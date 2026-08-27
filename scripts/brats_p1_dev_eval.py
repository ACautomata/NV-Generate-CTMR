# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P1 dev light-acceptance sidecar: fixed samples + FID trend + L2 trend (issue #57, spec #51 §6).

Runs beside the P1 finetune on a reserved GPU. For every ``epoch_<N>.pt`` the
trainer persists (N a multiple of ``--eval-every``), it:

1. generates the FIXED dev cohort — 16 dev cases x 4 target modalities
   (t1n/t1c/t2w/t2f), one sample per (case, modality) with a fixed
   per-(case, modality) seed, cfg=10, 30 steps, per-case spacing from the
   phase companions — the "fixed four-modality samples" the spec requires;
2. computes the 2.5D RadImageNet FID trend per target modality against the
   dev-side REAL volume bank (percentile 0-99.5 -> [0,1], RAS, 1 mm, zero pad
   240x240x160 — the pinned L1 MR preprocessing);
3. runs the frozen L2 instruments (nnUNetv2, ADR-0003 chain) on the generated
   pseudo-four-modality volumes and records WT/TC/ET volume medians plus
   input/run/hierarchy failure counts as the L2 trend;
4. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop`` for the trainer; ``select`` emits the final
   dev-side checkpoint selection (argmin mean FID) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts):
  metric m(N) = mean over the four target modalities of the plane-mean dev FID
  at epoch N; stop when N >= --min-epoch AND the last --patience consecutive
  evals produced no new best m; never past --max-epoch (= the trainer cap).

Usage (sugon, one reserved GPU):
    python -m scripts.brats_p1_dev_eval reference --dev-list ... --raw-root ... --eval-root DIR
    python -m scripts.brats_p1_dev_eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --emb-root ... -e env.json -c config.json -t network.json
    python -m scripts.brats_p1_dev_eval select --eval-root DIR --ckpt-dir DIR
    python -m scripts.brats_p1_dev_eval selftest --workdir TMP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))  # repo src layout: python -m scripts.<this>
sys.path.insert(0, str(_HERE / "src"))  # flat sugon deployment: src/ synced next to the script

# Forward shim (ticket 08 / ADR-0015 §2): the dev-eval engine (watch/select
# machinery), the shared cohort constants and the shared trend extractors
# (cohort/FID bank/instrument runner) moved to ctmr.application.shell and
# ctmr.application.generation.trends; this script consumes them until its own
# migration batch relocates it.
from ctmr.application.generation.trends import (  # noqa: E402
    EMPTY_SLICE_THRESHOLD,  # noqa: F401  (shim re-export: the retirement batch drops this module)
    PLANES,
    SAMPLE_EVERY_K,  # noqa: F401  (shim re-export)
    TREND_PREPROCESSING,  # noqa: F401  (shim re-export)
    DevCohortBuilder,
    L2TrendRunner,
    MrTrendFeatures,
    RealReferenceBank,
    TrendFid,
)
from ctmr.application.shell import (  # noqa: E402
    COHORT_QUOTAS,
    MODALITY_TOKENS,
    STOP_FILE,
    TARGET_MODALITIES,
    CheckpointWatcher,
    EarlyStopRule,
    TrendLedger,
)

from .diff_model_setting import load_config  # noqa: E402
from .utils import define_instance, dynamic_infer  # noqa: E402
from .utils_infer import ReconModel  # noqa: E402


class CohortSpacingSource:
    """Per-case post-resize spacing from the phase embedding companions (t1n entry)."""

    def __init__(self, dev_list_path, emb_root):
        self._emb_root = Path(emb_root)
        self._entries = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["modality"] == "mri_t1_skull_stripped":
                self._entries[entry["case"]] = entry["image"]

    def spacing_of(self, case):
        rel = self._entries[case].replace(".nii.gz", "_emb.nii.gz") + ".json"
        return json.loads((self._emb_root / rel).read_text())["spacing"]


class CandidateSampler:
    """Generates the fixed dev cohort samples with a candidate checkpoint (cfg=10, 30 steps)."""

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    @staticmethod
    def seed_of(case, modality):
        return int(hashlib.sha256(f"{case}|{modality}".encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def load_models(self, checkpoint_path):
        autoencoder = define_instance(self._args, "autoencoder_def").to(self._device)
        ae_ckpt = torch.load(self._args.trained_autoencoder_path, map_location=self._device, weights_only=True)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        unet = define_instance(self._args, "diffusion_unet_def").to(self._device)
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        autoencoder.eval()
        unet.eval()
        # Upstream inference convention is fp16 on the DCU (float16 latents);
        # a half-precision model keeps the conv input/weight/bias set consistent
        # (the HIP bf16 SDPA flash path emits fp16 and breaks the mixed chain).
        autoencoder = autoencoder.half()
        unet = unet.half()
        scale = float(ckpt["scale_factor"])
        return unet, ReconModel(autoencoder=autoencoder, scale_factor=scale).to(self._device).half()

    @torch.inference_mode()
    def sample_one(self, unet, recon_model, modality_token, spacing, seed, output_size=(256, 256, 128)):
        from monai.inferers import SlidingWindowInferer
        from monai.networks.schedulers import RFlowScheduler

        torch.manual_seed(seed)
        noise_scheduler = RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"})
        divisor = 4
        image = torch.randn((1, 4, output_size[0] // divisor, output_size[1] // divisor, output_size[2] // divisor), device=self._device)
        noise_scheduler.set_timesteps(
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(image.shape[2:])),
        )
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
        all_timesteps = noise_scheduler.timesteps
        all_next = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        cfg = self._args.cfg_guidance_scale
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            for t, next_t in zip(all_timesteps, all_next):
                unet_inputs = {
                    "x": image,
                    "timesteps": torch.Tensor((t,)).to(self._device),
                    "spacing_tensor": spacing_tensor,
                    "class_labels": modality_tensor,
                }
                if cfg > 0:
                    unet_inputs = {
                        key: (torch.cat([value, value]) if key != "class_labels" else torch.cat([value, torch.zeros_like(value)]))
                        for key, value in unet_inputs.items()
                    }
                    model_t, model_uncond = unet(**unet_inputs).chunk(2)
                    model_output = model_uncond + cfg * (model_t - model_uncond)
                else:
                    model_output = unet(**unet_inputs)
                image, _ = noise_scheduler.step(model_output, t, image, next_t)
        inferer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25, sw_device=self._device, device=torch.device("cpu"))
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            synthetic = dynamic_infer(inferer, recon_model, image).squeeze().float().cpu().numpy()
        data = synthetic * 1000.0  # [0,1] -> MR 0..1000 scale, upstream int16 convention
        return np.clip(data, 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cohort, spacings, out_dir):
        unet, recon = self.load_models(checkpoint_path)
        samples = []
        for item in cohort:
            for modality in TARGET_MODALITIES:
                seed = self.seed_of(item["case"], modality)
                out = Path(out_dir) / item["sub"] / f"{item['case']}_{modality}_seed{seed}.nii.gz"
                if not out.is_file():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data = self.sample_one(unet, recon, MODALITY_TOKENS[modality], spacings.spacing_of(item["case"]), seed)
                    image = nib.Nifti1Image(data, affine=np.diag([1.0, 1.0, 1.0, 1.0]))
                    nib.save(image, out)
                samples.append({"sub": item["sub"], "case": item["case"], "modality": modality, "path": str(out)})
        del unet, recon
        torch.cuda.empty_cache()
        return samples


class DevEvalSelfTest:
    """Fixture check of cohort/rule/selection logic (numpy/stdlib only, no GPU)."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        dev_list = self._workdir / "p1_dev.json"
        entries = []
        for challenge, quota in COHORT_QUOTAS.items():
            for index in range(quota + 2):
                entries.append({"sub": challenge, "case": f"FIX{challenge}-{index:04d}-000", "modality": "mri_t1_skull_stripped"})
        dev_list.write_text(json.dumps({"training": entries}))
        cohort = DevCohortBuilder(dev_list).build()
        if len(cohort) != sum(COHORT_QUOTAS.values()):
            self.failures.append(f"cohort size {len(cohort)} != {sum(COHORT_QUOTAS.values())}")
        if DevCohortBuilder(dev_list).build() != cohort:
            self.failures.append("cohort not deterministic")
        if {item["sub"] for item in cohort} != set(COHORT_QUOTAS):
            self.failures.append("cohort missing a challenge")

        rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
        improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
        stop, _ = rule.should_stop(improving)
        if stop:
            self.failures.append("rule stopped an improving trend")
        plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
        stop, reason = rule.should_stop(plateau)
        if not stop:
            self.failures.append(f"rule failed to stop a {3}-eval plateau ({reason})")
        short = improving + [{"epoch": 35, "m": 0.75}, {"epoch": 40, "m": 0.75}]
        stop, _ = rule.should_stop(short)
        if stop:
            self.failures.append("rule stopped before patience exhausted")
        selection = EarlyStopRule.selection(plateau)
        if selection["epoch"] != 30 or abs(selection["mean_fid"] - 0.70) > 1e-9:
            self.failures.append(f"selection picked {selection}, expected epoch 30 / m 0.70")

        trend = [{"epoch": 5, "m": 1.2, "checkpoint": "epoch_5.pt"}, {"epoch": 10, "m": 0.8, "checkpoint": "epoch_10.pt"}]
        ledger = TrendLedger(self._workdir)
        for record in trend:
            ledger.append(record)
        if ledger.read() != trend:
            self.failures.append("ledger roundtrip mismatch")
        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="build the dev real-feature bank once")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)

    p = sub.add_parser("watch", help="sidecar loop: evaluate epoch checkpoints as they land")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--skip-l2", action="store_true", help="FID-only trend (instruments unavailable)")
    p.add_argument("--instrument-results", action="append", default=[], help="CHALLENGE=nnUNet_results path")
    p.add_argument("--nnunet-raw", default="/root/private_data/brats2023_nnunet")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/nnUNet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("selftest")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        failures = DevEvalSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        features = MrTrendFeatures(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
        print(f"real reference bank -> {eval_root / 'reference' / 'real_reference_bank.pt'}")
        return 0

    if args.command == "select":
        trend = ledger.read()
        selection = EarlyStopRule.selection(trend)
        if selection is None:
            print("no eval points; nothing to select", file=sys.stderr)
            return 1
        selection["rule"] = "argmin mean dev FID over eval points (pre-recorded)"
        selection["trend"] = trend
        selection["recorded_utc"] = datetime.now(UTC).isoformat()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(selection, indent=2) + "\n")
        print(f"selection -> {out} (epoch {selection['epoch']}, mean_fid {selection['mean_fid']:.4f})")
        return 0

    # watch mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(args.dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = CohortSpacingSource(args.dev_list, args.emb_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
    fid = TrendFid(bank)
    sampler = CandidateSampler(merged, device, None)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.nnunet_raw, args.nnunet_preprocessed)
    watcher = CheckpointWatcher(args.ckpt_dir, args.eval_every, args.max_epoch, {r["epoch"] for r in ledger.read()})
    idle_since = None

    while True:
        pending = watcher.pending()
        if not pending:
            if args.idle_exit_seconds and idle_since is not None and time.time() - idle_since > args.idle_exit_seconds:
                break
            if args.idle_exit_seconds and idle_since is None:
                idle_since = time.time()
            time.sleep(args.poll_seconds)
            continue
        idle_since = None
        for epoch, path in pending:
            if any(r["epoch"] == epoch for r in ledger.read()):
                watcher.mark_done(epoch)
                continue
            epoch_dir = eval_root / f"epoch_{epoch}"
            try:
                samples = sampler.generate_cohort(path, cohort, spacings, epoch_dir / "samples")
                plane_cache = {sample["path"]: features.volume_features(sample["path"]) for sample in samples}
                generated = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
                for sample in samples:
                    for plane in PLANES:
                        matrix = plane_cache[sample["path"]][plane]
                        if matrix is not None:
                            generated[sample["modality"]][plane].append(matrix.mean(axis=0))
                report, mean_fid = fid.score(generated)
            except Exception as error:
                # A broken checkpoint, a transient network/model failure, or any
                # single-epoch hiccup must not kill the sidecar: without it
                # nobody writes .early_stop. Skip and retry on the next poll.
                print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                continue
            l2_trend = None
            if not args.skip_l2:
                try:
                    l2_trend = l2.run(samples, cohort, epoch_dir)
                except Exception as error:
                    print(f"[eval] epoch {epoch} l2 skipped: {error}", file=sys.stderr, flush=True)
            record = {
                "eval_utc": datetime.now(UTC).isoformat(),
                "epoch": epoch,
                "checkpoint": str(path),
                "fid": report,
                "m": mean_fid,
                "l2_trend": l2_trend,
                "cohort_file": str(cohort_path),
            }
            ledger.append(record)
            (epoch_dir / "trend.json").write_text(json.dumps(record, indent=2) + "\n")
            watcher.mark_done(epoch)
            stop, reason = rule.should_stop(ledger.read())
            print(f"[eval] epoch {epoch}: mean_fid={mean_fid} stop={stop} ({reason})", flush=True)
            if stop:
                (Path(args.ckpt_dir) / STOP_FILE).write_text(json.dumps({"reason": reason, "epoch": epoch}) + "\n")
                print(f"early-stop fired ({reason}); wrote {Path(args.ckpt_dir) / STOP_FILE}", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
