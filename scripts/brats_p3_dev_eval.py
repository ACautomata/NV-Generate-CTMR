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

"""P3 dev light-acceptance sidecar: fixed image-conditioned samples + FID trend (issue #61).

Runs beside the P3 ControlNet finetune on a reserved GPU. For every ``epoch_<N>.pt`` the
trainer persists (N a multiple of ``--eval-every``), it:

1. generates the FIXED dev cohort — the dev-side cases × the 12 ordered src->tgt pairs,
   conditioned on the case's **src-image latent** (``src_image``, 4ch, no mask) with the
   target modality label and **CFG off** (``cfg_guidance_scale=0``, issue #61 acceptance
   criterion 1-2), fixed per-(case, src, tgt) seed;
2. computes the 2.5D RadImageNet FID trend per target modality against the dev-side REAL
   volume bank (the pinned L1 MR preprocessing) — the same selection trend P2 uses;
3. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop``; ``select`` emits the final dev-side checkpoint selection
   (argmin mean FID) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts): metric
m(N) = mean over the four target modalities of the plane-mean dev FID at epoch N; stop when
N >= --min-epoch AND the last --patience consecutive evals produced no new best m.

The dev cohort / real bank / spacing / src-latent source are all filtered to the dev side:
``p3_pairs.json`` mixes train (fold=1) and dev (fold=0); this script uses only fold=0.

Usage (sugon, one reserved GPU):
    python -m scripts.brats_p3_dev_eval reference --dev-list ... --raw-root ... --eval-root DIR
    python -m scripts.brats_p3_dev_eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --phase-root ... -e env.json -c config.json -t network_p3.json
    python -m scripts.brats_p3_dev_eval select --eval-root DIR --ckpt-dir DIR
    python -m scripts.brats_p3_dev_eval selftest --workdir TMP
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch

from .brats_p1_dev_eval import (
    COHORT_QUOTAS,
    MODALITY_TOKENS,
    PLANES,
    STOP_FILE,
    CheckpointWatcher,
    EarlyStopRule,
    L2TrendRunner,
    MrTrendFeatures,
    RealReferenceBank,
    TrendFid,
    TrendLedger,
)
from .brats_p3_controlnet_manifest import P3CandidateInferenceConfig
from .brats_phase_prep import MODALITIES as PAIR_MODALITIES
from .diff_model_setting import load_config
from .utils_infer import load_image_models, run_controlnet_conditioned_image_dm

LATENT = (4, 64, 64, 32)
GRID = (256, 256, 128)


def read_src_latent(src_image_path, device):
    """Loads a 4ch src-image latent NIfTI as (1,4,H,W,D) on the pinned grid (RAS, float).

    Uses the same transform chain as the training dataloader (LoadImage ->
    EnsureChannelFirst -> Orientation RAS) so the dev-eval condition matches the
    training condition exactly: the stored 4D latent is (X,Y,Z,C) and
    EnsureChannelFirst moves the channel axis to the front.
    """
    import monai.transforms as monai_t

    transform = monai_t.Compose(
        [
            monai_t.LoadImage(image_only=True),
            monai_t.EnsureChannelFirst(),
            monai_t.Orientation(axcodes="RAS"),
            monai_t.EnsureType(dtype=torch.float32),
        ]
    )
    x = transform(str(src_image_path))  # (C,H,W,D)
    return x[None].to(device)  # (1,C,H,W,D)


class P3DevList:
    """The dev (fold=0) view of the ``p3_pairs.json`` list, with raw tgt paths for the real bank."""

    def __init__(self, dev_list_path, eval_root):
        self._source = Path(dev_list_path)
        self._eval_root = Path(eval_root)

    def built_path(self):
        return self._eval_root / "dev_list.json"

    def build(self):
        out = self.built_path()
        if out.is_file():
            return out
        entries = json.loads(self._source.read_text())["training"]
        dev = []
        for entry in entries:
            if entry["fold"] != 0:
                continue
            # p3 image is the *embedding* path (embeddings/.../<case>-<mod>_emb.nii.gz);
            # the real bank needs the raw tgt volume relative to --raw-root.
            raw = entry["image"].replace("embeddings/", "").replace("_emb.nii.gz", ".nii.gz")
            dev.append({**copy.deepcopy(entry), "image": raw})
        self._eval_root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"training": dev}, indent=1) + "\n")
        print(f"p3 dev list: {len(dev)} entries -> {out}")
        return out


class P3DevCohort:
    """Builds the dev-case × 12-ordered-pair generation plan from the dev list."""

    def __init__(self, dev_list_path):
        self._entries = json.loads(Path(dev_list_path).read_text())["training"]

    def cases(self):
        seen, cases = set(), []
        for entry in self._entries:
            if entry["case"] not in seen:
                seen.add(entry["case"])
                cases.append({"sub": entry["sub"], "case": entry["case"]})
        return cases

    def spacing_of(self, case):
        for entry in self._entries:
            if entry["case"] == case:
                return entry["spacing"]

    def src_image_of(self, case, src_suffix):
        # list fields carry the long mapping keys (mri_*), translate the BraTS file suffix
        for entry in self._entries:
            if entry["case"] == case and entry["src_modality"] == PAIR_MODALITIES[src_suffix][0]:
                return entry["src_image"]

    def tgt_of(self, case, tgt_suffix):
        for entry in self._entries:
            if entry["case"] == case and entry["modality"] == PAIR_MODALITIES[tgt_suffix][0]:
                return entry["image"]


class P3CandidateSampler:
    """Generates the fixed dev cohort with a P3 ControlNet checkpoint (cfg=0, 30 steps)."""

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    def load_models(self, checkpoint_path):
        self._args.trained_controlnet_path = str(checkpoint_path)
        autoencoder, unet, controlnet, scale_factor, _noise_scheduler = load_image_models(self._args, self._device)
        for model in (autoencoder, unet, controlnet):
            model.eval()
        torch.cuda.empty_cache()
        return autoencoder, unet, controlnet, scale_factor

    @torch.inference_mode()
    def sample_one(self, autoencoder, unet, controlnet, scale_factor, spacing, modality_token, seed, src_latent):
        from monai.networks.schedulers import RFlowScheduler

        torch.manual_seed(seed)
        noise_scheduler = RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"})
        # ControlNet condition: the (already scaled to the model's normalized space) src latent.
        cond = (src_latent * scale_factor).half().to(self._device)
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
        synthetic = run_controlnet_conditioned_image_dm(
            autoencoder,
            unet,
            controlnet,
            noise_scheduler,
            scale_factor,
            self._device,
            controlnet_cond_tensor=cond,
            spacing_tensor=spacing_tensor,
            latent_shape=LATENT,
            output_size=GRID,
            noise_factor=1.0,
            modality_tensor=modality_tensor,
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            cfg_guidance_scale=0.0,
            controlnet_uncond_tensor=None,
        )
        return np.clip(synthetic.squeeze().float().cpu().numpy(), 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cases, cohort_source, phase_root, out_dir):
        import nibabel as nib

        autoencoder, unet, controlnet, scale_factor = self.load_models(checkpoint_path)
        samples = []
        for case in cases:
            spacing = cohort_source.spacing_of(case["case"])
            for src in MODALITY_TOKENS:
                for tgt in MODALITY_TOKENS:
                    if src == tgt:
                        continue
                    seed = P3CandidateInferenceConfig.seed_of(case["case"], src, tgt)
                    src_latent = read_src_latent(phase_root / cohort_source.src_image_of(case["case"], src), self._device)
                    out = Path(out_dir) / case["sub"] / f"{case['case']}_{src}_to_{tgt}_seed{seed}.nii.gz"
                    if not out.is_file():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        data = self.sample_one(
                            autoencoder, unet, controlnet, scale_factor, spacing, MODALITY_TOKENS[tgt], seed, src_latent
                        )
                        nib.save(nib.Nifti1Image(data, np.diag([spacing[0], spacing[1], spacing[2], 1.0])), out)
                    samples.append({"sub": case["sub"], "case": case["case"], "src_modality": src, "target_modality": tgt, "path": str(out)})
        del autoencoder, unet, controlnet
        torch.cuda.empty_cache()
        return samples


class P3DevEvalSelfTest:
    """Fixture check of p3-specific logic: dev-view, cohort, src-latent read, cfg=0 (numpy/stdlib)."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        src_entries = []
        for challenge, quota in COHORT_QUOTAS.items():
            for index in range(quota):
                case = f"FIX{challenge}-{index:04d}-000"
                for src in ("t1n", "t1c", "t2w", "t2f"):
                    for tgt in ("t1n", "t1c", "t2w", "t2f"):
                        if src == tgt:
                            continue
                        src_entries.append(
                            {
                                "image": f"embeddings/{challenge}/{case}/{case}-{tgt}_emb.nii.gz",
                                "src_image": f"embeddings/{challenge}/{case}/{case}-{src}_emb.nii.gz",
                                "label": f"labels/{challenge}/{case}/{case}-tumor129.nii.gz",
                                "spacing": [1.0, 1.0, 1.0],
                                "modality": PAIR_MODALITIES[tgt][0],
                                "src_modality": PAIR_MODALITIES[src][0],
                                "fold": 0,
                                "sub": challenge,
                                "case": case,
                            }
                        )
        src = self._workdir / "p3_src.json"
        src.write_text(json.dumps({"training": src_entries}))
        out = P3DevList(src, self._workdir).build()
        entries = json.loads(out.read_text())["training"]
        expected = sum(quota for quota in COHORT_QUOTAS.values()) * 12
        if len(entries) != expected:
            self.failures.append(f"dev-view kept {len(entries)} entries, expected {expected} (12 ordered pairs per dev case)")
        if not entries[0]["image"].endswith("-t1c.nii.gz") or "_emb" in entries[0]["image"]:
            self.failures.append(f"raw tgt not derived from embedding path: {entries[0]['image']}")
        if "src_image" not in entries[0]:
            self.failures.append("dev-view dropped the src_image condition")

        cohort_source = P3DevCohort(out)
        cohort = cohort_source.cases()
        n_cases = sum(quota for quota in COHORT_QUOTAS.values())
        if len(cohort) != n_cases:
            self.failures.append(f"cohort has {len(cohort)} dev cases, expected {n_cases}")
        if {item["sub"] for item in cohort} != set(COHORT_QUOTAS):
            self.failures.append("cohort missing a challenge")
        # the real pairs list keys src_modality/modality by the long mapping keys (mri_*);
        # the lookups must resolve the BraTS suffixes through that translation
        for suffix in PAIR_MODALITIES:
            if cohort_source.src_image_of(cohort[0]["case"], suffix) is None:
                self.failures.append(f"src lookup unresolved for {cohort[0]['case']} {suffix}")

        # src-latent channel-axis read: write a (H,W,D,C) NIfTI and confirm (C,H,W,D).
        import nibabel as nib

        latent = np.zeros((32, 32, 16, 4), dtype=np.float32)
        latent[..., 0] = 1.0
        latent_path = self._workdir / "latent.nii.gz"
        nib.save(nib.Nifti1Image(latent, np.eye(4)), str(latent_path))
        tensor = read_src_latent(latent_path, torch.device("cpu"))
        if tuple(tensor.shape) != (1, 4, 32, 32, 16):
            self.failures.append(f"src-latent read shape {tuple(tensor.shape)} != (1,4,32,32,16)")
        elif float(tensor[0, 0].mean()) != 1.0:
            self.failures.append("src-latent channel axis mis-read (channel 0 not the brain-modality slot)")

        # early-stop rule + selection (shared with P1/P2)
        rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
        improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
        stop, _ = rule.should_stop(improving)
        if stop:
            self.failures.append("rule stopped an improving trend")
        plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
        stop, reason = rule.should_stop(plateau)
        if not stop:
            self.failures.append(f"rule failed to stop a 3-eval plateau ({reason})")
        selection = EarlyStopRule.selection([{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}, {"epoch": 20, "m": 0.8}])
        if selection["epoch"] != 10:
            self.failures.append(f"selection picked {selection}, expected epoch 10")
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
    p.add_argument("--phase-root", required=True, help="phase root holding embeddings/labels (src-image latents)")
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("selftest")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        failures = P3DevEvalSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        return 1 if failures else 0

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        dev_list = P3DevList(args.dev_list, eval_root).build()
        features = MrTrendFeatures(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        RealReferenceBank(dev_list, args.raw_root, features, eval_root / "reference").build()
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
    dev_list = P3DevList(args.dev_list, eval_root).build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_source = P3DevCohort(dev_list)
    cohort = cohort_source.cases()
    phase_root = Path(args.phase_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 0.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(dev_list, args.raw_root, features, eval_root / "reference").build()
    fid = TrendFid(bank)
    sampler = P3CandidateSampler(merged, device, None)
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
                samples = sampler.generate_cohort(path, cohort, cohort_source, phase_root, epoch_dir / "samples")
                plane_cache = {sample["path"]: features.volume_features(sample["path"]) for sample in samples}
                generated = {modality: {plane: [] for plane in PLANES} for modality in MODALITY_TOKENS}
                for sample in samples:
                    for plane in PLANES:
                        matrix = plane_cache[sample["path"]][plane]
                        if matrix is not None:
                            generated[sample["target_modality"]][plane].append(matrix.mean(axis=0))
                report, mean_fid = fid.score(generated)
            except Exception as error:
                print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                continue
            record = {
                "eval_utc": datetime.now(UTC).isoformat(),
                "epoch": epoch,
                "checkpoint": str(path),
                "fid": report,
                "m": mean_fid,
                "cohort_file": str(dev_list),
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
