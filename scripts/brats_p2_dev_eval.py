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

"""P2 dev light-acceptance sidecar: fixed mask-conditioned samples + FID trend + L2 trend + round-trip Dice.

Runs beside the P2 ControlNet finetune on a reserved GPU (issue #59, spec #51
decision 7, ADR-0007). For every ``epoch_<N>.pt`` the trainer persists (N a
multiple of ``--eval-every``), it:

1. generates the FIXED dev cohort — 16 dev cases x 4 target modalities
   (t1n/t1c/t2w/t2f), one per (case, modality) with a fixed per-(case,
   modality) seed, cfg=10, 30 steps, the case's ``-combined`` mask as the
   ControlNet condition and per-case spacing from the phase companions;
2. computes the 2.5D RadImageNet FID trend per target modality against the
   dev-side REAL volume bank (the pinned L1 MR preprocessing);
3. runs the frozen L2 instruments on the generated four-modality volumes and
   records WT/TC/ET volume medians plus input/run/hierarchy failure counts;
4. computes the P2 condition round-trip Dice trend (instrument-predicted mask
   vs the combined condition mask, nearest-neighbour aligned to the instrument
   grid, combined 22/129/130/131 remapped to 0/1/2/3) — a selection trend, the
   formal pass line lives in the L2 final-acceptance round_trip;
5. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop``; ``select`` emits the final dev-side checkpoint
   selection (argmin mean FID) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts):
  metric m(N) = mean over the four target modalities of the plane-mean dev FID
  at epoch N; stop when N >= --min-epoch AND the last --patience consecutive
  evals produced no new best m; never past --max-epoch (= the trainer cap).

The dev cohort / real bank / spacing / mask source are all filtered to the dev
side: ``p2_mask_cond.json`` mixes train (fold=1) and dev (fold=0); this script
uses only the fold=0 entries (matching the split manifest's dev side).

Usage (sugon, one reserved GPU):
    python -m scripts.brats_p2_dev_eval reference --dev-list ... --raw-root ... --eval-root DIR
    python -m scripts.brats_p2_dev_eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --label-root ... -e env.json -c config.json -t network.json
    python -m scripts.brats_p2_dev_eval select --eval-root DIR --ckpt-dir DIR
    python -m scripts.brats_p2_dev_eval selftest --workdir TMP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import torch

from .brats_p1_dev_eval import (
    COHORT_QUOTAS,
    MODALITY_TOKENS,
    TARGET_MODALITIES,
    PLANES,
    STOP_FILE,
    CheckpointWatcher,
    DevCohortBuilder,
    EarlyStopRule,
    L2TrendRunner,
    MrTrendFeatures,
    RealReferenceBank,
    TrendFid,
    TrendLedger,
)
from .diff_model_setting import load_config
from .infer_image_from_mask import ldm_conditional_sample_one_image_from_mask

# P2 condition combined mask -> instrument label space (REGION_LABELS = {1,2,3}).
COMBINED_TO_INSTRUMENT = {22: 0, 129: 1, 130: 2, 131: 3}
INSTRUMENT_REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType]
)


class P2DevList:
    """The dev (fold=0) view of the #52 ``p2_mask_cond.json`` list, with raw image paths.

    The p2 list carries train (fold=1) and dev (fold=0) in one file for the
    trainer's fold split. The dev-eval needs only the dev side, and the real
    reference bank needs the RAW image path (the p2 ``image`` field is the
    *embedding* path). This builds a dev-only list file with raw ``image`` and
    keeps ``label`` (combined mask) + ``spacing`` for the sampler and Dice.
    """

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
            image = entry["image"]
            # p2 image is the *embedding* path (embeddings/.../<case>-<mod>_emb.nii.gz);
            # the real bank needs the raw volume relative to --raw-root (raw/.../<case>-<mod>.nii.gz).
            raw = image.replace("embeddings/", "").replace("_emb.nii.gz", ".nii.gz")
            dev.append({**entry, "image": raw})
        self._eval_root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"training": dev}, indent=1) + "\n")
        print(f"p2 dev list: {len(dev)} entries -> {out}")
        return out


class P2CohortSpacingSource:
    """Per-case post-resize spacing from the p2 dev entries (inline ``spacing``)."""

    def __init__(self, dev_list_path):
        self._by_case = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["case"] not in self._by_case:
                self._by_case[entry["case"]] = entry["spacing"]

    def spacing_of(self, case):
        return self._by_case[case]


class P2CohortMaskSource:
    """Per-case P2 condition mask (``-combined.nii.gz``) relative to the phase label root."""

    def __init__(self, dev_list_path, label_root):
        self._label_root = Path(label_root)
        self._by_case = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["case"] not in self._by_case:
                self._by_case[entry["case"]] = entry["label"]

    def path_of(self, case):
        return self._label_root / self._by_case[case]


class P2CandidateSampler:
    """Generates the fixed dev cohort samples with a P2 ControlNet checkpoint (cfg=10, 30 steps)."""

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    @staticmethod
    def seed_of(case, modality):
        return int(hashlib.sha256(f"{case}|{modality}".encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def load_models(self, checkpoint_path):
        from .utils import define_instance

        autoencoder = define_instance(self._args, "autoencoder_def").to(self._device)
        ae_ckpt = torch.load(self._args.trained_autoencoder_path, map_location=self._device, weights_only=True)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        unet = define_instance(self._args, "diffusion_unet_def").to(self._device)
        dm_ckpt = torch.load(self._args.trained_diffusion_path, map_location=self._device, weights_only=True)
        unet.load_state_dict(dm_ckpt["unet_state_dict"], strict=False)
        controlnet = define_instance(self._args, "controlnet_def").to(self._device)
        cn_ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        controlnet.load_state_dict(cn_ckpt["controlnet_state_dict"], strict=False)
        for model in (autoencoder, unet, controlnet):
            model.eval()
        # Upstream inference convention is fp16 on the DCU (float16 latents); a
        # half-precision model keeps the conv input/weight/bias set consistent
        # with the HIP bf16 SDPA flash adapter that emits fp16 (P1 convention).
        autoencoder = autoencoder.half()
        unet = unet.half()
        controlnet = controlnet.half()
        scale = float(dm_ckpt["scale_factor"])
        return autoencoder, unet, controlnet, scale

    @staticmethod
    def load_condition_mask(mask_source, case, device):
        """Loads the case's combined mask as a (1,1,H,W,D) long tensor on the grid."""
        from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd

        path = mask_source.path_of(case)
        if not path.is_file():
            raise FileNotFoundError(f"combined mask missing: {path}")
        transform = Compose(
            [
                LoadImaged(keys=["label"], image_only=True),
                EnsureChannelFirstd(keys=["label"]),
                Orientationd(keys=["label"], axcodes="RAS"),
                EnsureTyped(keys=["label"], dtype=torch.long),
            ]
        )
        label = transform({"label": str(path)})["label"]
        if label.ndim == 4:
            label = label.unsqueeze(0)
        return label.to(device)

    @torch.inference_mode()
    def sample_one(self, autoencoder, unet, controlnet, scale, modality_token, spacing, seed, condition):
        from monai.networks.schedulers import RFlowScheduler

        torch.manual_seed(seed)
        noise_scheduler = RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"})
        noise_scheduler.set_timesteps(
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor((64, 64, 32))),
        )
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
        # Returns (synthetic_image, combine_label); only the image is used here.
        synthetic, _returned_label = ldm_conditional_sample_one_image_from_mask(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale,
            device=self._device,
            combine_label_or=condition,
            spacing_tensor=spacing_tensor,
            latent_shape=(4, 64, 64, 32),
            output_size=(256, 256, 128),
            noise_factor=1.0,
            modality_tensor=modality_tensor,
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            autoencoder_sliding_window_infer_size=(96, 96, 96),
            autoencoder_sliding_window_infer_overlap=0.25,
            cfg_guidance_scale=self._args.cfg_guidance_scale,
        )
        data = synthetic.squeeze().float().cpu().numpy()
        return np.clip(data, 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cohort, spacings, masks, out_dir):
        import nibabel as nib

        autoencoder, unet, controlnet, scale = self.load_models(checkpoint_path)
        samples = []
        for item in cohort:
            condition = self.load_condition_mask(masks, item["case"], self._device)
            for modality in TARGET_MODALITIES:
                seed = self.seed_of(item["case"], modality)
                out = Path(out_dir) / item["sub"] / f"{item['case']}_{modality}_seed{seed}.nii.gz"
                if not out.is_file():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data = self.sample_one(
                        autoencoder, unet, controlnet, scale, MODALITY_TOKENS[modality],
                        spacings.spacing_of(item["case"]), seed, condition,
                    )
                    nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), out)
                samples.append({"sub": item["sub"], "case": item["case"], "modality": modality, "path": str(out)})
        del autoencoder, unet, controlnet
        torch.cuda.empty_cache()
        return samples


class P2RoundTripDice:
    """P2 condition round-trip Dice trend: instrument prediction vs combined condition mask.

    The combined condition (22/129/130/131) is remapped to the instrument label
    space (0/1/2/3) and aligned to the instrument grid with the same
    nearest-neighbour resampler as the L2 final-acceptance path. Dice is
    undefined (None) when both the condition and prediction regions are empty --
    recorded as such, never a silent 0 (spec #51 decision 11).
    """

    def __init__(self, mask_source):
        self._mask_source = mask_source

    @classmethod
    def remap_combined_to_instrument(cls, arr):
        out = np.zeros_like(arr)
        for src, dst in COMBINED_TO_INSTRUMENT.items():
            out[arr == src] = dst
        return out

    @staticmethod
    def dice(pred, condition, region):
        gt = np.isin(condition, INSTRUMENT_REGION_LABELS[region])
        pm = np.isin(pred, INSTRUMENT_REGION_LABELS[region])
        denom = int(gt.sum()) + int(pm.sum())
        if denom == 0:
            return None
        return float(2 * np.logical_and(gt, pm).sum() / denom)

    def align_condition(self, condition_path):
        """Aligns the combined mask onto the instrument grid (reuse L2 resampler)."""
        from .nnunet_l2_final_acceptance_nifti import GeneratedVolumeResampler

        array = GeneratedVolumeResampler().label_to_grid(str(condition_path))
        if array is None:
            return None
        return self.remap_combined_to_instrument(array)

    def run(self, predictions_root, cohort):
        """Reads the L2 instrument predictions (``<sub>/<case>.nii.gz``), measures per-case Dice."""
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        rows = []
        region_medians = {region: [] for region in INSTRUMENT_REGION_LABELS}
        for item in cohort:
            pred_path = Path(predictions_root) / item["sub"] / f"{item['case']}.nii.gz"
            row = {"sub": item["sub"], "case": item["case"]}
            if not pred_path.is_file():
                row["run_fail"] = True
                rows.append(row)
                continue
            condition = self.align_condition(self._mask_source.path_of(item["case"]))
            if condition is None:
                row["cond_fail"] = True
                rows.append(row)
                continue
            try:
                pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.uint8, copy=False)
            except (RuntimeError, OSError):
                row["run_fail"] = True
                rows.append(row)
                continue
            for region in INSTRUMENT_REGION_LABELS:
                value = self.dice(pred_arr, condition, region)
                row[f"cond_dice_{region.lower()}"] = value
                if value is not None:
                    region_medians[region].append(value)
            rows.append(row)
        summary = {
            "per_case": rows,
            "n_run_fail": sum(1 for row in rows if row.get("run_fail")),
            "n_cond_fail": sum(1 for row in rows if row.get("cond_fail")),
            "median_cond_dice": {
                region: (float(np.median(values)) if values else None)
                for region, values in region_medians.items()
            },
        }
        return summary


class P2DevEvalSelfTest:
    """Fixture check of p2-specific logic: dev-view, cohort, remap/dice, selection (numpy/stdlib + nibabel).

    Mirrors the P1 dev-eval selftest and adds the P2 combined->instrument remap
    and round-trip Dice arithmetic. Runs without a GPU.
    """

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        # dev-view: fold0 kept, fold1 dropped, raw image derived from emb path.
        src_entries = []
        for challenge, quota in COHORT_QUOTAS.items():
            for index in range(quota + 2):
                src_entries.append(
                    {"image": f"embeddings/{challenge}/FIX{challenge}-{index:04d}-000-t1n_emb.nii.gz",
                     "label": f"labels/{challenge}/FIX{challenge}-{index:04d}-000/FIX{challenge}-{index:04d}-000-combined.nii.gz",
                     "spacing": [1.0, 1.0, 1.0], "modality": "mri_t1_skull_stripped",
                     "fold": 0, "sub": challenge, "case": f"FIX{challenge}-{index:04d}-000"}
                )
        # one train-side (fold=1) entry per challenge must be dropped.
        for challenge, quota in COHORT_QUOTAS.items():
            src_entries.append(
                {"image": f"embeddings/{challenge}/TRAIN{challenge}-000-t1n_emb.nii.gz",
                 "label": f"labels/{challenge}/TRAIN{challenge}-000/TRAIN{challenge}-000-combined.nii.gz",
                 "spacing": [1.0, 1.0, 1.0], "modality": "mri_t1_skull_stripped",
                 "fold": 1, "sub": challenge, "case": f"TRAIN{challenge}-000"}
            )
        src = self._workdir / "p2_src.json"
        src.write_text(json.dumps({"training": src_entries}))
        out = P2DevList(src, self._workdir).build()
        entries = json.loads(out.read_text())["training"]
        total_dev = sum(quota + 2 for quota in COHORT_QUOTAS.values())
        if len(entries) != total_dev:
            self.failures.append(f"dev-view kept {len(entries)} entries, expected {total_dev} (fold=1 not dropped?)")
        kept_cases = {entry["case"] for entry in entries}
        if any(case.startswith("TRAIN") for case in kept_cases):
            self.failures.append("dev-view leaked a train-side (fold=1) case")
        if not entries[0]["image"].endswith("-t1n.nii.gz") or entries[0]["image"].startswith("raw/") or "_emb" in entries[0]["image"]:
            self.failures.append(f"raw image not derived (emb/raw prefix leak): {entries[0]['image']}")

        cohort = DevCohortBuilder(out).build()
        if not cohort:
            self.failures.append("cohort empty for dev-view")
        if {item["sub"] for item in cohort} != set(COHORT_QUOTAS):
            self.failures.append("cohort missing a challenge")

        # round-trip dice arithmetic: combined 129/130/131 -> instrument 1/2/3.
        condition = np.array([[[0, 22], [129, 130], [131, 0]]], dtype=np.int16)
        remapped = P2RoundTripDice.remap_combined_to_instrument(condition)
        expected = np.array([[[0, 0], [1, 2], [3, 0]]], dtype=np.int16)
        if not np.array_equal(remapped, expected):
            self.failures.append(f"remap mismatch: {remapped.tolist()} != {expected.tolist()}")

        pred = np.array([[[0, 1], [1, 2], [3, 0]]], dtype=np.uint8)
        perfect = P2RoundTripDice.dice(pred, remapped, "WT")
        if abs(perfect - 1.0) > 1e-9:
            self.failures.append(f"perfect round-trip WT dice != 1.0: {perfect}")
        empty = P2RoundTripDice.dice(np.zeros_like(pred), np.zeros_like(remapped), "ET")
        if empty is not None:
            self.failures.append("both-empty dice must be None, not 0")

        rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
        improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
        stop, _ = rule.should_stop(improving)
        if stop:
            self.failures.append("rule stopped an improving trend")
        plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
        stop, reason = rule.should_stop(plateau)
        if not stop:
            self.failures.append(f"rule failed to stop a 3-eval plateau ({reason})")

        trend = [{"epoch": e, "m": m, "checkpoint": f"epoch_{e}.pt"} for e, m in ((5, 1.2), (10, 0.8), (15, 0.8))]
        ledger = TrendLedger(self._workdir)
        for record in trend:
            ledger.append(record)
        if ledger.read() != trend:
            self.failures.append("ledger roundtrip mismatch")
        selection = EarlyStopRule.selection(
            [{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}, {"epoch": 20, "m": 0.8}]
        )
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
    p.add_argument("--label-root", required=True)
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
    p.add_argument("--instrument-entry", default="scripts/l2_calibration_predict_entry.py")
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
        failures = P2DevEvalSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        dev_list = P2DevList(args.dev_list, eval_root).build()
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
    dev_list = P2DevList(args.dev_list, eval_root).build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = P2CohortSpacingSource(dev_list)
    masks = P2CohortMaskSource(dev_list, args.label_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(dev_list, args.raw_root, features, eval_root / "reference").build()
    fid = TrendFid(bank)
    sampler = P2CandidateSampler(merged, device, None)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.instrument_entry, args.nnunet_raw, args.nnunet_preprocessed)
    round_trip = P2RoundTripDice(masks)
    watcher = CheckpointWatcher(
        args.ckpt_dir, args.eval_every, args.max_epoch, {r["epoch"] for r in ledger.read()}
    )
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
                samples = sampler.generate_cohort(path, cohort, spacings, masks, epoch_dir / "samples")
                plane_cache = {sample["path"]: features.volume_features(sample["path"]) for sample in samples}
                generated = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
                for sample in samples:
                    for plane in PLANES:
                        matrix = plane_cache[sample["path"]][plane]
                        if matrix is not None:
                            generated[sample["modality"]][plane].append(matrix.mean(axis=0))
                report, mean_fid = fid.score(generated)
            except Exception as error:
                print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                continue
            l2_trend = None
            round_trip_dice = None
            if not args.skip_l2:
                try:
                    l2_trend = l2.run(samples, cohort, epoch_dir)
                    round_trip_dice = round_trip.run(epoch_dir / "nnunet_predictions", cohort)
                except Exception as error:
                    print(f"[eval] epoch {epoch} l2 skipped: {error}", file=sys.stderr, flush=True)
            record = {
                "eval_utc": datetime.now(UTC).isoformat(),
                "epoch": epoch,
                "checkpoint": str(path),
                "fid": report,
                "m": mean_fid,
                "l2_trend": l2_trend,
                "round_trip_dice": round_trip_dice,
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
