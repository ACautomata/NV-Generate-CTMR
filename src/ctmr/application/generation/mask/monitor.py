# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mask-conditioned dev light-acceptance sidecar: fixed samples + FID trend + L2 trend + round-trip Dice (issue #59, ADR-0007).

Runs beside the mask ControlNet finetune on a reserved GPU (spec #51
decision 7). For every ``epoch_<N>.pt`` the trainer persists (N a multiple of
``--eval-every``), it:

1. generates the FIXED dev cohort — 16 dev cases x 4 target modalities
   (t1n/t1c/t2w/t2f), one per (case, modality) with a fixed per-(case,
   modality) seed, cfg=10, 30 steps, the case's ``-combined`` mask as the
   ControlNet condition and per-case spacing from the phase companions;
2. computes the 2.5D RadImageNet FID trend per target modality against the
   dev-side REAL volume bank (the pinned L1 MR preprocessing);
3. runs the frozen L2 instruments on the generated four-modality volumes and
   records WT/TC/ET volume medians plus input/run/hierarchy failure counts;
4. computes the mask condition round-trip Dice trend (instrument-predicted
   mask vs the combined condition mask, nearest-neighbour aligned to the
   instrument grid, combined 22/129/130/131 remapped to 0/1/2/3) — a selection
   trend, the formal pass line lives in the L2 final-acceptance round_trip;
5. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop``; ``select`` emits the final dev-side checkpoint
   selection (argmin mean FID) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts):
  metric m(N) = mean over the four target modalities of the plane-mean dev FID
  at epoch N; stop when N >= --min-epoch AND the last --patience consecutive
  evals produced no new best m; never past --max-epoch (= the trainer cap).

The dev cohort / real bank / spacing / mask source are all filtered to the dev
side: ``p2_mask_cond.json`` mixes train (fold=1) and dev (fold=0); this sidecar
uses only the fold=0 entries (matching the split manifest's dev side).

Migrated from the retired mask dev-eval script entry (ticket 09, ADR-0015
§2); its ``selftest`` subcommand retired with it — its assertions live as
pytest functions in tests/application/generation/mask.

Usage (sugon, one reserved GPU):
    ctmr generate mask dev-eval reference --dev-list ... --raw-root ... --eval-root DIR
    ctmr generate mask dev-eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --label-root ... -e env.json -c config.json -t network.json
    ctmr generate mask dev-eval select --eval-root DIR --ckpt-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ctmr.application.generation.mask.sample import CandidateSampler
from ctmr.application.generation.trend import (
    PLANES,
    DevCohortBuilder,
    L2TrendRunner,
    MrTrendFeatures,
    RealReferenceBank,
    TrendFid,
)
from ctmr.application.shell import (
    STOP_FILE,
    TARGET_MODALITIES,
    CheckpointWatcher,
    EarlyStopRule,
    TrendLedger,
)
from ctmr.domain.grid import INSTRUMENT_GRID, InstrumentGridAdapter
from ctmr.infrastructure.maisi_engine.diff_model_setting import load_config

# Mask condition combined mask -> instrument label space (REGION_LABELS = {1,2,3}).
COMBINED_TO_INSTRUMENT = {22: 0, 129: 1, 130: 2, 131: 3}
INSTRUMENT_REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
# The terminal-acceptance-only DM RAS->LPS axis flip (zyx array axes y=1, x=2);
# the round-trip condition alignment must track the final-acceptance resampler
# path -- the parity is machine-guarded in tests/application/generation/mask.
DM_GRID_TO_LPS_AXIS_FLIP = (1, 2)
PREDICTION_SHAPE = tuple(reversed(INSTRUMENT_GRID.size))  # array layout is zyx


class DevList:
    """The dev (fold=0) view of the #52 ``p2_mask_cond.json`` list, with raw image paths.

    The mask list carries train (fold=1) and dev (fold=0) in one file for the
    trainer's fold split. The dev-eval needs only the dev side, and the real
    reference bank needs the RAW image path (the mask list's ``image`` field is
    the *embedding* path). This builds a dev-only list file with raw ``image``
    and keeps ``label`` (combined mask) + ``spacing`` for the sampler and Dice.
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
            # image is the *embedding* path (embeddings/.../<case>-<mod>_emb.nii.gz);
            # the real bank needs the raw volume relative to --raw-root (raw/.../<case>-<mod>.nii.gz).
            raw = image.replace("embeddings/", "").replace("_emb.nii.gz", ".nii.gz")
            dev.append({**entry, "image": raw})
        self._eval_root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"training": dev}, indent=1) + "\n")
        print(f"mask dev list: {len(dev)} entries -> {out}")
        return out


class CohortSpacingSource:
    """Per-case post-resize spacing from the mask dev entries (inline ``spacing``)."""

    def __init__(self, dev_list_path):
        self._by_case = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["case"] not in self._by_case:
                self._by_case[entry["case"]] = entry["spacing"]

    def spacing_of(self, case):
        return self._by_case[case]


class ConditionMaskSource:
    """Per-case mask condition (``-combined.nii.gz``) relative to the phase label root."""

    def __init__(self, dev_list_path, label_root):
        self._label_root = Path(label_root)
        self._by_case = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["case"] not in self._by_case:
                self._by_case[entry["case"]] = entry["label"]

    def path_of(self, case):
        return self._label_root / self._by_case[case]


class RoundTripDice:
    """Mask condition round-trip Dice trend: instrument prediction vs combined condition mask.

    The combined condition (22/129/130/131) is remapped to the instrument label
    space (0/1/2/3) and aligned to the instrument grid with the same
    nearest-neighbour resampler + terminal-acceptance DM RAS->LPS flip as the
    L2 final-acceptance path. Dice is undefined (None) when both the condition
    and prediction regions are empty -- recorded as such, never a silent 0
    (spec #51 decision 11).
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
        """Aligns the combined mask onto the instrument grid (the L2 final-acceptance resampler path)."""
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        try:
            image = sitk.ReadImage(str(condition_path))
            aligned = InstrumentGridAdapter.label().align(image)
            array = np.flip(sitk.GetArrayFromImage(aligned).astype(np.uint8, copy=False), axis=DM_GRID_TO_LPS_AXIS_FLIP)
        except (RuntimeError, OSError):
            return None
        if array.shape != PREDICTION_SHAPE:
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
            "median_cond_dice": {region: (float(np.median(values)) if values else None) for region, values in region_medians.items()},
        }
        return summary


def parse_args(argv=None):
    """The sidecar entry argparse surface (verbatim from the retired dev-eval script entry).

    Exposed for the argv↔namespace equivalence gate (ADR-0015 Testing: the
    assertion lives in tests/application/generation/mask).
    """
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
    p.add_argument("--nnunet-raw", default="/root/private_data/brats2023_nnunet")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/nnUNet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        dev_list = DevList(args.dev_list, eval_root).build()
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
    dev_list = DevList(args.dev_list, eval_root).build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = CohortSpacingSource(dev_list)
    masks = ConditionMaskSource(dev_list, args.label_root)
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
    sampler = CandidateSampler(merged, device, None)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.nnunet_raw, args.nnunet_preprocessed)
    round_trip = RoundTripDice(masks)
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
