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

"""Mask-conditioned offline dev light acceptance: fixed samples + FID trend + L2 trend + round-trip Dice (issue #59, ADR-0007).

Offline form (ADR-0019 §5, #279): one pass over ANY run's already-persisted
checkpoints, training live or finished. For every ``epoch_<N>.pt`` on disk
(N a multiple of ``--eval-every``) the run's ledger does not have yet, it:

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
side: ``p2_mask_cond.json`` mixes train (fold=1) and dev (fold=0); this entry
uses only the fold=0 entries (matching the split manifest's dev side).

Migrated from the retired mask dev-eval script entry (ticket 09, ADR-0015
§2); its ``selftest`` subcommand retired with it — its assertions live as
pytest functions in tests/application/generation/mask. Since #273
(ADR-0019 §2) the watch face locates the engine through the composition
root's ``mask_engine`` lookup: the merged config comes from the
``GenerationEngine`` port, handed on to the sampler; the family assembles no
infrastructure itself.

Usage:
    ctmr generate mask dev-eval reference --dev-list ... --raw-root ... --eval-root DIR
    ctmr generate mask dev-eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --label-root ... -e env.json -c config.json -t network.json
    ctmr generate mask dev-eval select --eval-root DIR --ckpt-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ctmr.application.generation.devices import add_device_flag, resolve_device
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
    TARGET_MODALITIES,
    EarlyStopRule,
    SelectionEmitter,
    WatchEngine,
)
from ctmr.domain.grid import INSTRUMENT_GRID, InstrumentGridAdapter
from ctmr.domain.measurement import REGIONS, DiceScore, RegionMasks
from ctmr.domain.orientation import RasOrientation
from ctmr.wiring.generate import mask_engine

# Mask condition combined mask -> instrument label space (REGION_LABELS = {1,2,3}).
COMBINED_TO_INSTRUMENT = {22: 0, 129: 1, 130: 2, 131: 3}
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
    nearest-neighbour resampler + RAS direction world as the L2 final-acceptance
    path (ADR-0020: both sides enter RAS -- the pre-#314 x/y flip here mirrored
    the condition onto a different world from the L2TrendRunner's un-flipped
    predictions, misaligning this trend by construction). Dice is undefined
    (None) when both the condition and prediction regions are empty -- recorded
    as such, never a silent 0 (spec #51 decision 11).
    """

    def __init__(self, mask_source):
        self._mask_source = mask_source
        self._orientation = RasOrientation()

    @classmethod
    def remap_combined_to_instrument(cls, arr):
        out = np.zeros_like(arr)
        for src, dst in COMBINED_TO_INSTRUMENT.items():
            out[arr == src] = dst
        return out

    @staticmethod
    def dice(pred, condition, region):
        """The canonical DiceScore on canonical region projections (#223): the
        empty-denominator sentinel is ``None`` (spec #51 decision 11)."""
        return DiceScore.of(RegionMasks(condition).of(region), RegionMasks(pred).of(region))

    def align_condition(self, condition_path):
        """Aligns the combined mask onto the instrument grid (the L2 final-acceptance resampler path, RAS world).

        Failure classes match the terminal-acceptance assembler: unreadable
        files degrade to None (the cond_fail row), while a direction-world
        violation (``NotRasWorldError``) fails loudly -- an upstream protocol
        break, not a per-case input failure (ADR-0020).
        """
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        try:
            image = sitk.ReadImage(str(condition_path))
            aligned = InstrumentGridAdapter.label().align(self._orientation.to_ras(image))
            array = sitk.GetArrayFromImage(aligned).astype(np.uint8, copy=False)
        except (RuntimeError, OSError):
            return None
        if array.shape != PREDICTION_SHAPE:
            return None
        return self.remap_combined_to_instrument(array)

    def run(self, predictions_root, cohort):
        """Reads the L2 instrument predictions (``<sub>/<case>.nii.gz``), measures per-case Dice."""
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        rows = []
        region_medians = {region: [] for region in REGIONS}
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
            for region in REGIONS:
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


class FidTrendScorer:
    """The watch scorer seam: plane-mean 2.5D RadImageNet FID per target modality."""

    def __init__(self, features, fid):
        self._features = features
        self._fid = fid

    def __call__(self, samples):
        plane_cache = {sample["path"]: self._features.volume_features(sample["path"]) for sample in samples}
        generated = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
        for sample in samples:
            for plane in PLANES:
                matrix = plane_cache[sample["path"]][plane]
                if matrix is not None:
                    generated[sample["modality"]][plane].append(matrix.mean(axis=0))
        report, mean_fid = self._fid.score(generated)
        return {"fid": report, "m": mean_fid}, f"mean_fid={mean_fid}"


class L2PostScore:
    """The optional post-score extension: the frozen L2 instruments trend + round-trip Dice (``--skip-l2`` degrades to None).

    The extension owns its failure tolerance: a single-epoch instrument hiccup
    records the None fields and must not kill the watch pass -- the engine's
    skip path is reserved for the score itself.
    """

    def __init__(self, l2, round_trip, cohort, skip):
        self._l2 = l2
        self._round_trip = round_trip
        self._cohort = cohort
        self._skip = skip

    def __call__(self, epoch, samples, epoch_dir):
        fields = {"l2_trend": None, "round_trip_dice": None}
        if self._skip:
            return fields
        try:
            fields["l2_trend"] = self._l2.run(samples, self._cohort, epoch_dir)
            fields["round_trip_dice"] = self._round_trip.run(epoch_dir / "nnunet_predictions", self._cohort)
        except Exception as error:
            print(f"[eval] epoch {epoch} l2 skipped: {error}", file=sys.stderr, flush=True)
        return fields


def parse_args(argv=None):
    """The dev-eval entry argparse surface (verbatim from the retired dev-eval script entry).

    Exposed for the argv↔namespace equivalence gate (ADR-0015 Testing: the
    assertion lives in tests/application/generation/mask).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="build the dev real-feature bank once")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)
    add_device_flag(p)

    p = sub.add_parser("watch", help="offline pass: evaluate a run's existing epoch checkpoints, then exit")
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
    p.add_argument("--skip-l2", action="store_true", help="FID-only trend (instruments unavailable)")
    p.add_argument("--instrument-results", action="append", default=[], help="CHALLENGE=nnUNet_results path")
    p.add_argument(
        "--instrument-specs-autodiscover",
        action="store_true",
        help="resolve each challenge's -tr/-p/-c from the results tree's live <trainer>__<plans>__<config> dir "
        "(the v2-tree adaptation; the frozen INSTRUMENT_SPECS anchor is never mutated)",
    )
    p.add_argument("--nnunet-raw", default="/root/private_data/ctmr/data/nnunet_raw")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/ctmr/data/nnunet_preprocessed")
    add_device_flag(p)

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    eval_root = Path(args.eval_root)

    if args.command == "reference":
        dev_list = DevList(args.dev_list, eval_root).build()
        features = MrTrendFeatures(resolve_device(args.device))
        RealReferenceBank(dev_list, args.raw_root, features, eval_root / "reference").build()
        print(f"real reference bank -> {eval_root / 'reference' / 'real_reference_bank.pt'}")
        return 0

    if args.command == "select":
        return SelectionEmitter(eval_root).emit(args.out, rule_text="argmin mean dev FID over eval points (pre-recorded)")

    # watch mode: assemble the stage collaborators, the shell engine drives the loop
    dev_list = DevList(args.dev_list, eval_root).build()
    device = resolve_device(args.device)
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = CohortSpacingSource(dev_list)
    masks = ConditionMaskSource(dev_list, args.label_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    engine = mask_engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(dev_list, args.raw_root, features, eval_root / "reference").build()
    sampler = CandidateSampler(merged, device, None, engine)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.nnunet_raw, args.nnunet_preprocessed, autodiscover_specs=args.instrument_specs_autodiscover)
    return WatchEngine(
        ckpt_dir=args.ckpt_dir,
        eval_root=eval_root,
        eval_every=args.eval_every,
        max_epoch=args.max_epoch,
        rule=rule,
        # The engine's factory contract is the positional (checkpoint_path,
        # out_dir) call; generate_cohort's own parameter order interposes
        # cohort/spacings/masks, so a keyword partial lets the engine's second
        # positional argument land on ``cohort`` (multiple-values TypeError on
        # every eval point -- #316). The lambda keeps each argument on its name.
        sampler_factory=lambda checkpoint_path, out_dir: sampler.generate_cohort(checkpoint_path, cohort, spacings, masks, out_dir),
        scorer=FidTrendScorer(features, TrendFid(bank)),
        post_score=L2PostScore(l2, RoundTripDice(masks), cohort, args.skip_l2),
    ).run(cohort_file=str(cohort_path))


if __name__ == "__main__":
    sys.exit(main())
