#!/usr/bin/env python3
"""Issue #55 NIfTI execution side for the L2 final-acceptance pipeline.

Runs on the sugon DCU host (needs SimpleITK / numpy / scipy); every judgement
rule lives in ``nnunet_l2_final_acceptance.py`` (stdlib-only). Two commands:

  assemble-execute   plan -> instrument inputs at ``<output-root>/inputs/<CH>/``
                     (generated side: the issue #38 ``InputPreparator``
                     geometry verbatim -- resample to 1mm B-spline, centre
                     crop/pad to 240x240x155; real side: byte-identical
                     pass-through, the native BraTS volumes already meet the
                     instrument contract). One file per observation and channel:
                     ``{obs_id}_{suffix}.nii.gz``.
  measure            plan + predictions -> measurement CSV (schema shared with
                     the judge). Semantics copied from the calibration mother
                     implementation (issue #36 ``measure_case``), with the
                     GT columns dropped -- final acceptance has no GT. Failure
                     flags (input_fail / run_fail / hier_viol) are checked on
                     BOTH sides; empty predictions are measurement results,
                     not failures.

The z-axis geometry fact (resampled 241x241x174 centred-cropped to 155 slices,
~19 slices dropped on the generated side only) is registered in the protocol
and carried into the report appendix -- not compensated here.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # sibling scripts (the stdlib judge module)
sys.path.insert(0, str(_HERE.parent / "src"))  # repo src layout: python scripts/<this>.py
sys.path.insert(0, str(_HERE / "src"))  # flat sugon deployment: src/ synced next to the script

from nnunet_l2_final_acceptance import (  # noqa: E402
    CHANNEL_SUFFIXES,
    REGION_LABELS,
    REGIONS,
    MeasurementTable,
)

from ctmr.domain.grid import INSTRUMENT_GRID, InstrumentGridAdapter  # noqa: E402

NNUNET_TARGET_SIZE = INSTRUMENT_GRID.size
PREDICTION_SHAPE = tuple(reversed(NNUNET_TARGET_SIZE))  # array layout is zyx

# P2 condition combined mask -> instrument label space (mirrors the dev sidecar
# COMBINED_TO_INSTRUMENT in scripts/brats_p2_dev_eval.py). The combined mask is
# stored in the BraTS 2023 label ids (22/129/130/131); the instrument predicts
# 0/1/2/3, so the round-trip Dice must remap first -- comparing raw ids against
# REGION_LABELS (1/2/3) matches nothing and yields a spurious exact 0.
COMBINED_TO_INSTRUMENT = {22: 0, 129: 1, 130: 2, 131: 3}

# The DM emits generated volumes (and the #52 condition masks) on the RAS grid,
# while the real BraTS side is passed through in its native LPS orientation.
# RAS<->LPS flips the x and y axes and preserves z, so the generated volume and
# the condition mask are x/y-flipped onto the instrument grid to align with the
# real reference (array layout is zyx: flip the last two axes). Without this the
# gen/real pair lands ~240mm apart and the TOST centroid test fails spuriously.
DM_GRID_TO_LPS_AXIS_FLIP = (1, 2)  # zyx array axes to reverse (y=1, x=2)


class GeneratedVolumeResampler:
    """Issue #38 InputPreparator geometry (protocol §2), with the axis handling
    corrected for the zyx array layout (#38 applied xyz slices to a zyx array).

    Since #105 (ADR-0008) the geometry itself lives in ctmr.domain.grid
    (InstrumentGridAdapter: B-spline continuum / nearest-neighbour label,
    centred crop/pad onto the instrument grid); this shell keeps only the
    terminal-acceptance-only DM-RAS -> LPS flip and the file IO.
    """

    def __init__(self):
        self._continuum = InstrumentGridAdapter.continuum()
        self._label = InstrumentGridAdapter.label()

    def write(self, source, destination):
        image = sitk.ReadImage(str(source))
        aligned = self._flip_dm_grid_to_lps(self._continuum.align(image))
        sitk.WriteImage(aligned, str(destination))

    def label_to_grid(self, source):
        """Aligns a label volume onto the instrument grid; None when unreadable."""
        try:
            image = sitk.ReadImage(str(source))
            aligned = self._flip_dm_grid_to_lps(self._label.align(image))
            array = sitk.GetArrayFromImage(aligned).astype(np.uint8, copy=False)
        except (RuntimeError, OSError):
            return None
        if array.shape != PREDICTION_SHAPE:
            return None
        return array

    @staticmethod
    def _flip_dm_grid_to_lps(image):
        """RAS(DM grid) -> LPS(instrument grid); see DM_GRID_TO_LPS_AXIS_FLIP."""
        array = sitk.GetArrayFromImage(image)
        for axis in DM_GRID_TO_LPS_AXIS_FLIP:  # zyx array axes to reverse (y=1, x=2)
            array = np.flip(array, axis=axis)
        result = sitk.GetImageFromArray(array)
        result.SetSpacing(image.GetSpacing())
        result.SetOrigin(image.GetOrigin())
        result.SetDirection(image.GetDirection())
        return result


class ObservationInputWriter:
    """Writes every plan observation into the instrument input tree."""

    def __init__(self, output_root):
        self._output_root = Path(output_root)
        self._resampler = GeneratedVolumeResampler()

    def write_all(self, plan):
        for observation in plan["observations"]:
            case_dir = self._output_root / "inputs" / observation["challenge"]
            case_dir.mkdir(parents=True, exist_ok=True)
            for suffix, source in sorted(observation["channels"].items()):
                destination = case_dir / f"{observation['obs_id']}_{suffix}.nii.gz"
                if destination.exists():
                    continue
                if observation["side"] == "gen":
                    self._resampler.write(source, destination)
                else:  # real side meets the contract natively: pass through
                    shutil.copyfile(source, destination)
        return self._output_root / "inputs"


class InstrumentFailureChecker:
    """input_fail / run_fail flags, semantics from calibration protocol §2/§5."""

    @staticmethod
    def input_fail(observation, input_dir):
        images = []
        for suffix in sorted(CHANNEL_SUFFIXES.values()):
            path = input_dir / f"{observation['obs_id']}_{suffix}.nii.gz"
            if not path.is_file():
                return True
            try:
                images.append(sitk.ReadImage(str(path)))
            except RuntimeError:
                return True
        reference = (images[0].GetSize(), images[0].GetSpacing(), images[0].GetOrigin())
        consistent = all((img.GetSize(), img.GetSpacing(), img.GetOrigin()) == reference for img in images[1:])
        isotropic = all(abs(s - 1.0) < 1e-3 for s in images[0].GetSpacing())
        if not (consistent and isotropic):
            return True
        if observation.get("condition_mask"):  # P2: the condition is part of the input contract
            try:
                sitk.ReadImage(str(observation["condition_mask"]))
            except (RuntimeError, OSError):
                return True
        return False

    @staticmethod
    def read_prediction(observation, pred_dir):
        path = pred_dir / f"{observation['obs_id']}.nii.gz"
        try:
            image = sitk.ReadImage(str(path))
            array = sitk.GetArrayFromImage(image)
        except (RuntimeError, OSError):
            return None
        if array.shape != PREDICTION_SHAPE:
            return None
        return array.astype(np.uint8, copy=False)

    @staticmethod
    def hierarchy_violation(pred):
        wt = np.isin(pred, (1, 2, 3))
        tc = np.isin(pred, (1, 3))
        et = pred == 3
        outside_domain = not np.isin(pred, (0, 1, 2, 3)).all()
        return bool(outside_domain or (et & ~tc).any() or (tc & ~wt).any())


class MaskMeasurer:
    """Region volumes, centroids, WT/brain, ET/WT and P2 condition round-trip dice."""

    @staticmethod
    def volumes_ml(pred):
        return {region: float(np.isin(pred, labels).sum()) * 0.001 for region, labels in REGION_LABELS.items()}

    @staticmethod
    def centroid_mm(pred, region):
        """Physical SRI24 centroid (x, y, z) mm; None when the region mask is empty."""
        mask = np.isin(pred, REGION_LABELS[region])
        if not mask.any():
            return None
        cz, cy, cx = ndimage.center_of_mass(mask)  # voxel indices on the 1mm grid
        return (float(cx), float(cy), float(cz))

    @staticmethod
    def brain_ml(input_dir, obs_id):
        """Four-channel union of non-zero voxels (protocol §3, rule pinned there)."""
        union = None
        for suffix in sorted(CHANNEL_SUFFIXES.values()):
            array = sitk.GetArrayFromImage(sitk.ReadImage(str(input_dir / f"{obs_id}_{suffix}.nii.gz")))
            nonzero = array > 0
            union = nonzero if union is None else (union | nonzero)
        return float(union.sum()) * 0.001

    @staticmethod
    def remap_condition(combined):
        """P2 combined mask ids (22/129/130/131) -> instrument labels (0/1/2/3)."""
        out = np.zeros_like(combined)
        for src, dst in COMBINED_TO_INSTRUMENT.items():
            out[combined == src] = dst
        return out

    @staticmethod
    def condition_dice(pred, condition, region):
        """Round-trip dice of the instrument prediction against the P2 condition mask.

        ``condition`` is already on the instrument grid (see label_to_grid):
        a condition mask that cannot be aligned is an input-contract failure of
        the observation, handled by the runner -- never a silent dice."""
        gt_mask = np.isin(condition, REGION_LABELS[region])
        pred_mask = np.isin(pred, REGION_LABELS[region])
        denom = int(gt_mask.sum()) + int(pred_mask.sum())
        if denom == 0:
            return None
        return float(2 * np.logical_and(gt_mask, pred_mask).sum() / denom)


class MeasurementRunner:
    """Runs every plan observation through the measurers into the judgement CSV."""

    def __init__(self, plan, input_root, pred_root):
        self._plan = plan
        self._input_root = Path(input_root)
        self._pred_root = Path(pred_root)
        self._checker = InstrumentFailureChecker()
        self._measurer = MaskMeasurer()
        self._resampler = GeneratedVolumeResampler()

    def measure_observation(self, observation):
        challenge_dir = self._input_root / observation["challenge"]
        row = {}
        input_fail = self._checker.input_fail(observation, challenge_dir)
        condition = None
        if observation.get("condition_mask"):
            # The P2 condition mask is part of the input contract: align it onto
            # the instrument grid (nearest neighbour); unalignable -> input_fail.
            condition = self._resampler.label_to_grid(observation["condition_mask"])
            if condition is None:
                input_fail = True
            else:
                condition = self._measurer.remap_condition(condition)
        pred = None if input_fail else self._checker.read_prediction(observation, self._pred_root / observation["challenge"])
        run_fail = pred is None
        row.update(
            obs_id=observation["obs_id"],
            challenge=observation["challenge"],
            case=observation["case"],
            side=observation["side"],
            anchor=observation["anchor"] or "",
            input_fail=int(input_fail),
            run_fail=int(run_fail),
            hier_viol=0 if pred is None else int(self._checker.hierarchy_violation(pred)),
            pred_empty="" if pred is None else int(not np.isin(pred, (1, 2, 3)).any()),
        )
        if pred is None:
            return row  # failure placeholder: counts in R_fail, quantities undefined
        volumes = self._measurer.volumes_ml(pred)
        brain = self._measurer.brain_ml(challenge_dir, observation["obs_id"])
        row.update(vol_wt_ml=volumes["WT"], vol_tc_ml=volumes["TC"], vol_et_ml=volumes["ET"], brain_ml=brain)
        row["wt_brain"] = volumes["WT"] / brain if brain > 0 else None
        row["et_wt"] = volumes["ET"] / volumes["WT"] if volumes["WT"] > 0 else None
        for region in REGIONS:
            centroid = self._measurer.centroid_mm(pred, region)
            values = {f"c{axis}_{region.lower()}_mm": (None if centroid is None else centroid[i]) for i, axis in enumerate("xyz")}
            row.update(values)
        if condition is not None:
            row.update({f"cond_dice_{region.lower()}": self._measurer.condition_dice(pred, condition, region) for region in REGIONS})
        return row

    def run(self):
        rows = [self.measure_observation(observation) for observation in self._plan["observations"]]
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("assemble-execute", help="plan -> instrument inputs (resample generated, pass real through)")
    p.add_argument("--plan", required=True)
    p.add_argument("--output-root", required=True)
    p.set_defaults(handler="assemble-execute")

    p = sub.add_parser("measure", help="plan + predictions -> measurement CSV for the judge")
    p.add_argument("--plan", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--pred-root", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(handler="measure")

    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())

    if args.handler == "assemble-execute":
        inputs = ObservationInputWriter(args.output_root).write_all(plan)
        print(f"[OK] instrument inputs -> {inputs} ({len(plan['observations'])} observations)")
        return 0
    rows = MeasurementRunner(plan, args.input_root, args.pred_root).run()
    path = MeasurementTable.write(rows, args.output)
    print(f"[OK] measurements -> {path} ({len(rows)} rows; controlled storage, subject ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
