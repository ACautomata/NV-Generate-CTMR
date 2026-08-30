"""Issue #55 NIfTI execution side for the L2 final-acceptance pipeline.

(needs SimpleITK / numpy / scipy; runs where the predictions were produced).
Every judgement rule lives in ``final_acceptance`` (stdlib-only). Two commands:

  assemble-execute   plan -> instrument inputs at ``<output-root>/inputs/<CH>/``
                     (generated side: the issue #38 ``InputPreparator``
                     geometry verbatim -- resample to 1mm B-spline, centre
                     crop/pad to 240x240x155; real side: byte-identical
                     pass-through, the native BraTS volumes already meet the
                     instrument contract). One file per observation and channel:
                     ``{obs_id}_{suffix}.nii.gz``.
  measure            plan + predictions -> measurement CSV (schema shared with
                     the judge). Measurement logic is the canonical
                     ``InstrumentMeasurer`` (ADR-0010, #224); this shell keeps
                     the caller-owned execution concerns -- input_fail/run_fail
                     policies, failure placeholder rows and their sentinels,
                     the P2 combined-mask remap, the DM-RAS->LPS flip and all
                     file IO. Failure flags (input_fail / run_fail / hier_viol)
                     are checked on BOTH sides; empty predictions are
                     measurement results, not failures.

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

from ctmr.application.acceptance.distribution.final_acceptance import CHANNEL_SUFFIXES, MeasurementTable
from ctmr.domain.grid import INSTRUMENT_GRID, InstrumentGridAdapter  # noqa: E402
from ctmr.domain.measurement import InstrumentMeasurer

NNUNET_TARGET_SIZE = INSTRUMENT_GRID.size
PREDICTION_SHAPE = tuple(reversed(NNUNET_TARGET_SIZE))  # array layout is zyx

# P2 condition combined mask -> instrument label space (mirrors the dev sidecar
# COMBINED_TO_INSTRUMENT in ctmr.application.generation.mask.monitor). The
# combined mask is stored in the BraTS 2023 label ids (22/129/130/131); the
# instrument predicts 0/1/2/3, so the round-trip Dice must remap first --
# comparing raw ids against REGION_LABELS (1/2/3) matches nothing and yields a
# spurious exact 0.
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


class MeasurementRunner:
    """Runs every plan observation through the canonical measurer into the judgement CSV.

    Measurement is ``InstrumentMeasurer.measure`` + the wide serialization
    (ADR-0010); the caller-owned execution concerns stay here -- the
    input_fail/run_fail policies, the failure placeholder row with its
    sentinels, the P2 combined-mask remap, the brain-channel file reads.
    """

    def __init__(self, plan, input_root, pred_root):
        self._plan = plan
        self._input_root = Path(input_root)
        self._pred_root = Path(pred_root)
        self._checker = InstrumentFailureChecker()
        self._measurer = InstrumentMeasurer()
        self._resampler = GeneratedVolumeResampler()

    @staticmethod
    def remap_condition(combined):
        """P2 combined mask ids (22/129/130/131) -> instrument labels (0/1/2/3)."""
        out = np.zeros_like(combined)
        for src, dst in COMBINED_TO_INSTRUMENT.items():
            out[combined == src] = dst
        return out

    def _brain_channels(self, challenge_dir, obs_id):
        """Reads the four input channels for the brain column family (file IO stays with the caller)."""
        return [
            sitk.GetArrayFromImage(sitk.ReadImage(str(challenge_dir / f"{obs_id}_{suffix}.nii.gz"))) for suffix in sorted(CHANNEL_SUFFIXES.values())
        ]

    def measure_observation(self, observation):
        challenge_dir = self._input_root / observation["challenge"]
        input_fail = self._checker.input_fail(observation, challenge_dir)
        condition = None
        if observation.get("condition_mask"):
            # The P2 condition mask is part of the input contract: align it onto
            # the instrument grid (nearest neighbour); unalignable -> input_fail.
            condition = self._resampler.label_to_grid(observation["condition_mask"])
            if condition is None:
                input_fail = True
            else:
                condition = self.remap_condition(condition)
        pred = None if input_fail else self._checker.read_prediction(observation, self._pred_root / observation["challenge"])
        run_fail = pred is None
        if pred is None:
            # failure placeholder: counts in R_fail, quantities undefined (caller-owned sentinels)
            return {
                "obs_id": observation["obs_id"],
                "challenge": observation["challenge"],
                "case": observation["case"],
                "side": observation["side"],
                "anchor": observation["anchor"] or "",
                "input_fail": int(input_fail),
                "run_fail": int(run_fail),
                "hier_viol": 0,
                "pred_empty": "",
            }
        measurement = self._measurer.measure(pred, condition=condition, brain=self._brain_channels(challenge_dir, observation["obs_id"]))
        return measurement.to_wide_row(
            obs_id=observation["obs_id"],
            challenge=observation["challenge"],
            case=observation["case"],
            side=observation["side"],
            anchor=observation["anchor"] or "",
            input_fail=int(input_fail),
            run_fail=int(run_fail),
        )

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
