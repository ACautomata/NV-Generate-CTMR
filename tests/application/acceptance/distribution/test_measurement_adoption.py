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

"""Convergence-gate tests for the frozen measurement adoption (#224, ADR-0010 decision 6).

The two frozen measurement sites -- the terminal-acceptance execution side
(``measurement_run``) and the calibration mother (``calibration_metrics``) --
draw their measurement logic from the canonical ``InstrumentMeasurer``. These
tests pin the adoption seam on synthetic fixtures (any machine, no cluster):

- the measurement column-family equality (judge ``MEASUREMENT_FIELDS`` ==
  canonical ``FINAL_ACCEPTANCE_FIELDS``) -- the restored #110 drift trigger;
- the registered per-case CSV divergences (the ``hier_viol`` -> ``case_usable``
  rename, the Dice ``nan`` -> ``None`` sentinel, the ml-ratio ulp; ADR-0010
  decision 3/4 and consequences);
- caller-owned behaviour stays in place: failure placeholder rows and their
  sentinels, the input_fail/run_fail policies, the P2 remap, the file IO;
- a tiny end-to-end calibration run writes the frozen aggregate JSON shape
  (key names intact -- the byte-level rerun gate itself lives with the
  integration window, #233).

The frozen reference snapshots (tests/domain/measurement) pin the canonical
module itself against the pre-adoption implementations; this file pins the two
call sites onto that canonical standard.
"""

import json
import math

import numpy as np
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.calibration_metrics import main as calibration_main
from ctmr.application.acceptance.distribution.calibration_metrics import measure_case
from ctmr.application.acceptance.distribution.measurement_run import (
    COMBINED_TO_INSTRUMENT,
    GeneratedVolumeResampler,
    MeasurementRunner,
)
from ctmr.application.acceptance.distribution.measurement_table import MEASUREMENT_FIELDS
from ctmr.domain.measurement import (
    CALIBRATION_FIELDS,
    FINAL_ACCEPTANCE_FIELDS,
    HierarchyChecker,
    InstrumentMeasurer,
    WilsonUpper,
)

FULL_GRID = (155, 240, 240)  # zyx instrument prediction shape (measurement_run.PREDICTION_SHAPE)
SMALL_GRID = (12, 14, 16)  # the calibration side checks shape consistency only, not the grid


# ── fixtures ────────────────────────────────────────────────────────────


def _write_nifti(path, array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _tumour_array(shape, origin=(2, 3, 4)):
    """Well-formed nested tumour (WT⊃TC⊃ET: labels 1/2/3) at origin in a zero volume."""
    z, y, x = origin
    array = np.zeros(shape, dtype=np.uint8)
    array[z : z + 4, y : y + 5, x : x + 5] = 1  # core: WT + TC
    array[z + 1 : z + 3, y + 1 : y + 4, x + 1 : x + 4] = 2  # oedema: WT only
    array[z + 2 : z + 4, y + 2 : y + 4, x + 2 : x + 3] = 3  # ET
    return array


def _channel_array(shape, index):
    array = np.zeros(shape, dtype=np.uint8)
    z0 = 10 + 3 * index
    array[z0 : z0 + 8, 2:6, 2:6] = 7  # any non-zero voxels feed the brain union
    return array


def _terminal_observation(case, side="gen", condition_mask=None):
    obs_id = f"{case}__{side}" + ("__aL" if condition_mask and side == "gen" else "")
    return {
        "obs_id": obs_id,
        "challenge": "GLI",
        "case": case,
        "side": side,
        "anchor": "L" if obs_id.endswith("__aL") else None,
        "channels": {suffix: "" for suffix in ("0000", "0001", "0002", "0003")},
        "condition_mask": condition_mask,
    }


def _write_terminal_fixtures(input_root, pred_root, observation, pred=None, condition_path=None):
    """Writes the four instrument inputs, the prediction and (optionally) a P2 condition mask."""
    channels = [_channel_array(FULL_GRID, index) for index in range(4)]
    for suffix, channel in zip(("0000", "0001", "0002", "0003"), channels):
        _write_nifti(input_root / "GLI" / f"{observation['obs_id']}_{suffix}.nii.gz", channel)
    pred = _tumour_array(FULL_GRID, origin=(70, 115, 116)) if pred is None else pred
    _write_nifti(pred_root / "GLI" / f"{observation['obs_id']}.nii.gz", pred)
    if condition_path is not None:
        _write_nifti(condition_path, pred)
    return channels, pred


def _calibration_job(root, case="C001", write_pred=True, pred=None, gt=None):
    inputs_dir = root / "inputs" / "GLI"
    gt_dir = root / "gt" / "GLI"
    pred_dir = root / "predictions" / "GLI" / "rep1"
    gt = _tumour_array(SMALL_GRID) if gt is None else gt
    pred = _tumour_array(SMALL_GRID, origin=(3, 4, 5)) if pred is None else pred
    for index in range(4):
        _write_nifti(inputs_dir / f"{case}_{index:04d}.nii.gz", _channel_array(SMALL_GRID, index))
    _write_nifti(gt_dir / f"{case}.nii.gz", gt)
    if write_pred:
        _write_nifti(pred_dir / f"{case}.nii.gz", pred)
    return {"challenge": "GLI", "case": case, "source": "dev", "rep": 1, "inputs_dir": inputs_dir, "gt_dir": gt_dir, "pred_dir": pred_dir}, gt, pred


# ── the restored drift triggers (measurement column-family equality) ────


def test_judge_measurement_fields_equal_the_canonical_wide_schema():
    assert MEASUREMENT_FIELDS == FINAL_ACCEPTANCE_FIELDS


def test_calibration_csv_schema_carries_only_the_registered_rename():
    assert "case_usable" in CALIBRATION_FIELDS
    assert "hier_viol" not in CALIBRATION_FIELDS
    # the wide (terminal) schema keeps hier_viol -- the frozen ADR-0004 columns
    assert "hier_viol" in MEASUREMENT_FIELDS


# ── terminal-acceptance execution side (measurement_run) ────────────────


def test_terminal_runner_row_is_the_canonical_wide_row(tmp_path):
    input_root, pred_root = tmp_path / "inputs", tmp_path / "predictions"
    observation = _terminal_observation("CASE001")
    channels, pred = _write_terminal_fixtures(input_root, pred_root, observation)

    row = MeasurementRunner({"observations": [observation]}, input_root, pred_root).measure_observation(observation)

    expected = (
        InstrumentMeasurer()
        .measure(pred, condition=None, brain=channels)
        .to_wide_row(
            obs_id=observation["obs_id"],
            challenge="GLI",
            case="CASE001",
            side="gen",
            anchor=None,
            input_fail=0,
            run_fail=0,
        )
    )
    assert row == expected


def test_terminal_runner_p2_condition_row_is_the_canonical_wide_row(tmp_path):
    input_root, pred_root = tmp_path / "inputs", tmp_path / "predictions"
    condition_path = tmp_path / "condition.nii.gz"
    observation = _terminal_observation("CASE001", condition_mask=str(condition_path))
    channels, pred = _write_terminal_fixtures(input_root, pred_root, observation, condition_path=condition_path)

    row = MeasurementRunner({"observations": [observation]}, input_root, pred_root).measure_observation(observation)

    # the alignment/flip and the combined->instrument remap are caller-owned
    # input adaptation: the test drives the same collaborators, then checks the
    # measured row against the canonical serialization of the aligned condition
    aligned = GeneratedVolumeResampler().label_to_grid(condition_path)
    condition = MeasurementRunner.remap_condition(aligned)
    expected = (
        InstrumentMeasurer()
        .measure(pred, condition=condition, brain=channels)
        .to_wide_row(
            obs_id=observation["obs_id"],
            challenge="GLI",
            case="CASE001",
            side="gen",
            anchor="L",
            input_fail=0,
            run_fail=0,
        )
    )
    assert row == expected
    assert COMBINED_TO_INSTRUMENT == {22: 0, 129: 1, 130: 2, 131: 3}  # the remap table is frozen caller bookkeeping


def test_terminal_runner_placeholder_row_keeps_the_failure_sentinels(tmp_path):
    input_root, pred_root = tmp_path / "inputs", tmp_path / "predictions"
    observation = _terminal_observation("CASE001")
    _write_terminal_fixtures(input_root, pred_root, observation)
    (pred_root / "GLI" / f"{observation['obs_id']}.nii.gz").unlink()  # prediction missing -> run_fail

    row = MeasurementRunner({"observations": [observation]}, input_root, pred_root).measure_observation(observation)

    assert row == {
        "obs_id": observation["obs_id"],
        "challenge": "GLI",
        "case": "CASE001",
        "side": "gen",
        "anchor": "",
        "input_fail": 0,
        "run_fail": 1,
        "hier_viol": 0,  # undefined, not measured
        "pred_empty": "",  # the caller-owned placeholder sentinel (not an int)
    }


# ── calibration mother (calibration_metrics) ────────────────────────────


def test_calibration_measure_case_is_the_canonical_long_serialization(tmp_path):
    job, gt, pred = _calibration_job(tmp_path)

    rows = measure_case(job)

    expected = (
        InstrumentMeasurer().measure(pred, gt=gt).to_long_rows(challenge="GLI", case="C001", source="dev", rep=1, input_fail=False, run_fail=False)
    )
    assert rows == expected
    assert all(list(row.keys()) == CALIBRATION_FIELDS for row in rows)
    assert all(row["case_usable"] is True for row in rows)  # usable cases, in every region row
    # the registered ulp divergence: et_wt_ratio_pred is the ml-volume ratio of
    # the frozen terminal et_wt -- within a few ulp of the mother's count ratio
    # (ADR-0010 consequences), not merely approx-equal
    count_ratio = float((pred == 3).sum()) / float(np.isin(pred, (1, 2, 3)).sum())
    assert abs(rows[0]["et_wt_ratio_pred"] - count_ratio) <= 8 * math.ulp(count_ratio)


def test_calibration_measure_case_placeholder_rows_on_run_fail(tmp_path):
    job, _, _ = _calibration_job(tmp_path, write_pred=False)  # prediction file missing -> run_fail

    rows = measure_case(job)

    assert len(rows) == 3
    for row in rows:
        assert row["run_fail"] is True
        assert row["input_fail"] is False
        assert row["case_usable"] is None  # unmeasured, not False -- failed rows never enter the hier_viol breakdown (ADR-0002)
        assert row["detected"] is None
        assert row["dice"] is None
        assert row["vol_pred_ml"] is None


def test_calibration_usability_gate_is_not_hierarchy_violation(tmp_path):
    pred = _tumour_array(SMALL_GRID, origin=(3, 4, 5))
    pred[5, 5, 5] = 9  # value-domain violation: unusable for calibration AND a hierarchy violation
    job, _, _ = _calibration_job(tmp_path, pred=pred)

    rows = measure_case(job)

    assert all(row["case_usable"] is False for row in rows)
    assert HierarchyChecker.violates(pred) is True


def test_calibration_dice_none_sentinel_on_the_calibrated_csv_path(tmp_path):
    """A both-side empty region (GT and pred without ET) hits the registered
    nan->None Dice sentinel on the calibration path itself, while the case
    stays usable (domains OK, GT WT present)."""
    gt = _tumour_array(SMALL_GRID)
    gt[gt == 3] = 0
    pred = _tumour_array(SMALL_GRID, origin=(3, 4, 5))
    pred[pred == 3] = 0
    job, _, _ = _calibration_job(tmp_path, gt=gt, pred=pred)

    rows = {row["region"]: row for row in measure_case(job)}

    assert rows["ET"]["dice"] is None
    assert rows["WT"]["dice"] is not None and rows["TC"]["dice"] is not None
    assert rows["ET"]["case_usable"] is True


def test_calibration_wilson_is_the_single_guarded_definition(tmp_path):
    root = tmp_path / "cal"
    _calibration_job(root)  # the fixture tree is what matters below

    manifest = {"cases": [{"case": "C001", "source": "dev"}]}
    (root / "protocol").mkdir(parents=True, exist_ok=True)
    (root / "protocol" / "calibration_cases_GLI.json").write_text(json.dumps(manifest))

    calibration_main(["--calibration-root", str(root), "--challenge", "GLI", "--reps", "1", "--workers", "1"])

    summary = json.loads((root / "metrics" / "summary_GLI.json").read_text())
    # the frozen aggregate JSON keeps its key names (byte-level rerun gate: #233)
    assert summary["R_fail"]["breakdown"] == {"input_fail": 0, "run_fail": 0, "hier_viol": 0}
    assert summary["R_fail"]["k"] == 0 and summary["R_fail"]["n"] == 1
    assert summary["R_fail"]["wilson_95_upper"] == WilsonUpper.of(0, 1)
    assert summary["R_miss"] == {"k": 0, "n": 1, "point": 0.0}
    measurement = InstrumentMeasurer().measure(
        sitk.GetArrayFromImage(sitk.ReadImage(str(root / "predictions" / "GLI" / "rep1" / "C001.nii.gz"))),
        gt=sitk.GetArrayFromImage(sitk.ReadImage(str(root / "gt" / "GLI" / "C001.nii.gz"))),
    )
    assert summary["per_region"]["WT"]["dice_median"] == measurement.gt_metrics["WT"].dice

    with (root / "metrics" / "GLI" / "per_case_GLI_rep1.csv").open() as handle:
        header = handle.readline().strip().split(",")
    assert header == CALIBRATION_FIELDS


def test_calibration_r_fail_breakdown_keeps_the_mother_semantics(tmp_path):
    """C001 usable, C002 run-fail (placeholder), C003 measured-but-unusable: the
    mother's breakdown counted the hier_viol component only for measured cases
    (a failed row has no arrays to check) -- the rename must not drift that."""
    root = tmp_path / "cal"
    _calibration_job(root, case="C001")
    _calibration_job(root, case="C002", write_pred=False)
    bad_pred = _tumour_array(SMALL_GRID, origin=(3, 4, 5))
    bad_pred[5, 5, 5] = 9
    _calibration_job(root, case="C003", pred=bad_pred)

    manifest = {"cases": [{"case": "C001", "source": "dev"}, {"case": "C002", "source": "dev"}, {"case": "C003", "source": "dev"}]}
    (root / "protocol").mkdir(parents=True, exist_ok=True)
    (root / "protocol" / "calibration_cases_GLI.json").write_text(json.dumps(manifest))

    calibration_main(["--calibration-root", str(root), "--challenge", "GLI", "--reps", "1", "--workers", "1"])

    summary = json.loads((root / "metrics" / "summary_GLI.json").read_text())
    assert summary["R_fail"]["n"] == 3 and summary["R_fail"]["k"] == 2
    assert summary["R_fail"]["breakdown"] == {"input_fail": 0, "run_fail": 1, "hier_viol": 1}
    assert summary["R_miss"] == {"k": 0, "n": 3, "point": 0.0}
