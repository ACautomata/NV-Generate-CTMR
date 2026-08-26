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

"""Convergence-gate tests for the canonical value object and its two serializations (ADR-0010, #109).

Pins ``CaseMeasurement.to_wide_row`` to the frozen terminal-acceptance schema
(judge ``MEASUREMENT_FIELDS``, 27 columns, verbatim) and the long rows to the
calibration mother schema -- with the one registered rename
(``hier_viol`` -> ``case_usable``; ADR-0010 decision 3). The execution flags
and observation identity are caller bookkeeping, passed in at serialization.
"""

import pytest

from ctmr.measure.measurement import CALIBRATION_FIELDS, FINAL_ACCEPTANCE_FIELDS, CaseMeasurement, GtRegionMetrics


def _sample_measurement(**overrides):
    base = dict(
        hierarchy_violation=True,
        pred_empty=False,
        volumes_ml={"WT": 50.0, "TC": 20.0, "ET": 5.0},
        centroids_mm={"WT": (120.5, 121.5, 77.5), "TC": (120.0, 121.0, 77.0), "ET": None},
        et_wt=0.1,
        brain_ml=1200.0,
        wt_brain=50.0 / 1200.0,
        condition_dice={"WT": 0.9, "TC": 0.8, "ET": None},
        gt_metrics={
            "WT": GtRegionMetrics(
                vol_gt_ml=52.0,
                vol_pred_ml=50.0,
                signed_bias_ml=-2.0,
                abs_err_ml=2.0,
                rel_vol_err=2.0 / 52.0,
                dice=0.95,
                sensitivity=0.9,
                precision=0.93,
                hd95_mm=4.2,
                centroid_distance_mm=1.1,
                n_components_gt=1,
                n_components_pred=2,
                n_false_positive_components=1,
            ),
            "TC": GtRegionMetrics(
                vol_gt_ml=21.0,
                vol_pred_ml=20.0,
                signed_bias_ml=-1.0,
                abs_err_ml=1.0,
                rel_vol_err=1.0 / 21.0,
                dice=0.85,
                sensitivity=0.8,
                precision=0.84,
                hd95_mm=3.5,
                centroid_distance_mm=2.0,
                n_components_gt=1,
                n_components_pred=1,
                n_false_positive_components=None,  # WT-only field
            ),
            "ET": GtRegionMetrics(
                vol_gt_ml=5.5,
                vol_pred_ml=5.0,
                signed_bias_ml=-0.5,
                abs_err_ml=0.5,
                rel_vol_err=0.5 / 5.5,
                dice=0.9,
                sensitivity=0.85,
                precision=0.9,
                hd95_mm=2.0,
                centroid_distance_mm=3.0,
                n_components_gt=1,
                n_components_pred=1,
                n_false_positive_components=None,
            ),
        },
        calibration_usable=True,
        et_wt_ratio_gt=5.5 / 52.0,
    )
    base.update(overrides)
    return CaseMeasurement(**base)


def test_wide_row_matches_the_frozen_terminal_schema():
    measurement = _sample_measurement()
    row = measurement.to_wide_row(obs_id="OBS-1", challenge="GLI", case="C-1", side="gen", anchor="L")
    assert list(row.keys()) == FINAL_ACCEPTANCE_FIELDS
    assert row == {
        "obs_id": "OBS-1",
        "challenge": "GLI",
        "case": "C-1",
        "side": "gen",
        "anchor": "L",
        "input_fail": 0,
        "run_fail": 0,
        "hier_viol": 1,
        "pred_empty": 0,
        "vol_wt_ml": 50.0,
        "vol_tc_ml": 20.0,
        "vol_et_ml": 5.0,
        "brain_ml": 1200.0,
        "wt_brain": 50.0 / 1200.0,
        "et_wt": 0.1,
        "cx_wt_mm": 120.5,
        "cy_wt_mm": 121.5,
        "cz_wt_mm": 77.5,
        "cx_tc_mm": 120.0,
        "cy_tc_mm": 121.0,
        "cz_tc_mm": 77.0,
        "cx_et_mm": None,  # empty ET region -> cell blank in the frozen CSV
        "cy_et_mm": None,
        "cz_et_mm": None,
        "cond_dice_wt": 0.9,
        "cond_dice_tc": 0.8,
        "cond_dice_et": None,
    }


def test_wide_row_backfills_caller_flags_and_anchor_default():
    row = _sample_measurement().to_wide_row(obs_id="O", challenge="SSA", case="C", side="real", input_fail=1, run_fail=0)
    assert row["input_fail"] == 1
    assert row["run_fail"] == 0
    assert row["anchor"] == ""


def test_long_rows_carry_one_row_per_region_with_mother_schema():
    rows = _sample_measurement().to_long_rows(challenge="GLI", case="C-1", source="dev", rep=1)
    assert [row["region"] for row in rows] == ["WT", "TC", "ET"]
    assert all(list(row.keys()) == CALIBRATION_FIELDS for row in rows)


def test_long_rows_keep_usability_as_case_usable_not_hier_viol():
    rows = _sample_measurement(calibration_usable=False).to_long_rows(challenge="GLI", case="C", source="dev", rep=1)
    assert all(row["case_usable"] is False for row in rows)
    assert "hier_viol" not in rows[0]


def test_long_rows_refuse_without_the_gt_gated_family():
    measurement = _sample_measurement(gt_metrics=None, calibration_usable=None, et_wt_ratio_gt=None)
    with pytest.raises(ValueError):
        measurement.to_long_rows(challenge="GLI", case="C", source="dev", rep=1)


def test_long_row_wt_only_n_fp_comp_field_is_none_off_wt():
    rows = _sample_measurement().to_long_rows(challenge="GLI", case="C", source="dev", rep=1)
    by_region = {row["region"]: row for row in rows}
    assert by_region["WT"]["n_fp_comp"] == 1
    assert by_region["TC"]["n_fp_comp"] is None
    assert by_region["ET"]["n_fp_comp"] is None


def test_closed_column_families_serialize_as_none():
    measurement = _sample_measurement(brain_ml=None, wt_brain=None, condition_dice=None, gt_metrics=None)
    wide = measurement.to_wide_row(obs_id="O", challenge="GLI", case="C", side="gen")
    assert wide["brain_ml"] is None
    assert wide["wt_brain"] is None
    assert wide["cond_dice_wt"] is None
    assert wide["cond_dice_tc"] is None
    assert wide["cond_dice_et"] is None


def test_the_two_schemas_are_the_union_of_the_six_sites():
    # Wide == judge columns; long == mother columns with the one rename. The
    # exact sets are the drift anchor for #110's 6-site consolidation. The two
    # schemas share only the caller identity/flag columns.
    assert len(FINAL_ACCEPTANCE_FIELDS) == 27
    assert len(CALIBRATION_FIELDS) == 24
    assert set(FINAL_ACCEPTANCE_FIELDS) & set(CALIBRATION_FIELDS) == {"challenge", "case", "input_fail", "run_fail"}
