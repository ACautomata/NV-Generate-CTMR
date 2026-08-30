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

"""Synthetic-domain measurement convergence gates (issue #223, ADR-0010 decision 6).

The #38 synthetic-domain evaluation draws its measurement primitives (region
literals, Wilson, Dice, hierarchy-violation verdict) from the canonical
``ctmr.domain.measurement`` module from now on, so the synthetic-domain chain
and the frozen terminal-acceptance chain can never drift apart semantically.
These gates pin the convergence at the live call site: the hierarchy count
equals ``HierarchyChecker.violates`` on crafted inputs (including the
ET-present-TC-empty regression pin for the guard-free canonical shape,
ADR-0010 decision 3), the per-region Dice is ``DiceScore.of`` with the
single ``None`` empty-denominator sentinel, and the R_fail Wilson bound is
``WilsonUpper.of``. The input_fail/run_fail chain, failure placeholder rows
and the file IO stay call-site concerns (issue #219 semantics diff, settled).
Light stack, any machine.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.synthetic_domain import MetricsCalculator
from ctmr.domain.measurement import DiceScore, HierarchyChecker, RegionMasks, WilsonUpper

GRID = (6, 6, 6)  # crafted zyx volume; 1 mm isotropic keeps input_fail False


def _write_case(tmp_path, pred_arr, gt_arr=None):
    """One synthetic case on disk: four consistent isotropic inputs + pred (+ optional GT)."""
    inputs_dir = tmp_path / "inputs" / "GLI"
    pred_dir = tmp_path / "preds" / "GLI"
    inputs_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    for suffix in ("0000", "0001", "0002", "0003"):
        image = sitk.GetImageFromArray(np.zeros(GRID, dtype=np.float32))
        image.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(image, str(inputs_dir / f"CASE_{suffix}.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(pred_arr), str(pred_dir / "CASE.nii.gz"))
    if gt_arr is None:
        return None
    gt_dir = tmp_path / "gt" / "GLI"
    gt_dir.mkdir(parents=True)
    sitk.WriteImage(sitk.GetImageFromArray(gt_arr), str(gt_dir / "CASE.nii.gz"))
    return gt_dir


def _evaluate(tmp_path, pred_arr, gt_arr=None):
    gt_dir = _write_case(tmp_path, pred_arr, gt_arr)
    return MetricsCalculator().evaluate_case("CASE", "GLI", tmp_path / "inputs" / "GLI", tmp_path / "preds" / "GLI", gt_dir)


def _well_formed():
    pred = np.zeros(GRID, dtype=np.uint8)
    pred[0, 0, 0] = 3  # ET inside TC
    pred[0, 0, 1] = 1  # TC-only, inside WT
    pred[1, 1, 1] = 2  # WT-only oedema
    return pred


def _et_present_tc_empty():
    """ET voxels present, no TC-only voxel anywhere: the crafted superset-empty case."""
    pred = np.zeros(GRID, dtype=np.uint8)
    pred[0, 0, 0] = 3
    return pred


def _domain_escape():
    pred = _well_formed()
    pred[2, 2, 2] = 4  # not a BraTS 2023 label
    return pred


def test_synthetic_hierarchy_count_matches_the_canonical_single_expression(tmp_path):
    """Crafted-input equivalence: the call-site hierarchy count equals
    ``HierarchyChecker.violates`` on every case. The ET-present-TC-empty input
    (ET voxels, no TC-only voxel) is the #219-named regression pin: under the
    current nested BraTS projection (ET{3} ⊂ TC{1,3}) an empty TC projection is
    unconstructible, so both implementations read False -- the case pins the
    guard-free canonical shape (no empty-superset exemption branch, ADR-0010
    decision 3) and catches an exemption bug if the REGIONS nesting ever drifts."""
    cases = {
        "well_formed": _well_formed(),
        "all_zero": np.zeros(GRID, dtype=np.uint8),
        "et_present_tc_empty": _et_present_tc_empty(),
        "domain_escape": _domain_escape(),
    }
    for name, pred in cases.items():
        result = _evaluate(tmp_path / name, pred)
        assert result["input_fail"] is False and result["run_fail"] is False, name
        assert result["hier_viol"] == HierarchyChecker.violates(pred), name
    # the equivalence is not vacuous: the escaped value flips the canonical verdict
    assert HierarchyChecker.violates(cases["domain_escape"])
    assert not HierarchyChecker.violates(cases["et_present_tc_empty"])


def test_per_region_dice_is_the_canonical_score_with_none_sentinel(tmp_path):
    """The per-region Dice column is ``DiceScore.of`` on canonical projections;
    a both-empty region reads ``None`` (the single sentinel), never a number."""
    pred = np.zeros(GRID, dtype=np.uint8)
    pred[0, 0, 0] = 3
    pred[0, 0, 1] = 1
    gt = np.zeros(GRID, dtype=np.uint8)
    gt[0, 0, 0] = 3
    gt[1, 1, 1] = 2  # GT WT-only voxel: pred misses it -> partial WT dice

    result = _evaluate(tmp_path, pred, gt)

    assert set(result["per_region"]) == {"WT", "TC", "ET"}
    for region in ("WT", "TC", "ET"):
        expected = DiceScore.of(RegionMasks(gt).of(region), RegionMasks(pred).of(region))
        assert result["per_region"][region]["dice"] == expected, region
    # WT: 1 shared voxel of 2 GT + 2 pred; TC: 1 shared of 1 GT + 2 pred; ET: perfect match
    assert result["per_region"]["WT"]["dice"] == pytest.approx(2 * 1 / (2 + 2))
    assert result["per_region"]["TC"]["dice"] == pytest.approx(2 * 1 / (1 + 2))
    assert result["per_region"]["ET"]["dice"] == 1.0


def test_per_region_dice_both_empty_reads_none(tmp_path):
    """No GT and no pred in a region -> the Dice sentinel is None."""
    pred = np.zeros(GRID, dtype=np.uint8)
    gt = np.zeros(GRID, dtype=np.uint8)

    result = _evaluate(tmp_path, pred, gt)

    for region in ("WT", "TC", "ET"):
        assert result["per_region"][region]["dice"] is None, region


def test_r_fail_wilson_bound_is_the_canonical_wilson():
    """The R_fail Wilson upper bound comes from ``WilsonUpper.of`` (the single
    guarded definition), with the input/run/hierarchy breakdown intact."""
    results = [
        {"input_fail": True, "run_fail": False, "hier_viol": False},
        {"input_fail": False, "run_fail": False, "hier_viol": True},
        {"input_fail": False, "run_fail": False, "hier_viol": False},
    ]

    r_fail = MetricsCalculator().compute_r_fail(results)

    assert (r_fail["k"], r_fail["n"]) == (2, 3)
    assert r_fail["wilson_95_upper"] == WilsonUpper.of(2, 3)
    assert r_fail["breakdown"] == {"input_fail": 1, "run_fail": 0, "hier_viol": 1}
