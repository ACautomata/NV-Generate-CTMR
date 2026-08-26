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

"""Convergence-gate tests for the shared metric primitives (ADR-0010, #109).

``DiceScore`` carries the single empty-denominator sentinel: ``None``, aligned
with the frozen terminal-acceptance semantics (``MaskMeasurer.condition_dice``
returns None; the calibration mother's ``math.nan`` is the registered
divergence, collapse onto this module in #110). ``WilsonUpper`` is the single
``n == 0``-guarded formula: the reference snapshots are the two frozen copies
verbatim (calibration ``wilson_upper`` and the judge's ``FailureGate``), and
the module must reproduce them on the shared domain -- at ``n == 0`` the
frozen judge call site keeps its own ``None`` guard, so its behaviour is
unchanged by this module's return value.
"""

import math

import numpy as np
import pytest

from ctmr.measure.metrics import DiceScore, WilsonUpper


def test_dice_of_perfect_overlap_is_one():
    mask = np.array([True, True, False])
    assert DiceScore.of(mask, mask) == 1.0


def test_dice_of_disjoint_masks_is_zero():
    first = np.array([True, True, True])
    second = np.array([False, False, False])
    assert DiceScore.of(first, second) == 0.0


def test_dice_of_partial_overlap_matches_manual_formula():
    first = np.array([1, 1, 1, 1], dtype=np.uint8).astype(bool)
    second = np.array([1, 1, 1, 0], dtype=np.uint8).astype(bool)
    assert DiceScore.of(first, second) == 2 * 3 / 7


def test_dice_empty_denominator_sentinel_is_none():
    assert DiceScore.of(np.zeros(5, dtype=bool), np.zeros(5, dtype=bool)) is None


def test_dice_one_side_empty_only_is_zero_not_none():
    gt = np.array([True, True])
    assert DiceScore.of(gt, np.zeros(2, dtype=bool)) == 0.0


def test_dice_matches_the_frozen_condition_dice_reference():
    # The frozen reference: MaskMeasurer.condition_dice (scripts side, pre-#109)
    # -- single empty-denominator None sentinel, the semantic this module unifies.
    rng = np.random.default_rng(109)
    for _ in range(50):
        first = rng.random(64) > 0.7
        second = rng.random(64) > 0.7
        gt_mask, pred_mask = first, second
        denom = int(gt_mask.sum()) + int(pred_mask.sum())
        expected = None if denom == 0 else float(2 * np.logical_and(gt_mask, pred_mask).sum() / denom)
        assert DiceScore.of(first, second) == expected


# The frozen reference: nnunet_l2_calibration_metrics.wilson_upper, verbatim.
class CalibrationWilsonReference:
    Z95 = 1.959963984540054

    @classmethod
    def of(cls, successes, trials):
        if trials == 0:
            return math.nan
        probability = successes / trials
        denom = 1 + cls.Z95**2 / trials
        center = (probability + cls.Z95**2 / (2 * trials)) / denom
        half = (cls.Z95 / denom) * math.sqrt(probability * (1 - probability) / trials + cls.Z95**2 / (4 * trials**2))
        return min(1.0, center + half)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 1), (0, 10), (3, 10), (7, 10), (1, 100), (42, 100), (250, 250), (5, 5)],
)
def test_wilson_matches_the_frozen_calibration_reference(successes, trials):
    assert WilsonUpper.of(successes, trials) == CalibrationWilsonReference.of(successes, trials)


def test_wilson_zero_trials_returns_the_guarded_nan():
    assert math.isnan(WilsonUpper.of(0, 0))  # the single n==0 guard, frozen semantics


def test_wilson_is_capped_at_one_for_the_perfect_proportion():
    # Center + half term peaks just below 1.0 in floating point for k == n;
    # the frozen formula caps at min(1.0, ...), reproducing the reference
    # bit-for-bit.
    assert WilsonUpper.of(250, 250) == CalibrationWilsonReference.of(250, 250)
    assert WilsonUpper.of(250, 250) <= 1.0


def test_wilson_uses_the_frozen_z_value():
    assert WilsonUpper.Z95 == 1.959963984540054
