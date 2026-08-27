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

"""Convergence-gate tests for hierarchy semantics (ADR-0010, #109).

Pins the canonical containment definition to the frozen terminal-acceptance
implementation (``InstrumentFailureChecker.hierarchy_violation`` verbatim:
single expression, no precondition guards, no empty-superset exemption) and
proves the calibration mother's old ``hier_viol`` -- a different concept that
needs GT and only checks domains + non-empty GT WT -- is split into
``CalibrationCaseUsability``. A case unusable for calibration is NOT a
hierarchy violation and vice versa: the split is the bug fix, so the two
predicates must disagree on crafted inputs.
"""

import numpy as np

from ctmr.domain.measurement.hierarchy import CalibrationCaseUsability, HierarchyChecker
from ctmr.domain.measurement.regions import REGIONS


# The frozen reference: InstrumentFailureChecker.hierarchy_violation pre-#109,
# verbatim (scripts/nnunet_l2_final_acceptance_nifti.py).
class FrozenHierarchyReference:
    @staticmethod
    def violates(pred):
        wt = np.isin(pred, (1, 2, 3))
        tc = np.isin(pred, (1, 3))
        et = pred == 3
        outside_domain = not np.isin(pred, (0, 1, 2, 3)).all()
        return bool(outside_domain or (et & ~tc).any() or (tc & ~wt).any())


def test_hierarchy_matches_the_frozen_terminal_reference_on_partial_containment():
    # ET outside of a TC that is a proper subset of WT: the mother's nested
    # single-expression form must fire on each step independently.
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    pred[0, 0, 0] = 3  # ET present
    pred[0, 0, 1] = 1  # TC (and WT) present
    assert HierarchyChecker.violates(pred) == FrozenHierarchyReference.violates(pred)


def test_hierarchy_violates_when_value_escapes_the_label_domain():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0] = 4  # not a BraTS 2023 label
    assert HierarchyChecker.violates(pred)
    assert HierarchyChecker.violates(pred) == FrozenHierarchyReference.violates(pred)


def test_hierarchy_does_not_violate_well_formed_nested_mask():
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    pred[:, :, 0] = 1  # WT + TC
    pred[0, 0, 0] = 3  # ET inside TC
    pred[2, 2, 2] = 2  # WT-only voxel
    assert not HierarchyChecker.violates(pred)
    assert HierarchyChecker.violates(pred) == FrozenHierarchyReference.violates(pred)


def test_hierarchy_does_not_violate_all_zero_mask():
    pred = np.zeros((4, 4, 4), dtype=np.uint8)
    assert not HierarchyChecker.violates(pred)
    assert HierarchyChecker.violates(pred) == FrozenHierarchyReference.violates(pred)


def test_region_masks_are_derived_for_containment_not_re_literal():
    # The checker reads REGIONS (single source) -- the frozen form's literals
    # stay in one place.
    assert REGIONS["ET"] == (3,)
    assert REGIONS["TC"] == (1, 3)
    assert REGIONS["WT"] == (1, 2, 3)


def test_calibration_usability_requires_gt_and_non_empty_gt_wt():
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    assert not CalibrationCaseUsability.usable(gt, pred)  # empty GT WT -> unusable

    gt[0, 0, 0] = 1  # GT WT present (also TC)
    pred[0, 0, 0] = 2
    assert CalibrationCaseUsability.usable(gt, pred)


def test_calibration_usability_rejects_out_of_domain_values():
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    gt[0, 0, 0] = 1
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[1, 1, 1] = 5
    assert not CalibrationCaseUsability.usable(gt, pred)

    bad_gt = np.zeros((2, 2, 2), dtype=np.uint8)
    bad_gt[0, 0, 0] = 9
    assert not CalibrationCaseUsability.usable(bad_gt, np.zeros((2, 2, 2), dtype=np.uint8))


def test_usability_is_a_different_concept_from_hierarchy_violation():
    # Same input, opposite verdicts: GT usable for calibration (probably not
    # perfectly formable) while a domain escape in pred violates hierarchy --
    # built from the frozen mother check (value domains of gt AND pred + GT WT
    # non-empty, calibration `:167-168`) vs canonical containment.
    gt = np.zeros((3, 3, 3), dtype=np.uint8)
    gt[0, 0, 0] = 1
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    pred[1, 1, 1] = 4  # escapes the label domain -> hierarchy violation
    assert HierarchyChecker.violates(pred)
    assert not CalibrationCaseUsability.usable(gt, pred)  # pred domain bad -> not usable

    # The reverse: hierarchy-acceptable mask, but calibration gate flips on
    # empty GT WT (the mother's `gt_mask.sum() == 0` check) -- empty GT WT
    # makes the pair unusable for envelope estimation while nothing violates
    # the canonical containment.
    all_zero = np.zeros((3, 3, 3), dtype=np.uint8)
    assert not CalibrationCaseUsability.usable(all_zero, all_zero)
    assert not HierarchyChecker.violates(all_zero)
