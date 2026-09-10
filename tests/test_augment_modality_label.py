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

"""Behavioral confirmation that BRATS labels 40–43 follow the MRI augment branch
(ticket T3, issue #19; spec #13 §④.6).

``augment_modality_label`` is expected to treat the new labels exactly like the
old MRI labels (>=9): ~10% collapse to 8 (``mri`` generic embedding) and ~10%
zero out (CFG unconditional) — never the CT collapse to 1. Requires torch;
the guard keeps environments without torch (e.g. minimal local setups) green.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from scripts.diff_model_train import augment_modality_label  # noqa: E402

BRATS_LABELS = (40, 41, 42, 43)
OLD_MRI_LABELS = (9, 10, 11, 16, 20, 29, 30, 31, 32, 33)


class TestAugmentModalityLabelBratsLabels:
    @staticmethod
    def make_tensor(labels: tuple[int, ...], repeats: int = 2000) -> torch.Tensor:
        return torch.tensor(labels * repeats).reshape(-1, 1)

    def test_zero_prob_leaves_labels_untouched(self) -> None:
        result = augment_modality_label(self.make_tensor(BRATS_LABELS), prob=0.0)

        assert set(result.unique().tolist()) == set(BRATS_LABELS)

    def test_default_prob_output_domain_is_mri_consistent(self) -> None:
        torch.manual_seed(42)
        result = augment_modality_label(self.make_tensor(BRATS_LABELS))

        # Permitted outcomes: unchanged label, generic-MRI 8, CFG-unconditional 0.
        # 1 (the CT collapse) must never appear.
        assert set(result.unique().tolist()) <= {0, 8, *BRATS_LABELS}

    def test_brats_labels_match_old_mri_labels_element_wise_under_same_seed(self) -> None:
        # The augment's random masks depend only on the tensor shape, not values, so
        # under one seed a 40–43 tensor and a 9–12 tensor must receive the *same*
        # collapse-to-8 / zero-out pattern — i.e. the new labels take the identical
        # MRI code path as the old ones.
        probe = (9, 10, 11, 12)  # old MRI labels shifted to match 40–43 slot by slot
        brats_tensor = self.make_tensor(BRATS_LABELS)
        torch.manual_seed(7)
        brats_result = augment_modality_label(brats_tensor)
        torch.manual_seed(7)
        probe_result = augment_modality_label(self.make_tensor(probe))

        assert (brats_result == 0).equal(probe_result == 0), "zero-out (CFG) pattern diverges from old MRI labels"
        assert (brats_result == 8).equal(probe_result == 8), "collapse-to-8 (generic MRI) pattern diverges from old MRI labels"
        kept = (brats_result != 8) & (brats_result != 0)
        assert brats_result[kept].equal(brats_tensor[kept]), "surviving elements must keep their own label"

    def test_both_mri_collapse_paths_fire_and_ct_never_does(self) -> None:
        # Across many seeds the outcome domain stays inside {0, 8, own labels} and
        # both MRI-branch outcomes (collapse to 8, zero-out to 0) actually occur —
        # the label 1 fingerprint of the CT branch (2..7 -> 1) never does.
        outputs = set()
        for seed in range(20):
            torch.manual_seed(seed)
            outputs |= set(augment_modality_label(self.make_tensor(BRATS_LABELS, repeats=500)).unique().tolist())

        assert outputs <= {0, 8, *BRATS_LABELS}
        assert 8 in outputs, "collapse-to-8 (generic MRI) never triggered"
        assert 0 in outputs, "zero-out (CFG unconditional) never triggered"

    def test_old_mri_labels_share_the_same_output_domain(self) -> None:
        # Guard the invariant from the other side: old labels must keep behaving
        # identically (no accidental code change flips either group's branch).
        torch.manual_seed(42)
        result = augment_modality_label(self.make_tensor(OLD_MRI_LABELS))

        outcomes = set(result.unique().tolist())
        assert outcomes <= {0, 8, *OLD_MRI_LABELS}
        assert 8 in outcomes, "old MRI labels no longer collapse to 8 — branch behavior changed"
        assert 0 in outcomes, "old MRI labels no longer zero out — CFG behavior changed"
