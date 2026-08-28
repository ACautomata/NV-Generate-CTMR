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

"""ModalityLabelPerturber gates (ADR-0016 generation objective, issue #170).

The perturbation semantics must stay the P1-pinned recipe (augment prob 0.1,
CT members → 1, MR members → 8, prob-decided zeroing) and numerically match
the vendored upstream ``augment_modality_label`` the migrated P1 entry used
(seed-replayed).  Torch-level: real execution on CPU.
"""

from __future__ import annotations

import pytest
import torch

from ctmr.domain.generation.objective import ModalityLabelPerturber
from ctmr.infrastructure.maisi_engine.diff_model_train import augment_modality_label

pytestmark = pytest.mark.torch

# the pinned augmentation probability (recipe code-literal, ADR-0005 kernel)
PINNED_PROB = 0.1


def test_pinned_augmentation_probability():
    assert ModalityLabelPerturber().prob == pytest.approx(PINNED_PROB)


def test_zero_prob_is_identity():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[9.0]], [[13.0]]]])
    torch.manual_seed(0)
    out = ModalityLabelPerturber(prob=0.0)(base.clone())
    assert torch.equal(out, base)


def test_one_prob_zeroes_every_element_via_the_final_mask():
    # prob=1 makes the final mask (rand > 1) all-False, so the whole tensor is
    # zeroed -- the last rule is observable directly; the 1/8 relabelling rules
    # are pinned by the numerics parity test below.
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[9.0]], [[13.0]]]])
    torch.manual_seed(7)
    out = ModalityLabelPerturber(prob=1.0)(base.clone())
    assert torch.all(out == 0.0)


def test_matches_the_vendored_legacy_augmentation_seed_replayed():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[5.0]], [[7.0]], [[8.0]], [[9.0]], [[12.0]], [[13.0]], [[20.0]]]])
    torch.manual_seed(11)
    legacy = augment_modality_label(base.clone(), prob=PINNED_PROB)
    torch.manual_seed(11)
    migrated = ModalityLabelPerturber()(base.clone())
    assert torch.equal(migrated, legacy)


def test_is_seed_deterministic_like_the_legacy():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[9.0]], [[13.0]]]])
    torch.manual_seed(7)
    first = ModalityLabelPerturber()(base.clone())
    torch.manual_seed(7)
    second = ModalityLabelPerturber()(base.clone())
    assert torch.equal(first, second)


def test_keeps_shape_and_element_dtype():
    base = torch.tensor([[[[3.0, 9.0], [13.0, 8.0]]]])
    torch.manual_seed(3)
    out = ModalityLabelPerturber()(base.clone())
    assert out.shape == base.shape
    assert out.dtype == base.dtype
