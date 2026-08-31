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

"""Behavior tests for the domain tumor-removal chain (ADR-0019 §3, #269).

The mask-to-tumor-free label surgery the P2 sampling path runs on generated
combined labels, floated up pure from the retired augmentation scripts: organ
tumors remap to their organs, lesion regions take pseudo labels (or the
majority organ of their neighborhood), all pure tensor logic. The
infrastructure re-export face stays covered by
tests/infrastructure/dataio/test_augmentation.py.
"""

import pytest
import torch

from ctmr.domain.generation.tumor_removal import remap_labels, remove_tumors, remove_tumors_majority_vote

pytestmark = pytest.mark.torch


def test_organ_tumors_remap_to_their_organs():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 1, 1, 1] = 26  # hepatic tumor -> liver
    labels[0, 2, 2, 2] = 24  # pancreatic tumor -> pancreas
    out = remove_tumors(labels)
    assert out[0, 1, 1, 1] == 1
    assert out[0, 2, 2, 2] == 4


def test_lesions_take_pseudo_labels_when_offered():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 1, 1, 1] = 23  # lung tumor
    pseudo = torch.ones_like(labels) * 29  # every voxel offers a plausible lung label
    out = remove_tumors(labels, pseudo_labels=pseudo)
    assert out[0, 1, 1, 1] == 29


def test_lung_tumor_without_pseudo_labels_takes_the_organ_ring_majority():
    volume = torch.zeros(1, 6, 6, 6, dtype=torch.long)
    volume[0, 1:3, 1:3, 1:3] = 23  # lung tumor core
    volume[0, 3:5, 1:3, 1:3] = 30  # neighboring organ ring
    out = remove_tumors(volume)
    assert out[0, 1, 1, 1] == 30  # the tumor region was filled from its neighborhood


def test_brain_tumors_remap_to_brain_without_pseudo_labels():
    labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    labels[0, 1, 1, 1] = 401  # BraTS channel 1
    labels[0, 2, 2, 2] = 403  # BraTS channel 3
    out = remove_tumors(labels)
    assert out[0, 1, 1, 1] == 22
    assert out[0, 2, 2, 2] == 22


def test_majority_vote_falls_back_to_the_most_common_organ_when_the_ring_is_empty():
    volume = torch.zeros(1, 5, 5, 5)
    volume[0, 1:4, 1:4, 1:4] = 32  # only a distant organ, no ring around the tumor
    tumor_mask = torch.zeros(1, 5, 5, 5)
    tumor_mask[0, 0, 0, 0] = 1
    out = remove_tumors_majority_vote(tumor_mask, volume, organ_label_lists=(28, 29, 30, 31, 32))
    assert out[0, 0, 0, 0] == 32


def test_rejects_a_2d_input():
    with pytest.raises(ValueError, match="3D/4D"):
        remove_tumors(torch.zeros(2, 2))


def test_remap_labels_applies_mapping():
    x = torch.tensor([[[[3, 1]]]], dtype=torch.long)
    out = remap_labels(x, {3: 200})
    assert out.tolist() == [[[[200, 1]]]]
