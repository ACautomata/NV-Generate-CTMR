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

"""Domain objective members (ADR-0016, issue #170).

``ModalityLabelPerturber`` is the unique domain definition of the modality
label augmentation the P1 continuation applies (pinned prob 0.1, CT members →
1, MR members → 8, prob-decided zeroing).  VAE Kullback-Leibler / loss
aggregation (``VaeObjective``) and tumour-region weighting belong to the VAE /
P2-P3 migrations and are not defined here yet.
"""

from __future__ import annotations

import torch


class ModalityLabelPerturber:
    """The pinned modality-label augmentation, semantically the migrated P1 code-literal.

    Rules (verbatim from the continuation's ``augment_modality_label``):
    members in [2, 8) relabel to 1; members >= 9 relabel to 8 (each with
    probability ``prob``); every element is then zeroed with probability
    ``prob``.  In-place on the label tensor, like the legacy augmentation.
    """

    PINNED_PROB = 0.1

    def __init__(self, prob: float = PINNED_PROB):
        self._prob = prob

    @property
    def prob(self) -> float:
        return self._prob

    def __call__(self, modality_tensor: torch.Tensor) -> torch.Tensor:
        mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
        prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < self._prob
        modality_tensor[mask_ct & prob_ct] = 1

        mask_mri = modality_tensor >= 9
        prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < self._prob
        modality_tensor[mask_mri & prob_mri] = 8

        mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > self._prob
        return modality_tensor * mask_zero.long()
