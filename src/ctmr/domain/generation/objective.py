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

"""Domain objective members (ADR-0016).

``ModalityLabelPerturber`` is the unique domain definition of the modality
label augmentation the P1 continuation applies (pinned prob 0.1, CT members →
1, MR members → 8, prob-decided zeroing) (issue #170), with the config-driven
per-token freeze the series-② rectification adds (token 34 P(keep)=1, issue
#250).  ``VaeObjective`` is
the single domain definition of the repo-owned VAE Kullback-Leibler and
generator loss aggregation, consolidating the retired ``ctmr.domain.losses.kl_loss``
and ``ctmr.application.vae_train.loss_weighted_sum`` business free functions
(issue #171).  Tumour-region weighting belongs to the P2-P3 migrations and is
not defined here yet.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


class VaeObjective:
    """The repo-owned VAE objective: KL regulariser plus generator loss aggregation.

    The single domain carrier of the two retired business free functions
    (ADR-0016, issue #171) -- ``kl`` is verbatim ``ctmr.domain.losses.kl_loss``
    (itself extracted from the retired ``utils.KL_loss``), and ``aggregate`` is
    verbatim ``ctmr.application.vae_train.loss_weighted_sum`` (the notebook
    cell-30 generator combination).  No VAE model entity is introduced: MONAI's
    native autoencoder stays the only VAE model object.  Stateless -- the
    weights travel with the application's epoch configuration.
    """

    EPS = 1e-10

    def kl(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """Kullback-Leibler divergence between N(z_mu, z_sigma^2) and N(0, 1),
        summed over all non-batch dims and averaged over the batch.

        Keeps the learned latent distribution close to a standard normal -- the
        regulariser term of the VAE objective.

        Args:
            z_mu: mean of the latent variable distribution, [N,C,H,W,D] or [N,C,H,W].
            z_sigma: standard deviation, same shape as ``z_mu``.

        Returns:
            torch.Tensor: scalar KL divergence averaged over the batch.
        """
        kl = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + self.EPS) - 1,
            dim=list(range(1, len(z_sigma.shape))),
        )
        return torch.sum(kl) / kl.shape[0]

    def aggregate(self, losses: dict[str, torch.Tensor], *, kl_weight: float, perceptual_weight: float) -> torch.Tensor:
        """recon + weighted KL + weighted perceptual -- the generator objective core."""
        return losses["recons_loss"] + kl_weight * losses["kl_loss"] + perceptual_weight * losses["p_loss"]


class ModalityLabelPerturber:
    """The pinned modality-label augmentation, semantically the migrated P1 code-literal.

    Rules (verbatim from the continuation's ``augment_modality_label``):
    members in [2, 8) relabel to 1; members >= 9 relabel to 8 (each with
    probability ``prob``); every element is then zeroed with probability
    ``prob``.  In-place on the label tensor, like the legacy augmentation.

    Freeze (issue #250, series-② T3): ``frozen_tokens`` names the modality
    tokens excluded from every rule -- each keeps its value with probability 1
    (no relabel, no zeroing), fixing the job-D-measured ≈40% bright-core
    dilution of t1c (34→8 share 0.6041 mixed).  Config-driven with an empty
    default: without the key the augmentation is the legacy one op-for-op --
    the three full-tensor ``torch.rand`` draws keep their shapes and order,
    so the frozen run under a given seed differs from the unfrozen one only
    at the frozen tokens' positions.  The augmentation's robustness purpose
    for the other tokens is untouched.
    """

    PINNED_PROB = 0.1

    def __init__(self, prob: float = PINNED_PROB, frozen_tokens: Iterable[int] = ()):
        self._prob = prob
        self._frozen_tokens = frozenset(int(token) for token in frozen_tokens)

    @property
    def prob(self) -> float:
        return self._prob

    @property
    def frozen_tokens(self) -> tuple[int, ...]:
        return tuple(sorted(self._frozen_tokens))

    def _frozen_mask(self, modality_tensor: torch.Tensor) -> torch.Tensor | None:
        """The elementwise freeze flag, or ``None`` when nothing is frozen.

        ``None`` keeps the no-freeze path the legacy augmentation verbatim;
        with a freeze, the masks narrow but the RNG draws do not.
        """
        if not self._frozen_tokens:
            return None
        tokens = torch.tensor(sorted(self._frozen_tokens), device=modality_tensor.device, dtype=modality_tensor.dtype)
        return torch.isin(modality_tensor, tokens)

    def __call__(self, modality_tensor: torch.Tensor) -> torch.Tensor:
        frozen = self._frozen_mask(modality_tensor)
        mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
        if frozen is not None:
            mask_ct = mask_ct & ~frozen
        prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < self._prob
        modality_tensor[mask_ct & prob_ct] = 1

        mask_mri = modality_tensor >= 9
        if frozen is not None:
            mask_mri = mask_mri & ~frozen
        prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < self._prob
        modality_tensor[mask_mri & prob_mri] = 8

        mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > self._prob
        if frozen is not None:
            mask_zero = mask_zero | frozen
        return modality_tensor * mask_zero.long()


class TumourWeightedTarget:
    """The pinned P2 weighted target: weights = 1, = the weight on the tumour ROI.

    The tumour-region weighting the ADR-0016 objective module carries for the
    mask-conditioned families: the combined label mask is nearest-neighbour
    aligned onto the (latent) image grid, the ROI is the union of the given
    label ids ({129, 130, 131} for the P2 recipe), and the weight broadcasts
    across the latent channels.  A weight at or below 1.0 disables the ROI
    entirely (``None`` -- the plain-L1 branch), matching the migrated kernel's
    branch condition verbatim.
    """

    def __init__(self, weight: float, labels: list[int]):
        self._weight = weight
        self._labels = list(labels)

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def weights(self, mask_labels, images) -> torch.Tensor | None:
        """The per-voxel weight tensor for one batch, or ``None`` when weighting is off.

        ``mask_labels`` is the combined label mask ``[B,1,X,Y,Z]`` as the
        loader hands it over; ``images`` fixes the target grid and device.
        """
        if self._weight <= 1.0:
            return None
        weights = torch.ones_like(images)
        roi = torch.zeros([images.shape[0], 1] + list(images.shape[2:]), device=images.device)
        interpolate_label = torch.nn.functional.interpolate(mask_labels.float(), size=images.shape[2:], mode="nearest").long()
        for label in self._labels:
            roi[interpolate_label == label] = 1
        weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = self._weight
        return weights
