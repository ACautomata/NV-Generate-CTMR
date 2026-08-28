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

"""Generation objective gates (ADR-0016): ModalityLabelPerturber (#170) and VaeObjective (#171).

The perturbation semantics must stay the P1-pinned recipe (augment prob 0.1,
CT members → 1, MR members → 8, prob-decided zeroing) and numerically match
the vendored upstream ``augment_modality_label`` the migrated P1 entry used
(seed-replayed).  ``VaeObjective`` is the single domain definition of the
repo-owned VAE KL and generator loss aggregation: its math must reproduce,
bit for bit, the retired ``ctmr.domain.losses.kl_loss`` (itself extracted from
the retired ``utils.KL_loss``) and ``ctmr.application.vae_train.loss_weighted_sum``
(notebook cell-30 combination), embedded verbatim below as the parity
reference.  Torch-level: real execution on CPU.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from ctmr.domain.generation import objective as objective_module
from ctmr.domain.generation.objective import ModalityLabelPerturber, VaeObjective
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


# ---------------------------------------------------------------------------
# VaeObjective (ADR-0016, issue #171): the single domain definition of the
# repo-owned VAE KL and generator loss aggregation.
# ---------------------------------------------------------------------------


def _legacy_kl(z_mu, z_sigma):
    """Verbatim the retired ``ctmr.domain.losses.kl_loss`` (from ``utils.KL_loss``)."""
    eps = 1e-10
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl) / kl.shape[0]


def _legacy_weighted_sum(losses, *, kl_weight, perceptual_weight):
    """Verbatim the retired ``ctmr.application.vae_train.loss_weighted_sum`` (cell 30)."""
    return losses["recons_loss"] + kl_weight * losses["kl_loss"] + perceptual_weight * losses["p_loss"]


def test_vae_kl_standard_normal_is_zero():
    mu = torch.zeros(2, 1, 4, 4, 4)
    sigma = torch.ones(2, 1, 4, 4, 4)
    assert VaeObjective().kl(mu, sigma).item() == pytest.approx(0.0, abs=1e-6)


def test_vae_kl_closed_form_pinned():
    """Single-element latents: KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2 + eps) - 1),
    averaged over the batch dimension only."""
    mu = torch.tensor([[[[1.0]]], [[[-1.0]]]])  # shape [2,1,1,1]
    sigma = torch.full((2, 1, 1, 1), 2.0)
    expected = 0.5 * (1.0**2 + 2.0**2 - math.log(2.0**2 + 1e-10) - 1)
    got = VaeObjective().kl(mu, sigma).item()
    assert got == pytest.approx(expected, rel=1e-6)


def test_vae_kl_sums_all_non_batch_dims():
    """Two-spatial-element latents sum over every non-batch dim (the retired ``utils.KL_loss`` contract)."""
    mu = torch.zeros(1, 1, 1, 2)
    sigma = torch.full((1, 1, 1, 2), 2.0)
    per_element = 0.5 * (0.0 + 4.0 - math.log(4.0 + 1e-10) - 1.0)
    assert VaeObjective().kl(mu, sigma).item() == pytest.approx(2 * per_element, rel=1e-6)


def test_vae_kl_averages_over_batch_not_sums():
    """Batch mean semantics, not batch sum: samples 0.5*1 and 0.5*9 average to 2.5."""
    mu = torch.tensor([[[[1.0]]], [[[3.0]]]])  # shape [2,1,1,1]
    sigma = torch.ones_like(mu)
    assert VaeObjective().kl(mu, sigma).item() == pytest.approx(2.5)


@pytest.mark.parametrize("shape", [(1, 1, 4, 4, 4), (3, 2, 8, 8, 8), (2, 1, 1, 1), (4, 2, 3, 5)])
def test_vae_kl_matches_the_retired_free_function(shape):
    """The domain KL must reproduce the retired free function bit for bit."""
    torch.manual_seed(31)
    z_mu = torch.randn(shape)
    z_sigma = torch.rand(shape) + 0.5  # strictly positive standard deviations
    assert torch.equal(VaeObjective().kl(z_mu, z_sigma), _legacy_kl(z_mu, z_sigma))


def test_vae_aggregate_pins_the_notebook_combination():
    """recon + kl_weight * KL + perceptual_weight * perceptual -- the cell-30 generator core."""
    losses = {
        "recons_loss": torch.tensor(4.0),
        "kl_loss": torch.tensor(3.0),
        "p_loss": torch.tensor(2.0),
    }
    total = VaeObjective().aggregate(losses, kl_weight=0.001, perceptual_weight=0.01)
    assert total.item() == pytest.approx(4.0 + 3.0e-3 + 2.0e-2)


def test_vae_aggregate_matches_the_retired_free_function():
    """The domain aggregation must reproduce the retired free function bit for bit."""
    torch.manual_seed(41)
    losses = {
        "recons_loss": torch.randn(()),
        "kl_loss": torch.randn(()),
        "p_loss": torch.randn(()),
    }
    kl_weight, perceptual_weight = 1e-6, 1e-5
    got = VaeObjective().aggregate(losses, kl_weight=kl_weight, perceptual_weight=perceptual_weight)
    expected = _legacy_weighted_sum(losses, kl_weight=kl_weight, perceptual_weight=perceptual_weight)
    assert torch.equal(got, expected)


def test_objective_module_has_no_io():
    """Domain purity guard: the objective module must not reference any I/O surface."""
    source = inspect.getsource(objective_module)
    banned = ["open(", "os.", "pathlib", "Path(", "requests", "urllib"]
    hits = [token for token in banned if token in source]
    assert hits == []
