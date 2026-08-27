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


"""Convergence gates for ctmr.domain.losses.KL_loss (ADR-0015 §2, #132).

Pins the VAE KL-divergence semantics lifted verbatim out of scripts/utils.py:
closed-form value against N(0,1), full non-batch-dim summation followed by a
batch mean, 4D/5D shape acceptance, scalar output, differentiability.
Torch-level: skips itself on light stacks via pytest.importorskip; runs on CPU.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip must precede the torch-dependent import)

from ctmr.domain.losses import KL_loss  # noqa: E402


def _expected_kl(mu, sigma):
    """Closed form of the summed expression in float64, batch-mean last."""
    kl = 0.5 * (mu**2 + sigma**2 - np.log(sigma**2) - 1)
    return float(kl.sum()) / mu.shape[0]


def test_standard_normal_inputs_give_zero():
    z_mu = torch.zeros(2, 1, 4, 4, 4)
    z_sigma = torch.ones(2, 1, 4, 4, 4)
    loss = KL_loss(z_mu, z_sigma)
    assert loss.dim() == 0
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_matches_closed_form_on_5d_batch():
    gen = torch.Generator().manual_seed(7)
    z_mu = torch.randn(3, 2, 4, 4, 4, generator=gen).double()
    z_sigma = (0.5 + torch.rand(3, 2, 4, 4, 4, generator=gen)).double()  # keep clear of eps
    loss = KL_loss(z_mu, z_sigma)
    eps_shift = abs(float(loss) - _expected_kl(z_mu.numpy(), z_sigma.numpy()))
    assert eps_shift < 1e-6  # residual is exactly the eps=1e-10 log guard, aggregated
    assert float(loss) == pytest.approx(_expected_kl(z_mu.numpy(), z_sigma.numpy()), rel=1e-7)


def test_reduction_sums_every_non_batch_dim_then_means_over_batch():
    # asymmetric batch elements so any wrong axis choice (e.g. channel mean,
    # spatial mean) lands away from the true full-sum / batch-mean answer
    mu = torch.zeros(2, 1, 4, 4)
    sigma = torch.ones(2, 1, 4, 4)
    mu[0] += 0.25
    sigma[1] = 2.0
    expected = (_expected_kl(mu[0].numpy(), sigma[0].numpy()) + _expected_kl(mu[1].numpy(), sigma[1].numpy())) / 2
    assert float(KL_loss(mu, sigma)) == pytest.approx(expected, rel=1e-10)


def test_accepts_4d_and_5d_and_returns_scalar():
    for shape in [(4, 3, 8, 8), (4, 2, 3, 3, 3)]:
        loss = KL_loss(torch.zeros(shape), torch.ones(shape))
        assert loss.dim() == 0


def test_backpropagates_to_both_inputs():
    mu = torch.randn(2, 1, 4, 4, requires_grad=True)
    sigma = torch.rand(2, 1, 4, 4) + 0.5
    sigma.requires_grad_(True)
    KL_loss(mu, sigma).backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()
