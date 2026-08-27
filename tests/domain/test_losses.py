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

"""Numerical pinning of the pure VAE loss math (ctmr.domain.losses).

Torch-level: runs real tensor math on CPU.
"""

import math

import pytest
import torch

from ctmr.domain.losses import kl_loss


@pytest.mark.torch
def test_kl_loss_standard_normal_is_zero():
    mu = torch.zeros(2, 1, 4, 4, 4)
    sigma = torch.ones(2, 1, 4, 4, 4)
    assert kl_loss(mu, sigma).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.torch
def test_kl_loss_closed_form_pinned():
    """Single-element latents: KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2 + eps) - 1),
    averaged over the batch dimension only."""
    mu = torch.tensor([[[[1.0]]], [[[-1.0]]]])  # shape [2,1,1,1]
    sigma = torch.full((2, 1, 1, 1), 2.0)
    expected = 0.5 * (1.0**2 + 2.0**2 - math.log(2.0**2 + 1e-10) - 1)
    got = kl_loss(mu, sigma).item()
    assert got == pytest.approx(expected, rel=1e-6)


@pytest.mark.torch
def test_kl_loss_sums_all_non_batch_dims():
    """Two-spatial-element latents sum over every non-batch dim (scripts/utils.KL_loss contract)."""
    mu = torch.zeros(1, 1, 1, 2)
    sigma = torch.full((1, 1, 1, 2), 2.0)
    per_element = 0.5 * (0.0 + 4.0 - math.log(4.0 + 1e-10) - 1.0)
    assert kl_loss(mu, sigma).item() == pytest.approx(2 * per_element, rel=1e-6)
