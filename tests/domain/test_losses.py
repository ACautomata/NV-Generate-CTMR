"""Numerical anchors for ctmr.domain.losses.KL_loss on synthetic tensors (CPU, no fixtures needed)."""

import inspect
import math

import pytest
import torch

from ctmr.domain import losses
from ctmr.domain.losses import KL_loss  # noqa: N802

pytestmark = pytest.mark.torch


def test_standard_normal_gives_zero():
    """mu=0, sigma=1 is the prior itself, so every term vanishes exactly."""
    z_mu = torch.zeros(2, 4, 8, 8, 8)
    z_sigma = torch.ones(2, 4, 8, 8, 8)
    assert KL_loss(z_mu, z_sigma).item() == pytest.approx(0.0, abs=1e-7)


def test_single_element_analytic_value():
    """0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1) summed over non-batch dims."""
    # mu=2, sigma=1 -> 0.5 * (4 + 1 - 0 - 1) = 2.0
    assert KL_loss(torch.tensor([[[[2.0]]]]), torch.tensor([[[[1.0]]]])).item() == pytest.approx(2.0)


def test_batch_mean_not_sum():
    """The loss averages over the batch dimension."""
    z_mu = torch.tensor([[[[1.0]]], [[[3.0]]]])  # shape [2,1,1,1]
    z_sigma = torch.ones_like(z_mu)
    # per-sample: 0.5*1 and 0.5*9 -> mean 2.5
    assert KL_loss(z_mu, z_sigma).item() == pytest.approx(2.5)


def test_sigma_deviation_contributes_log_term():
    """mu=0, sigma=2 -> 0.5 * (0 + 4 - log(4 + eps) - 1)."""
    expected = 0.5 * (4.0 - math.log(4.0) - 1.0)
    got = KL_loss(torch.zeros(1, 1), 2.0 * torch.ones(1, 1)).item()
    assert got == pytest.approx(expected, rel=1e-6)


def test_3d_latent_shape_agrees_with_2d_per_element():
    """Same per-element anchor values under [N,C,H,W,D] and [N,C,H,W] summation semantics."""
    mu_2d = torch.tensor([[[[1.0, 2.0]]]])  # [1,1,1,2]
    sigma_2d = torch.tensor([[[[1.0, 1.0]]]])
    mu_3d = mu_2d.unsqueeze(-1)  # [1,1,1,2,1]
    sigma_3d = sigma_2d.unsqueeze(-1)
    # 0.5 * (1 + 4) = 2.5 for both shapes
    assert KL_loss(mu_2d, sigma_2d).item() == pytest.approx(2.5)
    assert KL_loss(mu_3d, sigma_3d).item() == pytest.approx(2.5)


def test_loss_is_nonnegative():
    generator = torch.Generator().manual_seed(0)
    for _ in range(5):
        z_mu = torch.randn(3, 2, 4, 4, 4, generator=generator)
        z_sigma = torch.rand(3, 2, 4, 4, 4, generator=generator) + 0.5
        assert KL_loss(z_mu, z_sigma).item() >= 0.0


def test_losses_module_has_no_io():
    """Domain purity guard (terminal-state style): the losses module must not import or open anything."""
    source = inspect.getsource(losses)
    banned = ["open(", "os.", "pathlib", "Path(", "requests", "urllib"]
    hits = [token for token in banned if token in source]
    assert hits == []
