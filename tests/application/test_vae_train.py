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

"""Behaviour gates for ctmr.application.vae_train (ADR-0015 §8, issue #142).

The extracted train_vae_tutorial.ipynb GAN loop: builder recipe values are
pinned against the notebook literals (Adam eps split by amp, three-phase
warmup lambda, GradScaler init_scale=2**8 / growth_factor=1.5,
PatchDiscriminator channel/layer layout); the alternating generator /
discriminator update itself is executed for real on a synthetic CPU fixture
(tiny autoencoder stub + real PatchDiscriminator, 32^3 patches). Torch-level:
runs on CPU in the CI torch-stack job.
"""

from pathlib import Path

import pytest
import torch
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.networks.nets import PatchDiscriminator
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam, lr_scheduler

from ctmr.application.vae_train import (
    build_adversarial_loss,
    build_amp_scalers,
    build_discriminator,
    build_intensity_loss,
    build_lr_schedulers,
    build_optimizers,
    load_pretrained_weights,
    loss_weighted_sum,
    train_epoch,
    warmup_rule,
)

# 32^3 is the smallest cube the 3-layer patch discriminator keeps spatially
# alive at train time (16^3 collapses to a single element and trips norm layers).
IMAGE_VOLUME = (32, 32, 32)


class TinyAutoencoder(torch.nn.Module):
    """1-conv stand-in honouring the MAISI autoencoder contract:
    forward(images) -> (reconstruction, z_mu, z_sigma)."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def forward(self, images):
        reconstruction = self.net(images)
        z_mu = reconstruction.mean(dim=(2, 3, 4), keepdim=True)
        z_sigma = torch.ones_like(z_mu)
        return reconstruction, z_mu, z_sigma


class SyntheticLoader:
    """Dict-batch loader standing in for the VAE_Transform dataloader."""

    def __init__(self, n_batches):
        self.batches = [{"image": torch.rand(2, 1, *IMAGE_VOLUME)} for _ in range(n_batches)]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _absolute_parameters(module):
    return [p.detach().abs().sum().item() for p in module.parameters()]


@pytest.fixture()
def tiny_autoencoder():
    torch.manual_seed(0)
    return TinyAutoencoder()


@pytest.mark.torch
def test_warmup_rule_three_phases_pinned():
    assert warmup_rule(0) == 0.01
    assert warmup_rule(9) == 0.01
    assert warmup_rule(10) == 0.1
    assert warmup_rule(19) == 0.1
    assert warmup_rule(20) == 1.0
    assert warmup_rule(279) == 1.0


@pytest.mark.torch
def test_build_discriminator_matches_notebook_layout():
    d = build_discriminator(spatial_dims=3)
    assert isinstance(d, PatchDiscriminator)
    assert d.num_channels == 32 and d.num_layers_d == 3


@pytest.mark.torch
def test_build_intensity_loss_dispatches_on_recon_loss():
    assert isinstance(build_intensity_loss("l2"), MSELoss)
    assert isinstance(build_intensity_loss("l1"), L1Loss)


@pytest.mark.torch
def test_build_adversarial_loss_is_least_squares_patch_loss():
    adv = build_adversarial_loss()
    assert isinstance(adv, PatchAdversarialLoss)


@pytest.mark.torch
@pytest.mark.parametrize("amp,eps", [(True, 1e-6), (False, 1e-8)])
def test_build_optimizers_pin_eps_split_by_amp(tiny_autoencoder, amp, eps):
    discriminator = build_discriminator(spatial_dims=3)
    optimizer_g, optimizer_d = build_optimizers(tiny_autoencoder, discriminator, lr=2e-4, amp=amp)
    assert isinstance(optimizer_g, Adam) and isinstance(optimizer_d, Adam)
    for opt in (optimizer_g, optimizer_d):
        assert opt.param_groups[0]["lr"] == 2e-4
        assert opt.param_groups[0]["eps"] == eps
    g_params = {id(p) for group in optimizer_g.param_groups for p in group["params"]}
    d_params = {id(p) for group in optimizer_d.param_groups for p in group["params"]}
    assert g_params and d_params and not g_params & d_params


@pytest.mark.torch
@pytest.mark.filterwarnings("ignore::UserWarning")  # bare scheduler.step() without a prior optimizer.step()
def test_build_lr_schedulers_follow_warmup_rule(tiny_autoencoder):
    lr = 2e-4
    optimizer_g, optimizer_d = build_optimizers(tiny_autoencoder, build_discriminator(spatial_dims=3), lr=lr, amp=False)
    scheduler_g, scheduler_d = build_lr_schedulers(optimizer_g, optimizer_d)
    assert isinstance(scheduler_g, lr_scheduler.LambdaLR)
    assert scheduler_g.get_last_lr() == [pytest.approx(lr * warmup_rule(0))]
    # the pairing itself is pinned: each scheduler drives its own optimizer
    assert scheduler_g.optimizer is optimizer_g
    assert scheduler_d.optimizer is optimizer_d
    for _ in range(10):  # epochs 1..9 stay in phase one; stepping into epoch 10 jumps to 0.1
        scheduler_g.step()
        scheduler_d.step()
    assert scheduler_g.get_last_lr() == [pytest.approx(lr * warmup_rule(10))]
    assert scheduler_d.get_last_lr() == [pytest.approx(lr * warmup_rule(10))]
    scheduler_g.step()
    assert scheduler_g.get_last_lr() == [pytest.approx(lr * warmup_rule(11))]


@pytest.mark.torch
def test_build_amp_scalers_pinned_against_notebook_values(monkeypatch):
    """The GradScaler pairs are pinned by captured constructor arguments: a CPU-only
    environment self-disables a cuda scaler, so instance readbacks cannot carry the pin."""
    captured = {}

    class CapturingScaler:
        def __init__(self, device_type, **kwargs):
            captured["device_type"] = device_type
            captured.update(kwargs)

    monkeypatch.setattr(torch.amp, "GradScaler", CapturingScaler)
    scaler_g, scaler_d = build_amp_scalers(amp=True)
    assert isinstance(scaler_g, CapturingScaler) and isinstance(scaler_d, CapturingScaler)
    assert captured == {"device_type": "cuda", "init_scale": 2.0**8, "growth_factor": 1.5}
    assert build_amp_scalers(amp=False) is None


@pytest.mark.torch
def test_loss_weighted_sum_pins_the_notebook_combination():
    losses = {
        "recons_loss": torch.tensor(4.0),
        "kl_loss": torch.tensor(3.0),
        "p_loss": torch.tensor(2.0),
    }
    total = loss_weighted_sum(losses, kl_weight=0.001, perceptual_weight=0.01)
    assert total.item() == pytest.approx(4.0 + 3.0e-3 + 2.0e-2)


@pytest.mark.torch
def test_load_pretrained_weights_plain_state_dict(tmp_path: Path, tiny_autoencoder):
    drifted = TinyAutoencoder()
    with torch.no_grad():
        drifted.net.weight += 0.5  # distinct from any seeded init
    checkpoint_path = tmp_path / "autoencoder.pt"
    torch.save(drifted.state_dict(), checkpoint_path)
    load_pretrained_weights(tiny_autoencoder, checkpoint_path)
    assert torch.equal(tiny_autoencoder.net.weight, drifted.net.weight)


@pytest.mark.torch
def test_load_pretrained_weights_unwraps_unet_state_dict_key(tmp_path: Path, tiny_autoencoder):
    """cell-28 compat: checkpoints stored as {"unet_state_dict": ...} load transparently."""
    payload = {"unet_state_dict": tiny_autoencoder.state_dict(), "scale_factor": 0.9}
    checkpoint_path = tmp_path / "epoch_10.pt"
    torch.save(payload, checkpoint_path)
    drifted = TinyAutoencoder()
    with torch.no_grad():
        drifted.net.weight += 1.0
    load_pretrained_weights(drifted, checkpoint_path)
    assert torch.equal(drifted.net.weight, tiny_autoencoder.net.weight)


@pytest.mark.torch
def test_train_epoch_alternating_updates_execute_and_move_both_networks(tiny_autoencoder):
    """The loop body executes a real G-step then D-step per batch and moves both networks."""
    loader = SyntheticLoader(n_batches=2)
    discriminator = build_discriminator(spatial_dims=3)
    optimizer_g, optimizer_d = build_optimizers(tiny_autoencoder, discriminator, lr=2e-4, amp=False)
    perceptual_stub = lambda recon, target: (recon - target).abs().mean()  # noqa: E731 (synthetic stand-in for PerceptualLoss)
    before_g = _absolute_parameters(tiny_autoencoder)
    before_d = _absolute_parameters(discriminator)

    avg_losses = train_epoch(
        loader,
        autoencoder=tiny_autoencoder,
        discriminator=discriminator,
        intensity_loss=build_intensity_loss("l1"),
        adversarial_loss=build_adversarial_loss(),
        perceptual_loss=perceptual_stub,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        adv_weight=0.01,
        kl_weight=1e-6,
        perceptual_weight=1e-6,
        device=torch.device("cpu"),
        autocast_device_type="cpu",
        amp=False,
    )

    after_g = _absolute_parameters(tiny_autoencoder)
    after_d = _absolute_parameters(discriminator)
    assert any(a != b for a, b in zip(before_g, after_g)), "generator weights must move"
    assert any(a != b for a, b in zip(before_d, after_d)), "discriminator weights must move"
    assert set(avg_losses) == {"recons_loss", "kl_loss", "p_loss"}
    for value in avg_losses.values():
        assert isinstance(value, float) and value == value and abs(value) < float("inf")  # finite (not NaN/inf)
