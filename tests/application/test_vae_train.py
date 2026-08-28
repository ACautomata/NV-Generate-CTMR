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
    train_epoch,
    validate_epoch,
    warmup_rule,
)
from ctmr.domain.generation.objective import VaeObjective

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


def _legacy_kl(z_mu, z_sigma):
    """Verbatim the retired ``ctmr.domain.losses.kl_loss`` (from ``utils.KL_loss``)."""
    eps = 1e-10
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl) / kl.shape[0]


@pytest.mark.torch
def test_train_epoch_single_step_losses_match_the_legacy_free_functions(tiny_autoencoder):
    """GAN single-step numerical parity (issue #171): the real alternating
    update on a one-batch CPU fixture must return the exact loss triplet the
    pre-migration free-function math computes on the same pre-update weights.

    The loop is RNG-free on CPU fp32 (deterministic stubs and least-squares
    adversarial), so the returned per-batch values equal a plain forward
    replayed with the retired ``kl_loss`` on the same seeded fixture.
    """
    torch.manual_seed(20260828)
    batch = {"image": torch.rand(2, 1, *IMAGE_VOLUME)}
    loader = SyntheticLoader(n_batches=1)
    loader.batches = [batch]
    discriminator = build_discriminator(spatial_dims=3)
    optimizer_g, optimizer_d = build_optimizers(tiny_autoencoder, discriminator, lr=2e-4, amp=False)
    intensity_loss = build_intensity_loss("l1")
    perceptual_stub = lambda recon, target: (recon - target).abs().mean()  # noqa: E731

    # legacy replay: the same pre-update weights, retired free-function math
    with torch.no_grad():
        reconstruction, z_mu, z_sigma = tiny_autoencoder(batch["image"])
    expected = {
        "recons_loss": intensity_loss(reconstruction, batch["image"]).item(),
        "kl_loss": _legacy_kl(z_mu, z_sigma).item(),
        "p_loss": perceptual_stub(reconstruction.float(), batch["image"].float()).item(),
    }

    got = train_epoch(
        loader,
        autoencoder=tiny_autoencoder,
        discriminator=discriminator,
        intensity_loss=intensity_loss,
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

    for key in expected:
        assert got[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-12)


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


@pytest.mark.torch
def test_train_epoch_rejects_amp_without_scalers(tiny_autoencoder, tmp_path: Path):
    """amp=True autocasts but never scales: the scaler pair is mandatory, not optional.

    Codex review P2 (line 154): a caller setting amp=True while omitting both scalers
    silently fell back to plain backward under enabled autocast. The combination is
    now rejected up front instead of silently destablilising fp16 training.
    """
    loader = SyntheticLoader(n_batches=1)
    with pytest.raises(ValueError, match="amp"):
        train_epoch(
            loader,
            autoencoder=tiny_autoencoder,
            discriminator=build_discriminator(spatial_dims=3),
            intensity_loss=build_intensity_loss("l1"),
            adversarial_loss=build_adversarial_loss(),
            perceptual_loss=lambda recon, target: (recon - target).abs().mean(),
            optimizer_g=build_optimizers(tiny_autoencoder, build_discriminator(spatial_dims=3), lr=2e-4, amp=True)[0],
            optimizer_d=build_optimizers(tiny_autoencoder, build_discriminator(spatial_dims=3), lr=2e-4, amp=True)[1],
            adv_weight=0.01,
            kl_weight=1e-6,
            perceptual_weight=1e-6,
            device=torch.device("cpu"),
            autocast_device_type="cpu",
            amp=True,
        )


@pytest.mark.torch
def test_train_epoch_discriminator_update_excludes_generator_objective(tiny_autoencoder, monkeypatch):
    """The D-step gradients must come from the discriminator objective only.

    Codex review P1 (line 199): with a nonzero ``adv_weight`` the generator backward
    traverses the discriminator and accumulates gradients on its parameters, and the
    subsequent discriminator backward stacked on top of them -- so optimizer_d.step
    updated the discriminator by a mixture of opposing objectives. The loop must clear
    the generator-path gradients before the discriminator update.
    """
    batch = {"image": torch.rand(2, 1, *IMAGE_VOLUME)}
    loader = SyntheticLoader(n_batches=1)
    loader.batches = [batch]
    discriminator = build_discriminator(spatial_dims=3)
    d_params = list(discriminator.parameters())

    adv_loss = build_adversarial_loss()
    perceptual_stub = lambda recon, target: (recon - target).abs().mean()  # noqa: E731
    intensity_loss = build_intensity_loss("l1")
    adv_weight, kl_weight, perceptual_weight = 0.01, 1e-6, 1e-6

    def generator_step_graph():
        objective = VaeObjective()
        reconstruction, z_mu, z_sigma = tiny_autoencoder(batch["image"])
        losses = {
            "recons_loss": intensity_loss(reconstruction, batch["image"]),
            "kl_loss": objective.kl(z_mu, z_sigma),
            "p_loss": perceptual_stub(reconstruction.float(), batch["image"].float()),
        }
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = objective.aggregate(losses, kl_weight=kl_weight, perceptual_weight=perceptual_weight)
        return loss_g + adv_weight * generator_loss, reconstruction

    # Reference: the generator-objective and discriminator-objective gradient
    # contributions on the discriminator parameters, computed at initial weights.
    loss_g, reconstruction = generator_step_graph()
    grad_g = torch.autograd.grad(loss_g, d_params, retain_graph=True)
    with torch.no_grad():
        reconstruction = reconstruction.contiguous().detach()
    logits_fake = discriminator(reconstruction)[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = discriminator(batch["image"].contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    grad_d = torch.autograd.grad((loss_d_fake + loss_d_real) * 0.5, d_params)
    assert any(g is not None and g.abs().sum() > 0 for g in grad_g), "precondition: the generator path must reach the discriminator"

    # What the loop actually hands to optimizer_d.step (the D-step is the only one).
    captured = {}

    class SteppingOptimizer(torch.optim.Adam):
        def step(self, closure=None):
            captured["grads"] = [p.grad.detach().clone() for p in d_params]
            return super().step(closure)

    optimizer_g, _ = build_optimizers(tiny_autoencoder, discriminator, lr=2e-4, amp=False)
    optimizer_d = SteppingOptimizer(params=d_params, lr=2e-4, eps=1e-8)

    train_epoch(
        loader,
        autoencoder=tiny_autoencoder,
        discriminator=discriminator,
        intensity_loss=intensity_loss,
        adversarial_loss=adv_loss,
        perceptual_loss=perceptual_stub,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        adv_weight=adv_weight,
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        device=torch.device("cpu"),
        autocast_device_type="cpu",
        amp=False,
    )

    for got, d_only, g_plus_d in zip(captured["grads"], grad_d, (g + d for g, d in zip(grad_g, grad_d))):
        assert torch.allclose(got, d_only, atol=1e-6), (
            "discriminator update must be driven by the discriminator objective only "
            f"(max deviation from d-only {((got - d_only).abs().max().item()):.2e}, "
            f"from g+d {((got - g_plus_d).abs().max().item()):.2e})"
        )


@pytest.mark.torch
def test_validate_epoch_scores_the_loss_family_without_moving_weights(tiny_autoencoder):
    """The validation pass runs real tensor math and leaves weights untouched (cell 30 tail)."""
    loader = SyntheticLoader(n_batches=2)
    before = _absolute_parameters(tiny_autoencoder)
    perceptual_stub = lambda recon, target: (recon - target).abs().mean()  # noqa: E731
    intensity_loss = build_intensity_loss("l1")

    scores = validate_epoch(
        loader,
        autoencoder=tiny_autoencoder,
        intensity_loss=intensity_loss,
        perceptual_loss=perceptual_stub,
        infer=lambda images: tiny_autoencoder(images),
        device=torch.device("cpu"),
        autocast_device_type="cpu",
        amp=False,
    )

    assert _absolute_parameters(tiny_autoencoder) == before, "validation must not move weights"
    assert set(scores) == {"recons_loss", "kl_loss", "p_loss"}
    # the epoch averages are exactly the per-batch loss means, computed by hand
    expected = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0}
    for batch in loader.batches:
        with torch.no_grad():
            reconstruction, z_mu, z_sigma = tiny_autoencoder(batch["image"])
        expected["recons_loss"] += intensity_loss(reconstruction, batch["image"]).item()
        expected["kl_loss"] += VaeObjective().kl(z_mu, z_sigma).item()
        expected["p_loss"] += perceptual_stub(reconstruction.float(), batch["image"].float()).item()
    for key in expected:
        expected[key] /= len(loader.batches)
        assert scores[key] == pytest.approx(expected[key], rel=1e-6)
