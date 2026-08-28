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

"""VAE GAN training orchestration -- the repo's only implementation of the
MAISI autoencoder adversarial loop (ADR-0015 §8, issue #142).

Extracted verbatim from ``train_vae_tutorial.ipynb`` cells 24/26/28/30, which
were deleted with the rest of the tutorial notebooks; git history is the
tutorial's reproduction anchor. What lives here:

- the ``PatchDiscriminator`` layout and the loss family construction
  (intensity L1/L2 dispatch, least-squares patch adversarial, squeeze
  perceptual);
- the optimizer/warmup/AMP recipe facts: paired Adam optimizers whose ``eps``
  loosens under autocast, the three-phase warmup lambda (0.01 / 0.1 / 1.0 at
  epochs 10 / 20), paired ``GradScaler`` with init_scale=2**8,
  growth_factor=1.5;
- the alternating update itself: per batch a generator step (recon + KL +
  perceptual + adversarial) then a discriminator step on detached
  reconstructions, plus the finetune loading of pretrained weights;
- the validation pass: epoch-averaged recon/KL/perceptual scores through a
  caller-injected inferer (whole-volume / sliding-window lives on the caller).

The repo-owned KL and generator loss aggregation ride the domain
``VaeObjective`` (ADR-0016, issue #171): the retired ``kl_loss`` /
``loss_weighted_sum`` business free functions were consolidated there.

The IO ring stays out: TensorBoard logging, validation, checkpoint
publication and data transforms belong to the caller / future harness shells;
an epoch returns its averaged losses instead.
"""

import torch
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler

from ctmr.domain.generation.objective import VaeObjective

__all__ = [
    "build_adversarial_loss",
    "build_amp_scalers",
    "build_discriminator",
    "build_intensity_loss",
    "build_lr_schedulers",
    "build_optimizers",
    "build_perceptual_loss",
    "load_pretrained_weights",
    "train_epoch",
    "validate_epoch",
    "warmup_rule",
]


def warmup_rule(epoch: int) -> float:
    """Three-phase learning-rate multiplier applied per epoch (cell 26):
    warm up slowly for ten epochs, transition for another ten, full rate after."""
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    return 1.0


def build_discriminator(spatial_dims: int = 3) -> PatchDiscriminator:
    """The patch discriminator paired with the autoencoder (cell 24)."""
    return PatchDiscriminator(
        spatial_dims=spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    )


def build_intensity_loss(recon_loss: str):
    """MSE for ``"l2"``, mean-reduction L1 otherwise (cell 26)."""
    if recon_loss == "l2":
        return MSELoss()
    return L1Loss(reduction="mean")


def build_adversarial_loss() -> PatchAdversarialLoss:
    return PatchAdversarialLoss(criterion="least_squares")


def build_perceptual_loss(device) -> PerceptualLoss:
    """Squeeze-based fake-3D perceptual loss (downloads its network weights --
    a GPU-side construction path, exercised only outside the CPU test suite)."""
    return PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)


def build_optimizers(autoencoder, discriminator, *, lr: float, amp: bool):
    """Paired Adam optimizers; autocast tolerates the looser eps=1e-06 (cell 26)."""
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=lr, eps=1e-06 if amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=lr, eps=1e-06 if amp else 1e-08)
    return optimizer_g, optimizer_d


def build_lr_schedulers(optimizer_g, optimizer_d):
    """Per-epoch LambdaLR schedulers driven by :func:`warmup_rule` (cell 26)."""
    return (
        lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule),
        lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule),
    )


def build_amp_scalers(*, amp: bool, device_type: str = "cuda"):
    """Paired GradScalers pinned to the notebook values, or ``None`` when AMP is off."""
    if not amp:
        return None
    return (
        torch.amp.GradScaler(device_type, init_scale=2.0**8, growth_factor=1.5),
        torch.amp.GradScaler(device_type, init_scale=2.0**8, growth_factor=1.5),
    )


def load_pretrained_weights(module: torch.nn.Module, checkpoint_path) -> None:
    """Finetune loading (cell 28): accepts both bare state_dicts and the
    upstream ``{"unet_state_dict": ...}`` checkpoint wrapping."""
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and "unet_state_dict" in checkpoint.keys():
        checkpoint = checkpoint["unet_state_dict"]
    module.load_state_dict(checkpoint)


def train_epoch(
    dataloader,
    *,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module,
    intensity_loss,
    adversarial_loss,
    perceptual_loss,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    adv_weight: float,
    kl_weight: float,
    perceptual_weight: float,
    device,
    autocast_device_type: str = "cuda",
    amp: bool = True,
    scaler_g=None,
    scaler_d=None,
) -> dict[str, float]:
    """One epoch of alternating generator/discriminator updates (cell 30).

    Per batch: zero both optimizers, run the generator step inside autocast
    (recon + KL + perceptual + weighted adversarial), then train the
    discriminator on detached reconstructions against real images. When
    ``amp=True`` the paired scalers from :func:`build_amp_scalers` are
    mandatory -- enabled autocast without gradient scaling silently underflows
    fp16 grads -- and are unused when ``amp=False``. Callers step the paired LR
    schedulers once per epoch and own validation/checkpoint publication.

    Returns:
        dict: epoch-averaged ``{"recons_loss", "kl_loss", "p_loss"}`` floats.
    """
    if amp and (scaler_g is None or scaler_d is None):
        raise ValueError(
            "amp=True requires both scalers (build_amp_scalers) -- enabled autocast without gradient scaling silently underflows fp16 gradients."
        )
    autoencoder.train()
    discriminator.train()
    epoch_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0}
    n_batches = len(dataloader)
    objective = VaeObjective()

    for batch in dataloader:
        images = batch["image"].to(device).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.autocast(autocast_device_type, enabled=amp):
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": intensity_loss(reconstruction, images),
                "kl_loss": objective.kl(z_mu, z_sigma),
                "p_loss": perceptual_loss(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = objective.aggregate(losses, kl_weight=kl_weight, perceptual_weight=perceptual_weight)
            loss_g = loss_g + adv_weight * generator_loss

            if scaler_g is not None:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            # Train Discriminator
            # The generator backward traversed the discriminator and stacked
            # gradients on its parameters; only loss_d may drive optimizer_d.
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            if scaler_d is not None:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

        for loss_name, loss_value in losses.items():
            epoch_losses[loss_name] += loss_value.item()

    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    return epoch_losses


def validate_epoch(
    dataloader,
    *,
    autoencoder: torch.nn.Module,
    intensity_loss,
    perceptual_loss,
    infer,
    device,
    autocast_device_type: str = "cuda",
    amp: bool = True,
) -> dict[str, float]:
    """One validation pass over the training loss family (cell 30 tail).

    ``infer(images)`` returns the same ``(reconstruction, z_mu, z_sigma)``
    triple as the autoencoder forward: the caller hands in the cell-30 inferer
    wrapper (``dynamic_infer`` around a Simple/SlidingWindow inferer), so
    whole-volume evaluation stays on the caller. Runs under ``torch.no_grad``
    and does not modify weights.

    Returns:
        dict: epoch-averaged ``{"recons_loss", "kl_loss", "p_loss"}`` floats.
    """
    autoencoder.eval()
    epoch_losses = {"recons_loss": 0.0, "kl_loss": 0.0, "p_loss": 0.0}
    n_batches = len(dataloader)
    objective = VaeObjective()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            with torch.autocast(autocast_device_type, enabled=amp):
                reconstruction, z_mu, z_sigma = infer(images)
            reconstruction = reconstruction.to(device)
            epoch_losses["recons_loss"] += intensity_loss(reconstruction, images).item()
            epoch_losses["kl_loss"] += objective.kl(z_mu, z_sigma).item()
            epoch_losses["p_loss"] += perceptual_loss(reconstruction.float(), images.float()).item()

    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    return epoch_losses
