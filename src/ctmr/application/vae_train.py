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

"""VAE + PatchDiscriminator GAN training orchestration (ADR-0015 §2/§8, batch M6).

Extracted from ``train_vae_tutorial.ipynb`` cells 24/26/28/30 -- the repo's
only GAN training loop -- when the notebooks were retired. The construction
and schedule settings are PINNED by ``tests/application/test_vae_train.py``;
do not drift them without re-pinning there.

Composition over inheritance: the loss objects, optimizers, warmup
schedulers and AMP scalers are plain members built by the module-level
``build_*`` helpers (every component injectable for tests), while the
alternating Generator/Discriminator loop keeps the notebook's algorithmic
shape as methods of :class:`VaeGanTrainer`. The former notebook-only
visualisation (XYZ snapshot panels, interactive display) sinks into an
optional ``visualize`` callback instead of living inside the loop.
"""

import logging
import math
from collections.abc import Callable

import torch
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler

from ctmr.domain.losses import KL_loss

_logger = logging.getLogger(__name__)

#: Optional hook replacing the retired notebook visualisation: receives
#: ``(images, reconstruction, writer, epoch)`` for one validation batch.
Visualizer = Callable[[torch.Tensor, torch.Tensor, object, int], None]


def build_discriminator(spatial_dims: int) -> PatchDiscriminator:
    """Build the VAE adversarial partner (cell 24): INSTANCE-normed patch head."""
    return PatchDiscriminator(
        spatial_dims=spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    )


def build_intensity_loss(recon_loss: str):
    """L2 -> MSE, anything else -> L1 (cell 26 follows config ``recon_loss``)."""
    return MSELoss() if recon_loss == "l2" else L1Loss(reduction="mean")


def warmup_rule(epoch: int) -> float:
    """Three-segment LR warmup ladder (cell 26); fixture-pinned boundaries."""
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    else:
        return 1.0


def build_optimizers(autoencoder, discriminator, *, lr: float, amp: bool):
    """Paired Adams whose eps depends on AMP (cell 26: 1e-6 under AMP, else 1e-8)."""
    eps = 1e-06 if amp else 1e-08
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=lr, eps=eps)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=lr, eps=eps)
    return optimizer_g, optimizer_d


def build_schedulers(optimizer_g, optimizer_d):
    """LambdaLR pair sharing :func:`warmup_rule` (cell 26)."""
    return (
        lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule),
        lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule),
    )


def build_scalers(amp: bool):
    """AMP GradScaler pair (cell 26), or ``(None, None)`` in full precision."""
    if not amp:
        return None, None
    scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
    scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
    return scaler_g, scaler_d


def build_val_inferer(val_sliding_window_patch_size, sw_device) -> object:
    """Validation inferer (cell 30): sliding window when patches are given, else whole-volume."""
    if not val_sliding_window_patch_size:
        return SimpleInferer()
    return SlidingWindowInferer(
        roi_size=val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.0,
        device=torch.device("cpu"),
        sw_device=sw_device,
    )


def loss_weighted_sum(losses: dict, kl_weight: float, perceptual_weight: float):
    """Total VAE objective (cell 30 closure): recon + kl_weight*KL + perceptual_weight*p."""
    return losses["recons_loss"] + kl_weight * losses["kl_loss"] + perceptual_weight * losses["p_loss"]


class VaeGanTrainer:
    """Orchestrates the extracted VAE/GAN training loop (cells 24/26/28/30).

    Callers supply the autoencoder (any network returning
    ``(reconstruction, z_mu, z_sigma)``) and the discriminator; every loss,
    the metric writer, the logger and the validation visualiser are injectable
    so tests can drive the full loop with toy components on CPU.
    """

    def __init__(
        self,
        autoencoder,
        discriminator,
        device,
        writer,
        *,
        kl_weight: float,
        adv_weight: float,
        perceptual_weight: float,
        lr: float,
        recon_loss: str = "l1",
        amp: bool = False,
        intensity_loss=None,
        adv_loss=None,
        perceptual_loss=None,
        visualize: Visualizer | None = None,
    ) -> None:
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.device = device
        self.writer = writer
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        self.amp = amp
        self.visualize = visualize

        self.intensity_loss = intensity_loss if intensity_loss is not None else build_intensity_loss(recon_loss)
        self.adv_loss = adv_loss if adv_loss is not None else PatchAdversarialLoss(criterion="least_squares")
        self.perceptual = (
            perceptual_loss
            if perceptual_loss is not None
            else PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
            .eval()
            .to(device)
        )
        self.optimizer_g, self.optimizer_d = build_optimizers(autoencoder, discriminator, lr=lr, amp=amp)
        self.scheduler_g, self.scheduler_d = build_schedulers(self.optimizer_g, self.optimizer_d)
        self.scaler_g, self.scaler_d = build_scalers(amp)
        self.total_step = 0
        self.best_val_loss = 10000000.0

    def load_finetune_weights(self, path) -> None:
        """Finetune branch (cell 28): accepts raw state dicts and payload-style ones."""
        checkpoint_autoencoder = torch.load(path)
        if "unet_state_dict" in checkpoint_autoencoder.keys():
            checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
        self.autoencoder.load_state_dict(checkpoint_autoencoder)
        _logger.info("Finetune on pretrained model %s", path)

    @staticmethod
    def _forward_with_inferer(inferer, model, images):
        """The ported ``dynamic_infer`` semantics (scripts/utils.py): single-patch images skip the inferer."""
        if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
            return model(images)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output

    def validate(self, dataloader_val, val_inferer, epoch: int) -> tuple:
        """One validation pass (cell 30): eval-mode recon/KL/perceptual averages + monitoring."""
        self.autoencoder.eval()
        val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
        for batch in dataloader_val:
            with torch.no_grad():
                with autocast("cuda", enabled=self.amp):
                    images = batch["image"]
                    reconstruction, z_mu, z_sigma = self._forward_with_inferer(val_inferer, self.autoencoder, images)
                    reconstruction = reconstruction.to(self.device)
                    val_epoch_losses["recons_loss"] += self.intensity_loss(reconstruction, images.to(self.device)).item()
                    val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_epoch_losses["p_loss"] += self.perceptual(reconstruction, images.to(self.device)).item()

        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(dataloader_val)
        val_loss_g = loss_weighted_sum(val_epoch_losses, self.kl_weight, self.perceptual_weight)
        _logger.info("Epoch %d val_vae_loss %s: %s.", epoch, val_loss_g, val_epoch_losses)

        for loss_name, loss_value in val_epoch_losses.items():
            self.writer.add_scalar(loss_name, loss_value, epoch)

        # Monitor scale_factor: we'd like to tune kl_weights to make scale_factor close to 1.
        scale_factor_sample = 1.0 / z_mu.flatten().std()
        self.writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, epoch)

        if self.visualize is not None:
            self.visualize(images, reconstruction, self.writer, epoch)
        return val_epoch_losses, val_loss_g

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        *,
        n_epochs: int,
        val_interval: int,
        checkpoint_path_g,
        checkpoint_path_d,
        val_sliding_window_patch_size=None,
    ) -> dict:
        """Full epoch loop (cell 30): GAN alternation, schedules, saves, periodic validation."""
        max_epochs = n_epochs
        val_inferer = build_val_inferer(val_sliding_window_patch_size, self.device)
        for epoch in range(max_epochs):
            _logger.info("lr: %s", self.scheduler_g.get_lr())
            self.autoencoder.train()
            self.discriminator.train()
            train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

            for batch in dataloader_train:
                images = batch["image"].to(self.device).contiguous()
                self.optimizer_g.zero_grad(set_to_none=True)
                self.optimizer_d.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=self.amp):
                    # Train Generator
                    reconstruction, z_mu, z_sigma = self.autoencoder(images)
                    losses = {
                        "recons_loss": self.intensity_loss(reconstruction, images),
                        "kl_loss": KL_loss(z_mu, z_sigma),
                        "p_loss": self.perceptual(reconstruction.float(), images.float()),
                    }
                    logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g = (
                        loss_weighted_sum(losses, self.kl_weight, self.perceptual_weight)
                        + self.adv_weight * generator_loss
                    )

                    if self.amp:
                        self.scaler_g.scale(loss_g).backward()
                        self.scaler_g.unscale_(self.optimizer_g)
                        self.scaler_g.step(self.optimizer_g)
                        self.scaler_g.update()
                    else:
                        loss_g.backward()
                        self.optimizer_g.step()

                    # Train Discriminator
                    logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = self.discriminator(images.contiguous().detach())[-1]
                    loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    loss_d = (loss_d_fake + loss_d_real) * 0.5

                    if self.amp:
                        self.scaler_d.scale(loss_d).backward()
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                    else:
                        loss_d.backward()
                        self.optimizer_d.step()

                # Log training loss
                self.total_step += 1
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), self.total_step)
                    train_epoch_losses[loss_name] += loss_value.item()
                self.writer.add_scalar("train_adv_loss_iter", generator_loss, self.total_step)
                self.writer.add_scalar("train_fake_loss_iter", loss_d_fake, self.total_step)
                self.writer.add_scalar("train_real_loss_iter", loss_d_real, self.total_step)

            self.scheduler_g.step()
            self.scheduler_d.step()
            for key in train_epoch_losses:
                train_epoch_losses[key] /= len(dataloader_train)
            _logger.info(
                "Epoch %d train_vae_loss %s: %s.",
                epoch,
                loss_weighted_sum(train_epoch_losses, self.kl_weight, self.perceptual_weight),
                train_epoch_losses,
            )
            for loss_name, loss_value in train_epoch_losses.items():
                self.writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)
            torch.save(self.autoencoder.state_dict(), checkpoint_path_g)
            torch.save(self.discriminator.state_dict(), checkpoint_path_d)
            _logger.info("Save trained autoencoder to %s", checkpoint_path_g)
            _logger.info("Save trained discriminator to %s", checkpoint_path_d)

            # Validation
            if epoch % val_interval == 0:
                val_epoch_losses, val_loss_g = self.validate(dataloader_val, val_inferer, epoch)
                if val_loss_g < self.best_val_loss:
                    self.best_val_loss = val_loss_g
                    trained_g_path_epoch = f"{checkpoint_path_g[:-3]}_epoch{epoch}.pt"
                    torch.save(self.autoencoder.state_dict(), trained_g_path_epoch)
                    _logger.info("Got best val vae loss.")
                    _logger.info("Save trained autoencoder to %s", trained_g_path_epoch)

        return {
            "best_val_loss": self.best_val_loss,
            "total_steps": self.total_step,
        }
