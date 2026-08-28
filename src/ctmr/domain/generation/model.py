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

"""DiffusionModel: the rich behavioural entity of the generation side (ADR-0016, issues #170/#172).

``DiffusionModel`` carries the UNet weights, the scale_factor and the
training/sampling recipe as an anemic-checkpoint-free behavioural entity: its
``train_step`` drives one complete parameter update (batch loss → backward →
optimizer step, precision policy injected through ``GradientExecutor``) and
its ``sample``/``denoise`` drive the denoising loop through a
``DiffusionScheduler`` created fresh per ``sample`` call (CFG composition and
timestep advancement included).  Epoch loops, data loading, process spawning
and disk orchestration stay out of the entity (application layer); the entity
carries no identity -- checkpoint lineage stays expressed by ``WeightsRef`` and
the runtime instance is rebuilt by the application from checkpoint payloads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ctmr.domain.generation.scheduler import DiffusionScheduler
from ctmr.domain.generation.update import GradientExecutor


class DiffusionModel:
    """Behavioural carrier for one generation pipeline's train/sample rules.

    Constructor members: the (possibly DDP-wrapped) UNet, the scale_factor
    tensor, the MONAI noise scheduler instance carrying the RF configuration,
    the modality-label perturber (P1) or the ControlNet bypass (P2), and --
    needed only by ``train_step`` -- the optimizer and lr scheduler.
    """

    def __init__(self, unet, scale_factor, noise_scheduler, perturber=None, optimizer=None, lr_scheduler=None, bypass=None):
        self._unet = unet
        self._scale_factor = scale_factor
        self._noise_scheduler = noise_scheduler
        self._perturber = perturber
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._bypass = bypass

    # the unet-input keys the CFG batch-doubling must pass through untouched
    # (the bypass already ran them as the batch=2 conditioned|unconditional pair)
    _RESIDUAL_KEYS = ("down_block_additional_residuals", "mid_block_additional_residual")

    @property
    def unet(self) -> torch.nn.Module:
        return self._unet

    @property
    def scale_factor(self) -> torch.Tensor:
        return self._scale_factor

    # ------------------------------------------------------------------ training

    def train_step(self, images, spacing, modality, executor: GradientExecutor, mask_condition=None, target_weights=None) -> torch.Tensor:
        """One closed training step: loss → update → lr step.

        ``images`` / ``spacing`` / ``modality`` arrive device-placed (the
        application adapts the loaded batch).  The P1 recipe adds the modality
        perturbation; the P2 recipe (issue #172) passes the binarized mask as
        ``mask_condition`` (the bypass-conditioned forward) and the
        ``TumourWeightedTarget`` weights as ``target_weights`` (the weighted
        velocity L1).  ``executor`` carries the zero_grad / autocast / backward
        / step execution strategy.
        """
        if self._optimizer is None or self._lr_scheduler is None:
            raise ValueError("training session members (optimizer, lr_scheduler) required for train_step")
        if self._bypass is None and self._perturber is None:
            raise ValueError("training session members (perturber or bypass, optimizer, lr_scheduler) required for train_step")
        if self._bypass is not None and mask_condition is None:
            raise ValueError("bypass-conditioned train_step requires mask_condition")
        if self._bypass is None and mask_condition is not None:
            raise ValueError("mask_condition requires a configured ControlNetBypass")
        loss = executor.run(lambda: self._training_loss(images, spacing, modality, mask_condition, target_weights), self._unet, self._optimizer)
        self._lr_scheduler.step()
        return loss

    def _training_loss(self, images, spacing, modality, mask_condition=None, target_weights=None):
        scaled = images * self._scale_factor
        modality_tensor = self._perturber(modality) if self._perturber is not None else modality
        noise = torch.randn_like(scaled)
        timesteps = self._noise_scheduler.sample_timesteps(scaled)
        noisy_latent = self._noise_scheduler.add_noise(original_samples=scaled, noise=noise, timesteps=timesteps)
        if self._bypass is not None:
            down, mid = self._bypass.residuals(noisy_latent, timesteps, mask_condition, modality_tensor)
            model_output = self._unet(
                x=noisy_latent,
                timesteps=timesteps,
                spacing_tensor=spacing,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                class_labels=modality_tensor,
            )
        else:
            model_output = self._unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing, class_labels=modality_tensor)
        model_gt = scaled - noise
        if target_weights is not None:
            return (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * target_weights).mean()
        return F.l1_loss(model_output.float(), model_gt.float())

    # ------------------------------------------------------------------ sampling

    def begin_sampling(self, latent_shape, num_inference_steps: int) -> DiffusionScheduler:
        """Prepare a fresh denoising trajectory for one sample call (ADR-0016: never reused)."""
        return DiffusionScheduler.begin(self._noise_scheduler, num_inference_steps, latent_shape)

    def denoise(self, scheduler: DiffusionScheduler, latent, spacing, modality, cfg: float) -> torch.Tensor:
        """One CFG-composed denoising step, advancing ``scheduler`` one position.

        ``cfg > 0`` runs the classifier-free-guidance double forward (conditioned
        vs zero-label unconditional, combined ``uncond + cfg * (cond - uncond)``)
        exactly like the migrated sampler; ``cfg == 0`` is the plain single
        forward.  Returns the next latent; the trajectory completes at the
        chained ``next_timestep = 0`` boundary.
        """
        inputs = {
            "x": latent,
            "timesteps": torch.Tensor((scheduler.current_timestep,)).to(latent.device),
            "spacing_tensor": spacing,
            "class_labels": modality,
        }
        return scheduler.step(model_output=self._unet_output(inputs, cfg), sample=latent)

    def denoise_conditioned(self, scheduler: DiffusionScheduler, latent, spacing, modality, controlnet_cond, uncond_cond, cfg: float) -> torch.Tensor:
        """One bypass-conditioned denoising step (the P2 live sampler, issue #172).

        The bypass supplies the (down, mid) residuals -- the CFG pair forward
        when ``uncond_cond`` travels with the call and ``cfg > 0`` -- and the
        DM forward composes the guidance on top: the residuals stay the
        batch=2 pair they arrived as while every other input duplicates
        (``class_labels = (modality | zeros)``), then
        ``uncond + cfg * (cond - uncond)``.  ``cfg == 0`` is the plain single
        conditioned forward.
        """
        if self._bypass is None:
            raise ValueError("denoise_conditioned requires a configured ControlNetBypass")
        timestep = torch.Tensor((scheduler.current_timestep,)).to(latent.device)
        down, mid = self._bypass.residuals(latent, timestep, controlnet_cond, modality, uncond_cond if cfg > 0 else None)
        inputs = {
            "x": latent,
            "timesteps": timestep,
            "spacing_tensor": spacing,
            "down_block_additional_residuals": down,
            "mid_block_additional_residual": mid,
            "class_labels": modality,
        }
        return scheduler.step(model_output=self._unet_output(inputs, cfg), sample=latent)

    def _unet_output(self, inputs, cfg: float):
        """The CFG-composed UNet output shared by the plain and bypass-conditioned denoise.

        ``cfg > 0`` duplicates every input (``class_labels`` gets the zero-label
        unconditional half) except the bypass residuals, which stay the batch=2
        pair the bypass produced, then composes ``uncond + cfg * (cond - uncond)``.
        """
        if cfg > 0:
            batched = {}
            for key, value in inputs.items():
                if key in self._RESIDUAL_KEYS:
                    batched[key] = value
                elif key == "class_labels":
                    batched[key] = torch.cat([value, torch.zeros_like(value)])
                else:
                    batched[key] = torch.cat([value, value])
            model_t, model_uncond = self._unet(**batched).chunk(2)
            return model_uncond + cfg * (model_t - model_uncond)
        return self._unet(**inputs)

    def sample(self, initial_latent, spacing, modality, cfg: float, num_inference_steps: int) -> torch.Tensor:
        """The public sampling loop: fresh scheduler per call, trajectory to the end."""
        scheduler = self.begin_sampling(initial_latent.shape, num_inference_steps)
        latent = initial_latent
        while not scheduler.complete:
            latent = self.denoise(scheduler, latent, spacing, modality, cfg)
        return latent
