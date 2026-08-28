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

"""DiffusionModel: the rich behavioural entity of the generation side (ADR-0016, issue #170).

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
    the modality-label perturber, and -- needed only by ``train_step`` -- the
    optimizer and lr scheduler.  Sampling needs no training session members.
    """

    def __init__(self, unet, scale_factor, noise_scheduler, perturber=None, optimizer=None, lr_scheduler=None):
        self._unet = unet
        self._scale_factor = scale_factor
        self._noise_scheduler = noise_scheduler
        self._perturber = perturber
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

    @property
    def unet(self) -> torch.nn.Module:
        return self._unet

    @property
    def scale_factor(self) -> torch.Tensor:
        return self._scale_factor

    # ------------------------------------------------------------------ training

    def train_step(self, images, spacing, modality, executor: GradientExecutor) -> torch.Tensor:
        """One closed P1 training step: loss → update → lr step.

        ``images`` / ``spacing`` / ``modality`` arrive device-placed (the
        application adapts the loaded batch); the modality perturbation, RF
        uniform timestep draw (scale 1.4), noise injection, UNet forward and the
        velocity L1 target are the pinned P1 recipe.  ``executor`` carries the
        zero_grad / autocast / backward / step execution strategy.
        """
        if self._optimizer is None or self._lr_scheduler is None or self._perturber is None:
            raise ValueError("training session members (perturber, optimizer, lr_scheduler) required for train_step")
        loss = executor.run(lambda: self._training_loss(images, spacing, modality), self._unet, self._optimizer)
        self._lr_scheduler.step()
        return loss

    def _training_loss(self, images, spacing, modality):
        scaled = images * self._scale_factor
        modality_tensor = self._perturber(modality)
        noise = torch.randn_like(scaled)
        timesteps = self._noise_scheduler.sample_timesteps(scaled)
        noisy_latent = self._noise_scheduler.add_noise(original_samples=scaled, noise=noise, timesteps=timesteps)
        model_output = self._unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing, class_labels=modality_tensor)
        return F.l1_loss(model_output.float(), (scaled - noise).float())

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
        if cfg > 0:
            batched = {
                key: (torch.cat([value, value]) if key != "class_labels" else torch.cat([value, torch.zeros_like(value)]))
                for key, value in inputs.items()
            }
            model_t, model_uncond = self._unet(**batched).chunk(2)
            model_output = model_uncond + cfg * (model_t - model_uncond)
        else:
            model_output = self._unet(**inputs)
        return scheduler.step(model_output=model_output, sample=latent)

    def sample(self, initial_latent, spacing, modality, cfg: float, num_inference_steps: int) -> torch.Tensor:
        """The public sampling loop: fresh scheduler per call, trajectory to the end."""
        scheduler = self.begin_sampling(initial_latent.shape, num_inference_steps)
        latent = initial_latent
        while not scheduler.complete:
            latent = self.denoise(scheduler, latent, spacing, modality, cfg)
        return latent
