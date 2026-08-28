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

"""DiffusionScheduler: one denoising trajectory per sample call (ADR-0016, issue #170).

A stateful rich entity for a single concrete denoising run -- created by
``DiffusionModel`` on every ``sample`` call, it holds that trajectory's
timestep sequence and current advance position, and exposes prepare
(``begin``) / step (``step``) / completion (``complete``) behaviour.  It is
terminated when the sampling run ends, is never persisted, and never enters a
checkpoint; its identity holds only within the one sampling session and is
never confused with the weight lineage expressed by ``WeightsRef``.

The step arithmetic is delegated to the wrapped MONAI scheduler it carries
(the pinned production shape), so the entity adds state without changing
math -- per-tensor parity is machine-guarded in
``tests/domain/generation/test_diffusion_scheduler.py``.
"""

from __future__ import annotations

import torch


class DiffusionScheduler:
    """One prepared denoising trajectory over a MONAI scheduler.

    ``begin`` prepares the trajectory (the MONAI ``set_timesteps`` call plus the
    next-timestep chain); ``step`` advances one position; ``complete`` marks the
    terminal boundary after the last step.
    """

    def __init__(self, scheduler, timesteps, next_timesteps):
        self._scheduler = scheduler
        self._timesteps = timesteps
        self._next_timesteps = next_timesteps
        self._position = 0

    @classmethod
    def begin(cls, scheduler, num_inference_steps: int, latent_shape) -> DiffusionScheduler:
        """Prepare a fresh trajectory: MONAI timesteps for this latent grid, chained nexts.

        The sequence is snapshotted (cloned) so the trajectory is immune to any
        later ``set_timesteps`` on the shared MONAI instance -- one trajectory,
        one immutable chain.
        """
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor(latent_shape[2:])),
        )
        timesteps = scheduler.timesteps.clone()
        next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
        return cls(scheduler, timesteps, next_timesteps)

    @property
    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    @property
    def next_timesteps(self) -> torch.Tensor:
        return self._next_timesteps

    @property
    def position(self) -> int:
        return self._position

    @property
    def complete(self) -> bool:
        return self._position >= len(self._timesteps)

    @property
    def current_timestep(self) -> torch.Tensor:
        """The 0-d timestep at the current position (the raw value the MONAI step needs)."""
        return self._timesteps[self._position]

    @property
    def next_timestep(self) -> torch.Tensor:
        """The 0-d next timestep (the chain value the MONAI step needs)."""
        return self._next_timesteps[self._position]

    def add_noise(self, original_samples, noise, timesteps):
        return self._scheduler.add_noise(original_samples=original_samples, noise=noise, timesteps=timesteps)

    def sample_timesteps(self, samples):
        return self._scheduler.sample_timesteps(samples)

    def step(self, model_output: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Advance one position: the MONAI RF step at (t, next_t), then move on.

        Returns the next latent; the trajectory is complete after the last
        chain entry (next_t = 0).
        """
        if self.complete:
            raise RuntimeError("DiffusionScheduler: trajectory already completed; a fresh scheduler must begin each sample call")
        previous, _ = self._scheduler.step(
            model_output=model_output,
            timestep=self.current_timestep,
            sample=sample,
            next_timestep=self.next_timestep,
        )
        self._position += 1
        return previous
