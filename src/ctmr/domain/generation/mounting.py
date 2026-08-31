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

"""BypassMounting: the ControlNet-only hook-up port (ADR-0019 §3, #273).

The mount face the mask-family train kernel drives: ``mount`` runs one
hook-up (instantiate, load + freeze the DM, init the bypass, build the AdamW
+ PolynomialLR session members) and hands back the ``MountedBypass`` record;
``checkpoint_payload`` builds the per-epoch payload (the pinned
``controlnet_state_dict`` key set, ADR-0011 §4). Protocol only -- the
concrete hook-up sequence lives in ``ctmr.infrastructure.bypass_mounting``
and is assembled by the composition root (ADR-0019 §2); the record is a pure
dataclass, floated up here so the port can name what a mount produced.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from monai.networks.schedulers import Scheduler


@dataclass
class MountedBypass:
    """What one ControlNet-only mount produced: the trainable bypass, the frozen DM and the session members."""

    trainable: torch.nn.Module
    dm: torch.nn.Module
    noise_scheduler: Scheduler
    scale: torch.Tensor
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


@runtime_checkable
class BypassMounting(Protocol):
    """The one ControlNet-only hook-up: mount + payload, recipe values injected by the caller."""

    def mount(self, dataset_size: int, *, lr: float, n_epochs: int, batch_size: int) -> MountedBypass:
        """Run the hook-up; ``dataset_size`` and the recipe values shape the PolynomialLR span."""
        ...

    def checkpoint_payload(self, trainable: torch.nn.Module, epoch: int, avg_loss: float, scale) -> dict:
        """The per-epoch payload: DDP-unwrap the trainable bypass, keep the pinned key set (ADR-0015 §4)."""
        ...
