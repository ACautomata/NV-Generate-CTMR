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

"""BypassMounting -- the single ControlNet-only hook-up sequence (issue #226, spec #221 candidate 3).

The mount both bypass families (mask P2 / cross-modal P3) used to hand-copy in
their ``TrainKernel.load_models``, collapsed into one infrastructure
collaborator: network instantiation (``define_instance``), the allowlisted
DM-source load (``MonaiCheckpoint``), the ``strict=False`` continuation load,
``copy_model_state`` init from the DM encoder/mid, the DDP wrap of the
trainable bypass (the frozen DM stays unwrapped, MAISI convention), the
freeze, the AdamW + PolynomialLR(power 2.0) construction and the per-epoch
checkpoint payload. The recipe values (lr, epochs, batch size) are injected by
the family kernel -- the mount itself carries none (ADR-0005/0007 stay put,
ADR-0016's domain entity partition untouched: the kernel composes the
``DiffusionModel``/``ControlNetBypass`` entities on top of what the mount
produced). Numerics are byte-identical to the collapsed copies; the assembly
gate is tests/infrastructure/test_bypass_mounting.py (S5 seam).

Since #273 (ADR-0019 §3) this class realizes the domain ``BypassMounting``
port and its ``MountedBypass`` record lives in
``ctmr.domain.generation.mounting`` (imported here); the mask family receives
the mount through the port, assembled by the composition root -- the
cross-modal family migrates with #274.
"""

from __future__ import annotations

import monai
import torch
import torch.distributed as dist
from monai.networks.utils import copy_model_state
from torch.nn.parallel import DistributedDataParallel

from ctmr.domain.generation.mounting import MountedBypass
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

__all__ = ["BypassMounting", "MonaiCheckpoint", "MountedBypass"]


class MonaiCheckpoint:
    """The trusted MONAI-pickled training checkpoint loaded under the frozen weights_only allowlist.

    The training checkpoints (the P1 base and the registered P1-DM source)
    pickle MONAI meta-tensor globals alongside the weights; this single point
    activates the allowlist so ``weights_only`` stays enabled at the training
    load sites it serves (P1 full-param continuation, P2/P3 DM-source hook-up
    -- the path is the only thing that varies).
    """

    def __init__(self, path, device):
        self._path = path
        self._device = device

    def load(self):
        torch.serialization.add_safe_globals([monai.data.meta_tensor.MetaTensor, monai.utils.enums.TraceKeys])
        return torch.load(self._path, map_location=self._device, weights_only=True)


class BypassMounting:
    """The one ControlNet-only hook-up: instantiate, load + freeze the DM, init the bypass, session members.

    ``mount`` runs the whole sequence and hands the pieces back; the kernel
    composes the domain entity from them and injects only the recipe values.
    ``checkpoint_payload`` is the single DDP-unwrap + payload build the two
    families shared (the key set stays ``controlnet_state_dict``-shaped,
    ADR-0011 §4).
    """

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    def mount(self, dataset_size, *, lr, n_epochs, batch_size) -> MountedBypass:
        """Run the hook-up; ``dataset_size`` and the recipe values shape the PolynomialLR span."""
        args = self._args
        controlnet = define_instance(args, "controlnet_def").to(self._device)
        unet = define_instance(args, "diffusion_unet_def").to(self._device)
        dm_ckpt = MonaiCheckpoint(args.trained_diffusion_path, self._device).load()
        state = unet.load_state_dict(dm_ckpt["unet_state_dict"], strict=False)
        if state.missing_keys:
            raise ValueError(f"DM source checkpoint missing keys for frozen DM: {state.missing_keys}")
        if state.unexpected_keys:
            self._logger.warning(f"DM source checkpoint unexpected keys (ignored): {state.unexpected_keys}")
        # init the bypass from the frozen DM encoder/mid (spec #51 decision 7 / ADR-0007).
        copy_model_state(controlnet, unet.state_dict())
        # Only the ControlNet is trained; the DM stays a frozen, non-DDP module
        # (MAISI convention: the trainable bypass is DDP-wrapped, the frozen DM is not).
        if dist.is_initialized():
            controlnet = DistributedDataParallel(controlnet, device_ids=[self._device], find_unused_parameters=True)
        scale_factor = float(dm_ckpt["scale_factor"])
        for p in unet.parameters():
            p.requires_grad = False
        unet.eval()
        controlnet.train()
        self._logger.info(f"DM frozen (requires_grad=False); ControlNet init from DM encoder/mid -> {args.trained_diffusion_path}")
        self._logger.info(f"scale_factor reused from P1-DM checkpoint -> {scale_factor}")
        scale_tensor = torch.tensor(scale_factor, device=self._device)

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=lr)
        total_steps = (n_epochs * dataset_size) / batch_size
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
        return MountedBypass(
            trainable=controlnet,
            dm=unet,
            noise_scheduler=define_instance(args, "noise_scheduler"),
            scale=scale_tensor,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )

    def checkpoint_payload(self, trainable, epoch, avg_loss, scale):
        """The per-epoch payload: DDP-unwrap the trainable bypass, keep the pinned key set."""
        controlnet_state = trainable.module.state_dict() if isinstance(trainable, DistributedDataParallel) else trainable.state_dict()
        return {
            "epoch": epoch,
            "loss": avg_loss,
            "num_train_timesteps": self._args.noise_scheduler["num_train_timesteps"],
            "scale_factor": scale,
            "controlnet_state_dict": controlnet_state,
        }
