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

"""Single-step execution of the cross_modal train kernel on CPU (ticket 08).

Acceptance criterion 4: the family's training kernel must execute one closed
step on a synthetic mini fixture without a GPU, and the CI full-dependency tier
(torch-marked) must run them for real — never skipped around the torch mark.

Per ADR-0016 (issue #174) ``TrainKernel.train_step`` is the thin batch adapter
(scaled-src-latent condition, label-shape guard, weight build) handing the batch
to the domain ``DiffusionModel.train_step`` over a ``ControlNetBypass``
composition: the closed-update gate asserts a finite scalar loss, that the
bypass moved and the frozen P1-DM stayed put. The training-step numerics are
parity-locked in tests/domain/generation/test_candidate_bypass.py. Two small
``torch.nn.Module`` fakes stand in for the real MONAI ControlNet /
DiffusionModelUNet so the gate isolates the kernel logic (scale-factor
application to the condition, label-shape guard) from the network definitions,
which MONAI already covers. The scheduler is the real ``RFlowScheduler`` — the
same one the candidate live sampler and the baseline img2img chain drive — so
the sampling-closure gate exercises the production rectified-flow step on CPU.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from monai.data import MetaTensor
from monai.networks.schedulers import RFlowScheduler

from ctmr.application.generation.cross_modal.train import TrainKernel
from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.infrastructure.bypass_mounting import BypassMounting  # tests are exempt (ADR-0019 §1); the real mounting
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


class _TinyControlNet(torch.nn.Module):
    """Emits (down, mid) ControlNet residuals from the 4ch src condition."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.25))
        self.seen_cond = None

    def forward(self, x, timesteps, controlnet_cond, class_labels):
        self.seen_cond = controlnet_cond.detach()
        residual = controlnet_cond * self.gain
        return [residual], residual  # (down_block_residuals, mid_block_residual)


class _TinyUNet(torch.nn.Module):
    """Consumes the ControlNet residuals and returns a same-shape prediction."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, timesteps, spacing_tensor, down_block_additional_residuals, mid_block_additional_residual, class_labels):
        return x * self.gain + down_block_additional_residuals[0] + mid_block_additional_residual


def _kernel():
    args = SimpleNamespace(
        controlnet_train={"batch_size": 1, "n_epochs": 1, "lr": 1e-4, "weighted_loss": 1.0, "weighted_loss_label": [129, 130, 131]},
        noise_scheduler={"num_train_timesteps": 10},
    )
    # the real mounting collaborator rides in (production assembles it in the
    # composition root, ADR-0019 §2); the mount seam itself is exercised by
    # tests/infrastructure/test_bypass_mounting.py. The gate bypasses mount()
    # -- the kernel receives the tiny fakes directly below.
    logger = logging.getLogger("test-kernel")
    kernel = TrainKernel(args, device=CPU, logger=logger, local_rank=0, mounting=BypassMounting(args, CPU, logger))
    kernel._controlnet = _TinyControlNet()
    unet = _TinyUNet()
    optimizer = torch.optim.AdamW(kernel._controlnet.parameters(), lr=1e-4)
    kernel._model = DiffusionModel(
        unet=unet,
        scale_factor=torch.tensor(0.5, device=CPU),
        noise_scheduler=RFlowScheduler(num_train_timesteps=10),
        bypass=ControlNetBypass(kernel._controlnet),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=2.0),
    )
    return kernel


def _batch(spatial=(8, 8, 4)):
    return {
        "image": torch.randn(1, 4, *spatial),
        "src_image": torch.randn(1, 4, *spatial),
        # the production dataloader's EnsureTyped(track_meta=True) hands the kernel a MetaTensor
        "label": MetaTensor(torch.zeros(1, 1, *spatial, dtype=torch.long)),
        "spacing": torch.ones(1, 3),
        "modality": torch.tensor([29]),
    }


def test_train_step_executes_a_closed_update_on_cpu():
    kernel = _kernel()
    controlnet_before = kernel._controlnet.gain.detach().clone()
    unet_before = kernel._model.unet.gain.detach().clone()

    loss = kernel.train_step(_batch(), PlainGradientExecutor())

    assert loss.dim() == 0  # a scalar loss
    assert torch.isfinite(loss)
    # the closed update: the bypass moved, the frozen P1-DM stayed put
    assert not torch.equal(controlnet_before, kernel._controlnet.gain.detach())
    assert torch.equal(unet_before, kernel._model.unet.gain.detach())


def test_train_step_guards_the_label_channel_axis():
    kernel = _kernel()
    bad = _batch()
    bad["label"] = torch.zeros(1, 2, 8, 8, 4)  # labels must be [B,1,X,Y,Z]
    with pytest.raises(ValueError, match="labels"):
        kernel.train_step(bad, PlainGradientExecutor())


def test_train_step_conditions_on_the_scaled_src_latent():
    """The ONLY structural difference vs the mask family: the condition is the scaled 4ch src latent."""
    kernel = _kernel()
    kernel._model._scale_factor = torch.tensor(2.0, device=CPU)  # a visible scale
    batch = _batch()
    kernel.train_step(batch, PlainGradientExecutor())

    seen = kernel._controlnet.seen_cond
    assert seen.shape == (1, 4, 8, 8, 4)  # the 4ch src latent -- never the 8ch mask
    assert seen.dtype == torch.float32
    # the src latent enters scaled by the DM's scale_factor (the model's normalized space)
    assert torch.allclose(seen, batch["src_image"] * 2.0)


def test_checkpoint_payload_carries_the_controlnet_key_set():
    kernel = _kernel()
    payload = kernel.checkpoint_payload(epoch=7, avg_loss=0.75, scale=0.5)
    assert list(payload) == ["epoch", "loss", "num_train_timesteps", "scale_factor", "controlnet_state_dict"]
    assert payload["num_train_timesteps"] == 10
    assert payload["scale_factor"] == 0.5
    assert set(payload["controlnet_state_dict"]) == {"gain"}


def test_rflow_sampling_step_closes_on_cpu():
    scheduler = RFlowScheduler(num_train_timesteps=10)
    scheduler.set_timesteps(num_inference_steps=3)
    sample = torch.randn(1, 4, 8, 8, 4)
    model_output = torch.randn(1, 4, 8, 8, 4)  # the predicted velocity (images - noise)
    prev = scheduler.step(model_output=model_output, timestep=scheduler.timesteps[0], sample=sample)
    prev_sample = prev[0] if isinstance(prev, tuple) else prev.prev_sample
    assert prev_sample.shape == sample.shape
    assert torch.isfinite(prev_sample).all()
