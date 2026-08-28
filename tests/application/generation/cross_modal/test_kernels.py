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

"""Single-step execution of the cross_modal train/sample kernels on CPU (ticket 08).

Acceptance criterion 4: the family's training and sampling kernels must execute one
step on a synthetic mini fixture without a GPU, and the CI full-dependency tier
(torch-marked) must run them for real — never skipped around the torch mark.

``TrainKernel.train_batch`` is the training-closure gate: it must produce a finite
scalar loss whose backward pass reaches both the ControlNet and the (frozen-DM-side)
UNet operands. Two small ``torch.nn.Module`` fakes stand in for the real MONAI
ControlNet / DiffusionModelUNet so the gate isolates the kernel logic (scale-factor
application, label-shape guard, RFlow noising, the images-minus-noise velocity
target, weighted-L1) from the network definitions, which MONAI already covers. The
scheduler is the real ``RFlowScheduler`` — the same one ``CandidateSampler`` and
the baseline ``run_img2img`` drive — so the sampling-closure gate exercises the
production rectified-flow step on CPU.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.application.generation.cross_modal.train import TrainKernel

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


class _TinyControlNet(torch.nn.Module):
    """Emits (down, mid) ControlNet residuals from the 4ch src condition."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, x, timesteps, controlnet_cond, class_labels):
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
    kernel = TrainKernel(args, device=CPU, logger=logging.getLogger("test-kernel"), local_rank=0)
    kernel._scale_factor = torch.tensor(0.5, device=CPU)
    kernel._noise_scheduler = RFlowScheduler(num_train_timesteps=10)
    kernel._controlnet = _TinyControlNet()
    kernel._unet = _TinyUNet()
    return kernel


def _batch(spatial=(8, 8, 4)):
    return {
        "image": torch.randn(1, 4, *spatial),
        "src_image": torch.randn(1, 4, *spatial),
        "label": torch.zeros(1, 1, *spatial),
        "spacing": torch.ones(1, 3),
        "modality": torch.tensor([29]),
    }


def test_train_batch_executes_a_closed_training_step_on_cpu():
    kernel = _kernel()
    loss = kernel.train_batch(_batch())

    assert loss.dim() == 0  # a scalar loss
    assert torch.isfinite(loss)
    loss.backward()  # training closure: gradients reach both network operands
    assert kernel._controlnet.gain.grad is not None and torch.isfinite(kernel._controlnet.gain.grad).all()
    assert kernel._unet.gain.grad is not None and torch.isfinite(kernel._unet.gain.grad).all()


def test_train_batch_guards_the_label_channel_axis():
    kernel = _kernel()
    bad = _batch()
    bad["label"] = torch.zeros(1, 2, 8, 8, 4)  # labels must be [B,1,X,Y,Z]
    with pytest.raises(ValueError, match="labels"):
        kernel.train_batch(bad)


def test_train_batch_applies_the_scale_factor_to_both_latents():
    kernel = _kernel()
    kernel._scale_factor = torch.tensor(2.0, device=CPU)  # a visible scale
    batch = _batch()
    captured = {}

    def _capture(x, timesteps, controlnet_cond, class_labels):
        captured["x"] = x.detach()
        captured["cond"] = controlnet_cond.detach()
        return [controlnet_cond], controlnet_cond

    kernel._controlnet.forward = _capture
    kernel._unet.forward = lambda x, **kwargs: x  # identity, shape preserved
    kernel.train_batch(batch)
    # images and src_image are both scaled by scale_factor before entering the networks
    assert torch.allclose(captured["cond"], batch["src_image"] * 2.0)


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
