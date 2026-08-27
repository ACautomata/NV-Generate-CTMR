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

"""Single-step execution and value-level gates of the mask train kernel on CPU (ticket 09).

Acceptance criteria: the family's training kernel must execute one step on a
synthetic mini fixture without a GPU, the weighted-loss inner kernel must be
kept value-for-value (fixed torch-tensor comparison), and the ControlNet-only
initialization contract plus the pinned recipe guard values must stay itemwise
unchanged (ADR-0007).

``TrainKernel.train_batch`` is the training-closure gate: it must produce a
finite scalar loss whose backward pass reaches both the ControlNet and the
(frozen-DM-side) UNet operands. Two small ``torch.nn.Module`` fakes stand in
for the real MONAI ControlNet / DiffusionModelUNet so the gate isolates the
kernel logic (scale-factor application, mask binarization, label-shape guard,
RFlow noising, the images-minus-noise velocity target, weighted-L1) from the
network definitions, which MONAI already covers. The scheduler is the real
``RFlowScheduler``. Torch-marked, CPU: the CI full-dependency tier runs these
for real (ADR-0015 §6).
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from monai.networks.schedulers import RFlowScheduler

from ctmr.application.generation.mask.inference import binarize_labels
from ctmr.application.generation.mask.train import TrainKernel
from ctmr.domain.recipe import MaskRecipeSpec

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


class _TinyControlNet(torch.nn.Module):
    """Emits (down, mid) ControlNet residuals from the 8ch binary mask condition."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.25))
        self.seen_cond = None

    def forward(self, x, timesteps, controlnet_cond, class_labels):
        self.seen_cond = controlnet_cond.detach()
        residual = controlnet_cond.sum(dim=1, keepdim=True) * self.gain
        return [residual], residual  # (down_block_residuals, mid_block_residual)


class _TinyUNet(torch.nn.Module):
    """Consumes the ControlNet residuals and returns a same-shape prediction."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, timesteps, spacing_tensor, down_block_additional_residuals, mid_block_additional_residual, class_labels):
        return x * self.gain + down_block_additional_residuals[0] + mid_block_additional_residual


def _kernel(weighted_loss=100, weighted_loss_label=(129, 130, 131)):
    args = SimpleNamespace(
        controlnet_train={
            "batch_size": 1,
            "n_epochs": 1,
            "lr": 1e-4,
            "weighted_loss": weighted_loss,
            "weighted_loss_label": list(weighted_loss_label),
        },
        noise_scheduler={"num_train_timesteps": 10},
    )
    kernel = TrainKernel(args, device=CPU, logger=logging.getLogger("test-mask-kernel"), local_rank=0)
    kernel._scale_factor = torch.tensor(0.5, device=CPU)
    kernel._noise_scheduler = RFlowScheduler(num_train_timesteps=10)
    kernel._controlnet = _TinyControlNet()
    kernel._unet = _TinyUNet()
    return kernel


def _batch(spatial=(8, 8, 4), labels=None):
    if labels is None:
        labels = torch.zeros(1, 1, *spatial, dtype=torch.long)
    from monai.data import MetaTensor

    return {
        "image": torch.randn(1, 4, *spatial),
        # the production dataloader's EnsureTyped(track_meta=True) hands the kernel a MetaTensor
        "label": MetaTensor(labels),
        "spacing": torch.ones(1, 3),
        "modality": torch.tensor([29]),
    }


# ------------------------------------------------------------------- training closure


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


def test_train_batch_conditions_on_the_binarized_mask():
    """The ONLY structural difference vs cross_modal: the condition is the binarized mask."""
    kernel = _kernel()
    labels = torch.zeros(1, 1, 8, 8, 4, dtype=torch.long)
    labels[0, 0, 0, 0, 0] = 129
    labels[0, 0, 1, 1, 1] = 131
    kernel.train_batch(_batch(labels=labels))

    seen = kernel._controlnet.seen_cond
    assert seen.shape == (1, 8, 8, 8, 4)  # the 8-bit binary channels
    assert seen.dtype == torch.float32
    assert float(seen[0, 0, 0, 0, 0]) == 1.0 and float(seen[0, 7, 0, 0, 0]) == 1.0  # 129 -> bits {0,7}
    assert float(seen[0, 0, 1, 1, 1]) == 1.0 and float(seen[0, 1, 1, 1, 1]) == 1.0 and float(seen[0, 7, 1, 1, 1]) == 1.0  # 131
    assert float(seen.sum()) == 5.0  # 129+131 expand to 2+3 set bits, nothing else


# --------------------------------------------------- weighted-loss kernel, value-for-value


def test_weighted_target_kernel_is_value_identical_to_the_pinned_construction():
    """weights = 1 everywhere, = weighted_loss on the {129,130,131} ROI (fixed-tensor gate)."""
    kernel = _kernel(weighted_loss=100)
    labels = torch.zeros(1, 1, 4, 8, 8, dtype=torch.long)  # deliberately off the latent grid
    labels[0, 0, 0, 0, 0] = 129
    labels[0, 0, 1, 2, 3] = 130
    labels[0, 0, 3, 5, 1] = 131
    labels[0, 0, 2, 1, 1] = 22  # background id: never weighted
    labels[0, 0, 2, 6, 6] = 7  # an unrelated id: never weighted
    images = torch.randn(1, 4, 8, 8, 4)

    weights = kernel._weighted_target(labels, images)

    expected = torch.ones_like(images)
    # hand-built expectation: nearest-neighbour upsample of the three tumour ids onto the image grid,
    # then broadcast across the 4 latent channels (the kernel's repeat)
    upsampled = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest").long()
    roi = torch.isin(upsampled, torch.tensor([129, 130, 131])).repeat(1, images.shape[1], 1, 1, 1)
    expected[roi] = 100.0
    assert torch.equal(weights, expected)
    # and the ROI truly contains the labelled voxels after the upsample
    assert float(weights.max()) == 100.0
    assert int((weights == 100.0).sum()) == int(roi.sum())


def test_weighted_loss_changes_the_loss_value_by_exactly_the_pinned_weighting():
    """The weighted branch reproduces its arithmetic independently on a fixed batch."""
    torch.manual_seed(7)
    labels = torch.zeros(1, 1, 8, 8, 4, dtype=torch.long)
    labels[0, 0, 2, 2, 2] = 129
    batch = _batch(labels=labels)

    plain_kernel = _kernel(weighted_loss=1.0)  # <= 1.0 disables the ROI weight entirely
    weighted_kernel = _kernel(weighted_loss=100)
    torch.manual_seed(7)  # identical noise draw for both kernels
    plain = plain_kernel.train_batch(batch)
    torch.manual_seed(7)
    weighted = weighted_kernel.train_batch(batch)

    assert weighted > plain  # the x100 ROI term amplifies the same residual

    # independent re-derivation of the weighted branch's value
    images = batch["image"] * weighted_kernel._scale_factor
    torch.manual_seed(7)  # reproduce the exact noise draw the kernel saw (randn_like under seed 7)
    noise = torch.randn(images.shape)
    timesteps = weighted_kernel._noise_scheduler.sample_timesteps(images)
    noisy = weighted_kernel._noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
    cond = binarize_labels(batch["label"].as_tensor().to(torch.long)).float()
    down, mid = weighted_kernel._controlnet(x=noisy, timesteps=timesteps, controlnet_cond=cond, class_labels=batch["modality"])
    model_output = weighted_kernel._unet(
        x=noisy,
        timesteps=timesteps,
        spacing_tensor=batch["spacing"],
        down_block_additional_residuals=down,
        mid_block_additional_residual=mid,
        class_labels=batch["modality"],
    )
    model_gt = images - noise
    weights = torch.ones_like(images)
    roi = F.interpolate(batch["label"].float(), size=images.shape[2:], mode="nearest").long()
    weights[torch.isin(roi, torch.tensor([129, 130, 131])).repeat(1, images.shape[1], 1, 1, 1)] = 100.0
    expected = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
    assert torch.isfinite(expected)
    assert torch.allclose(weighted, expected)


def test_weighted_target_below_threshold_returns_none():
    kernel = _kernel(weighted_loss=1.0)
    images = torch.randn(1, 4, 8, 8, 4)
    assert kernel._weighted_target(torch.zeros(1, 1, 8, 8, 4, dtype=torch.long), images) is None


# ------------------------------------------------- ControlNet-only init + recipe guard values


def test_pinned_recipe_guard_values_are_itemwise_unchanged():
    """ADR-0007 pins, one assertion each -- no value may drift with the migration."""
    assert MaskRecipeSpec.PINNED_LR == 1e-5
    assert MaskRecipeSpec.PINNED_BATCH == 1
    assert MaskRecipeSpec.PINNED_WEIGHTED_LOSS == 100
    assert MaskRecipeSpec.PINNED_WEIGHTED_LABELS == [129, 130, 131]
    assert MaskRecipeSpec.PINNED_CACHE_RATE == 0
    assert MaskRecipeSpec.MAX_EPOCHS == 100


class _QuietLogger:
    def info(self, message):
        pass


def test_mask_recipe_guard_passes_the_pinned_config_and_blocks_deviations():
    pinned = {"lr": 1e-5, "batch_size": 1, "weighted_loss": 100, "weighted_loss_label": [129, 130, 131], "cache_rate": 0, "n_epochs": 100}
    assert MaskRecipeSpec(pinned, _QuietLogger()).check() is True
    deviated = {**pinned, "lr": 2e-5}
    with pytest.raises(ValueError, match=re.escape("pinned mask lr is 1e-05, got 2e-05 (ADR-0007)")):
        MaskRecipeSpec(deviated, _QuietLogger()).check()
    deviated = {**pinned, "weighted_loss": 50}
    with pytest.raises(ValueError, match="weighted_loss is 100"):
        MaskRecipeSpec(deviated, _QuietLogger()).check()


# ---------------------------------------------------------------- checkpoint payload


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
