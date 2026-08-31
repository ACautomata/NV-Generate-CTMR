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

"""Single-step execution and value-level gates of the mask train kernel on CPU (ticket 09, #273).

Acceptance criteria: the family's training kernel must execute one closed step
on a synthetic mini fixture without a GPU, the weighted-target definition must
be kept value-for-value (fixed torch-tensor comparison, now the domain
``TumourWeightedTarget`` gate in tests/domain/generation/test_objective.py),
and the ControlNet-only initialization contract plus the pinned recipe guard
values must stay itemwise unchanged (ADR-0007).

Per ADR-0016 (issue #172) ``TrainKernel.train_step`` is the thin batch adapter
(mask binarization, label-shape guard, weight build) handing the batch to the
domain ``DiffusionModel.train_step`` over a ``ControlNetBypass`` composition:
the closed-update gate asserts a finite scalar loss, that the bypass moved and
the frozen DM stayed put. The weighted-target and training-step numerics are
parity-locked in tests/domain/generation/test_controlnet_bypass.py. Two small
``torch.nn.Module`` fakes stand in for the real MONAI ControlNet /
DiffusionModelUNet; the scheduler is the real ``RFlowScheduler``.

Per ADR-0019 §2/#273 the kernel receives the bypass mounting as the injected
domain port (the composition root assembles the concrete hook-up): the gates
drive a stub mounting and pin the kernel's composition -- entity assembly from
the mount, payload delegation -- not the mount sequence itself (the
infrastructure gate is tests/infrastructure/test_bypass_mounting.py).
Torch-marked, CPU: the CI full-dependency tier runs these for real
(ADR-0015 §6).
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.application.generation.mask.train import TrainKernel
from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.recipe import MaskRecipeSpec
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")


class _StubMounting:
    """The injected mounting port: returns a canned mount, records the payload hand-off."""

    def __init__(self, args, mounted=None):
        self._args = args
        self._mounted = mounted
        self.payload_calls = []
        self.mount_calls = []

    def mount(self, dataset_size, *, lr, n_epochs, batch_size):
        self.mount_calls.append((dataset_size, lr, n_epochs, batch_size))
        return self._mounted

    def checkpoint_payload(self, trainable, epoch, avg_loss, scale):
        self.payload_calls.append((trainable, epoch, avg_loss, scale))
        return {"epoch": epoch, "loss": avg_loss, "passed": True}


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
    kernel = TrainKernel(args, device=CPU, logger=logging.getLogger("test-mask-kernel"), local_rank=0, mounting=_StubMounting(args))
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


def test_train_step_executes_a_closed_update_on_cpu():
    kernel = _kernel()
    controlnet_before = kernel._controlnet.gain.detach().clone()
    unet_before = kernel._model.unet.gain.detach().clone()

    loss = kernel.train_step(_batch(), PlainGradientExecutor())

    assert loss.dim() == 0  # a scalar loss
    assert torch.isfinite(loss)
    # the closed update: the bypass moved, the frozen DM stayed put
    assert not torch.equal(controlnet_before, kernel._controlnet.gain.detach())
    assert torch.equal(unet_before, kernel._model.unet.gain.detach())


def test_train_step_guards_the_label_channel_axis():
    kernel = _kernel()
    bad = _batch()
    bad["label"] = torch.zeros(1, 2, 8, 8, 4)  # labels must be [B,1,X,Y,Z]
    with pytest.raises(ValueError, match="labels"):
        kernel.train_step(bad, PlainGradientExecutor())


def test_train_step_conditions_on_the_binarized_mask():
    """The ONLY structural difference vs cross_modal: the condition is the binarized mask."""
    kernel = _kernel()
    labels = torch.zeros(1, 1, 8, 8, 4, dtype=torch.long)
    labels[0, 0, 0, 0, 0] = 129
    labels[0, 0, 1, 1, 1] = 131
    kernel.train_step(_batch(labels=labels), PlainGradientExecutor())

    seen = kernel._controlnet.seen_cond
    assert seen.shape == (1, 8, 8, 8, 4)  # the 8-bit binary channels
    assert seen.dtype == torch.float32
    assert float(seen[0, 0, 0, 0, 0]) == 1.0 and float(seen[0, 7, 0, 0, 0]) == 1.0  # 129 -> bits {0,7}
    assert float(seen[0, 0, 1, 1, 1]) == 1.0 and float(seen[0, 1, 1, 1, 1]) == 1.0 and float(seen[0, 7, 1, 1, 1]) == 1.0  # 131
    assert float(seen.sum()) == 5.0  # 129+131 expand to 2+3 set bits, nothing else


# --------------------------------------------------- weighted-loss kernel, value-for-value


def test_kernel_builds_the_pinned_weighting_from_the_recipe():
    """The kernel's TumourWeightedTarget carries the ADR-0007 recipe values verbatim.

    The weighted-target math itself (value-for-value against the pinned
    construction) is the domain gate in tests/domain/generation/test_objective.py,
    and the weighted-loss numerics ride the train-step parity in
    tests/domain/generation/test_controlnet_bypass.py.
    """
    kernel = _kernel()
    assert kernel._weighted_target.weight == 100
    assert kernel._weighted_target.labels == [129, 130, 131]


def test_weighted_loss_amplifies_the_same_residual():
    """The x100 ROI weight amplifies the identical residual (weighted > plain)."""
    torch.manual_seed(7)
    labels = torch.zeros(1, 1, 8, 8, 4, dtype=torch.long)
    labels[0, 0, 2, 2, 2] = 129
    batch = _batch(labels=labels)

    plain_kernel = _kernel(weighted_loss=1.0)  # <= 1.0 disables the ROI weight entirely
    weighted_kernel = _kernel(weighted_loss=100)
    torch.manual_seed(7)  # identical noise draw for both kernels
    plain = plain_kernel.train_step(batch, PlainGradientExecutor())
    torch.manual_seed(7)
    weighted = weighted_kernel.train_step(batch, PlainGradientExecutor())

    assert weighted > plain


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


def test_load_models_composes_the_domain_entity_from_the_mount():
    """The kernel composes the DiffusionModel + TrainContext from the mount's
    pieces -- recipe values injected, session members kept as the single
    shared optimizer/scheduler pair."""
    args = SimpleNamespace(
        controlnet_train={"batch_size": 1, "n_epochs": 2, "lr": 1e-5, "weighted_loss": 100, "weighted_loss_label": [129, 130, 131]},
        noise_scheduler={"num_train_timesteps": 10},
    )
    trainable = _TinyControlNet()
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=1e-5)
    mounted = SimpleNamespace(
        trainable=trainable,
        dm=_TinyUNet(),
        noise_scheduler=RFlowScheduler(num_train_timesteps=10),
        scale=torch.tensor(0.5, device=CPU),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=2.0),
    )
    mounting = _StubMounting(args, mounted)
    kernel = TrainKernel(args, device=CPU, logger=logging.getLogger("test-mask-kernel"), local_rank=0, mounting=mounting)

    ctx = kernel.load_models(SimpleNamespace(dataset=[1, 2, 3]))

    assert mounting.mount_calls == [(3, 1e-5, 2, 1)]  # dataset size + the recipe values, verbatim
    assert isinstance(kernel._model, DiffusionModel)
    assert kernel._model.unet is mounted.dm
    assert kernel._model._bypass.controlnet is mounted.trainable
    assert ctx.trainable is mounted.trainable
    assert ctx.optimizer is mounted.optimizer
    assert ctx.scheduler is mounted.scheduler
    assert ctx.scale is mounted.scale


# ---------------------------------------------------------------- checkpoint payload


def test_checkpoint_payload_delegates_to_the_mounting_port():
    """The per-epoch payload hand-off rides the injected mounting; the payload
    key-set contract itself is the infrastructure gate
    (tests/infrastructure/test_bypass_mounting.py)."""
    kernel = _kernel()
    payload = kernel.checkpoint_payload(epoch=7, avg_loss=0.75, scale=0.5)

    trainable, epoch, avg_loss, scale = kernel._mounting.payload_calls[0]
    assert trainable is kernel._controlnet
    assert (epoch, avg_loss, scale) == (7, 0.75, 0.5)
    assert payload == {"epoch": 7, "loss": 0.75, "passed": True}


def test_rflow_sampling_step_closes_on_cpu():
    scheduler = RFlowScheduler(num_train_timesteps=10)
    scheduler.set_timesteps(num_inference_steps=3)
    sample = torch.randn(1, 4, 8, 8, 4)
    model_output = torch.randn(1, 4, 8, 8, 4)  # the predicted velocity (images - noise)
    prev = scheduler.step(model_output=model_output, timestep=scheduler.timesteps[0], sample=sample)
    prev_sample = prev[0] if isinstance(prev, tuple) else prev.prev_sample
    assert prev_sample.shape == sample.shape
    assert torch.isfinite(prev_sample).all()
