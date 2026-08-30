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

"""Assembly gate of the single ControlNet-only mount (S5 seam, issue #226).

``BypassMounting`` is the one definition of the hook-up sequence the two
bypass families (mask P2 / cross-modal P3) used to hand-copy: network
instantiation, the allowlisted DM-source load (``MonaiCheckpoint``), the
``strict=False`` continuation load, ``copy_model_state`` init from the DM
encoder/mid, the DDP wrap of the trainable bypass, the freeze, the AdamW +
PolynomialLR(power 2.0) construction and the per-epoch checkpoint payload.
This gate runs the REAL sequence on CPU with the production MAISI networks at
toy width and asserts -- for both family conditioning shapes in one place
(one mount covers the two use-case families): the DM lands frozen, the bypass
trainable, the optimizer carries only bypass parameters, and the init source
is the DM checkpoint (the ADR-0016 "fixed inputs, any machine" tier; the CI
full-dependency torch tier runs these for real, ADR-0015 §6).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from monai.data import MetaTensor
from monai.networks.schedulers import RFlowScheduler

from ctmr.infrastructure.bypass_mounting import BypassMounting, MonaiCheckpoint
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
CKPT_SCALE_FACTOR = 0.87
LR = 1e-4
N_EPOCHS = 2
BATCH_SIZE = 1
DATASET_SIZE = 4
PAYLOAD_KEYS = ["epoch", "loss", "num_train_timesteps", "scale_factor", "controlnet_state_dict"]


def _toy_dm_def():
    """The production config_network_rflow.json DM topology at toy width."""
    return {
        "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 4,
        "num_channels": [32, 64],
        "attention_levels": [False, False],
        "num_head_channels": [0, 32],
        "num_res_blocks": 1,
        "use_flash_attention": False,
        "include_top_region_index_input": False,
        "include_bottom_region_index_input": False,
        "include_spacing_input": True,
        "num_class_embeds": 40,
        "resblock_updown": True,
        "include_fc": True,
    }


def _toy_controlnet_def(conditioning_in_channels):
    """The production ControlNet topology at toy width; 8ch mask (P2) vs 4ch src latent (P3)."""
    return {
        "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
        "spatial_dims": 3,
        "in_channels": 4,
        "num_channels": [32, 64],
        "attention_levels": [False, False],
        "num_head_channels": [0, 32],
        "num_res_blocks": 1,
        "use_flash_attention": False,
        "conditioning_embedding_in_channels": conditioning_in_channels,
        "conditioning_embedding_num_channels": [8],
        "num_class_embeds": 40,
        "resblock_updown": True,
        "include_fc": True,
    }


def _noise_scheduler_def():
    return {
        "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
        "num_train_timesteps": 10,
        "use_discrete_timesteps": False,
        "use_timestep_transform": True,
        "sample_method": "uniform",
        "scale": 1.4,
    }


def _args(tmp_path, conditioning_in_channels):
    dm_def_only = SimpleNamespace(diffusion_unet_def=_toy_dm_def())
    unet = define_instance(dm_def_only, "diffusion_unet_def")
    dm_ckpt = tmp_path / "dm_source.pt"
    # the real DM-source payload layout: unet_state_dict + scale_factor (issue #10 §7 reuse)
    torch.save({"unet_state_dict": unet.state_dict(), "scale_factor": CKPT_SCALE_FACTOR}, str(dm_ckpt))
    return SimpleNamespace(
        diffusion_unet_def=_toy_dm_def(),
        controlnet_def=_toy_controlnet_def(conditioning_in_channels),
        noise_scheduler=_noise_scheduler_def(),
        trained_diffusion_path=str(dm_ckpt),
    )


def _mount(args):
    return BypassMounting(args, device=CPU, logger=logging.getLogger("test-mount")).mount(
        DATASET_SIZE, lr=LR, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE
    )


@pytest.mark.parametrize("conditioning", [8, 4], ids=["mask-8ch", "cross-modal-4ch"])
def test_mount_lands_the_frozen_dm_and_the_trainable_bypass(tmp_path, conditioning):
    mounted = _mount(_args(tmp_path, conditioning))

    # the DM is frozen end to end: no gradient, inference mode
    assert all(not p.requires_grad for p in mounted.dm.parameters())
    assert not mounted.dm.training
    # the bypass is the trainable side
    assert all(p.requires_grad for p in mounted.trainable.parameters())
    assert mounted.trainable.training


@pytest.mark.parametrize("conditioning", [8, 4], ids=["mask-8ch", "cross-modal-4ch"])
def test_optimizer_carries_only_the_bypass_parameters_with_the_injected_recipe(tmp_path, conditioning):
    mounted = _mount(_args(tmp_path, conditioning))

    optimizer_ids = {id(p) for group in mounted.optimizer.param_groups for p in group["params"]}
    bypass_ids = {id(p) for p in mounted.trainable.parameters()}
    dm_ids = {id(p) for p in mounted.dm.parameters()}
    assert optimizer_ids == bypass_ids  # the optimizer sees exactly the bypass
    assert not (optimizer_ids & dm_ids)  # never the frozen DM
    assert mounted.optimizer.param_groups[0]["lr"] == LR
    assert isinstance(mounted.scheduler, torch.optim.lr_scheduler.PolynomialLR)
    assert mounted.scheduler.total_iters == (N_EPOCHS * DATASET_SIZE) / BATCH_SIZE


@pytest.mark.parametrize("conditioning", [8, 4], ids=["mask-8ch", "cross-modal-4ch"])
def test_bypass_init_source_is_the_dm_encoder_mid(tmp_path, conditioning):
    mounted = _mount(_args(tmp_path, conditioning))

    controlnet_state = mounted.trainable.state_dict()
    dm_state = mounted.dm.state_dict()
    shared = {key for key in controlnet_state if key in dm_state and controlnet_state[key].shape == dm_state[key].shape}
    assert shared  # the encoder/mid overlap is real, not an empty match
    for key in shared:
        assert torch.equal(controlnet_state[key], dm_state[key]), key  # copy_model_state copied the DM values
    assert len(controlnet_state) > len(shared)  # the bypass owns weights beyond the copied overlap


@pytest.mark.parametrize("conditioning", [8, 4], ids=["mask-8ch", "cross-modal-4ch"])
def test_scale_is_reused_and_payload_keeps_the_key_set(tmp_path, conditioning):
    args = _args(tmp_path, conditioning)
    mounted = _mount(args)

    assert float(mounted.scale) == pytest.approx(CKPT_SCALE_FACTOR)  # reused from the checkpoint, never recomputed
    assert isinstance(mounted.noise_scheduler, RFlowScheduler)

    payload = BypassMounting(args, device=CPU, logger=logging.getLogger("test-mount")).checkpoint_payload(mounted.trainable, 7, 0.75, mounted.scale)
    assert list(payload) == PAYLOAD_KEYS
    assert payload["epoch"] == 7
    assert payload["loss"] == 0.75
    assert payload["num_train_timesteps"] == 10
    assert set(payload["controlnet_state_dict"]) == set(mounted.trainable.state_dict())


def test_mount_rejects_a_dm_source_checkpoint_missing_keys(tmp_path):
    args = _args(tmp_path, 8)
    torch.save({"unet_state_dict": {}, "scale_factor": 1.0}, args.trained_diffusion_path)
    with pytest.raises(ValueError, match="missing keys for frozen DM"):
        _mount(args)


def test_monai_checkpoint_loads_the_meta_tensor_payload_under_weights_only(tmp_path):
    """The one allowlist activation: the MONAI-pickled payload loads with weights_only enabled."""
    payload_path = tmp_path / "monai_payload.pt"
    torch.save({"unet_state_dict": {"w": MetaTensor(torch.ones(1))}, "scale_factor": CKPT_SCALE_FACTOR}, str(payload_path))

    loaded = MonaiCheckpoint(str(payload_path), CPU).load()

    assert float(loaded["scale_factor"]) == CKPT_SCALE_FACTOR
    assert isinstance(loaded["unet_state_dict"]["w"], MetaTensor)
