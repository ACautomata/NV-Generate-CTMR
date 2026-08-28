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

"""Sampling-parity gates: DiffusionModel.denoise vs the migrated sampler loop (ADR-0016 testing, issue #170).

Fixed initial latent, spacing, modality, timesteps and CFG on CPU fp32: every
intermediate latent of the domain ``denoise`` trajectory must equal the
migrated ``CandidateSampler.sample_one`` loop (MONAI ``RFlowScheduler`` raw
sequence, CFG double forward, chained ``next_timestep``) -- tested for the
P1 dev-sidecar CFG>0 case and the CFG=0 single-forward case.  The domain path
creates a fresh ``DiffusionScheduler`` per sample call and never reuses
trajectory state across calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.domain.generation.model import DiffusionModel
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
SCALE_FACTOR = 0.87
SEED = 97
NUM_STEPS = 4
LATENT_SHAPE = (1, 4, 8, 8, 4)
SPACING = torch.tensor([[1.0, 1.2, 0.8]], device=CPU)
MODALITY = torch.tensor([29], device=CPU)


def _toy_unet_def():
    """The production config_network_rflow.json topology at toy width."""
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


def _rflow():
    return RFlowScheduler(num_train_timesteps=1000, use_discrete_timesteps=False, use_timestep_transform=True, sample_method="uniform", scale=1.4)


def _fresh_unet():
    args = SimpleNamespace(diffusion_unet_def=_toy_unet_def())
    return define_instance(args, "diffusion_unet_def")


def _legacy_denoise_loop(unet, rflow, initial_latent, spacing, modality, cfg):
    """The migrated ``CandidateSampler.sample_one`` denoising loop, verbatim math.

    Every returned latent is recorded per step (the trajectory is the parity
    target, not only the final image).
    """
    rflow.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=torch.prod(torch.tensor(initial_latent.shape[2:])))
    all_timesteps = rflow.timesteps
    all_next = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    trajectory = [initial_latent.clone()]
    latent = initial_latent
    for t, next_t in zip(all_timesteps, all_next):
        unet_inputs = {
            "x": latent,
            "timesteps": torch.Tensor((t,)).to(latent.device),
            "spacing_tensor": spacing,
            "class_labels": modality,
        }
        if cfg > 0:
            unet_inputs = {
                key: (torch.cat([value, value]) if key != "class_labels" else torch.cat([value, torch.zeros_like(value)]))
                for key, value in unet_inputs.items()
            }
            model_t, model_uncond = unet(**unet_inputs).chunk(2)
            model_output = model_uncond + cfg * (model_t - model_uncond)
        else:
            model_output = unet(**unet_inputs)
        latent, _ = rflow.step(model_output, t, latent, next_t)
        trajectory.append(latent.clone())
    return trajectory


def _assert_trajectories_equal(domain_trajectory, legacy_trajectory, label):
    assert len(domain_trajectory) == len(legacy_trajectory)
    for index, (domain_step, legacy_step) in enumerate(zip(domain_trajectory, legacy_trajectory)):
        assert torch.equal(domain_step, legacy_step), f"{label}: latent drifted at step {index}"


@pytest.mark.parametrize("cfg", [10.0, 0.0])
def test_denoise_trajectory_matches_the_migrated_sampler_loop(cfg):
    unet = _fresh_unet()
    torch.manual_seed(SEED)
    initial_latent = torch.randn(LATENT_SHAPE, device=CPU)  # the fixed initial latent

    legacy = _legacy_denoise_loop(unet, _rflow(), initial_latent, SPACING, MODALITY, cfg)

    model = DiffusionModel(unet=unet, scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())
    scheduler = model.begin_sampling(initial_latent.shape, NUM_STEPS)
    domain_trajectory = [initial_latent.clone()]
    latent = initial_latent
    while not scheduler.complete:
        latent = model.denoise(scheduler, latent, SPACING, MODALITY, cfg)
        domain_trajectory.append(latent.clone())

    _assert_trajectories_equal(domain_trajectory, legacy, f"cfg={cfg}")


def test_fresh_scheduler_per_sample_call_restarts_the_trajectory():
    """Two sample calls with the same seed reproduce the identical trajectory (no state leak)."""
    unet = _fresh_unet()
    model = DiffusionModel(unet=unet, scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())

    def run_once():
        torch.manual_seed(SEED)
        initial = torch.randn(LATENT_SHAPE, device=CPU)
        return model.sample(initial, SPACING, MODALITY, cfg=0.0, num_inference_steps=NUM_STEPS)

    first = run_once()
    second = run_once()
    assert torch.equal(first, second)
