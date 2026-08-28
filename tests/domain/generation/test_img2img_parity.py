# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""img2img sampling-parity gates: DiffusionModel.begin_img2img vs the migrated run_img2img chain (ADR-0016 testing, issue #173).

Fixed seed, src latent, spacing, modality and strength on CPU fp32: every
intermediate latent of the domain img2img trajectory must equal the migrated
``run_img2img`` chain (MONAI ``RFlowScheduler`` raw sequence, the strict-greater
strength truncation, the ``add_noise`` interpolation start
``x_t = (1-t)*src*scale_factor + t*noise``, CFG double forward, chained
``next_timestep``) -- tested for the CFG>0 and CFG=0 cases across the strength
boundaries (full trajectory at strength 1.0, the production 0.9, a deeper
0.75) plus the over-truncation rejection.  The noise draw happens after
``set_timesteps`` on both paths, so the fixed seed pins identical noise.
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
NUM_STEPS = 30
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


def _legacy_img2img_chain(unet, rflow, src_latent, scale_factor, strength, spacing, modality, cfg):
    """The migrated ``run_img2img`` img2img chain, verbatim math.

    Every returned latent is recorded per step (the trajectory is the parity
    target, not only the final image).
    """
    rflow.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=torch.prod(torch.tensor(src_latent.shape[2:])))
    all_timesteps = rflow.timesteps
    threshold = float(strength) * rflow.num_train_timesteps
    start_idx = int((all_timesteps > threshold).sum())
    if start_idx >= len(all_timesteps) - 1:
        raise ValueError(f"strength={strength} 截断后步数不足（timesteps={all_timesteps.tolist()[:5]}...）")
    timesteps = all_timesteps[start_idx:]
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))

    noise = torch.randn(src_latent.shape, device=src_latent.device, dtype=src_latent.dtype)
    latent_norm = src_latent * scale_factor
    latent = rflow.add_noise(original_samples=latent_norm, noise=noise, timesteps=timesteps[:1].to(src_latent.device))

    trajectory = [latent.clone()]
    for t, next_t in zip(timesteps, next_timesteps):
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


@pytest.mark.parametrize("strength", [1.0, 0.9, 0.75])
@pytest.mark.parametrize("cfg", [10.0, 0.0])
def test_img2img_trajectory_matches_the_migrated_run_img2img_chain(cfg, strength):
    # networks built before the seed so their random init never consumes the seeded stream (#170 parity order)
    unet = _fresh_unet()
    torch.manual_seed(SEED)
    src_latent = torch.randn(LATENT_SHAPE, device=CPU)  # the fixed src latent

    legacy = _legacy_img2img_chain(unet, _rflow(), src_latent, SCALE_FACTOR, strength, SPACING, MODALITY, cfg)

    model = DiffusionModel(unet=unet, scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())
    torch.manual_seed(SEED)  # replay the same RNG stream: src latent draw, then the img2img noise draw
    src_latent = torch.randn(LATENT_SHAPE, device=CPU)
    scheduler, latent = model.begin_img2img(src_latent, strength, NUM_STEPS)
    domain_trajectory = [latent.clone()]
    while not scheduler.complete:
        latent = model.denoise(scheduler, latent, SPACING, MODALITY, cfg)
        domain_trajectory.append(latent.clone())

    _assert_trajectories_equal(domain_trajectory, legacy, f"cfg={cfg} strength={strength}")


def test_strength_one_keeps_the_full_trajectory():
    """strength=1.0 truncates nothing: the strict-greater threshold keeps every timestep."""
    unet = _fresh_unet()
    model = DiffusionModel(unet=unet, scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())
    src_latent = torch.randn(LATENT_SHAPE, device=CPU)

    scheduler, _ = model.begin_img2img(src_latent, 1.0, NUM_STEPS)

    reference = _rflow()
    reference.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=int(torch.prod(torch.tensor(LATENT_SHAPE[2:]))))
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    assert len(scheduler.timesteps) == NUM_STEPS


def test_over_truncation_strength_is_rejected():
    model = DiffusionModel(unet=_fresh_unet(), scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())
    src_latent = torch.randn(LATENT_SHAPE, device=CPU)

    with pytest.raises(ValueError, match="fewer than two"):
        model.begin_img2img(src_latent, 0.001, NUM_STEPS)


def test_strength_leaving_exactly_two_steps_is_accepted():
    """The over-truncation boundary is strict: two kept steps still run (the legacy acceptance)."""
    reference = _rflow()
    reference.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=int(torch.prod(torch.tensor(LATENT_SHAPE[2:]))))
    timesteps = reference.timesteps
    threshold = (float(timesteps[-3]) + float(timesteps[-2])) / 2.0
    strength = threshold / reference.num_train_timesteps

    model = DiffusionModel(unet=_fresh_unet(), scale_factor=torch.tensor(SCALE_FACTOR), noise_scheduler=_rflow())
    src_latent = torch.randn(LATENT_SHAPE, device=CPU)
    scheduler, _ = model.begin_img2img(src_latent, strength, NUM_STEPS)

    assert len(scheduler.timesteps) == 2
    assert torch.equal(scheduler.timesteps, timesteps[-2:])
