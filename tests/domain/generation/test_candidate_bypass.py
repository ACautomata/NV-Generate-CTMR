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

"""Candidate-bypass gates: P3 image-conditioned numeric parity (ADR-0016, issue #174).

Fixed seed, CPU fp32, toy production-topology networks (the P3 ControlNet
takes the 4ch src latent -- ``conditioning_embedding_in_channels=4``, never
the P2 8ch mask) and synthetic batches:

- the domain ``train_step`` (image-conditioned forward, weighted L1, one
  closed update) must reproduce the legacy cross-modal kernel math verbatim --
  scale both latents -> RF timesteps/noise -> scaled-src-latent condition ->
  ControlNet/UNet forward -> weighted velocity L1 -> plain update -> lr step --
  compared over two consecutive steps (single continuous RNG stream): per-step
  loss, ControlNet parameter state and optimizer state;
- the domain ``denoise_conditioned`` trajectory must reproduce the
  retired ControlNet-conditioned core's loop math (git history; deleted with
  issue #175) for the candidate's CFG=0 recipe (single conditioned forward, no unconditional branch) --
  every intermediate latent.

The candidate's CFG=0 constraint itself is pinned one level up
(``CandidateInferenceConfig`` rejects any other value); this gate locks the
numbers the pinned recipe produces.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import TumourWeightedTarget
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
SCALE_FACTOR = 0.87
SEED = 174
LR = 1e-5
TOTAL_STEPS = 10
NUM_STEPS = 4
LATENT_SHAPE = (1, 4, 8, 8, 4)
SPACING = torch.tensor([[1.0, 1.2, 0.8]], device=CPU)
MODALITY = torch.tensor([29], device=CPU)


def _toy_unet_def():
    """The production config_network_p3.json topology at toy width."""
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
        "num_class_embeds": 128,
        "resblock_updown": True,
        "include_fc": True,
    }


def _toy_controlnet_def():
    """The production config_network_p3.json controlnet_def topology at toy width.

    The P3 condition is the 4ch src latent: ``conditioning_embedding_in_channels``
    is 4 -- the image-conditioned bypass never carries the P2 8ch mask face.
    """
    return {
        "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
        "spatial_dims": 3,
        "in_channels": 4,
        "num_channels": [32, 64],
        "attention_levels": [False, False],
        "num_head_channels": [0, 32],
        "num_res_blocks": 1,
        "use_flash_attention": False,
        "conditioning_embedding_in_channels": 4,
        "conditioning_embedding_num_channels": [8],
        "num_class_embeds": 128,
        "resblock_updown": True,
        "include_fc": True,
    }


def _rflow():
    return RFlowScheduler(num_train_timesteps=1000, use_discrete_timesteps=False, use_timestep_transform=True, sample_method="uniform", scale=1.4)


def _fresh_unet():
    return define_instance(SimpleNamespace(diffusion_unet_def=_toy_unet_def()), "diffusion_unet_def")


def _fresh_controlnet():
    return define_instance(SimpleNamespace(controlnet_def=_toy_controlnet_def()), "controlnet_def")


def _seed_batch():
    """One synthetic batch: target latent, 4ch src latent, tumour label."""
    torch.manual_seed(SEED)
    images = torch.randn(1, 4, 8, 8, 4)
    src_latent = torch.randn(1, 4, 8, 8, 4)
    labels = torch.zeros(1, 1, 32, 32, 16, dtype=torch.long)
    labels[0, 0, 4, 4, 4] = 129
    labels[0, 0, 20, 24, 8] = 131
    return images, src_latent, labels


# --------------------------------------------------------------------- training parity


def _legacy_training_step(unet, controlnet, scale_factor, rflow, images, src_latent, spacing, modality, labels, weights, optimizer, lr_scheduler):
    """The migrated cross-modal ``TrainKernel.train_batch`` live math, verbatim (ADR-0016 testing).

    Scale both latents -> RF timesteps/noise -> the scaled 4ch src latent as
    the ControlNet condition -> residuals forward -> weighted velocity L1,
    followed by the PhaseHarness non-AMP update sequence and the lr step the
    shell drove per batch.
    """
    scaled = images * scale_factor
    condition = src_latent * scale_factor
    noise = torch.randn_like(scaled)
    timesteps = rflow.sample_timesteps(scaled)
    noisy_latent = rflow.add_noise(original_samples=scaled, noise=noise, timesteps=timesteps)
    down, mid = controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=condition, class_labels=modality)
    model_output = unet(
        x=noisy_latent,
        timesteps=timesteps,
        spacing_tensor=spacing,
        down_block_additional_residuals=down,
        mid_block_additional_residual=mid,
        class_labels=modality,
    )
    model_gt = scaled - noise
    loss = (torch.nn.functional.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    return loss


def _assert_state_dict_close(actual, expected, rtol=1e-6, atol=1e-8):
    """Recursive tensor-level comparison over optimizer/param state dicts."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_state_dict_close(actual[key], expected[key], rtol, atol)
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_state_dict_close(left, right, rtol, atol)
    elif isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape
        assert torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol), f"{actual} vs {expected}"
    else:
        assert actual == expected, f"{actual} vs {expected}"


def test_two_image_conditioned_train_steps_match_legacy_loss_params_and_optimizer_state():
    images, src_latent, labels = _seed_batch()
    # the application-side members the migrated kernel drives: the scaled src
    # latent condition and the pinned tumour-weighted target (weights=1, x100 ROI)
    condition = src_latent * SCALE_FACTOR
    weights = TumourWeightedTarget(100, [129, 130, 131]).weights(labels, images)

    # ---- shared identical start for both paths ----
    base_controlnet = _fresh_controlnet()
    initial_state = {key: value.clone() for key, value in base_controlnet.state_dict().items()}

    # ---- legacy path (migrated live math, plain update) ----
    # One seed per path, two consecutive steps: a single continuous RNG stream
    # (no per-step re-seeding), so a drifted random-consumption order between
    # the two paths surfaces at step 2 instead of being masked by re-seeds.
    legacy_controlnet = _fresh_controlnet()
    legacy_controlnet.load_state_dict(initial_state)
    legacy_unet = _fresh_unet()
    for parameter in legacy_unet.parameters():
        parameter.requires_grad = False
    legacy_optimizer = torch.optim.AdamW(legacy_controlnet.parameters(), lr=LR)
    legacy_lr = torch.optim.lr_scheduler.PolynomialLR(legacy_optimizer, total_iters=TOTAL_STEPS, power=2.0)
    legacy_rflow = _rflow()
    torch.manual_seed(SEED)
    legacy_loss_1 = _legacy_training_step(
        legacy_unet,
        legacy_controlnet,
        SCALE_FACTOR,
        legacy_rflow,
        images,
        src_latent,
        SPACING,
        MODALITY,
        labels,
        weights,
        legacy_optimizer,
        legacy_lr,
    )
    legacy_loss_2 = _legacy_training_step(
        legacy_unet,
        legacy_controlnet,
        SCALE_FACTOR,
        legacy_rflow,
        images,
        src_latent,
        SPACING,
        MODALITY,
        labels,
        weights,
        legacy_optimizer,
        legacy_lr,
    )

    # ---- domain path (same start, same two closed updates) ----
    controlnet = _fresh_controlnet()
    controlnet.load_state_dict(initial_state)
    unet = _fresh_unet()
    for parameter in unet.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=TOTAL_STEPS, power=2.0)
    model = DiffusionModel(
        unet=unet,
        scale_factor=torch.tensor(SCALE_FACTOR),
        noise_scheduler=_rflow(),
        bypass=ControlNetBypass(controlnet),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    executor = PlainGradientExecutor()
    torch.manual_seed(SEED)
    domain_loss_1 = model.train_step(images, SPACING, MODALITY, executor, controlnet_cond=condition, target_weights=weights)
    domain_loss_2 = model.train_step(images, SPACING, MODALITY, executor, controlnet_cond=condition, target_weights=weights)

    assert torch.equal(legacy_loss_1, domain_loss_1), "step-1 loss drifted"
    assert torch.equal(legacy_loss_2, domain_loss_2), "step-2 loss drifted"

    _assert_state_dict_close(controlnet.state_dict(), legacy_controlnet.state_dict())
    _assert_state_dict_close(optimizer.state_dict(), legacy_optimizer.state_dict())
    # the lr schedule evolved identically (PolynomialLR power 2.0 after 2 steps)
    assert optimizer.param_groups[0]["lr"] == legacy_optimizer.param_groups[0]["lr"]
    # the bypass is the only trainable: the frozen P1-DM never moved
    assert not any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in unet.parameters())


# --------------------------------------------------------------------- sampling parity


def _legacy_denoise_loop(unet, controlnet, rflow, initial_latent, spacing, modality, cond):
    """The retired ControlNet-conditioned core's CFG=0 loop (git history; deleted with issue #175), verbatim math.

    The candidate recipe is CFG OFF: a single conditioned ControlNet forward
    (``class_labels`` = the target modality) composes into a single UNet
    forward -- no unconditional branch, no guidance composition.  Every
    returned latent is recorded per step (the trajectory is the parity
    target, not only the final image); the AE decode tail stays out -- it is
    an application adapter below the latent the entity produces.
    """
    rflow.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=torch.prod(torch.tensor(initial_latent.shape[2:])))
    all_timesteps = rflow.timesteps
    all_next = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    trajectory = [initial_latent.clone()]
    latent = initial_latent
    for t, next_t in zip(all_timesteps, all_next):
        down, mid = controlnet(x=latent, timesteps=torch.Tensor((t,)).to(latent.device), controlnet_cond=cond, class_labels=modality)
        model_output = unet(
            x=latent,
            timesteps=torch.Tensor((t,)).to(latent.device),
            spacing_tensor=spacing,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
            class_labels=modality,
        )
        latent, _ = rflow.step(model_output, t, latent, next_t)
        trajectory.append(latent.clone())
    return trajectory


def test_candidate_denoise_trajectory_matches_the_legacy_controlnet_loop_at_cfg0():
    _images, src_latent, _labels = _seed_batch()
    cond = (src_latent * SCALE_FACTOR).half().float()  # the migrated condition prep (half on device, fp32 here)
    unet = _fresh_unet()
    controlnet = _fresh_controlnet()
    for parameter in unet.parameters():
        parameter.requires_grad = False

    torch.manual_seed(SEED + 1)
    initial_latent = torch.randn(LATENT_SHAPE, device=CPU)  # the fixed initial noise

    legacy = _legacy_denoise_loop(unet, controlnet, _rflow(), initial_latent, SPACING, MODALITY, cond)

    model = DiffusionModel(
        unet=unet,
        scale_factor=torch.tensor(SCALE_FACTOR),
        noise_scheduler=_rflow(),
        bypass=ControlNetBypass(controlnet),
    )
    scheduler = model.begin_sampling(initial_latent.shape, NUM_STEPS)
    domain_trajectory = [initial_latent.clone()]
    latent = initial_latent
    while not scheduler.complete:
        latent = model.denoise_conditioned(scheduler, latent, SPACING, MODALITY, cond, None, 0.0)
        domain_trajectory.append(latent.clone())

    assert len(domain_trajectory) == len(legacy)
    for index, (domain_step, legacy_step) in enumerate(zip(domain_trajectory, legacy)):
        assert torch.equal(domain_step, legacy_step), f"cfg=0: latent drifted at step {index}"
