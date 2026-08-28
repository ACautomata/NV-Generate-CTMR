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

"""DiffusionModel.train_step numeric parity with the migrated P1 path (ADR-0016 testing, issue #170).

Fixed seed, CPU fp32, toy production-topology network and synthetic batch: the
domain ``train_step`` (loss → backward → optimizer step → lr step) must
reproduce the legacy live-path math -- ``TrainKernel.train_batch`` forward +
the PhaseHarness plain update sequence -- including the perturbation, RF
noise/scheduler draw order and the Adam/PolynomialLR evolution.  Compared
values: per-step loss, model parameter state and optimizer state after two
closed updates.  The legacy reference drives the same vendored
``augment_modality_label`` the migrated entry used, so any drift in the domain
re-implementation surfaces as a mismatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import ModalityLabelPerturber
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.diff_model_train import augment_modality_label
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
SCALE_FACTOR = 0.87
SEED = 20260828
LR = 2e-06

TRAIN_RECIPE = {"lr": LR, "batch_size": 1, "n_epochs": 2}
TOTAL_STEPS = 10


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


def _seed_batch():
    torch.manual_seed(SEED)
    return (
        torch.randn(1, 4, 8, 8, 4),
        torch.tensor([[1.0, 1.2, 0.8]]),
        torch.tensor([29]),
    )


def _legacy_training_step(unet, scale_factor, rflow, images, spacing, modality, optimizer, lr_scheduler):
    """The migrated P1 live-path single step, verbatim math (ADR-0016 testing).

    ``TrainKernel.train_batch`` forward (scale → perturb → noise → RF timesteps
    → noisy latent → L1 against the RF velocity target) followed by the
    PhaseHarness non-AMP update sequence (zero_grad → backward → step) and the
    scheduler step the shell drove per batch.
    """
    scaled = images * scale_factor
    perturbed = augment_modality_label(modality.clone()).to(CPU)
    noise = torch.randn_like(scaled)
    timesteps = rflow.sample_timesteps(scaled)
    noisy_latent = rflow.add_noise(original_samples=scaled, noise=noise, timesteps=timesteps)
    model_output = unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing, class_labels=perturbed)
    loss = torch.nn.functional.l1_loss(model_output.float(), (scaled - noise).float())
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


def _fresh_adam(unet):
    return torch.optim.Adam(unet.parameters(), lr=LR)


def _fresh_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=TOTAL_STEPS, power=2.0)


def test_two_train_steps_match_legacy_loss_params_and_optimizer_state():
    images, spacing, modality = _seed_batch()

    # ---- shared identical start for both paths ----
    base_unet = _fresh_unet()
    initial_state = {key: value.clone() for key, value in base_unet.state_dict().items()}

    # ---- legacy path (migrated live math, plain update) ----
    # One seed per path, two consecutive steps: a single continuous RNG stream
    # (no per-step re-seeding), so a drifted random-consumption order between
    # the two paths surfaces at step 2 instead of being masked by re-seeds.
    legacy_unet = _fresh_unet()
    legacy_unet.load_state_dict(initial_state)
    legacy_optimizer = _fresh_adam(legacy_unet)
    legacy_lr = _fresh_lr_scheduler(legacy_optimizer)
    legacy_rflow = _rflow()
    torch.manual_seed(SEED)
    legacy_loss_1 = _legacy_training_step(legacy_unet, SCALE_FACTOR, legacy_rflow, images, spacing, modality.clone(), legacy_optimizer, legacy_lr)
    legacy_loss_2 = _legacy_training_step(legacy_unet, SCALE_FACTOR, legacy_rflow, images, spacing, modality.clone(), legacy_optimizer, legacy_lr)

    # ---- domain path (same start, same two closed updates) ----
    domain_unet = _fresh_unet()
    domain_unet.load_state_dict(initial_state)
    optimizer = _fresh_adam(domain_unet)
    lr_scheduler = _fresh_lr_scheduler(optimizer)
    model = DiffusionModel(
        unet=domain_unet,
        scale_factor=torch.tensor(SCALE_FACTOR),
        noise_scheduler=_rflow(),
        perturber=ModalityLabelPerturber(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    executor = PlainGradientExecutor()
    torch.manual_seed(SEED)
    domain_loss_1 = model.train_step(images, spacing, modality.clone(), executor)
    domain_loss_2 = model.train_step(images, spacing, modality.clone(), executor)

    assert torch.equal(legacy_loss_1, domain_loss_1), "step-1 loss drifted"
    assert torch.equal(legacy_loss_2, domain_loss_2), "step-2 loss drifted"

    _assert_state_dict_close(domain_unet.state_dict(), legacy_unet.state_dict())
    _assert_state_dict_close(optimizer.state_dict(), legacy_optimizer.state_dict())
    # the lr schedule evolved identically (PolynomialLR power 2.0 after 2 steps)
    assert optimizer.param_groups[0]["lr"] == legacy_optimizer.param_groups[0]["lr"]
    assert lr_scheduler.last_epoch == legacy_lr.last_epoch


def test_train_step_refuses_without_training_session_members():
    model = DiffusionModel(
        unet=_fresh_unet(),
        scale_factor=torch.tensor(SCALE_FACTOR),
        noise_scheduler=_rflow(),
        perturber=None,
        optimizer=None,
        lr_scheduler=None,
    )
    images, spacing, modality = _seed_batch()
    with pytest.raises(ValueError, match="training session"):
        model.train_step(images, spacing, modality, PlainGradientExecutor())


def test_train_step_moves_the_parameters_once_per_call():
    unet = _fresh_unet()
    optimizer = _fresh_adam(unet)
    lr_scheduler = _fresh_lr_scheduler(optimizer)
    model = DiffusionModel(
        unet=unet,
        scale_factor=torch.tensor(SCALE_FACTOR),
        noise_scheduler=_rflow(),
        perturber=ModalityLabelPerturber(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    before = {key: value.clone() for key, value in unet.state_dict().items()}
    images, spacing, modality = _seed_batch()
    torch.manual_seed(SEED)
    model.train_step(images, spacing, modality, PlainGradientExecutor())
    after = unet.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)
