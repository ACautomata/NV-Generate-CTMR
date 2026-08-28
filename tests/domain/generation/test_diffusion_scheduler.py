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

"""DiffusionScheduler gates: per-tensor parity with MONAI's RFlowScheduler (ADR-0016 testing).

The domain scheduler owns one denoising trajectory (timestep sequence +
advance position) and delegates the step arithmetic to the MONAI scheduler it
wraps; every value it exposes must equal the raw RFlowScheduler sequence for
the same configuration -- noise injection, first/last timesteps, step
advancement and the completion boundary.  Torch-level: real execution on CPU.
"""

from __future__ import annotations

import pytest
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.domain.generation.scheduler import DiffusionScheduler

pytestmark = pytest.mark.torch

RF_CONFIG = {
    "num_train_timesteps": 1000,
    "use_discrete_timesteps": False,
    "use_timestep_transform": True,
    "sample_method": "uniform",
    "scale": 1.4,
}
LATENT_SHAPE = (1, 4, 8, 8, 4)
NUM_STEPS = 4
DIVISOR_NUMEL = int(torch.prod(torch.tensor(LATENT_SHAPE[2:])))


def _rflow():
    return RFlowScheduler(**RF_CONFIG)


def _trajectory():
    scheduler = DiffusionScheduler.begin(_rflow(), NUM_STEPS, LATENT_SHAPE)
    return scheduler, _rflow()


def test_begin_prepares_the_monai_timestep_sequence_verbatim():
    scheduler, reference = _trajectory()
    reference.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=DIVISOR_NUMEL)

    assert scheduler.timesteps.dtype == reference.timesteps.dtype
    assert torch.equal(scheduler.timesteps, reference.timesteps)
    # the trajectory the domain scheduler advances: t -> next_t per position
    expected_next = torch.cat((reference.timesteps[1:], torch.tensor([0], dtype=reference.timesteps.dtype)))
    assert torch.equal(scheduler.next_timesteps, expected_next)
    assert scheduler.position == 0
    assert not scheduler.complete


def test_noise_injection_and_timestep_draw_match_the_wrapped_scheduler():
    scheduler, reference = _trajectory()
    reference.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=DIVISOR_NUMEL)
    samples = torch.randn(LATENT_SHAPE)

    torch.manual_seed(21)
    drawn = scheduler.sample_timesteps(samples)
    torch.manual_seed(21)
    drawn_ref = reference.sample_timesteps(samples)
    assert torch.equal(drawn, drawn_ref)

    torch.manual_seed(22)
    noisy = scheduler.add_noise(original_samples=samples, noise=torch.randn_like(samples), timesteps=drawn)
    torch.manual_seed(22)
    noisy_ref = reference.add_noise(original_samples=samples, noise=torch.randn_like(samples), timesteps=drawn_ref)
    assert torch.equal(noisy, noisy_ref)


def test_each_step_advances_the_same_latent_as_the_monai_loop():
    scheduler, reference = _trajectory()
    reference.set_timesteps(num_inference_steps=NUM_STEPS, input_img_size_numel=DIVISOR_NUMEL)
    all_next = torch.cat((reference.timesteps[1:], torch.tensor([0], dtype=reference.timesteps.dtype)))

    latent = torch.randn(LATENT_SHAPE)
    reference_latent = latent.clone()
    for t, next_t in zip(reference.timesteps, all_next):
        model_output = torch.randn_like(reference_latent)
        reference_latent, _ = reference.step(model_output=model_output, timestep=t, sample=reference_latent, next_timestep=next_t)
        domain_latent = scheduler.step(model_output=model_output.detach().clone(), sample=latent)
        assert torch.equal(domain_latent, reference_latent)  # per-step parity
        latent = domain_latent

    assert latent.shape == LATENT_SHAPE
    assert scheduler.complete
    assert scheduler.position == NUM_STEPS


def test_step_after_completion_refuses_with_runtime_error():
    scheduler, _ = _trajectory()
    sample = torch.randn(LATENT_SHAPE)
    for _ in range(NUM_STEPS):
        scheduler.step(model_output=torch.randn_like(sample), sample=sample)
    assert scheduler.complete
    with pytest.raises(RuntimeError, match="completed"):
        scheduler.step(model_output=torch.randn_like(sample), sample=sample)


def test_each_sample_call_gets_a_fresh_trajectory():
    first = DiffusionScheduler.begin(_rflow(), NUM_STEPS, LATENT_SHAPE)
    for _ in range(2):
        first.step(model_output=torch.randn(LATENT_SHAPE), sample=torch.randn(LATENT_SHAPE))
    second = DiffusionScheduler.begin(_rflow(), NUM_STEPS, LATENT_SHAPE)

    assert second.position == 0  # no progress leaks across sample sessions
    assert not second.complete
    assert torch.equal(second.timesteps, first.timesteps)  # same config -> same sequence
