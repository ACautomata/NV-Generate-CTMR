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

"""GradientExecutor runtime-update strategy gates (ADR-0016 generation domain).

The protocol is the seam application injects the fp16 / bf16 / non-AMP
execution strategy through; the concrete strategies live in
``ctmr.infrastructure.gradient_executors`` (torch-amp assembly stays on the
adapter side).  Torch-level: runs for real on CPU in the CI full-dependency
tier -- never skipped around the torch mark.
"""

from __future__ import annotations

import pytest
import torch

from ctmr.domain.generation.update import GradientExecutor
from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor

pytestmark = pytest.mark.torch


class _ToyModel(torch.nn.Module):
    """One writable weight giving a controlled, reproducible gradient flow."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, target):
        return self.weight - target


def test_plain_executor_drives_one_closed_update():
    model = _ToyModel()
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    executor = PlainGradientExecutor()

    def compute_loss():
        return (model(torch.tensor([1.0])) ** 2).mean()

    expected = (model(torch.tensor([1.0])) ** 2).mean().detach()
    returned = executor.run(compute_loss, model, optimizer)

    assert torch.equal(returned.detach(), expected)  # the loss of the *pre-update* forward
    assert model.weight.item() < 2.0  # gradient descent moved the weight
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


def test_plain_executor_zeroes_previous_gradients():
    model = _ToyModel()
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    executor = PlainGradientExecutor()
    executor.run(lambda: (model(torch.tensor([1.0])) ** 2).mean(), model, optimizer)
    first_grad = model.weight.grad.clone()
    executor.run(lambda: (model(torch.tensor([2.0])) ** 2).mean(), model, optimizer)
    # zero_grad(set_to_none=True): only the newest backward's gradient survives
    assert not torch.equal(first_grad, model.weight.grad)


def test_executor_protocol_is_structural():
    """The protocol is structural: any object providing ``run`` satisfies it (no runtime check)."""

    class _Duck:
        def run(self, compute_loss, trainable, optimizer):
            return compute_loss()

    model = _ToyModel()
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    returned = _Duck().run(lambda: model(torch.tensor([0.0])) ** 2, model, optimizer)
    assert torch.equal(returned.detach(), (model(torch.tensor([0.0])) ** 2).detach())
    # GradientExecutor is a Protocol: isinstance cannot be satisfied at runtime
    with pytest.raises(TypeError):
        isinstance(_Duck(), GradientExecutor)


@pytest.mark.parametrize("executor", [PlainGradientExecutor(), Fp16GradientExecutor(), Bf16GradientExecutor()])
def test_every_strategy_drives_one_closed_update(executor):
    """All three strategies execute the scaler/autocast call sequence for real.

    On a CPU-only host the CUDA scaler/autocast auto-disable (the documented
    no-CUDA fallback), but the scale/step/update and autocast call chain still
    runs -- a regressed sequence fails here before any DCU run burns compute.
    """
    model = _ToyModel()
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    expected = (model(torch.tensor([1.0])) ** 2).mean().detach()
    returned = executor.run(lambda: (model(torch.tensor([1.0])) ** 2).mean(), model, optimizer)
    assert torch.equal(returned.detach(), expected)
    assert model.weight.item() < 2.0
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
