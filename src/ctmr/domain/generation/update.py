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

"""GradientExecutor: the runtime update strategy seam (ADR-0016, issue #170).

``DiffusionModel.train_step`` drives one complete parameter update (forward
loss → backward → optimizer step) but stays ignorant of the hardware precision
policy: the application injects a ``GradientExecutor`` carrying the fp16
(scaler), bf16 or non-AMP execution strategy.  Protocol only -- the concrete
strategies live in ``ctmr.infrastructure.gradient_executors`` (torch.amp
assembly stays on the adapter side).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class GradientExecutor(Protocol):
    """Execution strategy for one gradient update, injected by application.

    ``run`` executes: zero_grad → compute the loss inside the strategy's
    precision context → backward (with the fp16 gradient scaler when the
    strategy so decides) → optimizer step, and returns the computed loss so
    the caller can aggregate it without recomputation.
    """

    def run(self, compute_loss: Callable[[], torch.Tensor], trainable: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Drive one closed update; ``compute_loss`` runs inside the strategy's autocast context."""
        ...
