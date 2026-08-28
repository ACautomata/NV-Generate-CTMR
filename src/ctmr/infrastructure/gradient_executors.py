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

"""Runtime gradient-execution strategies injected into the generation domain (ADR-0016, issue #170).

The torch.amp assembly (autocast dtype, GradScaler lifecycle) stays on the
infrastructure adapter side; ``DiffusionModel.train_step`` only sees the
``GradientExecutor`` protocol.  Each strategy reproduces one branch of the
PhaseHarness mechanical sequence exactly (zero_grad → autocast-wrapped loss →
backward → step), so migrating a stage to the domain entities changes no
training-step arithmetic.
"""

from __future__ import annotations

import torch
from torch.amp import GradScaler, autocast


class PlainGradientExecutor:
    """Non-AMP strategy: plain fp32 forward/backward, no autocast, no scaler."""

    def run(self, compute_loss, trainable, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        return loss


class Fp16GradientExecutor:
    """fp16 autocast + GradScaler; the scaler lives for the executor's lifetime.

    The scaler's growth state is per-run (one executor per training run), the
    same lifetime the PhaseHarness previously gave the ``GradScaler`` it passed
    between batches.
    """

    def __init__(self):
        self._scaler = GradScaler("cuda")

    def run(self, compute_loss, trainable, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=True):
            loss = compute_loss()
        self._scaler.scale(loss).backward()
        self._scaler.step(optimizer)
        self._scaler.update()
        return loss


class Bf16GradientExecutor:
    """bf16 autocast without scaler (bf16 dynamic range skips GradScaler, the DCU default)."""

    def run(self, compute_loss, trainable, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss = compute_loss()
        loss.backward()
        optimizer.step()
        return loss
