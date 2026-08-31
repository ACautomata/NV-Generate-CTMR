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

"""Composition-root runtime gates (issue #274 / ADR-0015 §6).

The ``GenerateRuntime`` assembly is the one home of the concrete collaborator
knowledge the cross-modal family entries stopped carrying (ADR-0019 §2), so it
is born with tests: each seam hands back the collaborator it names -- the
precision-strategy selection (fp16 scaler / bf16 / plain, the same branches the
train entries used to inline), the checkpoint file identity (collapsing onto the
domain ``WeightsRef`` addressing rule), the frozen engine adapter behind the
``GenerationEngine`` port and the logger behind the ``Logger`` port.
``train_session`` is the one seam without a local gate: it bootstraps the real
distributed session, a cluster-side topology behavior. Torch-level (the seams
resolve torch/monai adapters), so the module is torch-marked and runs for real
in the CI full-dependency tier (ADR-0015 §6); the structural light-import gate
stays in tests/test_wiring.py.
"""

from __future__ import annotations

import logging

import pytest

from ctmr.domain.engine import GenerationEngine
from ctmr.domain.identity import WeightsRef
from ctmr.domain.logging import Logger
from ctmr.infrastructure.bypass_mounting import BypassMounting  # tests are exempt (ADR-0019 §1)
from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor
from ctmr.wiring.generate import GenerateRuntime

pytestmark = pytest.mark.torch


def test_gradient_executor_selects_the_pinned_precision_strategy():
    runtime = GenerateRuntime()
    assert isinstance(runtime.gradient_executor(amp=True, amp_dtype="fp16"), Fp16GradientExecutor)
    assert isinstance(runtime.gradient_executor(amp=True, amp_dtype="bf16"), Bf16GradientExecutor)
    assert isinstance(runtime.gradient_executor(amp=False, amp_dtype="bf16"), PlainGradientExecutor)


def test_weights_ref_of_file_collapses_onto_the_domain_identity(tmp_path):
    payload = b"checkpoint-fixture-bytes"
    path = tmp_path / "weights.pt"
    path.write_bytes(payload)
    assert GenerateRuntime().weights_ref_of_file()(path) == WeightsRef.of_bytes(payload)


def test_engine_and_logger_satisfy_the_domain_ports():
    runtime = GenerateRuntime()
    assert isinstance(runtime.engine(), GenerationEngine)
    assert isinstance(runtime.logger("runtime-gate"), Logger)


def test_bypass_mounting_assembles_the_hook_up_collaborator():
    args = type("Args", (), {})()  # the mount constructor only holds references
    mounted = GenerateRuntime().bypass_mounting(args, device=None, logger=logging.getLogger("runtime-gate"))
    assert isinstance(mounted, BypassMounting)
