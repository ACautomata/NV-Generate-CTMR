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

"""Modality-label family assembly gates (issue #272 / ADR-0019 §2-§3).

The family entries consume only domain ports; the composition root
(``ctmr.wiring.generate``) assembles the concrete set: the engine port
adapter, the distributed session + logger, the gradient executor chosen by
the amp declaration, and the MONAI-checkpoint archive behind the
``CheckpointRepository`` load face. One gate scans the three family modules
for infrastructure imports (the family-level face of the ADR-0019 §1
direction rule, ``-- the ratchet entries for this family shrink to zero);
the assembly behavior runs the real frozen functions behind the injected
session with only the distributed bootstrap faked (no GPU on this tier).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctmr.domain.engine import GenerationEngine
from ctmr.wiring.generate import MonaiCheckpointArchive, modality_label_engine, modality_label_train_session

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")

FAMILY_MODULES = (
    Path(__file__).resolve().parents[4] / "src" / "ctmr" / "application" / "generation" / "modality_label"
)


def test_family_carries_no_infrastructure_import():
    """AC of issue #272: the three family entries import zero infrastructure --
    every concrete edge they carried at guard birth is now a port injection."""
    violations = []
    for path in sorted(FAMILY_MODULES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            bases = []
            if isinstance(node, ast.Import):
                bases = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                bases = [node.module]
            for base in bases:
                if base.startswith("ctmr.infrastructure"):
                    violations.append(f"{path.name}: {base}")
    assert not violations, f"modality-label family imports infrastructure: {violations}"


def test_engine_assembly_satisfies_the_generation_engine_port():
    assert isinstance(modality_label_engine(), GenerationEngine)


def _fake_distributed_bootstrap(monkeypatch):
    """Fake the two session edges that need a GPU host: the distributed
    bootstrap (real one sets the cuda device unconditionally) and the logging
    setup (its RankFilter branch probes dist state)."""
    from ctmr.infrastructure.maisi_engine import diff_model_setting

    monkeypatch.setattr(diff_model_setting, "initialize_distributed", lambda num_gpus: (0, 1, CPU))
    monkeypatch.setattr(diff_model_setting, "setup_logging", lambda logger_name="": logging.getLogger("test-session"))


@pytest.mark.parametrize(
    ("amp", "amp_dtype", "executor_name"),
    [(True, "fp16", "Fp16GradientExecutor"), (True, "bf16", "Bf16GradientExecutor"), (False, "bf16", "PlainGradientExecutor")],
)
def test_train_session_selects_the_gradient_executor_by_the_amp_declaration(monkeypatch, amp, amp_dtype, executor_name):
    from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor

    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=amp, amp_dtype=amp_dtype)

    session = modality_label_train_session(args)

    expected = {"Fp16GradientExecutor": Fp16GradientExecutor, "Bf16GradientExecutor": Bf16GradientExecutor, "PlainGradientExecutor": PlainGradientExecutor}
    assert isinstance(session.gradient_executor, expected[executor_name])
    assert callable(session.gradient_executor.run)  # the domain GradientExecutor port's face


def test_train_session_mounts_the_engine_port_and_the_session_members(monkeypatch):
    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=True, amp_dtype="bf16")

    session = modality_label_train_session(args)

    assert isinstance(session.engine, GenerationEngine)  # the port, not a concrete family spelling
    assert session.local_rank == 0
    assert session.device == CPU
    assert session.logger.name == "test-session"  # the Logger port's injected sink


def test_train_session_base_checkpoint_archive_loads_published_payloads(tmp_path, monkeypatch):
    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=True, amp_dtype="bf16")
    payload = {"unet_state_dict": {"w": torch.eye(2)}, "scale_factor": 0.87}
    torch.save(payload, tmp_path / "base.pt")

    session = modality_label_train_session(args)
    loaded = session.base_checkpoints.load(tmp_path / "base.pt")

    assert torch.equal(loaded["unet_state_dict"]["w"], torch.eye(2))
    assert loaded["scale_factor"] == pytest.approx(0.87)
    # the archive stands behind the CheckpointRepository port's load face
    assert isinstance(MonaiCheckpointArchive(CPU).load(tmp_path / "base.pt"), dict)
