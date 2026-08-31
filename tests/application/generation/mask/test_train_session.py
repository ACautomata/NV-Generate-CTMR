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

"""Mask family assembly gates (issue #273 / ADR-0019 §2-§3).

The family entries consume only domain ports; the composition root
(``ctmr.wiring.generate``) assembles the concrete set: the engine port
adapter, the distributed session + logger, the gradient executor chosen by
the amp declaration, and the bypass mounting behind the ``BypassMounting``
port face. One gate scans the four family modules for infrastructure imports
(the family-level face of the ADR-0019 §1 direction rule -- the ratchet
entries for this family shrink to zero); the assembly behavior runs the real
frozen functions behind the injected session with only the distributed
bootstrap faked (no GPU on this tier).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctmr.domain.engine import GenerationEngine
from ctmr.wiring.generate import mask_engine, mask_train_session

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")

FAMILY_MODULES = Path(__file__).resolve().parents[4] / "src" / "ctmr" / "application" / "generation" / "mask"


def test_family_carries_no_infrastructure_import():
    """AC of issue #273: the four family entries import zero infrastructure --
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
    assert not violations, f"mask family imports infrastructure: {violations}"


def test_engine_assembly_satisfies_the_generation_engine_port():
    assert isinstance(mask_engine(), GenerationEngine)


def _fake_distributed_bootstrap(monkeypatch, order=None):
    """Fake the two session edges that need a GPU host: the distributed
    bootstrap (real one sets the cuda device unconditionally) and the logging
    setup (its RankFilter branch probes dist state). ``order`` records the
    bootstrap's firing, to pin the config-before-collective ordering."""
    from ctmr.infrastructure.maisi_engine import diff_model_setting

    def _initialize(num_gpus):
        if order is not None:
            order.append("dist")
        return 0, 1, CPU

    monkeypatch.setattr(diff_model_setting, "initialize_distributed", _initialize)
    monkeypatch.setattr(diff_model_setting, "setup_logging", lambda logger_name="": logging.getLogger("test-session"))


class _FakeSessionEngine:
    """Engine-port stand-in: canned config namespace, config call recorded."""

    def __init__(self, order=None):
        self._order = order

    def load_config(self, env_config_path, model_config_path, model_def_path):
        if self._order is not None:
            self._order.append("config")
        # the parsed namespace the real engine hands back carries the config
        # keys the assembled collaborators read (the mounting's payload face
        # reads noise_scheduler)
        return SimpleNamespace(env_config_path=env_config_path, noise_scheduler={"num_train_timesteps": 1000})


@pytest.mark.parametrize(
    ("amp", "amp_dtype", "executor_name"),
    [(True, "fp16", "Fp16GradientExecutor"), (True, "bf16", "Bf16GradientExecutor"), (False, "bf16", "PlainGradientExecutor")],
)
def test_train_session_selects_the_gradient_executor_by_the_amp_declaration(monkeypatch, amp, amp_dtype, executor_name):
    from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor

    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=amp, amp_dtype=amp_dtype, env_config_path="e.json", model_config_path="c.json", model_def_path="d.json")

    session = mask_train_session(args, engine=_FakeSessionEngine())

    expected = {
        "Fp16GradientExecutor": Fp16GradientExecutor,
        "Bf16GradientExecutor": Bf16GradientExecutor,
        "PlainGradientExecutor": PlainGradientExecutor,
    }
    assert isinstance(session.gradient_executor, expected[executor_name])
    assert callable(session.gradient_executor.run)  # the domain GradientExecutor port's face


def test_train_session_mounts_the_ports_and_resolves_config_before_the_collective(monkeypatch):
    order = []
    _fake_distributed_bootstrap(monkeypatch, order)
    args = SimpleNamespace(num_gpus=1, amp=True, amp_dtype="bf16", env_config_path="e.json", model_config_path="c.json", model_def_path="d.json")
    engine = _FakeSessionEngine(order)

    session = mask_train_session(args, engine=engine)

    assert session.engine is engine  # the injected engine is mounted verbatim
    assert session.local_rank == 0
    assert session.device == CPU
    assert session.logger.name == "test-session"  # the Logger port's injected sink
    assert session.merged.env_config_path == "e.json"  # the parsed config rides the session
    # config resolution strictly precedes the distributed bootstrap (the
    # pre-migration ordering -- the #272 review-locked sequence, family-shared)
    assert order == ["config", "dist"]


def test_train_session_bypass_mounting_is_the_injected_hookup(monkeypatch):
    """The mounting member stands behind the domain ``BypassMounting`` port:
    the real hook-up (DM-source load, copy_model_state init, DDP wrap) is the
    infrastructure gate's seam (tests/infrastructure/test_bypass_mounting.py);
    here the kernel-facing payload contract is pinned over the real sequence."""
    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=True, amp_dtype="bf16", env_config_path="e.json", model_config_path="c.json", model_def_path="d.json")

    session = mask_train_session(args, engine=_FakeSessionEngine())

    payload = session.mounting.checkpoint_payload(_BareModule(), epoch=7, avg_loss=0.5, scale=0.25)
    assert list(payload) == ["epoch", "loss", "num_train_timesteps", "scale_factor", "controlnet_state_dict"]
    # the mount recipe values are the kernel's to inject; the session mounts
    # the concrete sequence with the session config and device
    with pytest.raises(TypeError):
        session.mounting.mount(dataset_size=1)  # recipe values are keyword-only


class _BareModule:
    """A minimal DDP-unwrap target for the payload face (no DDP wrapper here)."""

    def state_dict(self):
        return {}
