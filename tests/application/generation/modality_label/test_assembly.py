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

"""Modality-label family assembly gates (issue #272 / ADR-0019 §2-§3; #276 terminal form).

The family entries consume only domain ports; the composition root
(``ctmr.wiring.generate``) assembles the concrete set: the engine port
adapter, the distributed session + logger, the gradient executor chosen by
the amp declaration, the MONAI-checkpoint archive behind the
``CheckpointRepository`` load face, and the shell-side publication store
behind the same port's save face (#276). One gate scans the three family
modules for infrastructure imports (the family-level face of the ADR-0019 §1
direction rule); the
assembly behavior runs the real frozen functions behind the injected session
with only the distributed bootstrap faked (no GPU on this tier).
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctmr.application.generation import trend as trend_mod
from ctmr.application.shell import PeriodicValidator, ValidationPhase
from ctmr.domain.checkpoints import CheckpointRepository
from ctmr.domain.engine import GenerationEngine
from ctmr.wiring.generate import MonaiCheckpointArchive, modality_label_engine, modality_label_train_session, modality_label_validation

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")

FAMILY_MODULES = Path(__file__).resolve().parents[4] / "src" / "ctmr" / "application" / "generation" / "modality_label"


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
        # the shell-side checkpoint repository reads model_dir off the parsed
        # namespace (a bare string -- the constructor never touches the path)
        return SimpleNamespace(env_config_path=env_config_path, model_dir="model-dir-under-test")


@pytest.mark.parametrize(
    ("amp", "amp_dtype", "executor_name"),
    [(True, "fp16", "Fp16GradientExecutor"), (True, "bf16", "Bf16GradientExecutor"), (False, "bf16", "PlainGradientExecutor")],
)
def test_train_session_selects_the_gradient_executor_by_the_amp_declaration(monkeypatch, amp, amp_dtype, executor_name):
    from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor

    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=amp, amp_dtype=amp_dtype, env_config_path="e.json", model_config_path="c.json", model_def_path="d.json")

    session = modality_label_train_session(args, engine=_FakeSessionEngine())

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

    session = modality_label_train_session(args, engine=engine)

    assert session.engine is engine  # the injected engine is mounted verbatim
    assert session.local_rank == 0
    assert session.device == CPU
    assert session.logger.name == "test-session"  # the Logger port's injected sink
    assert session.merged.env_config_path == "e.json"  # the parsed config rides the session
    # the shell-side weight store rides the session behind the domain port (#276)
    assert isinstance(session.checkpoint_repository, CheckpointRepository)
    # the malformed-config failure mode stays pre-collective: config resolution
    # strictly precedes the distributed bootstrap (the pre-migration ordering)
    assert order == ["config", "dist"]


def test_train_session_base_checkpoint_archive_loads_published_payloads(tmp_path, monkeypatch):
    _fake_distributed_bootstrap(monkeypatch)
    args = SimpleNamespace(num_gpus=1, amp=True, amp_dtype="bf16", env_config_path="e.json", model_config_path="c.json", model_def_path="d.json")
    payload = {"unet_state_dict": {"w": torch.eye(2)}, "scale_factor": 0.87}
    torch.save(payload, tmp_path / "base.pt")

    session = modality_label_train_session(args, engine=_FakeSessionEngine())
    loaded = session.base_checkpoints.load(tmp_path / "base.pt")

    assert torch.equal(loaded["unet_state_dict"]["w"], torch.eye(2))
    assert loaded["scale_factor"] == pytest.approx(0.87)
    # the archive stands behind the CheckpointRepository port's load face
    assert isinstance(MonaiCheckpointArchive(CPU).load(tmp_path / "base.pt"), dict)


# ----------------------------------------- embedded periodic validation assembly (#278)


class _FakeBank:
    """Stands in for the real-reference bank: no GPU, no raw volumes."""

    builds = []

    def __init__(self, dev_list_path, raw_root, features, out_dir):
        self._out_dir = Path(out_dir)

    def build(self):
        _FakeBank.builds.append(str(self._out_dir))
        return {"t1n": {"xy": [[0.0]]}}


def _validation_args(tmp_path):
    dev_list = tmp_path / "dev_list.json"
    cases = []
    for challenge, quota in (("GLI", 4), ("SSA", 2), ("MEN", 4), ("METS", 3), ("PED", 3)):
        cases += [
            {"sub": challenge, "case": f"{challenge}-{index:03d}", "image": f"{challenge}-{index:03d}.nii.gz", "modality": "mri_t1_skull_stripped"}
            for index in range(quota)
        ]
    dev_list.write_text(json.dumps({"training": cases}) + "\n")
    return SimpleNamespace(
        val_every=10,
        dev_list=str(dev_list),
        raw_root=str(tmp_path / "raw"),
        emb_root=str(tmp_path / "emb"),
    )


def _validation_session(tmp_path):
    return SimpleNamespace(local_rank=0, device=CPU, engine=_FakeSessionEngine(), logger=logging.getLogger("test-session"))


def _validation_merged(tmp_path):
    return SimpleNamespace(model_dir=str(tmp_path / "models"), diffusion_unet_train={"n_epochs": 100})


def test_modality_label_validation_assembles_the_periodic_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(trend_mod, "RealReferenceBank", _FakeBank)
    _FakeBank.builds.clear()
    args = _validation_args(tmp_path)
    merged = _validation_merged(tmp_path)

    phase = modality_label_validation(args, merged, _validation_session(tmp_path), kernel=None)

    assert isinstance(phase, ValidationPhase)
    assert phase.every == 10
    # the 16-case cohort unfolded to 64 (case, modality) shard items
    assert isinstance(phase.validator, PeriodicValidator)
    assert len(phase.validator._items) == 64
    assert {tuple(item) for item in ((entry["sub"], entry["case"], entry["modality"]) for entry in phase.validator._items)} == {
        (entry["sub"], entry["case"], modality)
        for entry in json.loads(Path(args.dev_list).read_text())["training"]
        for modality in ("t1n", "t1c", "t2w", "t2f")
    }
    assert phase.validator.cohort_file == str(Path(merged.model_dir) / "dev_eval" / "dev_cohort.json")
    # the pre-registered rule rides in verbatim (patience 3, min epoch 30, cap = trainer n_epochs)
    assert (phase.rule.patience, phase.rule.min_epoch, phase.rule.max_epoch, phase.rule.direction) == (3, 30, 100, "min")
    # rank 0 wrote the cohort + rule contracts under the ledger root
    eval_root = Path(merged.model_dir) / "dev_eval"
    cohort_doc = json.loads((eval_root / "dev_cohort.json").read_text())
    assert len(cohort_doc["cohort"]) == 16 and "quotas" in cohort_doc
    rule_doc = json.loads((eval_root / "early_stop_rule.json").read_text())
    assert rule_doc["patience"] == 3 and rule_doc["min_epoch"] == 30 and rule_doc["max_epoch"] == 100
    assert _FakeBank.builds == [str(eval_root / "reference")]  # the bank is pre-built once, at assembly


def test_modality_label_validation_refuses_missing_dev_inputs(tmp_path):
    merged = _validation_merged(tmp_path)
    args = _validation_args(tmp_path)
    args.dev_list = None
    with pytest.raises(ValueError, match="--dev-list"):
        modality_label_validation(args, merged, _validation_session(tmp_path), kernel=None)
    args = _validation_args(tmp_path)
    args.raw_root = None
    with pytest.raises(ValueError, match="--raw-root"):
        modality_label_validation(args, merged, _validation_session(tmp_path), kernel=None)
    args = _validation_args(tmp_path)
    args.emb_root = None
    with pytest.raises(ValueError, match="--emb-root"):
        modality_label_validation(args, merged, _validation_session(tmp_path), kernel=None)
