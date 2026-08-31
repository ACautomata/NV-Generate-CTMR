"""Torch-level gate tests for the weights_only whitelist adoption (ADR-0009, re-homed with #140; port migration with #275).

Proves decision 4's collapse end state, now in its layered form: importing the
judge-chain modules (``instrument_training`` / ``closing``) never mutates
global torch state and never touches ``torch.load`` at all -- the checkpoint
reads run behind the injected ``InstrumentCheckpointReader`` port, whose
adapter carries the ``nnunet_safe_globals()`` scope in
``ctmr.infrastructure.nnunet_runner`` (the legacy calibration entry and the
instrument reverse shim retired with the scripts layer / issue #175).
Torch-level tier: runs for real in the CI full-dependency set (ADR-0015 §6);
the AST half needs no torch but lives here to keep the gate in one file.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[4]
JUDGE_CHAIN_MODULES = (
    REPO_ROOT / "src/ctmr/application/acceptance/distribution/instrument_training.py",
    REPO_ROOT / "src/ctmr/application/acceptance/distribution/closing.py",
)
RUNNER = REPO_ROOT / "src/ctmr/infrastructure/nnunet_runner.py"


def _is_add_safe_globals_call(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_safe_globals"


def _is_torch_load_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _is_nnunet_safe_globals_with(node):
    return isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "nnunet_safe_globals"
        for item in node.items
    )


def test_import_of_the_judge_chain_modules_no_longer_mutates_torch_global_state():
    # the baseline is taken AFTER monai's own import chain (monai registers its own
    # MetaTensor/SpaceKeys allowlist on import -- third-party behaviour, not ours):
    # the gate is that OUR modules add nothing on top of it.
    check = (
        "import torch; import monai.apps.nnunet; "
        "before = list(torch.serialization.get_safe_globals()); "
        "import ctmr.application.acceptance.distribution.instrument_training; "
        "import ctmr.application.acceptance.distribution.closing; "
        "assert torch.serialization.get_safe_globals() == before"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_import_time_allowlist_call_remains_in_the_judge_chain():
    for path in JUDGE_CHAIN_MODULES:
        tree = ast.parse(path.read_text())
        assert not [node for node in ast.walk(tree) if _is_add_safe_globals_call(node)], path


def test_judge_chain_carries_no_torch_load_at_all():
    """The checkpoint reads moved behind the injected reader port (#275): the
    judge-chain modules own none themselves -- the scoped-allowlist load has
    exactly one home, the runner adapter."""
    for path in JUDGE_CHAIN_MODULES:
        tree = ast.parse(path.read_text())
        assert not [node for node in ast.walk(tree) if _is_torch_load_call(node)], path
    runner_tree = ast.parse(RUNNER.read_text())
    readers = [node for node in ast.walk(runner_tree) if _is_nnunet_safe_globals_with(node)]
    assert readers  # the scoped activation survives in the adapter


def test_legacy_entries_are_superseded_by_the_canonical_verb():
    assert not (REPO_ROOT / "scripts" / "l2_calibration_predict_entry.py").exists()
    # the instrument reverse shim package retired with issue #175 (ADR-0016 M5);
    # the canonical verb owns the scoped activation with no forwarding layer left
    assert not (REPO_ROOT / "src" / "ctmr" / "instrument").exists()


def test_instrument_run_reads_the_checkpoint_through_the_injected_reader(tmp_path, monkeypatch):
    """The instrument run's completion read is the injected reader port's to
    answer (#275): the completion record reflects the injected payload (the
    closing-side twin of this pin lives in test_closing)."""
    from ctmr.application.acceptance.distribution import instrument_training
    from ctmr.application.acceptance.distribution.instrument_training import (
        TRAINER_CLASS,
        ChallengeRegistry,
        InstrumentRun,
        RunConfiguration,
    )

    monkeypatch.setattr(instrument_training, "PERSISTENT_ROOT", tmp_path)
    results_root = tmp_path / "results"
    fold_dir = results_root / "Dataset503_BraTS2023MEN" / f"{TRAINER_CLASS}__nnUNetPlans__3d_fullres" / "fold_0"
    fold_dir.mkdir(parents=True)
    checkpoint = fold_dir / "checkpoint_final.pth"
    checkpoint.write_bytes(b"payload")  # hash target only; the reader is fake so the bytes are free
    (fold_dir / "training_log_0.txt").write_text("Epoch 249 done\n", encoding="utf-8")

    class _RecordingReader:
        def __init__(self):
            self.read_paths = []

        def read(self, path):
            self.read_paths.append(path)
            return {"current_epoch": 250, "trainer_name": TRAINER_CLASS}

    reader = _RecordingReader()
    under_root = tmp_path  # every root must resolve under the pinned persistent root
    configuration = RunConfiguration(
        challenge="MEN",
        raw_root=under_root,
        preprocessed_root=under_root,
        results_root=results_root,
        work_dir=under_root,
        audit_dir=under_root,
        gpu_ids=(0,),
        container_digest="sha256:test",
        repo_commit="0" * 40,
        monai_commit="0" * 40,
        nnunetv2_commit="0" * 40,
        nnunetv2_distribution_sha256="0" * 64,
        ssa_exception=False,
    )
    completion = InstrumentRun(configuration, ChallengeRegistry().get("MEN"), checkpoint_reader=reader)._completion()
    assert reader.read_paths == [checkpoint]
    assert completion["checkpoint_current_epoch"] == 250
    assert completion["checkpoint_trainer_name"] == TRAINER_CLASS
