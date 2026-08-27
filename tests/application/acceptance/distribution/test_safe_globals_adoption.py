"""Torch-level gate tests for the weights_only whitelist adoption (ADR-0009, re-homed with #140).

Proves decision 4's collapse end state: importing the judge-chain modules that
``torch.load`` checkpoints (``instrument_training`` / ``closing``, now in
``ctmr.application.acceptance.distribution``) never mutates global torch state,
their ``torch.load`` calls run inside the ``nnunet_safe_globals()`` scope, the
promoted canonical verb lives in ``ctmr.infrastructure.nnunet_runner`` (the
legacy ``l2_calibration_predict_entry.py`` and ``python -m ctmr.instrument.predict``
stay superseded), and the surviving different-payload whitelists stay untouched
(the modality-label family, the mask family and the shared trend machinery were
relocated to allowlist-at-the-load-point form in tickets 10 and 09). Torch-level tier: runs for real in the CI
full-dependency set (ADR-0015 §6); the AST half needs no torch but lives here
to keep the gate in one file.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[4]
ADOPTED_MODULES = (
    REPO_ROOT / "src/ctmr/application/acceptance/distribution/instrument_training.py",
    REPO_ROOT / "src/ctmr/application/acceptance/distribution/closing.py",
)
UNTOUCHED_WHITELIST_SITES = ()  # end state: the mask family allowlists at the load point too (ticket 09)


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
    for path in ADOPTED_MODULES:
        tree = ast.parse(path.read_text())
        assert not [node for node in ast.walk(tree) if _is_add_safe_globals_call(node)], path


def test_torch_load_points_run_inside_the_scoped_allowlist():
    for path in ADOPTED_MODULES:
        tree = ast.parse(path.read_text())
        scoped_lines = [(node.lineno, node.end_lineno) for node in ast.walk(tree) if _is_nnunet_safe_globals_with(node)]
        load_lines = [node.lineno for node in ast.walk(tree) if _is_torch_load_call(node)]
        assert load_lines, path  # both modules really load with weights_only=True
        assert all(any(start <= line <= end for start, end in scoped_lines) for line in load_lines), path


def test_legacy_entries_are_superseded_by_the_canonical_verb():
    assert not (REPO_ROOT / "scripts" / "l2_calibration_predict_entry.py").exists()
    shim_source = (REPO_ROOT / "src/ctmr/instrument/predict.py").read_text()  # reverse shim: no scope body of its own
    assert "ctmr.infrastructure.nnunet_runner" in shim_source  # forwards, adds nothing
    runner_tree = ast.parse((REPO_ROOT / "src/ctmr/infrastructure/nnunet_runner.py").read_text())
    assert [node for node in ast.walk(runner_tree) if _is_nnunet_safe_globals_with(node)]  # the scoped activation


def test_unrelated_payload_whitelists_stay_untouched():
    for rel in UNTOUCHED_WHITELIST_SITES:
        assert "add_safe_globals" in (REPO_ROOT / rel).read_text(), rel
