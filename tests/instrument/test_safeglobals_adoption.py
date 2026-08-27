"""Torch-level gate tests for the #108 weights_only whitelist adoption (ADR-0009).

Proves decision 4's collapse on the two remaining copy sites: importing
``scripts.nnunet_l2_instrument`` / ``scripts.nnunet_l2_closing_verification``
no longer mutates global torch state (import-time ``add_safe_globals`` gone),
their ``torch.load`` calls run inside the ``nnunet_safe_globals()`` scope, the
promoted ``l2_calibration_predict_entry.py`` is gone as a call site, and the
four surviving different-payload whitelists stay untouched (a fifth,
``prototype/p3_image_cond_controlnet/p3_common.py``, was removed with the
prototype tree in #145). Auto-skipped on light stacks
(``pytest.importorskip``, ADR-0013 §4); the AST half needs no torch but lives
here to keep the adoption gate in one file.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

pytestmark = pytest.mark.torch

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
REPO_ROOT = Path(__file__).resolve().parents[2]
ADOPTED_SCRIPTS = (SCRIPTS_DIR / "nnunet_l2_instrument.py", SCRIPTS_DIR / "nnunet_l2_closing_verification.py")
UNTOUCHED_WHITELIST_SITES = (
    "scripts/brats_p1_finetune.py",
    "src/ctmr/application/generation/mask/monitor.py",
    "src/ctmr/application/generation/mask/train.py",
    "src/ctmr/application/generation/trends.py",
)


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


def test_import_of_the_two_scripts_no_longer_mutates_torch_global_state():
    # the baseline is taken AFTER monai's own import chain (monai registers its own
    # MetaTensor/SpaceKeys allowlist on import -- third-party behaviour, not ours):
    # the gate is that OUR scripts add nothing on top of it.
    check = (
        "import torch; import monai.apps.nnunet; "
        "before = list(torch.serialization.get_safe_globals()); "
        "import scripts.nnunet_l2_instrument, scripts.nnunet_l2_closing_verification; "
        "assert torch.serialization.get_safe_globals() == before"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_import_time_allowlist_call_remains_in_the_two_scripts():
    for path in ADOPTED_SCRIPTS:
        tree = ast.parse(path.read_text())
        assert not [node for node in ast.walk(tree) if _is_add_safe_globals_call(node)], path


def test_torch_load_points_run_inside_the_scoped_allowlist():
    for path in ADOPTED_SCRIPTS:
        tree = ast.parse(path.read_text())
        scoped_lines = [(node.lineno, node.end_lineno) for node in ast.walk(tree) if _is_nnunet_safe_globals_with(node)]
        load_lines = [node.lineno for node in ast.walk(tree) if _is_torch_load_call(node)]
        assert load_lines, path  # both scripts really load with weights_only=True
        assert all(any(start <= line <= end for start, end in scoped_lines) for line in load_lines), path


def test_legacy_entry_script_is_superseded_by_the_canonical_entry():
    assert not (SCRIPTS_DIR / "l2_calibration_predict_entry.py").exists()
    # the promoted canonical entry carries the scoped activation (ADR-0009 decision 3)
    tree = ast.parse((REPO_ROOT / "src/ctmr/instrument/predict.py").read_text())
    assert [node for node in ast.walk(tree) if _is_nnunet_safe_globals_with(node)]


def test_unrelated_payload_whitelists_stay_untouched():
    for rel in UNTOUCHED_WHITELIST_SITES:
        assert "add_safe_globals" in (REPO_ROOT / rel).read_text(), rel
