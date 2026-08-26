"""nnunetv2-level gate tests for the canonical entry point (ADR-0009, #107).

Proves ``ctmr.instrument.predict`` runs the native nnUNetv2 entry inside the
``nnunet_safe_globals()`` scope and that ``python -m ctmr.instrument.predict``
is a working Python/shell entry. Auto-skipped when nnunetv2 / torch are absent
(``pytest.importorskip``, ADR-0013 §4); no cluster, no external data.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("nnunetv2")
pytest.importorskip("torch")
pytest.importorskip("numpy")

import numpy  # noqa: E402  (importorskip must precede the dependent imports)
import torch  # noqa: E402

import ctmr.instrument.predict  # noqa: E402

pytestmark = pytest.mark.torch


def test_main_runs_the_native_entry_inside_the_safe_globals_scope(monkeypatch):
    observed = {}

    def fake_entry_point():
        observed["active_globals"] = list(torch.serialization.get_safe_globals())
        return 0

    monkeypatch.setattr(ctmr.instrument.predict, "predict_entry_point", fake_entry_point)
    assert ctmr.instrument.predict.main() == 0
    assert numpy.core.multiarray.scalar in observed["active_globals"]  # the native entry ran inside the scope


def test_entry_point_is_python_m_runnable():
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run([sys.executable, "-m", "ctmr.instrument.predict", "--help"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage" in (result.stdout + result.stderr).lower()
