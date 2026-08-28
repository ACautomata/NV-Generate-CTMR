"""Gate tests for the frozen-instrument execution side (ADR-0009 decisions 3+4, #140).

Proves ``ctmr measure predict`` runs the native nnUNetv2 entry inside the
``nnunet_safe_globals()`` scope with the caller's flags passed through (the
canonical spelling since the reverse shim retired with issue #175), and that
both the ``ctmr.measure predict`` route and
the direct module form stay working Python/shell entries. Torch-level tier:
nnunetv2 / torch / numpy are part of the CI full-dependency set and these tests
run for real (ADR-0015 §6); no cluster, no external data.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest
import torch

import ctmr.infrastructure.nnunet_runner as runner
from ctmr.cli import CtmrCli

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_verb_runs_the_native_entry_inside_the_safe_globals_scope(monkeypatch):
    observed = {}

    def fake_entry_point():
        observed["active_globals"] = list(torch.serialization.get_safe_globals())
        observed["argv"] = list(sys.argv)
        return 0

    monkeypatch.setattr("nnunetv2.inference.predict_from_raw_data.predict_entry_point", fake_entry_point)
    assert runner.MeasurePredictVerb().run(["-i", "/raw/in", "-o", "/pred/out"]) == 0
    assert numpy.core.multiarray.scalar in observed["active_globals"]  # the native entry ran inside the scope
    assert observed["argv"] == [sys.argv[0], "-i", "/raw/in", "-o", "/pred/out"]  # tail-only argv for the native parser


def test_module_form_still_runs_the_same_path(monkeypatch):
    seen = {}

    def fake_entry_point():
        seen["argv"] = list(sys.argv)
        return 7

    monkeypatch.setattr("nnunetv2.inference.predict_from_raw_data.predict_entry_point", fake_entry_point)
    assert runner.main(["--help"]) == 7
    assert seen["argv"][-1] == "--help"


def test_scope_covers_the_whole_native_call(monkeypatch):
    """The allowlist must be active when the native entry body executes, not only at dispatch."""
    observed = {}

    def fake_entry_point():
        observed["in_scope"] = numpy.core.multiarray.scalar in torch.serialization.get_safe_globals()
        return 0

    torch.serialization.clear_safe_globals()  # no residue from other tests: the scope itself must introduce the payload
    monkeypatch.setattr("nnunetv2.inference.predict_from_raw_data.predict_entry_point", fake_entry_point)
    runner.MeasurePredictVerb().run([])
    assert observed["in_scope"]


def test_cli_route_dispatches_lazily_to_the_runner(monkeypatch):
    dispatched = {}

    class FakeVerb:
        def run(self, pass_through):
            dispatched["pass_through"] = pass_through
            return 5

    fake_module = type(sys)("fake-runner")
    fake_module.MeasurePredictVerb = FakeVerb
    monkeypatch.setitem(sys.modules, "ctmr.infrastructure.nnunet_runner", fake_module)
    code = CtmrCli().run(["measure", "predict", "-i", "/raw/in", "-o", "/out"])
    assert code == 5
    assert dispatched["pass_through"] == ["-i", "/raw/in", "-o", "/out"]


def test_cli_measure_predict_is_python_dash_m_runnable():
    """The frozen argv runs the package itself (``-m ctmr``), not just the cli module."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "ctmr", "measure", "predict", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    output = result.stdout + result.stderr
    assert result.returncode in (0, 2)  # --help exits via argparse; nnUNetv2's own parser may also answer
    assert "predict" in output.lower()
