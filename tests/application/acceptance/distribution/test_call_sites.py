"""Convergence-gate tests for the instrument call-site adoptions (ADR-0009; re-homed with #140).

The spec gate pins ``build`` output vs the canonical snapshot
(tests/domain/test_instrument_spec); this file proves the judge-chain *call
sites* land on that single construction point: each adopted Python call site
produces precisely the ``FrozenInstrumentCommand.build`` argv (decision 5), and
shell orchestration shares the canonical ``ctmr measure predict`` entry with the
Python callers (decision 3). The fatal ``--disable_tta False`` token and both
non-standard entry names stay gone repo-wide. The sugon self-contained copy's
gate died with its retirement (#140): its recipe lives in deploy/jobs.
Light stack, any machine, no cluster, no external data.
"""

import re
from pathlib import Path

import pytest

from ctmr.application.acceptance.distribution import final_acceptance
from ctmr.application.acceptance.distribution.final_acceptance import PredictScriptWriter
from ctmr.application.acceptance.distribution.synthetic_domain import InstrumentRunner
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand

FIVE_CHALLENGES = sorted(INSTRUMENT_SPECS)
REPO_ROOT = Path(__file__).resolve().parents[4]
JOBS_DIR = REPO_ROOT / "deploy" / "jobs"


# ── final_acceptance.PredictScriptWriter (frozen terminal acceptance) ────────────────


def test_predict_script_writer_emits_the_builder_argv(tmp_path):
    challenges = {challenge: {} for challenge in FIVE_CHALLENGES}  # the writer only reads the keys
    PredictScriptWriter({"challenges": challenges}, tmp_path).write()

    for challenge in FIVE_CHALLENGES:
        cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(tmp_path / "inputs" / challenge, tmp_path / "predictions" / challenge)
        line = (tmp_path / f"predict_{challenge}.sh").read_text().splitlines()[-1]
        assert line == " ".join(cmd)


def test_predict_script_writer_bootstraps_the_src_tree_onto_pythonpath(tmp_path):
    """The generated scripts run the canonical entry ``python -m ctmr measure predict``
    in a fresh shell; the writer pins this checkout's src tree onto PYTHONPATH so the
    verb stays importable on the machine that generated them."""
    PredictScriptWriter({"challenges": {"GLI": {}}}, tmp_path).write()
    lines = (tmp_path / "predict_GLI.sh").read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1] == "set -euo pipefail"
    assert lines[2].startswith("export PYTHONPATH=")
    package_src = Path(final_acceptance.__file__).resolve().parents[4]
    assert str(package_src) in lines[2]
    assert "-m ctmr measure predict" in lines[-1]


def test_predict_script_writer_runner_executes_every_challenge_script(tmp_path):
    challenges = {challenge: {} for challenge in ("GLI", "SSA")}  # a subset proves ordering
    runner = PredictScriptWriter({"challenges": challenges}, tmp_path).write()

    assert runner == tmp_path / "predict_all.sh"
    lines = (tmp_path / "predict_all.sh").read_text().splitlines()
    assert lines[:2] == ["#!/bin/bash", "set -euo pipefail"]
    assert lines[2:] == ["bash predict_GLI.sh", "bash predict_SSA.sh"]


# ── synthetic_domain.InstrumentRunner (non-frozen #38 family) ───────────────────────


@pytest.mark.parametrize("challenge", FIVE_CHALLENGES)
def test_synthetic_domain_eval_predict_script_is_the_builder_argv(tmp_path, challenge):
    input_dir = tmp_path / "inputs"
    pred_base = tmp_path / "predictions"
    InstrumentRunner(tmp_path).predict_challenge(challenge, input_dir / challenge, pred_base)

    script = (pred_base / f"predict_{challenge}.sh").read_text()  # script next to the per-challenge pred dir (pre-adoption convention)
    cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(input_dir / challenge, pred_base / challenge)
    lines = script.splitlines()
    assert lines[0:2] == ["#!/bin/bash", "set -euo pipefail"]
    assert lines[2].startswith("export PYTHONPATH=")  # the canonical entry is importable from the fresh shell (ADR-0009 decision 6)
    assert lines[-1].startswith(" ".join(cmd))
    # the fatal tokens are gone: --disable_tta (any value) and --verbose <path>
    # (store_true as well -- the pre-adoption form died the same numpy-argparse way)
    assert "--disable_tta" not in script
    assert "--verbose" not in script


# ── shell orchestration (shared canonical entry, decision 3) ────────────────────────

SHELL_SCRIPTS = ["run_l2_synth_domain_eval.sh", "p1_predict_all.sh", "l2_calibration_predict.sh"]

_DECLARE_MAP_RE = re.compile(r"declare -A (\w+)=\(([^)]*)\)")
_MAP_ENTRY_RE = re.compile(r"\[(\w+)\]=(\S+)")


def _declared_maps(text):
    maps = {}
    for name, body in _DECLARE_MAP_RE.findall(text):
        maps[name] = dict(_MAP_ENTRY_RE.findall(body))
    return maps


def test_shell_scripts_declare_the_canonical_per_challenge_spec():
    """Every orchestration script's per-challenge dataset/plans/config maps match
    ``INSTRUMENT_SPECS`` verbatim -- the shell consumers carry the single spec, not
    a second copy of it."""
    runners = {
        "run_l2_synth_domain_eval.sh": {"DATASET_NAME": "dataset_id", "PLANS": "plans", "CONFIG": "config"},
        "l2_calibration_predict.sh": {"DATASET": "dataset_id", "PLANS": "plans", "CONFIG": "config"},
    }
    for name, mappings in runners.items():
        maps = _declared_maps((JOBS_DIR / name).read_text())
        for map_name, field in mappings.items():
            for challenge in FIVE_CHALLENGES:
                assert maps[map_name][challenge] == getattr(INSTRUMENT_SPECS[challenge], field), (name, map_name, challenge)


def test_p1_predict_all_sh_runs_the_canonical_command_per_challenge():
    text = (JOBS_DIR / "p1_predict_all.sh").read_text()
    assert "-m ctmr measure predict" in text
    runs = {match[0]: (match[1], match[2], match[3]) for match in re.findall(r"run_pred (\w+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s*&", text)}
    assert runs == {challenge: (spec.dataset_id, spec.plans, spec.config) for challenge, spec in INSTRUMENT_SPECS.items()}


def test_shell_orchestration_calls_only_the_canonical_entry():
    for name in SHELL_SCRIPTS:
        text = (JOBS_DIR / name).read_text()
        assert "-m ctmr measure predict" in text, name
        assert "nnUNetv2_predict_from_raw_data" not in text, name
        assert "l2_calibration_predict_entry" not in text, name
        assert "--disable_tta False" not in text, name  # the fatal token; bare mentions in comments are prose


def test_no_fatal_token_or_legacy_entry_name_remains_anywhere_in_scripts_deploy_or_src():
    """Repo-wide drift guard: the fatal ``--disable_tta False`` token (argparse
    ``unrecognized arguments``, #78) and both legacy entry names must never come
    back on the call-site side. ``src/`` is scanned for the fatal token only --
    module docstrings deliberately narrate the promotions, which is history,
    not a call site."""
    offenders = []
    executable_call_sites = (
        sorted((REPO_ROOT / "scripts").glob("*.py"))
        + sorted((REPO_ROOT / "scripts").glob("*.sh"))
        + sorted((REPO_ROOT / "deploy").rglob("*.sh"))
        + [REPO_ROOT / "src/ctmr/instrument/predict.py"]  # reverse shim until the last consumer switches
    )
    for path in executable_call_sites:
        text = path.read_text(errors="replace")
        for token in ("nnUNetv2_predict_from_raw_data", "l2_calibration_predict_entry", "--disable_tta False"):
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if "--disable_tta False" in path.read_text(errors="replace"):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: --disable_tta False")
    assert offenders == []
