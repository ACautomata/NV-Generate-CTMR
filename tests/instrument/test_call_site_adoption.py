"""Convergence-gate tests for the #108 instrument call-site adoptions (ADR-0009).

The #107 gate pinned the module itself (``build`` output vs the canonical
snapshot); this file proves the *call sites* adopted in issue #108 land on that
single construction point: each adopted Python call site produces precisely the
``FrozenInstrumentCommand.build`` argv (decision 5), the fatal
``--disable_tta False`` token and the non-standard entry names
(``nnUNetv2_predict_from_raw_data`` / ``l2_calibration_predict_entry``) are gone
everywhere, and shell orchestration shares the canonical
``python -m ctmr.instrument.predict`` entry with the Python callers (decision 3).

Light stack, any machine, no cluster, no external data (ADR-0009 decision 7 /
ADR-0013 §4): the torch-heavy dev-eval call site is gated separately in
``tests/dev_eval``; the sugon subprocess call is exercised with monkeypatched
paths and a captured ``subprocess.run``.
"""

import re
import types
from pathlib import Path

import pytest

import scripts.l2_synth_domain_sugon as sugon  # noqa: E402
import scripts.nnunet_l2_final_acceptance as final_acceptance  # noqa: E402
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand
from scripts.nnunet_l2_final_acceptance import PredictScriptWriter  # noqa: E402
from scripts.nnunet_l2_synthetic_domain_eval import InstrumentRunner  # noqa: E402

FIVE_CHALLENGES = sorted(INSTRUMENT_SPECS)
REPO_ROOT = Path(__file__).resolve().parents[2]


# ── nnunet_l2_final_acceptance.PredictScriptWriter (frozen terminal acceptance) ──────


def test_predict_script_writer_emits_the_builder_argv(tmp_path):
    challenges = {challenge: {} for challenge in FIVE_CHALLENGES}  # the writer only reads the keys
    PredictScriptWriter({"challenges": challenges}, tmp_path).write()

    for challenge in FIVE_CHALLENGES:
        cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(tmp_path / "inputs" / challenge, tmp_path / "predictions" / challenge)
        line = (tmp_path / f"predict_{challenge}.sh").read_text().splitlines()[-1]
        assert line == " ".join(cmd)


def test_predict_script_writer_bootstraps_the_src_tree_onto_pythonpath(tmp_path):
    """The generated scripts run the canonical entry ``python -m ctmr.instrument.predict``
    in a fresh shell; the writer pins its own src tree onto PYTHONPATH so the module
    is importable on the machine that generated them (repo and flat-deployment spellings,
    matching the sibling executors' shim)."""
    PredictScriptWriter({"challenges": {"GLI": {}}}, tmp_path).write()
    lines = (tmp_path / "predict_GLI.sh").read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1] == "set -euo pipefail"
    assert lines[2].startswith("export PYTHONPATH=")
    src_root = Path(final_acceptance.__file__).resolve().parent.parent / "src"
    assert str(src_root) in lines[2]
    assert "-m ctmr.instrument.predict" in lines[-1]


def test_predict_script_writer_runner_executes_every_challenge_script(tmp_path):
    challenges = {challenge: {} for challenge in ("GLI", "SSA")}  # a subset proves ordering
    runner = PredictScriptWriter({"challenges": challenges}, tmp_path).write()

    assert runner == tmp_path / "predict_all.sh"
    lines = (tmp_path / "predict_all.sh").read_text().splitlines()
    assert lines[:2] == ["#!/bin/bash", "set -euo pipefail"]
    assert lines[2:] == ["bash predict_GLI.sh", "bash predict_SSA.sh"]


# ── nnunet_l2_synthetic_domain_eval.InstrumentRunner (non-frozen, #38 family) ─────────


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


# ── l2_synth_domain_sugon.cmd_predict (sugon self-contained copy → shim) ──────────────


def test_synth_domain_sugon_predict_uses_the_builder_argv(tmp_path, monkeypatch):
    eval_root = tmp_path / "eval"
    monkeypatch.setattr(sugon, "EVAL_ROOT", eval_root)
    monkeypatch.setattr(sugon, "NNUNET_ROOT", tmp_path / "nnunet")
    monkeypatch.setattr(sugon, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(sugon, "RESULTS_SSA", tmp_path / "results_ssa")
    monkeypatch.setattr(sugon, "RESULTS_52667", tmp_path / "results_52667")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("cmd", []).append(cmd)
        captured["env"] = kwargs.get("env", {})
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sugon.subprocess, "run", fake_run)

    for challenge in FIVE_CHALLENGES:
        case_dir = eval_root / "p1_nnunet_inputs" / challenge
        case_dir.mkdir(parents=True)
        for suffix in ("0000", "0001", "0002", "0003"):
            (case_dir / f"SYNTH-0001_{suffix}.nii.gz").write_bytes(b"dummy\n")

    sugon.cmd_predict(types.SimpleNamespace(mode="p1"))

    assert len(captured["cmd"]) == len(FIVE_CHALLENGES)
    for challenge in FIVE_CHALLENGES:
        tmp_dataset = eval_root / "_tmp_nnunet_dataset" / INSTRUMENT_SPECS[challenge].dataset_id
        expected = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(tmp_dataset, eval_root / "p1_predictions" / challenge)
        assert expected in captured["cmd"]
        assert "--disable_tta" not in expected
    # nnU-Net env wiring stays with the executor (execution side, ADR-0009 decision 1)
    assert captured["env"]["nnUNet_raw"]
    # the child process gets the module's src tree on PYTHONPATH (process-local
    # sys.path.insert does not reach a fresh `python -m ctmr.instrument.predict`)
    assert str(REPO_ROOT / "src") in captured["env"]["PYTHONPATH"]


# ── shell orchestration (shared canonical entry, decision 3) ──────────────────────────

SHELL_SCRIPTS = ["synth_domain_eval.sh", "predict_all.sh", "calibration_predict.sh"]
# cluster job recipes live in deploy/jobs since ticket #131 (ADR-0015 §5)
JOBS_DIR = REPO_ROOT / "deploy" / "jobs"

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
        "synth_domain_eval.sh": {"DATASET_NAME": "dataset_id", "PLANS": "plans", "CONFIG": "config"},
        "calibration_predict.sh": {"DATASET": "dataset_id", "PLANS": "plans", "CONFIG": "config"},
    }
    for name, mappings in runners.items():
        maps = _declared_maps((JOBS_DIR / name).read_text())
        for map_name, field in mappings.items():
            for challenge in FIVE_CHALLENGES:
                assert maps[map_name][challenge] == getattr(INSTRUMENT_SPECS[challenge], field), (name, map_name, challenge)


def test_predict_all_sh_runs_the_canonical_command_per_challenge():
    text = (JOBS_DIR / "predict_all.sh").read_text()
    assert "-m ctmr.instrument.predict" in text
    runs = {match[0]: (match[1], match[2], match[3]) for match in re.findall(r"run_pred (\w+)\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s*&", text)}
    assert runs == {challenge: (spec.dataset_id, spec.plans, spec.config) for challenge, spec in INSTRUMENT_SPECS.items()}


def test_shell_orchestration_calls_only_the_canonical_entry():
    for name in SHELL_SCRIPTS:
        text = (JOBS_DIR / name).read_text()
        assert "-m ctmr.instrument.predict" in text, name
        assert "nnUNetv2_predict_from_raw_data" not in text, name
        assert "l2_calibration_predict_entry" not in text, name
        assert "--disable_tta False" not in text, name  # the fatal token; bare mentions in comments are prose


def test_no_fatal_token_or_legacy_entry_name_remains_anywhere_in_scripts_or_src():
    """Repo-wide drift guard: the fatal ``--disable_tta False`` token (argparse
    ``unrecognized arguments``, #78) and both legacy entry names must never come
    back on the call-site side. ``src/`` is scanned for the fatal token only --
    its module docstrings deliberately narrate the promotion
    (``l2_calibration_predict_entry.py``), which is history, not a call site."""
    offenders = []
    for path in (
        sorted((REPO_ROOT / "scripts").glob("*.py")) + sorted((REPO_ROOT / "scripts").glob("*.sh")) + sorted((REPO_ROOT / "deploy").rglob("*.sh"))
    ):
        text = path.read_text(errors="replace")
        for token in ("nnUNetv2_predict_from_raw_data", "l2_calibration_predict_entry", "--disable_tta False"):
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if "--disable_tta False" in path.read_text(errors="replace"):
            offenders.append(f"{path.relative_to(REPO_ROOT)}: --disable_tta False")
    assert not offenders, "\n".join(offenders)
