# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified ``ctmr`` console-entry surface (issue #130 / ADR-0015 §3).

Observed purely through the CLI seam: ``ctmr --help`` lists the five command
families pinned by the ADR; every concrete invocation of a family whose verbs
have not landed yet answers a friendly "not migrated yet" message instead of an
error traceback; the live ``generate cross-modal`` family hands its entry argv
to the family module (ticket 08 -- the argv↔namespace gate lives there). The
stdlib-only purity of ``ctmr.cli`` keeps the light sci-stack CI job able to
exercise these paths (ADR-0013 §4).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ctmr import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["generate", "measure", "accept", "data", "experiment"]
NOT_MIGRATED_FAMILIES = ["accept", "data", "experiment"]  # generate/measure have live verbs; measure unknown verbs are argparse errors

_HEAVY_DEPS = [
    "torch",
    "monai",
    "numpy",
    "scipy",
    "skimage",
    "nibabel",
    "SimpleITK",
    "PIL",
    "matplotlib",
    "einops",
    "huggingface_hub",
    "tqdm",
    "fire",
    "tensorboard",
]


def test_help_lists_all_five_command_families(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for family in FAMILIES:
        assert family in out


def test_gen_alias_reaches_the_generate_family():
    peeled = cli.CtmrCli._peel_generate(["gen", "cross-modal", "train", "--bad"])
    assert peeled[0] is cli.CtmrCli._run_cross_modal_train
    assert list(peeled[1]) == ["--bad"]
    assert cli.CtmrCli._peel_generate(["gen", "cross-modal"]) is None


def test_every_family_without_verbs_answers_not_migrated_for_any_concrete_call(capsys):
    for family in NOT_MIGRATED_FAMILIES:
        assert cli.main([family, "some-future-verb"]) == 2
        err = capsys.readouterr().err
        assert "not migrated yet" in err
        assert f"ctmr {family}" in err
        assert "some-future-verb" in err


def test_unknown_measure_verb_is_a_clean_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["measure", "some-future-verb"])
    assert excinfo.value.code == 2


def test_measure_bare_invocation_still_answers_not_migrated(capsys):
    assert cli.main(["measure"]) == 2
    err = capsys.readouterr().err
    assert "ctmr measure" in err
    assert "not migrated yet" in err


def test_not_migrated_generate_cases_still_answer_not_migrated(capsys):
    assert cli.main(["generate", "modality-label", "train"]) == 2
    assert "modality-label" in capsys.readouterr().err
    assert cli.main(["gen", "mask", "watch"]) == 2
    assert "mask" in capsys.readouterr().err


def test_family_without_verb_also_answers_not_migrated(capsys):
    assert cli.main(["experiment"]) == 2
    assert "ctmr experiment" in capsys.readouterr().err


def test_bare_invocation_is_a_clean_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def test_python_dash_m_matches_console_behavior():
    result = subprocess.run(
        [sys.executable, "-m", "ctmr.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0
    assert all(family in result.stdout for family in FAMILIES)


def test_cli_import_pulls_no_third_party_dependency():
    probe = "import ctmr.cli, sys\nprint(sorted(name for name in sys.argv[1:] if name in sys.modules))\n"
    result = subprocess.run(
        [sys.executable, "-c", probe, *_HEAVY_DEPS],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.stdout.strip() == "[]"
