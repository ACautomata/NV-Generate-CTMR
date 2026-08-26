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

"""Convergence gate for the ``ctmr`` CLI skeleton (issue #130 / ADR-0015 §3).

The five command-family nouns must be registered exactly (``generate`` with
alias ``gen``), every not-yet-migrated verb must get the friendly skeleton
notice (exit 2, no traceback, nothing executed), bare ``ctmr`` prints help,
and unknown families stay argparse usage errors. Library-level only: these
tests ride the pythonpath dual track, no install needed.
"""

import pytest

from ctmr.cli import FAMILIES, CtmrCli


def test_exactly_five_command_families_registered():
    assert {family.name: family.aliases for family in FAMILIES} == {
        "generate": ("gen",),
        "measure": (),
        "accept": (),
        "data": (),
        "experiment": (),
    }


def test_help_lists_all_five_families(capsys):
    with pytest.raises(SystemExit) as excinfo:
        CtmrCli().run(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for family in FAMILIES:
        assert family.name in out
    assert "gen" in out


def test_bare_invocation_prints_help_and_exits_zero(capsys):
    assert CtmrCli().run([]) == 0
    assert "usage: ctmr" in capsys.readouterr().out


def test_every_family_answers_with_skeleton_notice(capsys):
    for family in FAMILIES:
        assert CtmrCli().run([family.name]) == 2
        err = capsys.readouterr().err
        assert f"ctmr {family.name}" in err
        assert "尚未迁移" in err
        assert "nothing was executed" in err


def test_alias_gen_resolves_to_generate_notice(capsys):
    assert CtmrCli().run(["gen"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ctmr generate:")  # canonical name echoed even via the alias


def test_received_verbs_are_echoed_without_executing(capsys):
    assert CtmrCli().run(["measure", "predict", "img-dir", "out-dir"]) == 2
    err = capsys.readouterr().err
    assert "(received: predict img-dir out-dir)" in err
    assert "planned verbs (ADR-0015 §3): predict, calibrate, fid" in err


def test_experiment_family_has_open_verb_plan(capsys):
    assert CtmrCli().run(["experiment"]) == 2
    assert "planned verbs (ADR-0015 §3): not fixed yet" in capsys.readouterr().err


def test_unknown_family_is_argparse_usage_error_not_traceback():
    with pytest.raises(SystemExit) as excinfo:
        CtmrCli().run(["bogus"])
    assert excinfo.value.code == 2
