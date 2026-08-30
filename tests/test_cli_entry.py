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

"""Unified ``ctmr`` console-entry surface (issue #130 / ADR-0015 §3; registry issue #228).

Observed purely through the CLI seam (argv in, dispatch/help/exit code out):
``ctmr --help`` lists the five command families pinned by the ADR; every
concrete invocation of a family whose verbs have not landed yet answers a
friendly "not migrated yet" message instead of an error traceback. The live
verbs are one registry (``ctmr.cli.VERBS``) that both the argparse spelling
tree and the dispatch router read, so these gates pin observed behavior --
help texts, unknown-verb exit codes, and the routed (handler, rest) result via
the public ``CtmrCli.route`` lookup -- and never import private methods: every
registry row must dispatch its handler argv verbatim (``peel_verb`` rows keep
the verb spelling for the module's own verb parser), and adding a verb is
provably one registry entry. The stdlib-only purity of ``ctmr.cli`` keeps the
light sci-stack CI job able to exercise these paths (ADR-0013 §4); the
dispatch gates pre-seed fake handler modules into ``sys.modules`` so the
torch/monai/nnunetv2 stacks stay out (lazy import hits ``sys.modules`` first).
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ctmr import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["generate", "measure", "accept", "data", "experiment"]
NOT_MIGRATED_FAMILIES = ["data", "experiment"]  # generate/measure/accept have live verbs; unknown accept layers/verbs are argparse errors

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
    # the migrated families advertise their blurb only; the not-yet-migrated
    # ones carry the migration banner
    for blurb in (
        "use-case chains modality-label|mask|cross-modal (train/dev-eval/generate/manifest)",
        "instrument-side prediction/calibration/FID",
        "quantitative/distribution/expert-review chains plus contract orchestration",
        "data preparation/encoding/download tooling -- not migrated yet",
        "experiment-record repository -- not migrated yet",
    ):
        # compare whitespace-squashed: argparse rewraps long help lines
        assert "".join(blurb.split()) in "".join(out.split())


def test_gen_alias_reaches_the_generate_family():
    route, rest = cli.CtmrCli().route(["gen", "cross-modal", "train", "--bad"])
    assert route.module == "ctmr.application.generation.cross_modal.train"
    assert route.kind == "train"
    assert list(rest) == ["--bad"]
    assert cli.CtmrCli().route(["gen", "cross-modal"]) is None
    route, rest = cli.CtmrCli().route(["gen", "mask", "train", "--bad"])
    assert route.module == "ctmr.application.generation.mask.train"
    assert list(rest) == ["--bad"]


def test_every_family_without_verbs_answers_not_migrated_for_any_concrete_call(capsys):
    for family in NOT_MIGRATED_FAMILIES:
        assert cli.main([family, "some-future-verb"]) == 2
        err = capsys.readouterr().err
        assert "not migrated yet" in err
        assert f"ctmr {family}" in err
        assert "some-future-verb" in err


def test_accept_routes_each_layer_to_its_module():
    route, rest = cli.CtmrCli().route(["accept", "quantitative", "evaluate", "--run", "r.json"])
    assert route.module == "ctmr.application.acceptance.quantitative.evaluate"
    assert list(rest) == ["--run", "r.json"]
    route, rest = cli.CtmrCli().route(["accept", "distribution", "assemble", "--phase", "P1"])
    assert route.module == "ctmr.application.acceptance.distribution.final_acceptance"
    assert list(rest) == ["assemble", "--phase", "P1"]  # peel_verb=False: the verb spelling stays in the handler argv
    route, rest = cli.CtmrCli().route(["accept", "expert-review", "build-package", "--seed", "7"])
    assert route.module == "ctmr.application.acceptance.expert_review.package"
    assert list(rest) == ["--seed", "7"]
    route, rest = cli.CtmrCli().route(["accept", "expert-review", "aggregate", "--seed", "7"])
    assert route.module == "ctmr.application.acceptance.expert_review.aggregate"
    assert list(rest) == ["--seed", "7"]
    route, rest = cli.CtmrCli().route(["accept", "contract", "conclude", "--run", "run.json"])
    assert route.module == "ctmr.application.acceptance.contract.cli"
    assert list(rest) == ["conclude", "--run", "run.json"]


def test_accept_unknown_layer_or_verb_falls_back_to_the_parser_tree():
    assert cli.CtmrCli().route(["accept", "biomarker", "run"]) is None  # unknown layer
    assert cli.CtmrCli().route(["accept", "contract", "rewind", "--run", "r.json"]) is None  # unknown verb
    assert cli.CtmrCli().route(["accept", "contract"]) is None  # verb required
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["accept", "biomarker", "run"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["accept", "contract", "rewind"])
    assert excinfo.value.code == 2


def test_accept_bare_invocation_is_a_clean_usage_error():
    # every accept layer is live now: a bare family call is an argparse usage
    # error (required <layer>), not a not-migrated pointer
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["accept"])
    assert excinfo.value.code == 2


def test_unknown_measure_verb_is_a_clean_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["measure", "some-future-verb"])
    assert excinfo.value.code == 2


def test_measure_bare_invocation_still_answers_not_migrated(capsys):
    assert cli.main(["measure"]) == 2
    err = capsys.readouterr().err
    assert "ctmr measure" in err
    assert "not migrated yet" in err


def test_migrated_generate_case_rejects_an_unknown_verb_as_a_usage_error():
    for argv in (["gen", "mask", "watch"], ["generate", "modality-label", "manifest"]):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 2


def test_live_generate_cases_route_to_their_family_handlers():
    route, rest = cli.CtmrCli().route(["generate", "modality-label", "train", "-e", "env.json"])
    assert route.module == "ctmr.application.generation.modality_label.train"
    assert route.kind == "train"
    assert list(rest) == ["-e", "env.json"]
    route, rest = cli.CtmrCli().route(["generate", "modality-label", "dev-eval", "select", "--out", "o.json"])
    assert route.module == "ctmr.application.generation.modality_label.monitor"
    assert list(rest) == ["select", "--out", "o.json"]


def test_bare_generate_case_answers_a_usage_pointer_not_a_traceback(capsys):
    assert cli.main(["generate", "modality-label"]) == 2
    err = capsys.readouterr().err
    assert "needs a verb" in err and "modality-label" in err
    assert cli.main(["gen", "cross-modal"]) == 2
    assert "needs a verb" in capsys.readouterr().err
    assert cli.main(["generate", "mask"]) == 2
    assert "needs a verb" in capsys.readouterr().err


def test_family_without_verb_also_answers_not_migrated(capsys):
    assert cli.main(["experiment"]) == 2
    assert "ctmr experiment" in capsys.readouterr().err


def test_bare_invocation_is_a_clean_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


def _fake_handler(monkeypatch, name="ctmr_cli_probe.fake_handler"):
    """Pre-seed a fake verb-handler module; lazy import hits ``sys.modules`` first."""
    fake = types.ModuleType(name)
    calls = []
    fake.main = lambda rest: calls.append(list(rest)) or 0
    monkeypatch.setitem(sys.modules, name, fake)
    return calls


@pytest.mark.parametrize("key", sorted(cli.VERBS))
def test_every_registry_row_routes_its_argv_verbatim(monkeypatch, key):
    """The table is the whole registration: every row dispatches its handler
    argv verbatim to the row's module -- peel_verb rows hand the flags only,
    passthrough rows keep the verb spelling for the module's own verb parser.
    The probe tokens would be argparse errors if the router pre-parsed them."""
    route = cli.VERBS[key]
    calls = _fake_handler(monkeypatch)
    monkeypatch.setitem(cli.VERBS, key, route._replace(module="ctmr_cli_probe.fake_handler", kind="main"))
    flags = ["STRICT", "--flag", "7"]  # STRICT would be an argparse error if the router pre-parsed it
    assert cli.main([*key, *flags]) == 0
    # peel_verb rows hand the flags only; passthrough rows keep the verb spelling
    expected = [*flags] if route.peel_verb else [key[-1], *flags]
    assert calls == [expected]


@pytest.mark.parametrize("key", sorted(cli.VERBS))
def test_every_registry_spelling_appears_in_the_parser_tree(key, capsys):
    """One table feeds both halves: each registry row's spelling shows up in
    the argparse help of its parent node, and the case/layer blurbs show up in
    the family-level help of the tree those nodes hang from. Text comparisons
    squash whitespace -- argparse rewraps long help lines."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*key[:-1], "--help"])
    assert excinfo.value.code == 0
    out = "".join(capsys.readouterr().out.split())
    assert key[-1] in out
    if len(key) > 2:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([key[0], "--help"])
        assert excinfo.value.code == 0
        family_help = "".join(capsys.readouterr().out.split())
        assert "".join(cli.CASE_BLURBS[(key[0], key[1])].split()) in family_help


def test_adding_a_generate_verb_is_one_registry_entry(monkeypatch, capsys):
    """Registration is exactly one table entry: a new generate row dispatches
    its argv verbatim and spells into the argparse tree with its help text --
    no other code or table changes."""
    calls = _fake_handler(monkeypatch, "ctmr_cli_probe.duplicate")
    monkeypatch.setitem(
        cli.VERBS,
        ("generate", "mask", "duplicate"),
        cli.VerbRoute("ctmr_cli_probe.duplicate", help="duplicate a mask case"),
    )
    assert cli.main(["generate", "mask", "duplicate", "--out", "x"]) == 0
    assert calls == [["--out", "x"]]  # argv verbatim past the fixed prefix
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["generate", "mask", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "duplicate" in out and "duplicate a mask case" in out


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
