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

"""torchrun spawn derivation for the cross-modal train entry (ticket 08, criterion 3).

``ctmr generate cross-modal train`` arrives WITHOUT torchrun; the application layer
derives the ``torchrun --nproc_per_node=<num_gpus> -m <module> <argv>`` child (spawn
precedent #123: no fork). These gates demonstrate that derivation: the exact child
command, the ``-g/--num_gpus`` extraction, the exit-code relay, and the CLI's
WORLD_SIZE fork (already a torchrun worker -> dispatch in-process; otherwise spawn).

Both ``ctmr.cli`` and ``ctmr.application.generation.launcher`` are stdlib-only, so
this module needs no torch mark. The CLI fork gates pre-seed a fake ``train`` module
into ``sys.modules`` so the in-process branch never imports the torch stack; the
spawn is observed by stubbing ``subprocess.run`` (no real child is started).
"""

from __future__ import annotations

import sys
import types

import pytest

from ctmr import cli
from ctmr.application.generation import launcher as launcher_mod
from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

TRAIN_MODULE = "ctmr.application.generation.cross_modal.train"


def test_command_is_the_spawn_torchrun_invocation():
    launcher = TorchrunLauncher(TRAIN_MODULE, ["-e", "env.json", "-c", "cfg.json"], num_gpus=4)
    assert launcher.command() == ["torchrun", "--nproc_per_node", "4", "--nnodes", "1", "-m", TRAIN_MODULE, "-e", "env.json", "-c", "cfg.json"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-e", "env.json", "-g", "7"], 7),
        (["-e", "env.json", "--num_gpus", "2"], 2),
        (["-e", "env.json"], 1),  # the TrainCli default
        (["-g"], 1),  # a valueless trailing flag falls back to the default
    ],
)
def test_num_gpus_is_extracted_from_the_train_argv(argv, expected):
    assert num_gpus_of(argv) == expected


def test_run_derives_the_child_and_relays_its_exit_code(monkeypatch):
    captured = {}

    class _Proc:
        returncode = 7

    def _fake_run(cmd, stdout=None, stderr=None, check=False):
        captured["cmd"] = cmd
        captured["check"] = check
        return _Proc()

    monkeypatch.setattr(launcher_mod.subprocess, "run", _fake_run)
    launcher = TorchrunLauncher(TRAIN_MODULE, ["-e", "env.json", "-g", "2"], num_gpus=2)
    assert launcher.run() == 7  # the trainer's exit code is relayed verbatim
    assert captured["cmd"] == launcher.command()
    assert captured["check"] is False


@pytest.fixture()
def _fake_train(monkeypatch):
    """Pre-seed a fake train module so the in-process branch never touches torch.

    ``from ctmr.application.generation.cross_modal import train`` resolves the
    ``train`` attribute on the already-imported package first (not ``sys.modules``),
    and once any torch-tier test has imported the real module that attribute points
    at it. Patch both the package attribute and ``sys.modules`` so the fake wins in
    either stack.
    """
    import ctmr.application.generation.cross_modal as cross_modal_pkg

    module = types.ModuleType(TRAIN_MODULE)
    calls = []

    def _main(rest):
        calls.append(list(rest))
        return 0

    module.main = _main
    monkeypatch.setattr(cross_modal_pkg, "train", module, raising=False)
    monkeypatch.setitem(sys.modules, TRAIN_MODULE, module)
    return calls


def test_cli_spawns_torchrun_when_world_size_is_absent(monkeypatch, _fake_train):
    captured = {}

    class _Proc:
        returncode = 0

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(launcher_mod.subprocess, "run", _fake_run)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    rc = cli.CtmrCli()._run_cross_modal_train(["-e", "env.json", "-g", "4"])
    assert rc == 0
    assert captured["cmd"][:2] == ["torchrun", "--nproc_per_node"]  # the spawn derivation fired
    assert captured["cmd"][2] == "4"
    assert _fake_train == []  # the in-process train entry was NOT called


def test_cli_dispatches_in_process_when_already_a_torchrun_worker(monkeypatch, _fake_train):
    def _no_spawn(cmd, **kw):  # pragma: no cover - must never fire on this branch
        raise AssertionError("torchrun must not be re-derived inside a torchrun worker")

    monkeypatch.setattr(launcher_mod.subprocess, "run", _no_spawn)
    monkeypatch.setenv("WORLD_SIZE", "4")
    rc = cli.CtmrCli()._run_cross_modal_train(["-e", "env.json"])
    assert rc == 0
    assert _fake_train == [["-e", "env.json"]]  # the train entry ran in-process with argv verbatim
