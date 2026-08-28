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

"""Convergence-gate tests for the common phase-train CLI surface (ADR-0011, #111).

The embedded reference parsers are the pre-#111 ``brats_p{1,2,3}_finetune.py`` (retired scripts layer, git history)
argparse blocks, verbatim (do not edit -- drift here is exactly what this gate
exists to catch). The gate: same argv must produce an equal argparse namespace
before and after the harness consolidation, and the torchrun WORLD_SIZE check
must raise / pass identically. Stdlib-only: any machine, no torch (ADR-0013 §4).
"""

import argparse

import pytest

from ctmr.application.train_cli import TrainCli

COMMON_ARGV = ["-e", "run/env.json", "-c", "configs/train.json", "-t", "configs/net.json", "-g", "7"]

P1_ARGV = COMMON_ARGV + ["--replay-list", "lists/a.json", "--replay-list", "lists/b.json"]
P2_ARGV = COMMON_ARGV
P3_ARGV = COMMON_ARGV + ["--data-list", "runs/p3/p3_pairs.json"]


def _reference_parser(stage_flags):
    """The pre-#111 finetune argparse construction, verbatim (shared block + stage flags)."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    stage_flags(parser)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    return parser


def _reference_p1_flags(parser):
    parser.add_argument(
        "--replay-list",
        dest="replay_list",
        action="append",
        required=True,
        help="MR-RATE replay data list (spec: list-level 1:1 mix; append once per list)",
    )


def _reference_p2_flags(parser):
    pass


def _reference_p3_flags(parser):
    parser.add_argument("--data-list", default=None, help="p3_pairs.json (defaults to env json_data_list)")


@pytest.mark.parametrize(
    "stage,argv,reference_flags",
    [("p1", P1_ARGV, _reference_p1_flags), ("p2", P2_ARGV, _reference_p2_flags), ("p3", P3_ARGV, _reference_p3_flags)],
)
def test_same_argv_yields_an_equal_namespace(stage, argv, reference_flags):
    reference = _reference_parser(reference_flags).parse_args(argv)
    consolidated = TrainCli("description", stage=stage).parse(argv)
    assert vars(consolidated) == vars(reference)


@pytest.mark.parametrize(
    "stage,argv,reference_flags,expected_amp",
    [
        ("p1", P1_ARGV + ["--no_amp", "--amp_dtype", "fp16"], _reference_p1_flags, False),
        ("p2", P2_ARGV + ["--no_amp"], _reference_p2_flags, False),
        ("p3", P3_ARGV + ["--amp_dtype", "fp16"], _reference_p3_flags, True),
    ],
)
def test_amp_switches_survive_the_consolidation(stage, argv, reference_flags, expected_amp):
    reference = _reference_parser(reference_flags).parse_args(argv)
    consolidated = TrainCli("description", stage=stage).parse(argv)
    assert vars(consolidated) == vars(reference)
    assert consolidated.amp is expected_amp  # the store_false destination is preserved


def test_p1_replay_list_appends_and_is_required():
    args = TrainCli("description", stage="p1").parse(P1_ARGV)
    assert args.replay_list == ["lists/a.json", "lists/b.json"]
    with pytest.raises(SystemExit):
        TrainCli("description", stage="p1").parse(COMMON_ARGV)  # missing the required replay list


def test_p3_data_list_defaults_to_none():
    args = TrainCli("description", stage="p3").parse(COMMON_ARGV)
    assert args.data_list is None


@pytest.mark.parametrize(
    "stage,argv,reference_flags",
    [("p1", P1_ARGV, _reference_p1_flags), ("p2", P2_ARGV, _reference_p2_flags), ("p3", P3_ARGV, _reference_p3_flags)],
)
def test_torchrun_disagreement_raises_identically(stage, argv, reference_flags, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "7")

    # Append -g AFTER the stage argv: --replay-list consumes the next token as its value.
    with pytest.raises(ValueError, match=r"--num_gpus 1 disagrees with torchrun WORLD_SIZE 7"):
        TrainCli("description", stage=stage).parse([*argv, "-g", "1"])

    reference = _reference_parser(reference_flags).parse_args([*argv, "-g", "1"])
    torchrun_world = int(__import__("os").environ["WORLD_SIZE"])
    with pytest.raises(ValueError):
        if torchrun_world != reference.num_gpus:
            raise ValueError(f"--num_gpus {reference.num_gpus} disagrees with torchrun WORLD_SIZE {torchrun_world}")


@pytest.mark.parametrize(
    "stage,argv",
    [("p1", P1_ARGV), ("p2", P2_ARGV), ("p3", P3_ARGV)],
)
def test_no_torchrun_means_no_validation(monkeypatch, stage, argv):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert TrainCli("description", stage=stage).parse(argv).num_gpus == 7


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match=r"unknown stage"):
        TrainCli("description", stage="p9")
