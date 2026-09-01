# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""argv↔namespace equivalence gate for the modality_label family entries (ticket 10).

One mapping table covering every migrated entry: the same argv that the
retired modality-label script entries accepted -- including the family's
``--replay-list`` replay-mix flag -- must yield an equal argparse namespace
through the new home. The reference parsers below are the retired entries'
argparse constructions, verbatim (do not edit -- drift here is exactly what
this gate exists to catch); self-referential metadata fields (help/description
text, provenance ``script`` values) are exempt -- only the flag-derived values
are compared. The retired dev-eval entry's ``selftest`` subcommand is not
covered: it retired with the entry (its assertions live as pytest functions).
The entry modules import monai/torch at module level (they are the production
entries), so this gate is torch-marked and runs for real in the CI
full-dependency tier (ADR-0015 §6) -- it must never be skipped around the
torch mark itself.
"""

from __future__ import annotations

import argparse

import pytest

from ctmr import cli
from ctmr.application.generation.launcher import num_gpus_of
from ctmr.application.generation.modality_label import monitor
from ctmr.application.train_cli import TrainCli

pytestmark = pytest.mark.torch

# ------------------------------------------------------------- old argv sets

FINETUNE_ARGV = [
    "-e",
    "run/environment.json",
    "-c",
    "configs/config_brats_p1_train.json",
    "-t",
    "configs/config_network_rflow.json",
    "-g",
    "7",
    "--replay-list",
    "run/lists/p1_mrrate_replay.json",
]
FINETUNE_ARGV_AMP = FINETUNE_ARGV + ["--no_amp", "--amp_dtype", "fp16"]

DEV_EVAL_REFERENCE_ARGV = ["reference", "--dev-list", "lists/dev.json", "--raw-root", "/phase/raw", "--eval-root", "/phase/dev"]
DEV_EVAL_WATCH_ARGV = [
    "watch",
    "--ckpt-dir",
    "/phase/ckpt",
    "--eval-root",
    "/phase/dev",
    "--dev-list",
    "lists/dev.json",
    "--raw-root",
    "/phase/raw",
    "--emb-root",
    "/phase/embeddings",
    "-e",
    "env.json",
    "-c",
    "config.json",
    "-t",
    "net.json",
    "--eval-every",
    "5",
    "--patience",
    "3",
    "--min-epoch",
    "30",
    "--max-epoch",
    "100",
    "--poll-seconds",
    "30.0",
    "--skip-l2",
    "--instrument-results",
    "GLI=/results/gli",
    "--instrument-results",
    "SSA=/results/ssa",
    "--nnunet-raw",
    "/nnunet/raw",
    "--nnunet-preprocessed",
    "/nnunet/pre",
    "--idle-exit-seconds",
    "120",
]
DEV_EVAL_SELECT_ARGV = ["select", "--eval-root", "/phase/dev", "--ckpt-dir", "/phase/ckpt", "--out", "/phase/select.json"]

# ------------------------------------------------- reference parsers (verbatim)


def _reference_finetune_parser():
    """The retired modality-label finetune entry's argparse surface, verbatim.

    The two declared evolutions since the migration (issue #278, ADR-0019 §4-§5,
    not drift): ``-g`` defaults to 8 and the entry carries ``--val-every``."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=8)
    parser.add_argument(
        "--replay-list",
        dest="replay_list",
        action="append",
        required=True,
        help="MR-RATE replay data list (spec: list-level 1:1 mix; append once per list)",
    )
    parser.add_argument("--val-every", dest="val_every", type=int, default=10, help="embedded periodic validation interval in epochs (0 disables)")
    parser.add_argument("--dev-list", dest="dev_list", default=None, help="dev cohort list json (embedded validation)")
    parser.add_argument("--raw-root", dest="raw_root", default=None, help="raw volume root for the dev real bank (embedded validation)")
    parser.add_argument("--emb-root", dest="emb_root", default=None, help="phase embedding root for per-case spacing (embedded validation)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    return parser


def _reference_dev_eval_parser():
    """The retired dev-eval entry's argparse surface, verbatim (minus the retired selftest verb)."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="build the dev real-feature bank once")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)

    p = sub.add_parser("watch", help="sidecar loop: evaluate epoch checkpoints as they land")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--skip-l2", action="store_true", help="FID-only trend (instruments unavailable)")
    p.add_argument("--instrument-results", action="append", default=[], help="CHALLENGE=nnUNet_results path")
    p.add_argument("--nnunet-raw", default="/root/private_data/ctmr/data/nnunet_raw")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/ctmr/data/nnunet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser


# --------------------------------------------------------------- the gates

TRAIN_MIGRATED = TrainCli("description", stage="p1").parse
ENTRY_TABLE = {
    "train": (FINETUNE_ARGV_AMP, _reference_finetune_parser, TRAIN_MIGRATED, "ctmr.application.generation.modality_label.train"),
    "dev-eval reference": (DEV_EVAL_REFERENCE_ARGV, _reference_dev_eval_parser, monitor.parse_args, monitor.__name__),
    "dev-eval watch": (DEV_EVAL_WATCH_ARGV, _reference_dev_eval_parser, monitor.parse_args, monitor.__name__),
    "dev-eval select": (DEV_EVAL_SELECT_ARGV, _reference_dev_eval_parser, monitor.parse_args, monitor.__name__),
}

CLI_PREFIXES = {
    "train": ["gen", "modality-label", "train"],
    "dev-eval reference": ["generate", "modality-label", "dev-eval"],
    "dev-eval watch": ["generate", "modality-label", "dev-eval"],
    "dev-eval select": ["generate", "modality-label", "dev-eval"],
}


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_cli_forwards_the_entry_argv_verbatim(name):
    argv, _, _, module = ENTRY_TABLE[name]
    route, rest = cli.CtmrCli().route([*CLI_PREFIXES[name], *argv])
    assert route.module == module
    assert list(rest) == argv


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_entry_namespace_is_unchanged_against_the_retired_parsers(name):
    argv, reference_builder, migrated, _ = ENTRY_TABLE[name]
    reference = reference_builder().parse_args(argv)
    assert vars(migrated(argv)) == vars(reference), f"{name}: migrated namespace drifted"


def test_replay_mix_flag_accepts_repeated_lists():
    """The replay-mix flag stays append-style: several --replay-list occurrences accumulate."""
    argv = FINETUNE_ARGV + ["--replay-list", "run/lists/p1_mrrate_replay_2.json"]
    args = TRAIN_MIGRATED(argv)
    reference = _reference_finetune_parser().parse_args(argv)
    assert args.replay_list == reference.replay_list == ["run/lists/p1_mrrate_replay.json", "run/lists/p1_mrrate_replay_2.json"]


def test_replay_mix_flag_is_required():
    with pytest.raises(SystemExit):
        TrainCli("description", stage="p1").parse(["-e", "e.json", "-c", "c.json", "-t", "t.json"])


def test_train_cli_derives_num_gpus_from_the_entry_argv():
    assert num_gpus_of(FINETUNE_ARGV) == 7
    assert num_gpus_of(["-e", "e.json", "-c", "c.json", "-t", "t.json"]) == 8  # the TrainCli default (issue #278)


@pytest.mark.parametrize("verb", ["generate", "sample", "batch"])
def test_retired_batch_generation_has_no_cli_verb(verb):
    """The #38 retired P1-style batch generation stays retired: the family's CLI
    face is exactly train/dev-eval -- no batch-sampling verb may come back."""
    assert cli.CtmrCli().route(["generate", "modality-label", verb]) is None
