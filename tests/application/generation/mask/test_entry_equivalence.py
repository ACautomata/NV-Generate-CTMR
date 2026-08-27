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

"""argv↔namespace equivalence gate for the mask family entries (ticket 09).

One mapping table covering every migrated entry: the same argv that the
retired mask script entries accepted must yield an equal argparse namespace
through the new home. The reference parsers below are the retired entries'
argparse constructions, verbatim (do not edit -- drift here is exactly what
this gate exists to catch); self-referential metadata fields (help/description
text, provenance ``script`` values) are exempt -- only the flag-derived values
are compared. The entry modules import monai/torch at module level (they are
the production entries), so this gate is torch-marked and runs for real in the
CI full-dependency tier (ADR-0015 §6) -- it must never be skipped around the
torch mark itself.
"""

from __future__ import annotations

import argparse

import pytest

from ctmr import cli
from ctmr.application.generation.launcher import num_gpus_of
from ctmr.application.generation.mask import monitor, sample
from ctmr.application.train_cli import TrainCli

pytestmark = pytest.mark.torch

# ------------------------------------------------------------- old argv sets

FINETUNE_ARGV = [
    "-e",
    "run/environment.json",
    "-c",
    "configs/config_brats_p2_train.json",
    "-t",
    "configs/config_network_rflow.json",
    "-g",
    "7",
]
FINETUNE_ARGV_AMP = FINETUNE_ARGV + ["--no_amp", "--amp_dtype", "fp16"]

DEV_EVAL_REFERENCE_ARGV = [
    "reference",
    "--dev-list",
    "lists/dev.json",
    "--raw-root",
    "/phase/raw",
    "--eval-root",
    "/phase/dev",
]
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
    "--label-root",
    "/phase",
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
    "--nnunet-raw",
    "/raw",
    "--nnunet-preprocessed",
    "/pre",
    "--idle-exit-seconds",
    "120",
]
DEV_EVAL_SELECT_ARGV = ["select", "--eval-root", "/phase/dev", "--ckpt-dir", "/phase/ckpt", "--out", "/phase/select.json"]

SAMPLE_ARGV = [
    "--run",
    "runs/p2/run.json",
    "--manifest",
    "/ctrl/phase/phase_manifest.json",
    "--out-root",
    "/ctrl/p2/holdout_generated",
    "--raw-root",
    "/ctrl/phase/raw",
    "--label-root",
    "/ctrl/phase",
    "-e",
    "env.json",
    "-c",
    "config.json",
    "-t",
    "net.json",
    "--shard",
    "0",
    "--num-shards",
    "8",
    "--limit",
    "2",
    "--challenge",
    "GLI",
    "--only-cases",
    "BraTS-GLI-0001-000",
    "BraTS-GLI-0002-000",
]

# ------------------------------------------------- reference parsers (verbatim)


def _reference_finetune_parser():
    """The retired mask finetune entry's argparse surface, verbatim."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    return parser


def _reference_dev_eval_parser():
    """The retired dev-eval entry's argparse surface, verbatim (selftest retired with the entry)."""
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
    p.add_argument("--label-root", required=True)
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
    p.add_argument("--nnunet-raw", default="/root/private_data/brats2023_nnunet")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/nnUNet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser


def _reference_sample_parser():
    """The retired holdout-generate entry's argparse surface, verbatim."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="mask brats-phase-run record with a recorded selection")
    parser.add_argument("--manifest", required=True, help="pinned phase phase_manifest.json")
    parser.add_argument("--out-root", required=True, help="controlled output root")
    parser.add_argument("--raw-root", required=True, help="phase raw root (holdout images land here)")
    parser.add_argument("--label-root", required=True, help="phase root holding labels/<CH>/<case>/")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="max holdout cases per challenge")
    parser.add_argument("--challenge", default=None, help="restrict to one challenge")
    parser.add_argument("--only-cases", nargs="*", default=None)
    return parser


# --------------------------------------------------------------- the gates

TRAIN_MIGRATED = TrainCli("description", stage="p2").parse
ENTRY_TABLE = {
    "train": (FINETUNE_ARGV + ["--no_amp", "--amp_dtype", "fp16"], _reference_finetune_parser, TRAIN_MIGRATED),
    "dev-eval reference": (DEV_EVAL_REFERENCE_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "dev-eval watch": (DEV_EVAL_WATCH_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "dev-eval select": (DEV_EVAL_SELECT_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "generate": (SAMPLE_ARGV, _reference_sample_parser, sample.parse_args),
}

CLI_PREFIXES = {
    "train": ["gen", "mask", "train"],
    "dev-eval reference": ["generate", "mask", "dev-eval"],
    "dev-eval watch": ["generate", "mask", "dev-eval"],
    "dev-eval select": ["generate", "mask", "dev-eval"],
    "generate": ["generate", "mask", "generate"],
}


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_cli_forwards_the_entry_argv_verbatim(name):
    argv, _, _ = ENTRY_TABLE[name]
    peeled = cli.CtmrCli._peel_generate([*CLI_PREFIXES[name], *argv])
    assert peeled is not None
    rest = peeled[-1]
    assert list(rest) == argv


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_entry_namespace_is_unchanged_against_the_retired_parsers(name):
    argv, reference_builder, migrated = ENTRY_TABLE[name]
    reference = reference_builder().parse_args(argv)
    assert vars(migrated(argv)) == vars(reference), f"{name}: migrated namespace drifted"


def test_train_cli_derives_num_gpus_from_the_entry_argv():
    assert num_gpus_of(FINETUNE_ARGV) == 7
    assert num_gpus_of(["-e", "e.json"]) == 1  # the TrainCli default


def test_mask_train_module_is_pinned_for_the_launcher():
    """The torchrun child module path stays the mask family train entry."""
    assert cli.TRAIN_MODULES["mask"] == "ctmr.application.generation.mask.train"
