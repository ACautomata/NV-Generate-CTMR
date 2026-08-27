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

"""argv↔namespace equivalence gate for the cross_modal family entries (ticket 08).

One mapping table covering every migrated entry: the same argv that the
retired cross-modal script entries accepted must yield an equal
argparse namespace through the new home. The reference parsers below are the
retired entries' argparse constructions, verbatim (do not edit -- drift here is
exactly what this gate exists to catch); self-referential metadata fields
(help/description text, provenance ``script`` values) are exempt -- only the
flag-derived values are compared. The entry modules import monai/torch at
module level (they are the production entries), so this gate is torch-marked
and runs for real in the CI full-dependency tier (ADR-0015 §6) -- it must
never be skipped around the torch mark itself.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from ctmr import cli  # noqa: E402
from ctmr.application.generation.cross_modal import baseline, candidate, monitor  # noqa: E402
from ctmr.application.generation.launcher import num_gpus_of  # noqa: E402
from ctmr.application.train_cli import TrainCli  # noqa: E402

pytestmark = pytest.mark.torch

# ------------------------------------------------------------- old argv sets

FINETUNE_ARGV = ["-e", "run/environment.json", "-c", "configs/config_brats_p3_train.json", "-t", "configs/config_network_p3.json", "-g", "7", "--data-list", "runs/p3/x/p3_pairs.json"]
FINETUNE_ARGV_AMP = FINETUNE_ARGV + ["--no_amp", "--amp_dtype", "fp16"]

DEV_EVAL_REFERENCE_ARGV = ["reference", "--dev-list", "lists/dev.json", "--raw-root", "/phase/raw", "--eval-root", "/phase/dev", "--score-workers", "8"]
DEV_EVAL_WATCH_ARGV = [
    "watch",
    "--ckpt-dir", "/phase/ckpt",
    "--eval-root", "/phase/dev",
    "--dev-list", "lists/dev.json",
    "--raw-root", "/phase/raw",
    "--phase-root", "/phase",
    "-e", "env.json",
    "-c", "config.json",
    "-t", "net.json",
    "--eval-every", "5", "--patience", "3", "--min-epoch", "30", "--max-epoch", "100",
    "--poll-seconds", "30.0", "--score-workers", "8", "--idle-exit-seconds", "120",
]
DEV_EVAL_SELECT_ARGV = ["select", "--eval-root", "/phase/dev", "--ckpt-dir", "/phase/ckpt", "--out", "/phase/select.json"]

BASELINE_ARGV = [
    "--run", "runs/p3-stage0/run.json",
    "--manifest", "/ctrl/phase/phase_manifest.json",
    "--out-root", "/ctrl/p3/stage0_holdout",
    "--raw-root", "/ctrl/phase/raw",
    "-e", "env.json", "-c", "config.json", "-t", "net.json",
    "--infer-config", "config_p3_stage0_infer.json",
    "--side", "holdout", "--shard", "0", "--num-shards", "8",
    "--limit", "2", "--challenge", "GLI", "--only-cases", "BraTS-GLI-0001-000", "BraTS-GLI-0002-000",
]
CANDIDATE_ARGV = [
    "--run", "runs/p3-candidate/run.json",
    "--manifest", "/ctrl/phase/phase_manifest.json",
    "--out-root", "/ctrl/p3/candidate_holdout",
    "--raw-root", "/ctrl/phase/raw",
    "--stage0-pairs", "/ctrl/p3/stage0_holdout/pairs.json",
    "-e", "env.json", "-c", "config.json", "-t", "net.json",
    "--infer-config", "config_p3_controlnet_infer.json",
    "--side", "holdout", "--shard", "0", "--num-shards", "8",
    "--limit", "2", "--challenge", "GLI", "--only-cases", "BraTS-GLI-0001-000",
]

# ------------------------------------------------- reference parsers (verbatim)


def _reference_finetune_parser():
    """The retired cross-modal finetune entry's argparse surface, verbatim."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--data-list", default=None, help="p3_pairs.json (defaults to env json_data_list)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    return parser


def _reference_dev_eval_parser():
    """The retired dev-eval entry's argparse surface, verbatim."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="pre-resample all dev real targets onto the generation grid")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--score-workers", type=int, default=32)

    p = sub.add_parser("watch", help="sidecar loop: evaluate epoch checkpoints as they land")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--phase-root", required=True, help="phase root holding embeddings/labels (src-image latents)")
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--score-workers", type=int, default=32, help="parallel CPU workers for reference resampling + PSNR/SSIM")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser


def _add_generation_flags(parser, with_stage0_pairs):
    """The shared baseline/candidate generate flag block, verbatim."""
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--infer-config", required=True)
    parser.add_argument("--side", default="holdout", choices=("dev", "holdout"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--challenge", default=None)
    parser.add_argument("--only-cases", nargs="*", default=None)
    if with_stage0_pairs:
        parser.add_argument("--stage0-pairs", required=True)
    return parser


# --------------------------------------------------------------- the gates

TRAIN_MIGRATED = TrainCli("description", stage="p3").parse
ENTRY_TABLE = {
    "train": (FINETUNE_ARGV + ["--no_amp", "--amp_dtype", "fp16"], _reference_finetune_parser, TRAIN_MIGRATED),
    "dev-eval reference": (DEV_EVAL_REFERENCE_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "dev-eval watch": (DEV_EVAL_WATCH_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "dev-eval select": (DEV_EVAL_SELECT_ARGV, _reference_dev_eval_parser, monitor.parse_args),
    "generate baseline": (BASELINE_ARGV, lambda: _add_generation_flags(argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter), False), baseline.parse_args),
    "generate candidate": (CANDIDATE_ARGV, lambda: _add_generation_flags(argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter), True), candidate.parse_args),
}

CLI_PREFIXES = {
    "train": ["gen", "cross-modal", "train"],
    "dev-eval reference": ["generate", "cross-modal", "dev-eval"],
    "dev-eval watch": ["generate", "cross-modal", "dev-eval"],
    "dev-eval select": ["generate", "cross-modal", "dev-eval"],
    "generate baseline": ["generate", "cross-modal", "generate", "baseline"],
    "generate candidate": ["generate", "cross-modal", "generate", "candidate"],
}


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_cli_forwards_the_entry_argv_verbatim(name):
    argv, _, _ = ENTRY_TABLE[name]
    peeled = cli.CtmrCli._peel_generate([*CLI_PREFIXES[name], *argv])
    assert peeled is not None
    rest = peeled[1] if len(peeled) == 2 else peeled[2]
    assert list(rest) == argv


@pytest.mark.parametrize("name", ENTRY_TABLE)
def test_entry_namespace_is_unchanged_against_the_retired_parsers(name):
    argv, reference_builder, migrated = ENTRY_TABLE[name]
    reference = reference_builder().parse_args(argv)
    assert vars(migrated(argv)) == vars(reference), f"{name}: migrated namespace drifted"


def test_train_cli_derives_num_gpus_from_the_entry_argv():
    assert num_gpus_of(FINETUNE_ARGV) == 7
    assert num_gpus_of(["ctmr", "x", "-e", "e.json", "-c", "c.json", "-t", "t.json", "-g", "2"]) == 2
    assert num_gpus_of(["-e", "e.json"]) == 1  # the TrainCli default


def test_cross_modal_generate_variant_is_not_consumed_as_entry_argv():
    """The ``generate baseline|candidate`` variant stays in the CLI namespace."""
    peeled = cli.CtmrCli._peel_generate(["generate", "cross-modal", "generate", "candidate", *CANDIDATE_ARGV])
    assert peeled[0] is cli.CtmrCli._run_cross_modal_generate
    assert peeled[1] == "candidate"
    assert list(peeled[2]) == CANDIDATE_ARGV
