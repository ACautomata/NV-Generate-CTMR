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

"""Common phase-train CLI surface (ADR-0011 decision 1, ADR-0015 §2, ticket 08).

The public argparse set shared by every generation-family train entry
(-e/-c/-t/-g, --no_amp/--amp_dtype) plus the torchrun WORLD_SIZE cross-check,
extracted from the three symmetric copies. Stage-specific flags (modality-label
--replay-list, cross-modal --data-list) ride in through ``stage_flags`` so the CLI face
stays verbatim: same argv must yield an equal argparse namespace before and
after the consolidation (the convergence gate in tests).

Stdlib-only: importable on any machine, no torch / monai (ADR-0013 §4).
"""

from __future__ import annotations

import argparse
import os


class TrainCli:
    """The shared finetune argument surface, parameterized by stage."""

    def __init__(self, description, stage="p2", formatter_class=argparse.RawDescriptionHelpFormatter):
        self.parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)
        self.parser.add_argument("-e", "--env_config_path", required=True)
        self.parser.add_argument("-c", "--model_config_path", required=True)
        self.parser.add_argument("-t", "--model_def_path", required=True)
        # Single-node topology (ADR-0019 §4, issue #278): per-GPU batch=1 stays
        # pinned, the whole node trains (--num_gpus 8 = the world_size default).
        self.parser.add_argument("-g", "--num_gpus", type=int, default=8)
        self._add_stage_flags(stage)
        self.parser.add_argument("--no_amp", dest="amp", action="store_false")
        self.parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")

    def _add_stage_flags(self, stage):
        # The ``p1``/``p2``/``p3`` discriminator keys date back to the retired
        # finetune entries (git history) but stay pinned: the argv↔namespace
        # equivalence gate freezes the migrated families' CLI face. New code
        # passes its family word only.
        if stage == "p1":
            self.parser.add_argument(
                "--replay-list",
                dest="replay_list",
                action="append",
                required=True,
                help="MR-RATE replay data list (spec: list-level 1:1 mix; append once per list)",
            )
            # Embedded periodic validation (ADR-0019 §5, issue #278): every N
            # epochs the trainer itself runs the sharded dev-cohort validation
            # stage; 0 disables the stage entirely.
            self.parser.add_argument(
                "--val-every",
                dest="val_every",
                type=int,
                default=10,
                help="embedded periodic validation interval in epochs (0 disables)",
            )
            self.parser.add_argument("--dev-list", dest="dev_list", default=None, help="dev cohort list json (embedded validation)")
            self.parser.add_argument("--raw-root", dest="raw_root", default=None, help="raw volume root for the dev real bank (embedded validation)")
            self.parser.add_argument(
                "--emb-root", dest="emb_root", default=None, help="phase embedding root for per-case spacing (embedded validation)"
            )
        elif stage == "p2":
            return
        elif stage == "p3":
            self.parser.add_argument("--data-list", default=None, help="p3_pairs.json (defaults to env json_data_list)")
        else:
            raise ValueError(f"unknown stage '{stage}' (expected one of ['p1', 'p2', 'p3'])")

    def parse(self, argv=None):
        """Parse argv, then enforce the torchrun WORLD_SIZE agreement."""
        args = self.parser.parse_args(argv)
        # torchrun sets WORLD_SIZE when launched via torchrun; -g must agree or
        # every worker would silently run as a world_size=1 replica on cuda:0.
        torchrun_world = int(os.environ["WORLD_SIZE"]) if os.environ.get("WORLD_SIZE") else None
        if torchrun_world is not None and torchrun_world != args.num_gpus:
            raise ValueError(f"--num_gpus {args.num_gpus} disagrees with torchrun WORLD_SIZE {torchrun_world}")
        return args
