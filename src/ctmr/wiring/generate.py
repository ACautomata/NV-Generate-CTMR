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

"""The generate family's assembly (ADR-0019 §2, issues #270/#272/#273).

The train verbs' runtime topology: ``ctmr generate <case> train`` arrives
WITHOUT torchrun and derives the ``torchrun --nproc_per_node=<num_gpus> -m
<module> <rest-argv>`` child; with ``WORLD_SIZE`` already set the process IS
the torchrun worker and runs the train entry in-process. Both branches ride
this one assembly -- the CLI face and the torchrun worker entry reuse it --
so the spawn topology has exactly one home, the composition root. The
collaborator classes (``TorchrunLauncher`` / ``num_gpus_of``, stdlib-light)
are imported at module top and called exactly as the interface layer used
to; the train modules themselves load lazily on dispatch (they are the
production torch entries).

The per-case port assemblies land with the family migration tickets: the
modality-label (#272) and mask (#273) families' are here -- the engine
adapter, the distributed session + logger, the gradient executor chosen by
the amp declaration, and the modality-label MONAI-checkpoint archive behind
the ``CheckpointRepository`` load face (ADR-0019 §2: concrete knowledge
settles nowhere else; §3: the family entries consume only domain ports). The
torchrun worker entry reuses the same assembly: the family ``main`` imports
it from here, so the worker process assembles through the composition root
too. cross-modal follows with #274.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

if TYPE_CHECKING:
    from argparse import Namespace

    import torch

    from ctmr.domain.checkpoints import CheckpointRepository
    from ctmr.domain.engine import GenerationEngine
    from ctmr.domain.generation import BypassMounting, GradientExecutor
    from ctmr.domain.logging import Logger


class TrainDispatch:
    """The one assembly behind every ``ctmr generate <case> train`` dispatch
    (ADR-0019 §2): outside torchrun it derives the torchrun child; already a
    worker, it runs the train entry in-process. Either way the return value
    is the trainer's exit code, and the child runs with the same argv
    namespace (spawn precedent #123: no fork)."""

    def __init__(self, module, argv):
        self._module = module
        self._argv = list(argv)

    def run(self):
        """Derive the topology and relay the trainer's exit code."""
        if os.environ.get("WORLD_SIZE"):
            return importlib.import_module(self._module).main(self._argv)
        return TorchrunLauncher(self._module, self._argv, num_gpus_of(self._argv)).run()


class MonaiCheckpointArchive:
    """MONAI-pickled training checkpoints behind the CheckpointRepository load
    face (ADR-0019 §3, #272).

    The P1 base checkpoint pickles MONAI meta-tensor globals: the allowlisted
    weights_only realization (``MonaiCheckpoint``) is mounted here in the
    composition root and reaches the family only as the domain port."""

    def __init__(self, device):
        self._device = device

    def load(self, path):
        bypass_mounting = importlib.import_module("ctmr.infrastructure.bypass_mounting")
        return bypass_mounting.MonaiCheckpoint(path, self._device).load()


@dataclass
class ModalityLabelTrainSession:
    """The assembled modality-label train runtime (ADR-0019 §2, #272): the
    port set the family entry consumes, constructed nowhere else. ``merged``
    is the parsed config namespace -- resolution happens inside the assembly,
    before the distributed group forms, so a bad config fails on every rank
    ahead of any collective (the pre-migration ordering)."""

    local_rank: int
    device: torch.device
    logger: Logger
    engine: GenerationEngine
    gradient_executor: GradientExecutor
    base_checkpoints: CheckpointRepository
    merged: Namespace


@dataclass
class MaskTrainSession:
    """The assembled mask train runtime (ADR-0019 §2, #273): the port set
    the family entry consumes, constructed nowhere else. ``merged`` is the
    parsed config namespace -- resolution happens inside the assembly, before
    the distributed group forms, so a bad config fails on every rank ahead of
    any collective (the pre-migration ordering). The bypass mounting is the
    domain port the kernel composes the entities from."""

    local_rank: int
    device: torch.device
    logger: Logger
    engine: GenerationEngine
    gradient_executor: GradientExecutor
    mounting: BypassMounting
    merged: Namespace


def modality_label_engine():
    """The modality-label family's GenerationEngine assembly (ADR-0019 §2, #272)."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()


def mask_engine():
    """The mask family's GenerationEngine assembly (ADR-0019 §2, #273)."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()


def modality_label_train_session(args, engine=None):
    """The modality-label train assembly (ADR-0019 §2, #272): the config
    resolution (strictly before the distributed bootstrap -- a malformed
    config must fail on every rank ahead of any collective), the session
    bootstrap, the logger, the gradient executor chosen by the amp
    declaration, and the base-checkpoint archive."""
    engine = engine if engine is not None else modality_label_engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    setting = importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting")
    executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
    local_rank, _world, device = setting.initialize_distributed(args.num_gpus)
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = executors.Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = executors.Bf16GradientExecutor()
    else:
        gradient_executor = executors.PlainGradientExecutor()
    return ModalityLabelTrainSession(
        local_rank=local_rank,
        device=device,
        logger=setting.setup_logging("modality-label-finetune"),
        engine=engine,
        gradient_executor=gradient_executor,
        base_checkpoints=MonaiCheckpointArchive(device),
        merged=merged,
    )


def mask_train_session(args, engine=None):
    """The mask train assembly (ADR-0019 §2, #273): the config resolution
    (strictly before the distributed bootstrap -- a malformed config must
    fail on every rank ahead of any collective), the session bootstrap, the
    logger, the gradient executor chosen by the amp declaration, and the
    bypass mounting the kernel composes the domain entities from."""
    engine = engine if engine is not None else mask_engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    setting = importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting")
    executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
    mounting = importlib.import_module("ctmr.infrastructure.bypass_mounting")
    local_rank, _world, device = setting.initialize_distributed(args.num_gpus)
    logger = setting.setup_logging("mask-finetune")
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = executors.Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = executors.Bf16GradientExecutor()
    else:
        gradient_executor = executors.PlainGradientExecutor()
    return MaskTrainSession(
        local_rank=local_rank,
        device=device,
        logger=logger,
        engine=engine,
        gradient_executor=gradient_executor,
        mounting=mounting.BypassMounting(merged, device, logger),
        merged=merged,
    )
