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

"""The generate family's assembly (ADR-0019 §2, issues #270/#273).

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
mask family (#273) is here -- ``mask_train_runtime`` hoists every concrete
construction the mask train entry used to make (the engine config merge with
the CLI flags patched in, the distributed session bootstrap, the run logger,
the amp-selected precision executor, the bypass mounting) and
``generation_engine`` is the lazy adapter lookup behind the sampling and
monitoring faces; modality-label (#272) and cross-modal (#274) follow.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

if TYPE_CHECKING:  # port types for the runtime record's annotations only --
    # the runtime imports stay lazy, the composition root stays stdlib-light
    from ctmr.domain.generation.mounting import BypassMounting
    from ctmr.domain.generation.update import GradientExecutor
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


def generation_engine():
    """The ``GenerationEngine`` adapter behind the generate families' config,
    model-loading and inference faces (ADR-0019 §2): one lazy lookup, so
    importing the composition root stays stdlib-light."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()


@dataclass
class MaskTrainRuntime:
    """What the mask train assembly hands the trainer entry (ADR-0019 §2, #273).

    The merged config namespace, the distributed session (local rank +
    device), the run logger, the injected runtime precision strategy and the
    bypass mounting -- every concrete construction the entry used to make
    itself, hoisted here so the application entry sees ports only.
    """

    merged: Any
    local_rank: int
    device: Any
    logger: Logger
    gradient_executor: GradientExecutor
    mounting: BypassMounting


def mask_train_runtime(args) -> MaskTrainRuntime:
    """Assemble the mask train runtime (ADR-0019 §2, #273): the engine
    config merge with the CLI flags patched in, the modality mapping read,
    the distributed session bootstrap, the run logger, the amp-flag-selected
    precision executor and the bypass mounting. The adapters load lazily on
    dispatch (the ``cli.py`` discipline)."""
    setting = importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting")
    executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
    mounting = importlib.import_module("ctmr.infrastructure.bypass_mounting")

    merged = generation_engine().load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path
    with open(merged.modality_mapping_path) as handle:
        merged.modality_mapping = json.load(handle)

    local_rank, _world, device = setting.initialize_distributed(args.num_gpus)
    logger = setting.setup_logging("mask-finetune")
    # The runtime precision strategy (ADR-0016): fp16 (scaler), bf16 (DCU
    # default) or non-AMP plain execution -- selected here, typed to the
    # application as the domain GradientExecutor port.
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = executors.Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = executors.Bf16GradientExecutor()
    else:
        gradient_executor = executors.PlainGradientExecutor()
    return MaskTrainRuntime(
        merged=merged,
        local_rank=local_rank,
        device=device,
        logger=logger,
        gradient_executor=gradient_executor,
        mounting=mounting.BypassMounting(merged, device, logger),
    )
