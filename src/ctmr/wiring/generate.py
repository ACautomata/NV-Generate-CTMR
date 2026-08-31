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

"""The generate family's assembly (ADR-0019 §2, issue #270).

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

The per-case port assemblies land with the family migration tickets
(#272-#274): the concrete collaborators the application entries stopped
constructing -- the frozen engine adapter behind the ``GenerationEngine``
port, the distributed session bootstrap, the run logger, the ControlNet
mounting, the pinned precision executor and the checkpoint file identity --
settle in :class:`GenerateRuntime`, resolved lazily on dispatch (importing
the composition root still pulls no third-party dependency). The family
entries reach it directly (the torchrun worker entry reuses the same
assembly, ADR-0019 §2), so the concrete knowledge stays here and nowhere
else.
"""

from __future__ import annotations

import importlib
import os

from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of


class GenerateRuntime:
    """The generate family's production runtime (ADR-0019 §2, #272-#274).

    One method per collaborator the application entries receive at their
    seams; each resolves its concrete adapter lazily (importlib on dispatch)
    so the frozen maisi_engine functions, the precision executors, the
    ControlNet mounting, the distributed session and the run logger load only
    when a verb actually runs. The application entries see the domain port
    faces (``GenerationEngine`` / ``Logger`` / ``GradientExecutor``) and the
    injected collaborators -- never the concrete addresses.
    """

    def engine(self):
        """The frozen maisi_engine adapter behind the GenerationEngine port."""
        return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()

    def logger(self, name):
        """The run logger (the Logger port's stdlib realization)."""
        return importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting").setup_logging(name)

    def train_session(self, args):
        """Bootstrap the distributed training session: (local_rank, world_size, device)."""
        return importlib.import_module("ctmr.infrastructure.maisi_engine.diff_model_setting").initialize_distributed(args.num_gpus)

    def bypass_mounting(self, args, device, logger):
        """The ControlNet-only hook-up collaborator the bypass kernels drive."""
        return importlib.import_module("ctmr.infrastructure.bypass_mounting").BypassMounting(args, device, logger)

    def gradient_executor(self, amp, amp_dtype):
        """The pinned precision strategy (ADR-0016): fp16 (scaler), bf16 (DCU default) or plain."""
        executors = importlib.import_module("ctmr.infrastructure.gradient_executors")
        if amp and amp_dtype == "fp16":
            return executors.Fp16GradientExecutor()
        if amp:
            return executors.Bf16GradientExecutor()
        return executors.PlainGradientExecutor()

    def weights_ref_of_file(self):
        """The checkpoint file identity callable (path -> domain WeightsRef)."""
        return importlib.import_module("ctmr.infrastructure.weightsref").weights_ref_of_file


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
