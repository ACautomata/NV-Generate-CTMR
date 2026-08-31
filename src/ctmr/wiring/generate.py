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
production torch entries). The per-case port assemblies (gradient executors,
checkpoint repository, engine loading, logging) land with the family
migration tickets #272-#274.
"""

from __future__ import annotations

import importlib
import os

from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of


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
