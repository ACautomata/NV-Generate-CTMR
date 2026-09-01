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

"""torchrun derivation for the generation-family train entries (ADR-0015 §3, ticket 08).

``ctmr generate … train`` arrives WITHOUT torchrun (plain ``ctmr`` process);
the application layer derives the ``torchrun --nproc_per_node=<num_gpus> -m
<module> <rest-argv>`` child so users never hand-write launcher details. When
WORLD_SIZE is already present (the process IS a torchrun worker), the CLI calls
the train module directly instead. The command is machine-checked by tests
(spawn precedent #123: no fork in the launcher path).
"""

from __future__ import annotations

import subprocess


class TorchrunLauncher:
    """Builds and runs a torchrun child for one train module (same argv namespace)."""

    def __init__(self, module, argv, num_gpus, stdout=None, stderr=None):
        self._module = module
        self._argv = list(argv)
        self._num_gpus = int(num_gpus)
        self._stdout = stdout
        self._stderr = stderr

    def command(self):
        """``torchrun --nproc_per_node N --nnodes 1 -m <module> <argv...>`` (spawn style: no shell)."""
        return ["torchrun", "--nproc_per_node", str(self._num_gpus), "--nnodes", "1", "-m", self._module, *self._argv]

    def run(self):
        """Derive the child and relay its exit code; the return value is the trainer's."""
        proc = subprocess.run(self.command(), stdout=self._stdout, stderr=self._stderr, check=False)
        return proc.returncode


def num_gpus_of(argv):
    """The ``-g/--num_gpus`` value from the train argv (default 8 = the whole
    node, the TrainCli default since issue #278 / ADR-0019 §4)."""
    for flag in ("-g", "--num_gpus"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                return int(argv[index + 1])
    return 8
