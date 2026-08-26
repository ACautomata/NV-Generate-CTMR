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

"""``ctmr.harness`` -- phase-script shells, the single definition (ADR-0011, #111).

The mechanical skeletons shared by every stage's finetune entry live here:
``cli`` is the public argparse set + torchrun WORLD_SIZE check; ``train_shell``
is the PhaseHarness epoch loop (early-stop file polling at epoch boundaries and
mid-epoch, DDP/amp mechanics, loss all_reduce, atomic checkpoint publishing +
latest.json) driven by an injected ``PhaseTrainKernel`` Protocol (composition,
never implementation inheritance), with the recipe guard as a first-class hook.
Stage kernels stay in their thin script entries -- the shell holds no recipe
values and no domain decisions.

Import policy: ``cli`` is stdlib-only (any machine); ``train_shell`` needs
torch -- import submodules directly. The pinned-recipe guards moved to
``ctmr.domain.recipe`` in #133 (thin re-export kept in ``recipe``).
"""
