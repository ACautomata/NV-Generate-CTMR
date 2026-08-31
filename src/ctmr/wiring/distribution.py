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

"""The distribution family's process-entry assembly (ADR-0019 §2, issue #275).

Three controlled ``python -m`` jobs -- the judge-chain closing verifier, the
instrument trainer and the diagnostic job C -- have no CLI verb route, so
their composition root is this module: the concrete-adapter knowledge their
process entries assemble settles here and nowhere else, composed lazily (the
``cli.py`` dispatch discipline -- torch / monai / nnunetv2 load only when the
job actually runs). The jobs themselves depend only on the domain ports
(``InstrumentCheckpointReader``, ``GenerationEngine``).
"""

import importlib


def instrument_checkpoint_reader():
    """The checkpoint-reader adapter behind the closing verifier and the instrument trainer."""
    return importlib.import_module("ctmr.infrastructure.nnunet_runner").InstrumentCheckpointReader()


def intensity_domain_engine():
    """The GenerationEngine adapter behind diagnostic job C's VAE arms."""
    return importlib.import_module("ctmr.infrastructure.engine").MaisiEngine()
