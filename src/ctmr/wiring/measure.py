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

"""The measure family's assembly (ADR-0019 §2, issue #270).

``ctmr measure predict`` is the canonical frozen-instrument execution entry
(ADR-0009 decision 3): the knowledge that the native nnUNetv2 predictor
adapter (``ctmr.infrastructure.nnunet_runner``) stands behind this verb
settles here -- the interface layer routes the row to this module and spells
no infrastructure address. The composition is a passthrough, exactly as the
registry used to compose it: everything after the verb reaches the nnUNetv2
predictor parser untouched, and the adapter loads lazily on dispatch (it is
an nnunetv2/torch surface).
"""

from __future__ import annotations

import importlib


def main(pass_through=None):
    """Run the frozen-instrument predict verb through the nnUNetv2 adapter; the exit code is relayed verbatim."""
    return importlib.import_module("ctmr.infrastructure.nnunet_runner").main(pass_through)
