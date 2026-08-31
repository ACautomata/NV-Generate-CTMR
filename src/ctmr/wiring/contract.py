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

"""The accept-contract family's assembly (ADR-0019 §2, issue #271).

The knowledge that the json-backed DM-source ledger
(``ctmr.infrastructure.dmsource.DmSourceLedger``) stands behind the run
contract settles here: the interface layer routes the six verbs to this
module and spells no infrastructure address, and the application verb face
receives the ``(record_root) -> DmSourceLedger`` port factory as an
injection -- the concrete adapter is this root's choice and nowhere else's.
The contract package import rides at module top (stdlib-light: the verb face
pulls no third-party stack); the adapter loads lazily on dispatch, exactly
as the other family modules compose.
"""

from __future__ import annotations

import importlib

from ctmr.application.acceptance.contract import cli as contract_cli


def main(pass_through=None):
    """Run one run-contract verb through the composition-root wiring; the exit code is relayed verbatim."""
    ledger_factory = importlib.import_module("ctmr.infrastructure.dmsource").DmSourceLedger
    return contract_cli.main(pass_through, ledger_factory=ledger_factory)
