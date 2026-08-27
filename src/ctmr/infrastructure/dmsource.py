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

"""dm_source.json ledger read/write (ADR-0015 section 4, issue #135).

``DmSourceRepository`` is the single IO mouth for the DM-source lineage ledger
at the root of the controlled record tree (CONTEXT.md "Checkpoint Identity":
the ledger is the authoritative registry of DM-source lineage), sunk from
``scripts/brats_phase_run_contract.py``'s DmSourceLedger (#58). It carries
parsed JSON both ways and pins the byte format (indent=2, sort_keys, trailing
newline); absence reads as ``None`` without creating the file. No schema
judgement happens here -- every verdict on top of the ledger (register
supersession, upstream init gates, verify-time mismatch detection) stays in
the contract layer above.

Stdlib-only.
"""

import json
from pathlib import Path


class DmSourceRepository:
    """Reads and writes the dm_source.json lineage ledger under a record root."""

    def __init__(self, record_root):
        self._root = Path(record_root)

    def path(self) -> Path:
        """The ledger file path (created only on write, never on read)."""
        return self._root / "dm_source.json"

    def read(self):
        """The parsed ledger, or ``None`` while no P1 candidate has been registered yet."""
        path = self.path()
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    def write(self, entry):
        """Persist one ledger entry (byte format pinned by tests; contract-layer callers only)."""
        path = self.path()
        path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return path
