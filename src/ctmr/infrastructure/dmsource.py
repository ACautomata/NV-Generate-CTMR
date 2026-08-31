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

"""dm_source.json ledger adapter (ADR-0015 §4, ADR-0019 §3, issues #135/#269).

The ledger's rules and violation live in ``ctmr.domain.dmsource`` (the domain
face); this module mounts them onto ``dm_source.json``: ``JsonDmSourceStore``
is the entry-store adapter (json read/write) and ``DmSourceLedger`` composes
it with the domain rules plus the two impurities -- the run-record file
digest (``weights_ref_of_file``, the single file-read point of checkpoint
identity) and the wall clock. The re-exports keep the historical import face
(``from ctmr.infrastructure.dmsource import DmSourceViolationError``) spelling
the domain types while the application contract family still imports from
here (the ratchet pins those edges until the family migration, #271).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from ctmr.domain.dmsource import DM_SOURCE_SCHEMA, DmSourceLedgerRules, DmSourceViolationError
from ctmr.domain.identity import WeightsRef
from ctmr.infrastructure.weightsref import weights_ref_of_file

__all__ = ["DM_SOURCE_SCHEMA", "DmSourceLedger", "DmSourceViolationError", "JsonDmSourceStore", "WeightsRef"]


class JsonDmSourceStore:
    """dm_source.json read/write (the ``DmSourceEntryStore`` adapter)."""

    def __init__(self, record_root):
        self._root = Path(record_root)

    @property
    def path(self):
        return self._root / "dm_source.json"

    def read(self):
        if not self.path.is_file():
            return None
        return json.loads(self.path.read_text())

    def write(self, entry):
        self.path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")


class DmSourceLedger:
    """The json-backed DM-source ledger face: domain rules over dm_source.json (issue #58)."""

    def __init__(self, record_root):
        self._store = JsonDmSourceStore(record_root)
        self._rules = DmSourceLedgerRules(self._store, run_record_ref=weights_ref_of_file, now_utc=self._now_utc)

    def path(self):
        return self._store.path

    def _now_utc(self):
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def current(self):
        return self._rules.current()

    def register(self, record, run_record_path):
        """Freezes the passing P1 candidate as the current DM source (superseding any previous)."""
        return self._rules.register(record, str(Path(run_record_path).resolve()))

    def check_upstream(self, upstream_run_id, checkpoint: WeightsRef):
        """Init-time gate: a P2/P3 bypass may only pin the registered DM source."""
        self._rules.check_upstream(upstream_run_id, checkpoint)

    def check_record(self, record):
        """Verify-time mismatch detection against the current DM source."""
        return self._rules.check_record(record)
