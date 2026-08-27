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

"""dm_source.json ledger port (ADR-0015 §4, issue #135).

``DmSourceLedger`` is the single read/write port for the DM-source ledger: the
registered P1-DM that P2/P3 bypasses may hang off (issue #58). Registering is
the freeze of a final-acceptance-passing P1 candidate's DM identity, configs and
provenance. Replacement is explicit: a later P1 candidate that passes final
acceptance supersedes the previous source, and every bypass pinned to the
superseded DM fails verification with a mismatch -- a retrained DM never
silently keeps old bypasses comparable (spec #51 user story 22 / CONTEXT.md
产物链). Ledger rules are pure reads/writes plus the DM-source identity gates
over the record root -- the wider run contract judgement chain (run records,
layer attachments, final acceptance) stays in the application layer. The
identity inputs of the gates are ``domain.WeightsRef``: the checkpoint byte
identity the ledger names, never a path.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ctmr.domain.identity import WeightsRef

DM_SOURCE_SCHEMA = "brats-dm-source/1"


class DmSourceViolationError(Exception):
    """A dm_source ledger rule violation (schema mismatch, unregistered or superseded source)."""


class DmSourceLedger:
    """The single registered P1-DM source that P2/P3 bypasses may hang off (issue #58)."""

    def __init__(self, record_root):
        self._root = Path(record_root)

    def path(self):
        return self._root / "dm_source.json"

    def _now_utc(self):
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def current(self):
        ledger_path = self.path()
        if not ledger_path.is_file():
            return None
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("schema") != DM_SOURCE_SCHEMA:
            raise DmSourceViolationError(f"dm_source ledger {ledger_path} has schema {ledger.get('schema')!r} != {DM_SOURCE_SCHEMA!r}")
        return ledger

    def register(self, record, run_record_path):
        """Freezes the passing P1 candidate as the current DM source (superseding any previous)."""
        if record["phase"] != "P1":
            raise DmSourceViolationError("only a P1 candidate can be registered as the DM source (P2/P3 are bypasses, not sources)")
        current = self.current()
        if current is not None and current["run_id"] == record["run_id"]:
            return current  # idempotent re-register of the same candidate
        entry = {
            "schema": DM_SOURCE_SCHEMA,
            "run_id": record["run_id"],
            "run_record": str(Path(run_record_path).resolve()),
            "run_record_sha256": self._file_sha256(run_record_path),
            "checkpoint": record["selection"]["checkpoint"],
            "configs": record["configs"],
            "manifest": record["manifest"],
            "base_ckpt": record.get("base_ckpt"),
            "code_version": record.get("code_version"),
            "registered_utc": self._now_utc(),
            "superseded_run_id": None,
        }
        if current is not None:
            entry["superseded_run_id"] = current["run_id"]
        self.path().write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return entry

    def check_upstream(self, upstream_run_id, checkpoint: WeightsRef):
        """Init-time gate: a P2/P3 bypass may only pin the registered DM source."""
        current = self.current()
        if current is None:
            raise DmSourceViolationError(
                "no P1 candidate has passed final acceptance yet; P2/P3 must hang off the registered DM source (conclude a passing P1 run first)"
            )
        if upstream_run_id != current["run_id"] or checkpoint != WeightsRef(sha256=current["checkpoint"]["sha256"]):
            raise DmSourceViolationError(
                f"upstream run {upstream_run_id} is not the registered DM source {current['run_id']}; "
                "P2/P3 may only hang off the final-acceptance-passing P1-DM"
            )

    def check_record(self, record):
        """Verify-time mismatch detection against the current DM source."""
        current = self.current()
        if current is None:
            return []
        current_ref = WeightsRef(sha256=current["checkpoint"]["sha256"])
        if record.get("phase") == "P1":
            if (
                record["run_id"] == current["run_id"]
                and WeightsRef(sha256=record.get("selection", {}).get("checkpoint", {}).get("sha256")) != current_ref
            ):
                return ["registered DM source checkpoint no longer matches its P1 run record"]
            return []
        upstream = record.get("upstream")
        if upstream and (WeightsRef(sha256=upstream["checkpoint"]["sha256"]) != current_ref or upstream["run_id"] != current["run_id"]):
            return [
                f"DM was retrained: this bypass is pinned to superseded DM {upstream['run_id']} "
                f"while the registered DM source is {current['run_id']}"
            ]
        return []
