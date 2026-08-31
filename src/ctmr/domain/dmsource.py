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

"""DM-source ledger domain face: the rules and their violation, IO-free (ADR-0019 §3, #269).

The ledger holds the one registered P1-DM that P2/P3 bypasses may hang off
(issue #58, CONTEXT.md "DM source"). Its rules live here since #269 -- only a
final-acceptance-passing P1 candidate may register; a bypass may pin only the
registered source; a retrained DM mismatches every bypass pinned to the
superseded one -- and the rules are pure decisions over an injected
``DmSourceEntryStore``: the json read/write, the run-record file digest and
the wall clock stay on the adapter side (infrastructure ``dmsource`` mounts
them as the ``DmSourceLedger`` face over ``dm_source.json``; domain owns no
IO, ADR-0015 §2).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from ctmr.domain.identity import WeightsRef

DM_SOURCE_SCHEMA = "brats-dm-source/1"


class DmSourceViolationError(Exception):
    """A dm_source ledger rule violation (schema mismatch, unregistered or superseded source)."""


@runtime_checkable
class DmSourceEntryStore(Protocol):
    """The entry-store seam the ledger rules ride on: one current entry or none.

    ``path`` is the store's location description (raised verbatim inside rule
    violation messages); ``read`` returns the raw current entry or ``None``
    when nothing is registered; ``write`` replaces the entry.
    """

    @property
    def path(self) -> Path: ...

    def read(self) -> dict | None: ...

    def write(self, entry: dict) -> None: ...


@runtime_checkable
class DmSourceLedger(Protocol):
    """The DM-source ledger domain face (the port the application injects).

    ``register`` freezes a passing P1 candidate as the current source,
    superseding any previous one; ``check_upstream`` is the init-time gate
    (a bypass may only pin the registered source); ``check_record`` is the
    verify-time mismatch detection against the current source.
    """

    def current(self) -> dict | None: ...

    def register(self, record: dict, run_record_path: str) -> dict: ...

    def check_upstream(self, upstream_run_id: str, checkpoint: WeightsRef) -> None: ...

    def check_record(self, record: dict) -> list[str]: ...


class DmSourceLedgerRules:
    """The ledger rules as pure decisions over an entry store.

    ``run_record_ref`` folds the run-record file into a ``WeightsRef`` and
    ``now_utc`` stamps the registration time -- both are injected callables
    (file digest and clock are the adapter's IO); the rules call them only
    for a real registration, keeping the idempotent re-register IO-free.
    """

    def __init__(self, store: DmSourceEntryStore, run_record_ref: Callable[[str], WeightsRef], now_utc: Callable[[], str]):
        self._store = store
        self._run_record_ref = run_record_ref
        self._now_utc = now_utc

    def current(self):
        entry = self._store.read()
        if entry is not None and entry.get("schema") != DM_SOURCE_SCHEMA:
            raise DmSourceViolationError(f"dm_source ledger {self._store.path} has schema {entry.get('schema')!r} != {DM_SOURCE_SCHEMA!r}")
        return entry

    def register(self, record, run_record_path):
        """Freezes the passing P1 candidate as the current DM source (superseding any previous).

        ``run_record_path`` is the resolved record path to pin in the entry.
        """
        if record["phase"] != "P1":
            raise DmSourceViolationError("only a P1 candidate can be registered as the DM source (P2/P3 are bypasses, not sources)")
        current = self.current()
        if current is not None and current["run_id"] == record["run_id"]:
            return current  # idempotent re-register of the same candidate
        entry = {
            "schema": DM_SOURCE_SCHEMA,
            "run_id": record["run_id"],
            "run_record": run_record_path,
            "run_record_sha256": self._run_record_ref(run_record_path).sha256,
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
        self._store.write(entry)
        return entry

    def check_upstream(self, upstream_run_id: str, checkpoint: WeightsRef):
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
                f"DM was retrained: this bypass is pinned to superseded DM {upstream['run_id']} while the registered DM source is {current['run_id']}"
            ]
        return []
