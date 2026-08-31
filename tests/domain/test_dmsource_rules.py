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

"""Contract tests for the DM-source ledger domain face (ADR-0019 §3, #269).

The rules (only a final-acceptance-passing P1 candidate registers; bypasses
hang only off the registered source; a retrained DM mismatches every bypass
pinned to the superseded one) live in ``ctmr.domain.dmsource`` and are driven
here by an in-memory fake entry store -- no filesystem, no json. The json
read/write adapter itself is gated by tests/infrastructure/test_dmsource.py
over the real ``DmSourceLedger`` facade, which must satisfy the same port.
"""

import re

import pytest

from ctmr.domain.dmsource import (
    DM_SOURCE_SCHEMA,
    DmSourceEntryStore,
    DmSourceLedger,
    DmSourceLedgerRules,
    DmSourceViolationError,
)
from ctmr.domain.identity import WeightsRef


class FakeEntryStore:
    """In-memory DmSourceEntryStore: one current entry or none, calls counted."""

    def __init__(self, entry=None):
        self.path = "memory://dm_source.json"
        self._entry = entry
        self.reads = 0
        self.writes = []

    def read(self):
        self.reads += 1
        return self._entry

    def write(self, entry):
        self.writes.append(entry)
        self._entry = entry


class Record:
    """Minimal valid phase-run record shape the ledger rules read."""

    def __init__(self, run_id, phase="P1", checkpoint_sha="1" * 64):
        self.record = {
            "schema": "brats-phase-run/1",
            "run_id": run_id,
            "phase": phase,
            "selection": {"checkpoint": {"path": "/ckpt.pt", "sha256": checkpoint_sha, "epoch": 5}},
            "configs": {"env_config": "env.json"},
            "manifest": "manifest.json",
            "base_ckpt": "/base.pt",
            "code_version": {"commit": "0" * 40},
        }

    def __getitem__(self, key):
        return self.record[key]

    def get(self, key, default=None):
        return self.record.get(key, default)


def fake_run_record_ref(path):
    """A deterministic stand-in for the file-digest injection."""
    return WeightsRef(sha256=f"run-record-{path}")


def fake_clock():
    return "2026-08-31T00:00:00Z"


def rules_with(store):
    return DmSourceLedgerRules(store, run_record_ref=fake_run_record_ref, now_utc=fake_clock)


# ---------------------------------------------------------------- current


def test_current_is_none_without_a_registered_source():
    assert rules_with(FakeEntryStore()).current() is None


def test_current_rejects_a_foreign_schema():
    store = FakeEntryStore(entry={"schema": "not-the-schema"})
    with pytest.raises(DmSourceViolationError, match=f"{store.path} has schema 'not-the-schema' != '{DM_SOURCE_SCHEMA}'"):
        rules_with(store).current()


def test_current_accepts_the_pinned_schema():
    entry = {"schema": DM_SOURCE_SCHEMA, "run_id": "p1"}
    assert rules_with(FakeEntryStore(entry=entry)).current() == entry


# ---------------------------------------------------------------- register


def test_register_rejects_a_non_p1_candidate():
    with pytest.raises(DmSourceViolationError, match="only a P1 candidate can be registered"):
        rules_with(FakeEntryStore()).register(Record("p2-fixture", phase="P2").record, "/runs/p2/run.json")


def test_register_freezes_the_passing_candidate_as_the_source():
    store = FakeEntryStore()
    record = Record("p1-final")
    entry = rules_with(store).register(record.record, "/runs/p1-final/run.json")

    assert store.writes == [entry]
    assert entry == {
        "schema": DM_SOURCE_SCHEMA,
        "run_id": "p1-final",
        "run_record": "/runs/p1-final/run.json",
        "run_record_sha256": "run-record-/runs/p1-final/run.json",
        "checkpoint": record.record["selection"]["checkpoint"],
        "configs": record.record["configs"],
        "manifest": record.record["manifest"],
        "base_ckpt": "/base.pt",
        "code_version": {"commit": "0" * 40},
        "registered_utc": "2026-08-31T00:00:00Z",
        "superseded_run_id": None,
    }


def test_register_is_idempotent_for_the_same_candidate():
    record = Record("p1-final")
    store = FakeEntryStore()
    rules = rules_with(store)
    first = rules.register(record.record, "/runs/p1-final/run.json")
    second = rules.register(record.record, "/runs/p1-final/run.json")

    assert second == first
    assert second["superseded_run_id"] is None
    assert len(store.writes) == 1  # the idempotent re-register writes nothing


def test_registering_a_fresh_p1_supersedes_the_current_source():
    store = FakeEntryStore()
    rules = rules_with(store)
    rules.register(Record("p1-final").record, "/runs/p1-final/run.json")
    entry = rules.register(Record("p1-retrained", checkpoint_sha="2" * 64).record, "/runs/p1-retrained/run.json")

    assert entry["run_id"] == "p1-retrained"
    assert entry["superseded_run_id"] == "p1-final"
    assert rules.current()["run_id"] == "p1-retrained"


def test_register_digests_and_stamps_the_clock_only_on_a_real_registration():
    store = FakeEntryStore()
    calls = []

    def counting_ref(path):
        calls.append(path)
        return fake_run_record_ref(path)

    rules = DmSourceLedgerRules(store, run_record_ref=counting_ref, now_utc=fake_clock)
    record = Record("p2-fixture", phase="P2")
    with pytest.raises(DmSourceViolationError):
        rules.register(record.record, "/runs/p2/run.json")
    assert calls == []  # a rejected candidate never reaches the file digest


# ---------------------------------------------------------------- check_upstream


def test_check_upstream_accepts_the_registered_source():
    rules = rules_with(FakeEntryStore())
    rules.register(Record("p1-final").record, "/runs/p1-final/run.json")
    rules.check_upstream("p1-final", WeightsRef(sha256="1" * 64))  # no raise


def test_check_upstream_rejects_when_no_source_is_registered():
    with pytest.raises(DmSourceViolationError, match="no P1 candidate has passed final acceptance yet"):
        rules_with(FakeEntryStore()).check_upstream("p1-final", WeightsRef(sha256="1" * 64))


def test_check_upstream_rejects_a_non_registered_upstream():
    rules = rules_with(FakeEntryStore())
    rules.register(Record("p1-final").record, "/runs/p1-final/run.json")
    with pytest.raises(DmSourceViolationError, match="p1-other is not the registered DM source p1-final"):
        rules.check_upstream("p1-other", WeightsRef(sha256="1" * 64))
    with pytest.raises(DmSourceViolationError, match="is not the registered DM source"):
        rules.check_upstream("p1-final", WeightsRef(sha256="3" * 64))


# ---------------------------------------------------------------- check_record


def test_check_record_flags_a_stale_bypass_after_a_retrained_dm():
    rules = rules_with(FakeEntryStore())
    rules.register(Record("p1-final").record, "/runs/p1-final/run.json")
    rules.register(Record("p1-retrained", checkpoint_sha="2" * 64).record, "/runs/p1-retrained/run.json")

    stale_p2 = {
        "phase": "P2",
        "run_id": "p2-old",
        "upstream": {"run_id": "p1-final", "checkpoint": {"sha256": "1" * 64}},
    }
    assert rules.check_record(stale_p2) == [
        "DM was retrained: this bypass is pinned to superseded DM p1-final while the registered DM source is p1-retrained"
    ]


def test_check_record_flags_a_p1_record_whose_checkpoint_changed():
    record = Record("p1-retrained", checkpoint_sha="2" * 64)
    rules = rules_with(FakeEntryStore())
    rules.register(record.record, "/runs/p1-retrained/run.json")

    tampered = dict(record.record)
    tampered["selection"] = {"checkpoint": {"sha256": "9" * 64}}
    assert rules.check_record(tampered) == ["registered DM source checkpoint no longer matches its P1 run record"]
    assert rules.check_record(record.record) == []


def test_check_record_passes_a_bypass_pinned_to_the_registered_source():
    rules = rules_with(FakeEntryStore())
    rules.register(Record("p1-final").record, "/runs/p1-final/run.json")
    fresh_p2 = {"phase": "P2", "run_id": "p2-new", "upstream": {"run_id": "p1-final", "checkpoint": {"sha256": "1" * 64}}}
    assert rules.check_record(fresh_p2) == []


def test_check_record_passes_everything_without_a_registered_source():
    assert rules_with(FakeEntryStore()).check_record({"phase": "P2", "run_id": "p2"}) == []


# ---------------------------------------------------------------- the port


def test_the_rules_and_the_violation_make_up_the_domain_face():
    assert issubclass(DmSourceViolationError, Exception)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", fake_clock())
    # the rules service and the fake entry store both satisfy the ports structurally
    assert isinstance(rules_with(FakeEntryStore()), DmSourceLedger)
    assert isinstance(FakeEntryStore(), DmSourceEntryStore)


def test_the_infrastructure_facade_satisfies_the_domain_port(tmp_path):
    from ctmr.infrastructure import dmsource as infrastructure

    assert isinstance(infrastructure.DmSourceLedger(tmp_path), DmSourceLedger)
    assert isinstance(infrastructure.JsonDmSourceStore(tmp_path), DmSourceEntryStore)
