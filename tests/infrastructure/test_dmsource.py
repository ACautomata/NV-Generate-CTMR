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

"""Behaviour gates for the dm_source.json ledger port (ADR-0015 §4, #135).

``DmSourceLedger`` is the single read/write port for the DM-source ledger: the
registered P1-DM that P2/P3 bypasses may hang off. The suite pins the register/
supersede protocol and -- the point of the port -- the mismatch verify behavior
survives verbatim: a retrained DM never silently keeps old bypasses comparable
("DM was retrained: ..."). Stdlib only.
"""

import hashlib
import json
import re

import pytest

from ctmr.domain.identity import WeightsRef
from ctmr.infrastructure.dmsource import DmSourceLedger

_SCHEMA_FIELD = "brats-dm-source/1"


class _Record:
    """Minimal valid phase-run record shape the ledger reads."""

    def __init__(self, run_id, phase="P1", checkpoint_sha="1" * 64, checkpoint_path="/ckpt.pt"):
        self.record = {
            "schema": "brats-phase-run/1",
            "run_id": run_id,
            "phase": phase,
            "selection": {"checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha, "epoch": 5}},
            "configs": {"env_config": "env.json"},
            "manifest": "manifest.json",
            "base_ckpt": "/base.pt",
            "code_version": {"commit": "0" * 40},
        }

    def __getitem__(self, key):
        return self.record[key]

    def get(self, key, default=None):
        return self.record.get(key, default)

    def write_run_record(self, root, filename="run.json"):
        path = root / "runs" / self.record["run_id"] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record, sort_keys=True))
        return path


def _register_p1(ledger, record, run_path=None, root=None):
    return ledger.register(record.record, run_path if run_path is not None else record.write_run_record(root))


def test_register_writes_the_ledger_entry_and_current_reads_it_back(tmp_path):
    record = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    entry = ledger.register(record.record, record.write_run_record(tmp_path))

    ledger_path = tmp_path / "dm_source.json"
    assert ledger_path.is_file()
    on_disk = json.loads(ledger_path.read_text())
    assert on_disk == entry
    assert on_disk["schema"] == _SCHEMA_FIELD
    assert on_disk["run_id"] == "p1-final"
    assert on_disk["checkpoint"]["sha256"] == "1" * 64
    assert on_disk["superseded_run_id"] is None
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", on_disk["registered_utc"])
    assert on_disk["run_record_sha256"] == hashlib.sha256(record.write_run_record(tmp_path).read_bytes()).hexdigest()

    assert ledger.current() == on_disk


def test_current_is_none_without_a_ledger(tmp_path):
    assert DmSourceLedger(tmp_path).current() is None


def test_current_rejects_a_foreign_schema(tmp_path):
    (tmp_path / "dm_source.json").write_text(json.dumps({"schema": "not-the-schema"}))
    with pytest.raises(Exception, match="has schema 'not-the-schema' != 'brats-dm-source/1'"):
        DmSourceLedger(tmp_path).current()


def test_register_rejects_a_non_p1_candidate(tmp_path):
    record = _Record("p2-fixture", phase="P2")
    with pytest.raises(Exception, match="only a P1 candidate can be registered"):
        DmSourceLedger(tmp_path).register(record.record, record.write_run_record(tmp_path))


def test_register_is_idempotent_for_the_same_candidate(tmp_path):
    record = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    first = ledger.register(record.record, record.write_run_record(tmp_path))
    second = ledger.register(record.record, record.write_run_record(tmp_path))
    assert second == first
    assert second["superseded_run_id"] is None


def test_registering_a_fresh_p1_supersedes_the_current_source(tmp_path):
    first = _Record("p1-final")
    second = _Record("p1-retrained", checkpoint_sha="2" * 64, checkpoint_path="/ckpt2.pt")
    ledger = DmSourceLedger(tmp_path)
    ledger.register(first.record, first.write_run_record(tmp_path))
    entry = ledger.register(second.record, second.write_run_record(tmp_path))

    assert entry["run_id"] == "p1-retrained"
    assert entry["superseded_run_id"] == "p1-final"
    assert ledger.current()["run_id"] == "p1-retrained"


def test_check_upstream_accepts_the_registered_source(tmp_path):
    record = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    ledger.register(record.record, record.write_run_record(tmp_path))
    ledger.check_upstream("p1-final", WeightsRef(sha256="1" * 64))  # no raise


def test_check_upstream_rejects_when_no_source_is_registered(tmp_path):
    with pytest.raises(Exception, match="no P1 candidate has passed final acceptance yet"):
        DmSourceLedger(tmp_path).check_upstream("p1-final", WeightsRef(sha256="1" * 64))


def test_check_upstream_rejects_a_non_registered_upstream(tmp_path):
    record = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    ledger.register(record.record, record.write_run_record(tmp_path))
    with pytest.raises(Exception, match="p1-other is not the registered DM source p1-final"):
        ledger.check_upstream("p1-other", WeightsRef(sha256="1" * 64))
    with pytest.raises(Exception, match="is not the registered DM source"):
        ledger.check_upstream("p1-final", WeightsRef(sha256="3" * 64))


def test_check_record_flags_a_stale_bypass_after_a_retrained_dm(tmp_path):
    first = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    ledger.register(first.record, first.write_run_record(tmp_path))
    retrained = _Record("p1-retrained", checkpoint_sha="2" * 64)
    ledger.register(retrained.record, retrained.write_run_record(tmp_path))

    stale_p2 = {
        "phase": "P2",
        "run_id": "p2-old",
        "upstream": {"run_id": "p1-final", "checkpoint": {"sha256": "1" * 64}},
    }
    failures = ledger.check_record(stale_p2)
    assert failures == ["DM was retrained: this bypass is pinned to superseded DM p1-final while the registered DM source is p1-retrained"]


def test_check_record_flags_a_p1_record_whose_checkpoint_changed(tmp_path):
    record = _Record("p1-retrained", checkpoint_sha="2" * 64)
    ledger = DmSourceLedger(tmp_path)
    ledger.register(record.record, record.write_run_record(tmp_path))

    tampered = dict(record.record)
    tampered["selection"] = {"checkpoint": {"sha256": "9" * 64}}
    assert ledger.check_record(tampered) == ["registered DM source checkpoint no longer matches its P1 run record"]
    assert ledger.check_record(record.record) == []


def test_check_record_passes_a_bypass_pinned_to_the_registered_source(tmp_path):
    record = _Record("p1-final")
    ledger = DmSourceLedger(tmp_path)
    ledger.register(record.record, record.write_run_record(tmp_path))
    fresh_p2 = {"phase": "P2", "run_id": "p2-new", "upstream": {"run_id": "p1-final", "checkpoint": {"sha256": "1" * 64}}}
    assert ledger.check_record(fresh_p2) == []
