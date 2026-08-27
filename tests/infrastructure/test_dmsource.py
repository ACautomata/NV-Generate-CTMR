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

"""Behaviour gates for DmSourceRepository and the sunk DmSourceLedger IO
(ADR-0015 section 4, issue #135).

The repository pins the ledger byte format (indent=2, sort_keys, trailing
newline) and absent-means-None semantics; read/write convergence is exercised
through the contract-layer consumer (``DmSourceLedger``), whose register /
schema-mismatch / verify-time mismatch behaviour must stay verbatim across the
sunk IO. Stdlib-only: runs on any machine (ADR-0013 section 4).
"""

import json

import pytest

from ctmr.infrastructure.dmsource import DmSourceRepository
from scripts.brats_phase_run_contract import (
    DM_SOURCE_SCHEMA,
    ContractViolationError,
    DmSourceLedger,
    RunRecordStore,
)


def _repo(root):
    return DmSourceRepository(root)


def _ledger(root):
    return DmSourceLedger(RunRecordStore(root))


def _registered_entry(run_id, sha):
    return {"schema": DM_SOURCE_SCHEMA, "run_id": run_id, "checkpoint": {"sha256": sha}}


def test_absent_ledger_reads_as_none_without_creating_the_file(tmp_path):
    assert _ledger(tmp_path).current() is None
    assert not _repo(tmp_path).path().exists()


def test_write_then_read_roundtrips_with_the_pinned_byte_format(tmp_path):
    entry = _registered_entry("run-a", "a" * 64)
    path = _repo(tmp_path).write(entry)
    text = path.read_text()
    assert text == json.dumps(entry, indent=2, sort_keys=True) + "\n"
    assert _repo(tmp_path).read() == entry


def test_register_persists_through_the_repository_and_is_idempotent(tmp_path):
    run_record = tmp_path / "runs" / "r" / "run.json"
    run_record.parent.mkdir(parents=True)
    run_record.write_text("{}\n")
    record = {
        "phase": "P1",
        "run_id": "run-a",
        "selection": {"checkpoint": {"sha256": "a" * 64}},
        "configs": {"env": 1},
        "manifest": {"unet": ["x"]},
    }
    first = _ledger(tmp_path).register(record, run_record)
    # Written by the single repository mouth: one file, pinned byte format.
    text = (tmp_path / "dm_source.json").read_text()
    assert text.endswith("\n")
    assert json.loads(text) == first
    before = (tmp_path / "dm_source.json").read_bytes()
    assert _ledger(tmp_path).register(record, run_record) == first  # idempotent re-register
    assert (tmp_path / "dm_source.json").read_bytes() == before


def test_current_raises_on_schema_mismatch_verbatim(tmp_path):
    _repo(tmp_path).write({"schema": "brats-dm-source/0"})
    with pytest.raises(ContractViolationError) as excinfo:
        _ledger(tmp_path).current()
    message = str(excinfo.value).replace(str(tmp_path), "<root>")
    assert message == "dm_source ledger <root>/dm_source.json has schema 'brats-dm-source/0' != 'brats-dm-source/1'"


def test_check_record_flags_a_bypass_pinned_to_a_superseded_dm_verbatim(tmp_path):
    _repo(tmp_path).write(_registered_entry("run-a", "a" * 64))
    bypass_retrained_dm = {"phase": "P2", "upstream": {"run_id": "run-b", "checkpoint": {"sha256": "b" * 64}}}
    assert _ledger(tmp_path).check_record(bypass_retrained_dm) == [
        "DM was retrained: this bypass is pinned to superseded DM run-b " "while the registered DM source is run-a"
    ]


def test_check_record_flags_a_registered_p1_whose_checkpoint_no_longer_matches(tmp_path):
    _repo(tmp_path).write(_registered_entry("run-a", "a" * 64))
    diverged_p1 = {"phase": "P1", "run_id": "run-a", "selection": {"checkpoint": {"sha256": "b" * 64}}}
    assert _ledger(tmp_path).check_record(diverged_p1) == ["registered DM source checkpoint no longer matches its P1 run record"]


def test_check_record_passes_records_still_matching_the_current_source(tmp_path):
    _repo(tmp_path).write(_registered_entry("run-a", "a" * 64))
    aligned_bypass = {"phase": "P2", "upstream": {"run_id": "run-a", "checkpoint": {"sha256": "a" * 64}}}
    other_p1 = {"phase": "P1", "run_id": "run-other", "selection": {"checkpoint": {"sha256": "b" * 64}}}
    no_upstream_p2 = {"phase": "P2"}
    for record in (aligned_bypass, other_p1, no_upstream_p2):
        assert _ledger(tmp_path).check_record(record) == []
