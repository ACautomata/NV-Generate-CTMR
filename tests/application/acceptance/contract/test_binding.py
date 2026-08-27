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

"""FrozenRunBinding tests (ADR-0012 决定 4 / CONTEXT.md 冻结候选绑定).

The five-key identity extraction is the single construction point with the
``require_frozen`` gate built in: extracting is validating the run state.
"""

import json

import pytest

from ctmr.application.acceptance.contract.binding import BINDING_KEYS, FrozenRunBinding, FrozenRunBindingError


def frozen_record(run_id="frozen-candidate-1"):
    return {
        "schema": "brats-phase-run/1",
        "run_id": run_id,
        "phase": "P1",  # frozen artifact domain value
        "status": "frozen",
        "manifest": {"path": "/tmp/manifest.json", "sha256": "a" * 64},
        "selection": {"checkpoint": {"path": "/tmp/candidate.pt", "sha256": "b" * 64}},
        "samples": {"path": "/tmp/samples.json", "sha256": "c" * 64},
        "frozen_utc": "2026-08-27T00:00:00Z",
    }


def test_from_record_extracts_the_five_keys():
    binding = FrozenRunBinding.from_record(frozen_record())

    assert binding.run_id == "frozen-candidate-1"
    assert binding.phase == "P1"
    assert binding.manifest_sha256 == "a" * 64
    assert binding.candidate_checkpoint_sha256 == "b" * 64
    assert binding.samples_sha256 == "c" * 64
    assert binding.as_dict() == {
        "run_id": "frozen-candidate-1",
        "phase": "P1",
        "manifest_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "samples_sha256": "c" * 64,
    }


def test_from_record_requires_frozen_state():
    record = frozen_record()
    record["status"] = "open"
    record.pop("frozen_utc")

    with pytest.raises(FrozenRunBindingError, match="is 'open'"):
        FrozenRunBinding.from_record(record)


def test_from_record_reports_missing_binding_keys():
    record = frozen_record()
    del record["samples"]

    with pytest.raises(FrozenRunBindingError, match="samples_sha256"):
        FrozenRunBinding.from_record(record)


def test_binding_keys_constant_pins_the_identity_shape():
    assert BINDING_KEYS == (
        "run_id",
        "phase",
        "manifest_sha256",
        "candidate_checkpoint_sha256",
        "samples_sha256",
    )


def test_from_path_loads_a_record_file(tmp_path):
    record_path = tmp_path / "run.json"
    record_path.write_text(json.dumps(frozen_record()))

    binding = FrozenRunBinding.from_path(record_path)

    assert binding.run_id == "frozen-candidate-1"
    assert binding.as_dict()["candidate_checkpoint_sha256"] == "b" * 64


def test_from_path_rejects_missing_record_file(tmp_path):
    with pytest.raises(FrozenRunBindingError, match="run record not found"):
        FrozenRunBinding.from_path(tmp_path / "missing.json")


def test_from_path_requires_frozen_state(tmp_path):
    record_path = tmp_path / "run.json"
    record = frozen_record()
    record["status"] = "open"
    record_path.write_text(json.dumps(record))

    with pytest.raises(FrozenRunBindingError, match="is 'open'"):
        FrozenRunBinding.from_path(record_path)


def test_mismatches_for_reports_diverging_keys():
    report_binding = {
        "run_id": "frozen-candidate-1",
        "phase": "P1",
        "manifest_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "0" * 64,
        "samples_sha256": "c" * 64,
    }

    assert FrozenRunBinding.mismatches_for(frozen_record(), report_binding) == ["candidate_checkpoint_sha256"]


def test_mismatches_for_requires_an_object_binding():
    assert FrozenRunBinding.mismatches_for(frozen_record(), None) is None


def test_mismatches_for_tolerates_partial_records():
    """Validator-side compare must not raise on records missing binding keys (the run is
    unfrozen or truncated); the missing key surfaces as a mismatch, not an exception."""
    record = frozen_record()
    del record["samples"]

    assert FrozenRunBinding.mismatches_for(record, FrozenRunBinding.from_record(frozen_record()).as_dict()) == ["samples_sha256"]
