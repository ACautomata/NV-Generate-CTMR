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

"""Convergence-gate tests for the frozen-run five-key binding home (issue #136).

Locks ADR-0012 decision 4 at its birthplace: one extraction point over a
single field-path set, key order stable for failure ordering, the
require-frozen gate built into ``FrozenRunBinding`` -- and the run contract's
validators delegating to the shared extraction with byte-for-byte unchanged
registered failure messages.
"""

import dataclasses

import pytest

from ctmr.application.acceptance.binding import (
    BINDING_KEYS,
    FrozenRunBinding,
    FrozenRunBindingError,
    expected_binding,
)

FROZEN_RECORD = {
    "run_id": "p1-selftest-000",
    "phase": "P1",
    "status": "frozen",
    "manifest": {"path": "/controlled/manifest.json", "sha256": "a" * 64},
    "selection": {"checkpoint": {"path": "/controlled/candidate.pt", "sha256": "b" * 64}},
    "samples": {"path": "/controlled/samples.json", "sha256": "c" * 64},
}


def test_expected_binding_extracts_the_five_keys_via_one_field_path_set():
    assert expected_binding(FROZEN_RECORD) == {
        "run_id": "p1-selftest-000",
        "phase": "P1",
        "manifest_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "samples_sha256": "c" * 64,
    }


def test_key_order_is_stable_and_matches_the_extraction():
    assert (
        tuple(expected_binding(FROZEN_RECORD))
        == BINDING_KEYS
        == (
            "run_id",
            "phase",
            "manifest_sha256",
            "candidate_checkpoint_sha256",
            "samples_sha256",
        )
    )


def test_pre_freeze_records_extract_without_raising():
    extracted = expected_binding({"run_id": "p1-open-000", "phase": "P1"})
    assert extracted["manifest_sha256"] is None
    assert extracted["candidate_checkpoint_sha256"] is None
    assert extracted["samples_sha256"] is None


def test_frozen_run_binding_carries_the_gate():
    binding = FrozenRunBinding.from_record(FROZEN_RECORD)
    assert binding.as_dict() == expected_binding(FROZEN_RECORD)

    with pytest.raises(FrozenRunBindingError, match="'open'"):
        FrozenRunBinding.from_record({**FROZEN_RECORD, "status": "open"})
    with pytest.raises(FrozenRunBindingError, match="binding requires a frozen candidate"):
        FrozenRunBinding.from_record({})


def test_binding_is_a_frozen_value_object_of_exactly_the_five_keys():
    binding = FrozenRunBinding.from_record(FROZEN_RECORD)
    assert tuple(field.name for field in dataclasses.fields(type(binding))) == BINDING_KEYS
    with pytest.raises(dataclasses.FrozenInstanceError):
        binding.run_id = "mutated"


def test_contract_validators_delegate_to_the_shared_extraction():
    from scripts import brats_phase_run_contract as contract_module

    # The big module holds no private copy: it binds the shared function itself.
    assert contract_module.expected_binding is expected_binding


def test_validator_binding_messages_are_byte_for_byte_unchanged():
    from scripts.brats_phase_run_contract import L1ReportValidator, L2ReportValidator, L3ReportValidator

    record = {"run_id": "p1-x", "phase": "P1"}

    unbound_l1 = []
    L1ReportValidator()._binding(record, {}, unbound_l1)
    assert unbound_l1[:2] == [
        "L1 report schema != brats-l1-report/1",
        "L1 report binding must be an object",
    ]

    unbound_l2 = []
    L2ReportValidator()._binding(record, {}, unbound_l2)
    assert unbound_l2[:2] == [
        "L2 report schema != l2-final-acceptance-report/1",
        "L2 report binding must be an object (evaluate with --run to bind the frozen candidate)",
    ]

    partially_bound = []
    report = {"schema": "brats-l3-report/1", "binding": {"run_id": "wrong-run"}}
    L3ReportValidator()._binding(record, report, partially_bound)
    assert partially_bound == [
        "L3 report binding run_id does not match frozen run",
        "L3 report binding phase does not match frozen run",
    ]

    every_key_wrong = []
    bad_binding = {key: "0" for key in BINDING_KEYS}
    report = {"schema": "brats-l1-report/1", "binding": bad_binding}
    L1ReportValidator()._binding(FROZEN_RECORD, report, every_key_wrong)
    assert every_key_wrong == [f"L1 report binding {key} does not match frozen run" for key in BINDING_KEYS]
