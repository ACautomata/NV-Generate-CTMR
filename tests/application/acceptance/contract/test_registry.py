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

"""AcceptanceLayer registry tests (ADR-0012 决定 2 / CONTEXT.md 验收层注册表).

The registry is the single declaration point; ATTACH_KINDS / FORMAL_LAYER_KINDS /
LAYER_KINDS and the contract-side schema strings all derive from it. These tests
pin the derived relations the way the former parallel wiring behaved. Frozen
artifact values (``l1_report``, ``L1`` ...) keep their literal compatibility --
they are run-record / verdict-artifact strings, not code naming.
"""

from dataclasses import FrozenInstanceError

import pytest

from ctmr.application.acceptance.contract.registry import (
    ACCEPTANCE_LAYERS,
    ATTACH_KINDS,
    DISTRIBUTION_SCHEMA,
    EXPERT_REVIEW_SCHEMA,
    FORMAL_LAYER_KINDS,
    LAYER_BY_KIND,
    LAYER_KINDS,
    QUANTITATIVE_SCHEMA,
    AcceptanceLayer,
)


def frozen_record():
    return {
        "run_id": "frozen-candidate-1",
        "phase": "P1",  # frozen artifact domain value
        "status": "frozen",
        "manifest": {"path": "/tmp/manifest.json", "sha256": "a" * 64},
        "selection": {"checkpoint": {"path": "/tmp/c.pt", "sha256": "b" * 64}},
        "samples": {"path": "/tmp/s.json", "sha256": "c" * 64},
    }


def test_registry_declares_exactly_the_three_formal_layers_in_stable_order():
    kinds = [layer.kind for layer in ACCEPTANCE_LAYERS]
    names = [layer.name for layer in ACCEPTANCE_LAYERS]

    # kind/name values are frozen artifact strings (attachments, verdict layers dict)
    assert kinds == ["l1_report", "l2_report", "l3_report"]
    assert names == ["L1", "L2", "L3"]


def test_registry_entries_are_frozen_declarations():
    for layer in ACCEPTANCE_LAYERS:
        assert isinstance(layer, AcceptanceLayer)
        with pytest.raises(FrozenInstanceError):  # frozen dataclass: no mutation
            layer.kind = "env"


def test_attach_kinds_derived_from_registry():
    # layer kinds plus env; env is the non-layer attachment: no validator, never concludes
    assert ATTACH_KINDS == ("l1_report", "l2_report", "l3_report", "env")


def test_formal_layer_kinds_derived_from_registry():
    assert FORMAL_LAYER_KINDS == ("l1_report", "l2_report", "l3_report")


def test_layer_kinds_derived_from_registry():
    assert LAYER_KINDS == {"L1": "l1_report", "L2": "l2_report", "L3": "l3_report"}
    assert list(LAYER_KINDS) == ["L1", "L2", "L3"]  # verdict-artifact layer order stays stable


def test_contract_schemas_derive_from_registry():
    assert QUANTITATIVE_SCHEMA == "brats-l1-report/1"
    assert DISTRIBUTION_SCHEMA == "l2-final-acceptance-report/1"
    assert EXPERT_REVIEW_SCHEMA == "brats-l3-report/1"
    assert LAYER_BY_KIND["l1_report"].schema == QUANTITATIVE_SCHEMA


def test_validator_factories_validate_reports(tmp_path):
    """Every formal layer's factory yields a validator with the (record, path) contract.

    Reported messages keep their frozen artifact prefixes (``L1 report not found`` ...)."""
    record = frozen_record()
    missing = tmp_path / "missing.json"
    expected = {
        "l1_report": f"L1 report not found: {missing}",
        "l2_report": f"L2 report not found: {missing}",
        "l3_report": f"L3 report not found: {missing}",
    }
    for layer in ACCEPTANCE_LAYERS:
        failures = layer.validator_factory().validate(record, missing)
        assert failures == [expected[layer.kind]]


def test_layer_kinds_reader_and_reasons_builder_shape():
    """The registry entry fields keep the per-layer verdict reader / reason builder
    behaviours the parallel wiring used to express (judgement stays inside the layers)."""
    quantitative = LAYER_BY_KIND["l1_report"]
    distribution = LAYER_BY_KIND["l2_report"]
    expert_review = LAYER_BY_KIND["l3_report"]

    assert quantitative.verdict_reader({"summary": {"verdict": "pass"}}) == "pass"
    assert distribution.verdict_reader({"overall_verdict": "undecided"}) == "undecided"
    assert expert_review.verdict_reader({"verdict": {"overall": "fail"}}) == "fail"

    quantitative_report = {
        "fid_results": [{"challenge": "GLI", "target_modality": "t1c", "verdict": "fail"}],
        "p3_paired_results": [{"challenge": "GLI", "src_modality": "t1n", "target_modality": "t1c", "gate_applicable": True, "verdict": "fail"}],
    }
    assert quantitative.reasons_builder(quantitative_report) == [
        "L1 FID GLI/t1c: fail",
        "L1 P3 paired GLI/t1n->t1c: fail",
    ]

    distribution_report = {
        "per_challenge": {
            "SSA": {"verdict": "undecided", "reason": "instrument failure on tested samples"},
            "GLI": {"verdict": "pass"},
        }
    }
    assert distribution.reasons_builder(distribution_report) == ["L2 SSA: undecided (instrument failure on tested samples)"]
    assert distribution.reasons_builder({"per_challenge": {"SSA": {"verdict": "fail"}}}) == ["L2 SSA: fail (0 TOST and 0 round-trip checks failed)"]

    expert_review_report = {
        "visual_turing": {"verdict": "pass"},
        "likert": [
            {"dimension": "overall_realism", "per_modality": {}, "phase": {"verdict": "fail"}},
        ],
    }
    assert expert_review.reasons_builder(expert_review_report) == ["L3 Likert overall_realism: lower-bound gate not met"]
    assert expert_review.reasons_builder({"visual_turing": {"verdict": "pass"}, "likert": []}) == []
