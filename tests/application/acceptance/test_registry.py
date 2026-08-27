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

"""Convergence-gate tests for the acceptance-layer registry home (issue #136).

Locks ADR-0012 decision 2 at its birthplace: one declaration point drives
every kind relation, entries are identity-only and frozen (gate constants and
verdict recomputation keep their dual-side sources, ADR-0006), and the legacy
run contract re-binds to these very objects (the reverse shim leaves no
private copy behind).
"""

import dataclasses

import pytest

from ctmr.application.acceptance.registry import (
    ACCEPTANCE_LAYERS,
    ATTACH_KINDS,
    FORMAL_LAYER_KINDS,
    L1_SCHEMA,
    L2_SCHEMA,
    L3_SCHEMA,
    LAYER_KINDS,
    AcceptanceLayer,
)


def test_single_declaration_point_drives_every_kind_relation():
    assert [(layer.layer_name, layer.kind, layer.report_schema) for layer in ACCEPTANCE_LAYERS] == [
        ("L1", "l1_report", L1_SCHEMA),
        ("L2", "l2_report", L2_SCHEMA),
        ("L3", "l3_report", L3_SCHEMA),
    ]
    assert FORMAL_LAYER_KINDS == ("l1_report", "l2_report", "l3_report")
    assert list(LAYER_KINDS) == ["L1", "L2", "L3"]  # order is load-bearing: reason collection and verdict entry layout
    assert ATTACH_KINDS == ("l1_report", "l2_report", "l3_report", "env")


def test_entries_are_identity_only():
    # No gate constants, verdict readers or reason builders in the base:
    # judgement stays dual-sourced (ADR-0006); validators stay with their layers.
    assert {field.name for field in dataclasses.fields(AcceptanceLayer)} == {"layer_name", "kind", "report_schema"}


def test_layer_entry_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ACCEPTANCE_LAYERS[0].kind = "env"


def test_script_side_contract_rebinds_the_home_objects_verbatim():
    from scripts import brats_phase_run_contract as contract_module

    # Reverse shim: the big module derives nothing locally anymore -- it re-binds
    # the same immutable objects under its historical names, so every existing
    # consumer keeps resolving them against the single declaration point.
    assert contract_module.ATTACH_KINDS is ATTACH_KINDS
    assert contract_module.FORMAL_LAYER_KINDS is FORMAL_LAYER_KINDS
    assert contract_module.LAYER_KINDS is LAYER_KINDS
    schemas = {layer.layer_name: layer.report_schema for layer in ACCEPTANCE_LAYERS}
    assert contract_module.L1_SCHEMA == schemas["L1"] == L1_SCHEMA
    assert contract_module.L2_SCHEMA == schemas["L2"] == L2_SCHEMA
    assert contract_module.L3_SCHEMA == schemas["L3"] == L3_SCHEMA
