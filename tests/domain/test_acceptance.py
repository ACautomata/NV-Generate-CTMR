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

"""Final-acceptance verdict kernel tests (CONTEXT.md 完整终验裁决 / undecided).

Pure domain: the non-compensatory AND with no compensation across layers, the
undecided third state blocking exactly like a fail, never-vacuous blocker
reasons, and the verify-time expected-overall re-derivation that catches a
hand-edited verdict flip. Reads do not change with the migration (#141): the
verdict pairs below are the same the legacy conclude chain produced.
"""

import pytest

from ctmr.domain.acceptance import FinalAcceptanceRule


@pytest.fixture()
def rule():
    return FinalAcceptanceRule()


def test_all_pass_layers_conclude_pass_with_no_blockers(rule):
    verdict, reasons = rule.judge({"L1": "pass", "L2": "pass", "L3": "pass"}, {})

    assert verdict == "pass"
    assert reasons == []


def test_any_failing_layer_blocks_and_is_not_offset_by_passing_layers(rule):
    verdict, reasons = rule.judge(
        {"L1": "fail", "L2": "pass", "L3": "pass"},
        {"L1": ["L1 FID GLI/t1n: fail"]},
    )

    assert verdict == "blocked"
    assert reasons == ["L1 FID GLI/t1n: fail"]


def test_l2_undecided_blocks_like_a_fail(rule):
    verdict, reasons = rule.judge(
        {"L1": "pass", "L2": "undecided", "L3": "pass"},
        {"L2": ["L2 SSA: undecided (instrument failure gate)"]},
    )

    assert verdict == "blocked"
    assert reasons == ["L2 SSA: undecided (instrument failure gate)"]


def test_blockers_accumulate_across_layers_in_input_order(rule):
    verdict, reasons = rule.judge(
        {"L1": "fail", "L2": "fail", "L3": "fail"},
        {
            "L1": ["L1 FID GLI/t1n: fail"],
            "L2": ["L2 MEN: fail (1 TOST and 0 round-trip checks failed)"],
            "L3": ["L3 visual-Turing: CI window gate not met"],
        },
    )

    assert verdict == "blocked"
    assert reasons == [
        "L1 FID GLI/t1n: fail",
        "L2 MEN: fail (1 TOST and 0 round-trip checks failed)",
        "L3 visual-Turing: CI window gate not met",
    ]


def test_blocker_layer_without_reasons_falls_back_to_a_named_verdict_line(rule):
    verdict, reasons = rule.judge({"L1": "pass", "L2": "undecided", "L3": "pass"}, {})

    assert verdict == "blocked"
    assert reasons == ["L2 verdict is undecided"]


def test_expected_overall_follows_the_and_of_layer_verdicts(rule):
    assert rule.expected_overall({"L1": "pass", "L2": "pass", "L3": "pass"}) == "pass"
    assert rule.expected_overall({"L1": "fail", "L2": "pass", "L3": "pass"}) == "blocked"
    assert rule.expected_overall({"L1": "pass", "L2": "undecided", "L3": "pass"}) == "blocked"


def test_recorded_flip_is_caught_by_the_expected_overall_derivation(rule):
    # a hand-edited verdict file claiming pass over a failing layer must
    # disagree with the kernel's AND (the legacy verify failure text)
    recorded = "pass"
    layer_verdicts = {"L1": "pass", "L2": "fail", "L3": "pass"}

    assert rule.expected_overall(layer_verdicts) != recorded
