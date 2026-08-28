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

"""Final-acceptance verdict kernel -- pure logic, no IO (ADR-0015 §2 domain/acceptance).

``FinalAcceptanceRule`` is the non-compensatory L1∧L2∧L3 AND (CONTEXT.md
完整终验裁决): every layer's own verdict must read ``pass`` for the verdict to
be ``pass``; any failing layer -- including an L2 ``undecided`` (the
instrument-unavailable third state, which blocks the full spec acceptance
exactly like a fail) -- writes ``blocked`` with the traceable per-layer
reasons, and no other layer's score can offset it. The layer verdicts and
blocked reasons are inputs, not recomputations here: gate-constant mirrors
and verdict recomputation stay judgement and remain inside each layer's
validator (different-sourced on both sides, ADR-0006 referee independence);
the kernel only owns the cross-layer AND. Hierarchy-violation constants have
their single canonical home in ``ctmr.domain.measurement`` and are not
mirrored here. Both the conclude path and the verify-time reconciliation
consume this one kernel so the two sides cannot drift.
"""


class FinalAcceptanceRule:
    """The non-compensatory cross-layer AND with the undecided-blocking state."""

    PASS = "pass"
    BLOCKED = "blocked"

    def judge(self, layer_verdicts, reasons_by_layer):
        """Judge layer verdicts (insertion-ordered) into (verdict, blocked_reasons).

        ``layer_verdicts`` maps layer name -> that layer's own verdict
        (``pass``/``fail``/``undecided``); ``reasons_by_layer`` maps layer
        name -> the layer's traceable blocker list. A layer is a blocker iff
        its verdict is not ``pass`` (undecided blocks like a fail); a blocker
        layer without reasons falls back to ``"{name} verdict is {verdict}"``
        so the blocking reason list is never vacuous.
        """
        blocked_reasons = []
        for name, verdict in layer_verdicts.items():
            if verdict == self.PASS:
                continue
            reasons = list(reasons_by_layer.get(name) or ())
            if not reasons:
                reasons = [f"{name} verdict is {verdict}"]
            blocked_reasons += reasons
        return (self.PASS if not blocked_reasons else self.BLOCKED), blocked_reasons

    def expected_overall(self, layer_verdicts):
        """The overall verdict implied by layer verdicts (verify-time AND re-derivation).

        A recorded verdict file must agree with this derivation even when its
        attachments are intact -- a hand-edited flip fails verification.
        """
        return self.PASS if all(verdict == self.PASS for verdict in layer_verdicts.values()) else self.BLOCKED
