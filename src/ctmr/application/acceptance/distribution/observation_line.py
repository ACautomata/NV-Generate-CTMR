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

"""Observation-line yellow flag (issue #253, parent #247): the dev selection
point's ET/WT monitoring line as a pure evaluation.

Adoption ruling #5 (``20260830-P1根因甄别-读数收编与整改方向决议.md`` §4)
de-blinds candidate selection: outside dev FID, the selection point reads the
job-B ET/WT vocabulary and evaluates a pre-recorded observation line --

  METS ET detection rate < 0.9, or a per-challenge vol_et_rel median > 2
  -> YELLOW FLAG.

The line hangs beside the ET discrimination module as a new seam: its input is
exactly the reading shape ``EtDiscrimination.discriminate`` produces, so the
monitor consumes both verbatim. It is a selection surface, never an acceptance
verdict -- zero contact with the judgement chain (no judge-module import;
source-pinned by test), no pass/fail semantics, only the flag and the fired
rule texts. The thresholds are the recorded literals; the first retrained
candidate's dev distribution may tighten them, never silently (new ticket).

Usage:
    ObservationLine().evaluate(readings) -> flag payload (json-serializable)
"""

from ctmr.application.acceptance.distribution.challenge_registry import CHALLENGES
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError


class ObservationLine:
    """监控读数 → 黄旗判定(选择面,非验收判定)。

    Strict inequalities: a rate exactly at the floor and a median exactly at
    the ceiling sit ON the line and do not fire. A missing METS reading or an
    empty METS denominator raises ``DiagnosticError`` -- the flag's primary
    subject cannot be silently absent. An undefined per-challenge rel median
    (no paired cases) is recorded with ``fires=False`` and the ``None`` median
    kept visible: honest not-evaluable, never an invented pass.
    """

    METS_CHALLENGE = "METS"
    METS_ET_RATE_FLOOR = 0.9
    VOL_ET_REL_MEDIAN_CEILING = 2.0

    def __init__(self, mets_rate_floor: float = METS_ET_RATE_FLOOR, rel_median_ceiling: float = VOL_ET_REL_MEDIAN_CEILING):
        self._mets_rate_floor = mets_rate_floor
        self._rel_median_ceiling = rel_median_ceiling

    def evaluate(self, readings):
        """Job-B readings (the ``EtDiscrimination.discriminate`` output) -> flag payload.

        ``per_challenge`` carries each challenge's rule outcomes and flag;
        ``fired`` the human-readable rule texts; ``flag`` the overall yellow.
        """
        by_challenge = {reading["challenge"]: reading for reading in readings}
        if self.METS_CHALLENGE not in by_challenge:
            raise DiagnosticError(f"observation line needs METS readings; challenges present: {sorted(by_challenge) or 'none'}")
        mets_rate = by_challenge[self.METS_CHALLENGE]["gen"]["rate"]
        if mets_rate is None:
            raise DiagnosticError("observation line needs a non-empty METS denominator; gen rate is undefined")

        per_challenge = {}
        fired = []
        for challenge in sorted(by_challenge, key=CHALLENGES.index):
            reading = by_challenge[challenge]
            rate_rule = None
            rules_fire = False
            if challenge == self.METS_CHALLENGE:
                rate_rule = {"rate": mets_rate, "floor": self._mets_rate_floor, "fires": mets_rate < self._mets_rate_floor}
                if rate_rule["fires"]:
                    fired.append(f"METS ET 检出率 {mets_rate:.4f} < {self._mets_rate_floor}")
                    rules_fire = True
            median = reading["rel_diff"]["median"]
            median_rule = {
                "median": median,
                "ceiling": self._rel_median_ceiling,
                "fires": median is not None and median > self._rel_median_ceiling,
            }
            if median_rule["fires"]:
                fired.append(f"{challenge} vol_et_rel 中位 {median:.4f} > {self._rel_median_ceiling}")
                rules_fire = True
            per_challenge[challenge] = {
                "mets_et_rate": rate_rule,
                "vol_et_rel_median": median_rule,
                "flag": rules_fire,
            }
        return {
            "rules": {"mets_et_rate_floor": self._mets_rate_floor, "vol_et_rel_median_ceiling": self._rel_median_ceiling},
            "per_challenge": per_challenge,
            "fired": fired,
            "flag": bool(fired),
        }
