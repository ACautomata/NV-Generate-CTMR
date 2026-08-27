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

"""Shared measurement primitives, one definition each (ADR-0010, issue #109).

``DiceScore`` is the single Dice definition behind the four drifted copies
(calibration ``dice_of``, synthetic-domain eval, terminal-acceptance
``condition_dice``, P2 dev eval): the empty-denominator sentinel is *one*
value -- ``None`` -- aligned with the frozen terminal-acceptance semantics.
``WilsonUpper`` is the single Wilson-score-95% upper-bound definition behind
the five copies, with the one ``n == 0`` guard (the terminal-acceptance call
site guards before calling, so its frozen output is unchanged by this module).
Both are class-method namespace classes, not free functions (repo python.md).
"""

import math

import numpy as np


class DiceScore:
    """Dice similarity of two boolean masks; ``None`` when both are empty (the one sentinel)."""

    @classmethod
    def of(cls, first: np.ndarray, second: np.ndarray) -> float | None:
        denominator = int(first.sum()) + int(second.sum())
        if denominator == 0:
            return None  # the single empty-denominator sentinel (ADR-0010 decision 4)
        return float(2 * np.logical_and(first, second).sum() / denominator)


class WilsonUpper:
    """Wilson score-interval 95% upper bound of a binomial proportion, ``n == 0`` guarded."""

    Z95 = 1.959963984540054  # the frozen z-value from all five drifted copies

    @classmethod
    def of(cls, successes: int, trials: int) -> float:
        if trials == 0:
            return math.nan  # the single n==0 guard; terminal-acceptance's call site kept its own None
        probability = successes / trials
        denom = 1 + cls.Z95**2 / trials
        center = (probability + cls.Z95**2 / (2 * trials)) / denom
        half = (cls.Z95 / denom) * math.sqrt(probability * (1 - probability) / trials + cls.Z95**2 / (4 * trials**2))
        return min(1.0, center + half)
