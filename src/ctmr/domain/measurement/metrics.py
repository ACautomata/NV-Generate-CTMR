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
``WilsonUpper`` moved to the stdlib-only leaf ``ctmr.domain.vocabulary``
(ADR-0017 decision 3) and is re-exported here -- this module's import surface
is unchanged. ``DiceScore`` is a class-method namespace class, not a free
function (repo python.md); the re-exported ``WilsonUpper`` keeps the same
shape.
"""

import numpy as np

from ctmr.domain.vocabulary import WilsonUpper  # noqa: F401  (re-export, ADR-0017 decision 3)


class DiceScore:
    """Dice similarity of two boolean masks; ``None`` when both are empty (the one sentinel)."""

    @classmethod
    def of(cls, first: np.ndarray, second: np.ndarray) -> float | None:
        denominator = int(first.sum()) + int(second.sum())
        if denominator == 0:
            return None  # the single empty-denominator sentinel (ADR-0010 decision 4)
        return float(2 * np.logical_and(first, second).sum() / denominator)
