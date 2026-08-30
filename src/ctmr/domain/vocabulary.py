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

"""Frozen instrument vocabulary -- the stdlib-only leaf (ADR-0017 decision 3, issue #222).

The single definition point of the region/label vocabulary and the frozen
Wilson constants. ``REGIONS`` is the one dict behind the seven drifted
literals (calibration mother, synthetic-domain eval, P1/P2 dev eval,
terminal-acceptance judge tuple + labels dict): instrumentation session masks
are BraTS-2023 label arrays (background 0, non-enhancing core 1, peritumoral
oedema 2, enhancing tumour 3) whose region projections are WT = {1,2,3},
TC = {1,3} and ET = {3}. The judge's tuple-of-names form derives from this
dict (``REGION_NAMES``), never a second literal; ``LABEL_DOMAIN`` pins every
value a well-formed mask may carry. ``WilsonUpper`` is the single
Wilson-score-95% upper bound behind the five drifted copies, with the one
``n == 0`` guard (the terminal-acceptance call site guards before calling, so
its frozen output is unchanged).

The dependency closure is third-party-free -- stdlib only, numpy/scipy/torch
unreachable -- so both the numpy side (``ctmr.domain.measurement`` re-exports
all of it, consumer surface unchanged) and the stdlib side (the
terminal-acceptance judge and the shared vocabulary) can draw from this one
home. This revises the *address* of ADR-0010 decision 4 -- "unique definition
in measurement" becomes "unique definition in the vocabulary leaf, measurement
re-exports"; the single-definition-point semantics are unchanged. This module
is itself the registration of the stdlib-only property: an import-face test
(``tests/domain/test_vocabulary.py``) guards the closure, replacing the
judge's former docstring-only discipline.
"""

import math

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
"""Per-region label tuples, keyed in canonical WT / TC / ET order (dict order)."""

REGION_NAMES = tuple(REGIONS)
"""The judge's tuple-of-names form, derived -- not a second literal (ADR-0010 decision 1)."""

LABEL_DOMAIN = (0, 1, 2, 3)
"""Every label value a well-formed instrument mask may carry (0 = background)."""


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
