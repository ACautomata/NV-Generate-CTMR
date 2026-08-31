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

"""L2 shared vocabulary: pure statistics primitives and the cluster bootstrap
(ADR-0017 decision 1, issues #229 + #231).

Moved verbatim out of the terminal-acceptance judge (``final_acceptance``),
which now imports this module. ``RelativeDifference.of`` is the shared
``(gen - real) / real`` read-out -- the judge's relative TOST quantity families
and the ET-discrimination job draw it from here, while the exclusion/pairing
policies stay with the callers (the judge's ``real_denominator_zero``
exclusion, the diagnostic job's pairing classes). ``DistributionReadout.of``
is the quantile/mean block over one value list, the diagnostic jobs' shared
read-out (issue #232). ``ClusterBootstrap.quantile`` is the shared quantile
read-out (linear interpolation, ``index = q*(n-1)`` -- the same rule as the
calibration side's ``numpy.quantile`` defaults); the bootstrap itself
resamples cases as clusters. RNG / bit-stream discipline is registered on the
class below.

The dependency closure is third-party-free -- stdlib only, numpy/scipy/torch
unreachable -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py``.
"""

import math
import random


class RelativeDifference:
    """Relative difference ``(gen - real) / real``, the one definition (ADR-0017 decision 1).

    The real-side denominator must exist and be positive: a generated-side
    empty prediction stays in the distributions at -1.0 (protocol §4), while a
    non-positive (or undefined) real side leaves the quantity undefined --
    ``None``. The callers own what an undefined difference means: the judge's
    quantity families exclude it as ``real_denominator_zero``, the ET
    discrimination job captures the shape in its pairing classes.
    """

    @classmethod
    def of(cls, gen_vol, real_vol):
        if gen_vol is None or real_vol is None or real_vol <= 0:
            return None
        return (gen_vol - real_vol) / real_vol


class DistributionReadout:
    """Quantile/mean read-out over one value list (ADR-0017 decision 1, issue #232).

    The linear ``q*(n-1)`` rule of ``ClusterBootstrap.quantile`` with a
    ``None``-sentinel for an empty list (json has no NaN). Moved verbatim from
    the diagnostic jobs' shared copies (#232); the paired ``(gen - real) /
    real`` difference lives next door in ``RelativeDifference``.
    """

    @classmethod
    def of(cls, values):
        if not values:
            return {"median": None, "mean": None, "q05": None, "q95": None}
        return {
            "median": ClusterBootstrap.quantile(values, 0.5),
            "mean": sum(values) / len(values),
            "q05": ClusterBootstrap.quantile(values, 0.05),
            "q95": ClusterBootstrap.quantile(values, 0.95),
        }


class ClusterBootstrap:
    """Case-level cluster bootstrap with linear-interpolated quantiles.

    Percentile CIs use the same index = q*(n-1) linear rule as the calibration
    side's numpy.quantile defaults. The RNG is random.Random(seed); the
    calibration bit-stream (PCG64) is deliberately not reproduced -- this is a
    new computation, not a recomputation of ADR-0002 numbers (protocol §4).
    """

    def __init__(self, b):
        self._b = b

    @staticmethod
    def quantile(values, q):
        ordered = sorted(values)
        n = len(ordered)
        if n == 0:
            return math.nan
        if n == 1:
            return ordered[0]
        index = q * (n - 1)
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[int(index)]
        return ordered[low] + (index - low) * (ordered[high] - ordered[low])

    def ci90(self, per_case_values, seed):
        """Two-sided 90% CI of the pooled population, resampling cases (clusters)."""
        pool = [group for group in per_case_values if group]
        n = len(pool)
        if n == 0:
            return None
        rng = random.Random(seed)
        q05_samples, q95_samples = [], []
        for _ in range(self._b):
            pooled = []
            for _ in range(n):
                pooled += pool[rng.randrange(n)]
            q05_samples.append(self.quantile(pooled, 0.05))
            q95_samples.append(self.quantile(pooled, 0.95))
        return {"low": self.quantile(q05_samples, 0.05), "high": self.quantile(q95_samples, 0.95), "n_cases": n}

    def q5_lower_bound(self, per_case_values, seed):
        """One-sided bootstrap 95% lower bound of the population 5th percentile (D_r,low statistic)."""
        pool = [group for group in per_case_values if group]
        n = len(pool)
        if n == 0:
            return None
        rng = random.Random(seed)
        q05_samples = []
        for _ in range(self._b):
            pooled = []
            for _ in range(n):
                pooled += pool[rng.randrange(n)]
            q05_samples.append(self.quantile(pooled, 0.05))
        return {"bound": self.quantile(q05_samples, 0.05), "n_cases": n}
