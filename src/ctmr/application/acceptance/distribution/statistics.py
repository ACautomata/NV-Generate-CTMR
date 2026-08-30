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
(ADR-0017 decision 1, issue #229).

Moved verbatim out of the terminal-acceptance judge (``final_acceptance``),
which now imports this module. ``ClusterBootstrap.quantile`` is the shared
quantile read-out (linear interpolation, ``index = q*(n-1)`` -- the same rule
as the calibration side's ``numpy.quantile`` defaults); the bootstrap itself
resamples cases as clusters. RNG / bit-stream discipline is registered on the
class below.

The dependency closure is third-party-free -- stdlib only, numpy/scipy/torch
unreachable -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py``.
"""

import math
import random


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
