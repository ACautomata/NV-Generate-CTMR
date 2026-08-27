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
``FidScoreCalculator`` is the Fréchet-distance kernel the L1 quantitative
chain and the dev-side plane-FID trend share (ticket 09 relocation, verbatim
arithmetic from the retired ``scripts/brats_l1_quantitative.py`` definition).
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


class FidScoreCalculator:
    """Calculates Fréchet distance between two feature distributions."""

    def score(self, real_features, generated_features):
        real = self._validated_features(real_features, "real")
        generated = self._validated_features(generated_features, "generated")
        if real.shape[1] != generated.shape[1]:
            raise ValueError(f"feature dimensions differ: real={real.shape[1]}, generated={generated.shape[1]}")
        real_mean, real_covariance = self._statistics(real)
        generated_mean, generated_covariance = self._statistics(generated)
        mean_distance = np.sum((real_mean - generated_mean) ** 2)
        covariance_trace = np.trace(real_covariance + generated_covariance)
        covariance_sqrt_trace = self._symmetric_product_sqrt_trace(real_covariance, generated_covariance)
        return float(max(mean_distance + covariance_trace - 2.0 * covariance_sqrt_trace, 0.0))

    def _validated_features(self, features, label):
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
            raise ValueError(f"{label} features must be a finite 2D array with at least two rows")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} features contain non-finite values")
        return array

    def _statistics(self, features):
        covariance = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
        return np.mean(features, axis=0), covariance

    def _symmetric_product_sqrt_trace(self, first_covariance, second_covariance):
        first_values, first_vectors = np.linalg.eigh(first_covariance)
        first_sqrt = (first_vectors * np.sqrt(np.clip(first_values, 0.0, None))) @ first_vectors.T
        symmetric_product = first_sqrt @ second_covariance @ first_sqrt
        symmetric_product = (symmetric_product + symmetric_product.T) / 2.0
        product_values = np.linalg.eigvalsh(symmetric_product)
        return float(np.sum(np.sqrt(np.clip(product_values, 0.0, None))))


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
