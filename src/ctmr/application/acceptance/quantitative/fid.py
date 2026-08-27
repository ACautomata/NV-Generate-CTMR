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

"""Fréchet-distance metric computation (pure numpy, no IO).

Lifted verbatim out of ``scripts/brats_l1_quantitative.py`` (ticket 10): the
calculator and its error type are the shared metric face consumed by the
quantitative chain and the generation families' dev-trend machinery alike,
and the L1 script consumes them back from here until its own migration batch
relocates the rest of the chain into this package.
"""

import numpy as np


class L1QuantitativeError(Exception):
    """Raised when L1 inputs cannot produce an auditable quantitative conclusion."""


class FidScoreCalculator:
    """Calculates Fréchet distance between two feature distributions."""

    def score(self, real_features, generated_features):
        real = self._validated_features(real_features, "real")
        generated = self._validated_features(generated_features, "generated")
        if real.shape[1] != generated.shape[1]:
            raise L1QuantitativeError(f"feature dimensions differ: real={real.shape[1]}, generated={generated.shape[1]}")
        real_mean, real_covariance = self._statistics(real)
        generated_mean, generated_covariance = self._statistics(generated)
        mean_distance = np.sum((real_mean - generated_mean) ** 2)
        covariance_trace = np.trace(real_covariance + generated_covariance)
        covariance_sqrt_trace = self._symmetric_product_sqrt_trace(real_covariance, generated_covariance)
        return float(max(mean_distance + covariance_trace - 2.0 * covariance_sqrt_trace, 0.0))

    def _validated_features(self, features, label):
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
            raise L1QuantitativeError(f"{label} features must be a finite 2D array with at least two rows")
        if not np.isfinite(array).all():
            raise L1QuantitativeError(f"{label} features contain non-finite values")
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
