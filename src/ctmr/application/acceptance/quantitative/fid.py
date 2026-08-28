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

"""Fréchet-distance metric, case-level bootstrap, and the three-plane FID assessment.

``FidScoreCalculator`` is the shared metric face consumed by the quantitative
chain and the generation families' dev-trend machinery alike (ticket 10); the
pre-registered bootstrap protocol, the case-resampling assessor and the
feature-cohort catalog migrated verbatim from ``scripts/brats_l1_quantitative.py``
(#141). Pure numpy: no IO, no feature extractor, no network.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class BootstrapProtocol:
    """The pre-registered percentile-bootstrap parameters for an L1 report."""

    resamples: int
    seed: int
    confidence_level: float = 0.95

    def __post_init__(self):
        if self.resamples < 1:
            raise L1QuantitativeError("bootstrap resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise L1QuantitativeError("bootstrap confidence_level must be between zero and one")


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate and percentile confidence interval."""

    point: float
    ci95: tuple[float, float]


class CaseFeatureBootstrap:
    """Resamples complete cases before calculating an unpaired FID interval."""

    def __init__(self, calculator, protocol):
        self._calculator = calculator
        self._protocol = protocol

    def evaluate(self, real_cases, generated_cases):
        real = self._validated_cases(real_cases, "real")
        generated = self._validated_cases(generated_cases, "generated")
        point = self._calculator.score(self._stack_cases(real, sorted(real)), self._stack_cases(generated, sorted(generated)))
        random = np.random.Generator(np.random.PCG64(self._protocol.seed))
        samples = []
        real_keys = tuple(sorted(real))
        generated_keys = tuple(sorted(generated))
        for _ in range(self._protocol.resamples):
            sampled_real = random.choice(real_keys, size=len(real_keys), replace=True)
            sampled_generated = random.choice(generated_keys, size=len(generated_keys), replace=True)
            samples.append(self._calculator.score(self._stack_cases(real, sampled_real), self._stack_cases(generated, sampled_generated)))
        lower, upper = np.quantile(
            samples,
            [(1.0 - self._protocol.confidence_level) / 2.0, (1.0 + self._protocol.confidence_level) / 2.0],
        )
        return BootstrapInterval(point=point, ci95=(float(lower), float(upper)))

    def _validated_cases(self, cases, label):
        if not isinstance(cases, dict) or len(cases) < 2:
            raise L1QuantitativeError(f"{label} feature cohort needs at least two complete cases")
        return cases

    def _stack_cases(self, cases, keys):
        return np.concatenate([np.asarray(cases[key], dtype=np.float64) for key in keys], axis=0)


@dataclass(frozen=True)
class FeatureRecord:
    """One case and plane of controlled RadImageNet feature vectors."""

    cohort: str
    challenge: str
    case: str
    target_modality: str
    plane: str
    features: np.ndarray
    src_modality: str | None = None


@dataclass(frozen=True)
class ThreePlaneFidResult:
    """FID estimates for each orthogonal plane, their CI, and bootstrap median."""

    planes: dict
    mean: BootstrapInterval
    mean_bootstrap_median: float


class ThreePlaneFidAssessor:
    """Calculates case-resampled FID for XY, YZ, ZX, and their arithmetic mean."""

    PLANES = ("xy", "yz", "zx")

    def __init__(self, calculator, protocol):
        self._calculator = calculator
        self._protocol = protocol

    def assess(self, real_by_plane, generated_by_plane):
        real = self._validated_cohort(real_by_plane, "real")
        generated = self._validated_cohort(generated_by_plane, "generated")
        point_scores = self._scores(real, generated, tuple(sorted(real["xy"])), tuple(sorted(generated["xy"])))
        random = np.random.Generator(np.random.PCG64(self._protocol.seed))
        samples = {plane: [] for plane in self.PLANES}
        mean_samples = []
        real_keys = tuple(sorted(real["xy"]))
        generated_keys = tuple(sorted(generated["xy"]))
        for _ in range(self._protocol.resamples):
            score = self._scores(
                real,
                generated,
                random.choice(real_keys, size=len(real_keys), replace=True),
                random.choice(generated_keys, size=len(generated_keys), replace=True),
            )
            for plane in self.PLANES:
                samples[plane].append(score[plane])
            mean_samples.append(float(np.mean(tuple(score.values()))))
        return ThreePlaneFidResult(
            planes={plane: self._interval(point_scores[plane], samples[plane]) for plane in self.PLANES},
            mean=self._interval(float(np.mean(tuple(point_scores.values()))), mean_samples),
            mean_bootstrap_median=float(np.median(mean_samples)),
        )

    def _validated_cohort(self, by_plane, label):
        if set(by_plane) != set(self.PLANES):
            raise L1QuantitativeError(f"{label} feature cohort must contain exactly {self.PLANES} planes")
        cases = None
        for plane in self.PLANES:
            plane_cases = by_plane[plane]
            if len(plane_cases) < 2:
                raise L1QuantitativeError(f"{label} {plane} features need at least two complete cases")
            keys = set(plane_cases)
            if cases is None:
                cases = keys
            elif keys != cases:
                raise L1QuantitativeError(f"{label} feature cases differ between orthogonal planes")
        return by_plane

    def _scores(self, real, generated, real_keys, generated_keys):
        return {
            plane: self._calculator.score(self._stack(real[plane], real_keys), self._stack(generated[plane], generated_keys)) for plane in self.PLANES
        }

    def _stack(self, cases, keys):
        return np.concatenate([np.asarray(cases[key], dtype=np.float64) for key in keys], axis=0)

    def _interval(self, point, samples):
        lower, upper = np.quantile(
            samples,
            [(1.0 - self._protocol.confidence_level) / 2.0, (1.0 + self._protocol.confidence_level) / 2.0],
        )
        return BootstrapInterval(point=float(point), ci95=(float(lower), float(upper)))


class FeatureCohortCatalog:
    """Indexes controlled feature records by cohort, challenge, target modality, plane, and case."""

    def __init__(self, records):
        self._records = records

    def cohort(self, challenge, target_modality, cohort):
        grouped = {plane: {} for plane in ThreePlaneFidAssessor.PLANES}
        for record in self._records:
            if (record.challenge, record.target_modality, record.cohort) != (challenge, target_modality, cohort):
                continue
            if record.plane not in grouped:
                raise L1QuantitativeError(f"unknown feature plane {record.plane!r}")
            source = record.src_modality if cohort == "generated" else None
            per_case = grouped[record.plane].setdefault(record.case, {})
            if source in per_case:
                raise L1QuantitativeError(f"duplicate feature record for {challenge}/{target_modality}/{record.plane}/{record.case}/{source}")
            per_case[source] = record.features
        return {
            plane: {case: np.concatenate(tuple(features.values()), axis=0) for case, features in grouped[plane].items()}
            for plane in ThreePlaneFidAssessor.PLANES
        }

    def generated_source_modalities(self, challenge, target_modality):
        return tuple(
            sorted(
                {
                    record.src_modality
                    for record in self._records
                    if (record.challenge, record.target_modality, record.cohort) == (challenge, target_modality, "generated")
                    and record.src_modality is not None
                }
            )
        )
