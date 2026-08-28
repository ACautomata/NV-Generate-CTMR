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

"""Paired P3 candidate-vs-baseline error assessment (MAE / 3D SSIM).

Migrated verbatim from ``brats_l1_quantitative.py`` (retired scripts layer, git history) (#141): per-case
metrics on aligned MR volumes in the fixed [0, 1] protocol, the paired
case-level percentile bootstrap, and the pre-registered improvement gate with
the explicit ``t1n->t1c`` known-unobservable exception (CONTEXT.md T1→T1c
已知限制). The intensity protocol itself lives in ``ctmr.domain.intensity_protocol``.
"""

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity

from ctmr.application.acceptance.quantitative.fid import BootstrapInterval, L1QuantitativeError


@dataclass(frozen=True)
class P3PairObservation:
    """One same-case P3 target, stage-0 baseline, and candidate volume triplet."""

    challenge: str
    case: str
    src_modality: str
    target_modality: str
    reference: np.ndarray
    baseline: np.ndarray
    candidate: np.ndarray


@dataclass(frozen=True)
class P3PairMetrics:
    """Per-case error measurements for one P3 direction."""

    baseline_mae: float
    candidate_mae: float
    baseline_ssim: float
    candidate_ssim: float


@dataclass(frozen=True)
class P3DirectionResult:
    """The pre-registered P3 paired acceptance conclusion for one direction."""

    challenge: str
    src_modality: str
    target_modality: str
    case_count: int
    mae_relative_reduction: BootstrapInterval
    ssim_increase: BootstrapInterval
    gate_applicable: bool
    verdict: str


class P3PairMetricCalculator:
    """Measures MAE and 3D SSIM for aligned MR volumes in the fixed [0, 1] protocol."""

    def score(self, observation):
        reference, baseline, candidate = self._validated_volumes(observation)
        return P3PairMetrics(
            baseline_mae=float(np.mean(np.abs(reference - baseline))),
            candidate_mae=float(np.mean(np.abs(reference - candidate))),
            baseline_ssim=float(structural_similarity(reference, baseline, data_range=1.0, channel_axis=None, win_size=7)),
            candidate_ssim=float(structural_similarity(reference, candidate, data_range=1.0, channel_axis=None, win_size=7)),
        )

    def _validated_volumes(self, observation):
        volumes = tuple(np.asarray(volume, dtype=np.float64) for volume in (observation.reference, observation.baseline, observation.candidate))
        reference = volumes[0]
        if reference.ndim != 3 or any(volume.shape != reference.shape for volume in volumes[1:]):
            raise L1QuantitativeError(f"P3 pair {observation.case} must contain same-shape 3D volumes")
        if any(size < 7 for size in reference.shape):
            raise L1QuantitativeError(f"P3 pair {observation.case} has a dimension smaller than SSIM win_size=7")
        if not all(np.isfinite(volume).all() for volume in volumes):
            raise L1QuantitativeError(f"P3 pair {observation.case} contains non-finite values")
        return volumes


class P3DirectionAssessor:
    """Applies paired bootstrap and the pre-registered P3 improvement gate."""

    def __init__(self, protocol):
        self._protocol = protocol
        self._calculator = P3PairMetricCalculator()

    def assess(self, observations):
        records = self._validated_observations(observations)
        metrics = [self._calculator.score(observation) for observation in records]
        reductions = self._relative_mae_reductions(metrics, records[0].case)
        ssim_increases = np.array([metric.candidate_ssim - metric.baseline_ssim for metric in metrics])
        mae_interval, ssim_interval = self._paired_intervals(reductions, ssim_increases)
        gate_applicable = (records[0].src_modality, records[0].target_modality) != ("t1n", "t1c")
        verdict = self._verdict(mae_interval, ssim_interval, gate_applicable)
        return P3DirectionResult(
            challenge=records[0].challenge,
            src_modality=records[0].src_modality,
            target_modality=records[0].target_modality,
            case_count=len(records),
            mae_relative_reduction=mae_interval,
            ssim_increase=ssim_interval,
            gate_applicable=gate_applicable,
            verdict=verdict,
        )

    def _validated_observations(self, observations):
        if len(observations) < 2:
            raise L1QuantitativeError("P3 paired assessment needs at least two complete cases")
        expected = (observations[0].challenge, observations[0].src_modality, observations[0].target_modality)
        seen_cases = set()
        for observation in observations:
            identity = (observation.challenge, observation.src_modality, observation.target_modality)
            if identity != expected:
                raise L1QuantitativeError("P3 paired assessment must contain exactly one challenge and direction")
            if observation.case in seen_cases:
                raise L1QuantitativeError(f"P3 paired assessment has a duplicate case: {observation.case}")
            seen_cases.add(observation.case)
        return observations

    def _relative_mae_reductions(self, metrics, case):
        baseline = np.array([metric.baseline_mae for metric in metrics])
        if np.any(baseline <= 0.0):
            raise L1QuantitativeError(f"P3 pair {case} has zero baseline MAE; relative improvement is undefined")
        candidate = np.array([metric.candidate_mae for metric in metrics])
        return (baseline - candidate) / baseline

    def _paired_intervals(self, reductions, ssim_increases):
        random = np.random.Generator(np.random.PCG64(self._protocol.seed))
        size = len(reductions)
        mae_samples, ssim_samples = [], []
        for _ in range(self._protocol.resamples):
            indices = random.choice(size, size=size, replace=True)
            mae_samples.append(float(np.mean(reductions[indices])))
            ssim_samples.append(float(np.mean(ssim_increases[indices])))
        return self._interval(reductions, mae_samples), self._interval(ssim_increases, ssim_samples)

    def _interval(self, values, samples):
        lower, upper = np.quantile(
            samples,
            [(1.0 - self._protocol.confidence_level) / 2.0, (1.0 + self._protocol.confidence_level) / 2.0],
        )
        return BootstrapInterval(point=float(np.mean(values)), ci95=(float(lower), float(upper)))

    def _verdict(self, mae_interval, ssim_interval, gate_applicable):
        if not gate_applicable:
            return "not_applicable_known_unobservable"
        if mae_interval.point >= 0.10 and ssim_interval.point >= 0.02 and mae_interval.ci95[0] > 0.0 and ssim_interval.ci95[0] > 0.0:
            return "pass"
        return "fail"
