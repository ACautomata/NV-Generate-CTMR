# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantitative L1 MR acceptance for frozen BraTS phase candidates (issue #54)."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # repo src layout: python -m scripts.<this>
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # flat sugon deployment: src/ synced next to the script

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity

from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError  # noqa: E402

SCHEMA = "brats-l1-report/1"
FEATURE_EXTRACTOR = "radimagenet_resnet50"
MR_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad"


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


class L1ReportProducer:
    """Builds the versioned, candidate-bound L1 quantitative report."""

    MODALITIES = ("t1n", "t1c", "t2w", "t2f")

    def __init__(self, protocol):
        self._protocol = protocol
        self._fid_assessor = ThreePlaneFidAssessor(FidScoreCalculator(), protocol)
        self._p3_assessor = P3DirectionAssessor(protocol)

    @staticmethod
    def _run_binding(run_record):
        """The frozen-candidate five-key binding with the freeze gate built in (ADR-0012 决定 4)."""
        try:
            return FrozenRunBinding.from_record(run_record)
        except FrozenRunBindingError as error:
            raise L1QuantitativeError(str(error)) from error

    def produce(self, run_record, challenges, feature_records, p3_observations, feature_protocol):
        binding = self._run_binding(run_record)
        catalog = FeatureCohortCatalog(feature_records)
        fid_results = self._fid_results(binding.phase, tuple(challenges), catalog)
        p3_results = self._p3_results(binding.phase, tuple(challenges), p3_observations)
        return {
            "schema": SCHEMA,
            "binding": binding.as_dict(),
            "protocol": {
                "feature_extractor": feature_protocol["feature_extractor"],
                "mr_preprocessing": feature_protocol["mr_preprocessing"],
                "planes": list(ThreePlaneFidAssessor.PLANES),
                "bootstrap": {
                    "method": "case_level_percentile_pcg64",
                    "confidence_level": self._protocol.confidence_level,
                    "resamples": self._protocol.resamples,
                    "seed": self._protocol.seed,
                },
                "fid_multiplier": 2.5,
                "p3_pair_metrics": {"mae": "whole_volume", "ssim": "3d_win_size_7_data_range_1"},
            },
            "fid_results": fid_results,
            "p3_paired_results": p3_results,
            "summary": {"verdict": self._summary_verdict(fid_results, p3_results)},
        }

    def _fid_results(self, phase, challenges, catalog):
        results = []
        for challenge in challenges:
            for modality in self.MODALITIES:
                source_modalities = catalog.generated_source_modalities(challenge, modality)
                try:
                    if phase == "P3" and set(source_modalities) != {source for source in self.MODALITIES if source != modality}:
                        raise L1QuantitativeError(f"P3 target {modality} FID must cover every src!=tgt direction, got {source_modalities}")
                    baseline = self._fid_assessor.assess(catalog.cohort(challenge, modality, "train"), catalog.cohort(challenge, modality, "holdout"))
                    generated = self._fid_assessor.assess(
                        catalog.cohort(challenge, modality, "holdout"), catalog.cohort(challenge, modality, "generated")
                    )
                    threshold = 2.5 * baseline.mean_bootstrap_median
                    verdict = "pass" if generated.mean.ci95[1] <= threshold else "fail"
                    results.append(
                        {
                            "challenge": challenge,
                            "target_modality": modality,
                            "generated_source_modalities": list(source_modalities),
                            "generated_vs_holdout": self._fid_result_dict(generated),
                            "train_vs_holdout_baseline": self._fid_result_dict(baseline),
                            "threshold": threshold,
                            "verdict": verdict,
                        }
                    )
                except L1QuantitativeError as error:
                    results.append(
                        {
                            "challenge": challenge,
                            "target_modality": modality,
                            "generated_source_modalities": list(source_modalities),
                            "generated_vs_holdout": None,
                            "train_vs_holdout_baseline": None,
                            "threshold": None,
                            "verdict": "undecided",
                            "reason": str(error),
                        }
                    )
        return results

    def _fid_result_dict(self, result):
        return {
            "planes": {plane: self._interval_dict(result.planes[plane]) for plane in ThreePlaneFidAssessor.PLANES},
            "mean": self._interval_dict(result.mean),
            "mean_bootstrap_median": result.mean_bootstrap_median,
        }

    def _p3_results(self, phase, challenges, observations):
        if phase != "P3":
            return []
        grouped = {}
        for observation in observations:
            grouped.setdefault((observation.challenge, observation.src_modality, observation.target_modality), []).append(observation)
        results = []
        for challenge in challenges:
            for source in self.MODALITIES:
                for target in self.MODALITIES:
                    if source == target:
                        continue
                    key = (challenge, source, target)
                    try:
                        result = self._p3_assessor.assess(grouped[key])
                        results.append(self._p3_result_dict(result))
                    except (KeyError, L1QuantitativeError) as error:
                        results.append(
                            {
                                "challenge": challenge,
                                "src_modality": source,
                                "target_modality": target,
                                "case_count": 0,
                                "mae_relative_reduction": None,
                                "ssim_increase": None,
                                "gate_applicable": key[1:] != ("t1n", "t1c"),
                                "verdict": "undecided",
                                "reason": str(error),
                            }
                        )
        return results

    def _p3_result_dict(self, result):
        return {
            "challenge": result.challenge,
            "src_modality": result.src_modality,
            "target_modality": result.target_modality,
            "case_count": result.case_count,
            "mae_relative_reduction": self._interval_dict(result.mae_relative_reduction),
            "ssim_increase": self._interval_dict(result.ssim_increase),
            "gate_applicable": result.gate_applicable,
            "verdict": result.verdict,
        }

    def _interval_dict(self, interval):
        return {"point": interval.point, "ci95": list(interval.ci95)}

    def _summary_verdict(self, fid_results, p3_results):
        verdicts = [result["verdict"] for result in fid_results]
        verdicts += [result["verdict"] for result in p3_results if result["gate_applicable"]]
        if "undecided" in verdicts:
            return "undecided"
        if "fail" in verdicts:
            return "fail"
        return "pass"


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


class L1SelfTest:
    """Synthetic public-CLI checks for deterministic L1 statistical behavior."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._check_fid_location_shift()
        self._check_case_bootstrap_is_deterministic()
        self._check_mr_normalizer_clips_metric_range()
        self._check_p3_pair_gate_and_t1n_t1c_exception()
        self._check_report_binding_and_relative_fid_gate()
        self._check_evaluate_command_writes_bound_report()
        self._check_p3_evaluate_command_writes_paired_results()
        self._check_public_report_output_is_rejected()
        return self.failures

    def _check_fid_location_shift(self):
        real_features = np.array([[0.0], [2.0]])
        generated_features = np.array([[1.0], [3.0]])
        fid = FidScoreCalculator().score(real_features, generated_features)
        if not np.isclose(fid, 1.0):
            self.failures.append(f"FID location-shift fixture expected 1.0, got {fid}")

    def _check_case_bootstrap_is_deterministic(self):
        real_cases = {"case-a": np.array([[0.0]]), "case-b": np.array([[2.0]])}
        generated_cases = {"case-a": np.array([[1.0]]), "case-b": np.array([[3.0]])}
        protocol = BootstrapProtocol(resamples=32, seed=20260821)
        first = CaseFeatureBootstrap(FidScoreCalculator(), protocol).evaluate(real_cases, generated_cases)
        second = CaseFeatureBootstrap(FidScoreCalculator(), protocol).evaluate(real_cases, generated_cases)
        if not np.isclose(first.point, 1.0):
            self.failures.append(f"case bootstrap point FID expected 1.0, got {first.point}")
        if first != second:
            self.failures.append("case bootstrap must be deterministic for a fixed PCG64 seed")

    def _check_mr_normalizer_clips_metric_range(self):
        volume = np.linspace(0.0, 100.0, 7**3).reshape((7, 7, 7))
        normalized = MRIntensityNormalizer().normalize(volume, "selftest")
        if normalized.min() < 0.0 or normalized.max() > 1.0:
            self.failures.append("MR metric normalization must remain within the declared [0,1] range")

    def _check_p3_pair_gate_and_t1n_t1c_exception(self):
        reference = np.linspace(0.0, 1.0, 7**3).reshape((7, 7, 7))
        observations = [P3PairObservation("GLI", f"case-{index}", "t2w", "t1n", reference, reference + 0.2, reference + 0.01) for index in range(2)]
        assessor = P3DirectionAssessor(BootstrapProtocol(resamples=32, seed=20260821))
        informative = assessor.assess(observations)
        exceptional = assessor.assess(
            [P3PairObservation("GLI", f"case-{index}", "t1n", "t1c", reference, reference + 0.2, reference + 0.01) for index in range(2)]
        )
        if informative.verdict != "pass":
            self.failures.append(f"P3 informative pair expected pass, got {informative.verdict}")
        if exceptional.verdict != "not_applicable_known_unobservable":
            self.failures.append(f"P3 t1n->t1c expected exception, got {exceptional.verdict}")

    def _check_report_binding_and_relative_fid_gate(self):
        run = {
            "run_id": "p1-fixture",
            "phase": "P1",
            "status": "frozen",
            "manifest": {"sha256": "manifest-sha"},
            "selection": {"checkpoint": {"sha256": "candidate-sha"}},
            "samples": {"sha256": "samples-sha"},
        }
        records = []
        values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            for cohort, cohort_values in values.items():
                for plane in ("xy", "yz", "zx"):
                    for index, value in enumerate(cohort_values):
                        records.append(FeatureRecord(cohort, "GLI", f"case-{index}", modality, plane, np.array([[value]])))
        report = L1ReportProducer(BootstrapProtocol(resamples=32, seed=20260821)).produce(
            run,
            ("GLI",),
            records,
            [],
            {
                "feature_extractor": {"name": FEATURE_EXTRACTOR, "weights_sha256": "f" * 64},
                "mr_preprocessing": MR_PREPROCESSING,
            },
        )
        if report["binding"]["candidate_checkpoint_sha256"] != "candidate-sha":
            self.failures.append("L1 report did not bind the frozen candidate checkpoint")
        if report["summary"]["verdict"] != "pass":
            self.failures.append(f"relative FID fixture expected pass, got {report['summary']['verdict']}")
        first_fid = report["fid_results"][0]
        bootstrap_median = first_fid["train_vs_holdout_baseline"].get("mean_bootstrap_median")
        if bootstrap_median is None or not np.isclose(first_fid["threshold"], 2.5 * bootstrap_median):
            self.failures.append("relative FID gate must use the real train-vs-holdout bootstrap median")

    def _check_evaluate_command_writes_bound_report(self):
        root = self._workdir / "evaluate-command"
        features_dir = root / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        phase_manifest_path = root / "phase_manifest.json"
        phase_manifest_path.write_text(json.dumps({"challenges": {"GLI": {"cases": {}}}}))
        run_path = root / "run.json"
        run_path.write_text(
            json.dumps(
                {
                    "run_id": "p1-cli-fixture",
                    "phase": "P1",
                    "status": "frozen",
                    "manifest": {"path": str(phase_manifest_path), "sha256": "manifest-sha"},
                    "selection": {"checkpoint": {"sha256": "candidate-sha"}},
                    "samples": {"sha256": "samples-sha"},
                }
            )
        )
        rows = []
        values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            for cohort, cohort_values in values.items():
                for plane in ("xy", "yz", "zx"):
                    for index, value in enumerate(cohort_values):
                        feature_path = features_dir / f"{cohort}-{modality}-{plane}-{index}.npy"
                        np.save(feature_path, np.array([[value]]))
                        rows.append(
                            {
                                "cohort": cohort,
                                "challenge": "GLI",
                                "case": f"case-{index}",
                                "target_modality": modality,
                                "plane": plane,
                                "path": str(feature_path),
                            }
                        )
        features_path = root / "features.json"
        features_path.write_text(
            json.dumps(
                {
                    "schema": "brats-l1-features/1",
                    "protocol": {
                        "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
                        "mr_preprocessing": "percentile_0_99.5_to_0_1_ras_1mm_zero_pad",
                    },
                    "records": rows,
                }
            )
        )
        invalid_features = json.loads(features_path.read_text())
        invalid_features["protocol"].pop("mr_preprocessing")
        invalid_features_path = root / "invalid_features.json"
        invalid_features_path.write_text(json.dumps(invalid_features))
        try:
            FeatureManifestReader(ControlledJsonReader()).read(invalid_features_path)
        except L1QuantitativeError:
            pass
        else:
            self.failures.append("feature manifest without fixed MR preprocessing provenance was accepted")
        report_path = root / "l1_report.json"
        status = CommandLine().run(
            [
                "evaluate",
                "--run",
                str(run_path),
                "--features",
                str(features_path),
                "--output",
                str(report_path),
                "--bootstrap-resamples",
                "32",
                "--seed",
                "20260821",
            ]
        )
        if status != 0:
            self.failures.append(f"evaluate command returned {status}")
        elif json.loads(report_path.read_text())["binding"]["run_id"] != "p1-cli-fixture":
            self.failures.append("evaluate command wrote an incorrectly bound report")

    def _check_p3_evaluate_command_writes_paired_results(self):
        root = self._workdir / "p3-evaluate-command"
        features_dir = root / "features"
        volumes_dir = root / "volumes"
        features_dir.mkdir(parents=True, exist_ok=True)
        volumes_dir.mkdir(parents=True, exist_ok=True)
        phase_manifest_path = root / "phase_manifest.json"
        phase_manifest_path.write_text(json.dumps({"challenges": {"GLI": {"cases": {}}}}))
        run_path = root / "run.json"
        run_path.write_text(
            json.dumps(
                {
                    "run_id": "p3-cli-fixture",
                    "phase": "P3",
                    "status": "frozen",
                    "manifest": {"path": str(phase_manifest_path), "sha256": "manifest-sha"},
                    "selection": {"checkpoint": {"sha256": "candidate-sha"}},
                    "samples": {"sha256": "samples-sha"},
                }
            )
        )
        feature_rows = []
        values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
        for modality in ("t1n", "t1c", "t2w", "t2f"):
            for cohort, cohort_values in values.items():
                sources = (None,) if cohort != "generated" else tuple(source for source in ("t1n", "t1c", "t2w", "t2f") if source != modality)
                for source in sources:
                    for plane in ("xy", "yz", "zx"):
                        for index, value in enumerate(cohort_values):
                            suffix = "" if source is None else f"-{source}"
                            feature_path = features_dir / f"{cohort}-{modality}-{plane}-{index}{suffix}.npy"
                            np.save(feature_path, np.array([[value]]))
                            feature_rows.append(
                                {
                                    "cohort": cohort,
                                    "challenge": "GLI",
                                    "case": f"case-{index}",
                                    "target_modality": modality,
                                    "plane": plane,
                                    "path": str(feature_path),
                                    "src_modality": source,
                                }
                            )
        features_path = root / "features.json"
        features_path.write_text(
            json.dumps(
                {
                    "schema": "brats-l1-features/1",
                    "protocol": {
                        "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
                        "mr_preprocessing": "percentile_0_99.5_to_0_1_ras_1mm_zero_pad",
                    },
                    "records": feature_rows,
                }
            )
        )
        reference = np.linspace(0.0, 1.0, 7**3).reshape((7, 7, 7))
        pair_rows = []
        for source in ("t1n", "t1c", "t2w", "t2f"):
            for target in ("t1n", "t1c", "t2w", "t2f"):
                if source == target:
                    continue
                for index in range(2):
                    paths = {}
                    for label, volume in {
                        "reference": reference,
                        "baseline": reference**2,
                        "candidate": reference + (0.01 + index * 0.005) * np.sin(np.pi * reference),
                    }.items():
                        path = volumes_dir / f"{source}-{target}-{index}-{label}.nii.gz"
                        nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
                        paths[label] = str(path)
                    pair_rows.append(
                        {
                            "challenge": "GLI",
                            "case": f"case-{index}",
                            "src_modality": source,
                            "target_modality": target,
                            **paths,
                        }
                    )
        pairs_path = root / "pairs.json"
        pairs_path.write_text(json.dumps({"schema": "brats-l1-pairs/1", "records": pair_rows}))
        report_path = root / "l1_report.json"
        status = CommandLine().run(
            [
                "evaluate",
                "--run",
                str(run_path),
                "--features",
                str(features_path),
                "--pairs",
                str(pairs_path),
                "--output",
                str(report_path),
                "--bootstrap-resamples",
                "32",
                "--seed",
                "20260821",
            ]
        )
        if status != 0:
            self.failures.append(f"P3 evaluate command returned {status}")
            return
        report = json.loads(report_path.read_text())
        p3_results = report["p3_paired_results"]
        t1n_t1c = next(result for result in p3_results if (result["src_modality"], result["target_modality"]) == ("t1n", "t1c"))
        t1c_fid = next(result for result in report["fid_results"] if result["target_modality"] == "t1c")
        if len(p3_results) != 12 or t1n_t1c["verdict"] != "not_applicable_known_unobservable":
            self.failures.append("P3 evaluate command did not preserve all directions and the t1n->t1c exception")
        if "t1n" not in t1c_fid.get("generated_source_modalities", []):
            self.failures.append("P3 t1n->t1c must remain covered by the target-t1c FID report")
        if report["summary"]["verdict"] != "pass":
            self.failures.append("P3 FID must aggregate all same-case source directions without becoming undecided")

    def _check_public_report_output_is_rejected(self):
        output = self._workdir / "public-report-repo" / "l1_report.json"
        (output.parent / ".git").mkdir(parents=True, exist_ok=True)
        try:
            L1ReportWriter().write({"fixture": True}, output)
        except L1QuantitativeError:
            return
        self.failures.append("L1 report writer accepted a path inside a git work tree")


FEATURE_SCHEMA = "brats-l1-features/1"


@dataclass(frozen=True)
class FeatureManifest:
    """Controlled FID feature records and their extractor provenance."""

    records: tuple[FeatureRecord, ...]
    protocol: dict


class ControlledJsonReader:
    """Reads an auditable JSON document from controlled storage."""

    def read(self, path, label):
        resolved = Path(path)
        if not resolved.is_file():
            raise L1QuantitativeError(f"{label} not found: {resolved}")
        try:
            return json.loads(resolved.read_text())
        except json.JSONDecodeError as error:
            raise L1QuantitativeError(f"{label} is not valid JSON: {resolved} ({error})") from error


class FrozenRunRecordReader:
    """Loads a frozen phase run and its pinned challenge manifest."""

    def __init__(self, documents):
        self._documents = documents

    def read(self, path):
        record = self._documents.read(path, "run record")
        try:
            FrozenRunBinding.from_record(record)  # freeze gate: binding extract validates the run state
        except FrozenRunBindingError as error:
            raise L1QuantitativeError(str(error)) from error
        return record

    def challenges(self, record):
        try:
            manifest_path = record["manifest"]["path"]
            manifest = self._documents.read(manifest_path, "phase manifest")
            challenges = tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"run record has no readable phase manifest: {error}") from error
        if not challenges:
            raise L1QuantitativeError("phase manifest has no challenges")
        return challenges


class FeatureManifestReader:
    """Loads case-level three-plane feature arrays without accessing an extractor or network."""

    def __init__(self, documents):
        self._documents = documents

    def read(self, path):
        manifest_path = Path(path)
        payload = self._documents.read(manifest_path, "L1 feature manifest")
        if payload.get("schema") != FEATURE_SCHEMA:
            raise L1QuantitativeError(f"feature manifest schema must be {FEATURE_SCHEMA!r}")
        protocol = payload.get("protocol")
        extractor = protocol.get("feature_extractor") if isinstance(protocol, dict) else None
        if not isinstance(extractor, dict) or extractor.get("name") != FEATURE_EXTRACTOR or not self._sha256(extractor.get("weights_sha256")):
            raise L1QuantitativeError(f"feature manifest must record {FEATURE_EXTRACTOR} and a SHA-256 weights hash")
        if protocol.get("mr_preprocessing") != MR_PREPROCESSING:
            raise L1QuantitativeError(f"feature manifest mr_preprocessing must be {MR_PREPROCESSING}")
        records = tuple(self._record(manifest_path.parent, row) for row in payload.get("records", []))
        if not records:
            raise L1QuantitativeError("feature manifest has no records")
        return FeatureManifest(records=records, protocol=protocol)

    def _sha256(self, value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    def _record(self, root, row):
        try:
            relative_path = Path(row["path"])
            feature_path = relative_path if relative_path.is_absolute() else root / relative_path
            if not feature_path.is_file():
                raise L1QuantitativeError(f"feature array not found: {feature_path}")
            return FeatureRecord(
                cohort=row["cohort"],
                challenge=row["challenge"],
                case=row["case"],
                target_modality=row["target_modality"],
                plane=row["plane"],
                features=np.load(feature_path, allow_pickle=False),
                src_modality=row.get("src_modality"),
            )
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"feature manifest record is incomplete: {error}") from error


PAIR_SCHEMA = "brats-l1-pairs/1"


@dataclass(frozen=True)
class NiftiVolume:
    """A loaded MR volume and its spatial transform."""

    data: np.ndarray
    affine: np.ndarray


# Reverse shim (ticket 08 / ADR-0015 §2): the pinned MR [0,1] intensity protocol
# moved to ctmr.domain.intensity_protocol; this module re-exports it so its
# consumers (and the legacy evaluate chain) keep working until the L1 batch
# relocates them. The protocol's error class is IntensityProtocolError (message
# text unchanged) -- nothing here catches it by type.
from ctmr.domain.intensity_protocol import MRIntensityNormalizer  # noqa: E402  (module-level position preserved)


class NiftiVolumeReader:
    """Loads a NIfTI image without resampling its established evaluation grid."""

    def read(self, path, label):
        image_path = Path(path)
        if not image_path.is_file():
            raise L1QuantitativeError(f"{label} NIfTI not found: {image_path}")
        try:
            image = nib.load(image_path)
            return NiftiVolume(data=image.get_fdata(dtype=np.float64), affine=image.affine)
        except (OSError, ValueError) as error:
            raise L1QuantitativeError(f"{label} NIfTI cannot be read: {image_path} ({error})") from error


class P3PairManifestReader:
    """Loads same-case P3 target, stage-0 baseline, and candidate NIfTI triplets."""

    def __init__(self, documents, volumes, normalizer):
        self._documents = documents
        self._volumes = volumes
        self._normalizer = normalizer

    def read(self, path):
        manifest_path = Path(path)
        payload = self._documents.read(manifest_path, "P3 pair manifest")
        if payload.get("schema") != PAIR_SCHEMA:
            raise L1QuantitativeError(f"P3 pair manifest schema must be {PAIR_SCHEMA!r}")
        records = tuple(self._record(manifest_path.parent, row) for row in payload.get("records", []))
        if not records:
            raise L1QuantitativeError("P3 pair manifest has no records")
        return records

    def _record(self, root, row):
        try:
            reference = self._load(root, row["reference"], "reference")
            baseline = self._load(root, row["baseline"], "stage-0 baseline")
            candidate = self._load(root, row["candidate"], "candidate")
            self._same_geometry(reference, baseline, candidate, row["case"])
            return P3PairObservation(
                challenge=row["challenge"],
                case=row["case"],
                src_modality=row["src_modality"],
                target_modality=row["target_modality"],
                reference=self._normalizer.normalize(reference.data, "P3 reference"),
                baseline=self._normalizer.normalize(baseline.data, "P3 stage-0 baseline"),
                candidate=self._normalizer.normalize(candidate.data, "P3 candidate"),
            )
        except (KeyError, TypeError) as error:
            raise L1QuantitativeError(f"P3 pair record is incomplete: {error}") from error

    def _load(self, root, text_path, label):
        path = Path(text_path)
        return self._volumes.read(path if path.is_absolute() else root / path, label)

    def _same_geometry(self, reference, baseline, candidate, case):
        volumes = (reference, baseline, candidate)
        if any(volume.data.shape != reference.data.shape for volume in volumes[1:]):
            raise L1QuantitativeError(f"P3 pair {case} has mismatched NIfTI shapes")
        if any(not np.allclose(volume.affine, reference.affine) for volume in volumes[1:]):
            raise L1QuantitativeError(f"P3 pair {case} has mismatched NIfTI affines")


class L1ReportWriter:
    """Persists the final machine-readable report without allowing NaN JSON values."""

    def write(self, report, path):
        output = Path(path)
        self._assert_controlled(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        except (TypeError, ValueError) as error:
            raise L1QuantitativeError(f"L1 report cannot be represented as strict JSON: {error}") from error
        return output

    def _assert_controlled(self, output):
        for parent in output.resolve().parents:
            if (parent / ".git").exists():
                raise L1QuantitativeError(f"L1 report output lives inside a git work tree ({parent}); controlled reports must stay outside the repo")


class L1EvaluationCommand:
    """Coordinates frozen-run evidence into a controlled L1 report."""

    def __init__(self, run_reader, feature_reader, pair_reader, writer):
        self._run_reader = run_reader
        self._feature_reader = feature_reader
        self._pair_reader = pair_reader
        self._writer = writer

    def evaluate(self, run_path, feature_path, pair_path, output_path, bootstrap_protocol):
        record = self._run_reader.read(run_path)
        features = self._feature_reader.read(feature_path)
        if record["phase"] == "P3" and pair_path is None:
            raise L1QuantitativeError("P3 L1 assessment requires --pairs with stage-0 and candidate same-case volumes")
        if record["phase"] != "P3" and pair_path is not None:
            raise L1QuantitativeError("--pairs applies only to P3 L1 assessment")
        observations = self._pair_reader.read(pair_path) if pair_path is not None else []
        report = L1ReportProducer(bootstrap_protocol).produce(
            record,
            self._run_reader.challenges(record),
            features.records,
            observations,
            features.protocol,
        )
        return self._writer.write(report, output_path)


class CommandLine:
    """The L1 CLI command dispatcher."""

    def run(self, argv):
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="command", required=True)
        selftest = sub.add_parser("selftest", help="run synthetic L1 acceptance checks")
        selftest.add_argument("--workdir", required=True)
        evaluate = sub.add_parser("evaluate", help="write a candidate-bound L1 report from controlled evidence")
        evaluate.add_argument("--run", required=True, help="frozen brats-phase-run record")
        evaluate.add_argument("--features", required=True, help=f"controlled {FEATURE_SCHEMA} manifest")
        evaluate.add_argument("--pairs", help="P3 only: stage-0/candidate/reference pair manifest")
        evaluate.add_argument("--output", required=True, help="controlled output report path")
        evaluate.add_argument("--bootstrap-resamples", type=int, default=1000)
        evaluate.add_argument("--seed", type=int, default=20260821)
        args = parser.parse_args(argv)
        if args.command == "evaluate":
            documents = ControlledJsonReader()
            report_path = L1EvaluationCommand(
                FrozenRunRecordReader(documents),
                FeatureManifestReader(documents),
                P3PairManifestReader(documents, NiftiVolumeReader(), MRIntensityNormalizer()),
                L1ReportWriter(),
            ).evaluate(
                args.run,
                args.features,
                args.pairs,
                args.output,
                BootstrapProtocol(resamples=args.bootstrap_resamples, seed=args.seed),
            )
            print(f"L1 report written -> {report_path}")
            return 0
        failures = L1SelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0


def main(argv=None):
    """Run the L1 quantitative CLI."""
    try:
        return CommandLine().run(argv)
    except L1QuantitativeError as error:
        print(f"L1 INPUT ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
