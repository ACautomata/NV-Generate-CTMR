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

"""Versioned candidate-bound L1 quantitative report assembly.

Migrated verbatim from ``scripts/brats_l1_quantitative.py`` (#141). The
producer binds the report to the frozen candidate's five keys via the shared
``FrozenRunBinding`` (freeze gate built in), assesses per-challenge FID and
P3 paired evidence, and derives the summary verdict with the undecided third
state -- an assessment that could not run (incomplete evidence) records
``undecided`` with its reason instead of guessing.
"""

from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.quantitative.fid import (
    FeatureCohortCatalog,
    FidScoreCalculator,
    L1QuantitativeError,
    ThreePlaneFidAssessor,
)
from ctmr.application.acceptance.quantitative.paired import P3DirectionAssessor

SCHEMA = "brats-l1-report/1"


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
