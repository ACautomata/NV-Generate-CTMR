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

"""Per-layer evidence validators: quantitative / distribution / expert review (ADR-0012, judgement stays layered).

Each validator checks the versioned report schema of its layer and the frozen-candidate
binding, then applies the layer's own Rules (protocol / coverage / result lines /
verdict recomputation / gate checks). The gate constants on this side and the
production-side mirrors (``brats_l1_quantitative`` / ``brats_l3_blind_eval``) stay
independently sourced: a production bug must not be let through by a same-source
checker (ADR-0006 referee independence). Report-schema strings and attached-report
kinds keep their frozen artifact values (``l1_report`` etc.); the capability-phase
codepoints live in CONTEXT.md word entries, issues and ADRs, not in code naming.
"""

import json
import math
from pathlib import Path

from ctmr.application.acceptance.contract.binding import FrozenRunBinding

# Quantitative-layer constants (validate/contract side; production-side constants stay independent).
QUANTITATIVE_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
QUANTITATIVE_PLANES = ("xy", "yz", "zx")
QUANTITATIVE_VERDICTS = ("pass", "fail", "undecided")
QUANTITATIVE_T1N_TO_T1C = ("t1n", "t1c")
QUANTITATIVE_FEATURE_EXTRACTOR = "radimagenet_resnet50"
QUANTITATIVE_MR_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad"

# Expert-review layer constants (gate mirror of scripts/brats_l3_blind_eval -- different source on purpose).
EXPERT_REVIEW_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
EXPERT_REVIEW_DIMENSIONS = (
    "overall_realism",
    "anatomical_plausibility",
    "tumor_authenticity",
    "artifact_slice_consistency",
)
EXPERT_REVIEW_TURING_WINDOW = (0.40, 0.60)
EXPERT_REVIEW_LIKERT_BOUND = 4.0
EXPERT_REVIEW_VERDICTS = ("pass", "fail")

# Distribution-layer constants.
DISTRIBUTION_CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")  # the frozen five; formal L2 evidence covers all
DISTRIBUTION_VERDICTS = ("pass", "fail", "undecided")


class QuantitativeReportValidator:
    """Validates the versioned quantitative evidence schema and its frozen-candidate binding."""

    REPORT_SCHEMA = "brats-l1-report/1"

    def __init__(self, schema=None):
        self._schema = schema or self.REPORT_SCHEMA

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L1 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L1 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L1 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        challenges = self._challenges(record, failures)
        self._protocol(report, failures)
        self._fid_results(record["phase"], report, challenges, failures)
        self._p3_results(record["phase"], report, challenges, failures)
        self._summary(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        if report.get("schema") != self._schema:
            failures.append(f"L1 report schema != {self._schema}")
        mismatches = FrozenRunBinding.mismatches_for(record, binding)
        if mismatches is None:
            failures.append("L1 report binding must be an object")
            return
        for key in mismatches:
            failures.append(f"L1 report binding {key} does not match frozen run")

    def _challenges(self, record, failures):
        try:
            manifest = json.loads(Path(record["manifest"]["path"]).read_text())
            return tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot read pinned manifest for L1 coverage: {error}")
            return ()

    def _protocol(self, report, failures):
        protocol = report.get("protocol")
        if not isinstance(protocol, dict):
            failures.append("L1 report protocol must be an object")
            return
        extractor = protocol.get("feature_extractor")
        if (
            not isinstance(extractor, dict)
            or extractor.get("name") != QUANTITATIVE_FEATURE_EXTRACTOR
            or not self._sha256(extractor.get("weights_sha256"))
        ):
            failures.append(f"L1 report must record {QUANTITATIVE_FEATURE_EXTRACTOR} and a SHA-256 weights hash")
        if protocol.get("mr_preprocessing") != QUANTITATIVE_MR_PREPROCESSING:
            failures.append(f"L1 report mr_preprocessing must be {QUANTITATIVE_MR_PREPROCESSING}")
        if tuple(protocol.get("planes", ())) != QUANTITATIVE_PLANES:
            failures.append(f"L1 report planes must be {QUANTITATIVE_PLANES}")
        bootstrap = protocol.get("bootstrap")
        if not isinstance(bootstrap, dict) or bootstrap.get("method") != "case_level_percentile_pcg64":
            failures.append("L1 report bootstrap must record case_level_percentile_pcg64")
        elif bootstrap.get("confidence_level") != 0.95 or not isinstance(bootstrap.get("resamples"), int):
            failures.append("L1 report bootstrap must record 95% CI and integer resamples")
        if protocol.get("fid_multiplier") != 2.5:
            failures.append("L1 report FID multiplier must be 2.5")

    def _fid_results(self, phase, report, challenges, failures):
        results = report.get("fid_results")
        if not isinstance(results, list):
            failures.append("L1 report fid_results must be a list")
            return
        expected = {(challenge, modality) for challenge in challenges for modality in QUANTITATIVE_MODALITIES}
        actual = {(item.get("challenge"), item.get("target_modality")) for item in results if isinstance(item, dict)}
        if len(results) != len(expected) or actual != expected:
            failures.append("L1 report must cover each pinned challenge and all four target modalities exactly once")
        for result in results:
            if isinstance(result, dict):
                self._fid_result(phase, result, failures)

    def _fid_result(self, phase, result, failures):
        sources = result.get("generated_source_modalities")
        if phase == "P3":
            expected_sources = {source for source in QUANTITATIVE_MODALITIES if source != result.get("target_modality")}
            if not isinstance(sources, list) or set(sources) != expected_sources:
                failures.append("P3 target-modality FID must record all src!=tgt generated_source_modalities")
        elif sources != []:
            failures.append(f"{phase} target-modality FID must not record P3 source modalities")
        verdict = result.get("verdict")
        if verdict not in QUANTITATIVE_VERDICTS:
            failures.append("L1 FID verdict must be pass, fail, or undecided")
            return
        if verdict == "undecided":
            failures.append("formal L1 report cannot attach an undecided FID result")
            return
        generated = result.get("generated_vs_holdout")
        baseline = result.get("train_vs_holdout_baseline")
        threshold = result.get("threshold")
        self._fid_bundle(generated, "generated-vs-holdout", failures)
        self._fid_bundle(baseline, "train-vs-holdout baseline", failures)
        if not self._number(threshold):
            failures.append("L1 FID threshold must be finite")
            return
        try:
            bootstrap_median = baseline["mean_bootstrap_median"]
            expected_threshold = 2.5 * bootstrap_median
            upper = generated["mean"]["ci95"][1]
        except (KeyError, TypeError, IndexError):
            failures.append("L1 FID baseline must record a finite mean_bootstrap_median")
            return
        if not self._number(bootstrap_median):
            failures.append("L1 FID baseline mean_bootstrap_median must be finite")
            return
        if not math.isclose(threshold, expected_threshold, rel_tol=1e-9, abs_tol=1e-12):
            failures.append("L1 FID threshold must equal 2.5 times real train-vs-holdout bootstrap median")
        expected_verdict = "pass" if upper <= threshold else "fail"
        if verdict != expected_verdict:
            failures.append("L1 FID verdict disagrees with its CI upper-bound gate")

    def _fid_bundle(self, bundle, label, failures):
        if not isinstance(bundle, dict):
            failures.append(f"L1 {label} FID bundle must be an object")
            return
        planes = bundle.get("planes")
        if not isinstance(planes, dict) or tuple(sorted(planes)) != QUANTITATIVE_PLANES:
            failures.append(f"L1 {label} FID bundle must contain xy/yz/zx")
        elif all(plane in planes for plane in QUANTITATIVE_PLANES):
            for plane in QUANTITATIVE_PLANES:
                self._interval(planes[plane], f"L1 {label} {plane} FID", failures)
        self._interval(bundle.get("mean"), f"L1 {label} mean FID", failures)

    def _p3_results(self, phase, report, challenges, failures):
        results = report.get("p3_paired_results")
        if phase != "P3":
            if results != []:
                failures.append(f"{phase} L1 report must not carry P3 paired results")
            return
        if not isinstance(results, list):
            failures.append("P3 L1 report p3_paired_results must be a list")
            return
        expected = {
            (challenge, source, target)
            for challenge in challenges
            for source in QUANTITATIVE_MODALITIES
            for target in QUANTITATIVE_MODALITIES
            if source != target
        }
        actual = {(item.get("challenge"), item.get("src_modality"), item.get("target_modality")) for item in results if isinstance(item, dict)}
        if len(results) != len(expected) or actual != expected:
            failures.append("P3 L1 report must cover all 12 ordered directions for every pinned challenge")
        for result in results:
            if isinstance(result, dict):
                self._p3_result(result, failures)

    def _p3_result(self, result, failures):
        source, target = result.get("src_modality"), result.get("target_modality")
        applicable = (source, target) != QUANTITATIVE_T1N_TO_T1C
        if result.get("gate_applicable") != applicable:
            failures.append("P3 L1 gate_applicable must preserve the t1n->t1c exception")
            return
        verdict = result.get("verdict")
        mae = result.get("mae_relative_reduction")
        ssim = result.get("ssim_increase")
        if not applicable:
            if verdict != "not_applicable_known_unobservable":
                failures.append("P3 t1n->t1c must be explicitly not_applicable_known_unobservable")
            self._interval(mae, "P3 t1n->t1c MAE diagnostic", failures)
            self._interval(ssim, "P3 t1n->t1c SSIM diagnostic", failures)
            return
        if verdict not in QUANTITATIVE_VERDICTS:
            failures.append("P3 paired verdict must be pass, fail, or undecided")
            return
        if verdict == "undecided":
            failures.append("formal L1 report cannot attach an undecided P3 paired result")
            return
        self._interval(mae, "P3 MAE relative reduction", failures)
        self._interval(ssim, "P3 SSIM increase", failures)
        try:
            expected = "pass" if mae["point"] >= 0.10 and ssim["point"] >= 0.02 and mae["ci95"][0] > 0.0 and ssim["ci95"][0] > 0.0 else "fail"
        except (KeyError, TypeError, IndexError):
            return
        if verdict != expected:
            failures.append("P3 paired verdict disagrees with the pre-registered MAE/SSIM gate")

    def _summary(self, report, failures):
        summary = report.get("summary")
        if not isinstance(summary, dict) or summary.get("verdict") not in QUANTITATIVE_VERDICTS:
            failures.append("L1 report summary verdict must be pass, fail, or undecided")
            return
        verdicts = [result.get("verdict") for result in report.get("fid_results", []) if isinstance(result, dict)]
        verdicts += [
            result.get("verdict")
            for result in report.get("p3_paired_results", [])
            if isinstance(result, dict) and result.get("gate_applicable") is True
        ]
        expected = "undecided" if "undecided" in verdicts else "fail" if "fail" in verdicts else "pass"
        if summary["verdict"] != expected:
            failures.append("L1 report summary verdict disagrees with its applicable FID/P3 results")

    def _interval(self, interval, label, failures):
        if not isinstance(interval, dict) or not self._number(interval.get("point")):
            failures.append(f"{label} must record a finite point estimate")
            return
        ci = interval.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2 or not all(self._number(value) for value in ci) or ci[0] > ci[1]:
            failures.append(f"{label} must record an ordered finite 95% CI")

    @staticmethod
    def verdict_of(report):
        """The quantitative layer verdict read from a report (registry ``verdict_reader``)."""
        return report.get("summary", {}).get("verdict")

    @staticmethod
    def blocked_reasons(report):
        """Traceable quantitative blockers: failing FID/paired-result criteria, never offset by other layers."""
        reasons = []
        for result in report.get("fid_results", []):
            if result.get("verdict") != "pass":
                reasons.append(f"L1 FID {result.get('challenge')}/{result.get('target_modality')}: {result.get('verdict')}")
        for result in report.get("p3_paired_results", []):
            if result.get("gate_applicable") and result.get("verdict") != "pass":
                reasons.append(
                    f"L1 P3 paired {result.get('challenge')}/{result.get('src_modality')}->{result.get('target_modality')}: {result.get('verdict')}"
                )
        return reasons

    def _number(self, value):
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)

    def _sha256(self, value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class ExpertReviewReportValidator:
    """Validates the versioned expert-review evidence schema and its frozen-candidate binding."""

    REPORT_SCHEMA = "brats-l3-report/1"

    def __init__(self, schema=None):
        self._schema = schema or self.REPORT_SCHEMA

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L3 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L3 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L3 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        challenges = self._challenges(record, failures)
        protocol = self._protocol(report, failures)
        self._coverage(challenges, report, protocol, failures)
        self._visual_turing(report, protocol, failures)
        self._likert(report, protocol, failures)
        self._verdict(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        if report.get("schema") != self._schema:
            failures.append(f"L3 report schema != {self._schema}")
        mismatches = FrozenRunBinding.mismatches_for(record, binding)
        if mismatches is None:
            failures.append("L3 report binding must be an object")
            return
        for key in mismatches:
            failures.append(f"L3 report binding {key} does not match frozen run")

    def _challenges(self, record, failures):
        try:
            manifest = json.loads(Path(record["manifest"]["path"]).read_text())
            return tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot read pinned manifest for L3 coverage: {error}")
            return ()

    def _protocol(self, report, failures):
        protocol = report.get("protocol")
        if not isinstance(protocol, dict):
            failures.append("L3 report protocol must be an object")
            return None
        reviewers = protocol.get("reviewers")
        if not isinstance(reviewers, int) or reviewers < 2:
            failures.append("L3 report must record at least two independent reviewers")
        dimensions = protocol.get("dimensions")
        if not isinstance(dimensions, list) or tuple(dimensions) != EXPERT_REVIEW_DIMENSIONS:
            failures.append(f"L3 report dimensions must be {EXPERT_REVIEW_DIMENSIONS}")
        modalities = protocol.get("target_modalities")
        if not isinstance(modalities, list) or tuple(modalities) != EXPERT_REVIEW_MODALITIES:
            failures.append(f"L3 report target modalities must be {EXPERT_REVIEW_MODALITIES}")
        window = protocol.get("visual_turing_ci_window")
        if not isinstance(window, list) or len(window) != 2 or window != list(EXPERT_REVIEW_TURING_WINDOW):
            failures.append(f"L3 report visual-Turing CI window must be {list(EXPERT_REVIEW_TURING_WINDOW)}")
        if protocol.get("likert_minimum") != EXPERT_REVIEW_LIKERT_BOUND:
            failures.append(f"L3 report Likert bound must be {EXPERT_REVIEW_LIKERT_BOUND}")
        if protocol.get("confidence_level") != 0.95:
            failures.append("L3 report must record 95% confidence")
        bootstrap = protocol.get("bootstrap")
        if not isinstance(bootstrap, dict) or not bootstrap.get("method") or not isinstance(bootstrap.get("resamples"), int):
            failures.append("L3 report bootstrap must record a method and integer resamples")
        elif not isinstance(bootstrap.get("seed"), int):
            failures.append("L3 report bootstrap must record an integer seed")
        if not isinstance(protocol.get("per_cell"), int) or protocol.get("per_cell", 0) < 1:
            failures.append("L3 report per_cell must be a positive integer")
        if not isinstance(protocol.get("total_entries"), int):
            failures.append("L3 report total_entries must be an integer")
        return protocol

    def _coverage(self, challenges, report, protocol, failures):
        coverage = report.get("coverage")
        if not isinstance(coverage, list):
            failures.append("L3 report coverage must be a list")
            return
        per_cell = (protocol or {}).get("per_cell")
        expected = {(challenge, modality) for challenge in challenges for modality in EXPERT_REVIEW_MODALITIES}
        actual = {(row.get("challenge"), row.get("target_modality")) for row in coverage if isinstance(row, dict)}
        if not expected or len(coverage) != len(expected) or actual != expected:
            failures.append("L3 report must cover each pinned challenge and all four target modalities exactly once")
            return
        if per_cell is None:
            return
        total = 0
        for row in coverage:
            if isinstance(row, dict):
                real_count, synth_count = row.get("real"), row.get("synth")
                if real_count != per_cell or synth_count != per_cell:
                    failures.append(f"L3 coverage {row.get('challenge')}/{row.get('target_modality')} must be {per_cell} real + {per_cell} synth")
                if isinstance(real_count, int) and isinstance(synth_count, int):
                    total += real_count + synth_count
        if protocol.get("total_entries") != total:
            failures.append("L3 report total_entries must equal the sum of the per-cell coverage")

    def _visual_turing(self, report, protocol, failures):
        vt = report.get("visual_turing")
        if not isinstance(vt, dict):
            failures.append("L3 report visual_turing must be an object")
            return
        expected_reviewers = (protocol or {}).get("reviewers")
        per_reviewer = vt.get("per_reviewer")
        if not isinstance(per_reviewer, list) or len(per_reviewer) != expected_reviewers:
            failures.append("L3 report visual_turing per_reviewer must match the recorded reviewer count")
            return
        for result in per_reviewer:
            if isinstance(result, dict):
                self._vt_result(result, failures)
        pooled = vt.get("pooled")
        if isinstance(pooled, dict):
            self._vt_result(pooled, failures, pooled=True)
        else:
            failures.append("L3 report must record the pooled visual-Turing result")
        if isinstance(pooled, dict):
            recorded = vt.get("verdict")
            expected = "pass" if all(isinstance(item, dict) and item.get("verdict") == "pass" for item in per_reviewer + [pooled]) else "fail"
            if recorded != expected:
                failures.append("L3 report visual_turing verdict disagrees with its per-reviewer/pooled CI window gates")

    def _vt_result(self, result, failures, pooled=False):
        verdict = result.get("verdict")
        if verdict not in EXPERT_REVIEW_VERDICTS:
            failures.append("L3 visual-Turing verdict must be pass or fail")
            return
        if not self._number(result.get("balanced_accuracy")):
            failures.append("L3 visual-Turing balanced accuracy must be finite")
            return
        ci = result.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2 or not all(self._number(value) for value in ci) or ci[0] > ci[1]:
            failures.append("L3 visual-Turing CI must be an ordered finite 95% CI")
            return
        if not (ci[0] <= result["balanced_accuracy"] <= ci[1]):
            failures.append("L3 visual-Turing CI must contain its balanced-accuracy point estimate")
        expected = "pass" if ci[0] >= EXPERT_REVIEW_TURING_WINDOW[0] and ci[1] <= EXPERT_REVIEW_TURING_WINDOW[1] else "fail"
        if verdict != expected:
            failures.append("L3 visual-Turing verdict disagrees with its CI window gate")
        if pooled:
            return
        confusion = result.get("confusion")
        if not isinstance(confusion, dict):
            failures.append("L3 per-reviewer visual-Turing must record a confusion matrix")
            return
        try:
            real_total = confusion.get("real_said_real", 0) + confusion.get("real_said_synth", 0)
            synth_total = confusion.get("synth_said_real", 0) + confusion.get("synth_said_synth", 0)
            if real_total <= 0 or synth_total <= 0:
                failures.append("L3 per-reviewer visual-Turing confusion must have both real and synth entries")
                return
            if result.get("n") != real_total + synth_total:
                failures.append("L3 per-reviewer visual-Turing n must equal the confusion total")
            rederived = 0.5 * (confusion["real_said_real"] / real_total + confusion["synth_said_synth"] / synth_total)
        except (KeyError, TypeError):
            failures.append("L3 per-reviewer visual-Turing confusion must carry integer counts")
            return
        if not math.isclose(rederived, result["balanced_accuracy"], rel_tol=1e-9, abs_tol=1e-12):
            failures.append("L3 per-reviewer visual-Turing balanced accuracy disagrees with its confusion matrix")

    def _likert(self, report, protocol, failures):
        likert = report.get("likert")
        if not isinstance(likert, list):
            failures.append("L3 report likert must be a list")
            return
        dimensions = {item.get("dimension") for item in likert if isinstance(item, dict)}
        if len(likert) != len(EXPERT_REVIEW_DIMENSIONS) or dimensions != set(EXPERT_REVIEW_DIMENSIONS):
            failures.append(f"L3 report likert must cover each of {EXPERT_REVIEW_DIMENSIONS} exactly once")
            return
        for item in likert:
            if isinstance(item, dict):
                self._likert_item(item, failures)

    def _likert_item(self, item, failures):
        dimension = item.get("dimension")
        phase = item.get("phase")
        self._likert_bundle(phase, f"L3 Likert {dimension} phase", failures)
        per_modality = item.get("per_modality")
        if not isinstance(per_modality, dict) or set(per_modality) != set(EXPERT_REVIEW_MODALITIES):
            failures.append(f"L3 Likert {dimension} per_modality must cover all four target modalities")
            return
        for modality in EXPERT_REVIEW_MODALITIES:
            self._likert_bundle(per_modality.get(modality), f"L3 Likert {dimension} {modality}", failures)
        if not (item.get("fleiss_kappa") is None or self._number(item["fleiss_kappa"])):
            failures.append(f"L3 Likert {dimension} Fleiss' kappa must be finite or null")

    def _likert_bundle(self, bundle, label, failures):
        if not isinstance(bundle, dict):
            failures.append(f"{label} must be an object")
            return
        point = bundle.get("point")
        lower = bundle.get("ci95_lower")
        if not self._number(point) or not self._number(lower) or lower > point:
            failures.append(f"{label} must record a finite point and a one-sided lower CI not above the mean")
            return
        if not isinstance(bundle.get("n"), int) or bundle["n"] < 1:
            failures.append(f"{label} must record a positive integer n")
            return
        if not isinstance(bundle.get("na"), int) or bundle["na"] < 0:
            failures.append(f"{label} must record a non-negative integer NA count")
            return
        verdict = bundle.get("verdict")
        if verdict not in EXPERT_REVIEW_VERDICTS:
            failures.append(f"{label} verdict must be pass or fail")
            return
        expected = "pass" if lower >= EXPERT_REVIEW_LIKERT_BOUND else "fail"
        if verdict != expected:
            failures.append(f"{label} verdict disagrees with its {EXPERT_REVIEW_LIKERT_BOUND} lower-bound gate")

    def _verdict(self, report, failures):
        verdict = report.get("verdict")
        if not isinstance(verdict, dict):
            failures.append("L3 report verdict must be an object")
            return
        for key in ("visual_turing", "likert", "overall"):
            if verdict.get(key) not in EXPERT_REVIEW_VERDICTS:
                failures.append(f"L3 report verdict {key} must be pass or fail")
        likert = report.get("likert") or []
        likert_expected = (
            "pass"
            if all(
                isinstance(item, dict)
                and (item.get("phase") or {}).get("verdict") == "pass"
                and all(isinstance(modality, dict) and modality.get("verdict") == "pass" for modality in (item.get("per_modality") or {}).values())
                for item in likert
            )
            else "fail"
        )
        overall_expected = "pass" if (report.get("visual_turing") or {}).get("verdict") == "pass" and likert_expected == "pass" else "fail"
        if verdict.get("likert") != likert_expected:
            failures.append("L3 report verdict likert disagrees with its dimension lower-bound gates")
        if verdict.get("overall") != overall_expected:
            failures.append("L3 report verdict overall must be the non-compensatory AND of visual-Turing and Likert")

    @staticmethod
    def verdict_of(report):
        """The expert-review layer verdict read from a report (registry ``verdict_reader``)."""
        return (report.get("verdict") or {}).get("overall")

    @staticmethod
    def blocked_reasons(report):
        """Traceable expert-review blockers: visual-Turing window and Likert lower-bound gates."""
        reasons = []
        if (report.get("visual_turing") or {}).get("verdict") != "pass":
            reasons.append("L3 visual-Turing: CI window gate not met")
        for item in report.get("likert") or []:
            failing = [m for m, b in (item.get("per_modality") or {}).items() if b.get("verdict") != "pass"]
            if (item.get("phase") or {}).get("verdict") != "pass" or failing:
                detail = f"; per-modality fail: {', '.join(sorted(failing))}" if failing else ""
                reasons.append(f"L3 Likert {item.get('dimension')}: lower-bound gate not met{detail}")
        return reasons

    def _number(self, value):
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


class DistributionReportValidator:
    """Validates the versioned distribution evidence schema and its frozen-candidate binding.

    Formal distribution evidence (``l2-final-acceptance-report/1``, issue #55) attaches
    only as a complete five-challenge report: ``challenges_missing`` empty, no
    provisional challenge, ``complete_coverage`` true (spec Further Notes --
    a run over a subset of the five challenges is provisional smoke, never
    full-spec acceptance evidence). Verdict consistency mirrors the issue #55
    judgement chain: any failure-audit count > 0 forces that challenge
    ``undecided``; otherwise all TOST (and, for P2, round-trip) checks passing
    forces ``pass``; the overall verdict is undecided > fail > pass.
    """

    REPORT_SCHEMA = "l2-final-acceptance-report/1"  # mirrors the distribution final_acceptance REPORT_SCHEMA

    def __init__(self, schema=None):
        self._schema = schema or self.REPORT_SCHEMA

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L2 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L2 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L2 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        self._coverage(report, failures)
        self._per_challenge(record, report, failures)
        self._overall(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        if report.get("schema") != self._schema:
            failures.append(f"L2 report schema != {self._schema}")
        mismatches = FrozenRunBinding.mismatches_for(record, binding)
        if mismatches is None:
            failures.append("L2 report binding must be an object (evaluate with --run to bind the frozen candidate)")
            return
        for key in mismatches:
            failures.append(f"L2 report binding {key} does not match frozen run")

    def _coverage(self, report, failures):
        if report.get("challenges_missing") != []:
            failures.append("formal L2 evidence must cover all five challenges (challenges_missing must be empty)")
        if report.get("provisional_challenges") != []:
            failures.append("formal L2 evidence must meet every frozen holdout quota (no provisional challenge)")
        if report.get("complete_coverage") is not True:
            failures.append("formal L2 evidence must record complete_coverage true")

    def _per_challenge(self, record, report, failures):
        per_challenge = report.get("per_challenge")
        if not isinstance(per_challenge, dict):
            failures.append("L2 report per_challenge must be an object")
            return
        if set(per_challenge) != set(DISTRIBUTION_CHALLENGES):
            failures.append(f"L2 report per_challenge must cover exactly {DISTRIBUTION_CHALLENGES}")
            return
        for challenge, verdict in per_challenge.items():
            if not isinstance(verdict, dict):
                failures.append(f"L2 per_challenge {challenge} must be an object")
            else:
                self._challenge_verdict(challenge, verdict, record.get("phase"), failures)

    def _challenge_verdict(self, challenge, verdict, phase, failures):
        recorded = verdict.get("verdict")
        if recorded not in DISTRIBUTION_VERDICTS:
            failures.append(f"L2 {challenge} verdict must be pass, fail, or undecided")
            return
        audit = verdict.get("failure_audit")
        n_failed = audit.get("n_failed") if isinstance(audit, dict) else None
        if not isinstance(n_failed, int) or n_failed < 0:
            failures.append(f"L2 {challenge} failure_audit must record a non-negative integer n_failed")
            return
        checks = [item.get("passed") for item in verdict.get("tost") or []]
        if verdict.get("round_trip") is not None:
            if phase != "P2":
                failures.append(f"L2 {challenge} round_trip evidence is P2-only; {phase} must not carry it")
                return
            checks += [item.get("passed") for item in verdict["round_trip"] or []]
        elif phase == "P2":
            failures.append(f"L2 {challenge} P2 evidence must carry the condition round-trip results")
            return
        if not checks:
            failures.append(f"L2 {challenge} carries no TOST checks")
            return
        if n_failed:
            expected = "undecided"
        elif all(checks):
            expected = "pass"
        else:
            expected = "fail"
        if recorded != expected:
            failures.append(f"L2 {challenge} verdict {recorded!r} disagrees with its failure gate/TOST/round-trip evidence")

    def _overall(self, report, failures):
        overall = report.get("overall_verdict")
        if overall not in DISTRIBUTION_VERDICTS:
            failures.append("L2 report overall_verdict must be pass, fail, or undecided")
            return
        per_challenge = report.get("per_challenge")
        verdicts = (
            [verdict.get("verdict") for verdict in per_challenge.values() if isinstance(verdict, dict)] if isinstance(per_challenge, dict) else []
        )
        if len(verdicts) != len(DISTRIBUTION_CHALLENGES) or any(v not in DISTRIBUTION_VERDICTS for v in verdicts):
            return  # already reported by _per_challenge
        expected = "undecided" if "undecided" in verdicts else "pass" if all(v == "pass" for v in verdicts) else "fail"
        if overall != expected:
            failures.append("L2 report overall_verdict disagrees with its per-challenge verdicts")

    @staticmethod
    def verdict_of(report):
        """The distribution layer verdict read from a report (registry ``verdict_reader``)."""
        return report.get("overall_verdict")

    @staticmethod
    def blocked_reasons(report):
        """Traceable distribution blockers: per-challenge verdicts with their reason or TOST/round-trip audit."""
        reasons = []
        for challenge, verdict in (report.get("per_challenge") or {}).items():
            if verdict.get("verdict") != "pass":
                reason = verdict.get("reason") or DistributionReportValidator._l2_challenge_reason(verdict)
                reasons.append(f"L2 {challenge}: {verdict.get('verdict')} ({reason})")
        return reasons

    @staticmethod
    def _l2_challenge_reason(verdict):
        if verdict.get("verdict") == "undecided":
            return "instrument failure gate; fix direction is the instrument or a re-run"
        tost_failed = sum(0 if verdict.get("tost") is None else (not item.get("passed")) for item in (verdict.get("tost") or []))
        rt_failed = sum(0 if verdict.get("round_trip") is None else (not item.get("passed")) for item in (verdict.get("round_trip") or []))
        return f"{tost_failed} TOST and {rt_failed} round-trip checks failed"
