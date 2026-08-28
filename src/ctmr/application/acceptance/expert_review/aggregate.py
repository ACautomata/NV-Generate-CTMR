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

"""Blinded-judgment aggregation into the candidate-bound L3 report.

Migrated verbatim from ``scripts/brats_l3_blind_eval.py`` (#141). Combines
>=2 independent reviewers' blinded judgments (``brats-l3-responses/1``) with
the blind map into a ``brats-l3-report/1`` conclusion: per-reviewer
visual-Turing balanced accuracy with a class-stratified 95% CI (must lie
entirely inside the window), the pooled CI, the four Likert dimensions with
phase-total and per-target-modality one-sided lower 95% bounds (each >= the
bound), and Fleiss' kappa (report-only). The verdict is a non-compensatory
AND. Stdlib only.

Reached as ``ctmr accept expert-review aggregate ...``.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.expert_review.catalog import (
    BLIND_MAP_SCHEMA,
    DIMENSIONS,
    LIKERT_BOUND,
    LIKERT_MAX,
    LIKERT_MIN,
    MODALITIES,
    NA,
    REPORT_SCHEMA,
    RESPONSES_SCHEMA,
    TURING_LABELS,
    TURING_WINDOW,
    Catalog,
    L3Error,
)


class VisualTuringAggregator:
    """Balanced accuracy and a class-stratified percentile bootstrap 95% CI."""

    @staticmethod
    def balanced_accuracy(real_judgments, synth_judgments):
        """real/synth_judgments are the reviewer's predictions for entries of that true source."""
        tpr = sum(1 for prediction in real_judgments if prediction == "real") / len(real_judgments)
        tnr = sum(1 for prediction in synth_judgments if prediction == "synth") / len(synth_judgments)
        return 0.5 * (tpr + tnr)

    def bootstrap_ci(self, real_judgments, synth_judgments, resamples, seed):
        """Class-stratified resample bootstrap: keep each class size fixed per resample."""
        rng = random.Random(seed)
        point = self.balanced_accuracy(real_judgments, synth_judgments)
        n_real, n_synth = len(real_judgments), len(synth_judgments)
        estimates = []
        for _ in range(resamples):
            real_sample = [real_judgments[rng.randrange(n_real)] for _ in range(n_real)]
            synth_sample = [synth_judgments[rng.randrange(n_synth)] for _ in range(n_synth)]
            estimates.append(self.balanced_accuracy(real_sample, synth_sample))
        estimates.sort()
        lo_index = int(0.025 * resamples)
        hi_index = max(0, int(0.975 * resamples) - 1)
        return {"point": point, "ci95": [estimates[lo_index], estimates[hi_index]]}


class LikertAggregator:
    """One-sided bootstrap lower bound of a dimension mean and Fleiss' kappa."""

    @staticmethod
    def _mean(scores):
        return statistics.fmean(scores)

    def one_sided_lower_ci(self, scores, resamples, seed):
        rng = random.Random(seed)
        point = self._mean(scores)
        n = len(scores)
        means = []
        for _ in range(resamples):
            resampled = [scores[rng.randrange(n)] for _ in range(n)]
            means.append(self._mean(resampled))
        means.sort()
        return {"point": point, "ci95_lower": means[int(0.05 * resamples)]}

    def fleiss_kappa(self, category_lists):
        """Fleiss' kappa across reviewers for whole-subject category assignments.

        ``category_lists`` is the per-subject list of reviewer category labels
        (each inner list length == reviewer count, no NA/missing entries). Returns
        None on a degenerate chance-agreement=1 configuration."""
        subjects = len(category_lists)
        if subjects == 0:
            return None
        rater_count = len(category_lists[0])
        if rater_count < 2:
            return None
        categories = sorted({category for subject in category_lists for category in subject})
        assignments = subjects * rater_count
        if assignments == 0:
            return None
        p_bar = sum(self._subject_agreement(subject, categories, rater_count) for subject in category_lists) / subjects
        chance = 0.0
        for category in categories:
            probability = sum(subject.count(category) for subject in category_lists) / assignments
            chance += probability * probability
        if chance >= 1.0:
            return None
        return (p_bar - chance) / (1 - chance)

    @staticmethod
    def _subject_agreement(subject, categories, rater_count):
        if rater_count < 2:
            return 0.0
        denominator = rater_count * (rater_count - 1)
        return sum(count * (count - 1) for count in (subject.count(category) for category in categories)) / denominator


class L3ReportProducer:
    """Aggregates blinded reviewer judgments into a candidate-bound L3 report."""

    def __init__(self, resamples, seed):
        self._resamples = resamples
        self._seed = seed
        self._vt = VisualTuringAggregator()
        self._likert_agg = LikertAggregator()

    @staticmethod
    def _sha256_text(payload):
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run_binding(self, run_record):
        """The frozen-candidate five-key binding with the freeze gate built in (ADR-0012 决定 4)."""
        try:
            return FrozenRunBinding.from_record(run_record).as_dict()
        except FrozenRunBindingError as error:
            raise L3Error(str(error)) from error

    def _validate_responses(self, responses, expected_ids):
        if len(responses) < 2:
            raise L3Error("L3 final acceptance requires at least two independent reviewers")
        for response in responses:
            if response.get("schema") != RESPONSES_SCHEMA:
                raise L3Error(f"reviewer responses schema must be {RESPONSES_SCHEMA!r}")
            rated_ids = {entry["entry_id"] for entry in response["entries"]}
            if rated_ids != expected_ids:
                raise L3Error(f"reviewer {response['reviewer']} must rate every blinded entry exactly once")
            for entry in response["entries"]:
                if entry["turing"] not in TURING_LABELS:
                    raise L3Error(f"reviewer {response['reviewer']} entry {entry['entry_id']} turing must be real/synth")

    def _visual_turing(self, responses, true_source):
        per_reviewer = []
        for response in responses:
            real_judgments, synth_judgments = [], []
            confusion = {"real_said_real": 0, "real_said_synth": 0, "synth_said_real": 0, "synth_said_synth": 0}
            for record in response["entries"]:
                prediction = record["turing"]
                if true_source[record["entry_id"]] == "real":
                    real_judgments.append(prediction)
                    confusion["real_said_real" if prediction == "real" else "real_said_synth"] += 1
                else:
                    synth_judgments.append(prediction)
                    confusion["synth_said_real" if prediction == "real" else "synth_said_synth"] += 1
            result = self._vt.bootstrap_ci(real_judgments, synth_judgments, self._resamples, self._seed)
            result["balanced_accuracy"] = result.pop("point")
            result["reviewer"] = response["reviewer"]
            result["n"] = len(real_judgments) + len(synth_judgments)
            result["confusion"] = confusion
            result["verdict"] = self._window_verdict(result["ci95"])
            per_reviewer.append(result)
        # Pool: each blinded entry contributes one judgment per reviewer.
        pooled_real, pooled_synth = [], []
        for response in responses:
            for record in response["entries"]:
                (pooled_real if true_source[record["entry_id"]] == "real" else pooled_synth).append(record["turing"])
        pooled = self._vt.bootstrap_ci(pooled_real, pooled_synth, self._resamples, self._seed)
        pooled["balanced_accuracy"] = pooled.pop("point")
        pooled["reviewers"] = len(responses)
        pooled["n"] = len(pooled_real) + len(pooled_synth)
        pooled["verdict"] = self._window_verdict(pooled["ci95"])
        verdict = "pass" if all(item["verdict"] == "pass" for item in per_reviewer + [pooled]) else "fail"
        return {"per_reviewer": per_reviewer, "pooled": pooled, "verdict": verdict, "fleiss_kappa": self._turing_fleiss(responses)}

    @staticmethod
    def _window_verdict(ci95):
        lo, hi = ci95
        return "pass" if lo >= TURING_WINDOW[0] and hi <= TURING_WINDOW[1] else "fail"

    def _turing_fleiss(self, responses):
        by_entry = {}
        for response in responses:
            for record in response["entries"]:
                by_entry.setdefault(record["entry_id"], []).append(record["turing"])
        return self._likert_agg.fleiss_kappa([labels for _, labels in sorted(by_entry.items())])

    def _likert_dimension_scores(self, responses, blind_map_index, dimension):
        modality_of = {entry_id: entry["target_modality"] for entry_id, entry in blind_map_index.items()}
        scores_by_modality = {modality: [] for modality in MODALITIES}
        na_by_modality = {modality: 0 for modality in MODALITIES}
        phase_scores = []
        for response in responses:
            for record in response["entries"]:
                value = record.get(dimension, NA)
                modality = modality_of[record["entry_id"]]
                if value == NA:
                    na_by_modality[modality] += 1
                    continue
                scores_by_modality[modality].append(value)
                phase_scores.append(value)
        per_modality = {}
        for modality in MODALITIES:
            scores = scores_by_modality[modality]
            if not scores:
                raise L3Error(f"Likert {dimension} modality {modality} has no non-NA ratings; cannot form a bound")
            bundle = self._likert_agg.one_sided_lower_ci(scores, self._resamples, self._seed)
            bundle["n"] = len(scores)
            bundle["na"] = na_by_modality[modality]
            bundle["verdict"] = "pass" if bundle["ci95_lower"] >= LIKERT_BOUND else "fail"
            per_modality[modality] = bundle
        if not phase_scores:
            raise L3Error(f"Likert {dimension} phase has no non-NA ratings; cannot form a bound")
        phase = self._likert_agg.one_sided_lower_ci(phase_scores, self._resamples, self._seed)
        phase["n"] = len(phase_scores)
        phase["na"] = sum(na_by_modality.values())
        phase["verdict"] = "pass" if phase["ci95_lower"] >= LIKERT_BOUND else "fail"
        return {"phase": phase, "per_modality": per_modality}

    def _likert_fleiss(self, responses, dimension):
        per_entry = {}
        for response in responses:
            for record in response["entries"]:
                value = record.get(dimension, NA)
                per_entry.setdefault(record["entry_id"], []).append(value)
        # Fleiss' kappa needs every reviewer's category for a subject: drop entries with any NA.
        complete = [labels for _, labels in sorted(per_entry.items()) if all(label != NA for label in labels)]
        return self._likert_agg.fleiss_kappa(complete)

    def _likert(self, responses, blind_map_index):
        dimensions = []
        for dimension in DIMENSIONS:
            bundle = self._likert_dimension_scores(responses, blind_map_index, dimension)
            bundle["dimension"] = dimension
            bundle["fleiss_kappa"] = self._likert_fleiss(responses, dimension)
            dimensions.append(bundle)
        return dimensions

    def _coverage(self, blind_map_entries):
        cells = {}
        for entry in blind_map_entries:
            cells[(entry["challenge"], entry["target_modality"], entry["source"])] = (
                cells.get((entry["challenge"], entry["target_modality"], entry["source"]), 0) + 1
            )
        rows = []
        for (challenge, modality, source), count in sorted(cells.items()):
            row = next((candidate for candidate in rows if candidate["challenge"] == challenge and candidate["target_modality"] == modality), None)
            if row is None:
                row = {"challenge": challenge, "target_modality": modality, "real": 0, "synth": 0}
                rows.append(row)
            row[source] = count
        return rows

    def _produce(self, run_record, responses, blind_map_doc, catalog_payload):
        blind_map_entries = blind_map_doc["entries"]
        per_cell = blind_map_doc["sampling"]["per_cell"]["real"]
        coverage = self._coverage(blind_map_entries)
        for row in coverage:
            if row["real"] != per_cell or row["synth"] != per_cell:
                raise L3Error(
                    f"cell {row['challenge']}/{row['target_modality']} holds {row['real']} real + {row['synth']} synth; "
                    f"every cell must hold exactly {per_cell} of each source"
                )
        total_entries = sum(row["real"] + row["synth"] for row in coverage)
        expected_ids = {entry["entry_id"] for entry in blind_map_entries}
        if len(expected_ids) != total_entries:
            raise L3Error(f"blind map entry ids do not match the coverage total ({total_entries})")
        self._validate_responses(responses, expected_ids)
        blind_map_index = {entry["entry_id"]: entry for entry in blind_map_entries}
        true_source = {entry_id: entry["source"] for entry_id, entry in blind_map_index.items()}
        visual_turing = self._visual_turing(responses, true_source)
        likert = self._likert(responses, blind_map_index)
        likert_verdict = (
            "pass"
            if all(
                dimension["phase"]["verdict"] == "pass" and all(modality["verdict"] == "pass" for modality in dimension["per_modality"].values())
                for dimension in likert
            )
            else "fail"
        )
        verdict = {
            "visual_turing": visual_turing["verdict"],
            "likert": likert_verdict,
            "overall": "pass" if visual_turing["verdict"] == "pass" and likert_verdict == "pass" else "fail",
        }
        protocol = {
            "reviewers": len(responses),
            "dimensions": list(DIMENSIONS),
            "target_modalities": list(MODALITIES),
            "visual_turing_ci_window": list(TURING_WINDOW),
            "likert_minimum": LIKERT_BOUND,
            "likert_scale": {"min": LIKERT_MIN, "max": LIKERT_MAX},
            "confidence_level": 0.95,
            "bootstrap": {"method": "entry_level_stratified_percentile_mt19937", "resamples": self._resamples, "seed": self._seed},
            "per_cell": per_cell,
            "total_entries": total_entries,
        }
        return {
            "schema": REPORT_SCHEMA,
            "binding": self._run_binding(run_record),
            "protocol": protocol,
            "coverage": coverage,
            "provenance": {
                "catalog_sha256": self._sha256_text(catalog_payload),
                "blind_map_sha256": self._sha256_text({"schema": BLIND_MAP_SCHEMA, "entries": blind_map_entries}),
            },
            "visual_turing": visual_turing,
            "likert": likert,
            "verdict": verdict,
        }


def main(argv=None):
    """Run the aggregate verb (``ctmr accept expert-review aggregate``)."""
    parser = argparse.ArgumentParser(
        prog="ctmr accept expert-review aggregate",
        description="Aggregate blinded reviewer judgments into a candidate-bound expert-review report.",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--responses", dest="responses", action="append", required=True, help="one blinded reviewer response per flag (brats-l3-responses/1)"
    )
    parser.add_argument("--blind-map", required=True, help="controlled blind map (brats-l3-blind-map/1)")
    parser.add_argument("--catalog", required=True, help="catalog manifest (brats-l3-catalog/1)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)
    try:
        run_record = json.loads(Path(args.run).read_text())
        blind_map_doc = json.loads(Path(args.blind_map).read_text())
        responses = [json.loads(Path(path).read_text()) for path in args.responses]
        report = L3ReportProducer(args.resamples, args.seed)._produce(run_record, responses, blind_map_doc, Catalog.from_path(args.catalog).payload)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"L3 report written -> {args.output}")
        return 0
    except L3Error as error:
        print(f"L3 ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
