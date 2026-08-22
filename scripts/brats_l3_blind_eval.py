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

"""L3 blind evaluation package and visual-Turing / Likert aggregation (issue #56, spec #51).

A frozen P1/P2/P3 candidate is turned into a neuroradiology-review blinding
package, then a set of independent blinded reviewer judgments is aggregated to
a ``brats-l3-report/1`` final-acceptance conclusion.

Two subcommands (each standalone, stdlib only):

``build-package``
    Sample a reproducible blinding package. For the frozen candidate's target
    modalities x sub-challenges we draw ``per-cell`` real + ``per-cell``
    synthetic images for a fixed total of 200 entries, blind them into opaque
    entry ids (L3-0001..L3-0200) in a seeded presentation order, and emit the
    reviewer-facing ``brats-l3-package/1`` plus the controlled ``brats-l3-
    blind-map/1`` that records the unblinded source, case and image path. The
    seed, per-cell quota and the package/blind-map sha256 make the draw and the
    blinding auditable. A cell with fewer than ``per-cell`` available images of
    either source refuses to build (no silent partial sub-sampling).

``aggregate``
    Combine >=2 independent reviewers' blinded judgments (``brats-l3-responses/1``)
    with the blind map into a ``brats-l3-report/1`` final conclusion: per-reviewer
    visual-Turing balanced accuracy with a 95% CI (must lie entirely inside
    [0.40, 0.60]); the pooled visual-Turing balanced accuracy and CI; the four
    Likert dimensions with their phase-total and per-target-modality one-sided
    lower 95% bounds (each must be >= 4.0/5.0); and Fleiss' kappa (report-only).
    The verdict is a non-compensatory AND: the final acceptance passes only when
    every reviewer's visual-Turing CI is inside the window and every Likert lower
    bound is at least 4.0.

Usage::

    python -m scripts.brats_l3_blind_eval build-package \
        --run runs/p1-xxx/run.json --catalog catalog.json --output /ctrl/l3/p1 \
        --seed 20260821 --per-cell 5
    python -m scripts.brats_l3_blind_eval aggregate \
        --run runs/p1-xxx/run.json --responses R1.json --responses R2.json \
        --blind-map /ctrl/l3/p1/blind_map.json --catalog catalog.json \
        --output /ctrl/l3/p1/l3_report.json --resamples 1000 --seed 20260821
    python -m scripts.brats_l3_blind_eval selftest --workdir TMP
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

CATALOG_SCHEMA = "brats-l3-catalog/1"
PACKAGE_SCHEMA = "brats-l3-package/1"
BLIND_MAP_SCHEMA = "brats-l3-blind-map/1"
RESPONSES_SCHEMA = "brats-l3-responses/1"
REPORT_SCHEMA = "brats-l3-report/1"

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
DIMENSIONS = (
    "overall_realism",
    "anatomical_plausibility",
    "tumor_authenticity",
    "artifact_slice_consistency",
)
TURING_LABELS = ("real", "synth")
LIKERT_MIN, LIKERT_MAX = 1, 5
NA = "NA"
TURING_WINDOW = (0.40, 0.60)
LIKERT_BOUND = 4.0
STATUS_FROZEN = "frozen"
TOTAL_ENTRIES = 200


class L3Error(Exception):
    """Raised when a blinding package or judgment does not satisfy the L3 protocol."""


@dataclass
class CatalogEntry:
    """One available candidate image of a given source on a given target modality."""

    challenge: str
    case: str
    target_modality: str
    source: str
    path: str
    sha256: str
    src_modality: str | None = None


class Catalog:
    """The controlled catalog of candidate images available for L3 sampling."""

    def __init__(self, payload):
        self._payload = payload
        self._entries = [CatalogEntry(**record) for record in payload["records"]]

    @classmethod
    def from_path(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != CATALOG_SCHEMA:
            raise L3Error(f"catalog schema must be {CATALOG_SCHEMA!r}, got {payload.get('schema')!r}")
        return cls(payload)

    @property
    def payload(self):
        return self._payload

    def entries(self):
        return self._entries

    def cell(self, challenge, target_modality):
        real, synth = [], []
        for entry in self._entries:
            if (entry.challenge, entry.target_modality) == (challenge, target_modality):
                (real if entry.source == "real" else synth).append(entry)
        return real, synth

    def challenges(self):
        return tuple(sorted({entry.challenge for entry in self._entries}))


class BlindSampler:
    """Deterministic, auditable sampling and blinding of the 200-entry package."""

    def __init__(self, seed, per_cell):
        self._seed = seed
        self._per_cell = per_cell

    def _draw(self, candidates, count, rng):
        ordered = sorted(candidates, key=lambda e: (e.challenge, e.case, e.target_modality, e.source, e.path))
        if len(ordered) < count:
            raise L3Error(f"cell has only {len(ordered)} available images of one source; need {count}")
        return rng.sample(ordered, count)

    def sample(self, catalog):
        """Draw per-cell real+synth entries, then blind them into a presentation order.

        Returns ``(package_entries, blind_map_entries)``, each a list of dicts in the
        blinded presentation order."""
        rng = random.Random(self._seed)
        drawn = []
        for challenge in catalog.challenges():
            for modality in MODALITIES:
                real, synth = catalog.cell(challenge, modality)
                drawn.extend(self._draw(real, self._per_cell, rng))
                drawn.extend(self._draw(synth, self._per_cell, rng))
        rng.shuffle(drawn)
        package, blind_map = [], []
        for index, entry in enumerate(drawn, start=1):
            entry_id = f"L3-{index:04d}"
            package.append({"entry_id": entry_id, "challenge": entry.challenge, "target_modality": entry.target_modality, "image_path": entry.path})
            blind_entry = {
                "entry_id": entry_id,
                "challenge": entry.challenge,
                "case": entry.case,
                "target_modality": entry.target_modality,
                "source": entry.source,
                "image_path": entry.path,
            }
            if entry.src_modality is not None:
                blind_entry["src_modality"] = entry.src_modality
            blind_map.append(blind_entry)
        return package, blind_map


class BlindPackageBuilder:
    """Assembles and writes the blinded package and the controlled blind map."""

    def __init__(self, seed, per_cell):
        self._seed = seed
        self._per_cell = per_cell

    def build(self, run_record, catalog):
        if run_record.get("status") != STATUS_FROZEN:
            raise L3Error(f"run {run_record.get('run_id')} is {run_record.get('status')}; L3 blinds only frozen candidates")
        package, blind_map = BlindSampler(self._seed, self._per_cell).sample(catalog)
        sampling = {
            "seed": self._seed,
            "per_cell": {"real": self._per_cell, "synth": self._per_cell},
            "total_entries": len(package),
        }
        package_doc = {
            "schema": PACKAGE_SCHEMA,
            "phase": run_record["phase"],
            "run_id": run_record["run_id"],
            "sampling": sampling,
            "entries": package,
        }
        blind_map_doc = {
            "schema": BLIND_MAP_SCHEMA,
            "phase": run_record["phase"],
            "run_id": run_record["run_id"],
            "sampling": sampling,
            "entries": blind_map,
        }
        return package_doc, blind_map_doc


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

    @staticmethod
    def _binding(run_record):
        return {
            "run_id": run_record.get("run_id"),
            "phase": run_record.get("phase"),
            "manifest_sha256": run_record.get("manifest", {}).get("sha256"),
            "candidate_checkpoint_sha256": run_record.get("selection", {}).get("checkpoint", {}).get("sha256"),
            "samples_sha256": run_record.get("samples", {}).get("sha256"),
        }

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
            "binding": self._binding(run_record),
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


class L3SelfTest:
    """Fixture-driven synthetic checks of sampling/blinding and aggregation (stdlib only)."""

    FIXTURE_CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def _make_catalog(self):
        records = []
        for challenge in self.FIXTURE_CHALLENGES:
            for modality in MODALITIES:
                for index in range(6):
                    real_id = f"REAL-{challenge}-{modality}-{index}"
                    synth_id = f"SYNTH-{challenge}-{modality}-{index}"
                    records.append(
                        {
                            "challenge": challenge,
                            "case": real_id,
                            "target_modality": modality,
                            "source": "real",
                            "path": f"/ctrl/real/{real_id}.nii.gz",
                            "sha256": "d" * 64,
                        }
                    )
                    records.append(
                        {
                            "challenge": challenge,
                            "case": synth_id,
                            "target_modality": modality,
                            "source": "synth",
                            "path": f"/ctrl/synth/{synth_id}.nii.gz",
                            "sha256": "f" * 64,
                        }
                    )
        return {"schema": CATALOG_SCHEMA, "records": records}

    @staticmethod
    def _run_record():
        return {
            "run_id": "p1-fixture",
            "phase": "P1",
            "status": "frozen",
            "manifest": {"sha256": "m" * 64},
            "selection": {"checkpoint": {"sha256": "c" * 64}},
            "samples": {"sha256": "s" * 64},
        }

    @staticmethod
    def _deterministic_bit(entry_id):
        return int(hashlib.sha256(entry_id.encode()).hexdigest(), 16) % 2

    def _responses(self, blind_map_entries):
        """R1 is near chance (pass); R2 is a strong distinguisher (fails the gate)."""
        responses = []
        for reviewer in ("R1", "R2"):
            entries = []
            for entry in blind_map_entries:
                real = entry["source"] == "real"
                if reviewer == "R1":
                    prediction = "real" if self._deterministic_bit(entry["entry_id"]) == 0 else "synth"
                else:
                    # R2 correctly tells real vs synth (balanced accuracy near 1.0 -> fail).
                    prediction = "real" if real else "synth"
                entries.append(
                    {
                        "entry_id": entry["entry_id"],
                        "turing": prediction,
                        "overall_realism": 4,
                        "anatomical_plausibility": 4,
                        "tumor_authenticity": 4,
                        "artifact_slice_consistency": 4,
                        "notes": "",
                    }
                )
            responses.append({"schema": RESPONSES_SCHEMA, "reviewer": reviewer, "entries": entries})
        return responses

    def _test_package_build(self):
        run = self._run_record()
        catalog = Catalog(self._make_catalog())
        package_doc, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run, catalog)
        if package_doc["schema"] != PACKAGE_SCHEMA or blind_map_doc["schema"] != BLIND_MAP_SCHEMA:
            self.failures.append("package build did not emit the expected schemas")
        if len(package_doc["entries"]) != TOTAL_ENTRIES:
            self.failures.append(f"expected {TOTAL_ENTRIES} blinded entries, got {len(package_doc['entries'])}")
        entry_ids = [entry["entry_id"] for entry in blind_map_doc["entries"]]
        if entry_ids != sorted(entry_ids) or len(set(entry_ids)) != TOTAL_ENTRIES:
            self.failures.append("blind map entries must carry unique, fully blinded L3-XXXX ids")
        package_again, blind_map_again = BlindPackageBuilder(seed=20260821, per_cell=5).build(run, catalog)
        if package_again["entries"] != package_doc["entries"] or blind_map_again["entries"] != blind_map_doc["entries"]:
            self.failures.append("package build must be deterministic for a fixed seed and catalog")
        if any("source" in entry or "case" in entry for entry in package_doc["entries"]):
            self.failures.append("reviewer package must not expose source or case id")
        per_cell_counts = {}
        for entry in blind_map_doc["entries"]:
            key = (entry["challenge"], entry["target_modality"])
            per_cell_counts[key] = per_cell_counts.get(key, 0) + 1
        if len(per_cell_counts) != 20 or any(count != 10 for count in per_cell_counts.values()):
            self.failures.append("each challenge x modality cell must hold exactly 10 entries (5 real + 5 synth)")
        # A cell with too few candidates must refuse to build (no silent partial substitution).
        sparse = {"schema": CATALOG_SCHEMA, "records": self._make_catalog()["records"][:4]}
        try:
            BlindPackageBuilder(seed=20260821, per_cell=5).build(run, Catalog(sparse))
            self.failures.append("a cell with fewer than per-cell images must refuse to build")
        except L3Error:
            pass

    def _test_aggregate_verdict(self):
        run = self._run_record()
        catalog = Catalog(self._make_catalog())
        _, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run, catalog)
        responses = self._responses(blind_map_doc["entries"])
        report = L3ReportProducer(resamples=200, seed=20260821)._produce(run, responses, blind_map_doc, catalog.payload)
        if report["schema"] != REPORT_SCHEMA:
            self.failures.append("aggregate did not emit the expected report schema")
        if report["binding"]["candidate_checkpoint_sha256"] != "c" * 64:
            self.failures.append("report did not bind the frozen candidate checkpoint")
        if not all(result["verdict"] in ("pass", "fail") for result in report["visual_turing"]["per_reviewer"]):
            self.failures.append("per-reviewer visual-Turing verdict must be pass or fail")
        if report["visual_turing"]["verdict"] != "fail":
            self.failures.append(f"an over-confident reviewer must fail the visual-Turing gate, got {report['visual_turing']['verdict']}")
        if report["verdict"]["overall"] != "fail":
            self.failures.append("non-compensatory AND must fail when visual-Turing fails")

    def _test_likert_gate(self):
        # A reviewer set that passes visual-Turing (both near chance) but under-rates Likert must fail the AND.
        run = self._run_record()
        catalog = Catalog(self._make_catalog())
        _, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run, catalog)
        responses = []
        for reviewer in ("R1", "R2"):
            entries = []
            for entry in blind_map_doc["entries"]:
                prediction = "real" if self._deterministic_bit(entry["entry_id"]) == 0 else "synth"
                entries.append(
                    {
                        "entry_id": entry["entry_id"],
                        "turing": prediction,
                        "overall_realism": 4,
                        "anatomical_plausibility": 3,
                        "tumor_authenticity": 4,
                        "artifact_slice_consistency": 4,
                        "notes": "",
                    }
                )
            responses.append({"schema": RESPONSES_SCHEMA, "reviewer": reviewer, "entries": entries})
        report = L3ReportProducer(resamples=200, seed=20260821)._produce(run, responses, blind_map_doc, catalog.payload)
        if report["visual_turing"]["verdict"] != "pass":
            self.failures.append(f"near-chance reviewers must pass visual-Turing, got {report['visual_turing']['verdict']}")
        if report["verdict"]["overall"] != "fail":
            self.failures.append("a sub-4.0 Likert lower bound must fail the non-compensatory AND")

    def _test_determinism(self):
        run = self._run_record()
        catalog = Catalog(self._make_catalog())
        _, blind_map_doc = BlindPackageBuilder(seed=7, per_cell=5).build(run, catalog)
        responses = self._responses(blind_map_doc["entries"])
        first = L3ReportProducer(resamples=100, seed=7)._produce(run, responses, blind_map_doc, catalog.payload)
        second = L3ReportProducer(resamples=100, seed=7)._produce(run, responses, blind_map_doc, catalog.payload)
        if first != second:
            self.failures.append("aggregate must be deterministic for a fixed seed and responses")

    def _test_insufficient_reviewers(self):
        run = self._run_record()
        catalog = Catalog(self._make_catalog())
        _, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run, catalog)
        responses = self._responses(blind_map_doc["entries"])[:1]
        try:
            L3ReportProducer(resamples=100, seed=20260821)._produce(run, responses, blind_map_doc, catalog.payload)
            self.failures.append("aggregate must reject fewer than two independent reviewers")
        except L3Error:
            pass

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._test_package_build()
        self._test_aggregate_verdict()
        self._test_likert_gate()
        self._test_determinism()
        self._test_insufficient_reviewers()
        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-package", help="sample and blind a 200-entry reviewer package")
    p.add_argument("--run", required=True, help="frozen run.json record (binding + freeze guard)")
    p.add_argument("--catalog", required=True, help=f"catalog manifest (schema {CATALOG_SCHEMA})")
    p.add_argument("--output", required=True, help="output directory for package.json + blind_map.json")
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--per-cell", type=int, default=5, help="real and synthetic images drawn per challenge x modality cell")
    p.set_defaults(handler="build-package")

    p = sub.add_parser("aggregate", help="aggregate blinded judgments into a candidate-bound L3 report")
    p.add_argument("--run", required=True)
    p.add_argument(
        "--responses", dest="responses", action="append", required=True, help=f"one blinded reviewer response per flag (schema {RESPONSES_SCHEMA})"
    )
    p.add_argument("--blind-map", required=True, help=f"controlled blind map (schema {BLIND_MAP_SCHEMA})")
    p.add_argument("--catalog", required=True, help=f"catalog manifest (schema {CATALOG_SCHEMA})")
    p.add_argument("--output", required=True)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260821)
    p.set_defaults(handler="aggregate")

    p = sub.add_parser("selftest", help="fixture-driven synthetic L3 checks (stdlib only)")
    p.add_argument("--workdir", required=True)
    p.set_defaults(handler="selftest")

    args = parser.parse_args(argv)
    try:
        if args.handler == "build-package":
            run_record = json.loads(Path(args.run).read_text())
            package_doc, blind_map_doc = BlindPackageBuilder(args.seed, args.per_cell).build(run_record, Catalog.from_path(args.catalog))
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "package.json").write_text(json.dumps(package_doc, indent=2) + "\n")
            (out / "blind_map.json").write_text(json.dumps(blind_map_doc, indent=2) + "\n")
            print(f"package + blind map -> {out}")
            return 0
        if args.handler == "aggregate":
            run_record = json.loads(Path(args.run).read_text())
            blind_map_doc = json.loads(Path(args.blind_map).read_text())
            responses = [json.loads(Path(path).read_text()) for path in args.responses]
            report = L3ReportProducer(args.resamples, args.seed)._produce(
                run_record, responses, blind_map_doc, Catalog.from_path(args.catalog).payload
            )
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(f"L3 report written -> {args.output}")
            return 0
        failures = L3SelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0
    except L3Error as error:
        print(f"L3 ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
