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

"""L2 frozen-instrument final acceptance (issue #55, spec #51 decision 10).

Gives any frozen candidate (P1/P2/P3, issue #53 run contract) its per-challenge
distribution-alignment, P2 round-trip and instrument-applicability verdict on
the five frozen MONAI nnU-Net instruments -- without touching any frozen
artifact. Pass lines are pre-registered in docs/adr/0004 and
docs/calibration/l2-final-acceptance-protocol.md; every number comes from
ADR-0002 (calibration envelopes) and ADR-0003 (frozen-artifact audit).

Pipeline (judgement chain in this file is stdlib-only; NIfTI execution side
lives in nnunet_l2_final_acceptance_nifti.py and runs on sugon):

  assemble    samples manifest + holdout phase manifest -> assembly plan JSON
              (unique obs_id per observation: <case>__real / <case>__gen[__a<anchor>];
              asserts every case is holdout-side and flags provisional when the
              case count falls short of the frozen quota)
  predict     plan -> per-challenge frozen-instrument inference scripts
              (nnUNetv2_predict defaults: mirror TTA on by omission -- the flag
              is store_true, passing "False" enables it -- overlap 0.5, fold 0,
              nnUNetTrainer250Epochs; SSA uses the derived bs16 plans, ADR-0001)
  measure     (nifti side) plan + predictions -> per-observation measurement CSV
  evaluate    measurement CSV + frozen envelopes + freeze-audit verdict ->
              undecided | pass | fail per challenge, non-compensatory AND overall

Verdict chain (ADR-0004): any input_fail / run_fail / ET<=TC<=WT hierarchy
violation on EITHER side -> that challenge is ``undecided`` (instrument
unavailable; blocks the full final acceptance; fix the instrument or re-run,
not the candidate). Without failures: TOST equivalence (per-region rel-volume
diff within +-E_r,vol; per-axis centroid diff within +-E_r,centroid; WT/brain
within E_r,vol[WT]; ET/WT within E_r,vol[ET]+E_r,vol[WT]; case-level cluster
bootstrap, P3 resamples each case's four anchor rounds as one cluster) and,
for P2, condition round-trip Dice whose bootstrap 5th-percentile lower bound
must clear floor == D_r,low (METS floor 0 -> vacuous pass, reported as such).
Wide METS/PED envelopes are carried verbatim and can never be narrowed: any
loaded envelope value drifting from the ADR-0002 literals rejects the run.

Usage:
    python -m scripts.nnunet_l2_final_acceptance assemble --phase P2 \
        --samples samples.json --real-root /root/private_data/raw \
        --holdout-manifest phase_manifest.json --run-id p2-... --output-dir DIR
    python -m scripts.nnunet_l2_final_acceptance predict --plan DIR/plan.json \
        --output-dir DIR
    python -m scripts.nnunet_l2_final_acceptance evaluate --phase P2 \
        --table measurements.csv --freeze-audit freeze-audit.json \
        --run-id p2-... --output-dir DIR
    python -m scripts.nnunet_l2_final_acceptance verify-frozen --freeze-audit freeze-audit.json
    python -m scripts.nnunet_l2_final_acceptance selftest --workdir TMP
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brats_phase_run_contract import ArtifactFingerprinter, ManifestSides  # noqa: E402

PLAN_SCHEMA = "l2-final-acceptance-plan/1"
REPORT_SCHEMA = "l2-final-acceptance-report/1"
PHASES = ("P1", "P2", "P3")
CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")
REGIONS = ("WT", "TC", "ET")
REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNEL_SUFFIXES = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
DATASET_IDS = {"GLI": 501, "SSA": 502, "MEN": 503, "METS": 504, "PED": 505}

# Frozen 20% final-holdout quotas (spec #51 decision 3 / split manifest).
HOLDOUT_QUOTAS = {"GLI": 250, "SSA": 12, "MEN": 200, "METS": 48, "PED": 20}

# Bootstrap registration (protocol §4): B, global seed, per-challenge offsets.
BOOTSTRAP_B = 10_000
GLOBAL_SEED = 20260821
CHALLENGE_SEED_OFFSET = {"GLI": 1, "SSA": 2, "MEN": 3, "METS": 4, "PED": 5}

# ADR-0003 §6: the accepted frozen-artifact audit verdict anchor.
FROZEN_AUDIT_SHA256 = "9121e8ac73f2bdb3999f83c10666c387ccd4c488b3e71b24cf7ab208353e0f82"

# ADR-0002 frozen envelopes (published 4-dp literals; the authoritative source
# for every pass line -- equality against a controlled calibration summary is
# enforced to +-5e-5 so drift AND narrowing both reject).
FROZEN_ENVELOPES = {
    #            region: D_r,low   E_r,vol   E_r,centroid(mm)   R_fail Wilson 95% upper
    "GLI":  {"WT": (0.8053, 0.2802, 5.38),  "TC": (0.6819, 0.4373, 4.79),  "ET": (0.4093, 0.5702, 4.41),  "r_fail_upper": 0.0043},
    "SSA":  {"WT": (0.7046, 0.7223, 12.36), "TC": (0.6578, 0.7767, 8.64),  "ET": (0.7111, 0.7786, 8.42),  "r_fail_upper": 0.0838},
    "MEN":  {"WT": (0.7562, 0.3235, 3.92),  "TC": (0.7208, 0.3576, 6.17),  "ET": (0.7501, 0.3367, 5.70),  "r_fail_upper": 0.0053},
    "METS": {"WT": (0.0000, 1.6510, 28.58), "TC": (0.0000, 1.0000, 35.08), "ET": (0.0000, 1.0000, 35.08), "r_fail_upper": 0.0220},
    "PED":  {"WT": (0.0093, 0.9946, 17.87), "TC": (0.0105, 0.9939, 18.33), "ET": (0.0000, 1.0000, 22.65), "r_fail_upper": 0.0507},
}

MEASUREMENT_FIELDS = [
    "obs_id", "challenge", "case", "side", "anchor",
    "input_fail", "run_fail", "hier_viol", "pred_empty",
    "vol_wt_ml", "vol_tc_ml", "vol_et_ml", "brain_ml", "wt_brain", "et_wt",
    "cx_wt_mm", "cy_wt_mm", "cz_wt_mm",
    "cx_tc_mm", "cy_tc_mm", "cz_tc_mm",
    "cx_et_mm", "cy_et_mm", "cz_et_mm",
    "cond_dice_wt", "cond_dice_tc", "cond_dice_et",
]


class AcceptanceError(Exception):
    """Raised when acceptance setup, freeze verification or judgement rules break."""


# ── frozen envelopes ────────────────────────────────────────────────────

class FrozenEnvelopes:
    """ADR-0002 envelope literals plus the load-and-verify gate (protocol §4).

    Every pass line reads its numbers from here. When a controlled calibration
    summary is supplied, each value must equal the published literal to within
    5e-5 (the 4-dp publishing grid): drift in either direction -- including a
    narrowed margin or a raised floor -- rejects the evaluation.
    """

    TOLERANCE = 5e-5

    def __init__(self, table=None):
        self._table = table or FROZEN_ENVELOPES

    def d_r_low(self, challenge, region):
        return self._table[challenge][region][0]

    def e_r_vol(self, challenge, region):
        return self._table[challenge][region][1]

    def e_r_centroid(self, challenge, region):
        return self._table[challenge][region][2]

    def r_fail_upper(self, challenge):
        return self._table[challenge]["r_fail_upper"]

    def verify_against_summary(self, summary_dir):
        """Checks per-region D_r,low / E_r,vol / E_r,centroid of every challenge."""
        summary_dir = Path(summary_dir)
        problems = []
        for challenge in CHALLENGES:
            path = summary_dir / f"summary_{challenge}.json"
            if not path.is_file():
                problems.append(f"{challenge}: calibration summary missing: {path}")
                continue
            summary = json.loads(path.read_text())
            for region in REGIONS:
                loaded = summary["per_region"][region]
                pairs = (
                    ("D_r_low", loaded["D_r_low"], self.d_r_low(challenge, region)),
                    ("E_r_vol", loaded["E_r_vol"], self.e_r_vol(challenge, region)),
                    ("E_r_centroid", loaded["E_r_centroid"], self.e_r_centroid(challenge, region)),
                )
                for name, live, frozen in pairs:
                    if live is None or math.isnan(live) or abs(live - frozen) > self.TOLERANCE:
                        problems.append(
                            f"{challenge}/{region} {name}: controlled summary {live} != ADR-0002 literal {frozen} "
                            "(envelope drift and narrowing both reject; re-run only via a new ADR)"
                        )
        if problems:
            raise AcceptanceError("frozen-envelope verification failed:\n  " + "\n  ".join(problems))
        return True


# ── freeze guard ────────────────────────────────────────────────────────

class FreezeGuard:
    """Accepts only the ADR-0003 §6 frozen audit verdict (or a re-run of it).

    Default mode pins the published verdict SHA-256; ``--any-verdict`` accepts
    a fresh re-run of scripts/nnunet_l2_freeze_audit.py whose all_passed is
    true (its own hash is recorded into the report instead). Anything narrower
    (spot-checking a subset of hashes) is deliberately not offered.
    """

    def __init__(self, fingerprinter):
        self._fingerprinter = fingerprinter

    def verify(self, verdict_path, expect_sha256=FROZEN_AUDIT_SHA256):
        path = Path(verdict_path)
        if not path.is_file():
            raise AcceptanceError(f"freeze-audit verdict not found: {path}")
        verdict = json.loads(path.read_text())
        if verdict.get("all_passed") is not True:
            raise AcceptanceError("freeze-audit verdict all_passed is not true; the frozen chain is broken, evaluation refuses to run")
        actual_sha = self._fingerprinter.file_sha256(path)
        if expect_sha256 is not None and actual_sha != expect_sha256:
            raise AcceptanceError(
                f"freeze-audit verdict sha256 {actual_sha} != pinned {expect_sha256}; "
                "either pin the ADR-0003 verdict or explicitly accept a fresh re-run"
            )
        return {"path": str(path.resolve()), "sha256": actual_sha, "pinned": expect_sha256 is not None}


# ── assembly planning ───────────────────────────────────────────────────

class RealReferenceResolver:
    """Real-side four-modality paths: explicit ``real_paths`` beat layout lookup.

    Layout fallback mirrors the official BraTS layout used by issue #38:
    ``<real-root>/<CHALLENGE>/<case>/<case>-<suffix>.nii.gz``.
    """

    def __init__(self, real_root):
        self._real_root = Path(real_root)

    def channels(self, entry):
        explicit = entry.get("real_paths")
        if not explicit and "anchors" in entry:  # P3 carries per-anchor real paths
            explicit = {m: entry["anchors"][m]["real"] for m in MODALITIES if entry["anchors"].get(m, {}).get("real")}
        if explicit:
            missing = [m for m in MODALITIES if not explicit.get(m)]
            if missing:
                raise AcceptanceError(f"{entry['case_id']}: real_paths missing modalities {missing}")
            return {CHANNEL_SUFFIXES[m]: explicit[m] for m in MODALITIES}
        case_dir = self._real_root / entry["challenge"] / entry["case_id"]
        paths = {}
        for modality in MODALITIES:
            path = case_dir / f"{entry['case_id']}-{modality}.nii.gz"
            if not path.is_file():
                raise AcceptanceError(
                    f"real reference image not found: {path} "
                    f"(supply real_paths for {entry['case_id']} or fix --real-root)"
                )
            paths[CHANNEL_SUFFIXES[modality]] = str(path)
        return paths


class P1PseudoQuadPlan:
    """P1 assembly protocol: four independent per-modality samples per case.

    Independence is a manifest obligation: the four entries must carry pairwise
    distinct noise seeds (same seed = same noise = not independent sampling).
    """

    def observations(self, entry, resolver):
        samples = self._samples(entry)
        seeds = [samples[m]["seed"] for m in MODALITIES]
        if len(set(seeds)) != len(seeds):
            raise AcceptanceError(f"{entry['case_id']}: P1 pseudo-quad needs four distinct noise seeds, got {seeds}")
        channels = {CHANNEL_SUFFIXES[m]: samples[m]["path"] for m in MODALITIES}
        return [
            {"obs_id": f"{entry['case_id']}__real", "challenge": entry["challenge"], "case": entry["case_id"],
             "side": "real", "anchor": None, "channels": resolver.channels(entry), "condition_mask": None},
            {"obs_id": f"{entry['case_id']}__gen", "challenge": entry["challenge"], "case": entry["case_id"],
             "side": "gen", "anchor": None, "channels": channels, "condition_mask": None},
        ]

    @staticmethod
    def _samples(entry):
        samples = entry.get("samples")
        if not samples or set(samples) != set(MODALITIES):
            raise AcceptanceError(f"{entry.get('case_id')}: P1 samples must carry exactly {list(MODALITIES)}")
        for modality in MODALITIES:
            item = samples[modality]
            if not item.get("path") or "seed" not in item:
                raise AcceptanceError(f"{entry.get('case_id')}: samples[{modality}] needs path and seed")
        return samples


class P2SharedMaskPlan:
    """P2 assembly protocol: one condition mask, four modalities of the same case."""

    def observations(self, entry, resolver):
        condition_mask = entry.get("condition_mask")
        if not condition_mask:
            raise AcceptanceError(f"{entry['case_id']}: P2 requires condition_mask")
        samples = entry.get("samples")
        if not samples or set(samples) != set(MODALITIES):
            raise AcceptanceError(f"{entry['case_id']}: P2 samples must carry exactly {list(MODALITIES)}")
        channels = {}
        for modality in MODALITIES:
            if not samples[modality].get("path"):
                raise AcceptanceError(f"{entry['case_id']}: samples[{modality}] needs a path")
            channels[CHANNEL_SUFFIXES[modality]] = samples[modality]["path"]
        return [
            {"obs_id": f"{entry['case_id']}__real", "challenge": entry["challenge"], "case": entry["case_id"],
             "side": "real", "anchor": None, "channels": resolver.channels(entry), "condition_mask": None},
            {"obs_id": f"{entry['case_id']}__gen", "challenge": entry["challenge"], "case": entry["case_id"],
             "side": "gen", "anchor": None, "channels": channels, "condition_mask": condition_mask},
        ]


class P3FourAnchorPlan:
    """P3 assembly protocol: four anchor rounds covering all 12 ordered pairs.

    One real observation per case (shared reference measurement); each anchor
    round contributes one generated observation whose four channels are the
    real anchor modality plus three anchor-conditioned generations.
    """

    def observations(self, entry, resolver):
        anchors = entry.get("anchors")
        if not anchors or set(anchors) != set(MODALITIES):
            raise AcceptanceError(f"{entry.get('case_id')}: P3 anchors must carry exactly {list(MODALITIES)}")
        real_channels = resolver.channels(entry)
        observations = [{"obs_id": f"{entry['case_id']}__real", "challenge": entry["challenge"],
                         "case": entry["case_id"], "side": "real", "anchor": None,
                         "channels": real_channels, "condition_mask": None}]
        for anchor in MODALITIES:
            round_info = anchors[anchor]
            if not round_info.get("real") or not round_info.get("generated"):
                raise AcceptanceError(f"{entry['case_id']}: anchor {anchor} needs real and generated")
            targets = [m for m in MODALITIES if m != anchor]
            if set(round_info["generated"]) != set(targets):
                raise AcceptanceError(
                    f"{entry['case_id']}: anchor {anchor} must generate exactly {targets}, got {sorted(round_info['generated'])}"
                )
            channels = dict(real_channels)
            channels[CHANNEL_SUFFIXES[anchor]] = round_info["real"]
            for target in targets:
                if not round_info["generated"][target].get("path"):
                    raise AcceptanceError(f"{entry['case_id']}: anchor {anchor} -> {target} needs a path")
                channels[CHANNEL_SUFFIXES[target]] = round_info["generated"][target]["path"]
            observations.append({
                "obs_id": f"{entry['case_id']}__gen__a{anchor}", "challenge": entry["challenge"],
                "case": entry["case_id"], "side": "gen", "anchor": anchor,
                "channels": channels, "condition_mask": None,
            })
        return observations


class AssemblyPlanner:
    """Turns a frozen candidate's samples manifest into the two-sided assembly plan."""

    def __init__(self, phase, plan_strategy, resolver, holdout_sides, fingerprinter):
        self._phase = phase
        self._strategy = plan_strategy
        self._resolver = resolver
        self._holdout_sides = holdout_sides
        self._fingerprinter = fingerprinter

    def build(self, samples_path, holdout_manifest_path, run_id, real_root):
        entries = json.loads(Path(samples_path).read_text())
        if not entries:
            raise AcceptanceError("samples manifest is empty")
        observations = []
        case_counts = {}
        for entry in entries:
            if entry.get("phase") != self._phase:
                raise AcceptanceError(
                    f"entry {entry.get('case_id')} declares phase {entry.get('phase')!r}; this assembly is {self._phase}"
                )
            side = self._holdout_sides.side_of(entry["challenge"], entry["case_id"])
            if side is None:
                raise AcceptanceError(f"({entry['challenge']}, {entry['case_id']}) is not in the holdout manifest")
            if side != "holdout":
                raise AcceptanceError(
                    f"({entry['challenge']}, {entry['case_id']}) is {side}-side; final acceptance runs on the "
                    "20% holdout side only (spec decision 3)"
                )
            observations += self._strategy.observations(entry, self._resolver)
            case_counts.setdefault(entry["challenge"], set()).add(entry["case_id"])
        obs_ids = [obs["obs_id"] for obs in observations]
        if len(set(obs_ids)) != len(obs_ids):
            raise AcceptanceError("duplicate obs_id in assembly plan (observations would overwrite each other)")
        challenges = {}
        for challenge, cases in sorted(case_counts.items()):
            quota = HOLDOUT_QUOTAS[challenge]
            challenges[challenge] = {
                "n_cases": len(cases),
                "quota": quota,
                "provisional": len(cases) < quota,
            }
        return {
            "schema": PLAN_SCHEMA,
            "phase": self._phase,
            "run_id": run_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "samples_manifest": self._fingerprinter.must_fingerprint(samples_path, "samples manifest"),
            "holdout_manifest": self._fingerprinter.must_fingerprint(holdout_manifest_path, "holdout manifest"),
            "real_root": str(Path(real_root).resolve()),
            "challenges": challenges,
            "observations": observations,
        }


# ── frozen-instrument inference scripting ───────────────────────────────

class PredictScriptWriter:
    """Per-challenge inference scripts under the frozen ADR-0003 §4 config.

    Mirror TTA stays ON by omission (--disable_tta is store_true; passing any
    value, including False, would turn it off). SSA uses the ADR-0001 derived
    batch-16 plans identifier and configuration.
    """

    def __init__(self, plan, output_dir):
        self._plan = plan
        self._output_dir = Path(output_dir)

    def write(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        input_root = self._output_dir / "inputs"
        scripts = []
        for challenge in sorted(self._plan["challenges"]):
            cmd = [
                "nnUNetv2_predict",
                "-i", str(input_root / challenge),
                "-o", str(self._output_dir / "predictions" / challenge),
                "-d", str(DATASET_IDS[challenge]),
                "-c", "3d_fullres_bs16" if challenge == "SSA" else "3d_fullres",
                "-f", "0",
                "-tr", "nnUNetTrainer250Epochs",
            ]
            if challenge == "SSA":
                cmd += ["-p", "nnUNetPlans_SSA_bs16_v1"]
            script_path = self._output_dir / f"predict_{challenge}.sh"
            script_path.write_text("#!/bin/bash\nset -euo pipefail\n" + " ".join(cmd) + "\n")
            script_path.chmod(0o755)
            scripts.append(script_path)
        runner = self._output_dir / "predict_all.sh"
        runner.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            + "".join(f"bash {script.name}\n" for script in scripts)
        )
        runner.chmod(0o755)
        print(f"[OK] wrote {len(scripts)} frozen-instrument predict scripts; runner: {runner}")
        return runner


# ── measurement table ───────────────────────────────────────────────────

class MeasurementTable:
    """CSV persistence for per-observation measurements (controlled storage, subject IDs)."""

    @classmethod
    def write(cls, rows, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @classmethod
    def read(cls, path):
        with open(Path(path), newline="") as handle:
            rows = list(csv.DictReader(handle))
        missing = set(MEASUREMENT_FIELDS) - set(rows[0].keys()) if rows else set(MEASUREMENT_FIELDS)
        if missing:
            raise AcceptanceError(f"measurement table {path} is missing columns: {sorted(missing)}")
        return rows

    @staticmethod
    def number(row, field):
        """Parses a measurement cell; empty/None -> None (undefined quantity)."""
        value = row.get(field)
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if math.isnan(parsed) else parsed

    @staticmethod
    def flag(row, field):
        return str(row.get(field, "")).strip().lower() in ("1", "true", "yes")


# ── statistics ──────────────────────────────────────────────────────────

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


class QuantityFamily:
    """One registered TOST quantity: how to pair, exclude and bound it.

    ``relative`` quantities test (gen - real)/real against an additive margin
    built from frozen E_r,vol values; centroid axes test signed mm differences
    against E_r,centroid. Exclusions happen only when the real-side denominator
    (or either-side mask) is undefined -- a generated-side empty prediction
    always stays in the volume distributions at rel diff -1.0 (protocol §4).
    """

    def __init__(self, name, field, margin_builder, relative, exclusion):
        self.name = name
        self.field = field
        self.margin_builder = margin_builder
        self.relative = relative
        self.exclusion = exclusion

    def margin(self, envelopes, challenge):
        return self.margin_builder(envelopes, challenge)

    def pair(self, gen_row, real_row):
        real = MeasurementTable.number(real_row, self.field)
        gen = MeasurementTable.number(gen_row, self.field)
        reason = self.exclusion(real_row=real_row, gen_row=gen_row)
        if reason:
            return None, reason
        if real is None or gen is None:
            return None, "undefined_measurement"
        if self.relative:
            if real == 0:
                return None, "real_denominator_zero"
            return (gen - real) / real, None
        return gen - real, None


class QuantityRegistry:
    """The pre-registered quantity list (protocol §4 / ADR-0004 decision 1)."""

    def __init__(self):
        vol_margin = {region: (lambda envelopes, challenge, region=region: envelopes.e_r_vol(challenge, region))
                      for region in REGIONS}
        cent_margin = {region: (lambda envelopes, challenge, region=region: envelopes.e_r_centroid(challenge, region))
                       for region in REGIONS}
        self._quantities = []
        for region in REGIONS:
            vol_field = {"WT": "vol_wt_ml", "TC": "vol_tc_ml", "ET": "vol_et_ml"}[region]
            self._quantities.append(QuantityFamily(
                f"vol_{region.lower()}_rel", vol_field, vol_margin[region], relative=True,
                exclusion=lambda real_row, gen_row, region=region:
                    None if (MeasurementTable.number(real_row, vol_field) or 0) > 0 else "real_volume_zero",
            ))
            for axis in ("x", "y", "z"):
                centroid_field = f"c{axis}_{region.lower()}_mm"
                self._quantities.append(QuantityFamily(
                    f"centroid_{region.lower()}_{axis}", centroid_field, cent_margin[region], relative=False,
                    exclusion=lambda real_row, gen_row, region=region:
                        None if (MeasurementTable.number(real_row, vol_field) or 0) > 0
                        and (MeasurementTable.number(gen_row, vol_field) or 0) > 0 else "empty_mask_side",
                ))
        self._quantities.append(QuantityFamily(
            "wt_brain_rel", "wt_brain", vol_margin["WT"], relative=True,
            exclusion=lambda real_row, gen_row:
                None if (MeasurementTable.number(real_row, "wt_brain") or 0) > 0 else "real_wt_brain_zero",
        ))
        self._quantities.append(QuantityFamily(
            "et_wt_rel", "et_wt",
            lambda envelopes, challenge: envelopes.e_r_vol(challenge, "ET") + envelopes.e_r_vol(challenge, "WT"),
            relative=True,
            exclusion=lambda real_row, gen_row:
                None if (MeasurementTable.number(real_row, "et_wt") or 0) > 0 else "real_et_wt_undefined",
        ))

    def all(self):
        return list(self._quantities)


# ── judgement chain ─────────────────────────────────────────────────────

class FailureGate:
    """The undecided gate: any input/run/hierarchy failure on either side (ADR-0004 decision 4)."""

    @staticmethod
    def audit(rows):
        breakdown = {"input_fail": 0, "run_fail": 0, "hier_viol": 0}
        failing = []
        for row in rows:
            failures = [name for name in breakdown if MeasurementTable.flag(row, name)]
            if failures:
                failing.append({"obs_id": row["obs_id"], "side": row["side"], "failures": failures})
                for name in failures:
                    breakdown[name] += 1
        return {"n_failed": len(failing), "breakdown": breakdown, "failing": failing}


class ChallengeJudge:
    """Full per-challenge verdict: failure gate -> TOST (+ P2 round-trip)."""

    def __init__(self, envelopes, bootstrap, phase):
        self._envelopes = envelopes
        self._bootstrap = bootstrap
        self._phase = phase
        self._registry = QuantityRegistry()

    def _pair_cases(self, rows):
        real_by_case = {}
        gen_by_case = {}
        for row in rows:
            if row["side"] == "real":
                if row["case"] in real_by_case:
                    raise AcceptanceError(f"multiple real rows for case {row['case']}")
                real_by_case[row["case"]] = row
            else:
                gen_by_case.setdefault(row["case"], []).append(row)
        missing = sorted(set(gen_by_case) - set(real_by_case))
        if missing:
            raise AcceptanceError(f"generated observations without a real reference row: {missing[:3]}")
        return real_by_case, gen_by_case

    def tost(self, rows, challenge, seed_base):
        real_by_case, gen_by_case = self._pair_cases(rows)
        results = []
        for index, quantity in enumerate(self._registry.all()):
            margin = quantity.margin(self._envelopes, challenge)
            per_case, exclusions = [], {}
            for case, gen_rows in sorted(gen_by_case.items()):
                diffs = []
                for gen_row in gen_rows:
                    diff, reason = quantity.pair(gen_row, real_by_case[case])
                    if diff is None:
                        exclusions[reason] = exclusions.get(reason, 0) + 1
                    else:
                        diffs.append(diff)
                per_case.append(diffs)
            ci = self._bootstrap.ci90(per_case, seed=seed_base + index)
            if ci is None:
                passed = False
                ci = {"low": None, "high": None, "n_cases": 0}
            else:
                passed = ci["low"] >= -margin - 1e-12 and ci["high"] <= margin + 1e-12
            results.append({
                "quantity": quantity.name, "margin": margin,
                "ci90_low": ci["low"], "ci90_high": ci["high"], "n_cases": ci["n_cases"],
                "n_excluded": sum(exclusions.values()), "exclusion_reasons": exclusions,
                "passed": passed,
            })
        return results

    def round_trip(self, rows, challenge, seed_base):
        """P2 condition round-trip: bootstrap q5 lower bound per region vs floor == D_r,low."""
        if self._phase != "P2":
            return None
        real_by_case, gen_by_case = self._pair_cases(rows)
        results = []
        for region_index, region in enumerate(REGIONS):
            field = f"cond_dice_{region.lower()}"
            floor = self._envelopes.d_r_low(challenge, region)
            per_case = []
            for case, gen_rows in sorted(gen_by_case.items()):
                for gen_row in gen_rows:
                    value = MeasurementTable.number(gen_row, field)
                    per_case.append([] if value is None else [value])
            stats = self._bootstrap.q5_lower_bound(per_case, seed=seed_base + 100 + region_index)
            vacuous = floor == 0.0
            if stats is None:
                passed, bound = False, None
            else:
                bound = stats["bound"]
                passed = vacuous or bound >= floor - 1e-12
            results.append({
                "region": region, "floor": floor, "bound": bound,
                "n_cases": stats["n_cases"] if stats else 0,
                "vacuous_pass": vacuous, "passed": passed,
            })
        return results

    def judge(self, rows, challenge, seed):
        audit = FailureGate.audit(rows)
        verdict = {
            "challenge": challenge,
            "n_observations": len(rows),
            "failure_audit": audit,
            "r_fail_point": audit["n_failed"] / len(rows) if rows else None,
            "r_fail_diagnostic_upper_envelope": self._envelopes.r_fail_upper(challenge),
            "tost": None,
            "round_trip": None,
        }
        if audit["n_failed"]:
            verdict["verdict"] = "undecided"
            verdict["reason"] = (
                "instrument failure on tested samples (input/run/hierarchy); blocks final acceptance -- "
                "fix direction is the instrument or a re-run, not the candidate"
            )
            return verdict
        verdict["tost"] = self.tost(rows, challenge, seed)
        verdict["round_trip"] = self.round_trip(rows, challenge, seed)
        checks = [item["passed"] for item in verdict["tost"]]
        if verdict["round_trip"] is not None:
            checks += [item["passed"] for item in verdict["round_trip"]]
        verdict["verdict"] = "pass" if all(checks) else "fail"
        return verdict


class AcceptanceReport:
    """Aggregate, subject-id-free JSON + markdown report (protocol §5)."""

    def __init__(self, phase, run_id, bootstrap_b, freeze_record, provisional_challenges):
        self._phase = phase
        self._run_id = run_id
        self._bootstrap_b = bootstrap_b
        self._freeze_record = freeze_record
        self._provisional = provisional_challenges

    def build(self, challenge_verdicts):
        order = {"undecided": 0, "fail": 1, "pass": 2}
        overall = "undecided" if any(v["verdict"] == "undecided" for v in challenge_verdicts) else (
            "pass" if all(v["verdict"] == "pass" for v in challenge_verdicts) else "fail"
        )
        report = {
            "schema": REPORT_SCHEMA,
            "title": "L2 冻结仪器最终验收报告",
            "issue": 55,
            "phase": self._phase,
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bootstrap": {
                "B": self._bootstrap_b, "method": "病例级 cluster bootstrap, percentile CI90, linear 插值 (protocol §4)",
                "global_seed": GLOBAL_SEED, "per_challenge_offset": CHALLENGE_SEED_OFFSET,
            },
            "frozen_audit": self._freeze_record,
            "envelopes_source": "ADR-0002 literals (docs/adr/0002-l2-instrument-calibration-envelopes.md)",
            "pass_lines": "ADR-0004 (docs/adr/0004-l2-final-acceptance-pass-lines.md)",
            "provisional_challenges": self._provisional,
            "overall_verdict": overall,
            "per_challenge": {v["challenge"]: v for v in challenge_verdicts},
        }
        if "METS" in report["per_challenge"]:
            report["mets_resolving_power_note"] = (
                "METS 宽包络如实保留 (ADR-0002): ±1.0–1.65 体积 margin 下 TC/ET 全漏仍可通过等价检验, "
                "这是该仪器在 METS 上的真实区分力极限, 不构成生成质量证据"
            )
        return report

    def write(self, report, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"l2_final_acceptance_{self._phase.lower()}.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / f"l2_final_acceptance_{self._phase.lower()}.md"
        md_path.write_text(self._markdown(report))
        return json_path, md_path

    def _markdown(self, report):
        lines = [
            f"# {report['title']}",
            "",
            f"**Issue**: [#55](https://github.com/ACautomata/NV-Generate-CTMR/issues/55) · "
            f"**Phase**: {report['phase']} · **Run**: `{report['run_id']}`",
            f"**总体判定**: **{report['overall_verdict'].upper()}**"
            + (" (provisional: " + ", ".join(sorted(self._provisional)) + " 观测不足冻结配额)" if self._provisional else ""),
            "",
            "| 挑战 | 观测数 | 失败数 | TOST 未过 | 回切未过 | 判定 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for challenge, verdict in report["per_challenge"].items():
            tost_failed = sum(0 if verdict["tost"] is None else (not item["passed"]) for item in (verdict["tost"] or []))
            rt_failed = sum(0 if verdict["round_trip"] is None else (not item["passed"]) for item in (verdict["round_trip"] or []))
            lines.append(
                f"| {challenge} | {verdict['n_observations']} | {verdict['failure_audit']['n_failed']} "
                f"| {tost_failed} | {rt_failed} | **{verdict['verdict']}** |"
            )
        lines += ["", "## TOST 明细(未通过项)", ""]
        for challenge, verdict in report["per_challenge"].items():
            for item in verdict["tost"] or []:
                if not item["passed"]:
                    lines.append(
                        f"- {challenge}/{item['quantity']}: CI90 [{item['ci90_low']:.4f}, {item['ci90_high']:.4f}] "
                        f"⊄ ±{item['margin']:.4f} (排除 {item['n_excluded']})"
                    )
        if report["phase"] == "P2":
            lines += ["", "## P2 条件回切", ""]
            for challenge, verdict in report["per_challenge"].items():
                for item in verdict["round_trip"] or []:
                    tag = " (vacuous-pass, floor=0)" if item["vacuous_pass"] else ""
                    lines.append(
                        f"- {challenge}/{item['region']}: q5 下界 {item['bound']:.4f} vs floor {item['floor']:.4f}{tag}"
                    )
        lines += ["", "## 冻结与合规", "",
                  f"- 冻结审计 verdict: `{report['frozen_audit']['sha256'][:16]}…` (pinned={report['frozen_audit']['pinned']})",
                  "- 仪器权重/plans/推理配置/校准包络未做任何修改; 逐病例测量(含 subject ID)只在受控存储。",
                  "- METS/PED 宽包络按 ADR-0002 原样适用, 未收窄。" + report.get("mets_resolving_power_note", ""),
                  ""]
        return "\n".join(lines)


# ── selftest (stdlib-only, synthetic non-subject ids) ───────────────────

class SelfTest:
    """Fixture-driven end-to-end check of the judgement chain and guards."""

    WORK_ENVELOPES = FROZEN_ENVELOPES  # the real frozen literals: boundaries must hold against them

    def __init__(self, workdir, bootstrap_b=400):
        self._workdir = Path(workdir)
        self._bootstrap_b = bootstrap_b
        self.failures = []

    def expect_reject(self, action, label):
        try:
            action()
        except AcceptanceError:
            return
        self.failures.append(f"expected rejection but succeeded: {label}")

    def fixture_measurement_row(self, obs_id, challenge, case, side, anchor=None, **overrides):
        row = {field: "" for field in MEASUREMENT_FIELDS}
        row.update(
            obs_id=obs_id, challenge=challenge, case=case, side=side, anchor=anchor or "",
            input_fail="0", run_fail="0", hier_viol="0", pred_empty="0",
            vol_wt_ml="50.0", vol_tc_ml="30.0", vol_et_ml="10.0", brain_ml="1200.0",
            wt_brain="0.0417", et_wt="0.20",
            cx_wt_mm="120.0", cy_wt_mm="120.0", cz_wt_mm="77.0",
            cx_tc_mm="121.0", cy_tc_mm="121.0", cz_tc_mm="78.0",
            cx_et_mm="122.0", cy_et_mm="122.0", cz_et_mm="79.0",
            cond_dice_wt="0.95", cond_dice_tc="0.93", cond_dice_et="0.90",
        )
        row.update(overrides)
        return row

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._test_envelope_verification()
        self._test_freeze_guard()
        self._test_assembly_plans()
        self._test_verdict_chain()
        return self.failures

    def _test_envelope_verification(self):
        envelopes = FrozenEnvelopes()
        summary_dir = self._workdir / "calibration_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        for challenge in CHALLENGES:
            summary = {"per_region": {
                region: {
                    "D_r_low": envelopes.d_r_low(challenge, region),
                    "E_r_vol": envelopes.e_r_vol(challenge, region),
                    "E_r_centroid": envelopes.e_r_centroid(challenge, region),
                } for region in REGIONS}}
            (summary_dir / f"summary_{challenge}.json").write_text(json.dumps(summary))
        envelopes.verify_against_summary(summary_dir)  # exact literals pass

        drifted = json.loads((summary_dir / "summary_GLI.json").read_text())
        drifted["per_region"]["WT"]["E_r_vol"] = 0.2000  # narrowed / drifted
        (summary_dir / "summary_GLI.json").write_text(json.dumps(drifted))
        self.expect_reject(lambda: envelopes.verify_against_summary(summary_dir), "drifted envelope value")
        (summary_dir / "summary_GLI.json").unlink()
        self.expect_reject(lambda: envelopes.verify_against_summary(summary_dir), "missing calibration summary")
        # METS floors stay zero: the wide envelope is carried, never narrowed.
        for region in REGIONS:
            if envelopes.d_r_low("METS", region) != 0.0:
                self.failures.append(f"METS {region} D_r,low is not the frozen 0")

    def _test_freeze_guard(self):
        guard = FreezeGuard(ArtifactFingerprinter())
        verdict = {"all_passed": True, "challenges": []}
        good = self._workdir / "freeze_audit_good.json"
        payload = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
        good.write_text(payload)
        pinned = hashlib.sha256(payload.encode()).hexdigest()
        guard.verify(good, expect_sha256=pinned)  # pinned hash passes
        bad_hash = self._workdir / "freeze_audit_bad_hash.json"
        bad_hash.write_text(payload)
        self.expect_reject(lambda: guard.verify(bad_hash, expect_sha256="0" * 64), "verdict hash mismatch")
        failed = self._workdir / "freeze_audit_failed.json"
        failed_payload = json.dumps({"all_passed": False}) + "\n"
        failed.write_text(failed_payload)
        self.expect_reject(lambda: guard.verify(failed, expect_sha256=hashlib.sha256(failed_payload.encode()).hexdigest()),
                           "all_passed false")
        # ADR-0003 anchor itself is loadable.
        record = guard.verify(good, expect_sha256=None)
        if record["pinned"] is not False:
            self.failures.append("freeze guard pinned flag misreported")

    def _fixture_entry(self, challenge, case, phase):
        real_paths = {m: f"/private/real/{challenge}/{case}/{case}-{m}.nii.gz" for m in MODALITIES}
        if phase == "P1":
            return {"case_id": case, "challenge": challenge, "phase": "P1",
                    "samples": {m: {"path": f"/private/gen/{case}-{m}.nii.gz", "seed": 100 + i}
                                for i, m in enumerate(MODALITIES)},
                    "real_paths": real_paths}
        if phase == "P2":
            return {"case_id": case, "challenge": challenge, "phase": "P2",
                    "condition_mask": f"/private/cond/{case}-cond.nii.gz",
                    "samples": {m: {"path": f"/private/gen/{case}-{m}.nii.gz"} for m in MODALITIES},
                    "real_paths": real_paths}
        return {"case_id": case, "challenge": challenge, "phase": "P3",
                "anchors": {m: {"real": f"/private/real/{challenge}/{case}/{case}-{m}.nii.gz",
                                "generated": {t: {"path": f"/private/gen/{case}-{t}-from-{m}.nii.gz"}
                                              for t in MODALITIES if t != m}}
                            for m in MODALITIES}}

    def _holdout_manifest(self):
        manifest = {"split_id": "selftest", "challenges": {}}
        for challenge in CHALLENGES:
            manifest["challenges"][challenge] = {"cases": {
                "train": [f"FIX{challenge}-0000-{i:03d}" for i in range(2)],
                "dev": [f"FIX{challenge}-0100-{i:03d}" for i in range(2)],
                "holdout": [f"FIX{challenge}-0200-{i:03d}" for i in range(HOLDOUT_QUOTAS[challenge])],
            }}
        path = self._workdir / "holdout_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path, manifest

    def _planner(self, phase, manifest_path):
        strategies = {"P1": P1PseudoQuadPlan(), "P2": P2SharedMaskPlan(), "P3": P3FourAnchorPlan()}
        return AssemblyPlanner(phase, strategies[phase], RealReferenceResolver(self._workdir / "real"),
                               ManifestSides(json.loads(Path(manifest_path).read_text())), ArtifactFingerprinter())

    def _write_samples(self, entries, name):
        path = self._workdir / name
        path.write_text(json.dumps(entries, indent=2) + "\n")
        return path

    def _test_assembly_plans(self):
        manifest_path, manifest = self._holdout_manifest()
        holdout_case = {ch: manifest["challenges"][ch]["cases"]["holdout"][0] for ch in CHALLENGES}

        # P1: distinct seeds pass; identical seeds reject; wrong side rejects; quota shortfall -> provisional.
        planner = self._planner("P1", manifest_path)
        entry = self._fixture_entry("GLI", holdout_case["GLI"], "P1")
        samples = self._write_samples([entry], "p1_samples.json")
        plan = planner.build(samples, manifest_path, "p1-run", self._workdir / "real")
        obs_ids = [obs["obs_id"] for obs in plan["observations"]]
        if sorted(obs_ids) != sorted([f"{holdout_case['GLI']}__real", f"{holdout_case['GLI']}__gen"]):
            self.failures.append(f"P1 plan observations wrong: {obs_ids}")
        if not plan["challenges"]["GLI"]["provisional"]:
            self.failures.append("P1 single-case plan must be provisional (quota 250)")
        same_seed = {m: {"path": entry["samples"][m]["path"], "seed": 7} for m in MODALITIES}
        bad_entry = {**entry, "samples": same_seed}
        bad_samples = self._write_samples([bad_entry], "p1_bad_seed.json")
        self.expect_reject(lambda: planner.build(bad_samples, manifest_path, "p1-bad", self._workdir / "real"),
                           "P1 with identical seeds")
        dev_entry = self._fixture_entry("GLI", manifest["challenges"]["GLI"]["cases"]["dev"][0], "P1")
        dev_samples = self._write_samples([dev_entry], "p1_dev_case.json")
        self.expect_reject(lambda: planner.build(dev_samples, manifest_path, "p1-dev", self._workdir / "real"),
                           "dev-side case in final acceptance")
        wrong_phase = self._write_samples([self._fixture_entry("GLI", holdout_case["GLI"], "P2")], "p1_wrong_phase.json")
        self.expect_reject(lambda: planner.build(wrong_phase, manifest_path, "p1-wrong", self._workdir / "real"),
                           "phase-mismatched sample entry")

        # P2: missing condition mask rejects.
        planner2 = self._planner("P2", manifest_path)
        entry2 = self._fixture_entry("MEN", holdout_case["MEN"], "P2")
        samples2 = self._write_samples([entry2], "p2_samples.json")
        planner2.build(samples2, manifest_path, "p2-run", self._workdir / "real")
        maskless = {k: v for k, v in entry2.items() if k != "condition_mask"}
        self.expect_reject(
            lambda: planner2.build(self._write_samples([maskless], "p2_maskless.json"), manifest_path, "p2-bad", self._workdir / "real"),
            "P2 without condition mask")

        # P3: four anchor rounds per case, unique obs ids; a dropped round rejects.
        planner3 = self._planner("P3", manifest_path)
        entry3 = self._fixture_entry("SSA", holdout_case["SSA"], "P3")
        samples3 = self._write_samples([entry3], "p3_samples.json")
        plan3 = planner3.build(samples3, manifest_path, "p3-run", self._workdir / "real")
        gen_obs = [obs for obs in plan3["observations"] if obs["side"] == "gen"]
        if len(gen_obs) != 4 or len({obs["anchor"] for obs in gen_obs}) != 4:
            self.failures.append(f"P3 plan must carry four distinct anchor rounds, got {len(gen_obs)}")
        if len({obs["obs_id"] for obs in plan3["observations"]}) != 5:
            self.failures.append("P3 plan obs_ids are not unique")
        broken = {m: entry3["anchors"][m] for m in ("t1n", "t1c", "t2w")}
        broken_entry = {**entry3, "anchors": broken}
        self.expect_reject(
            lambda: planner3.build(self._write_samples([broken_entry], "p3_broken.json"), manifest_path, "p3-bad", self._workdir / "real"),
            "P3 with a missing anchor round")

    def _challenge_rows(self, challenge, cases, phase, mutate=None):
        rows = []
        for index, case in enumerate(cases):
            real = self.fixture_measurement_row(f"{case}__real", challenge, case, "real")
            gen_anchors = [None] if phase != "P3" else list(MODALITIES)
            for anchor in gen_anchors:
                suffix = "" if anchor is None else f"__a{anchor}"
                gen = self.fixture_measurement_row(f"{case}__gen{suffix}", challenge, case, "gen", anchor)
                if mutate is not None:
                    mutate(index, case, anchor, real, gen)
                rows.append(gen)
            rows.append(real)
        return rows

    def _test_verdict_chain(self):
        envelopes = FrozenEnvelopes()
        bootstrap = ClusterBootstrap(self._bootstrap_b)
        seed = GLOBAL_SEED + CHALLENGE_SEED_OFFSET["GLI"]

        # Equivalent volumes -> TOST pass (differences well inside GLI margins).
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: gen.update(vol_wt_ml="51.0", vol_tc_ml="30.5", vol_et_ml="10.2"))
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        if verdict["verdict"] != "pass":
            self.failures.append(f"GLI P1 equivalent volumes should pass, got {verdict['verdict']}: "
                                 f"{[q for q in verdict['tost'] if not q['passed']]}")

        # One hierarchy violation anywhere -> undecided and it blocks.
        def break_one(index, case, anchor, real, gen):
            if index == 2 and anchor is None:
                gen.update(hier_viol="1")
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1", mutate=break_one)
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        if verdict["verdict"] != "undecided" or verdict["failure_audit"]["n_failed"] != 1:
            self.failures.append(f"single hier violation must give undecided, got {verdict['verdict']}")
        # A real-side input failure is equally undecided (reference chain broken).
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: real.update(input_fail="1") if i == 0 else None)
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        if verdict["verdict"] != "undecided":
            self.failures.append("real-side input failure must give undecided")

        # Volume bias far outside the GLI WT margin -> fail.
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: gen.update(vol_wt_ml="80.0"))
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        if verdict["verdict"] != "fail" or all(q["passed"] for q in verdict["tost"]):
            self.failures.append("60% WT volume bias must fail the GLI TOST")

        # METS keeps its wide envelope: the same 60% bias is inside +-1.651 -> pass.
        rows = self._challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: gen.update(vol_wt_ml="80.0"))
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "METS", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["METS"])
        if not any(q["quantity"] == "vol_wt_rel" and q["passed"] for q in verdict["tost"]):
            self.failures.append("METS wide volume envelope must carry the same bias as pass (resolving-power limit)")

        # Centroid axis shift beyond E_r,centroid -> fail.
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: gen.update(cx_wt_mm="130.0", cx_tc_mm="131.0", cx_et_mm="132.0"))
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        if verdict["verdict"] != "fail":
            self.failures.append("10mm centroid shift must fail GLI (E_r,centroid 5.38/4.79/4.41)")

        # Real-side WT volume zero -> exclusion is counted, never silent; ET/WT margin = ET+WT.
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: real.update(vol_et_ml="0.0", et_wt="") if i < 3 else None)
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "GLI", seed)
        et_vol = next(q for q in verdict["tost"] if q["quantity"] == "vol_et_rel")
        et_wt = next(q for q in verdict["tost"] if q["quantity"] == "et_wt_rel")
        if et_vol["n_excluded"] != 3 or et_wt["n_excluded"] != 3:
            self.failures.append(f"real-side zero ET must exclude and count 3, got {et_vol['n_excluded']}/{et_wt['n_excluded']}")
        if et_wt["margin"] != envelopes.e_r_vol("GLI", "ET") + envelopes.e_r_vol("GLI", "WT"):
            self.failures.append("et_wt margin must be E_r,vol[ET] + E_r,vol[WT]")

        # Generated-side empty prediction stays in the volume distribution at -1.0 (no exclusion).
        rows = self._challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P1",
                                    mutate=lambda i, case, a, real, gen: gen.update(vol_wt_ml="0.0") if i < 3 else None)
        verdict = ChallengeJudge(envelopes, bootstrap, "P1").judge(rows, "METS", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["METS"])
        vol_wt = next(q for q in verdict["tost"] if q["quantity"] == "vol_wt_rel")
        if vol_wt["n_excluded"] != 0:
            self.failures.append("generated-side empty prediction must not be excluded from the volume distribution")

        # P3 cluster bootstrap: four anchor rounds per case enter as one cluster.
        rows = self._challenge_rows("SSA", [f"FIXSSA-0200-{i:03d}" for i in range(5)], "P3",
                                    mutate=lambda i, case, a, real, gen: gen.update(vol_wt_ml="52.0"))
        verdict = ChallengeJudge(envelopes, bootstrap, "P3").judge(rows, "SSA", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["SSA"])
        if verdict["verdict"] != "pass":
            self.failures.append(f"SSA P3 small shift should stay inside the wide SSA envelope, got {verdict['verdict']}")
        vol_wt = next(q for q in verdict["tost"] if q["quantity"] == "vol_wt_rel")
        if vol_wt["n_cases"] != 5:
            self.failures.append(f"P3 cluster bootstrap must resample 5 cases, got {vol_wt['n_cases']}")

        # P2 round-trip: high dice passes; collapsed dice fails; METS floor 0 is vacuous.
        judge2 = ChallengeJudge(envelopes, bootstrap, "P2")
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P2")
        verdict = judge2.judge(rows, "GLI", seed)
        rt = {item["region"]: item for item in verdict["round_trip"]}
        if verdict["verdict"] != "pass" or not all(item["passed"] for item in rt.values()):
            self.failures.append("P2 round-trip with dice ~0.9+ must pass the D_r,low floors")
        rows = self._challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P2",
                                    mutate=lambda i, case, a, real, gen: gen.update(cond_dice_et="0.10"))
        verdict = judge2.judge(rows, "GLI", seed)
        if verdict["verdict"] != "fail":
            self.failures.append("P2 collapsed ET round-trip dice must fail (floor 0.4093)")
        rows = self._challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P2",
                                    mutate=lambda i, case, a, real, gen: gen.update(cond_dice_wt="0.0"))
        verdict = judge2.judge(rows, "METS", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["METS"])
        rt = {item["region"]: item for item in verdict["round_trip"]}
        if not all(item["vacuous_pass"] and item["passed"] for item in rt.values()):
            self.failures.append("METS floor-0 round-trip must be an explicit vacuous pass")

        # Report is aggregate: no case ids anywhere in JSON or markdown.
        report = AcceptanceReport("P2", "p2-selftest", self._bootstrap_b,
                                  {"path": "/private/freeze-audit.json", "sha256": "0" * 64, "pinned": True},
                                  provisional_challenges=["METS"]).build([verdict])
        blob = json.dumps(report) + "\n".join(AcceptanceReport(
            "P2", "p2-selftest", self._bootstrap_b, report["frozen_audit"], ["METS"])._markdown(report))
        for challenge in CHALLENGES:
            if f"FIX{challenge}" in blob:
                self.failures.append(f"report leaks case ids for {challenge}")
        if report["overall_verdict"] not in ("pass", "fail", "undecided"):
            self.failures.append("overall verdict malformed")
        if "METS" not in report["provisional_challenges"]:
            self.failures.append("provisional challenge list lost in report")


# ── CLI ─────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("assemble", help="samples manifest + holdout manifest -> assembly plan JSON")
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--samples", required=True, help="frozen candidate's samples manifest (per-phase schema)")
    p.add_argument("--real-root", required=True, help="root of real BraTS data (<CH>/<case>/<case>-<mod>.nii.gz)")
    p.add_argument("--holdout-manifest", required=True, help="phase/split manifest naming holdout sides")
    p.add_argument("--run-id", default=None, help="candidate run id from the #53 run contract")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(handler="assemble")

    p = sub.add_parser("predict", help="write frozen-instrument inference scripts from a plan")
    p.add_argument("--plan", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(handler="predict")

    p = sub.add_parser("evaluate", help="measurement CSV -> per-challenge verdicts and the report")
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--table", required=True, help="measurement CSV (see nnunet_l2_final_acceptance_nifti measure)")
    p.add_argument("--freeze-audit", required=True, help="ADR-0003 §6 freeze-audit verdict JSON")
    p.add_argument("--any-verdict", action="store_true",
                   help="accept a fresh re-run verdict (all_passed checked, pinned hash not)")
    p.add_argument("--calibration-summary", default=None,
                   help="controlled calibration summary_<CH>.json directory; verified against ADR-0002 literals")
    p.add_argument("--run-id", default=None)
    p.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B,
                   help=f"bootstrap resamples (default {BOOTSTRAP_B}; selftest may lower it)")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(handler="evaluate")

    p = sub.add_parser("verify-frozen", help="verify the freeze-audit verdict alone")
    p.add_argument("--freeze-audit", required=True)
    p.add_argument("--any-verdict", action="store_true")
    p.set_defaults(handler="verify-frozen")

    p = sub.add_parser("selftest", help="fixture-driven end-to-end check (synthetic ids, stdlib only)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--bootstrap-b", type=int, default=400)
    p.set_defaults(handler="selftest")

    args = parser.parse_args(argv)
    fingerprinter = ArtifactFingerprinter()

    try:
        if args.handler == "assemble":
            strategies = {"P1": P1PseudoQuadPlan(), "P2": P2SharedMaskPlan(), "P3": P3FourAnchorPlan()}
            planner = AssemblyPlanner(
                args.phase, strategies[args.phase], RealReferenceResolver(args.real_root),
                ManifestSides.from_path(args.holdout_manifest), fingerprinter)
            plan = planner.build(args.samples, args.holdout_manifest, args.run_id, args.real_root)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_path = output_dir / "plan.json"
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            provisional = [ch for ch, info in plan["challenges"].items() if info["provisional"]]
            print(f"[OK] plan -> {plan_path} ({len(plan['observations'])} observations, "
                  f"{len(plan['challenges'])} challenges"
                  + (f", PROVISIONAL: {provisional}" if provisional else "") + ")")
            return 0
        if args.handler == "predict":
            plan = json.loads(Path(args.plan).read_text())
            if plan.get("schema") != PLAN_SCHEMA:
                raise AcceptanceError(f"plan schema {plan.get('schema')!r} != {PLAN_SCHEMA!r}")
            PredictScriptWriter(plan, args.output_dir).write()
            return 0
        if args.handler == "verify-frozen":
            record = FreezeGuard(fingerprinter).verify(args.freeze_audit, expect_sha256=None if args.any_verdict else FROZEN_AUDIT_SHA256)
            print(f"FREEZE VERIFY PASS ({record['sha256'][:16]}…, pinned={record['pinned']})")
            return 0
        if args.handler == "evaluate":
            envelopes = FrozenEnvelopes()
            if args.calibration_summary:
                envelopes.verify_against_summary(args.calibration_summary)
            freeze_record = FreezeGuard(fingerprinter).verify(
                args.freeze_audit, expect_sha256=None if args.any_verdict else FROZEN_AUDIT_SHA256)
            rows = MeasurementTable.read(args.table)
            challenges_present = sorted({row["challenge"] for row in rows})
            unknown = [ch for ch in challenges_present if ch not in CHALLENGES]
            if unknown:
                raise AcceptanceError(f"unknown challenges in measurement table: {unknown}")
            bootstrap = ClusterBootstrap(args.bootstrap_b)
            verdicts = [
                ChallengeJudge(envelopes, bootstrap, args.phase).judge(
                    [row for row in rows if row["challenge"] == challenge],
                    challenge, GLOBAL_SEED + CHALLENGE_SEED_OFFSET[challenge])
                for challenge in challenges_present
            ]
            provisional = [ch for ch in challenges_present
                           if len({row["case"] for row in rows if row["challenge"] == ch}) < HOLDOUT_QUOTAS[ch]]
            reporter = AcceptanceReport(args.phase, args.run_id, args.bootstrap_b, freeze_record, provisional)
            report = reporter.build(verdicts)
            json_path, md_path = reporter.write(report, args.output_dir)
            print(f"[OK] overall={report['overall_verdict']} -> {json_path}")
            if report["overall_verdict"] != "pass":
                return 1
            return 0
        failures = SelfTest(args.workdir, bootstrap_b=args.bootstrap_b).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0
    except AcceptanceError as error:
        print(f"ACCEPTANCE VIOLATION: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
