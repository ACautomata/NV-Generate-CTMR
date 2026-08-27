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
lives in ``measurement_run``, the NIfTI execution side of this package):

  assemble    samples manifest + holdout phase manifest -> assembly plan JSON
              (unique obs_id per observation: <case>__real / <case>__gen[__a<anchor>];
              asserts every case is holdout-side and flags provisional when the
              case count falls short of the frozen quota)
  predict     plan -> per-challenge frozen-instrument inference scripts
              (ctmr measure predict, the ADR-0009 canonical entry
              sharing nnUNetv2_predict defaults: mirror TTA on by omission --
              the flag is store_true, appending a value is an argparse fatal
              exit 2, not a TTA switch -- overlap 0.5,
              fold 0, nnUNetTrainer250Epochs; SSA uses the derived bs16 plans,
              ADR-0001)
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
    python -m ctmr.application.acceptance.distribution.final_acceptance assemble --phase P2 \
        --samples samples.json --real-root /root/private_data/raw \
        --holdout-manifest phase_manifest.json --run-id p2-... --output-dir DIR
    python -m ctmr.application.acceptance.distribution.final_acceptance predict --plan DIR/plan.json \
        --output-dir DIR
    python -m ctmr.application.acceptance.distribution.final_acceptance evaluate --phase P2 \
        --table measurements.csv --freeze-audit freeze-audit.json \
        --run-id p2-... --run runs/p2-.../run.json --output-dir DIR
    python -m ctmr.application.acceptance.distribution.final_acceptance verify-frozen --freeze-audit freeze-audit.json

``--run`` (issue #58) #140 migration binds the report to the frozen candidate of the #53 run
contract; the bound report then passes ``brats_phase_run_contract attach
--kind l2_report`` and feeds ``conclude`` (non-compensatory L1∧L2∧L3).
"""

import argparse
import csv
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from ctmr.application.acceptance.contract.artifacts import ArtifactFingerprinter, ManifestSides
from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand

PLAN_SCHEMA = "l2-final-acceptance-plan/1"
REPORT_SCHEMA = "l2-final-acceptance-report/1"
PHASES = ("P1", "P2", "P3")
CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")
REGIONS = ("WT", "TC", "ET")
REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNEL_SUFFIXES = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}

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
    "GLI": {"WT": (0.8053, 0.2802, 5.38), "TC": (0.6819, 0.4373, 4.79), "ET": (0.4093, 0.5702, 4.41), "r_fail_upper": 0.0043},
    "SSA": {"WT": (0.7046, 0.7223, 12.36), "TC": (0.6578, 0.7767, 8.64), "ET": (0.7111, 0.7786, 8.42), "r_fail_upper": 0.0838},
    "MEN": {"WT": (0.7562, 0.3235, 3.92), "TC": (0.7208, 0.3576, 6.17), "ET": (0.7501, 0.3367, 5.70), "r_fail_upper": 0.0053},
    "METS": {"WT": (0.0000, 1.6510, 28.58), "TC": (0.0000, 1.0000, 35.08), "ET": (0.0000, 1.0000, 35.08), "r_fail_upper": 0.0220},
    "PED": {"WT": (0.0093, 0.9946, 17.87), "TC": (0.0105, 0.9939, 18.33), "ET": (0.0000, 1.0000, 22.65), "r_fail_upper": 0.0507},
}

MEASUREMENT_FIELDS = [
    "obs_id",
    "challenge",
    "case",
    "side",
    "anchor",
    "input_fail",
    "run_fail",
    "hier_viol",
    "pred_empty",
    "vol_wt_ml",
    "vol_tc_ml",
    "vol_et_ml",
    "brain_ml",
    "wt_brain",
    "et_wt",
    "cx_wt_mm",
    "cy_wt_mm",
    "cz_wt_mm",
    "cx_tc_mm",
    "cy_tc_mm",
    "cz_tc_mm",
    "cx_et_mm",
    "cy_et_mm",
    "cz_et_mm",
    "cond_dice_wt",
    "cond_dice_tc",
    "cond_dice_et",
]


class AcceptanceError(Exception):
    """Raised when acceptance setup, freeze verification or judgement rules break."""


# ── frozen envelopes ────────────────────────────────────────────────────


class FrozenEnvelopes:
    """ADR-0002 envelope literals plus the load-and-verify gate (protocol §4).

    Every pass line reads its numbers from here. When a controlled calibration
    summary is supplied, each value must equal the published literal to within
    its own column's publishing grid: D_r,low and E_r,vol are frozen on the
    4-dp grid (tolerance 5e-5), while E_r,centroid is frozen on the 2-dp grid
    (tolerance 5e-3) -- ADR-0002 publishes centroid to two decimals. Drift in
    either direction -- including a narrowed margin or a raised floor --
    rejects the evaluation.
    """

    TOLERANCE = 5e-5  # 4-dp publishing grid: D_r,low, E_r,vol
    CENTROID_TOLERANCE = 5e-3  # 2-dp publishing grid: E_r,centroid

    def __init__(self):
        self._table = FROZEN_ENVELOPES

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
                    ("D_r_low", loaded["D_r_low"], self.d_r_low(challenge, region), self.TOLERANCE),
                    ("E_r_vol", loaded["E_r_vol"], self.e_r_vol(challenge, region), self.TOLERANCE),
                    ("E_r_centroid", loaded["E_r_centroid"], self.e_r_centroid(challenge, region), self.CENTROID_TOLERANCE),
                )
                for name, live, frozen, tol in pairs:
                    if live is None or math.isnan(live) or abs(live - frozen) > tol:
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
    a fresh re-run of ``freeze_audit`` whose all_passed is
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
                raise AcceptanceError(f"real reference image not found: {path} (supply real_paths for {entry['case_id']} or fix --real-root)")
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
            {
                "obs_id": f"{entry['case_id']}__real",
                "challenge": entry["challenge"],
                "case": entry["case_id"],
                "side": "real",
                "anchor": None,
                "channels": resolver.channels(entry),
                "condition_mask": None,
            },
            {
                "obs_id": f"{entry['case_id']}__gen",
                "challenge": entry["challenge"],
                "case": entry["case_id"],
                "side": "gen",
                "anchor": None,
                "channels": channels,
                "condition_mask": None,
            },
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
            {
                "obs_id": f"{entry['case_id']}__real",
                "challenge": entry["challenge"],
                "case": entry["case_id"],
                "side": "real",
                "anchor": None,
                "channels": resolver.channels(entry),
                "condition_mask": None,
            },
            {
                "obs_id": f"{entry['case_id']}__gen",
                "challenge": entry["challenge"],
                "case": entry["case_id"],
                "side": "gen",
                "anchor": None,
                "channels": channels,
                "condition_mask": condition_mask,
            },
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
        observations = [
            {
                "obs_id": f"{entry['case_id']}__real",
                "challenge": entry["challenge"],
                "case": entry["case_id"],
                "side": "real",
                "anchor": None,
                "channels": real_channels,
                "condition_mask": None,
            }
        ]
        for anchor in MODALITIES:
            round_info = anchors[anchor]
            if not round_info.get("real") or not round_info.get("generated"):
                raise AcceptanceError(f"{entry['case_id']}: anchor {anchor} needs real and generated")
            targets = [m for m in MODALITIES if m != anchor]
            if set(round_info["generated"]) != set(targets):
                raise AcceptanceError(f"{entry['case_id']}: anchor {anchor} must generate exactly {targets}, got {sorted(round_info['generated'])}")
            channels = dict(real_channels)
            channels[CHANNEL_SUFFIXES[anchor]] = round_info["real"]
            for target in targets:
                if not round_info["generated"][target].get("path"):
                    raise AcceptanceError(f"{entry['case_id']}: anchor {anchor} -> {target} needs a path")
                channels[CHANNEL_SUFFIXES[target]] = round_info["generated"][target]["path"]
            observations.append(
                {
                    "obs_id": f"{entry['case_id']}__gen__a{anchor}",
                    "challenge": entry["challenge"],
                    "case": entry["case_id"],
                    "side": "gen",
                    "anchor": anchor,
                    "channels": channels,
                    "condition_mask": None,
                }
            )
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
                raise AcceptanceError(f"entry {entry.get('case_id')} declares phase {entry.get('phase')!r}; this assembly is {self._phase}")
            side = self._holdout_sides.side_of(entry["challenge"], entry["case_id"])
            if side is None:
                raise AcceptanceError(f"({entry['challenge']}, {entry['case_id']}) is not in the holdout manifest")
            if side != "holdout":
                raise AcceptanceError(
                    f"({entry['challenge']}, {entry['case_id']}) is {side}-side; final acceptance runs on the 20% holdout side only (spec decision 3)"
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

    The command line is exactly ``FrozenInstrumentCommand.build`` (ADR-0009
    decision 1): the canonical entry ``ctmr measure predict``, the
    per-challenge frozen spec (SSA uses the ADR-0001 derived batch-16 plans and
    configuration), fold 0, ``nnUNetTrainer250Epochs``. Mirror TTA stays ON by
    omission (--disable_tta is store_true; appending a value, ``False``
    included, is an argparse fatal exit 2, not a TTA switch). The generated
    shell also puts this module's src tree on
    PYTHONPATH, so the canonical entry is importable from the fresh shell the
    script runs in (repo and flat-deployment spellings, the ADR-0009 decision 6
    shim).
    """

    def __init__(self, plan, output_dir):
        self._plan = plan
        self._output_dir = Path(output_dir)

    @staticmethod
    def _pythonpath_export():
        """The generated scripts run the canonical verb in a fresh shell; pin this
        checkout's ``src`` tree onto PYTHONPATH (repo and flat-deployment spellings
        collapse to the installed package root since #140)."""
        package_src = Path(__file__).resolve().parents[4]
        return f'export PYTHONPATH="{package_src}${{PYTHONPATH:+:$PYTHONPATH}}"\n'

    def write(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        input_root = self._output_dir / "inputs"
        scripts = []
        for challenge in sorted(self._plan["challenges"]):
            cmd = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(input_root / challenge, self._output_dir / "predictions" / challenge)
            script_path = self._output_dir / f"predict_{challenge}.sh"
            script_path.write_text("#!/bin/bash\nset -euo pipefail\n" + self._pythonpath_export() + " ".join(cmd) + "\n")
            script_path.chmod(0o755)
            scripts.append(script_path)
        runner = self._output_dir / "predict_all.sh"
        runner.write_text("#!/bin/bash\nset -euo pipefail\n" + "".join(f"bash {script.name}\n" for script in scripts))
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
        vol_margin = {region: (lambda envelopes, challenge, region=region: envelopes.e_r_vol(challenge, region)) for region in REGIONS}
        cent_margin = {region: (lambda envelopes, challenge, region=region: envelopes.e_r_centroid(challenge, region)) for region in REGIONS}
        self._quantities = []
        for region in REGIONS:
            vol_field = {"WT": "vol_wt_ml", "TC": "vol_tc_ml", "ET": "vol_et_ml"}[region]
            self._quantities.append(
                QuantityFamily(
                    f"vol_{region.lower()}_rel",
                    vol_field,
                    vol_margin[region],
                    relative=True,
                    exclusion=lambda real_row, gen_row, vol_field=vol_field: (
                        None if (MeasurementTable.number(real_row, vol_field) or 0) > 0 else "real_volume_zero"
                    ),
                )
            )
            for axis in ("x", "y", "z"):
                centroid_field = f"c{axis}_{region.lower()}_mm"
                self._quantities.append(
                    QuantityFamily(
                        f"centroid_{region.lower()}_{axis}",
                        centroid_field,
                        cent_margin[region],
                        relative=False,
                        exclusion=lambda real_row, gen_row, vol_field=vol_field: (
                            None
                            if (MeasurementTable.number(real_row, vol_field) or 0) > 0 and (MeasurementTable.number(gen_row, vol_field) or 0) > 0
                            else "empty_mask_side"
                        ),
                    )
                )
        self._quantities.append(
            QuantityFamily(
                "wt_brain_rel",
                "wt_brain",
                vol_margin["WT"],
                relative=True,
                exclusion=lambda real_row, gen_row: None if (MeasurementTable.number(real_row, "wt_brain") or 0) > 0 else "real_ratio_zero",
            )
        )
        self._quantities.append(
            QuantityFamily(
                "et_wt_rel",
                "et_wt",
                lambda envelopes, challenge: envelopes.e_r_vol(challenge, "ET") + envelopes.e_r_vol(challenge, "WT"),
                relative=True,
                exclusion=lambda real_row, gen_row: None if (MeasurementTable.number(real_row, "et_wt") or 0) > 0 else "real_ratio_zero",
            )
        )

    def all(self):
        return list(self._quantities)


# ── judgement chain ─────────────────────────────────────────────────────


class FailureGate:
    """The undecided gate: any input/run/hierarchy failure on either side (ADR-0004 decision 4).

    Counts only: the aggregate report carries no obs_id (and therefore no subject
    id, protocol §5); per-observation detail lives in the controlled CSV.
    """

    @staticmethod
    def audit(rows):
        breakdown = {"input_fail": 0, "run_fail": 0, "hier_viol": 0}
        by_side = {"gen": 0, "real": 0}
        n_failed = 0
        for row in rows:
            failures = [name for name in breakdown if MeasurementTable.flag(row, name)]
            if failures:
                n_failed += 1
                by_side[row["side"]] = by_side.get(row["side"], 0) + 1
                for name in failures:
                    breakdown[name] += 1
        n_obs = len(rows)
        return {
            "n_failed": n_failed,
            "breakdown": breakdown,
            "n_failed_by_side": by_side,
            # Observed-side Wilson 95% upper, same formula as calibration (ADR-0002):
            # diagnostic only -- any single failure already forces undecided.
            "wilson_95_upper": FailureGate.wilson_upper(n_failed, n_obs) if n_obs else None,
        }

    Z95 = 1.959963984540054

    @staticmethod
    def wilson_upper(k, n):
        p = k / n
        denom = 1 + FailureGate.Z95**2 / n
        center = (p + FailureGate.Z95**2 / (2 * n)) / denom
        half = (FailureGate.Z95 / denom) * math.sqrt(p * (1 - p) / n + FailureGate.Z95**2 / (4 * n**2))
        return min(1.0, center + half)


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
            results.append(
                {
                    "quantity": quantity.name,
                    "margin": margin,
                    "ci90_low": ci["low"],
                    "ci90_high": ci["high"],
                    "n_cases": ci["n_cases"],
                    "n_excluded": sum(exclusions.values()),
                    "exclusion_reasons": exclusions,
                    "passed": passed,
                }
            )
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
            n_excluded = 0
            for case, gen_rows in sorted(gen_by_case.items()):
                for gen_row in gen_rows:
                    value = MeasurementTable.number(gen_row, field)
                    if value is None:  # undefined dice (both masks empty): excluded and counted, never silent
                        n_excluded += 1
                        per_case.append([])
                    else:
                        per_case.append([value])
            stats = self._bootstrap.q5_lower_bound(per_case, seed=seed_base + 100 + region_index)
            vacuous = floor == 0.0
            if stats is None:
                passed, bound = False, None
            else:
                bound = stats["bound"]
                passed = vacuous or bound >= floor - 1e-12
            results.append(
                {
                    "region": region,
                    "floor": floor,
                    "bound": bound,
                    "n_cases": stats["n_cases"] if stats else 0,
                    "n_excluded": n_excluded,
                    "vacuous_pass": vacuous,
                    "passed": passed,
                }
            )
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

    def __init__(self, phase, run_id, bootstrap_b, freeze_record, provisional_challenges, binding=None):
        self._phase = phase
        self._run_id = run_id
        self._bootstrap_b = bootstrap_b
        self._freeze_record = freeze_record
        self._provisional = provisional_challenges
        self._binding = binding

    def build(self, challenge_verdicts, challenges_missing):
        overall = (
            "undecided"
            if any(v["verdict"] == "undecided" for v in challenge_verdicts)
            else ("pass" if all(v["verdict"] == "pass" for v in challenge_verdicts) else "fail")
        )
        report = {
            "schema": REPORT_SCHEMA,
            "title": "L2 冻结仪器最终验收报告",
            "issue": 55,
            "phase": self._phase,
            "run_id": self._run_id,
            "binding": self._binding.as_dict() if self._binding is not None else None,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bootstrap": {
                "B": self._bootstrap_b,
                "method": "病例级 cluster bootstrap, percentile CI90, linear 插值 (protocol §4)",
                "global_seed": GLOBAL_SEED,
                "per_challenge_offset": CHALLENGE_SEED_OFFSET,
            },
            "frozen_audit": self._freeze_record,
            "envelopes_source": "ADR-0002 literals (docs/adr/0002-l2-instrument-calibration-envelopes.md)",
            "pass_lines": "ADR-0004 (docs/adr/0004-l2-final-acceptance-pass-lines.md)",
            "provisional_challenges": self._provisional,
            "challenges_missing": challenges_missing,
            "complete_coverage": not challenges_missing and not self._provisional,
            "z_crop_bias_note": (
                "生成侧 1mm 重采样后为 241×241×174, 居中裁到 240×240×155 砍掉 ~19 个 z 层; 真实侧原生 155 层"
                "不裁剪 — WT/brain 分母与质心 z 向存在系统性偏差轴(含 DM 训练数据 z 向 pad 对齐方式未单独验证), "
                "如实注册于协议 §2, 不补偿"
            ),
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

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, report):
        lines = [
            f"# {report['title']}",
            "",
            f"**Issue**: [#55](https://github.com/ACautomata/NV-Generate-CTMR/issues/55) · "
            f"**Phase**: {report['phase']} · **Run**: `{report['run_id']}`",
            f"**总体判定**: **{report['overall_verdict'].upper()}**"
            + (" (provisional: " + ", ".join(sorted(self._provisional)) + " 观测不足冻结配额)" if self._provisional else "")
            + (
                " (缺挑战: " + ", ".join(report["challenges_missing"]) + " — 仅子集 AND, 不构成完整 spec 终验)"
                if report["challenges_missing"]
                else ""
            ),
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
                        f"- {challenge}/{item['quantity']}: CI90 [{self._fmt(item['ci90_low'])}, {self._fmt(item['ci90_high'])}] "
                        f"⊄ ±{self._fmt(item['margin'])} (排除 {item['n_excluded']})"
                    )
        if report["phase"] == "P2":
            lines += ["", "## P2 条件回切", ""]
            for challenge, verdict in report["per_challenge"].items():
                for item in verdict["round_trip"] or []:
                    tag = " (vacuous-pass, floor=0)" if item["vacuous_pass"] else ""
                    lines.append(
                        f"- {challenge}/{item['region']}: q5 下界 {self._fmt(item['bound'])} vs floor "
                        f"{self._fmt(item['floor'])} (排除 {item['n_excluded']}){tag}"
                    )
        lines += [
            "",
            "## 冻结与合规",
            "",
            f"- 冻结审计 verdict: `{report['frozen_audit']['sha256'][:16]}…` (pinned={report['frozen_audit']['pinned']})",
            "- 仪器权重/plans/推理配置/校准包络未做任何修改; 逐病例测量(含 subject ID)只在受控存储。",
            "- " + report["z_crop_bias_note"],
            "- METS/PED 宽包络按 ADR-0002 原样适用, 未收窄。" + report.get("mets_resolving_power_note", ""),
            "",
        ]
        return "\n".join(lines)


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
    p.add_argument("--table", required=True, help="measurement CSV (see measurement_run measure)")
    p.add_argument("--freeze-audit", required=True, help="ADR-0003 §6 freeze-audit verdict JSON")
    p.add_argument("--any-verdict", action="store_true", help="accept a fresh re-run verdict (all_passed checked, pinned hash not)")
    p.add_argument(
        "--calibration-summary", default=None, help="controlled calibration summary_<CH>.json directory; verified against ADR-0002 literals"
    )
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--run",
        default=None,
        help="run.json of the frozen candidate (#53 contract); binds the report for attach --kind l2_report (issue #58) #140 migration",
    )
    p.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B}; selftest may lower it)")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(handler="evaluate")

    p = sub.add_parser("verify-frozen", help="verify the freeze-audit verdict alone")
    p.add_argument("--freeze-audit", required=True)
    p.add_argument("--any-verdict", action="store_true")
    p.set_defaults(handler="verify-frozen")

    args = parser.parse_args(argv)
    fingerprinter = ArtifactFingerprinter()

    try:
        if args.handler == "assemble":
            strategies = {"P1": P1PseudoQuadPlan(), "P2": P2SharedMaskPlan(), "P3": P3FourAnchorPlan()}
            planner = AssemblyPlanner(
                args.phase,
                strategies[args.phase],
                RealReferenceResolver(args.real_root),
                ManifestSides.from_path(args.holdout_manifest),
                fingerprinter,
            )
            plan = planner.build(args.samples, args.holdout_manifest, args.run_id, args.real_root)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plan_path = output_dir / "plan.json"
            plan_path.write_text(json.dumps(plan, indent=2) + "\n")
            provisional = [ch for ch, info in plan["challenges"].items() if info["provisional"]]
            print(
                f"[OK] plan -> {plan_path} ({len(plan['observations'])} observations, "
                f"{len(plan['challenges'])} challenges" + (f", PROVISIONAL: {provisional}" if provisional else "") + ")"
            )
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
            freeze_record = FreezeGuard(fingerprinter).verify(args.freeze_audit, expect_sha256=None if args.any_verdict else FROZEN_AUDIT_SHA256)
            rows = MeasurementTable.read(args.table)
            challenges_present = sorted({row["challenge"] for row in rows})
            unknown = [ch for ch in challenges_present if ch not in CHALLENGES]
            if unknown:
                raise AcceptanceError(f"unknown challenges in measurement table: {unknown}")
            bootstrap = ClusterBootstrap(args.bootstrap_b)
            verdicts = [
                ChallengeJudge(envelopes, bootstrap, args.phase).judge(
                    [row for row in rows if row["challenge"] == challenge], challenge, GLOBAL_SEED + CHALLENGE_SEED_OFFSET[challenge]
                )
                for challenge in challenges_present
            ]
            # Provisional is judged against ALL five frozen quotas (an absent
            # challenge is a shortfall too): the overall AND must never read as
            # full-spec acceptance over a subset (spec Further Notes).
            provisional = [ch for ch in CHALLENGES if len({row["case"] for row in rows if row["challenge"] == ch}) < HOLDOUT_QUOTAS[ch]]
            challenges_missing = [ch for ch in CHALLENGES if ch not in challenges_present]
            binding = None
            if args.run:
                try:
                    binding = FrozenRunBinding.from_path(args.run)
                except FrozenRunBindingError as error:
                    raise AcceptanceError(str(error)) from error
                bound = binding.as_dict()
                if args.run_id and args.run_id != bound["run_id"]:
                    raise AcceptanceError(f"--run-id {args.run_id} contradicts the run record {bound['run_id']}")
                if bound["phase"] != args.phase:
                    raise AcceptanceError(f"run record phase {bound['phase']!r} != --phase {args.phase!r}; the L2 protocol differs per phase")
                if not args.run_id:
                    args.run_id = bound["run_id"]
            reporter = AcceptanceReport(args.phase, args.run_id, args.bootstrap_b, freeze_record, provisional, binding=binding)
            report = reporter.build(verdicts, challenges_missing)
            json_path, md_path = reporter.write(report, args.output_dir)
            print(f"[OK] overall={report['overall_verdict']} -> {json_path}")
            if report["overall_verdict"] != "pass":
                return 1
            return 0
        raise ValueError(f"unhandled verb: {args.handler}")  # pragma: no cover - argparse constrains choices
    except AcceptanceError as error:
        print(f"ACCEPTANCE VIOLATION: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
