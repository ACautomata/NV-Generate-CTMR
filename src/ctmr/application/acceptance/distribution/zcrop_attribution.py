"""Z-crop compensation attribution (issue #206, parent #205 diagnostic job A).

Re-measures WT relative volume and centroid z after restricting BOTH sides to
the overlapping z range of the 19-slice crop, then attributes the L2
final-acceptance FAIL to the measurement axis vs the candidate defect.

``variant=diagnostic``: this module produces diagnostic readings only -- it
issues NO acceptance verdict of any kind, registers no ``ctmr accept``
surface, and never touches the frozen instruments or envelopes (ADR-0002/0004).

Geometry (protocol-registered z-axis fact, resampled 241x241x174
centre-cropped to 155 slices on the generated side only): both instrument
mask arrays are 155 slices, but their z coordinates differ by the crop start
``(174-155)//2 = 9`` -- the same formula as ``ctmr.domain.grid``
CenterCropOrPad. The generated side's slice i sits at physical z i+9, the
real side's slice i at physical z i. Restricting both sides to the overlap
therefore means gen [0,146) and real [9,155): two 146-slice windows mapping
onto the same physical range [9,155) mm via local+9 on BOTH sides.

Coordinate semantics (the smoke-run finding): the measurement CSV's
``cz_wt_mm`` is an instrument-array voxel index on each side -- the two
sides' absolute values live on different scales and are NOT comparable; the
comparable per-case quantity is the diff (``{quantity}_before/after``,
gen-real, the same direction as the TOST). ``before`` replays the final
acceptance on the CSV values (reconciled against both the full-array
remeasurement and the frozen acceptance JSON); ``after`` re-judges on the
compensated overlap re-measurement.

Reconciliation guards (any failure is a FATAL exit, never a silent pass):
  1. full-array re-measurement reproduces the measurement CSV within the
     1e-9 tolerance (scipy ``center_of_mass`` vs numpy accumulation order);
  2. the replayed before-CI reproduces the final-acceptance JSON CI within
     the same tolerance (same ClusterBootstrap, same B, same registry-order
     seed stream: vol_wt_rel=0, centroid_wt_z=3);
  3. replayed exclusion counts match the JSON.

Usage (sugon, via deploy/jobs/run_zcrop_attribution.sh)::

  python -m ctmr.application.acceptance.distribution.zcrop_attribution \\
      --plan DIR/plan.json --table DIR/measurements.csv --preds DIR/predictions \\
      --report DIR/l2_final_acceptance_p1.json --output-dir DIR/zcrop_attribution
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.final_acceptance import (
    CHALLENGE_SEED_OFFSET,
    GLOBAL_SEED,
    ClusterBootstrap,
    FrozenEnvelopes,
    MeasurementTable,
    QuantityRegistry,
)
from ctmr.application.acceptance.distribution.measurement_run import PREDICTION_SHAPE

REPORT_SCHEMA = "zcrop-attribution-diagnostic/1"
REPORT_TITLE = "z-crop 补偿重算归因报告(诊断)"
REPORT_ISSUE = 206
VARIANT = "diagnostic"
DISCLAIMER = "诊断读数(variant=diagnostic),不产生任何 L1/L2/L3 验收判定;冻结仪器、包络与判定线未做任何修改。"

# Instrument z geometry: resampled 174 -> centre-cropped 155, start 9 (grid.py formula).
RESAMPLED_Z = 174  # round(128 x 1.36)
TARGET_Z = 155
CROP_START = (RESAMPLED_Z - TARGET_Z) // 2  # = 9, the domain/grid CenterCropOrPad formula
OVERLAP_SLICES = TARGET_Z - CROP_START  # = 146

# Pre-registered TOST quantities of job A and their QuantityRegistry order
# (the judge enumerates registry.all(), so the CI seed stream is GLOBAL_SEED +
# CHALLENGE_SEED_OFFSET[challenge] + this index).
QUANTITY_TOST_INDEX = {"vol_wt_rel": 0, "centroid_wt_z": 3}

WT_LABELS = (1, 2, 3)  # REGION_LABELS["WT"]
RECONCILIATION_TOLERANCE = 1e-9  # scipy vs numpy accumulation order
MARGIN_TOLERANCE = 1e-12  # the final-acceptance pass-line epsilon

COORDINATE_NOTE = (
    "before 的 cz_wt_mm 为仪器数组 z 体素索引(measurement_run 原生口径,对账基准),"
    "after 为重叠域物理 mm(生成侧仪器数组索引 i 对应物理 i+9);"
    "绝对值跨口径不可比,逐 case 可比量为 {quantity}_before/after 差值(与 TOST 同向:gen−real)。"
)

OUTCOME_REPAIRED = "measurement_axis_repaired"
OUTCOME_PERSISTS = "defect_persists"
OUTCOME_NO_FAILURE = "no_failure_to_attribute"
VERDICT_MEASUREMENT = "measurement_axis_dominant"
VERDICT_CANDIDATE = "candidate_defect_dominant"
VERDICT_MIXED = "mixed"
VERDICT_NOTHING = "no_failure_to_attribute"


class AttributionError(Exception):
    """Raised when job-A inputs, reconciliation guards or attribution rules break."""


# ── geometry ────────────────────────────────────────────────────────────


class OverlapGeometry:
    """The two overlap windows and their shared physical mapping.

    Both windows are 146 slices; both map local slice -> physical z via +9,
    so the compensated readings of the two sides live on one scale.
    """

    @staticmethod
    def gen_window():
        """Generated side: instrument slices [0,146), physical [9,155) mm."""
        return slice(0, OVERLAP_SLICES)

    @staticmethod
    def real_window():
        """Real side: instrument slices [9,155), physical [9,155) mm."""
        return slice(CROP_START, TARGET_Z)

    @staticmethod
    def side_window(side):
        if side == "gen":
            return OverlapGeometry.gen_window()
        if side == "real":
            return OverlapGeometry.real_window()
        raise ValueError(f"unknown side {side!r}; expected 'gen' or 'real'")

    @staticmethod
    def overlap_z_mm(local_z):
        """Physical mm of a window-local slice index (+9 on BOTH sides)."""
        return local_z + CROP_START

    @staticmethod
    def describe():
        return {
            "resampled_z": RESAMPLED_Z,
            "target_z": TARGET_Z,
            "crop_start": CROP_START,
            "gen_window": [0, OVERLAP_SLICES],
            "real_window": [CROP_START, TARGET_Z],
            "overlap_slices": OVERLAP_SLICES,
            "overlap_phys_mm": [CROP_START, TARGET_Z],
            "offset_mm": CROP_START,
        }


# ── re-measurement ──────────────────────────────────────────────────────


class OverlapRemeasurer:
    """WT readings on the two scales: full array (instrument index, the CSV
    reconciliation baseline) and overlap window (physical mm)."""

    @staticmethod
    def measure(pred, side):
        """Measures one instrument-array prediction for ``side``'s window.

        Volumes in ml (1 mm isotropic voxels, the MaskMeasurer convention);
        ``cz_index`` is the full-array centroid (instrument voxel index),
        ``cz_overlap_mm`` the window centroid mapped to physical mm; both are
        None on an empty WT. The windowed centroid uses ``nonzero().mean()``,
        matching scipy ``center_of_mass`` up to the 1e-9 accumulation-order
        tolerance the reconciliation guard allows.
        """
        wt = np.isin(pred, WT_LABELS)
        window = OverlapGeometry.side_window(side)
        wt_overlap = wt[window]
        overlap_centroid = OverlapRemeasurer._mean_z(wt_overlap)
        return {
            "vol_ml": float(wt.sum()) * 0.001,
            "cz_index": OverlapRemeasurer._mean_z(wt),
            "vol_ml_overlap": float(wt_overlap.sum()) * 0.001,
            "cz_overlap_mm": None if overlap_centroid is None else OverlapGeometry.overlap_z_mm(overlap_centroid),
        }

    @staticmethod
    def _mean_z(wt):
        indices = np.nonzero(wt)[0]
        return None if indices.size == 0 else float(indices.mean())


# ── before replay + compensated judgement ───────────────────────────────


class CompensationJudge:
    """Replays the final-acceptance TOST for job A's two quantities, then
    judges the compensated (overlap) diffs with the identical method.

    Before-side diffs come from the measurement CSV through the frozen
    QuantityFamily (pairing order: sorted cases, plan-order gen rows, the
    exact enumeration the judge used); the baseline JSON is reconciled
    against the replay. After-side diffs come from the overlap pseudo-rows
    through the same QuantityFamily, so exclusion semantics cannot drift.
    """

    def __init__(self, acceptance_report):
        self._report = acceptance_report
        baseline_b = acceptance_report.get("bootstrap", {}).get("B")
        if not baseline_b:
            raise AttributionError("final-acceptance JSON carries no bootstrap B; the replay cannot mirror the judge")
        self._bootstrap = ClusterBootstrap(baseline_b)

    def judge(self, challenge, quantity_name, real_by_case, gen_by_case, after_real_by_case, after_gen_by_case, verify_baseline=True):
        registry_index = QUANTITY_TOST_INDEX[quantity_name]
        family = QuantityRegistry().all()[registry_index]
        margin = family.margin(FrozenEnvelopes(), challenge)
        seed = GLOBAL_SEED + CHALLENGE_SEED_OFFSET[challenge] + registry_index

        before_diffs, before_exclusions, before_flat = self._pair_all(family, real_by_case, gen_by_case)
        before_ci = self._bootstrap.ci90(before_diffs, seed=seed)
        if verify_baseline:
            self._verify_baseline(challenge, quantity_name, before_ci, before_exclusions)

        after_diffs, _, after_flat = self._pair_all(family, after_real_by_case, after_gen_by_case)
        after_ci = self._bootstrap.ci90(after_diffs, seed=seed)

        per_obs = {}
        for obs_id, diff in before_flat.items():
            per_obs.setdefault(obs_id, {})["before"] = diff
        for obs_id, diff in after_flat.items():
            per_obs.setdefault(obs_id, {})["after"] = diff
        return {
            "challenge": challenge,
            "quantity": quantity_name,
            "margin": margin,
            "ci90_before": before_ci if before_ci is not None else {"low": None, "high": None, "n_cases": 0},
            "before_passed": CompensationJudge._within_margin(before_ci, margin),
            "ci90_after": after_ci if after_ci is not None else {"low": None, "high": None, "n_cases": 0},
            "after_passed": CompensationJudge._within_margin(after_ci, margin),
            "n_excluded": sum(before_exclusions.values()),
            "exclusion_reasons": before_exclusions,
            "per_obs_diff": per_obs,
        }

    @staticmethod
    def _pair_all(family, real_by_case, gen_by_case):
        """The judge's exact pairing: sorted cases, plan-order gen rows.

        Deliberately a shape-copy of ``ChallengeJudge.tost``'s enumeration
        (final_acceptance.py): the bootstrap seed stream consumes the cluster
        order, so any drift here breaks the before-CI reconciliation guard.
        If the frozen judge's pairing ever changes, this must change with it.
        """
        diffs, exclusions, flat = [], {}, {}
        for case in sorted(gen_by_case):
            case_diffs = []
            for gen_row in gen_by_case[case]:
                diff, reason = family.pair(gen_row, real_by_case[case])
                flat[gen_row["obs_id"]] = diff
                if diff is None:
                    exclusions[reason] = exclusions.get(reason, 0) + 1
                else:
                    case_diffs.append(diff)
            diffs.append(case_diffs)
        return diffs, exclusions, flat

    def _verify_baseline(self, challenge, quantity_name, before_ci, exclusions):
        verdict = self._report.get("per_challenge", {}).get(challenge)
        tost = (verdict or {}).get("tost")
        if tost is None:
            raise AttributionError(
                f"{challenge}: final-acceptance JSON carries no TOST section (undecided report?) -- "
                "job A attributes TOST failures and cannot run without that baseline"
            )
        baseline = next((item for item in tost if item["quantity"] == quantity_name), None)
        if baseline is None:
            raise AttributionError(f"{challenge}: final-acceptance JSON TOST has no {quantity_name} entry")
        replayed = before_ci if before_ci is not None else {"low": None, "high": None}
        for key in ("ci90_low", "ci90_high"):
            if not CompensationJudge._close(replayed[key.replace("ci90_", "")], baseline.get(key)):
                raise AttributionError(
                    f"{challenge}/{quantity_name}: replayed before {key} {replayed[key.replace('ci90_', '')]} != "
                    f"final-acceptance JSON {baseline.get(key)} (tolerance {RECONCILIATION_TOLERANCE:g}); "
                    "the replay does not reproduce the frozen judge, refusing to attribute"
                )
        if sum(exclusions.values()) != baseline["n_excluded"] or exclusions != {
            key: value for key, value in baseline.get("exclusion_reasons", {}).items()
        }:
            raise AttributionError(
                f"{challenge}/{quantity_name}: replayed exclusions {exclusions} != final-acceptance JSON "
                f"(n_excluded={baseline['n_excluded']}, reasons={baseline.get('exclusion_reasons')})"
            )

    @staticmethod
    def _close(left, right):
        if left is None or right is None:
            return left is None and right is None
        return abs(left - right) <= RECONCILIATION_TOLERANCE

    @staticmethod
    def _within_margin(ci, margin):
        """The final-acceptance pass line (protocol §4), verbatim from
        ``ChallengeJudge.tost`` -- keep in sync with the frozen judge."""
        if ci is None:
            return False
        return ci["low"] >= -margin - MARGIN_TOLERANCE and ci["high"] <= margin + MARGIN_TOLERANCE


# ── attribution ─────────────────────────────────────────────────────────


class CompensationAttributor:
    """Three-way attribution over the before-FAIL quantities, with shares.

    A before-FAIL quantity whose compensated CI passes is explained by the
    measurement axis; one that still fails points at the candidate defect.
    Quantities that passed before had no failure to attribute (boundary
    state, reported as such). The base is the number of before-FAIL
    quantities (2 per challenge); per-axis sub-attributions decompose the
    same items per quantity family.
    """

    def attribute(self, entries):
        items = []
        for entry in entries:
            if entry["before_passed"]:
                outcome = OUTCOME_NO_FAILURE
            elif entry["after_passed"]:
                outcome = OUTCOME_REPAIRED
            else:
                outcome = OUTCOME_PERSISTS
            items.append(
                {
                    "challenge": entry["challenge"],
                    "quantity": entry["quantity"],
                    "before_passed": entry["before_passed"],
                    "after_passed": entry["after_passed"],
                    "outcome": outcome,
                }
            )
        base = [item for item in items if item["outcome"] != OUTCOME_NO_FAILURE]
        repaired = sum(item["outcome"] == OUTCOME_REPAIRED for item in base)
        persistent = sum(item["outcome"] == OUTCOME_PERSISTS for item in base)
        attribution = {
            "verdict": CompensationAttributor._verdict(repaired, persistent, len(base)),
            "base": len(base),
            "repaired": repaired,
            "persistent": persistent,
            "shares": {
                "measurement_axis": repaired / len(base) if base else 0.0,
                "candidate_defect": persistent / len(base) if base else 0.0,
            },
            "per_axis": {},
            "items": items,
        }
        for quantity in QUANTITY_TOST_INDEX:
            axis_items = [item for item in base if item["quantity"] == quantity]
            axis_repaired = sum(item["outcome"] == OUTCOME_REPAIRED for item in axis_items)
            axis_persistent = sum(item["outcome"] == OUTCOME_PERSISTS for item in axis_items)
            attribution["per_axis"][quantity] = {
                "verdict": CompensationAttributor._verdict(axis_repaired, axis_persistent, len(axis_items)),
                "base": len(axis_items),
                "repaired": axis_repaired,
                "persistent": axis_persistent,
            }
        return attribution

    @staticmethod
    def _verdict(repaired, persistent, base):
        if base == 0:
            return VERDICT_NOTHING
        if repaired == base:
            return VERDICT_MEASUREMENT
        if persistent == base:
            return VERDICT_CANDIDATE
        return VERDICT_MIXED


# ── report ──────────────────────────────────────────────────────────────


class AttributionReport:
    """Diagnostic JSON + markdown report; no acceptance verdict anywhere."""

    def __init__(self, run_id, bootstrap_b):
        self._run_id = run_id
        self._bootstrap_b = bootstrap_b

    def build(self, entries, per_case_rows, attribution, reconciliation):
        return {
            "schema": REPORT_SCHEMA,
            "title": REPORT_TITLE,
            "issue": REPORT_ISSUE,
            "variant": VARIANT,
            "disclaimer": DISCLAIMER,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": self._run_id,
            "bootstrap": {"B": self._bootstrap_b, "seed_stream": "GLOBAL_SEED + CHALLENGE_SEED_OFFSET + QuantityRegistry order"},
            "geometry": OverlapGeometry.describe(),
            "coordinate_note": COORDINATE_NOTE,
            "per_quantity": entries,
            "per_case": per_case_rows,
            "attribution": attribution,
            "reconciliation": reconciliation,
        }

    def write(self, report, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "zcrop_attribution_report.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / "zcrop_attribution_report.md"
        md_path.write_text(self.markdown(report))
        return json_path, md_path

    def markdown(self, report):
        attribution = report["attribution"]
        shares = attribution["shares"]
        lines = [
            f"# {report['title']}",
            "",
            f"**Issue**: [#{report['issue']}](https://github.com/ACautomata/NV-Generate-CTMR/issues/{report['issue']})"
            f" · **variant**: {report['variant']} · **Run**: `{report['run_id']}`",
            f"**声明**: {report['disclaimer']}",
            "",
            f"**归因结论**: **{attribution['verdict']}** — "
            f"测量轴 {attribution['repaired']}/{attribution['base']}"
            f"({AttributionReport._format_share(shares['measurement_axis'])}),"
            f"候选缺陷 {attribution['persistent']}/{attribution['base']}"
            f"({AttributionReport._format_share(shares['candidate_defect'])})",
            "",
            "## 重叠几何",
            "",
            f"- 仪器数组 z:{report['geometry']['resampled_z']} 层重采样 → {report['geometry']['target_z']} 层,"
            f"crop start={report['geometry']['crop_start']}(domain/grid CenterCropOrPad 同式)",
            f"- 重叠窗口:生成侧 {report['geometry']['gen_window']}、真实侧 {report['geometry']['real_window']},"
            f"各 {report['geometry']['overlap_slices']} 层,同一物理域 [9,155) mm(两侧同式 +9)",
            f"- {report['coordinate_note']}",
            "",
            "## 逐量判定(before → after)",
            "",
            "| 挑战 | 量 | margin | before CI90 | before | after CI90 | after |",
            "|---|---|---:|---|---|---|---|",
        ]
        for entry in report["per_quantity"]:
            lines.append(
                f"| {entry['challenge']} | {entry['quantity']} | ±{AttributionReport._fmt(entry['margin'])} "
                f"| [{AttributionReport._fmt(entry['ci90_before']['low'])}, {AttributionReport._fmt(entry['ci90_before']['high'])}] "
                f"| {'pass' if entry['before_passed'] else 'FAIL'} "
                f"| [{AttributionReport._fmt(entry['ci90_after']['low'])}, {AttributionReport._fmt(entry['ci90_after']['high'])}] "
                f"| {'pass' if entry['after_passed'] else 'FAIL'} |"
            )
        lines += ["", "## 归因明细", ""]
        for item in attribution["items"]:
            lines.append(
                f"- {item['challenge']}/{item['quantity']}: "
                f"{'PASS' if item['before_passed'] else 'FAIL'} → {'PASS' if item['after_passed'] else 'FAIL'}"
                f"({item['outcome']})"
            )
        lines += ["", "## 对账", ""]
        reconciliation = report["reconciliation"]
        lines += [
            f"- 全域重算 vs 测量表:容差(1e-9)内复现,n={reconciliation['n_reconciled']}" f"(无测量跳过 {reconciliation['n_skipped_no_measurement']})",
            "- 终验 TOST 重放:before CI 容差(1e-9)内复现终验 json,排除计数一致,seed 流=QuantityRegistry 注册序",
            "- 任一 guard 失败即 FATAL 退出,不存在静默降级。",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_share(share):
        """Renders a share as Chinese tenths ("7 成" for 0.7)."""
        return f"{round(share * 10, 1):g} 成"

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"


# ── job orchestration ───────────────────────────────────────────────────


class ZcropAttributionJob:
    """Assembles plan + measurement CSV + predictions + the frozen
    final-acceptance JSON into the diagnostic attribution report."""

    def __init__(self, plan_path, table_path, preds_root, acceptance_json_path, output_dir):
        self._plan_path = Path(plan_path)
        self._table_path = Path(table_path)
        self._preds_root = Path(preds_root)
        self._acceptance_path = Path(acceptance_json_path)
        self._output_dir = Path(output_dir)

    def run(self):
        plan = json.loads(self._plan_path.read_text())
        acceptance = json.loads(self._acceptance_path.read_text())
        rows = MeasurementTable.read(self._table_path)
        csv_by_obs = {row["obs_id"]: row for row in rows}
        observations = plan["observations"]
        missing = sorted({obs["obs_id"] for obs in observations} - set(csv_by_obs))
        extra = sorted(set(csv_by_obs) - {obs["obs_id"] for obs in observations})
        if missing or extra:
            raise AttributionError(f"plan and measurement table disagree: plan-only={missing[:3]} csv-only={extra[:3]}")

        baseline_b = acceptance.get("bootstrap", {}).get("B")
        if not baseline_b:
            raise AttributionError("final-acceptance JSON carries no bootstrap B")
        judge = CompensationJudge(acceptance)

        remeasured, per_case_rows, reconciliation = self._remeasure_all(observations, csv_by_obs)

        entries = []
        for challenge in sorted({obs["challenge"] for obs in observations}):
            challenge_obs = [obs for obs in observations if obs["challenge"] == challenge]
            before_real, before_gen = self._pair_rows([csv_by_obs[obs["obs_id"]] for obs in challenge_obs])
            after_real, after_gen = self._pair_rows([self._pseudo_row(obs, remeasured.get(obs["obs_id"])) for obs in challenge_obs])
            for quantity_name in QUANTITY_TOST_INDEX:
                entries.append(judge.judge(challenge, quantity_name, before_real, before_gen, after_real, after_gen))

        attribution = CompensationAttributor().attribute(entries)
        for entry in entries:  # per-case diff columns read from the judged entries
            for row in per_case_rows:
                diff = entry["per_obs_diff"].get(row["obs_id"], {})
                row[f"{entry['quantity']}_before"] = diff.get("before")
                row[f"{entry['quantity']}_after"] = diff.get("after")

        reporter = AttributionReport(acceptance.get("run_id"), baseline_b)
        report = reporter.build(entries, per_case_rows, attribution, reconciliation)
        json_path, _ = reporter.write(report, self._output_dir)
        print(f"[OK] variant={report['variant']} verdict={report['attribution']['verdict']} -> {json_path}")
        return report

    def _remeasure_all(self, observations, csv_by_obs):
        """Full-array + overlap re-measurement of every observation, with the
        CSV reconciliation guard (tolerance 1e-9, never silent)."""
        remeasured, per_case_rows = {}, []
        reconciled = skipped = 0
        for observation in observations:
            obs_id = observation["obs_id"]
            csv_vol = MeasurementTable.number(csv_by_obs[obs_id], "vol_wt_ml")
            csv_cz = MeasurementTable.number(csv_by_obs[obs_id], "cz_wt_mm")
            row = {
                "obs_id": obs_id,
                "challenge": observation["challenge"],
                "case": observation["case"],
                "side": observation["side"],
                "anchor": csv_by_obs[obs_id].get("anchor", ""),
                "vol_wt_ml_before": None,
                "cz_wt_mm_before": None,
                "vol_wt_ml_after": None,
                "cz_wt_mm_after": None,
            }
            if csv_vol is None and csv_cz is None:  # failed-run row: no measurement to reconcile
                skipped += 1
            else:
                measured = OverlapRemeasurer.measure(self._read_prediction(observation), observation["side"])
                if csv_vol is not None and abs(measured["vol_ml"] - csv_vol) > RECONCILIATION_TOLERANCE:
                    raise AttributionError(
                        f"{obs_id}: full-array WT volume {measured['vol_ml']} != measurement CSV {csv_vol} "
                        f"(tolerance {RECONCILIATION_TOLERANCE:g})"
                    )
                measured_cz = measured["cz_index"]
                cz_mismatch = (measured_cz is None) != (csv_cz is None) or (
                    measured_cz is not None and abs(measured_cz - csv_cz) > RECONCILIATION_TOLERANCE
                )
                if cz_mismatch:
                    raise AttributionError(
                        f"{obs_id}: full-array WT centroid z {measured_cz} != measurement CSV {csv_cz} " f"(tolerance {RECONCILIATION_TOLERANCE:g})"
                    )
                reconciled += 1
                remeasured[obs_id] = measured
                row["vol_wt_ml_before"] = measured["vol_ml"]
                row["cz_wt_mm_before"] = measured["cz_index"]
                row["vol_wt_ml_after"] = measured["vol_ml_overlap"]
                row["cz_wt_mm_after"] = measured["cz_overlap_mm"]
            per_case_rows.append(row)
        reconciliation = {
            "guards_passed": True,
            "tolerance": RECONCILIATION_TOLERANCE,
            "n_reconciled": reconciled,
            "n_skipped_no_measurement": skipped,
            "statement": "全域重算容差(1e-9)内复现测量表;终验 TOST 重放容差(1e-9)内复现 json;排除计数一致",
        }
        return remeasured, per_case_rows, reconciliation

    def _read_prediction(self, observation):
        path = self._preds_root / observation["challenge"] / f"{observation['obs_id']}.nii.gz"
        if not path.is_file():
            raise AttributionError(f"prediction mask not found: {path}")
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if array.shape != PREDICTION_SHAPE:
            raise AttributionError(f"{observation['obs_id']}: mask shape {array.shape} != instrument prediction shape {PREDICTION_SHAPE}")
        return array

    @staticmethod
    def _pair_rows(rows):
        """The judge's split: one real row per case, plan-order gen rows."""
        real_by_case, gen_by_case = {}, {}
        for row in rows:
            if row["side"] == "real":
                if row["case"] in real_by_case:
                    raise AttributionError(f"multiple real rows for case {row['case']}")
                real_by_case[row["case"]] = row
            else:
                gen_by_case.setdefault(row["case"], []).append(row)
        orphans = sorted(set(gen_by_case) - set(real_by_case))
        if orphans:
            raise AttributionError(f"generated observations without a real reference row: {orphans[:3]}")
        return real_by_case, gen_by_case

    @staticmethod
    def _pseudo_row(observation, measured):
        """The compensated measurement as a QuantityFamily-compatible row;
        an unmeasurable observation (failed run) stays undefined and is
        excluded by the family's own semantics, never silently dropped."""
        return {
            "obs_id": observation["obs_id"],
            "case": observation["case"],
            "side": observation["side"],
            "vol_wt_ml": None if measured is None else measured["vol_ml_overlap"],
            "cz_wt_mm": None if measured is None else measured["cz_overlap_mm"],
        }


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, help="assembly plan JSON (L2 final acceptance)")
    parser.add_argument("--table", required=True, help="measurement CSV (measurement_run measure)")
    parser.add_argument("--preds", required=True, help="prediction root (<challenge>/<obs_id>.nii.gz)")
    parser.add_argument("--report", required=True, help="frozen l2_final_acceptance_<phase>.json baseline to reconcile against")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    try:
        ZcropAttributionJob(args.plan, args.table, args.preds, args.report, args.output_dir).run()
        return 0
    except AttributionError as error:
        print(f"ZCROP ATTRIBUTION FATAL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
