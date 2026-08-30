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

"""Geometry audit of diagnostic job A (issue #217, parent #205): readings under
the workpiece affine.

Job A (``zcrop_compensation.py``, #206/#213) registered the modality-label-stage
sampling geometry -- 256x256x128 @ (0.94, 0.94, 1.36) mm resampled to 1 mm into
174 z slices, centre-cropped onto the instrument grid (crop start 9, 19 slices
dropped) -- and compensated the centroid-z axis by +9 mm under it. While
executing job C (#208, PR #216) the holdout generated NIfTI artifacts were
found to carry the sidecar writer's unit affine ``np.diag([1, 1, 1])``, so on
the artifacts the 1 mm resample is a no-op and the instrument chain PADS the
128-slice volume onto the 155-slice grid (13 below, 14 above) instead of
cropping 19 slices. This module re-audits job A's centroid axis under the
workpiece affine WITHOUT touching any frozen artifact or the already-recorded
job A module: the same prediction masks are re-measured under both windows and
reported head-to-head.

The two geometries re-interpret one and the same mask set; the instrument's
uncompensated readings are shared. For a case whose tumour crosses no window
edge the compensated readings sit exactly 22 mm apart (+9 vs -13):

    comp_pad = uncomp - 13 = (uncomp + 9) - 22 = comp_crop - 22.

Cases violating the identity are the window-edge set -- real tumours crossing
the window floor (z=9 under the registered crop, z=13 under the workpiece pad;
the latter is a superset). The physics narrative changes too: under the pad
geometry the generated DECLARED domain is [0, 128) mm (1 mm x 128 slices), so
the job A signal "real tumours beyond the generator field of view at z<9 mm"
inverts -- that mass lies INSIDE the declared domain, while real mass at
z>=128 mm (and the content-margin gaps z<13 / z>=141) is what actually leaves
the generated space. ``FieldOfViewAudit`` quantifies all three from the real
and generated prediction masks.

This module is ``variant=diagnostic``: it never produces an acceptance verdict,
and the sugon host recipe lives at ``deploy/jobs/run_zcrop_geometry_audit_217.sh``;
reports land in the sugon artifact area (controlled storage), never in git.

Usage:
    python -m ctmr.application.acceptance.distribution.zcrop_geometry_audit \
        --measurements <l2 run tree>/measurements.csv \
        --pred-root <l2 run tree>/predictions \
        --output-dir <artifact area>/zcrop_geometry_audit [--run-id <run>]
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ctmr.application.acceptance.distribution.final_acceptance import (
    BOOTSTRAP_B,
    CHALLENGE_SEED_OFFSET,
    CHALLENGES,
    FROZEN_ENVELOPES,
    MeasurementTable,
)
from ctmr.application.acceptance.distribution.zcrop_compensation import (
    COMPENSATED_SEED_STRIDE,
    DIAGNOSTIC_SEED_BASE,
    GEN_RESAMPLED_Z,
    INSTRUMENT_Z,
    REGION_LABELS,
    AttributionJudge,
    DiagnosticError,
    NiftiMaskRepository,
    OverlapWindow,
    PairedCompensation,
    ZCropCompensation,
)

# The holdout generated NIfTI artifacts (sidecar writer convention, modality_label
# monitor): 256x256x128 voxels declared at unit 1 mm spacing.
GEN_WORKPIECE_Z = 128

# Diagnostic bootstrap seeds share job A's namespace discipline: job A occupies
# slots 0/1 (uncompensated) and 100/101 (compensated) of each challenge's
# 1000-wide band; job B takes 200; this audit takes the next free slot so no
# two diagnostic quantities ever draw one seed. The comp_crop block re-draws
# job A's compensated seed bit-stream EXACTLY (slot 1 + stride 100), so its CI
# is bit-identical to the recorded job A report for the same cases.
JOB_AUDIT_SEED_SLOT = 300

# In-window cases satisfy comp_pad - (uncomp - 13) == 0 exactly; anything
# beyond this tolerance is a window-edge case.
IDENTITY_TOLERANCE = 1e-9


class GeometryAuditError(DiagnosticError):
    """Raised when the audit inputs cannot support a geometry re-audit."""


# ── workpiece geometry ──────────────────────────────────────────────────


class WorkpieceGeometry:
    """The instrument-grid geometry the artifacts actually carry (pure functions).

    With the unit affine the 1 mm resample is a no-op and CenterCropOrPad
    (grid.py: pad start (target - source) // 2) places the generated content
    onto the grid margins [pad_start, pad_start + gen_z). The grid index i of
    the generated array then sits at declared physical z = i - pad_start --
    the negative dual of the registered crop offset.
    """

    @staticmethod
    def pad_start(source_z: int, target_z: int) -> int:
        """The centre-pad start index (grid.py ``CenterCropOrPad``: (t - s) // 2)."""
        return (target_z - source_z) // 2

    @staticmethod
    def overlap_window(gen_native_z: int, instrument_z: int) -> OverlapWindow:
        """Both sides restricted to the physical z range they actually share.

        The generated content occupies grid [pad_start, pad_start + gen_z) and
        the real domain is [0, instrument_z); the shared window is the content
        domain itself (pad margins carry no generated mass). ``crop_start``
        carries the negative grid-to-physical offset: generated array index 0
        sits at declared physical z = -pad_start mm. Pad-only: for a source
        taller than the grid use ``ZCropCompensation.overlap_window``.
        """
        start = WorkpieceGeometry.pad_start(gen_native_z, instrument_z)
        stop = start + gen_native_z
        return OverlapWindow(
            gen_slice=slice(start, stop),
            real_slice=slice(start, min(stop, instrument_z)),
            crop_start=-start,
            phys_lo=start,
            phys_hi=min(stop, instrument_z),
        )


WORKPIECE_WINDOW = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
"""The audit's pad-geometry window: content [13, 141) mm on the 155-slice grid."""

# The two geometries' grid-to-physical offsets, derived from the windows they
# name (compensated readings sit at comp = uncomp + offset), so the in-window
# identity spread is their difference -- one fact, no parallel literal copies.
CROP_COMPENSATION_MM = float(ZCropCompensation.crop_start(GEN_RESAMPLED_Z, INSTRUMENT_Z))
PAD_COMPENSATION_MM = -float(WORKPIECE_WINDOW.phys_lo)
IDENTITY_SPREAD_MM = CROP_COMPENSATION_MM - PAD_COMPENSATION_MM


# ── field-of-view audit ─────────────────────────────────────────────────


class FieldOfViewAudit:
    """Real tumour mass outside the generated domains, and generated mass in
    the pad margins (pure functions).

    Under the workpiece affine the generated DECLARED domain is [0, 128) mm --
    real WT mass at z >= 128 mm leaves it (the true field-of-view mismatch,
    inverting job A's z<9 mm narrative). The CONTENT domain [13, 141) mm adds
    the two placement gaps: real mass at z<13 mm and z>=141 mm faces pad-zero
    generated margins on the instrument grid -- a placement gap, not a
    field-of-view one (z<13 is inside the declared domain). Generated mass
    inside the pad margins should be exactly zero (pure zeros went in); any
    mass there is an instrument-chain anomaly, counted separately.
    """

    @staticmethod
    def real_wt_outside(mask, window: OverlapWindow = WORKPIECE_WINDOW, declared_hi: int = GEN_WORKPIECE_Z) -> dict:
        """Real WT mass (ml) below the content floor, above the content ceiling
        and above the declared ceiling (the last contains the second)."""
        wt = FieldOfViewAudit.wt_region(mask)
        return {
            "below_content_ml": float(wt[: window.phys_lo].sum()) * 0.001,
            "above_content_ml": float(wt[window.phys_hi :].sum()) * 0.001,
            "above_declared_ml": float(wt[declared_hi:].sum()) * 0.001,
        }

    @staticmethod
    def gen_mass_in_padding(mask, window: OverlapWindow = WORKPIECE_WINDOW) -> float:
        """Generated mass (ml) inside the pad margins, any label -- expected 0.0."""
        return float(np.count_nonzero(mask[: window.phys_lo]) + np.count_nonzero(mask[window.phys_hi :])) * 0.001

    @staticmethod
    def wt_region(mask):
        """The WT region of one prediction mask (labels 1/2/3), the instrument's own rule."""
        return np.isin(mask, REGION_LABELS["WT"])


# ── paired audit ────────────────────────────────────────────────────────


class PairedGeometryAudit:
    """Per-case head-to-head of the two geometry re-measurements of one mask set.

    Both windows run the frozen pairing rules verbatim (``PairedCompensation``:
    the registered crop window of job A and the workpiece pad window); the
    audit pairs the two compensated centroid readings per case and flags the
    identity residual ``comp_pad - (uncomp - 13)`` -- zero in-window, negative
    when the real tumour crosses the pad floor. The field-of-view scan runs
    over the same masks once per side, per challenge.
    """

    def __init__(self, bootstrap_b: int = BOOTSTRAP_B):
        self._bootstrap_b = bootstrap_b

    def audit(self, rows, repository) -> dict:
        crop_window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
        pad_window = WORKPIECE_WINDOW
        crop_readings = PairedCompensation(crop_window, bootstrap_b=self._bootstrap_b).read_cases(rows, repository)
        pad_readings = PairedCompensation(pad_window, bootstrap_b=self._bootstrap_b).read_cases(rows, repository)
        per_case = [self._pair_case(crop, pad) for crop, pad in zip(crop_readings, pad_readings)]
        return {
            "per_case": per_case,
            "window_edge_cases": {
                "crop_window": self._edge_cases(per_case, "comp_crop", CROP_COMPENSATION_MM),
                "pad_window": self._edge_cases(per_case, "comp_pad", PAD_COMPENSATION_MM),
            },
        }

    @staticmethod
    def _pair_case(crop: dict, pad: dict) -> dict:
        uncomp = pad["centroid_wt_z_uncomp"]
        comp_crop = crop["centroid_wt_z_comp"]
        comp_pad = pad["centroid_wt_z_comp"]
        residual = None
        if uncomp is not None and comp_pad is not None:
            residual = comp_pad - (uncomp + PAD_COMPENSATION_MM)
        return {
            "challenge": pad["challenge"],
            "case": pad["case"],
            "uncomp": uncomp,
            "comp_crop": comp_crop,
            "comp_pad": comp_pad,
            "identity_residual": residual,
            "excluded": crop["excluded"] or pad["excluded"],
        }

    @staticmethod
    def _edge_cases(per_case, comp_key: str, shift: float) -> list[dict]:
        """Cases violating the geometry's own identity (comp = uncomp + shift)."""
        edges = []
        for item in per_case:
            if item[comp_key] is None or item["uncomp"] is None:
                continue
            residual = item[comp_key] - (item["uncomp"] + shift)
            if abs(residual) > IDENTITY_TOLERANCE:
                edges.append(
                    {
                        "challenge": item["challenge"],
                        "case": item["case"],
                        "uncomp": item["uncomp"],
                        "comp": item[comp_key],
                        "residual": residual,
                    }
                )
        return edges

    def summary_stats(self, values, challenge: str, block: str) -> dict:
        """Quantiles + cluster-bootstrap CI90 under the audit's seed formula."""
        seed = GeometryAuditReport.centroid_seeds(challenge)[block]
        return PairedCompensation.summary_stats(values, self._bootstrap_b, seed)


# ── field-of-view scan ──────────────────────────────────────────────────


class FieldOfViewScan:
    """Per-challenge field-of-view tallies over the prediction masks."""

    def scan(self, rows, repository) -> dict:
        cases = {}
        for row in rows:
            cases.setdefault((row["challenge"], row["case"]), {})[row["side"]] = row
        per_challenge = {}
        for (challenge, case), sides in sorted(cases.items()):
            real_mask = repository.wt_mask(challenge, PairedCompensation.obs_id(case, "real"))
            gen_mask = repository.wt_mask(challenge, PairedCompensation.obs_id(case, "gen"))
            tally = per_challenge.setdefault(
                challenge,
                {
                    "real_below_content_ml": 0.0,
                    "real_above_content_ml": 0.0,
                    "real_above_declared_ml": 0.0,
                    "real_over_declared_cases": 0,
                    "worst_over_declared": None,
                    "gen_padding_ml": 0.0,
                    "gen_padding_cases": 0,
                },
            )
            if real_mask is not None:
                outside = FieldOfViewAudit.real_wt_outside(real_mask)
                for key in ("below_content_ml", "above_content_ml", "above_declared_ml"):
                    tally[f"real_{key}"] += outside[key]
                if outside["above_declared_ml"] > 0:
                    tally["real_over_declared_cases"] += 1
                    if tally["worst_over_declared"] is None or outside["above_declared_ml"] > tally["worst_over_declared"]["ml"]:
                        tally["worst_over_declared"] = {"case": case, "ml": outside["above_declared_ml"]}
            if gen_mask is not None:
                padding = FieldOfViewAudit.gen_mass_in_padding(gen_mask)
                tally["gen_padding_ml"] += padding
                if padding > 0:
                    tally["gen_padding_cases"] += 1
        return per_challenge


# ── report ──────────────────────────────────────────────────────────────


class GeometryAuditReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git).

    Margins quote the frozen ADR-0002 literals for orientation only -- this run
    registers no verdict and never touches the recorded job A artifacts.
    """

    SCHEMA = "zcrop-geometry-audit-diagnostic/1"
    TITLE = "复核作业 A:z-crop 几何基座复核(以工件 affine 为准)"
    QUANTITY = "centroid_wt_z"

    def __init__(self, measurements_path, pred_root, bootstrap_b: int = BOOTSTRAP_B, run_id: str | None = None):
        self._measurements_path = Path(measurements_path)
        self._pred_root = Path(pred_root)
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id

    @staticmethod
    def centroid_seeds(challenge: str) -> dict:
        """The audit's diagnostic seed formula: job A's bit-stream for the crop
        blocks (uncomp slot 1, comp slot 1+stride), the next free slot for pad."""
        band = DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET[challenge] * 1000
        return {
            "uncomp": band + 1,
            "comp_crop": band + 1 + COMPENSATED_SEED_STRIDE,
            "comp_pad": band + JOB_AUDIT_SEED_SLOT,
        }

    @staticmethod
    def wt_centroid_margin(challenge: str) -> float:
        return FROZEN_ENVELOPES[challenge]["WT"][2]

    @staticmethod
    def _geometry_blocks() -> dict:
        return {
            "registered_crop": {
                "assumption": (
                    "作业 A 注册几何:生成采样 256×256×128 @ (0.94, 0.94, 1.36) mm,"
                    f"1 mm 重采样后 z {GEN_RESAMPLED_Z} 层,居中裁 19 层(z 砍前 {int(CROP_COMPENSATION_MM)} 后 {GEN_RESAMPLED_Z - INSTRUMENT_Z - int(CROP_COMPENSATION_MM)})"
                ),
                "crop_start": int(CROP_COMPENSATION_MM),
                "overlap_window_mm": [9, 155],
                "compensation_mm": CROP_COMPENSATION_MM,
                "identity": "comp_crop = uncomp + 9",
            },
            "workpiece_pad": {
                "assumption": (
                    "工件现状几何(sidecar 写出约定 np.diag([1,1,1]),单位 1 mm affine):"
                    f"1 mm 重采样为 no-op,z 向 pad 前 {WORKPIECE_WINDOW.phys_lo} 后 {INSTRUMENT_Z - WORKPIECE_WINDOW.phys_hi},"
                    f"生成内容域 [{WORKPIECE_WINDOW.phys_lo}, {WORKPIECE_WINDOW.phys_hi}) mm"
                ),
                "pad_start": WORKPIECE_WINDOW.phys_lo,
                "content_domain_mm": [WORKPIECE_WINDOW.phys_lo, WORKPIECE_WINDOW.phys_hi],
                "overlap_window_mm": [WORKPIECE_WINDOW.phys_lo, WORKPIECE_WINDOW.phys_hi],
                "declared_domain_mm": [0, GEN_WORKPIECE_Z],
                "compensation_mm": PAD_COMPENSATION_MM,
                "identity": "comp_pad = uncomp - 13",
            },
            "identity_spread_mm": IDENTITY_SPREAD_MM,
            "note": (
                f"两种几何是对同一批预测掩码的两种解读,仪器未补偿读数共享;"
                f"不跨窗 case 的补偿读数解析间距恰为 {CROP_COMPENSATION_MM:+.0f}-({PAD_COMPENSATION_MM:+.0f})={IDENTITY_SPREAD_MM:.0f} mm。"
                "复核不回改已落盘的作业 A 模块与实验记录(P1 L2 仍 FAIL,判定不变)"
            ),
        }

    def write(self, audit: dict, field_of_view: dict | None, output_dir):
        challenges = sorted({reading["challenge"] for reading in audit["per_case"]})
        per_challenge = {challenge: {self.QUANTITY: self._quantity_block(audit, challenge)} for challenge in challenges}
        payload = {
            "schema": self.SCHEMA,
            "title": self.TITLE,
            "issue": 217,
            "variant": "diagnostic",
            "disclaimer": (
                "诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#205 复核作业,#217)。"
                f"bootstrap 种子独立于正式判定链(诊断基 {DIAGNOSTIC_SEED_BASE},本作业 slot {JOB_AUDIT_SEED_SLOT})。"
            ),
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"measurements": str(self._measurements_path), "pred_root": str(self._pred_root)},
            "geometries": self._geometry_blocks(),
            "per_case": audit["per_case"],
            "per_challenge": per_challenge,
            "window_edge_cases": audit["window_edge_cases"],
            "field_of_view": field_of_view or {},
            "attribution_overall": self._attribution_overall(per_challenge),
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "zcrop_geometry_audit_diagnostic.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / "zcrop_geometry_audit_diagnostic.md"
        md_path.write_text(self._markdown(payload))
        return json_path, md_path

    def _quantity_block(self, audit: dict, challenge: str) -> dict:
        margin = self.wt_centroid_margin(challenge)
        cases = [reading for reading in audit["per_case"] if reading["challenge"] == challenge]
        seeds = self.centroid_seeds(challenge)
        uncomp_stats = PairedCompensation.summary_stats(
            [case["uncomp"] for case in cases if case["uncomp"] is not None], self._bootstrap_b, seeds["uncomp"]
        )
        comp_crop_stats = PairedCompensation.summary_stats(
            [case["comp_crop"] for case in cases if case["comp_crop"] is not None], self._bootstrap_b, seeds["comp_crop"]
        )
        comp_pad_stats = PairedCompensation.summary_stats(
            [case["comp_pad"] for case in cases if case["comp_pad"] is not None], self._bootstrap_b, seeds["comp_pad"]
        )
        judge = AttributionJudge()
        return {
            "margin": margin,
            "uncomp": uncomp_stats,
            "comp_crop": comp_crop_stats,
            "comp_pad": comp_pad_stats,
            "attribution_crop": judge.classify(uncomp_stats["median"], comp_crop_stats["median"]),
            "attribution_pad": judge.classify(uncomp_stats["median"], comp_pad_stats["median"]),
        }

    @staticmethod
    def _attribution_overall(per_challenge: dict) -> dict:
        """Cross-challenge tally of the per-geometry attribution classes."""
        overall = {}
        for key in ("attribution_crop", "attribution_pad"):
            counts: dict[str, int] = {}
            for challenge in sorted(per_challenge):
                classification = per_challenge[challenge][GeometryAuditReport.QUANTITY][key]["classification"]
                counts[classification] = counts.get(classification, 0) + 1
            overall[key] = {"counts": counts, "majority": max(counts, key=counts.get) if counts else None}
        return overall

    @staticmethod
    def _fmt(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload: dict) -> str:
        geometries = payload["geometries"]
        lines = [
            f"# {payload['title']}",
            "",
            f"**Issue**: [#217](https://github.com/ACautomata/NV-Generate-CTMR/issues/217)(父 #205;发现来源 作业 C #208/PR #216)"
            f" · **run**: `{payload['run_id'] or '未绑定'}`",
            f"**variant: diagnostic —— {payload['disclaimer']}**",
            "",
            "## 几何口径",
            "",
            f"- 注册几何(作业 A):{geometries['registered_crop']['assumption']};{geometries['registered_crop']['identity']}",
            f"- 工件几何(本复核):{geometries['workpiece_pad']['assumption']};{geometries['workpiece_pad']['identity']}",
            f"- {geometries['note']}",
            f"- 输入:measurements `{payload['inputs']['measurements']}`;predictions `{payload['inputs']['pred_root']}`",
            "",
            "## 逐挑战对照读数(centroid_wt_z)",
            "",
            "| 挑战 | 未补偿 median (CI90) | comp_crop median (CI90) | comp_pad median (CI90) | margin | 归因(crop) | 归因(pad) |",
            "|---|---|---|---|---:|---|---|",
        ]
        for challenge, blocks in payload["per_challenge"].items():
            block = blocks[self.QUANTITY]
            attribution_crop = block["attribution_crop"]
            attribution_pad = block["attribution_pad"]
            lines.append(
                f"| {challenge} "
                f"| {self._fmt(block['uncomp']['median'])} ({self._fmt(block['uncomp']['ci90_low'])}, {self._fmt(block['uncomp']['ci90_high'])}) "
                f"| {self._fmt(block['comp_crop']['median'])} ({self._fmt(block['comp_crop']['ci90_low'])}, {self._fmt(block['comp_crop']['ci90_high'])}) "
                f"| {self._fmt(block['comp_pad']['median'])} ({self._fmt(block['comp_pad']['ci90_low'])}, {self._fmt(block['comp_pad']['ci90_high'])}) "
                f"| ±{self._fmt(block['margin'])} "
                f"| {attribution_crop['classification']} ({self._fmt(attribution_crop['measurement_fraction'])}) "
                f"| {attribution_pad['classification']} ({self._fmt(attribution_pad['measurement_fraction'])}) |"
            )
        lines += ["", "## 越窗清单(恒等式残差超容差)", ""]
        for window_name, shift in (("crop_window", "comp_crop = uncomp + 9"), ("pad_window", "comp_pad = uncomp - 13")):
            edges = payload["window_edge_cases"][window_name]
            lines.append(f"### {window_name}({shift}),{len(edges)} 例")
            lines.append("")
            if edges:
                lines.append("| 挑战 | case | 未补偿 | 补偿后 | 残差 |")
                lines.append("|---|---|---:|---:|---:|")
                for edge in edges:
                    lines.append(
                        f"| {edge['challenge']} | {edge['case']} | {self._fmt(edge['uncomp'])} "
                        f"| {self._fmt(edge['comp'])} | {self._fmt(edge['residual'])} |"
                    )
                lines.append("")
        lines += [
            "",
            "## 视场缺口定量(real WT 质量,ml)",
            "",
            "| 挑战 | z<13(内容域下缺口) | z≥141(内容域上缺口) | z≥128(声明域外) | 声明域外例数 | 最大声明域外 case | gen pad 区质量 |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for challenge, tally in payload["field_of_view"].items():
            worst = tally["worst_over_declared"]
            lines.append(
                f"| {challenge} | {self._fmt(tally['real_below_content_ml'])} | {self._fmt(tally['real_above_content_ml'])} "
                f"| {self._fmt(tally['real_above_declared_ml'])} | {tally['real_over_declared_cases']} "
                f"| {worst['case'] if worst else '无'} ({self._fmt(worst['ml']) if worst else 'n/a'}) "
                f"| {self._fmt(tally['gen_padding_ml'])} |"
            )
        lines += ["", "## 跨挑战归因汇总", ""]
        for key, label in (("attribution_crop", "注册几何(作业 A)"), ("attribution_pad", "工件几何(本复核)")):
            tally = payload["attribution_overall"][key]
            counts = "、".join(f"{name} ×{count}" for name, count in sorted(tally["counts"].items())) or "无可用挑战"
            lines.append(f"- {label}:多数归因 **{tally['majority']}**({counts})")
        lines.append("")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", required=True, help="the L2 run tree's per-observation measurement CSV (controlled storage)")
    parser.add_argument("--pred-root", required=True, help="the L2 run tree's predictions directory (<challenge>/<obs_id>.nii.gz)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the audit report (never git)")
    parser.add_argument("--challenges", nargs="+", default=list(CHALLENGES), choices=CHALLENGES, help="challenges to re-audit")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    parser.add_argument("--run-id", default=None, help="the candidate's L2 terminal-acceptance run id, recorded into the report")
    args = parser.parse_args(argv)

    rows = [row for row in MeasurementTable.read(args.measurements) if row["challenge"] in set(args.challenges)]
    if not rows:
        raise GeometryAuditError(f"no observations for challenges {sorted(set(args.challenges))} in {args.measurements}")
    repository = NiftiMaskRepository(args.pred_root)
    audit = PairedGeometryAudit(bootstrap_b=args.bootstrap_b).audit(rows, repository)
    field_of_view = FieldOfViewScan().scan(rows, repository)
    report = GeometryAuditReport(Path(args.measurements), Path(args.pred_root), args.bootstrap_b, run_id=args.run_id)
    json_path, md_path = report.write(audit, field_of_view, Path(args.output_dir))
    skipped = sum(1 for reading in audit["per_case"] if reading["excluded"])
    print(f"[OK] {len(audit['per_case'])} cases ({skipped} skipped, variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
