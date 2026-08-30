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

"""Diagnostic job A (issue #206, parent #205): z-crop compensated re-measurement.

The frozen L2 instrument chain measures the modality-label-conditioned candidate
on asymmetric z domains: generated volumes are resampled to 1 mm (241x241x174)
then centre-cropped onto the 240x240x155 instrument grid -- 19 z slices dropped
(crop start 9, so 9 below and 10 above) -- while the real side passes through
natively at 155 slices. The registered z-crop bias axis (protocol §2, report
appendix) is quantified here WITHOUT touching any frozen artifact: both sides
are restricted to the overlapping physical z window [crop_start, 155) mm and
WT relative volume / centroid z are re-measured from the retained per-observation
prediction masks, head-to-head with the uncompensated instrument readings.

This module is ``variant=diagnostic``: it never produces an acceptance verdict,
shares no bootstrap seed with the formal judge chain, and is deliberately not a
``ctmr accept`` verb -- diagnostic readings stay strictly separated from the
formal acceptance surface (parent decision). The sugon host recipe lives at
``deploy/jobs/run_zcrop_compensation_a.sh``; reports land in the sugon artifact
area (controlled storage), never in git, ready for the read-out ticket to
transcribe.

Physical coordinate convention (the instrument's own, see the canonical
``InstrumentMeasurer`` centroid in ctmr.domain.measurement): centroids are
array voxel indices on the 1 mm grid with no origin added. The generated array
index z therefore sits ``crop_start`` mm below the physical z of the same
tissue -- the coordinate artefact job A removes: compensated centroids are
reported in physical mm (generated side shifted by +crop_start) so both sides
compare inside the same window.

Usage:
    python -m ctmr.application.acceptance.distribution.zcrop_compensation \
        --measurements <l2 run tree>/measurements.csv \
        --pred-root <l2 run tree>/predictions \
        --output-dir <artifact area>/zcrop_compensation [--run-id <run>]
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from ctmr.application.acceptance.distribution.final_acceptance import (
    BOOTSTRAP_B,
    CHALLENGE_SEED_OFFSET,
    CHALLENGES,
    FROZEN_ENVELOPES,
    REGION_LABELS,
    ClusterBootstrap,
    MeasurementTable,
)
from ctmr.domain.grid import INSTRUMENT_GRID

# The registered modality-label-stage sampling (256x256x128 @ 0.94/0.94/1.36 mm)
# resampled to 1 mm: round(128 * 1.36) = 174 z slices before the centre crop.
GEN_RESAMPLED_Z = 174
INSTRUMENT_Z = INSTRUMENT_GRID.size[2]  # 155, the frozen instrument grid z

# Diagnostic bootstrap seeds are offset far away from the formal judge chain's
# GLOBAL_SEED (20260821): a diagnostic CI must never be mistaken for the
# registered TOST bit-stream. Each challenge occupies its own 1000-wide seed
# band (offset x 1000) so no (challenge, quantity, side) pair ever shares one.
DIAGNOSTIC_SEED_BASE = 900_000_000
# Uncompensated and compensated blocks of the same quantity draw disjoint
# diagnostic seed namespaces via this stride (kept inside the challenge band).
COMPENSATED_SEED_STRIDE = 100


class DiagnosticError(Exception):
    """Raised when the diagnostic inputs cannot support a compensation run."""


# ── compensation geometry ───────────────────────────────────────────────


@dataclass(frozen=True)
class OverlapWindow:
    """The overlapping physical z window, as per-side array slices (zyx layout).

    ``gen_slice``/``real_slice`` cut each side's 155-slice prediction array down
    to the shared window; ``crop_start`` is the generated array's physical offset
    (the generated array index 0 sits at physical z = crop_start mm);
    ``phys_lo``/``phys_hi`` are the window bounds in physical mm.
    """

    gen_slice: slice
    real_slice: slice
    crop_start: int
    phys_lo: int
    phys_hi: int


class ZCropCompensation:
    """Overlap-window geometry and mask re-measurement (pure functions).

    The re-measurement mirrors the frozen instrument's own rules (the canonical
    ``InstrumentMeasurer`` in ctmr.domain.measurement: volumes in ml at
    0.001 ml/voxel, centroids as 1 mm-grid array indices) restricted to the
    window, with the generated side shifted into physical mm so both sides
    compare inside one window.
    """

    @staticmethod
    def crop_start(source_z: int, target_z: int) -> int:
        """The centre-crop start index (grid.py ``CenterCropOrPad``: (s - t) // 2)."""
        return (source_z - target_z) // 2

    @staticmethod
    def overlap_window(gen_resampled_z: int, instrument_z: int) -> OverlapWindow:
        """Both sides restricted to the physical z range they actually share.

        The generated measurement domain spans physical [crop_start, crop_start
        + instrument_z); the real domain spans [0, instrument_z); the shared
        window is [crop_start, instrument_z). Slices keep equal window depth on
        both sides, so neither side is compensated alone (parent decision).
        """
        crop = ZCropCompensation.crop_start(gen_resampled_z, instrument_z)
        return OverlapWindow(
            gen_slice=slice(0, instrument_z - crop),
            real_slice=slice(crop, instrument_z),
            crop_start=crop,
            phys_lo=crop,
            phys_hi=instrument_z,
        )

    @staticmethod
    def restrict_and_measure(mask: np.ndarray, window: OverlapWindow, side: str) -> dict:
        """WT volume (ml) and physical-mm centroid z of one side, inside the window.

        An empty (or wholly out-of-window) WT region measures as volume 0.0 with
        no centroid -- a measurement result, never an error.
        """
        z_slice, crop_offset = {"gen": (window.gen_slice, window.crop_start), "real": (window.real_slice, 0)}[side]
        wt = np.isin(mask[z_slice], REGION_LABELS["WT"])
        volume_ml = float(wt.sum()) * 0.001
        if not wt.any():
            return {"vol_ml": 0.0, "centroid_z_mm": None}
        centroid_z, _y, _x = ndimage.center_of_mass(wt)
        # center_of_mass indexes the restricted array; physical mm = restricted
        # index + slice start + the generated array's crop offset
        return {"vol_ml": volume_ml, "centroid_z_mm": float(centroid_z + z_slice.start + crop_offset)}


class MaskRepository(Protocol):
    """Per-observation prediction-mask source, injected into the pairing stage."""

    def wt_mask(self, challenge: str, obs_id: str) -> np.ndarray | None: ...


class NiftiMaskRepository:
    """Reads retained per-observation prediction masks from the L2 run tree.

    Layout mirrors the measurement run: ``<pred_root>/<challenge>/<obs_id>.nii.gz``
    (controlled storage). A missing or grid-inconsistent mask reads as ``None`` --
    the case is skipped with a stated reason, never silently imputed.
    """

    def __init__(self, pred_root):
        self._pred_root = Path(pred_root)

    def wt_mask(self, challenge: str, obs_id: str) -> np.ndarray | None:
        path = self._pred_root / challenge / f"{obs_id}.nii.gz"
        try:
            array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except (RuntimeError, OSError):
            return None
        expected = (INSTRUMENT_Z, INSTRUMENT_GRID.size[1], INSTRUMENT_GRID.size[0])  # zyx
        if array.shape != expected:
            return None
        return array.astype(np.uint8, copy=False)


# ── per-case pairing ────────────────────────────────────────────────────


class PairedCompensation:
    """Per-case uncompensated vs compensated readings for the WT quantities.

    Uncompensated quantities reuse the instrument's own pairing rules verbatim
    (``QuantityFamily``: relative diff against the real denominator, a
    generated-side empty prediction kept at rel diff -1.0, centroid axes
    requiring a non-empty mask on both sides); compensated quantities apply the
    same rules to the in-window re-measurement. The two quantity families have
    independent availability -- a case can carry an uncompensated relative
    volume and no compensated centroid.
    """

    def __init__(self, window: OverlapWindow, bootstrap_b: int = BOOTSTRAP_B):
        self._window = window
        self._bootstrap_b = bootstrap_b

    @staticmethod
    def obs_id(case: str, side: str) -> str:
        """The pseudo-quad assembly spells observation ids ``<case>__real`` / ``<case>__gen``."""
        return f"{case}__{side}"

    def read_cases(self, rows, repository) -> list[dict]:
        pairs: dict[tuple[str, str], dict[str, dict]] = {}
        for row in rows:
            pairs.setdefault((row["challenge"], row["case"]), {})[row["side"]] = row
        return [self._read_case(challenge, case, sides, repository) for (challenge, case), sides in sorted(pairs.items())]

    def _read_case(self, challenge: str, case: str, sides: dict, repository) -> dict:
        real_row, gen_row = sides.get("real"), sides.get("gen")
        real_vol = MeasurementTable.number(real_row, "vol_wt_ml") if real_row else None
        gen_vol = MeasurementTable.number(gen_row, "vol_wt_ml") if gen_row else None
        real_cz = MeasurementTable.number(real_row, "cz_wt_mm") if real_row else None
        gen_cz = MeasurementTable.number(gen_row, "cz_wt_mm") if gen_row else None
        reading = {
            "challenge": challenge,
            "case": case,
            "vol_wt_ml_real": real_vol,
            "vol_wt_ml_gen": gen_vol,
            "cz_wt_mm_real": real_cz,
            "cz_wt_mm_gen": gen_cz,
            "vol_wt_ml_real_comp": None,
            "vol_wt_ml_gen_comp": None,
            "vol_wt_rel_uncomp": self._relative_diff(gen_vol, real_vol),
            "vol_wt_rel_comp": None,
            "centroid_wt_z_uncomp": self._signed_diff(gen_vol, real_vol, gen_cz, real_cz),
            "centroid_wt_z_comp": None,
            "excluded": None if real_row is not None and gen_row is not None else "missing_side_row",
        }
        real_mask = repository.wt_mask(challenge, self.obs_id(case, "real"))
        gen_mask = repository.wt_mask(challenge, self.obs_id(case, "gen"))
        if real_mask is None or gen_mask is None:
            reading["excluded"] = "missing_prediction"
            return reading
        real = ZCropCompensation.restrict_and_measure(real_mask, self._window, "real")
        gen = ZCropCompensation.restrict_and_measure(gen_mask, self._window, "gen")
        reading["vol_wt_ml_real_comp"] = real["vol_ml"]
        reading["vol_wt_ml_gen_comp"] = gen["vol_ml"]
        reading["vol_wt_rel_comp"] = self._relative_diff(gen["vol_ml"], real["vol_ml"])
        reading["centroid_wt_z_comp"] = self._signed_diff(gen["vol_ml"], real["vol_ml"], gen["centroid_z_mm"], real["centroid_z_mm"])
        return reading

    @staticmethod
    def _relative_diff(gen_value: float | None, real_value: float | None) -> float | None:
        """(gen - real) / real; the real denominator must exist and be positive."""
        if gen_value is None or real_value is None or real_value <= 0:
            return None
        return (gen_value - real_value) / real_value

    @staticmethod
    def _signed_diff(gen_vol, real_vol, gen_value, real_value):
        """Signed mm difference; both sides need a non-empty WT for the axis to exist."""
        if gen_vol is None or real_vol is None or gen_vol <= 0 or real_vol <= 0:
            return None
        if gen_value is None or real_value is None:
            return None
        return gen_value - real_value

    @staticmethod
    def summary_stats(values, bootstrap_b: int, seed: int) -> dict:
        """Distribution read-out of one per-case quantity: quantiles + cluster-bootstrap CI90.

        Quantiles use the same q*(n-1) linear rule as the calibration side
        (``ClusterBootstrap.quantile``); the CI resamples cases exactly like the
        formal judge, but under the diagnostic seed namespace.
        """
        if not values:
            return {"median": None, "mean": None, "q05": None, "q95": None, "ci90_low": None, "ci90_high": None, "n_cases": 0}
        ci = ClusterBootstrap(bootstrap_b).ci90([[value] for value in values], seed)
        return {
            "median": ClusterBootstrap.quantile(values, 0.5),
            "mean": sum(values) / len(values),
            "q05": ClusterBootstrap.quantile(values, 0.05),
            "q95": ClusterBootstrap.quantile(values, 0.95),
            "ci90_low": ci["low"],
            "ci90_high": ci["high"],
            "n_cases": len(values),
        }


# ── attribution ─────────────────────────────────────────────────────────


class AttributionJudge:
    """Three-way attribution of a failed reading's central shift.

    The fraction of the uncompensated median's offset from zero that the
    compensation removes is the measurement axis' share; whatever offset remains
    is the candidate's share. Dominance bands: measurement axis at or above 2/3
    of the offset, candidate at or above 1/3 remaining, otherwise mixed. A
    median at (or without) a defined uncompensated offset has no central shift
    to attribute -- its failure, if any, is a spread problem, not a shift one.
    """

    MEASUREMENT_DOMINANT_FRACTION = 2 / 3
    CANDIDATE_DOMINANT_FRACTION = 1 / 3

    def classify(self, median_uncomp: float | None, median_comp: float | None) -> dict:
        if median_uncomp is None or median_comp is None or median_uncomp == 0:
            return {"measurement_fraction": None, "classification": "no_central_shift"}
        fraction = max(0.0, min(1.0, (abs(median_uncomp) - abs(median_comp)) / abs(median_uncomp)))
        if fraction >= self.MEASUREMENT_DOMINANT_FRACTION:
            classification = "measurement_axis_dominant"
        elif fraction <= self.CANDIDATE_DOMINANT_FRACTION:
            classification = "candidate_dominant"
        else:
            classification = "mixed"
        return {"measurement_fraction": fraction, "classification": classification}


# ── report ──────────────────────────────────────────────────────────────


class DiagnosticReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git).

    Margins quote the frozen ADR-0002 literals for orientation only -- the
    within-margin flags compare like-for-like with the formal report, but this
    run registers no verdict.
    """

    SCHEMA = "zcrop-compensation-diagnostic/1"
    TITLE = "诊断作业 A:z-crop 补偿重算(测量轴归因)"
    QUANTITIES = ("vol_wt_rel", "centroid_wt_z")

    def __init__(self, measurements_path, pred_root, bootstrap_b: int, run_id: str | None = None):
        self._measurements_path = Path(measurements_path)
        self._pred_root = Path(pred_root)
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id

    @staticmethod
    def wt_vol_margin(challenge: str) -> float:
        return FROZEN_ENVELOPES[challenge]["WT"][1]

    @staticmethod
    def wt_centroid_margin(challenge: str) -> float:
        return FROZEN_ENVELOPES[challenge]["WT"][2]

    def write(self, readings, window: OverlapWindow, output_dir):
        challenges = sorted({reading["challenge"] for reading in readings})
        per_challenge = {
            challenge: {
                quantity: self._quantity_block(
                    [reading for reading in readings if reading["challenge"] == challenge],
                    challenge,
                    quantity,
                    DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET[challenge] * 1000 + index,
                )
                for index, quantity in enumerate(self.QUANTITIES)
            }
            for challenge in challenges
        }
        payload = {
            "schema": self.SCHEMA,
            "title": self.TITLE,
            "issue": 206,
            "variant": "diagnostic",
            "disclaimer": (
                f"诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#205 作业 A)。bootstrap 种子独立于正式判定链(诊断基 {DIAGNOSTIC_SEED_BASE})。"
            ),
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"measurements": str(self._measurements_path), "pred_root": str(self._pred_root)},
            "geometry": self._geometry(window),
            "per_case": readings,
            "per_challenge": per_challenge,
            "attribution_overall": self._attribution_overall(per_challenge),
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "zcrop_compensation_diagnostic.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / "zcrop_compensation_diagnostic.md"
        md_path.write_text(self._markdown(payload))
        return json_path, md_path

    @staticmethod
    def _geometry(window: OverlapWindow) -> dict:
        return {
            "sampling": "模态标签条件生成采样 256×256×128 @ (0.94, 0.94, 1.36) mm,1 mm 重采样后 z 174 层",
            "gen_resampled_z": GEN_RESAMPLED_Z,
            "instrument_z": INSTRUMENT_Z,
            "crop_start": window.crop_start,
            "overlap_window_mm": [window.phys_lo, window.phys_hi],
            "gen_slice": [window.gen_slice.start, window.gen_slice.stop],
            "real_slice": [window.real_slice.start, window.real_slice.stop],
            "note": (
                "生成侧重采样后居中裁到仪器栅格 240×240×155(z 砍前 9 后 10 层,共 19 层),真实侧原生 155 层不裁;"
                "本作业把两侧都限制到重叠物理域 [crop_start, 155) mm 后重算 WT 体积与质心 z,"
                "质心统一换算到物理 mm 口径(生成侧数组索引 + crop_start),避免单侧补偿引入新偏差"
            ),
        }

    def _quantity_block(self, cases, challenge: str, quantity: str, challenge_seed: int) -> dict:
        margin = self.wt_vol_margin(challenge) if quantity == "vol_wt_rel" else self.wt_centroid_margin(challenge)
        uncomp_stats = PairedCompensation.summary_stats(
            [case[f"{quantity}_uncomp"] for case in cases if case[f"{quantity}_uncomp"] is not None],
            self._bootstrap_b,
            challenge_seed,
        )
        comp_stats = PairedCompensation.summary_stats(
            [case[f"{quantity}_comp"] for case in cases if case[f"{quantity}_comp"] is not None],
            self._bootstrap_b,
            challenge_seed + COMPENSATED_SEED_STRIDE,
        )
        return {
            "margin": margin,
            "uncomp": uncomp_stats,
            "comp": comp_stats,
            "uncomp_ci_within_margin": self._within_margin(uncomp_stats, margin),
            "comp_ci_within_margin": self._within_margin(comp_stats, margin),
            "attribution": AttributionJudge().classify(uncomp_stats["median"], comp_stats["median"]),
        }

    @staticmethod
    def _within_margin(stats: dict, margin: float) -> bool | None:
        if stats["ci90_low"] is None:
            return None
        return stats["ci90_low"] >= -margin - 1e-12 and stats["ci90_high"] <= margin + 1e-12

    @staticmethod
    def _attribution_overall(per_challenge: dict) -> dict:
        """Cross-challenge tally of the per-quantity attribution classes."""
        overall = {}
        for quantity in DiagnosticReport.QUANTITIES:
            counts: dict[str, int] = {}
            for challenge in sorted(per_challenge):
                classification = per_challenge[challenge][quantity]["attribution"]["classification"]
                counts[classification] = counts.get(classification, 0) + 1
            overall[quantity] = {"counts": counts, "majority": max(counts, key=counts.get) if counts else None}
        return overall

    @staticmethod
    def _fmt(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload: dict) -> str:
        geometry = payload["geometry"]
        lines = [
            f"# {payload['title']}",
            "",
            f"**Issue**: [#206](https://github.com/ACautomata/NV-Generate-CTMR/issues/206)(父 #205 作业 A)"
            f" · **run**: `{payload['run_id'] or '未绑定'}`",
            f"**variant: diagnostic —— {payload['disclaimer']}**",
            "",
            "## 几何口径",
            "",
            f"- 采样与重采样:{geometry['sampling']}",
            f"- 居中裁剪:生成侧 z 砍前 {geometry['crop_start']} 后 {GEN_RESAMPLED_Z - geometry['instrument_z'] - geometry['crop_start']} 层;"
            f"真实侧原生 {geometry['instrument_z']} 层不裁",
            f"- 重叠物理域:[{geometry['overlap_window_mm'][0]}, {geometry['overlap_window_mm'][1]}) mm"
            f"(生成侧数组切片 z{geometry['gen_slice']},真实侧 z{geometry['real_slice']});{geometry['note']}",
            f"- 输入:measurements `{payload['inputs']['measurements']}`;predictions `{payload['inputs']['pred_root']}`",
            "",
            "## 逐量归因读数",
            "",
            "| 挑战 | 量 | 未补偿 median (CI90) | 补偿后 median (CI90) | margin | 未补偿 CI ⊆ 包络 | 补偿后 CI ⊆ 包络 | 测量轴占比 | 归因 |",
            "|---|---|---|---|---:|---|---|---:|---|",
        ]
        for challenge, quantities in payload["per_challenge"].items():
            for quantity, block in quantities.items():
                attribution = block["attribution"]
                lines.append(
                    f"| {challenge} | {quantity} "
                    f"| {self._fmt(block['uncomp']['median'])} ({self._fmt(block['uncomp']['ci90_low'])}, {self._fmt(block['uncomp']['ci90_high'])}) "
                    f"| {self._fmt(block['comp']['median'])} ({self._fmt(block['comp']['ci90_low'])}, {self._fmt(block['comp']['ci90_high'])}) "
                    f"| ±{self._fmt(block['margin'])} "
                    f"| {self._fmt_flag(block['uncomp_ci_within_margin'])} | {self._fmt_flag(block['comp_ci_within_margin'])} "
                    f"| {self._fmt(attribution['measurement_fraction'])} | {attribution['classification']} |"
                )
        lines += ["", "## 跨挑战归因汇总", ""]
        for quantity, tally in payload["attribution_overall"].items():
            counts = "、".join(f"{name} ×{count}" for name, count in sorted(tally["counts"].items())) or "无可用挑战"
            lines.append(f"- {quantity}:多数归因 **{tally['majority']}**({counts})")
        lines += [
            "",
            "## 逐 case 明细",
            "",
            "| 挑战 | case | vol_wt_rel 未补偿 | vol_wt_rel 补偿后 | centroid_wt_z 未补偿 | centroid_wt_z 补偿后 | 跳过原因 |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for reading in payload["per_case"]:
            lines.append(
                f"| {reading['challenge']} | {reading['case']} "
                f"| {self._fmt(reading['vol_wt_rel_uncomp'])} | {self._fmt(reading['vol_wt_rel_comp'])} "
                f"| {self._fmt(reading['centroid_wt_z_uncomp'])} | {self._fmt(reading['centroid_wt_z_comp'])} "
                f"| {reading['excluded'] or ''} |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _fmt_flag(value: bool | None) -> str:
        return "n/a" if value is None else ("是" if value else "否")


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", required=True, help="the L2 run tree's per-observation measurement CSV (controlled storage)")
    parser.add_argument("--pred-root", required=True, help="the L2 run tree's predictions directory (<challenge>/<obs_id>.nii.gz)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the diagnostic report (never git)")
    parser.add_argument("--challenges", nargs="+", default=list(CHALLENGES), choices=CHALLENGES, help="challenges to re-measure")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    parser.add_argument("--run-id", default=None, help="the candidate's L2 terminal-acceptance run id, recorded into the report")
    args = parser.parse_args(argv)

    rows = [row for row in MeasurementTable.read(args.measurements) if row["challenge"] in set(args.challenges)]
    if not rows:
        raise DiagnosticError(f"no observations for challenges {sorted(set(args.challenges))} in {args.measurements}")
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    readings = PairedCompensation(window, bootstrap_b=args.bootstrap_b).read_cases(rows, NiftiMaskRepository(args.pred_root))
    report = DiagnosticReport(Path(args.measurements), Path(args.pred_root), args.bootstrap_b, run_id=args.run_id)
    json_path, md_path = report.write(readings, window, Path(args.output_dir))
    skipped = sum(1 for reading in readings if reading["excluded"])
    print(f"[OK] {len(readings)} cases ({skipped} skipped, variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
