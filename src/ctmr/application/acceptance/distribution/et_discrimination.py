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

"""Diagnostic job B (issue #207, parent #205): frozen-instrument ET discrimination.

Expert review (#58) reads missing enhancing tumour (ET) in the modality-label-
conditioned candidate's generated t1c; job B turns that impression into
per-challenge numbers. Every generated holdout pseudo-quad (530 cases = frozen
quotas 250 GLI + 200 MEN + 48 METS + 20 PED + 12 SSA) already passed the frozen
instrument during the P1 L2 terminal acceptance, so this job re-reads the
retained per-observation measurement CSV -- the instrument readings themselves
(``vol_et_ml``/``pred_empty`` per observation) -- and never touches any frozen
artifact, mask or envelope. Readings reuse the #38 synthetic-domain vocabulary:
empty pred = instrument argmax all-zero (a measurement result, never a
failure), Wilson 95% upper bounds for the detection rates, and the terminal-
acceptance convention that a generated-side empty prediction stays in the
relative-volume distributions at rel diff -1.0 (protocol §4).

This module is ``variant=diagnostic``: it never produces an acceptance verdict
and is deliberately not a ``ctmr accept`` verb -- diagnostic readings stay
strictly separated from the formal acceptance surface (parent decision). The
sugon host recipe lives at ``deploy/jobs/run_et_discrimination_b.sh``; reports
land in the sugon artifact area (controlled storage), never in git.

P3 candidate reuse (#205 series-③ merge point): the input surface is the
per-observation measurement CSV (``measurement_table.MEASUREMENT_FIELDS``
contract), which is phase-agnostic -- P3's four anchor rounds simply append
``__gen__a<anchor>`` observations. Whether the detection denominator then stays
per-observation or collapses to per-case is a series-③ protocol decision,
recorded here as the reuse hook, not pre-decided.

Usage:
    python -m ctmr.application.acceptance.distribution.et_discrimination \
        --measurements <l2 run tree>/measurements.csv \
        --output-dir <artifact area>/et_discrimination [--run-id <run>]
"""

import argparse
import sys
from pathlib import Path

from ctmr.application.acceptance.distribution.challenge_registry import (
    BOOTSTRAP_B,
    CHALLENGES,
    DIAGNOSTIC_SEED_SLOTS,
    HOLDOUT_QUOTAS,
)
from ctmr.application.acceptance.distribution.diagnostic_support import (
    DiagnosticError,
    DiagnosticReportWriter,
    DiagnosticSeedAllocator,
)
from ctmr.application.acceptance.distribution.measurement_table import AcceptanceError, MeasurementTable
from ctmr.application.acceptance.distribution.statistics import ClusterBootstrap, DistributionReadout, RelativeDifference
from ctmr.domain.measurement import WilsonUpper


class EtDiscrimination:
    """Per-challenge ET detection / volume / empty-pred readings (pure functions).

    A row is a *valid measurement* unless input_fail or run_fail fired -- only
    valid rows enter the detection denominators (an instrument that never ran
    is not a "no ET" result). Hierarchy violations are measured results and
    stay in every denominator, tallied separately. Detection means
    ``vol_et_ml > 0``; a case pairs its ``gen`` and ``real`` observations and
    the ``real_only`` class IS the ET-missing read-out the job exists for.
    """

    @staticmethod
    def wilson_95_upper(k, n):
        """Wilson score 95% upper bound -- the domain's single definition -- with the
        diagnostic report's ``None`` sentinel for an empty denominator (json has no NaN)."""
        if n == 0:
            return None
        return WilsonUpper.of(k, n)

    @staticmethod
    def rel_diff(gen_vol, real_vol):
        """(gen - real) / real -- ``statistics.RelativeDifference.of``, the shared
        definition; this name stays for the reading conventions (see ``rel_diff``
        in the report payload) and the tests pinned on it."""
        return RelativeDifference.of(gen_vol, real_vol)

    @staticmethod
    def pair_class(gen_vol, real_vol):
        """The four-quadrant detection pairing; ``real_only`` is the ET-missing class."""
        gen_detected = gen_vol is not None and gen_vol > 0
        real_detected = real_vol is not None and real_vol > 0
        if gen_detected and real_detected:
            return "both_detected"
        if real_detected:
            return "real_only"
        if gen_detected:
            return "gen_only"
        return "neither"

    def __init__(self, bootstrap_b: int = BOOTSTRAP_B):
        self._bootstrap_b = bootstrap_b

    def discriminate(self, rows):
        """Per-challenge readings, challenges sorted; empty row sets yield empty lists."""
        by_challenge: dict[str, list[dict]] = {}
        for row in rows:
            by_challenge.setdefault(row["challenge"], []).append(row)
        return [self._discriminate_challenge(challenge, by_challenge[challenge]) for challenge in sorted(by_challenge)]

    def _discriminate_challenge(self, challenge, rows):
        valid = {"gen": [], "real": []}
        excluded = {"input_fail": 0, "run_fail": 0}
        hier_viol = 0
        case_rows = []
        for row in rows:
            vol = MeasurementTable.number(row, "vol_et_ml")
            excluded_reason = None
            if MeasurementTable.flag(row, "input_fail"):
                excluded["input_fail"] += 1
                excluded_reason = "input_fail"
            elif MeasurementTable.flag(row, "run_fail"):
                excluded["run_fail"] += 1
                excluded_reason = "run_fail"
            else:
                valid[row["side"]].append(row)
                if MeasurementTable.flag(row, "hier_viol"):
                    hier_viol += 1
            case_rows.append(
                {
                    "challenge": challenge,
                    "case": row["case"],
                    "side": row["side"],
                    "vol_et_ml": vol,
                    "detected": vol is not None and vol > 0,
                    "pred_empty": MeasurementTable.flag(row, "pred_empty"),
                    "pair_class": None,
                    "rel_diff": None,
                    "excluded": excluded_reason,
                }
            )

        sides = {name: self._side_stats(rows) for name, rows in valid.items()}
        pairing, rel_values, pair_by_case = self._pair(valid["gen"], valid["real"])
        # annotate the per-case detail rows where the pairing lives: pair_class and
        # rel_diff only exist for cases present on both sides of a valid measurement
        for row in case_rows:
            if row["excluded"] is None and row["case"] in pair_by_case:
                row["pair_class"], row["rel_diff"] = pair_by_case[row["case"]]
        reading = {
            "challenge": challenge,
            "gen": sides["gen"],
            "real": sides["real"],
            "empty_pred": {name: {"k": self._empty_pred(rows), "n": len(rows)} for name, rows in valid.items()},
            "pairing": pairing,
            "excluded": excluded,
            "hier_viol": hier_viol,
            "per_case_rows": case_rows,
        }
        reading["rel_diff"] = self._rel_diff_stats(rel_values, challenge)
        return reading

    def _side_stats(self, rows):
        """Detection and ET-volume distribution of one side's valid rows."""
        vols = [MeasurementTable.number(row, "vol_et_ml") for row in rows]
        detected = [volume for volume in vols if volume is not None and volume > 0]
        return {
            "n": len(rows),
            "k_detected": len(detected),
            "rate": len(detected) / len(rows) if rows else None,
            "wilson_95_upper": self.wilson_95_upper(len(detected), len(rows)),
            "vol_ml": DistributionReadout.of([volume for volume in vols if volume is not None]),
        }

    @staticmethod
    def _empty_pred(rows):
        return sum(1 for row in rows if MeasurementTable.flag(row, "pred_empty"))

    @staticmethod
    def _pair(gen_rows, real_rows):
        """Four-quadrant pairing over the case intersection; unmatched sides tally as unpaired.

        Returns the class counts, the per-case relative differences (for the
        distribution statistics) and the per-case ``(pair_class, rel_diff)``
        annotations for the report's detail rows.
        """
        gen = {row["case"]: MeasurementTable.number(row, "vol_et_ml") for row in gen_rows}
        real = {row["case"]: MeasurementTable.number(row, "vol_et_ml") for row in real_rows}
        counts = {"both_detected": 0, "real_only": 0, "gen_only": 0, "neither": 0, "unpaired": 0}
        rel_values = []
        pair_by_case = {}
        for case in sorted(set(gen) | set(real)):
            if case not in gen or case not in real:
                counts["unpaired"] += 1
                continue
            pair = EtDiscrimination.pair_class(gen[case], real[case])
            diff = EtDiscrimination.rel_diff(gen[case], real[case])
            counts[pair] += 1
            pair_by_case[case] = (pair, diff)
            if diff is not None:
                rel_values.append(diff)
        return counts, rel_values, pair_by_case

    def _rel_diff_stats(self, rel_values, challenge):
        """Distribution read-out of the per-case relative differences, diagnostic-seed CI90."""
        stats = DistributionReadout.of(rel_values)
        stats.update({"ci90_low": None, "ci90_high": None, "n_cases": len(rel_values)})
        if rel_values:
            seed = DiagnosticSeedAllocator.seed(challenge, DIAGNOSTIC_SEED_SLOTS["et_rel_diff"])
            ci = ClusterBootstrap(self._bootstrap_b).ci90([[value] for value in rel_values], seed)
            stats["ci90_low"], stats["ci90_high"] = ci["low"], ci["high"]
        return stats


# ── report ──────────────────────────────────────────────────────────────


class EtDiscriminationReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git)."""

    SCHEMA = "et-discrimination-diagnostic/1"
    TITLE = "诊断作业 B:纯生成样本冻结仪器 ET 甄别"

    def __init__(self, measurements_path, bootstrap_b: int = BOOTSTRAP_B, run_id: str | None = None):
        self._measurements_path = Path(measurements_path)
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id
        self._writer = DiagnosticReportWriter(
            schema=self.SCHEMA,
            title=self.TITLE,
            issue=207,
            job_label="作业 B",
            stem="et_discrimination_diagnostic",
            inputs={"measurements": str(self._measurements_path)},
            run_id=self._run_id,
        )

    def write(self, readings, output_dir):
        payload = self._writer.payload(
            {
                "reading_conventions": {
                    "detection": "检出 = vol_et_ml > 0(仪器测量结果);分母为 input_fail/run_fail 之外的有效观测",
                    "empty_pred": "空 pred = 仪器 argmax 全 0(整例未检出),测量结果而非失败(#38 口径)",
                    "real_only": "real 侧 ET 检出而 gen 侧未检出——ET 缺失定量读数(本作业的核心甄别量)",
                    "rel_diff": "(gen - real)/real;gen 侧 ET 空保留 -1.0(协议 §4),real 侧空不定义",
                    "p3_reuse_hook": "输入面为逐观测测量 CSV(measurement_table.MEASUREMENT_FIELDS 契约),phase 无关;"
                    "P3 四锚轮的分子分母口径(逐观测 vs 逐 case 聚合)由 #205 序列③拍板",
                },
                "per_challenge": {
                    reading["challenge"]: {key: value for key, value in reading.items() if key != "per_case_rows"} for reading in readings
                },
                "cross_challenge": self._cross_challenge(readings),
                "per_case": self._per_case(readings),
            }
        )
        return self._writer.write(payload, self._markdown(payload), output_dir)

    @staticmethod
    def _cross_challenge(readings):
        """Tally across challenges: total ET-missing and empty-pred counts over valid gen rows."""
        real_only = empty = n = 0
        for reading in readings:
            real_only += reading["pairing"]["real_only"]
            empty += reading["empty_pred"]["gen"]["k"]
            n += reading["gen"]["n"]
        return {"real_only": {"k": real_only, "n": n}, "gen_empty_pred": {"k": empty, "n": n}}

    @staticmethod
    def _per_case(readings):
        return [row for reading in readings for row in reading["per_case_rows"]]

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload):
        lines = self._writer.markdown_preamble(payload) + [
            "## 读数口径",
            "",
            f"- 输入:measurements `{payload['inputs']['measurements']}`(逐观测仪器读数,只读)",
            f"- 检出:{payload['reading_conventions']['detection']}",
            f"- 空 pred:{payload['reading_conventions']['empty_pred']}",
            f"- ET 缺失:{payload['reading_conventions']['real_only']}",
            f"- 体积相对差:{payload['reading_conventions']['rel_diff']}",
            f"- 分母对照:各挑战 gen n 应达冻结持出配额 "
            f"({'/'.join(f'{ch} {HOLDOUT_QUOTAS[ch]}' for ch in ('GLI', 'MEN', 'METS', 'PED', 'SSA'))},合计 530)"
            "——不足即测量 CSV 不完整,读数如实呈现但不代表全量",
            "",
            "## 逐挑战甄别读数",
            "",
            "| 挑战 | gen n | gen ET 检出 k/n | 检出率 | Wilson 95% 上界 | gen 空 pred | real n | real ET 检出 k/n"
            " | ET 缺失 real_only | vol_et gen median (ml) | vol_et real median (ml) | 相对差 median (CI90) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for challenge, reading in payload["per_challenge"].items():
            gen, real = reading["gen"], reading["real"]
            rel = reading["rel_diff"]
            lines.append(
                f"| {challenge} | {gen['n']} | {gen['k_detected']}/{gen['n']} | {self._fmt(gen['rate'])} "
                f"| {self._fmt(gen['wilson_95_upper'])} | {reading['empty_pred']['gen']['k']}/{reading['empty_pred']['gen']['n']} "
                f"| {real['n']} | {real['k_detected']}/{real['n']} | {reading['pairing']['real_only']} "
                f"| {self._fmt(gen['vol_ml']['median'])} | {self._fmt(real['vol_ml']['median'])} "
                f"| {self._fmt(rel['median'])} ({self._fmt(rel['ci90_low'])}, {self._fmt(rel['ci90_high'])}) |"
            )
        cross = payload["cross_challenge"]
        lines += [
            "",
            f"跨挑战合计:ET 缺失 real_only **{cross['real_only']['k']}/{cross['real_only']['n']}**;"
            f"gen 空 pred **{cross['gen_empty_pred']['k']}/{cross['gen_empty_pred']['n']}**"
            "(与 #38 前证 P1 直出 METS 2/20、img2img 基线 METS 42/80 + MEN 2/80 横向对比)。",
            "",
            "## 逐 case 明细",
            "",
            "| 挑战 | case | side | vol_et (ml) | 检出 | 空 pred | 配对分类 | 相对差 | 排除 |",
            "|---|---|---|---:|---|---|---|---:|---|",
        ]
        for case in payload["per_case"]:
            lines.append(
                f"| {case['challenge']} | {case['case']} | {case['side']} | {self._fmt(case['vol_et_ml'])} "
                f"| {case['detected']} | {case['pred_empty']} | {case['pair_class'] or ''} "
                f"| {self._fmt(case['rel_diff'])} | {case['excluded'] or ''} |"
            )
        lines.append("")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", required=True, help="the L2 run tree's per-observation measurement CSV (controlled storage)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the diagnostic report (never git)")
    parser.add_argument("--challenges", nargs="+", default=list(CHALLENGES), choices=CHALLENGES, help="challenges to discriminate")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    parser.add_argument("--run-id", default=None, help="the candidate's L2 terminal-acceptance run id, recorded into the report")
    args = parser.parse_args(argv)

    try:
        rows = [row for row in MeasurementTable.read(args.measurements) if row["challenge"] in set(args.challenges)]
    except AcceptanceError as error:
        raise DiagnosticError(f"measurement table {args.measurements} is not a usable MEASUREMENT_FIELDS CSV: {error}") from error
    readings = EtDiscrimination(bootstrap_b=args.bootstrap_b).discriminate(rows)
    report = EtDiscriminationReport(Path(args.measurements), bootstrap_b=args.bootstrap_b, run_id=args.run_id)
    json_path, md_path = report.write(readings, Path(args.output_dir))
    print(f"[OK] {len(rows)} observations ({len(readings)} challenges, variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
