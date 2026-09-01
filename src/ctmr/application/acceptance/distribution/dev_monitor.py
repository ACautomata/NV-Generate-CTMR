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

"""Dev selection-point ET/WT monitor (issue #253, parent #247): the diagnostic
reading face of the de-blinded candidate selection.

The selection ruling (#210 adoption #5) replaces dev FID-only blindness with
the job-B measurement vocabulary plus a pre-recorded observation line. This
module is the monitor's report arm: it reads a dev-side per-observation
measurement CSV (the ``measurement_table.MEASUREMENT_FIELDS`` contract, the
frozen instrument's readings produced read-only by the sampling arm's plan and
the ``measurement_run`` execution side), reuses ``EtDiscrimination`` verbatim
for the ET axis, adds the WT volume addendum (the overestimation axis job B
measured on the holdout: GLI/PED/SSA rel medians +1.96/+12.5/+15.0), and
evaluates ``ObservationLine`` -- YELLOW FLAG on METS ET detection rate < 0.9
or any per-challenge vol_et_rel median > 2.

``variant=diagnostic``: a selection surface, never an acceptance verdict --
no judge import, no pass/fail vocabulary (source-pinned by test). The report
carries the per-case rows and per-challenge readings so the retrained
candidate's T8 run compares against this baseline like-for-like. The bootstrap
draw of the WT addendum takes the monitoring job's own registered slot
(``dev_monitor_wt_rel_diff``, block 600 -- re-registered from a 400 filing
before its first draw, after T5's 400/500 bands landed); the ET axis keeps drawing job B's
slot 200 through the unchanged ``EtDiscrimination``.

Usage:
    python -m ctmr.application.acceptance.distribution.dev_monitor \
        --measurements <monitor work root>/measurements_dev.csv \
        --sample-plan <monitor work root>/plan.json \
        --output-dir <monitor work root>/report [--run-id <run>]
"""

import argparse
import json
import sys
from pathlib import Path

from ctmr.application.acceptance.distribution.challenge_registry import BOOTSTRAP_B, DIAGNOSTIC_SEED_SLOTS
from ctmr.application.acceptance.distribution.diagnostic_support import (
    DiagnosticError,
    DiagnosticReportWriter,
    DiagnosticSeedAllocator,
)
from ctmr.application.acceptance.distribution.et_discrimination import EtDiscrimination, EtDiscriminationReport
from ctmr.application.acceptance.distribution.measurement_table import AcceptanceError, MeasurementTable
from ctmr.application.acceptance.distribution.observation_line import ObservationLine
from ctmr.application.acceptance.distribution.statistics import ClusterBootstrap, DistributionReadout, RelativeDifference


class WtMonitor:
    """Per-challenge WT volume addendum (the「ET/WT 监控」's WT axis).

    Same exclusion semantics as the ET discrimination -- input_fail/run_fail
    leave the denominator, hierarchy violations are measurement results and
    stay -- the shared ``RelativeDifference``/``DistributionReadout``
    definitions, and the monitoring job's own bootstrap slot for the per-case
    relative-difference CI90.
    """

    def __init__(self, bootstrap_b: int = BOOTSTRAP_B):
        self._bootstrap_b = bootstrap_b

    def readings(self, rows):
        """Per-challenge WT blocks, challenges sorted; empty row sets yield empty lists."""
        by_challenge: dict[str, list[dict]] = {}
        for row in rows:
            by_challenge.setdefault(row["challenge"], []).append(row)
        return [self._challenge_reading(challenge, by_challenge[challenge]) for challenge in sorted(by_challenge)]

    def _challenge_reading(self, challenge, rows):
        valid = {"gen": {}, "real": {}}
        for row in rows:
            if MeasurementTable.flag(row, "input_fail") or MeasurementTable.flag(row, "run_fail"):
                continue
            volume = MeasurementTable.number(row, "vol_wt_ml")
            if volume is not None:
                valid[row["side"]][row["case"]] = volume
        rel_values = [
            RelativeDifference.of(gen_volume, valid["real"][case])
            for case, gen_volume in valid["gen"].items()
            if case in valid["real"] and valid["real"][case] is not None
        ]
        reading = {
            "challenge": challenge,
            "gen": {"n": len(valid["gen"]), "vol_ml": DistributionReadout.of(list(valid["gen"].values()))},
            "real": {"n": len(valid["real"]), "vol_ml": DistributionReadout.of(list(valid["real"].values()))},
            "rel_diff": self._rel_diff_stats(rel_values, challenge),
        }
        return reading

    def _rel_diff_stats(self, rel_values, challenge):
        """Distribution read-out of the per-case relative differences, monitoring-slot CI90."""
        stats = DistributionReadout.of(rel_values)
        stats.update({"ci90_low": None, "ci90_high": None, "n_cases": len(rel_values)})
        if rel_values:
            seed = DiagnosticSeedAllocator.seed(challenge, DIAGNOSTIC_SEED_SLOTS["dev_monitor_wt_rel_diff"])
            ci = ClusterBootstrap(self._bootstrap_b).ci90([[value] for value in rel_values], seed)
            stats["ci90_low"], stats["ci90_high"] = ci["low"], ci["high"]
        return stats


class DevMonitorReport:
    """Dev monitor json + markdown artifacts (sugon artifact area, never git).

    One write composes the whole reading face: job-B ET discrimination, the WT
    addendum, the observation-line flag and the sample-protocol provenance from
    the sampling arm's plan.
    """

    SCHEMA = "dev-etwt-monitor-diagnostic/1"
    TITLE = "dev 监控:选择点 ET/WT 观察线读数"

    def __init__(self, measurements_path, sample_plan=None, run_id: str | None = None, bootstrap_b: int = BOOTSTRAP_B):
        self._measurements_path = Path(measurements_path)
        self._sample_plan_path = Path(sample_plan) if sample_plan else None
        self._run_id = run_id
        self._bootstrap_b = bootstrap_b
        self._writer = DiagnosticReportWriter(
            schema=self.SCHEMA,
            title=self.TITLE,
            issue=253,
            parent_issue=247,
            job_label="T6 dev 监控",
            stem="dev_monitor_diagnostic",
            inputs={
                "measurements": str(self._measurements_path),
                "sample_plan": str(self._sample_plan_path) if self._sample_plan_path else None,
            },
            run_id=self._run_id,
        )

    def write(self, output_dir, rows=None):
        if rows is None:
            rows = self.read_rows()
        readings = EtDiscrimination(bootstrap_b=self._bootstrap_b).discriminate(rows)
        flag = ObservationLine().evaluate(readings)
        wt_readings = {reading["challenge"]: reading for reading in WtMonitor(bootstrap_b=self._bootstrap_b).readings(rows)}
        for reading in readings:
            reading["wt"] = wt_readings.get(reading["challenge"])
        payload = self._writer.payload(
            {
                "sample": self._sample_block(),
                "reading_conventions": {
                    "et": "ET 轴复用作业 B 甄别口径:检出 = vol_et_ml > 0,分母为 input_fail/run_fail 之外的有效观测;"
                    "ET 缺失 = real_only;相对差 = (gen − real)/real,gen 侧 ET 空保留 −1.0",
                    "wt": "WT 添注:同排除语义的 vol_wt 分布与逐 case 相对差(median + CI90,监控槽位 600)",
                    "line": "观察线:METS ET 检出率 < 0.9 或任一挑战 vol_et_rel 中位 > 2 → 黄旗(选择面,非验收判定)",
                },
                "observation_line": flag,
                "per_challenge": {
                    reading["challenge"]: {key: value for key, value in reading.items() if key != "per_case_rows"} for reading in readings
                },
                "cross_challenge": EtDiscriminationReport.cross_challenge(readings),
                "per_case": EtDiscriminationReport.per_case(readings),
            }
        )
        return self._writer.write(payload, self._markdown(payload), output_dir)

    def read_rows(self):
        try:
            return list(MeasurementTable.read(self._measurements_path))
        except AcceptanceError as error:
            raise DiagnosticError(f"measurement table {self._measurements_path} is not a usable MEASUREMENT_FIELDS CSV: {error}") from error

    def _sample_block(self):
        """The sampling arm's protocol provenance (population/quota table), or None."""
        if self._sample_plan_path is None:
            return None
        plan = json.loads(self._sample_plan_path.read_text())
        return {
            "population": plan.get("population"),
            "sampling_rule": plan.get("sampling_rule"),
            "challenges": plan.get("challenges"),
        }

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload):
        flag = payload["observation_line"]
        sample = payload["sample"]
        flag_state = "黄旗" if flag["flag"] else "未触发"
        lines = self._writer.markdown_preamble(payload) + [
            "## 采样协议(dev 选择面,零 holdout 接触)",
            "",
            (
                f"- population: **{sample['population']}**;逐挑战样本量: "
                + ", ".join(f"{ch} {info['n_cases']}/{info['quota']}" for ch, info in sample["challenges"].items())
                if sample
                else "- (未提供 sample plan,采样协议块缺省)"
            ),
            f"- 采样规则:{(sample or {}).get('sampling_rule') or '未记录'}",
            "",
            "## 观察线判定(黄旗,选择面非验收判定)",
            "",
            f"- 判定线:METS ET 检出率 < {flag['rules']['mets_et_rate_floor']}"
            f" 或任一挑战 vol_et_rel 中位 > {flag['rules']['vol_et_rel_median_ceiling']}",
            f"- **状态:{flag_state}**",
        ]
        for fired in flag["fired"]:
            lines.append(f"  - 触发:{fired}")
        lines += [
            "",
            "## 逐挑战 ET/WT 读数",
            "",
            "| 挑战 | gen n | ET 检出 k/n | 检出率 | 空 pred | ET 缺失 real_only | ET vol median gen/real (ml) | ET 相对差 median (CI90)"
            " | WT vol median gen/real (ml) | WT 相对差 median (CI90) |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
        for challenge, reading in payload["per_challenge"].items():
            gen, real, rel, wt = reading["gen"], reading["real"], reading["rel_diff"], reading["wt"]
            lines.append(
                f"| {challenge} | {gen['n']} | {gen['k_detected']}/{gen['n']} | {self._fmt(gen['rate'])} "
                f"| {reading['empty_pred']['gen']['k']}/{reading['empty_pred']['gen']['n']} | {reading['pairing']['real_only']} "
                f"| {self._fmt(gen['vol_ml']['median'])} / {self._fmt(real['vol_ml']['median'])} "
                f"| {self._fmt(rel['median'])} ({self._fmt(rel['ci90_low'])}, {self._fmt(rel['ci90_high'])}) "
                f"| {self._fmt(wt['gen']['vol_ml']['median'])} / {self._fmt(wt['real']['vol_ml']['median'])} "
                f"| {self._fmt(wt['rel_diff']['median'])} ({self._fmt(wt['rel_diff']['ci90_low'])}, {self._fmt(wt['rel_diff']['ci90_high'])}) |"
            )
        cross = payload["cross_challenge"]
        lines += [
            "",
            f"跨挑战合计:ET 缺失 real_only **{cross['real_only']['k']}/{cross['real_only']['n']}**;"
            f"gen 空 pred **{cross['gen_empty_pred']['k']}/{cross['gen_empty_pred']['n']}**。",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--measurements", required=True, help="the monitor run's per-observation measurement CSV (MEASUREMENT_FIELDS, controlled storage)"
    )
    parser.add_argument("--sample-plan", default=None, help="the sampling arm's plan.json (sample-protocol provenance)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the monitor report (never git)")
    parser.add_argument("--run-id", default=None, help="the candidate's run id, recorded into the report")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    args = parser.parse_args(argv)

    report = DevMonitorReport(
        Path(args.measurements),
        sample_plan=args.sample_plan,
        run_id=args.run_id,
        bootstrap_b=args.bootstrap_b,
    )
    rows = report.read_rows()
    json_path, md_path = report.write(Path(args.output_dir), rows=rows)
    print(f"[OK] {len(rows)} observations (variant=diagnostic, selection surface -- no verdict) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
