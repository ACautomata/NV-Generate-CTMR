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

"""Diagnostic job D (issue #209, parent #205): same-seed token-swap bright-core
discrimination (modality-label dilution, RC-2/L3).

The modality-label perturber (``ModalityLabelPerturber``, PINNED_PROB=0.1,
per-element Bernoulli) spends ~19% of the t1c training steps teaching the
candidate that token 34 carries no specific semantics (→8 pan-MR, →0
unknown), and CFG=10 amplifies the weakened (cond−uncond) guidance. Job D
isolates that axis from every other suspect: the frozen sampling rule keeps
its per-case seed -- ``sha256(case|t1c)`` exactly like the holdout/dev
samplers -- and every recipe knob (cfg=10, 30 steps, RFlowScheduler), ONLY
the condition token is swapped across five arms per case (t1n 29 / t1c 34 /
t2w 30 / t2f 31 plus the pan-MR control 8 the augmentation itself perturbs
into). Within one case the initial noise is therefore bit-identical across
arms, and every bright-core difference attributes to the token condition
alone. The discriminating read-out is the token-34 gain share over the
pan-MR control on the top-0.5% mean: a near-zero share reads dilution
dominant (token 34 ≈ token 8 -- the augmentation taught the candidate well),
a large share reads token semantics intact and moves the suspicion off RC-2.

This module is ``variant=diagnostic``: it never produces an acceptance
verdict and is deliberately not a ``ctmr accept`` verb -- diagnostic readings
stay strictly separated from the formal acceptance surface (parent decision).
It owns the statistics and the report only; the sampling arm that produces
the five per-case volumes lives at
``ctmr.application.generation.modality_label.token_swap_sampling`` and the
sugon host recipe at ``deploy/jobs/run_token_dilution_d.sh``. Reports land in
the sugon artifact area (controlled storage), never in git.

Diagnostic seed discipline (shared with jobs A/B, ADR-0017 pre-registration):
seeds hang off the diagnostic base 900,000,000, never the formal judge chain's
GLOBAL_SEED. Job D has no challenge band, so its slots hang directly off the
base: arms 300..304, contrasts 310..313, gain share 320 -- clear of job A
(slots 0/1/100/101) and job B (slot 200).

Usage (pure CPU statistics over the sampling arm's products):
    python -m ctmr.application.acceptance.distribution.token_dilution \
        --samples-dir <sampling arm output> --output-dir <artifact area>/token_dilution \
        [--checkpoint <frozen candidate .pt>] [--run-id <run>]
"""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.final_acceptance import (
    BOOTSTRAP_B,
    ClusterBootstrap,
)

# Diagnostic bootstrap seeds share jobs A/B's namespace discipline (zcrop_
# compensation.py DIAGNOSTIC_SEED_BASE): far away from the formal judge chain's
# GLOBAL_SEED (20260821) so a diagnostic CI can never be mistaken for the
# registered TOST bit-stream. See the module docstring for job D's slot map.
DIAGNOSTIC_SEED_BASE = 900_000_000
ARM_CI_SLOT_BASE = 300
CONTRAST_CI_SLOT_BASE = 310
SHARE_CI_SLOT = 320

# The five sampling arms: the four frozen modality tokens (shell.MODALITY_
# TOKENS, verbatim -- changes gate through the frozen sampling-rule surface)
# plus token 8, the pan-MR token the augmentation itself perturbs 34 into.
TOKEN_ARMS = {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31, "panmr": 8}
ARM_ORDER = ("t1n", "t1c", "t2w", "t2f", "panmr")
CANDIDATE_ARM = "t1c"  # the discriminated channel (ET's only conditional clue)
CONTROL_ARM = "panmr"  # the augmentation's own 34→8 target
CONTRAST_PAIRS = (("t1c", "t1n"), ("t1c", "t2w"), ("t1c", "t2f"), ("t1c", "panmr"))

# The seed anchor: one seed per case shared by all five arms (the frozen
# sampling rule's per-(case, modality) hash pinned at the discriminated
# channel), so cross-arm noise is bit-identical and only the token varies.
ANCHOR_MODALITY = "t1c"

# Bright-core top statistics on the nonzero-voxel basis (0 = air/background in
# the ×1000 int16 output domain; values >1000 are the >1.0 pre-clip output
# domain and are measured as-is -- job E's axis, never silently clipped here).
BRIGHT_QUANTILES = (0.99, 0.999)
TOP_FRACTION = 0.0005


class DiagnosticError(Exception):
    """Raised when the diagnostic inputs cannot support a token-dilution run."""


# ── bright-core statistics ──────────────────────────────────────────────


class BrightCoreStats:
    """Top-intensity read-out of one generated volume (pure functions).

    Basis: the nonzero voxels. Quantiles follow the calibration side's
    linear q*(n-1) rule (``ClusterBootstrap.quantile``); the top-0.5% mean
    keeps at least one voxel so tiny volumes stay measurable. An all-zero
    volume has no bright core -- that is a measurement result (nulls), never
    an error.
    """

    @staticmethod
    def of(volume) -> dict:
        values = np.asarray(volume).ravel()
        nonzero = np.sort(values[values > 0]).astype(np.float64)
        n = int(nonzero.size)
        if n == 0:
            return {"p99": None, "p99_9": None, "top05pct_mean": None, "max": None, "n_nonzero": 0}
        top_k = max(1, math.ceil(TOP_FRACTION * n))
        return {
            "p99": ClusterBootstrap.quantile(list(nonzero), BRIGHT_QUANTILES[0]),
            "p99_9": ClusterBootstrap.quantile(list(nonzero), BRIGHT_QUANTILES[1]),
            "top05pct_mean": float(nonzero[-top_k:].mean()),
            "max": float(nonzero[-1]),
            "n_nonzero": n,
        }


class SeedAnchor:
    """The per-case shared seed of the five arms (pure functions).

    Literal replica of the frozen sampling rule's per-(case, modality) hash
    (``CandidateSampler.seed_of``) pinned at the discriminated channel: the
    sampling arm derives its seeds from here, and the statistics arm
    re-derives them from the artifact filenames -- a mismatch fails the run
    instead of silently comparing across seeds.
    """

    @staticmethod
    def of(case: str) -> int:
        return int(hashlib.sha256(f"{case}|{ANCHOR_MODALITY}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


# ── sample repository ───────────────────────────────────────────────────


class NiftiSampleRepository:
    """Reads the sampling arm's five per-case volumes from the artifact area.

    Layout: ``<samples_dir>/<case>_<arm>_seed<seed>.nii.gz`` (the holdout
    sampler's filename family, generalized to the diagnostic arms). The seed
    suffix is load-bearing: it is checked against the frozen-rule anchor per
    case and for cross-arm consistency, turning the "same seed, token only"
    premise of the job into an enforced invariant. Cases with missing arms
    stay in the report's detail table but out of every aggregate; a directory
    without any job artifact, or any seed violation, is a hard error.
    """

    def __init__(self, samples_dir):
        self._samples_dir = Path(samples_dir)

    def load_cohort(self) -> list[dict]:
        arms_alt = "|".join(sorted(TOKEN_ARMS))
        pattern = re.compile(rf"^(?P<case>.+)_(?P<arm>{arms_alt})_seed(?P<seed>\d+)\.nii\.gz$")
        found: dict[str, dict[str, tuple[int, Path]]] = {}
        for path in sorted(self._samples_dir.glob("*.nii.gz")):
            match = pattern.match(path.name)
            if match is None:
                continue  # not a job artifact; the directory may hold other diagnostics
            found.setdefault(match["case"], {})[match["arm"]] = (int(match["seed"]), path)
        if not found:
            raise DiagnosticError(f"采样产物目录 {self._samples_dir} 未发现任何 <case>_<arm>_seed<N>.nii.gz 工件(#209 五臂命名)")
        cohort = []
        for case, arms in sorted(found.items()):
            anchor = SeedAnchor.of(case)
            seeds = {seed for seed, _ in arms.values()}
            if len(seeds) > 1:
                raise DiagnosticError(f"{case}: 跨臂 seed 不一致 {sorted(seeds)} —— 违反同 seed 严格可比前提(#209)")
            if seeds != {anchor}:
                raise DiagnosticError(
                    f"{case}: 文件 seed {sorted(seeds)} ≠ 冻结采样规则锚 {anchor}(sha256(case|t1c))—— seed 规则被改动,同 seed 换 token 前提失效(#209)"
                )
            arms_out, missing = {}, []
            for arm in ARM_ORDER:
                if arm in arms:
                    arms_out[arm] = self._read(arms[arm][1])
                else:
                    missing.append(arm)
            cohort.append({"case": case, "sub": self._sub_of(case), "seed": anchor, "arms": arms_out, "missing_arms": missing})
        return cohort

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        try:
            array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        except (RuntimeError, OSError) as error:
            raise DiagnosticError(f"采样工件不可读: {path} ({error})") from error
        return array.astype(np.int16, copy=False)

    @staticmethod
    def _sub_of(case: str) -> str | None:
        parts = case.split("-")
        return parts[1] if len(parts) == 4 and parts[0] == "BraTS" else None


# ── per-case readings & attribution ─────────────────────────────────────


class TokenDilution:
    """Per-case five-arm bright-core readings and the token-34 gain share.

    The share compares the candidate arm's top-0.5% mean against the pan-MR
    control arm -- the token the augmentation itself perturbs 34 into -- so a
    near-zero share means the candidate cannot tell 34 from 8. Bands follow
    job A's attribution precedent: control at or above 2/3 of the candidate's
    core reads semantics intact, at or below 1/3 reads dilution dominant,
    otherwise mixed. Bands apply per case (detail table) and to the
    cross-case median (the discriminating read-out).
    """

    METRICS = ("p99", "p99_9", "top05pct_mean", "max")
    DILUTION_DOMINANT_FRACTION = 1 / 3
    SEMANTICS_INTACT_FRACTION = 2 / 3

    def __init__(self, bootstrap_b: int = BOOTSTRAP_B):
        self._bootstrap_b = bootstrap_b

    @staticmethod
    def gain_share(bright_candidate, bright_control):
        """(bright(34) − bright(8)) / bright(34), clamped to [0, 1].

        An all-zero control arm has no bright core at all -- its core is 0,
        so the full candidate core is semantics gain. A candidate arm without
        a bright core has no share to attribute.
        """
        if bright_candidate is None or bright_candidate <= 0:
            return None, "no_bright_core"
        control = 0.0 if bright_control is None else bright_control
        share = max(0.0, min(1.0, (bright_candidate - control) / bright_candidate))
        return share, TokenDilution.classify(share)

    @staticmethod
    def classify(share):
        if share is None:
            return "no_bright_core"
        if share <= TokenDilution.DILUTION_DOMINANT_FRACTION:
            return "dilution_dominant"
        if share >= TokenDilution.SEMANTICS_INTACT_FRACTION:
            return "semantics_intact"
        return "mixed"

    def read_cases(self, cohort: list[dict]) -> list[dict]:
        return [self._read_case(entry) for entry in cohort]

    def _read_case(self, entry: dict) -> dict:
        arms = {arm: (BrightCoreStats.of(entry["arms"][arm]) if arm in entry["arms"] else None) for arm in ARM_ORDER}
        bright = {arm: (stats["top05pct_mean"] if stats is not None else None) for arm, stats in arms.items()}
        share, classification = self.gain_share(bright[CANDIDATE_ARM], bright[CONTROL_ARM])
        return {
            "case": entry["case"],
            "sub": entry["sub"],
            "seed": entry["seed"],
            "arms": arms,
            "gain_share": share,
            "classification": classification,
            "excluded": None if not entry["missing_arms"] else f"missing_arms:{','.join(entry['missing_arms'])}",
        }

    def aggregate(self, cases: list[dict]) -> dict:
        """Cross-case read-out over the five-arm-complete cases only."""
        complete = [case for case in cases if case["excluded"] is None]
        per_arm = {}
        for index, arm in enumerate(ARM_ORDER):
            block = {"token": TOKEN_ARMS[arm]}
            for metric in self.METRICS:
                values = [case["arms"][arm][metric] for case in complete if case["arms"][arm] is not None and case["arms"][arm][metric] is not None]
                seed = DIAGNOSTIC_SEED_BASE + ARM_CI_SLOT_BASE + index if metric == "top05pct_mean" else None
                block[metric] = self._distribution(values, seed)
            per_arm[arm] = block
        contrasts = {}
        for index, (candidate, reference) in enumerate(CONTRAST_PAIRS):
            values = [
                case["arms"][candidate]["top05pct_mean"] - case["arms"][reference]["top05pct_mean"]
                for case in complete
                if case["arms"][candidate] is not None
                and case["arms"][reference] is not None
                and case["arms"][candidate]["top05pct_mean"] is not None
                and case["arms"][reference]["top05pct_mean"] is not None
            ]
            contrasts[f"{candidate}_vs_{reference}"] = {
                "candidate_token": TOKEN_ARMS[candidate],
                "reference_token": TOKEN_ARMS[reference],
                **self._distribution(values, DIAGNOSTIC_SEED_BASE + CONTRAST_CI_SLOT_BASE + index),
            }
        shares = [case["gain_share"] for case in complete if case["gain_share"] is not None]
        share_stats = self._distribution(shares, DIAGNOSTIC_SEED_BASE + SHARE_CI_SLOT)
        median_share = share_stats["median"]
        attribution = {
            "metric": "top05pct_mean",
            "candidate_token": TOKEN_ARMS[CANDIDATE_ARM],
            "control_token": TOKEN_ARMS[CONTROL_ARM],
            "n_cases": share_stats["n_cases"],
            "median_share": median_share,
            "ci90_low": share_stats["ci90_low"],
            "ci90_high": share_stats["ci90_high"],
            "classification": self.classify(median_share),
        }
        return {"per_arm": per_arm, "contrasts": contrasts, "attribution": attribution, "n_excluded": len(cases) - len(complete)}

    def _distribution(self, values, seed: int | None) -> dict:
        """Quantile/mean read-out of one value list (+ diagnostic-seed CI90 on request)."""
        stats = {"median": None, "mean": None, "q05": None, "q95": None, "ci90_low": None, "ci90_high": None, "n_cases": len(values)}
        if not values:
            return stats
        stats.update(
            median=ClusterBootstrap.quantile(values, 0.5),
            mean=sum(values) / len(values),
            q05=ClusterBootstrap.quantile(values, 0.05),
            q95=ClusterBootstrap.quantile(values, 0.95),
        )
        if seed is not None:
            ci = ClusterBootstrap(self._bootstrap_b).ci90([[value] for value in values], seed)
            stats["ci90_low"], stats["ci90_high"] = ci["low"], ci["high"]
        return stats


# ── report ──────────────────────────────────────────────────────────────


class TokenDilutionReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git)."""

    SCHEMA = "token-dilution-diagnostic/1"
    TITLE = "诊断作业 D:同 seed 换 token 采样亮核甄别(模态条件稀释)"

    ARM_LABELS = {"t1n": "t1n(29)", "t1c": "t1c(34)", "t2w": "t2w(30)", "t2f": "t2f(31)", "panmr": "泛 MR(8)"}

    def __init__(self, samples_dir, bootstrap_b: int = BOOTSTRAP_B, run_id: str | None = None, checkpoint: str | None = None):
        self._samples_dir = Path(samples_dir)
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id
        self._checkpoint = checkpoint

    def write(self, cases, aggregate, output_dir):
        payload = {
            "schema": self.SCHEMA,
            "title": self.TITLE,
            "issue": 209,
            "variant": "diagnostic",
            "disclaimer": (
                f"诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#205 作业 D)。bootstrap 种子独立于正式判定链(诊断基 {DIAGNOSTIC_SEED_BASE},作业 D 无挑战带,槽位 300+)。"
            ),
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"samples_dir": str(self._samples_dir), "checkpoint": self._checkpoint},
            "reading_conventions": {
                "arms": "每 case 五臂:t1n(29)/t1c(34)/t2w(30)/t2f(31) + 对照泛 MR(8,增广把 34 扰到的目标);token 表与冻结 MODALITY_TOKENS 一致",
                "same_seed": "seed 锚 = sha256(case|t1c),五臂共用——噪声逐位一致,输出差异全部归因 token 条件;非 t1c 臂的 seed 刻意不取冻结规则在该 modality 的正式 seed(同 seed 对照使然,仅 t1c 臂与冻结产物 seed 一致);统计端从工件文件名复核 seed 锚,不一致即拒绝",
                "pipeline": "冻结 P1 候选 checkpoint + 冻结采样配方(cfg=10、30 步、RFlowScheduler),仅替换条件 token;训练产物零改动",
                "bright_core": "亮核顶部指标取非零体素基底(0 为空气/背景):P99/P99.9(线性 q*(n-1) 规则)/前 0.5% 均值/max;×1000 int16 输出域,>1000 为 >1.0 输出域(作业 E 轴),如实保留不 clip",
                "gain_share": "增益份额 = clamp((亮核(t1c 34) − 亮核(泛 MR 8)) / 亮核(t1c 34), 0, 1),主指标前 0.5% 均值;对照臂全零按 0 计",
                "classification": "份额 ≤ 1/3 → dilution_dominant(34 与 8 无实质差,增广稀释坐实);≥ 2/3 → semantics_intact(34 仍承载增强语义,亮核缺失的嫌疑移出 RC-2);之间 → mixed。分带沿诊断作业 A 的 2/3–1/3 归因先例;跨 case 以份额中位数定类",
            },
            "per_arm": aggregate["per_arm"],
            "contrasts": aggregate["contrasts"],
            "attribution": aggregate["attribution"],
            "n_excluded": aggregate["n_excluded"],
            "per_case": cases,
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "token_dilution_diagnostic.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / "token_dilution_diagnostic.md"
        md_path.write_text(self._markdown(payload))
        return json_path, md_path

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload: dict) -> str:
        conventions = payload["reading_conventions"]
        lines = [
            f"# {payload['title']}",
            "",
            f"**Issue**: [#209](https://github.com/ACautomata/NV-Generate-CTMR/issues/209)(父 #205 作业 D)"
            f" · **run**: `{payload['run_id'] or '未绑定'}`",
            f"**variant: diagnostic —— {payload['disclaimer']}**",
            "",
            "## 读数口径",
            "",
            f"- 输入:采样产物 `{payload['inputs']['samples_dir']}`(冻结候选 checkpoint `{payload['inputs']['checkpoint'] or '未记录'}`,推理零改动)",
            f"- 五臂与同 seed:{conventions['arms']};{conventions['same_seed']}",
            f"- 亮核口径:{conventions['bright_core']}",
            f"- 增益份额:{conventions['gain_share']}",
            f"- 分带:{conventions['classification']}",
            "",
            "## 逐臂亮核统计(非零体素基底)",
            "",
            "| 臂 | token | n | 前 0.5% 均值 median (CI90) | P99 median | P99.9 median | max median |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for arm, block in payload["per_arm"].items():
            top = block["top05pct_mean"]
            lines.append(
                f"| {self.ARM_LABELS[arm]} | {block['token']} | {top['n_cases']} "
                f"| {self._fmt(top['median'])} ({self._fmt(top['ci90_low'])}, {self._fmt(top['ci90_high'])}) "
                f"| {self._fmt(block['p99']['median'])} | {self._fmt(block['p99_9']['median'])} "
                f"| {self._fmt(block['max']['median'])} |"
            )
        lines += [
            "",
            "## token 34 配对差与前 0.5% 均值增益份额",
            "",
            "| 对照 | 候选 token | 参考 token | n | 差 median (CI90) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, contrast in payload["contrasts"].items():
            lines.append(
                f"| {name} | {contrast['candidate_token']} | {contrast['reference_token']} | {contrast['n_cases']} "
                f"| {self._fmt(contrast['median'])} ({self._fmt(contrast['ci90_low'])}, {self._fmt(contrast['ci90_high'])}) |"
            )
        attribution = payload["attribution"]
        lines += [
            "",
            f"**甄别读数**:t1c(34) 对 泛MR(8) 的前 0.5% 均值增益份额中位数 **{self._fmt(attribution['median_share'])}**"
            f"(CI90 {self._fmt(attribution['ci90_low'])}, {self._fmt(attribution['ci90_high'])};n={attribution['n_cases']})"
            f" → **{attribution['classification']}**。",
            "",
            "## 逐 case 明细",
            "",
            "| case | 挑战 | seed | t1n | t1c | t2w | t2f | 泛MR | 增益份额 | 分带 | 排除 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for case in payload["per_case"]:
            arms = [self._fmt(case["arms"][arm]["top05pct_mean"]) if case["arms"][arm] is not None else "n/a" for arm in ARM_ORDER]
            lines.append(
                f"| {case['case']} | {case['sub'] or 'n/a'} | {case['seed']} | "
                + " | ".join(arms)
                + f" | {self._fmt(case['gain_share'])} | {case['classification']} | {case['excluded'] or ''} |"
            )
        lines.append("")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-dir", required=True, help="the sampling arm's five-per-case artifact directory (controlled storage)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the diagnostic report (never git)")
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    parser.add_argument("--run-id", default=None, help="the candidate's L2 terminal-acceptance run id, recorded into the report")
    parser.add_argument("--checkpoint", default=None, help="the frozen candidate checkpoint path, recorded into the report")
    args = parser.parse_args(argv)

    cohort = NiftiSampleRepository(args.samples_dir).load_cohort()
    dilution = TokenDilution(bootstrap_b=args.bootstrap_b)
    cases = dilution.read_cases(cohort)
    aggregate = dilution.aggregate(cases)
    report = TokenDilutionReport(args.samples_dir, bootstrap_b=args.bootstrap_b, run_id=args.run_id, checkpoint=args.checkpoint)
    json_path, md_path = report.write(cases, aggregate, Path(args.output_dir))
    print(f"[OK] {len(cases)} cases ({aggregate['n_excluded']} excluded, variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
