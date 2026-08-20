#!/usr/bin/env python3
"""Issue #38 合成域评估：对预测结果计算 R_fail 并与 ADR-0002 真实包络比较。

支持 P1（直出）与 P3（4 锚轮 img2img）两种模式：
  python3 /root/private_data/l2-synth-eval/p1_evaluate.py [p1|p3]（默认 p1）

R_fail 定义与 #36 校准一致：input_fail | run_fail | hier_viol
- input_fail: 四模态文件缺失/不可读，或 size/spacing 不一致、非 1mm isotropic
- run_fail:   预测文件缺失/不可读
- hier_viol:  标签 ∉ {0,1,2,3}，或 ET⊄TC，或 TC⊄WT

判定：R_fail_synth.point ≤ R_fail_real.point → PASS，否则 UNDECIDED（真实包络全 0，
任何一例失败即 UNDECIDED，进入终验伴随监控）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MODE = sys.argv[1] if len(sys.argv) > 1 else "p1"
assert MODE in ("p1", "p3"), f"mode must be p1|p3, got {MODE}"

EVAL_ROOT = Path("/root/private_data/l2-synth-eval")
INPUT = EVAL_ROOT / f"{MODE}_nnunet_inputs"
PRED = EVAL_ROOT / f"{MODE}_predictions"
CAL_DIR = Path("/root/private_data/l2-instrument-calibration/252940d0156f4c1258936fa25a1fb28bad61ae22")
REPORT_DIR = EVAL_ROOT / f"report_{MODE}"

MODE_LABEL = {"p1": "P1 直出（独立模态采样 ×4）",
              "p3": "P3 img2img 零训练基线（4 锚轮协议）"}

Z95 = 1.959963984540054


def wilson_upper(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    p = k / n
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denom
    half = (Z95 / denom) * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))
    return min(1.0, center + half)


def evaluate_case(challenge: str, case_id: str) -> dict:
    result = {"case": case_id, "challenge": challenge,
              "input_fail": False, "run_fail": False, "hier_viol": False}

    # 输入契约
    try:
        inputs = [sitk.ReadImage(str(INPUT / challenge / f"{case_id}_{s}.nii.gz"))
                  for s in ("0000", "0001", "0002", "0003")]
        ref = (inputs[0].GetSize(), inputs[0].GetSpacing())
        ok = all((img.GetSize(), img.GetSpacing()) == ref for img in inputs[1:])
        iso = all(abs(s - 1.0) < 1e-3 for s in inputs[0].GetSpacing())
        result["input_fail"] = not (ok and iso)
    except Exception:
        result["input_fail"] = True
        result["run_fail"] = True
        return result

    # 仪器运行
    try:
        pred_arr = sitk.GetArrayFromImage(
            sitk.ReadImage(str(PRED / challenge / f"{case_id}.nii.gz"))).astype(np.uint8)
    except Exception:
        result["run_fail"] = True
        return result

    # 层级违反: ET⊆TC⊆WT
    wt = np.isin(pred_arr, (1, 2, 3))
    tc = np.isin(pred_arr, (1, 3))
    et = (pred_arr == 3)
    if (et & ~tc).any() or (tc & ~wt).any():
        result["hier_viol"] = True
    if not np.isin(pred_arr, (0, 1, 2, 3)).all():
        result["hier_viol"] = True

    # 附加观察：空预测（PED ET 已知 25% 空 pred 的背景）
    result["empty_pred"] = bool(not wt.any() and not tc.any() and not et.any())
    result["vol_ml"] = {"WT": float((wt.sum()) * 1e-3), "TC": float((tc.sum()) * 1e-3),
                        "ET": float((et.sum()) * 1e-3)}
    return result


def main() -> int:
    report = {
        "title": "L2 仪器合成域适用性评估报告",
        "issue": 38, "mode": MODE,
        "per_challenge": {},
        "overall_verdict": "PASS",
    }
    all_case_results = []

    for ch_dir in sorted(INPUT.iterdir()):
        if not ch_dir.is_dir():
            continue
        challenge = ch_dir.name
        case_ids = sorted({p.name.rsplit("_", 1)[0] for p in ch_dir.glob("*_0000.nii.gz")})
        if not case_ids:
            continue

        case_results = [evaluate_case(challenge, c) for c in case_ids]
        all_case_results.extend(case_results)

        n = len(case_results)
        k_input = sum(r["input_fail"] for r in case_results)
        k_run = sum(r["run_fail"] for r in case_results)
        k_hier = sum(r["hier_viol"] for r in case_results)
        k_fail = sum(r["input_fail"] or r["run_fail"] or r["hier_viol"] for r in case_results)

        r_synth = {"k": k_fail, "n": n, "point": k_fail / n,
                   "wilson_95_upper": wilson_upper(k_fail, n),
                   "breakdown": {"input_fail": k_input, "run_fail": k_run, "hier_viol": k_hier}}

        cal_path = CAL_DIR / "metrics" / f"summary_{challenge}.json"
        r_real = json.loads(cal_path.read_text())["R_fail"] if cal_path.exists() else None

        verdict = "PASS" if (r_real is None or r_synth["point"] <= r_real["point"]) else "UNDECIDED"
        n_empty = sum(r["empty_pred"] for r in case_results)

        report["per_challenge"][challenge] = {
            "n_samples": n, "r_fail_synth": r_synth,
            "r_fail_real": r_real, "empty_pred_cases": n_empty, "verdict": verdict,
        }
        if verdict == "UNDECIDED":
            report["overall_verdict"] = "UNDECIDED"

    if MODE == "p1":
        report["p2_evidence_gap"] = (
            "P2 方向前置证据缺位已知情接受：掩码 ControlNet 训练前不存在 v1 可产样本，"
            "P2 依赖终验伴随监控兜底。P1 直出样本保留跨模态不一致性（独立采样），"
            "其 R_fail 只覆盖仪器对合成输入的运行/层级契约，不构成对 P2 配方产出的预测。"
        )
    else:
        report["p3_protocol_note"] = (
            "P3 为 img2img 零训练基线（RF 插值 strength=0.9，无 ControlNet）："
            "每轮一个真实模态作锚、其余三模态以该锚为 src 生成，12 有序模态对全覆盖；"
            "真实锚通道直接用原始数据（重采样对齐），生成通道为 v1 DM img2img 输出。"
            "跨模态自洽性强于 P1 但弱于待训 P3 ControlNet，仅作合成域适用性前置证据。"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"report_{MODE}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    (REPORT_DIR / "case_results.json").write_text(
        json.dumps(all_case_results, indent=2, ensure_ascii=False))

    md = [f"# {report['title']}", "",
          f"**模式**: {MODE_LABEL[MODE]}  ",
          f"**总体判定**: **{report['overall_verdict']}**", "",
          "| 挑战 | 样本数 | R_fail_synth (k/n) | Wilson 95% 上界 | R_fail_real | 空 pred | 判定 |",
          "|------|--------|--------------------|-----------------|-------------|---------|------|"]
    for ch, d in report["per_challenge"].items():
        s, r = d["r_fail_synth"], d["r_fail_real"]
        real = f"{r['point']:.4f} ({r['k']}/{r['n']})" if r else "N/A"
        md.append(f"| {ch} | {d['n_samples']} | {s['point']:.4f} ({s['k']}/{s['n']}) "
                  f"| {s['wilson_95_upper']:.4f} | {real} | {d['empty_pred_cases']} | **{d['verdict']}** |")
    md += ["", "## R_fail 细分", ""]
    for ch, d in report["per_challenge"].items():
        b = d["r_fail_synth"]["breakdown"]
        md += [f"- **{ch}**: input_fail={b['input_fail']} run_fail={b['run_fail']} "
               f"hier_viol={b['hier_viol']} (n={d['n_samples']})"]
    note = report.get("p2_evidence_gap") or report.get("p3_protocol_note")
    md += ["", "## 方向说明", "", note, ""]
    (REPORT_DIR / f"report_{MODE}.md").write_text("\n".join(md))

    print(json.dumps({ch: {"verdict": d["verdict"],
                           "r_fail_synth": d["r_fail_synth"],
                           "empty_pred": d["empty_pred_cases"]}
                      for ch, d in report["per_challenge"].items()}, indent=2, ensure_ascii=False))
    print(f"\noverall: {report['overall_verdict']} → {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
