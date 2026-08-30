"""计算 Issue #36 L2 仪器校准的七类误差与预注册误差包络。

输入为 ``calibration_prep`` 冻结的校准集与 3 次独立推理输出；
逐病例 × 区域 × rep 的原始指标写 CSV，预注册统计量（``D_r,low``、
``E_r,*``、``R_fail``、重复性、ET<1 mL 分层）按 docs/calibration/
l2-instrument-calibration-protocol.md §5–§6 的冻结定义计算并写 JSON。

测量逻辑走 canonical ``InstrumentMeasurer``（ADR-0010，#224），长表序列化
经 GT 门控；本模块只留执行侧职责——input_fail/run_fail 站策略、失败占位行、
文件 IO 与 cohort 聚合（聚合喂入 None→nan 回映射）。注册分歧（ADR-0010
决定 3/4 与后果）：CSV 列 ``hier_viol`` 改名 ``case_usable``（校准病例可用性
不是层级违反），Dice 空分母哨兵 ``nan``→``None``，``et_wt_ratio_pred`` 用
ml 体积比（与母版计数比差 ≤ 数 ulp）——三者均不属冻结聚合。聚合 JSON 的
键名（含 breakdown 的 ``hier_viol``）逐字保持，逐字节重跑兑付于 #233。

所有输出均约束于 ``/root/private_data``；CSV 含 subject ID，不入库。
"""

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ctmr.domain.measurement import CALIBRATION_FIELDS, REGIONS, InstrumentMeasurer, WilsonUpper

B = 10_000
GLOBAL_SEED = 20260820
CHALLENGE_SEED_OFFSET = {"GLI": 1, "SSA": 2, "MEN": 3, "METS": 4, "PED": 5}


def read_seg(path: Path) -> np.ndarray:
    """读分割 NIfTI，返回 array[z,y,x] uint8（测量口径假设冻结 1mm 网格，ADR-0008）。"""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8, copy=False)


def measure_case(job: dict) -> list[dict]:
    """一个 (case, rep) 的全部区域测量；执行侧判定与占位行留原地，测量走 canonical。"""
    challenge, case, source, rep = job["challenge"], job["case"], job["source"], job["rep"]
    gt_path = job["gt_dir"] / f"{case}.nii.gz"
    pred_path = job["pred_dir"] / f"{case}.nii.gz"

    input_fail = run_fail = False
    gt_arr = pred_arr = None

    try:
        inputs = [sitk.ReadImage(str(job["inputs_dir"] / f"{case}_{s}.nii.gz")) for s in ("0000", "0001", "0002", "0003")]
        gt_img = sitk.ReadImage(str(gt_path))
        reference = (inputs[0].GetSize(), inputs[0].GetSpacing(), inputs[0].GetOrigin())
        consistent = all((img.GetSize(), img.GetSpacing(), img.GetOrigin()) == reference for img in inputs[1:] + [gt_img])
        isotropic = all(abs(s - 1.0) < 1e-3 for s in inputs[0].GetSpacing())
        input_fail = not (consistent and isotropic)

        gt_arr = read_seg(gt_path)
        pred_arr = read_seg(pred_path)
        if pred_arr.shape != gt_arr.shape:
            run_fail = True
    except (RuntimeError, OSError):  # sitk 读失败 = 输出/输入缺失或损坏
        input_fail = (
            input_fail
            or not all((job["inputs_dir"] / f"{case}_{s}.nii.gz").is_file() for s in ("0000", "0001", "0002", "0003"))
            or not gt_path.is_file()
        )
        run_fail = True

    if gt_arr is None or pred_arr is None:
        # 失败占位行：计 R_fail 分母，各量为空（调用方职责，canonical 不掺和）。
        # case_usable 留空（未测）而非 False——母版失败行的 hier_viol 恒 False，
        # 从不进 breakdown 的 hier_viol 分量（ADR-0002 语义，聚合见 main）。
        rows = []
        for region in REGIONS:
            row = dict.fromkeys(CALIBRATION_FIELDS, None)
            row.update(challenge=challenge, case=case, source=source, rep=rep, region=region, input_fail=input_fail, run_fail=run_fail)
            rows.append(row)
        return rows

    measurement = InstrumentMeasurer().measure(pred_arr, gt=gt_arr)
    return measurement.to_long_rows(challenge=challenge, case=case, source=source, rep=rep, input_fail=input_fail, run_fail=run_fail)


def bootstrap_envelope(values: np.ndarray, quantile: float, upper: bool, seed: int) -> tuple[float, float, float]:
    """点估计、单侧 95% bootstrap 界（percentile 法，B=10,000，病例级重采样）。"""
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return math.nan, math.nan, 0
    point = float(np.quantile(values, quantile))
    rng = np.random.Generator(np.random.PCG64(seed))
    idx = rng.integers(0, n, size=(B, n))
    resampled = values[idx]
    boot_stats = np.quantile(resampled, quantile, axis=1)
    bound = float(np.quantile(boot_stats, 0.95 if upper else 0.05))
    return point, bound, n


def summarize_region(region_rows: list[dict], seed: int) -> dict:
    def col(name):
        return np.array([r[name] if r[name] is not None else math.nan for r in region_rows], dtype=float)

    dice = col("dice")
    rel_err = col("rel_vol_err")
    hd95 = col("hd95_mm")
    centroid = col("centroid_mm")
    vol_gt = col("vol_gt_ml")
    bias = col("signed_bias_ml")
    d_point, d_low, _ = bootstrap_envelope(dice, 0.05, upper=False, seed=seed)
    v_point, v_up, _ = bootstrap_envelope(rel_err, 0.95, upper=True, seed=seed)
    h_point, h_up, h_n = bootstrap_envelope(hd95, 0.95, upper=True, seed=seed)
    c_point, c_up, c_n = bootstrap_envelope(centroid, 0.95, upper=True, seed=seed)
    return {
        "dice_median": float(np.nanmedian(dice)),
        "dice_iqr": [float(np.nanquantile(dice, 0.25)), float(np.nanquantile(dice, 0.75))],
        "D_r_low": d_low,
        "dice_q5_point": d_point,
        "rel_vol_err_p95_point": v_point,
        "E_r_vol": v_up,
        "vol_bias_median_ml": float(np.nanmedian(bias)),
        "vol_gt_median_ml": float(np.nanmedian(vol_gt)),
        "E_r_hd95": h_up,
        "hd95_p95_point": h_point,
        "n_hd95_defined": h_n,
        "E_r_centroid": c_up,
        "centroid_p95_point": c_point,
        "n_centroid_defined": c_n,
        "n_excluded_empty_pred": int(np.isnan(hd95).sum()),
        "sensitivity_median": float(np.nanmedian(col("sensitivity"))),
        "precision_median": float(np.nanmedian(col("precision"))),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--challenge", required=True, choices=list(CHALLENGE_SEED_OFFSET))
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    challenge = args.challenge
    seed = GLOBAL_SEED + CHALLENGE_SEED_OFFSET[challenge]
    root = args.calibration_root
    inputs_dir = root / "inputs" / challenge
    gt_dir = root / "gt" / challenge
    metrics_dir = root / "metrics" / challenge
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((root / "protocol" / f"calibration_cases_{challenge}.json").read_text())
    jobs, all_rows = [], []
    for rep in args.reps:
        pred_dir = root / "predictions" / challenge / f"rep{rep}"
        for entry in manifest["cases"]:
            jobs.append(
                {
                    "challenge": challenge,
                    "case": entry["case"],
                    "source": entry["source"],
                    "rep": rep,
                    "inputs_dir": inputs_dir,
                    "gt_dir": gt_dir,
                    "pred_dir": pred_dir,
                }
            )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(measure_case, jobs, chunksize=4):
            all_rows.extend(rows)

    for rep in args.reps:
        rep_rows = [r for r in all_rows if r["rep"] == rep]
        out = metrics_dir / f"per_case_{challenge}_rep{rep}.csv"
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CALIBRATION_FIELDS)
            writer.writeheader()
            writer.writerows(rep_rows)

    # 重复性：同一病例同一区域跨 rep 的最大极差（max − min）
    repeatability = []
    by_case_region: dict[tuple, list[dict]] = {}
    for r in all_rows:
        by_case_region.setdefault((r["case"], r["source"], r["region"]), []).append(r)
    for (case, source, region), rows in sorted(by_case_region.items()):

        def rng_of(name):
            vals = [r[name] for r in rows if r[name] is not None and not math.isnan(r[name])]
            return max(vals) - min(vals) if len(vals) == len(rows) and vals else math.nan

        repeatability.append(
            {
                "case": case,
                "source": source,
                "region": region,
                "vol_ml_range": rng_of("vol_pred_ml"),
                "dice_range": rng_of("dice"),
                "centroid_mm_range": rng_of("centroid_mm"),
                "hd95_mm_range": rng_of("hd95_mm"),
            }
        )
    with (metrics_dir / f"repeatability_{challenge}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "source", "region", "vol_ml_range", "dice_range", "centroid_mm_range", "hd95_mm_range"])
        writer.writeheader()
        writer.writerows(repeatability)

    def rep_p95(name):
        vals = [r[name] for r in repeatability if r["region"] == "WT" and r[name] is not None and not math.isnan(r[name])]
        return float(np.quantile(vals, 0.95)) if vals else math.nan

    # R_fail 与 R_miss 按（病例 × rep）；R_miss 只在非失败观测上计（协议 §5：
    # 空 pred 是测量结果，文件级失败归 R_fail）。聚合 JSON 键名逐字保持
    # （breakdown 的 ``hier_viol``），数据源是改名后的 ``case_usable`` 列。
    case_level = {}
    for r in all_rows:
        key = (r["case"], r["rep"])
        c = case_level.setdefault(key, {"input_fail": False, "run_fail": False, "hier_viol": False, "detected": True})
        c["input_fail"] |= bool(r["input_fail"])
        c["run_fail"] |= bool(r["run_fail"])
        c["hier_viol"] |= r["case_usable"] is False  # 母版 hier_viol 语义 = 成功测量且不可用；失败行（None）不计（ADR-0002 语义）
        c["detected"] &= bool(r["detected"])
    n_obs = len(case_level)
    k_fail = sum(c["input_fail"] or c["run_fail"] or c["hier_viol"] for c in case_level.values())
    k_breakdown = {name: sum(c[name] for c in case_level.values()) for name in ("input_fail", "run_fail", "hier_viol")}
    k_miss = sum(not c["detected"] and not (c["input_fail"] or c["run_fail"] or c["hier_viol"]) for c in case_level.values())

    et_gt_by_case = {r["case"]: r["vol_gt_ml"] for r in all_rows if r["region"] == "ET"}

    summary = {
        "challenge": challenge,
        "n_cases": len(manifest["cases"]),
        "n_reps": len(args.reps),
        "n_observations": n_obs,
        "bootstrap": {
            "B": B,
            "method": "病例级重采样 percentile 法（协议 §6）",
            "seed": GLOBAL_SEED,
            "seed_offset": CHALLENGE_SEED_OFFSET[challenge],
        },
        "R_fail": {
            "k": k_fail,
            "n": n_obs,
            "point": k_fail / n_obs if n_obs else math.nan,
            "wilson_95_upper": WilsonUpper.of(k_fail, n_obs),
            "breakdown": k_breakdown,
        },
        "R_miss": {"k": k_miss, "n": n_obs, "point": k_miss / n_obs if n_obs else math.nan},
        "repeatability_p95": {
            "vol_ml_range": rep_p95("vol_ml_range"),
            "dice_range": rep_p95("dice_range"),
            "centroid_mm_range": rep_p95("centroid_mm_range"),
            "hd95_mm_range": rep_p95("hd95_mm_range"),
        },
        "per_region": {},
        "et_lt_1ml_stratum": {"n_cases": sum(v is not None and not math.isnan(v) and v < 1.0 for v in et_gt_by_case.values()), "per_region": {}},
    }
    for region in REGIONS:
        region_rows = [r for r in all_rows if r["region"] == region]
        summary["per_region"][region] = summarize_region(region_rows, seed=seed)
        stratum_cases = {c for c, v in et_gt_by_case.items() if v is not None and not math.isnan(v) and v < 1.0}
        stratum_rows = [r for r in region_rows if r["case"] in stratum_cases]
        summary["et_lt_1ml_stratum"]["per_region"][region] = summarize_region(stratum_rows, seed=seed) if stratum_rows else None

    protocol_sums = root / "protocol" / "SHA256SUMS"
    summary["protocol_sha256s_head"] = protocol_sums.read_text().splitlines()[0] if protocol_sums.is_file() else None
    out_path = root / "metrics" / f"summary_{challenge}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "challenge": challenge,
                "R_fail": summary["R_fail"],
                "R_miss": summary["R_miss"],
                "D_r_low": {r: summary["per_region"][r]["D_r_low"] for r in REGIONS},
                "E_r_vol": {r: summary["per_region"][r]["E_r_vol"] for r in REGIONS},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
