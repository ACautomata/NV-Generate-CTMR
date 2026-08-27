"""计算 Issue #36 L2 仪器校准的七类误差与预注册误差包络。

输入为 ``nnunet_l2_calibration_prep.py`` 冻结的校准集与 3 次独立推理输出；
逐病例 × 区域 × rep 的原始指标写 CSV，预注册统计量（``D_r,low``、
``E_r,*``、``R_fail``、重复性、ET<1 mL 分层）按 docs/calibration/
l2-instrument-calibration-protocol.md §5–§6 的冻结定义计算并写 JSON。

所有输出均约束于 ``/root/private_data``；CSV 含 subject ID，不入库。
"""

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
B = 10_000
GLOBAL_SEED = 20260820
CHALLENGE_SEED_OFFSET = {"GLI": 1, "SSA": 2, "MEN": 3, "METS": 4, "PED": 5}
Z95 = 1.959963984540054

CSV_FIELDS = [
    "challenge",
    "case",
    "source",
    "rep",
    "region",
    "input_fail",
    "run_fail",
    "hier_viol",
    "detected",
    "dice",
    "sensitivity",
    "precision",
    "vol_gt_ml",
    "vol_pred_ml",
    "signed_bias_ml",
    "abs_err_ml",
    "rel_vol_err",
    "et_wt_ratio_gt",
    "et_wt_ratio_pred",
    "hd95_mm",
    "centroid_mm",
    "n_comp_gt",
    "n_comp_pred",
    "n_fp_comp",
]


def read_seg(path: Path):
    """读分割 NIfTI，返回 (array[z,y,x] uint8, spacing[z,y,x] mm)。"""
    image = sitk.ReadImage(str(path))
    spacing_zyx = tuple(reversed(image.GetSpacing()))
    array = sitk.GetArrayFromImage(image)
    return array.astype(np.uint8, copy=False), spacing_zyx


def region_mask(array: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    return np.isin(array, labels)


def dice_of(gt: np.ndarray, pred: np.ndarray) -> float:
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return math.nan  # GT 空（BraTS 不发生）与 pred 空的协议在调用处处理
    return float(2 * np.logical_and(gt, pred).sum() / denom)


def hd95_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx) -> float:
    """HD95 = max(p95(d_gt→pred), p95(d_pred→gt))；表面 = mask XOR binary_erosion。"""
    gt_surf = gt ^ ndimage.binary_erosion(gt)
    pred_surf = pred ^ ndimage.binary_erosion(pred)
    if gt_surf.sum() == 0 or pred_surf.sum() == 0:
        return math.nan
    dist_to_gt = ndimage.distance_transform_edt(~gt, sampling=spacing_zyx)
    dist_to_pred = ndimage.distance_transform_edt(~pred, sampling=spacing_zyx)
    d_gt_to_pred = dist_to_pred[gt_surf]
    d_pred_to_gt = dist_to_gt[pred_surf]
    return float(max(np.quantile(d_gt_to_pred, 0.95), np.quantile(d_pred_to_gt, 0.95)))


def centroid_distance_mm(gt: np.ndarray, pred: np.ndarray, spacing_zyx) -> float:
    if gt.sum() == 0 or pred.sum() == 0:
        return math.nan
    c_gt = np.array(ndimage.center_of_mass(gt))
    c_pred = np.array(ndimage.center_of_mass(pred))
    return float(np.linalg.norm((c_gt - c_pred) * np.array(spacing_zyx)))


def component_stats(gt_wt: np.ndarray, pred_wt: np.ndarray):
    """(GT 组件数, pred 组件数, 与 GT WT 零重叠的 pred 组件数)；26 连通。"""
    gt_labels, n_gt = cc3d.connected_components(gt_wt, connectivity=26, return_N=True)
    pred_labels, n_pred = cc3d.connected_components(pred_wt, connectivity=26, return_N=True)
    n_fp = 0
    if n_gt > 0 and n_pred > 0:
        overlap = np.unique(pred_labels[gt_labels > 0])
        n_fp = n_pred - (len(overlap) - (1 if 0 in overlap else 0))
    return int(n_gt), int(n_pred), int(n_fp)


def measure_case(job: dict) -> list[dict]:
    """一个 (case, rep) 的全部区域测量；返回长表行（每区域一行）。"""
    challenge, case, source, rep = job["challenge"], job["case"], job["source"], job["rep"]
    gt_path = job["gt_dir"] / f"{case}.nii.gz"
    pred_path = job["pred_dir"] / f"{case}.nii.gz"

    input_fail = run_fail = hier_viol = False
    gt_arr = pred_arr = None
    spacing_zyx = None

    try:
        inputs = [sitk.ReadImage(str(job["inputs_dir"] / f"{case}_{s}.nii.gz")) for s in ("0000", "0001", "0002", "0003")]
        gt_img = sitk.ReadImage(str(gt_path))
        reference = (inputs[0].GetSize(), inputs[0].GetSpacing(), inputs[0].GetOrigin())
        consistent = all((img.GetSize(), img.GetSpacing(), img.GetOrigin()) == reference for img in inputs[1:] + [gt_img])
        isotropic = all(abs(s - 1.0) < 1e-3 for s in inputs[0].GetSpacing())
        input_fail = not (consistent and isotropic)

        gt_arr, spacing_zyx = read_seg(gt_path)
        pred_arr, _ = read_seg(pred_path)
        if pred_arr.shape != gt_arr.shape:
            run_fail = True
    except (RuntimeError, OSError):  # sitk 读失败 = 输出/输入缺失或损坏
        input_fail = (
            input_fail
            or not all((job["inputs_dir"] / f"{case}_{s}.nii.gz").is_file() for s in ("0000", "0001", "0002", "0003"))
            or not gt_path.is_file()
        )
        run_fail = True

    rows = []
    for region, labels in REGIONS.items():
        row = dict.fromkeys(CSV_FIELDS, None)
        row.update(
            challenge=challenge, case=case, source=source, rep=rep, region=region, input_fail=input_fail, run_fail=run_fail, hier_viol=hier_viol
        )
        if gt_arr is None or pred_arr is None:
            rows.append(row)  # 失败占位行：计 R_fail 分母，各量为空
            continue
        gt_mask = region_mask(gt_arr, labels)
        pred_mask = region_mask(pred_arr, labels)
        vol_gt = float(gt_mask.sum()) * 0.001
        vol_pred = float(pred_mask.sum()) * 0.001
        intersection = int(np.logical_and(gt_mask, pred_mask).sum())

        row["detected"] = bool(region_mask(pred_arr, REGIONS["WT"]).sum() > 0)
        row["dice"] = 0.0 if pred_mask.sum() == 0 and gt_mask.sum() > 0 else dice_of(gt_mask, pred_mask)
        row["sensitivity"] = 0.0 if pred_mask.sum() == 0 else (intersection / float(gt_mask.sum()) if gt_mask.sum() > 0 else math.nan)
        row["precision"] = 0.0 if pred_mask.sum() == 0 else (intersection / float(pred_mask.sum()) if intersection > 0 else 0.0)
        row["vol_gt_ml"], row["vol_pred_ml"] = vol_gt, vol_pred
        row["signed_bias_ml"] = vol_pred - vol_gt
        row["abs_err_ml"] = abs(vol_pred - vol_gt)
        row["rel_vol_err"] = abs(vol_pred - vol_gt) / vol_gt if vol_gt > 0 else math.nan
        row["hd95_mm"] = hd95_mm(gt_mask, pred_mask, spacing_zyx)
        row["centroid_mm"] = centroid_distance_mm(gt_mask, pred_mask, spacing_zyx)
        n_gt, n_pred, n_fp = component_stats(gt_mask, pred_mask)
        row["n_comp_gt"], row["n_comp_pred"] = n_gt, n_pred
        row["n_fp_comp"] = n_fp if region == "WT" else None  # 假阳性病灶组件只对整瘤定义
        if region == "WT":  # 值域/GT 空防御检查（协议 §5 层级行）
            row["hier_viol"] = bool(not np.isin(gt_arr, (0, 1, 2, 3)).all() or not np.isin(pred_arr, (0, 1, 2, 3)).all() or gt_mask.sum() == 0)
        rows.append(row)

    # ET/WT 比值逐病例仅算一次（region 循环外补）
    if gt_arr is not None and pred_arr is not None:
        et_gt = float(region_mask(gt_arr, REGIONS["ET"]).sum())
        wt_gt = float(region_mask(gt_arr, REGIONS["WT"]).sum())
        et_pred = float(region_mask(pred_arr, REGIONS["ET"]).sum())
        wt_pred = float(region_mask(pred_arr, REGIONS["WT"]).sum())
        ratio_gt = et_gt / wt_gt if wt_gt > 0 else math.nan
        ratio_pred = et_pred / wt_pred if wt_pred > 0 else math.nan
        for row in rows:
            row["et_wt_ratio_gt"], row["et_wt_ratio_pred"] = ratio_gt, ratio_pred
    return rows


def wilson_upper(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    p = k / n
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denom
    half = (Z95 / denom) * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))
    return min(1.0, center + half)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--challenge", required=True, choices=list(CHALLENGE_SEED_OFFSET))
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

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
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
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
    # 空 pred 是测量结果，文件级失败归 R_fail）
    case_level = {}
    for r in all_rows:
        key = (r["case"], r["rep"])
        c = case_level.setdefault(key, {"input_fail": False, "run_fail": False, "hier_viol": False, "detected": True})
        c["input_fail"] |= bool(r["input_fail"])
        c["run_fail"] |= bool(r["run_fail"])
        c["hier_viol"] |= bool(r["hier_viol"])
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
            "wilson_95_upper": wilson_upper(k_fail, n_obs),
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
