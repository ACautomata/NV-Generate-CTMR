#!/usr/bin/env python3
"""Issue #38 合成域评估——sugon 端自包含执行脚本。

独立于仓库 module 结构，直接在 sugon 上运行。
复制到 sugon: /root/private_data/l2-synth-eval/run_eval.py

用法:
  python3 /root/private_data/l2-synth-eval/run_eval.py --mode p1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

# ── sugon 固定路径 ──────────────────────────────────────────────────────

EVAL_ROOT = Path("/root/private_data/l2-synth-eval")
NNUNET_ROOT = Path("/root/private_data/brats2023_nnunet")
CAL_DIR = Path("/root/private_data/l2-instrument-calibration/252940d0156f4c1258936fa25a1fb28bad61ae22")
REPO_DIR = Path("/root/private_data/nv-dcu-smoke/NV-Generate-CTMR")
DM_MODEL = Path("/root/private_data/nv-dcu-smoke/models_trained/diff_unet_3d_rflow-mr-brain_v1.pt")
AE_MODEL = Path("/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/autoencoder_v1.pt")

# 训练结果路径（按 commit 分）
RESULTS_52667 = Path("/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results")
RESULTS_SSA = Path("/root/private_data/nnUNet_results/Dataset502_BraTS2023SSA")

# BraTS 模态 → v1 DM modality label
MODALITY_LABELS = {"t1n": 9, "t1c": 17, "t2w": 10, "t2f": 11}
# nnU-Net 通道
NNUNET_CHANNELS = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}
# BraTS 标签
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

# 每挑战评估样本数
SAMPLES = {"GLI": 20, "SSA": 14, "MEN": 20, "METS": 20, "PED": 14}

# v1 DM 输出参数
V1_SIZE = (256, 256, 128)
V1_SPACING = (0.94, 0.94, 1.36)
# nnU-Net 输入
NN_SIZE = (240, 240, 155)
NN_SPACING = (1.0, 1.0, 1.0)

DATASET_IDS = {"GLI": 501, "SSA": 502, "MEN": 503, "METS": 504, "PED": 505}
EXPECTED_COUNTS = {
    "GLI": {"dev": 125, "fold_val": 175},
    "SSA": {"dev": 6, "fold_val": 8},
    "MEN": {"dev": 100, "fold_val": 140},
    "METS": {"dev": 24, "fold_val": 33},
    "PED": {"dev": 10, "fold_val": 14},
}


# ── Step 0: 选择病例 ───────────────────────────────────────────────────

def select_cases(challenge: str, n: int) -> list[dict]:
    """从 nnU-Net 数据集中选取评估病例。"""
    manifest_path = NNUNET_ROOT / "splits/split_manifest_brats2023-rflow-v1.json"
    manifest = json.loads(manifest_path.read_text())
    dev = sorted(manifest["challenges"][challenge]["cases"]["dev"])
    fv_path = NNUNET_ROOT / f"splits/fold0_val_cases_{challenge}.txt"
    fv = sorted(fv_path.read_text().split()) if fv_path.exists() else []

    selected = dev[:n]
    remaining = n - len(selected)
    if remaining > 0:
        selected.extend(fv[:remaining])

    raw_dir = NNUNET_ROOT / f"Dataset{DATASET_IDS[challenge]:03d}_BraTS2023{challenge}"
    return [{"case_id": c, "challenge": challenge, "images_dir": str(raw_dir)} for c in selected]


def cmd_create_lists(args):
    """创建全部挑战的病例列表。"""
    output_dir = EVAL_ROOT / "case_lists"
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("p1", "p3"):
        cases = []
        for ch, n in SAMPLES.items():
            cases.extend(select_cases(ch, n))
        path = output_dir / f"{mode}_cases.json"
        path.write_text(json.dumps(cases, indent=2))
        print(f"[OK] {mode}: {len(cases)} cases → {path}")


# ── Step 1: 生成 v1 DM 样本 ────────────────────────────────────────────

def run_v1_dm_inference(
    modality_label: int,
    output_path: Path,
    seed: int,
    num_steps: int = 30,
) -> None:
    """调用 v1 DM 推理生成一个模态。"""
    config = {
        "diffusion_unet_inference": {
            "dim": list(V1_SIZE),
            "spacing": list(V1_SPACING),
            "top_region_index": [0, 1, 0, 0],
            "bottom_region_index": [0, 0, 1, 0],
            "random_seed": seed,
            "num_inference_steps": num_steps,
            "modality": modality_label,
            "cfg_guidance_scale": 2,
        }
    }

    work_dir = output_path.parent
    config_path = work_dir / "infer_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # 构造 diff_model_infer 需要的环境配置
    env_config = REPO_DIR / "configs/environment_maisi_diff_model_rflow-mr-brain.json"
    model_def = REPO_DIR / "configs/config_maisi_diff_model_rflow-mr-brain.json"

    cmd = [
        sys.executable, "-m", "scripts.diff_model_infer",
        "-e", str(env_config),
        "-c", str(config_path),
        "-t", str(model_def),
        "-g", "1",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_DIR)

    print(f"  [GEN] modality={modality_label} → {output_path.name}")
    result = subprocess.run(
        cmd, cwd=str(REPO_DIR), env=env,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  [ERROR] modality {modality_label} failed:\n{result.stderr[-500:]}")
        raise RuntimeError(f"v1 DM inference failed for modality {modality_label}")

    # diff_model_infer 输出到 output_dir，需要移动到目标位置
    # 查找生成的文件
    output_dir = REPO_DIR / "output"
    generated = list(output_dir.glob(f"*modality{modality_label}*.nii.gz"))
    if generated:
        import shutil
        shutil.move(str(generated[0]), str(output_path))
    else:
        raise FileNotFoundError(f"no output found for modality {modality_label}")


def generate_p1_case(case: dict, output_dir: Path, seed: int) -> dict:
    """P1 式：为一个病例独立生成四个模态。"""
    case_dir = output_dir / case["challenge"] / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    modality_paths = {}
    for mod_name, mod_label in MODALITY_LABELS.items():
        out = case_dir / f"{mod_name}.nii.gz"
        if out.exists():
            modality_paths[mod_name] = str(out)
            continue
        run_v1_dm_inference(mod_label, out, seed)
        modality_paths[mod_name] = str(out)

    return {
        "case_id": case["case_id"],
        "challenge": case["challenge"],
        "mode": "p1",
        "modalities": modality_paths,
    }


def cmd_generate(args):
    """生成 v1 DM 直出样本。"""
    case_list = json.loads((EVAL_ROOT / "case_lists" / f"{args.mode}_cases.json").read_text())
    output_dir = EVAL_ROOT / f"{args.mode}_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, case in enumerate(case_list):
        print(f"\n[{i+1}/{len(case_list)}] {case['challenge']}/{case['case_id']}")
        if args.mode == "p1":
            r = generate_p1_case(case, output_dir, args.seed + i)
            results.append(r)
        else:
            print("  [SKIP] P3 img2img 需要额外实现")
            continue

    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Generated {len(results)} cases → {manifest_path}")


# ── Step 2: 组装 nnU-Net 输入 ──────────────────────────────────────────

def resample_to_1mm(img: sitk.Image) -> sitk.Image:
    """重采样到 1mm isotropic。"""
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [int(round(s * sp / 1.0)) for s, sp in zip(orig_size, orig_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(NN_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(img.GetPixelIDValue()))
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(img)


def crop_or_pad(arr: np.ndarray, target: tuple) -> np.ndarray:
    """居中裁剪或零填充。"""
    result = np.zeros(target, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for s, t in zip(arr.shape, target):
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            start = (t - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(start, start + s))
    result[tuple(dst_slices)] = arr[tuple(src_slices)]
    return result


def prep_one_case(entry: dict, sample_dir: Path, output_dir: Path) -> None:
    """组装一个病例的 nnU-Net 输入。"""
    challenge = entry["challenge"]
    case_id = entry["case_id"]
    case_input_dir = output_dir / challenge
    case_input_dir.mkdir(parents=True, exist_ok=True)

    modality_paths = entry["modalities"]

    for suffix, mod_name in NNUNET_CHANNELS.items():
        dst = case_input_dir / f"{case_id}_{suffix}.nii.gz"
        if dst.exists():
            continue
        src = Path(modality_paths[mod_name])
        img = sitk.ReadImage(str(src))
        resampled = resample_to_1mm(img)
        arr = sitk.GetArrayFromImage(resampled)
        cropped = crop_or_pad(arr, NN_SIZE)
        out_img = sitk.GetImageFromArray(cropped)
        out_img.SetSpacing(NN_SPACING)
        sitk.WriteImage(out_img, str(dst))


def cmd_prep_inputs(args):
    """组装 nnU-Net 输入。"""
    sample_dir = EVAL_ROOT / f"{args.mode}_samples"
    output_dir = EVAL_ROOT / f"{args.mode}_nnunet_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((sample_dir / "generation_manifest.json").read_text())
    for i, entry in enumerate(manifest):
        print(f"[{i+1}/{len(manifest)}] Prepping {entry['challenge']}/{entry['case_id']}")
        prep_one_case(entry, sample_dir, output_dir)

    print(f"[OK] Prepared {len(manifest)} cases → {output_dir}")


# ── Step 3: 仪器推理 ───────────────────────────────────────────────────

def cmd_predict(args):
    """运行冻结仪器推理。"""
    input_dir = EVAL_ROOT / f"{args.mode}_nnunet_inputs"
    pred_base = EVAL_ROOT / f"{args.mode}_predictions"
    pred_base.mkdir(parents=True, exist_ok=True)

    for challenge in SAMPLES:
        case_input_dir = input_dir / challenge
        if not case_input_dir.is_dir():
            print(f"[SKIP] {challenge}: no input dir")
            continue

        pred_dir = pred_base / challenge
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 需要构建临时 nnU-Net dataset 格式
        # predict_from_raw_data 需要 DatasetXXXX 格式输入目录
        dataset_id = DATASET_IDS[challenge]
        dataset_name = f"Dataset{dataset_id:03d}_BraTS2023{challenge}"

        # 创建临时 dataset 目录（符号链接到输入）
        tmp_dataset = EVAL_ROOT / f"_tmp_nnunet_dataset/{dataset_name}"
        tmp_dataset.mkdir(parents=True, exist_ok=True)
        # 链接输入文件
        for f in case_input_dir.glob(f"*_{challenge}*.nii.gz"):
            # nnU-Net 期望 <case>_0000.nii.gz 格式
            pass
        # 直接把 case_input_dir 内容链接过去
        for f in case_input_dir.iterdir():
            link = tmp_dataset / f.name
            if not link.exists():
                link.symlink_to(f.resolve())

        # 设置环境
        env = os.environ.copy()
        env["nnUNet_raw"] = str(NNUNET_ROOT)
        env["nnUNet_preprocessed"] = str(NNUNET_ROOT.parent / "nnUNet_preprocessed")

        # 训练结果位置
        if challenge == "SSA":
            results_dir = RESULTS_SSA.parent
        else:
            results_dir = RESULTS_52667.parent

        env["nnUNet_results"] = str(results_dir)

        extra = ["-p", "nnUNetPlans_SSA_bs16_v1"] if challenge == "SSA" else []

        cmd = [
            sys.executable, "-m", "scripts.l2_calibration_predict_entry",
            "-i", str(tmp_dataset),
            "-o", str(pred_dir),
            "-d", str(dataset_id),
            "-c", "3d_fullres",
            "-f", "0",
            "-tr", "nnUNetTrainer250Epochs",
            "--disable_tta", "False",
        ] + extra

        print(f"[PREDICT] {challenge} ({len(list(case_input_dir.glob('*_0000.nii.gz')))} cases)...")
        # 逐病例推理（nnUNet predict_from_raw_data 会自动找 checkpoint）
        result = subprocess.run(
            cmd, cwd=str(REPO_DIR), env=env,
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"  [WARN] {challenge} prediction failed: {result.stderr[-300:]}")
        else:
            print(f"  [OK] {challenge} predictions → {pred_dir}")


# ── Step 4: 评估 ───────────────────────────────────────────────────────

Z95 = 1.959963984540054

def wilson_upper(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    p = k / n
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2 * n)) / denom
    half = (Z95 / denom) * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))
    return min(1.0, center + half)


def evaluate_case(case_id: str, challenge: str, input_dir: Path, pred_dir: Path) -> dict:
    """评估一个合成病例。"""
    result = {
        "case": case_id, "challenge": challenge,
        "input_fail": False, "run_fail": False, "hier_viol": False,
    }

    # 输入契约检查
    try:
        inputs = [sitk.ReadImage(str(input_dir / f"{case_id}_{s}.nii.gz"))
                  for s in ("0000", "0001", "0002", "0003")]
        ref = (inputs[0].GetSize(), inputs[0].GetSpacing())
        ok = all((img.GetSize(), img.GetSpacing()) == ref for img in inputs[1:])
        iso = all(abs(s - 1.0) < 1e-3 for s in inputs[0].GetSpacing())
        result["input_fail"] = not (ok and iso)
    except Exception:
        result["input_fail"] = True
        result["run_fail"] = True
        return result

    # 读预测
    pred_path = pred_dir / f"{case_id}.nii.gz"
    try:
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.uint8)
    except Exception:
        result["run_fail"] = True
        return result

    # 层级违反: ET⊆TC⊆WT
    wt = np.isin(pred_arr, (1, 2, 3))
    tc = np.isin(pred_arr, (1, 3))
    et = (pred_arr == 3)
    if et.sum() > 0 and tc.sum() > 0:
        if (et & ~tc).sum() > 0:
            result["hier_viol"] = True
    if tc.sum() > 0 and wt.sum() > 0:
        if (tc & ~wt).sum() > 0:
            result["hier_viol"] = True
    if not np.isin(pred_arr, (0, 1, 2, 3)).all():
        result["hier_viol"] = True

    return result


def cmd_evaluate(args):
    """计算指标并生成报告。"""
    mode = args.mode
    sample_dir = EVAL_ROOT / f"{mode}_samples"
    input_dir = EVAL_ROOT / f"{mode}_nnunet_inputs"
    pred_dir = EVAL_ROOT / f"{mode}_predictions"
    output_dir = EVAL_ROOT / f"report_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((sample_dir / "generation_manifest.json").read_text())
    by_challenge = {}
    for entry in manifest:
        by_challenge.setdefault(entry["challenge"], []).append(entry)

    report = {
        "title": "L2 仪器合成域适用性评估报告",
        "issue": 38, "mode": mode,
        "per_challenge": {},
        "overall_verdict": "PASS",
    }

    for challenge, entries in by_challenge.items():
        case_results = []
        for entry in entries:
            r = evaluate_case(
                entry["case_id"], challenge,
                input_dir / challenge, pred_dir / challenge,
            )
            case_results.append(r)

        n_obs = len(case_results)
        k_input = sum(r["input_fail"] for r in case_results)
        k_run = sum(r["run_fail"] for r in case_results)
        k_hier = sum(r["hier_viol"] for r in case_results)
        k_fail = sum(r["input_fail"] or r["run_fail"] or r["hier_viol"] for r in case_results)

        r_fail_synth = {
            "k": k_fail, "n": n_obs,
            "point": k_fail / n_obs if n_obs else 0,
            "wilson_95_upper": wilson_upper(k_fail, n_obs),
            "breakdown": {"input_fail": k_input, "run_fail": k_run, "hier_viol": k_hier},
        }

        # 载入真实 R_fail
        cal_summary = CAL_DIR / "metrics" / f"summary_{challenge}.json"
        r_fail_real = json.loads(cal_summary.read_text())["R_fail"] if cal_summary.exists() else {"k": 0, "n": 0, "point": 0}

        verdict = "PASS" if r_fail_synth["point"] <= r_fail_real.get("point", 0) else "UNDECIDED"

        report["per_challenge"][challenge] = {
            "n_samples": n_obs,
            "r_fail_synth": r_fail_synth,
            "r_fail_real": r_fail_real,
            "verdict": verdict,
        }
        if verdict == "UNDECIDED":
            report["overall_verdict"] = "UNDECIDED"

    if mode == "p1":
        report["p2_evidence_gap"] = "P2 方向前置证据缺位已知情接受（掩码 ControlNet 训练前不存在 v1 可产样本），P2 依赖终验伴随监控兜底"

    # 写报告
    (output_dir / f"report_{mode}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # 生成 markdown
    md = [f"# {report['title']}", "", f"**模式**: {mode}", f"**总体判定**: **{report['overall_verdict']}**", "",
          "| 挑战 | 样本数 | R_fail_synth | R_fail_real | 判定 |", "|------|--------|-------------|-------------|------|"]
    for ch, d in report["per_challenge"].items():
        s, r = d["r_fail_synth"], d["r_fail_real"]
        md.append(f"| {ch} | {d['n_samples']} | {s['point']:.4f} ({s['k']}/{s['n']}) | {r.get('point',0):.4f} ({r.get('k',0)}/{r.get('n',0)}) | **{d['verdict']}** |")
    md.extend(["", "## R_fail 细分", ""])
    for ch, d in report["per_challenge"].items():
        b = d["r_fail_synth"]["breakdown"]
        md.extend([f"### {ch}", f"- 输入失败: {b['input_fail']}/{d['n_samples']}", f"- 运行失败: {b['run_fail']}/{d['n_samples']}", f"- 层级违反: {b['hier_viol']}/{d['n_samples']}", ""])
    if "p2_evidence_gap" in report:
        md.extend(["## P2 方向说明", report["p2_evidence_gap"], ""])
    (output_dir / f"report_{mode}.md").write_text("\n".join(md))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[OK] Report → {output_dir}")


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("create-case-lists")
    gen = sub.add_parser("generate")
    gen.add_argument("--mode", choices=("p1", "p3"), default="p1")
    gen.add_argument("--seed", type=int, default=42)
    prep = sub.add_parser("prep-inputs")
    prep.add_argument("--mode", choices=("p1", "p3"), default="p1")
    pred = sub.add_parser("predict")
    pred.add_argument("--mode", choices=("p1", "p3"), default="p1")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--mode", choices=("p1", "p3"), default="p1")

    args = parser.parse_args()
    args.mode = getattr(args, "mode", "p1")

    {
        "create-case-lists": cmd_create_lists,
        "generate": cmd_generate,
        "prep-inputs": cmd_prep_inputs,
        "predict": cmd_predict,
        "evaluate": cmd_evaluate,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
