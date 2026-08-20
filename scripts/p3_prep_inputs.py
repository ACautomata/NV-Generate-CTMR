#!/usr/bin/env python3
"""Issue #38 P3 输入组装：4 锚轮四模态体 → nnU-Net 输入。

每病例每锚 a 一个四模态体：通道 a 用真实锚（jobs jsonl 中的 anchor 路径），
其余 3 通道用 p3_samples/<CH>/<case>/a<a>/<tgt>.nii.gz 生成文件。
全部重采样 1mm + 居中 crop/pad 到 240×240×155，写
p3_nnunet_inputs/<CH>/<case>_a<a>_000{0..3}.nii.gz（通道映射与 P1/校准一致）。

在 sugon 上运行：python3 /root/private_data/l2-synth-eval/p3_prep_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

EVAL_ROOT = Path("/root/private_data/l2-synth-eval")
SAMPLES = EVAL_ROOT / "p3_samples"
OUTPUT = EVAL_ROOT / "p3_nnunet_inputs"

NN_CHANNELS = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
NN_SIZE = (240, 240, 155)
NN_SPACING = (1.0, 1.0, 1.0)


def resample_to_1mm(img: sitk.Image) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [int(round(s * sp)) for s, sp in zip(orig_size, orig_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(NN_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)


def crop_or_pad(arr: np.ndarray, target: tuple) -> np.ndarray:
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


def write_channel(src: Path, dst: Path) -> None:
    img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(resample_to_1mm(img))
    out_img = sitk.GetImageFromArray(crop_or_pad(arr, NN_SIZE[::-1]))
    out_img.SetSpacing(NN_SPACING)
    sitk.WriteImage(out_img, str(dst))


def main() -> int:
    # 由 jobs 文件驱动：每锚收集 (anchor 真实路径, 各 tgt 生成路径)
    volumes: dict[tuple[str, str, str], dict[str, str]] = {}
    for gpu in range(8):
        jobs_path = EVAL_ROOT / f"p3_jobs_gpu{gpu}.jsonl"
        if not jobs_path.exists():
            continue
        for line in jobs_path.read_text().splitlines():
            job = json.loads(line)
            key = (job["anchor_key"].split("/")[0],
                   job["anchor_key"].split("/")[1],
                   job["anchor_key"].split("/")[2])  # (CH, case, anchor_name)
            vol = volumes.setdefault(key, {"anchor_file": job["anchor"]})
            vol[job["tgt_name"]] = job["out"]

    n_done, n_skip, n_incomplete = 0, 0, 0
    for (ch, case_id, anchor_name), vol in sorted(volumes.items()):
        tgts = [m for m in NN_CHANNELS if m != anchor_name]
        if any(m not in vol for m in tgts) or any(not Path(vol[m]).is_file() for m in tgts):
            n_incomplete += 1
            continue
        out_dir = OUTPUT / ch
        out_dir.mkdir(parents=True, exist_ok=True)
        vol_id = f"{case_id}_a{anchor_name}"
        dsts = {m: out_dir / f"{vol_id}_{s}.nii.gz" for m, s in NN_CHANNELS.items()}
        if all(p.exists() for p in dsts.values()):
            n_skip += 1
            continue
        write_channel(Path(vol["anchor_file"]), dsts[anchor_name])
        for m in tgts:
            write_channel(Path(vol[m]), dsts[m])
        n_done += 1
        print(f"[OK] {ch}/{vol_id}", flush=True)

    print(f"p3 prep complete: {n_done} prepared, {n_skip} skipped, {n_incomplete} incomplete (missing generations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
