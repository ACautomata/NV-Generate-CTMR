#!/usr/bin/env python3
"""Issue #38 P1 输入组装：p1_samples/<CH>/<case>/<mod>.nii.gz → nnU-Net 输入。

对每个病例的四个模态：重采样 256×256×128@0.94×0.94×1.36mm → 1mm isotropic
（linear 插值，背景填 0，与 nnUNet 预处理一致），居中 crop/pad 到 240×240×155，
写为 p1_nnunet_inputs/<CH>/<case>_000{0..3}.nii.gz（0000=t1n 0001=t1c 0002=t2w 0003=t2f，
与校准管线通道映射一致）。

在 sugon 上运行：python3 /root/private_data/l2-synth-eval/p1_prep_inputs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

EVAL_ROOT = Path("/root/private_data/l2-synth-eval")
SAMPLES = EVAL_ROOT / "p1_samples"
OUTPUT = EVAL_ROOT / "p1_nnunet_inputs"

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


def main() -> int:
    n_done, n_skip = 0, 0
    for ch_dir in sorted(SAMPLES.iterdir()):
        if not ch_dir.is_dir():
            continue
        out_ch = OUTPUT / ch_dir.name
        for case_dir in sorted(ch_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            out_ch.mkdir(parents=True, exist_ok=True)
            dst_paths = {m: out_ch / f"{case_id}_{s}.nii.gz" for m, s in NN_CHANNELS.items()}
            if all(p.exists() for p in dst_paths.values()):
                n_skip += 1
                continue
            for mod, dst in dst_paths.items():
                if dst.exists():
                    continue
                img = sitk.ReadImage(str(case_dir / f"{mod}.nii.gz"))
                arr = sitk.GetArrayFromImage(resample_to_1mm(img))
                cropped = crop_or_pad(arr, NN_SIZE[::-1])  # sitk 数组是 (z,y,x)
                out_img = sitk.GetImageFromArray(cropped)
                out_img.SetSpacing(NN_SPACING)
                sitk.WriteImage(out_img, str(dst))
            n_done += 1
            print(f"[OK] {ch_dir.name}/{case_id}", flush=True)
    print(f"prep complete: {n_done} prepared, {n_skip} skipped (already exist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
