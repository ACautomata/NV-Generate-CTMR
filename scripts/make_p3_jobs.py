#!/usr/bin/env python3
"""Issue #38 P3 jobs 生成器：4 锚轮协议 × 病例表 → 8 GPU 分片 jsonl。

每病例 4 轮（每轮一个真实模态作锚），每轮其余 3 模态各一个 img2img job，
12 个有序模态对全覆盖。seed = IDX*10000 + ANCHOR_LABEL*100 + TGT_LABEL
（IDX 为病例表 1-based 全局位置，与 P1 惯例一致）。
同病例的 12 个 job 按锚排序连续排列，且整病例分到同一分片（anchor latent 缓存友好）。

锚解析：dev 病例优先取 dev_raw（BraTS 原始命名 <case>-<mod>.nii.gz），
fold_val fallback 病例取 nnU-Net imagesTr（<case>_000X.nii.gz）。

在 sugon 上运行：python3 /root/private_data/l2-synth-eval/make_p3_jobs.py
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_ROOT = Path("/root/private_data/l2-synth-eval")
DEV_RAW = Path("/root/private_data/l2-instrument-calibration/252940d0156f4c1258936fa25a1fb28bad61ae22/dev_raw")
N_GPUS = 8

# nnU-Net 通道后缀 -> (模态名, v1 DM label)
CHANNELS = {"0000": ("t1n", 9), "0001": ("t1c", 17), "0002": ("t2w", 10), "0003": ("t2f", 11)}
# 挑战 -> dev_raw 的 BraTS 目录名（METS 对应 MET）
CH_DIRNAME = {"GLI": "GLI", "SSA": "SSA", "MEN": "MEN", "METS": "MET", "PED": "PED"}


def resolve_anchor(challenge: str, case_id: str, images_dir: str, suffix: str, mod_name: str) -> str | None:
    raw = DEV_RAW / f"ASNR-MICCAI-BraTS2023-{CH_DIRNAME[challenge]}-Challenge-TrainingData" / case_id
    dev_path = raw / f"{case_id}-{mod_name}.nii.gz"
    if dev_path.is_file():
        return str(dev_path)
    nnunet_path = Path(images_dir) / "imagesTr" / f"{case_id}_{suffix}.nii.gz"
    if nnunet_path.is_file():
        return str(nnunet_path)
    return None


def main() -> None:
    cases = json.loads((EVAL_ROOT / "case_lists" / "p3_cases.json").read_text())
    shards: list[list[dict]] = [[] for _ in range(N_GPUS)]
    missing: list[str] = []

    for idx0, case in enumerate(cases):
        idx = idx0 + 1
        ch, case_id, images_dir = case["challenge"], case["case_id"], case["images_dir"]
        for suffix, (anchor_name, anchor_label) in CHANNELS.items():
            anchor_path = resolve_anchor(ch, case_id, images_dir, suffix, anchor_name)
            if anchor_path is None:
                missing.append(f"{ch}/{case_id}/{anchor_name}")
                continue
            anchor_key = f"{ch}/{case_id}/{anchor_name}"
            for _tgt_suffix, (tgt_name, tgt_label) in CHANNELS.items():
                if tgt_name == anchor_name:
                    continue
                shards[idx0 % N_GPUS].append(
                    {
                        "anchor": anchor_path,
                        "anchor_key": anchor_key,
                        "tgt_label": tgt_label,
                        "tgt_name": tgt_name,
                        "seed": idx * 10000 + anchor_label * 100 + tgt_label,
                        "out": str(EVAL_ROOT / "p3_samples" / ch / case_id / f"a{anchor_name}" / f"{tgt_name}.nii.gz"),
                    }
                )

    total = sum(len(s) for s in shards)
    for gpu, jobs in enumerate(shards):
        path = EVAL_ROOT / f"p3_jobs_gpu{gpu}.jsonl"
        with open(path, "w") as fh:
            for job in jobs:
                fh.write(json.dumps(job) + "\n")
        print(f"[OK] gpu{gpu}: {len(jobs)} jobs -> {path}")
    print(f"total: {total} jobs (expect 12 * {len(cases)} minus missing anchors)")
    if missing:
        print(f"[WARN] {len(missing)} anchors unresolved (jobs skipped):")
        for m in missing[:10]:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
