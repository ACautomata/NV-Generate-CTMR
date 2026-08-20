# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""img2img 批量推理（Issue #38 P3 式 4 锚轮协议）。

一次加载模型，循环处理 job 列表；同一锚的多个目标模态共享一次 encode。
job 为 jsonl，每行: {"anchor": <nii路径>, "tgt_label": <int>, "seed": <int>,
"out": <输出nii.gz>, "anchor_key": <锚唯一标识>}
（同 anchor_key 连续出现时复用 latent；生成 jobs 时请按 anchor_key 排序。）

用法：
  python3 -m scripts.img2img_batch -e <env> -c <model_cfg> -t <net_def> \
      --jobs <jobs.jsonl> --strength 0.9 -g 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np
import torch
from monai.utils import set_determinism

from .diff_model_infer import load_models, prepare_tensors
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .img2img_infer import load_anchor_latent, run_img2img


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Batched rectified-flow img2img inference")
    parser.add_argument("-e", "--env_config", type=str, required=True)
    parser.add_argument("-c", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--model_def", type=str, required=True)
    parser.add_argument("--jobs", type=str, required=True, help="jsonl job list")
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    args_cli = parser.parse_args()

    with open(args_cli.jobs) as fh:
        jobs = [json.loads(line) for line in fh if line.strip()]
    if not jobs:
        raise SystemExit("empty job list")

    args = load_config(args_cli.env_config, args_cli.model_config, args_cli.model_def)
    local_rank, world_size, device = initialize_distributed(args_cli.num_gpus)
    logger = setup_logging("inference")
    logger.info(f"[batch] {len(jobs)} img2img jobs, strength={args_cli.strength}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    args.cfg_guidance_scale = args.diffusion_unet_inference["cfg_guidance_scale"]

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    top_ri, bottom_ri, spacing_tensor, _modality_tensor = prepare_tensors(args, device)

    # 每 job 需要独立的 modality tensor（目标模态不同）
    modality_of = lambda label: (label * torch.ones((len(spacing_tensor)), dtype=torch.long)).to(device)

    cached_key, anchor_latent = None, None
    n_done, n_skip, n_fail = 0, 0, 0
    for i, job in enumerate(jobs):
        out_path = job["out"]
        if os.path.isfile(out_path):
            n_skip += 1
            continue
        try:
            if job["anchor_key"] != cached_key:
                anchor_latent = load_anchor_latent(
                    job["anchor"], autoencoder, device, output_size, logger)
                cached_key = job["anchor_key"]
            # 每个 job 独立 seed（噪声与截断起点一致性由 set_determinism 控制）
            set_determinism(job["seed"])

            modality_tensor = modality_of(job["tgt_label"])
            data = run_img2img(
                args, device, autoencoder, unet, scale_factor, anchor_latent,
                top_ri, bottom_ri, spacing_tensor, modality_tensor,
                args_cli.strength, logger,
            )
            out_affine = np.eye(4)
            for k in range(3):
                out_affine[k, k] = out_spacing[k]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            nib.save(nib.Nifti1Image(data, affine=out_affine), out_path)
            n_done += 1
            logger.info(f"[batch {i + 1}/{len(jobs)}] saved {out_path}")
        except Exception as exc:  # 单 job 失败不终止批次
            n_fail += 1
            logger.error(f"[batch {i + 1}/{len(jobs)}] FAILED {job.get('out')}: {exc}")
    logger.info(f"[batch] done={n_done} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
