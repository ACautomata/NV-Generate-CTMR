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

# ---------------------------------------------------------------------------
# Vendored snapshot (issued #134 style, ticked 08 / ADR-0015 §2 maisi_engine):
# byte-for-byte copy of ``img2img_infer.py`` (retired scripts layer, git history) with import lines rewritten
# to this package home. Behavior must stay stable -- machine-guarded by
# tests/infrastructure/maisi_engine/test_vendored_parity.py (AST equality).
# ---------------------------------------------------------------------------
"""Rectified-flow img2img 推理（Issue #38 P3 式零训练基线）。

真实锚影像 → 与训练一致的预处理（percentile 99.5 归一化 + RAS + resize 256×256×128）
→ AE encode → ·scale_factor → add_noise 到 t_start → 以目标模态标签从截断的
timestep 序列去噪到 0 → decode → ·1000 → int16。

用法（与 diff_model_infer 相同的配置体系，另加 -i 锚图像与 --strength）：
  python3 -m ctmr.infrastructure.maisi_engine.img2img_infer -e <env> -c <model_cfg> -t <net_def> \
      -i <anchor.nii.gz> --strength 0.9 -g 1
模型配置中 diffusion_unet_inference.modality 为目标模态标签。
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime

import monai.transforms as monai_t
import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism
from tqdm import tqdm

from ctmr.infrastructure.maisi_engine.diff_model_infer import load_models, prepare_tensors, save_image
from ctmr.infrastructure.maisi_engine.diff_model_setting import initialize_distributed, load_config, setup_logging
from ctmr.infrastructure.maisi_engine.inference_primitives import dynamic_infer
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance
from ctmr.infrastructure.maisi_engine.utils_infer import ReconModel


def set_random_seed(seed: int) -> int:
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def load_anchor_latent(
    anchor_path: str,
    autoencoder: torch.nn.Module,
    device: torch.device,
    output_size: tuple,
    logger: logging.Logger,
) -> torch.Tensor:
    """锚影像 → 训练同款预处理 → encode → ·scale_factor 后的归一化 latent (1,C,H,W,D)。"""
    transforms = monai_t.Compose(
        [
            monai_t.LoadImage(image_only=True),
            monai_t.EnsureChannelFirst(),
            monai_t.Orientation(axcodes="RAS"),
            monai_t.EnsureType(dtype=torch.float32),
            monai_t.ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=False),
            monai_t.Resize(spatial_size=output_size, mode="trilinear"),
        ]
    )
    x = transforms(anchor_path).unsqueeze(0).to(device)  # [1,1,X,Y,Z]
    logger.info(f"anchor {anchor_path} -> {tuple(x.shape)}")

    # encode 走训练数据构建同款 sliding-window（roi 大于输入时整体前向）
    encode_inferer = SlidingWindowInferer(
        roi_size=[320, 320, 160],
        sw_batch_size=1,
        progress=False,
        mode="gaussian",
        overlap=0.4,
        sw_device=device,
        device=device,
    )
    with torch.amp.autocast("cuda", enabled=True):
        z = dynamic_infer(encode_inferer, autoencoder.encode_stage_2_inputs, x)
    # encode 的中间激活占用大，释放缓存后再进入去噪循环（否则叠出 OOM）
    torch.cuda.empty_cache()
    # encode 在 autocast 下输出 half；去噪循环的输入统一 float32（与 P1 的 randn 一致）
    return z.float()


def run_img2img(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    anchor_latent: torch.Tensor,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    strength: float,
    logger: logging.Logger,
) -> np.ndarray:
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None

    noise_scheduler = define_instance(args, "noise_scheduler")
    assert isinstance(noise_scheduler, RFlowScheduler), "img2img 仅支持 RFlow 调度器"
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(anchor_latent.shape[2:])),
    )

    # 截断：从第一个 timestep ≤ strength·1000 的位置开始（t 轴：0 干净，1000 纯噪声）
    all_timesteps = noise_scheduler.timesteps
    threshold = float(strength) * noise_scheduler.num_train_timesteps
    start_idx = int((all_timesteps > threshold).sum())
    if start_idx >= len(all_timesteps) - 1:
        raise ValueError(f"strength={strength} 截断后步数不足（timesteps={all_timesteps.tolist()[:5]}...）")
    timesteps = all_timesteps[start_idx:]
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
    logger.info(f"img2img: strength={strength}, skip first {start_idx} steps, t {float(timesteps[0])} -> 0 over {len(timesteps)} steps")

    # x_t = (1 - t/1000)·x0 + (t/1000)·ε —— 与训练 add_noise 同一概率路径
    noise = torch.randn(anchor_latent.shape, device=device, dtype=anchor_latent.dtype)
    latent_norm = anchor_latent * scale_factor
    image = noise_scheduler.add_noise(original_samples=latent_norm, noise=noise, timesteps=timesteps[:1].to(device))

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    cfg_guidance_scale = args.cfg_guidance_scale
    progress_bar = tqdm(list(zip(timesteps, next_timesteps)), total=len(timesteps))
    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:
            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update({"class_labels": modality_tensor})

            if cfg_guidance_scale > 0:
                for k in unet_inputs.keys():
                    if k != "class_labels":
                        unet_inputs[k] = torch.cat([unet_inputs[k]] * 2)
                    else:
                        unet_inputs[k] = torch.cat([unet_inputs[k], torch.zeros_like(modality_tensor)])
                model_t, model_uncond = unet(**unet_inputs).chunk(2)
                model_output = model_uncond + cfg_guidance_scale * (model_t - model_uncond)
            else:
                model_output = unet(**unet_inputs)

            image, _ = noise_scheduler.step(model_output, t, image, next_t)

        inferer = SlidingWindowInferer(
            roi_size=[96, 96, 96],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.25,
            sw_device=device,
            device=torch.device("cpu"),
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)
    data = synthetic_images.squeeze().cpu().detach().numpy()
    modality = int(modality_tensor.cpu().item())
    if modality >= 8:  # MR: 模型输出 [0,1] -> [0,1000]
        data = data * 1000.0
        data = np.clip(data, 0, None)
    else:  # CT
        data = data * 2000.0 - 1000.0
        data = np.clip(data, -1000, 1000)
    return np.int16(data)


@torch.inference_mode()
def img2img_infer(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    anchor_path: str,
    strength: float,
    num_gpus: int,
) -> None:
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank if "random_seed" in args.diffusion_unet_inference.keys() else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])
    output_prefix = args.output_prefix

    modality = args.diffusion_unet_inference["modality"]
    args.cfg_guidance_scale = args.diffusion_unet_inference["cfg_guidance_scale"]

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)

    anchor_latent = load_anchor_latent(anchor_path, autoencoder, device, output_size, logger)
    data = run_img2img(
        args,
        device,
        autoencoder,
        unet,
        scale_factor,
        anchor_latent,
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
        strength,
        logger,
    )

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = (
        f"{args.output_dir}/{output_prefix}_seed{random_seed}"
        f"_size{output_size[0]:d}x{output_size[1]:d}x{output_size[2]:d}"
        f"_spacing{out_spacing[0]:.2f}x{out_spacing[1]:.2f}x{out_spacing[2]:.2f}"
        f"_{timestamp}_rank{local_rank}_modality{modality}.nii.gz"
    )
    save_image(data, output_size, out_spacing, output_path, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rectified-flow img2img inference")
    parser.add_argument("-e", "--env_config", type=str, required=True)
    parser.add_argument("-c", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--model_def", type=str, required=True)
    parser.add_argument("-i", "--input_image", type=str, required=True, help="真实锚图像 (nii.gz)")
    parser.add_argument("--strength", type=float, default=0.9, help="起始噪声水平 t_start/1000")
    parser.add_argument("-g", "--num_gpus", type=int, default=1)

    args = parser.parse_args()
    img2img_infer(args.env_config, args.model_config, args.model_def, args.input_image, args.strength, args.num_gpus)
