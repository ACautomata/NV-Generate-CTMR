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
# Vendored snapshot (issue #134, ADR-0015 §2 maisi_engine): byte-for-byte
# copy of ``utils_infer.py`` (retired scripts layer, git history) with import lines rewritten to this package home.
# Behavior must stay stable — machine-guarded by
# tests/infrastructure/maisi_engine/test_engine_smoke.py (execution smoke).
#
# Issue #175 (ADR-0016 M4) surgically deleted the members the domain
# generation entities replaced or that lost every caller:
# ``run_controlnet_conditioned_image_dm`` (the retired ControlNet-conditioned
# denoise core), ``load_mask_models`` / ``load_paired_inference_models`` /
# ``build_conditioning_tensors`` (the retired LDMSampler paired-path loaders).
# Git history is their reproduction anchor.
# ---------------------------------------------------------------------------
"""
Shared inference helpers reused across:

- the cross-modal family (live) — P3 candidate/monitor image-side model loading

What lives here:

- ``ReconModel``                — wraps an autoencoder for scale-corrected decode
- ``initialize_noise_latents``  — fp16 random-noise latent generator
- ``load_image_models``         — image AE + image DM + ControlNet + scheduler

The retired ControlNet-conditioned denoise core and the paired-path mask-DM
loaders were deleted with issue #175 (git history; the domain
``DiffusionModel`` + ``ControlNetBypass`` composition is their canonical
replacement).
"""

from __future__ import annotations

import monai
import torch


class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.

    Attributes:
        autoencoder: The autoencoder model used for decoding.
        scale_factor: Scaling factor applied to the input before decoding.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.

        Args:
            z (torch.Tensor): The input latent representation.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def initialize_noise_latents(latent_shape, device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                1,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )


def load_image_models(args, device: torch.device):
    """
    Load **image-side** networks (image AE + image DM + ControlNet) + the
    image noise scheduler from disk.

    Args:
        args: a config namespace already populated by ``load_config``. Must
            contain the keys: ``trained_autoencoder_path``,
            ``trained_diffusion_path``, ``trained_controlnet_path``,
            plus the network defs (``autoencoder_def``, ``diffusion_unet_def``,
            ``controlnet_def``, ``noise_scheduler``).
        device: target device.

    Returns:
        ``(autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler)``.
        All networks are moved to ``device`` and set to ``.eval()`` mode.
    """
    from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt = torch.load(args.trained_autoencoder_path, weights_only=False)
    if "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    autoencoder.load_state_dict(ckpt)

    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    ckpt_dm = torch.load(args.trained_diffusion_path, weights_only=False)
    diffusion_unet.load_state_dict(ckpt_dm["unet_state_dict"], strict=False)
    scale_factor = ckpt_dm["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    ckpt_cn = torch.load(args.trained_controlnet_path, weights_only=False)
    monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
    controlnet.load_state_dict(ckpt_cn["controlnet_state_dict"], strict=False)

    noise_scheduler = define_instance(args, "noise_scheduler")

    autoencoder.eval()
    diffusion_unet.eval()
    controlnet.eval()
    return autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler
