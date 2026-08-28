# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mask-conditioned inference wrapper around the modality-agnostic DM core (ticket 09).

``ldm_conditional_sample_one_image_from_mask`` is the mask-specific wrapper the
mask family's sampler (and the user-facing ``infer_image_from_mask`` tools)
drive: ``binarize_labels`` converts the 1-channel integer mask into the 8-channel
binary ControlNet conditioning tensor, the CFG branch builds the tumor-free
unconditional counterpart via ``remove_tumors``, and ``crop_img_body_mask``
regularizes background voxels after decoding. The inner timestep loop is
delegated to ``ctmr.infrastructure.maiisi_engine.utils_infer``. Relocated
verbatim from ``infer_image_from_mask.py`` (retired scripts layer, git history) / ``utils.py`` (retired scripts layer, git history)
(git history is the provenance anchor).
"""

import logging

import torch

from ctmr.infrastructure.dataio.augmentation import remove_tumors
from ctmr.infrastructure.maiisi_engine.utils_infer import run_controlnet_conditioned_image_dm


def binarize_labels(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Convert input tensor to binary representation.

    This function takes an input tensor and converts it to a binary representation
    using the specified number of bits.

    Args:
        x (torch.Tensor): Input tensor with shape (B, 1, H, W, D).
        bits (int, optional): Number of bits to use for binary representation. Defaults to 8.

    Returns:
        torch.Tensor: Binary representation of the input tensor with shape (B, bits, H, W, D).
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1).permute(0, 4, 1, 2, 3)


def crop_img_body_mask(synthetic_images, combine_label, a_min=-1000):
    """
    Crop the synthetic image using a body mask.

    Args:
        synthetic_images (torch.Tensor): The synthetic images.
        combine_label (torch.Tensor): The body mask.

    Returns:
        torch.Tensor: The cropped synthetic images.
    """
    synthetic_images[combine_label == 0] = a_min
    return synthetic_images


def ldm_conditional_sample_one_image_from_mask(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    spacing_tensor,
    latent_shape,
    output_size,
    noise_factor,
    top_region_index_tensor=None,
    bottom_region_index_tensor=None,
    modality_tensor=None,
    num_inference_steps=1000,
    autoencoder_sliding_window_infer_size=(96, 96, 96),
    autoencoder_sliding_window_infer_overlap=0.6667,
    cfg_guidance_scale=0,
):
    """
    Generate a CT/MR image from a **3D label mask** via the ControlNet-
    conditioned image LDM.

    This is the **mask-specific** wrapper around the modality-agnostic core
    ``run_controlnet_conditioned_image_dm``. It does three mask-specific things:

      1. Pre-process: ``binarize_labels`` converts the 1-channel integer mask
         to the 8-channel binary ControlNet conditioning tensor.
      2. CFG (when ``cfg_guidance_scale > 0``): builds a tumor-free
         unconditional counterpart via ``remove_tumors`` + ``binarize_labels``.
      3. Post-process: ``crop_img_body_mask`` regularizes background voxels
         to ``a_min`` (CT: -1000; MR: 0) using the mask.

    Returns ``(synthetic_image, combine_label)`` — the mask is returned for
    downstream filtering (e.g. ``filter_mask_with_organs``).
    """
    # modality_tensor can be scalar (single mask) or shape (B,) (batch infer);
    # collapse to a single int so `if` doesn't choke on a multi-element bool tensor.
    if modality_tensor is not None and int(modality_tensor.flatten()[0]) <= 7:
        a_min = -1000  # CT background floor
    else:
        a_min = 0  # MR background floor

    combine_label = combine_label_or.to(device)
    if output_size[0] != combine_label.shape[2] or output_size[1] != combine_label.shape[3] or output_size[2] != combine_label.shape[4]:
        logging.info(
            "output_size is not a desired value. Need to interpolate the mask to match with output_size. The result image will be very low quality."
        )
        combine_label = torch.nn.functional.interpolate(combine_label, size=output_size, mode="nearest")

    # ── Mask-specific pre-processing ───────────────────────────────────────────
    # NOTE (modality-specific): the next line converts mask → ControlNet
    # conditioning. A future image-conditioned ControlNet would replace this
    # with image normalization in its own wrapper module.
    controlnet_cond_tensor = binarize_labels(combine_label.as_tensor().long()).half()

    controlnet_uncond_tensor = None
    if cfg_guidance_scale > 0:
        # Mask-specific unconditional branch: same mask with tumors removed.
        combine_label_no_tumor = torch.nn.functional.interpolate(
            remove_tumors(combine_label.squeeze(0)).unsqueeze(0).float(),
            size=output_size,
            mode="nearest",
        ).to(combine_label.dtype)
        controlnet_uncond_tensor = binarize_labels(combine_label_no_tumor.as_tensor().long()).half()
        del combine_label_no_tumor

    # ── Modality-agnostic core ─────────────────────────────────────────────────
    synthetic_images = run_controlnet_conditioned_image_dm(
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        device=device,
        controlnet_cond_tensor=controlnet_cond_tensor,
        spacing_tensor=spacing_tensor,
        latent_shape=latent_shape,
        output_size=output_size,
        noise_factor=noise_factor,
        top_region_index_tensor=top_region_index_tensor,
        bottom_region_index_tensor=bottom_region_index_tensor,
        modality_tensor=modality_tensor,
        num_inference_steps=num_inference_steps,
        autoencoder_sliding_window_infer_size=autoencoder_sliding_window_infer_size,
        autoencoder_sliding_window_infer_overlap=autoencoder_sliding_window_infer_overlap,
        cfg_guidance_scale=cfg_guidance_scale,
        controlnet_uncond_tensor=controlnet_uncond_tensor,
    )

    # ── Mask-specific post-processing ──────────────────────────────────────────
    # Regularize background HU using the mask: voxels where mask==0 → a_min.
    synthetic_images = crop_img_body_mask(synthetic_images, combine_label, a_min=a_min)
    return synthetic_images, combine_label


# Backward-compat alias — existing callers (LDMSampler, infer_image_from_mask_batch,
# notebooks) import the old name. Keep it pointing at the mask wrapper.
ldm_conditional_sample_one_image = ldm_conditional_sample_one_image_from_mask
