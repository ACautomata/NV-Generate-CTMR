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

"""Mask-conditioning helpers for the mask family (ticket 09).

``binarize_labels`` converts the 1-channel integer mask into the 8-channel
binary ControlNet conditioning tensor, and ``crop_img_body_mask`` regularizes
background voxels after decoding.  Relocated verbatim from ``utils.py``
(retired scripts layer, git history; git history is the provenance anchor).

The former mask wrapper ``ldm_conditional_sample_one_image_from_mask`` — which
drove these helpers plus the retired ControlNet-conditioned denoise core — lost
its last live caller with issue #172 and was deleted with issue #175 (ADR-0016
M4); git history is its reproduction anchor.
"""

import torch


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
