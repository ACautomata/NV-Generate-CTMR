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

"""Label-volume tumor removal: the pure tensor surgery the mask sampling path runs (ADR-0019 §3, #269).

Floated up verbatim from the retired augmentation scripts via
``infrastructure.dataio.augmentation`` (git history) -- pure tensor logic, no
IO. ``remove_tumors`` turns a generated combined-label volume tumor-free:
organ tumors remap to their organs, lesion regions take the offered pseudo
labels (or, unoffered, the majority organ of their immediate neighborhood /
the brain label). The infrastructure augmentation module's re-export alias
retired with the mask-family migration (#273): the sampling path imports this
domain module directly.
"""

import torch
from torch import Tensor

from ctmr.domain.morphology import dilate_one_img


def remove_tumors(orig_labels, pseudo_labels=None):
    """
    Replace tumor-related class ids with organ ids or pseudo labels.

    The function:
    - Maps tumor labels (e.g. hepatic, pancreatic, colon) to their organ counterparts.
    - Replaces ambiguous lesion regions with pseudo labels.

    Args:
        orig_labels (torch.Tensor): Original ground-truth labels.
        pseudo_labels (torch.Tensor): Pseudo labels used for replacement for tumor region, same size with orig_labels.

    Returns:
        torch.Tensor: Modified labels with tumor regions reassigned.
    """
    # hepatic tumor->liver, pancreatic tumor->pancreas, colon primaries timor->colon, kidney cyst-> kidney
    if len(orig_labels.shape) not in [3, 4]:
        raise ValueError(f"input has to be 3D/4D, [1,X,Y,Z] or [1,X,Y]. Yet got {orig_labels.shape}.")
    x = remap_labels(orig_labels, {26: 1, 24: 4, 27: 62, 116: 14, 117: 5})

    if pseudo_labels is not None:
        # replace with pseudo_labels, for lung tumor, bone lesion, brain tumors
        for lesion_id in [23, 128, 401, 402, 403, 176]:
            mask = x == lesion_id
            if mask.any():
                x[mask] = pseudo_labels[mask]
    else:
        # Replace lesion-like classes by pseudo labels (explicit and ordered)
        # replace lung tumor with majority vote of its immediate neighborhood
        x = remove_tumors_majority_vote(
            (x == 23),
            x,
            organ_label_lists=(28, 29, 30, 31, 32),
        )
        # replace brain tumors->brain
        x = remap_labels(x, {401: 22, 402: 22, 403: 22, 176: 22})
    return x


def remove_tumors_majority_vote(
    tumor_mask_: Tensor,
    volume: Tensor,
    organ_label_lists=(28, 29, 30, 31, 32),
):
    """
    Replace tumor voxels in a segmentation volume with the majority organ label
    from their immediate neighborhood.

    Steps:
    1. Dilate the tumor mask with a small kernel to get a surrounding "ring."
    2. Remove the original tumor voxels from that dilation (keeping only the ring).
    3. Restrict the ring to voxels whose labels are in `organ_label_lists`.
    4. Extract labels from that ring and compute the majority vote (mode).
    - If the ring is empty, fall back to the most frequent organ label
        in the entire volume.
    5. Overwrite the original tumor region in `volume` with the chosen organ label.

    Args:
        tumor_mask_ (Tensor): Binary (0/1) tumor mask of shape [1, D, H, W] or [1, ...].
        volume (Tensor): Segmentation label map of shape [D, H, W] or [1, D, H, W],
                        containing integer organ labels.
        organ_label_lists (iterable): Labels considered valid "organ" labels
                                    to fill the tumor region (default [28–32]).

    Returns:
        Tensor: A copy of `volume` where tumor voxels are replaced with the
                majority-vote organ label.

    Notes:
        - `volume` is expected to contain integer labels, not raw intensities.
        - `dilate_one_img` should be a morphological dilation that accepts
        a single-channel binary tensor and returns the dilated mask.
    """
    # 1) Build a ring: dilate tumor, then remove original tumor voxels
    dil = dilate_one_img(tumor_mask_.squeeze(0), filter_size=3, pad_value=1.0).unsqueeze(0)
    ring = dil.bool() & (~tumor_mask_.bool())

    # 2) Keep only organ labels in the ring
    organ_set = torch.tensor(organ_label_lists, device=volume.device, dtype=volume.dtype)
    ring = ring & torch.isin(volume, organ_set)

    # 3) Majority vote from the ring (fallback if empty)
    tmp = volume[ring]
    if tmp.numel() == 0:
        # Fallback: pick the most common organ label in the whole volume
        counts = torch.stack([(volume == lbl).sum() for lbl in organ_set])
        mode = organ_set[counts.argmax()].item()
    else:
        mode = torch.mode(tmp, 0)[0].item()

    # 4) Fill original tumor with the chosen organ label
    out = volume.clone()
    out[tumor_mask_.bool()] = mode
    return out


def remap_labels(x, mapping: dict[int, int]):
    """
    Remap integer labels in a tensor.

    Args:
        x (torch.Tensor): Input label tensor.
        mapping (dict[int,int]): Mapping from old label ids to new label ids.

    Returns:
        torch.Tensor: A copy of `x` with values replaced according to mapping.
    """
    out = x.clone()
    for old, new in mapping.items():
        out[x == old] = new
    return out
