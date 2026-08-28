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
# Freeze-side support module (issue #134, ADR-0015 §2 maisi_engine).
#
# Function bodies are copied byte-for-byte from their legacy homes so that
# vendored engine files keep upstream behavior with no numeric drift:
#
# - ``dynamic_infer``                    from the retired scripts layer (git history; ``utils``).py
# - ``check_input_ct`` / ``check_input_mr`` from the retired scripts layer (git history; ``sample_mask``).py
#
# ``get_body_region_index_from_mask`` was deleted with issue #175 (ADR-0016
# M4): its only consumer, ``utils_infer.build_conditioning_tensors``, retired
# with the paired-path loaders — git history is the reproduction anchor.
#
# Only these extractions and this import block are new; everything below the
# imports is guarded by tests/infrastructure/maisi_engine/test_engine_smoke.py.
# ---------------------------------------------------------------------------

import json
import logging
import math

import torch


def dynamic_infer(inferer, model, images):
    """
    Perform dynamic inference using a model and an inferer, typically a monai SlidingWindowInferer.

    This function determines whether to use the model directly or to use the provided inferer
    (such as a sliding window inferer) based on the size of the input images.

    Args:
        inferer: An inference object, typically a monai SlidingWindowInferer, which handles patch-based inference.
        model (torch.nn.Module): The model used for inference.
        images (torch.Tensor): The input images for inference, shape [N,C,H,W,D] or [N,C,H,W].

    Returns:
        torch.Tensor: The output from the model or the inferer, depending on the input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        # Extract the spatial dimensions from the images tensor (H, W, D)
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size

        # Check that roi has the same number of dimensions as spatial_dims
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

        # Iterate and adjust each ROI dimension
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        try:
            output = inferer(network=model, inputs=images)
        finally:
            inferer.roi_size = orig_roi
        return output


def check_input_ct(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    """
    Validate input parameters for image generation.

    Args:
        body_region (list): List of body regions.
        anatomy_list (list): List of anatomical structures.
        label_dict_json (str): Path to the label dictionary JSON file.
        output_size (tuple): Desired output size of the image.
        spacing (tuple): Desired voxel spacing.
        controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # check output_size and spacing format
    if output_size[0] != output_size[1]:
        raise ValueError(f"The first two components of output_size need to be equal, yet got {output_size}.")
    if (output_size[0] not in [256, 384, 512]) or (output_size[2] not in [128, 256, 384, 512, 640, 768]):
        raise ValueError(
            f"The output_size[0] have to be chosen from [256, 384, 512], and output_size[2] have to be chosen from [128, 256, 384, 512, 640, 768], yet got {output_size}."
        )

    if spacing[0] != spacing[1]:
        raise ValueError(f"The first two components of spacing need to be equal, yet got {spacing}.")
    if spacing[0] < 0.5 or spacing[0] > 3.0 or spacing[2] < 0.5 or spacing[2] > 5.0:
        raise ValueError(f"spacing[0] have to be between 0.5 and 3.0 mm, spacing[2] have to be between 0.5 and 5.0 mm, yet got {spacing}.")

    if output_size[0] * spacing[0] < 256:
        FOV = [output_size[axis] * spacing[axis] for axis in range(3)]  # noqa: N806
        raise ValueError(
            f"`'spacing'({spacing}mm) and 'output_size'({output_size}) together decide the output field of view (FOV). The FOV will be {FOV}mm. We recommend the FOV in x and y axis to be at least 256mm for head, and at least 384mm for other body regions like abdomen. There is no such restriction for z-axis."
        )

    if controllable_anatomy_size is None:
        logging.info("`controllable_anatomy_size` is not provided.")
        return

    # check controllable_anatomy_size format
    if len(controllable_anatomy_size) > 10:
        raise ValueError(
            f"The length of list controllable_anatomy_size has to be less than 10. Yet got length equal to {len(controllable_anatomy_size)}."
        )
    available_controllable_organ = [
        "liver",
        "gallbladder",
        "stomach",
        "pancreas",
        "colon",
    ]
    available_controllable_tumor = [
        "hepatic tumor",
        "bone lesion",
        "lung tumor",
        "colon cancer primaries",
        "pancreatic tumor",
    ]
    available_controllable_anatomy = available_controllable_organ + available_controllable_tumor
    controllable_tumor = []
    controllable_organ = []
    for controllable_anatomy_size_pair in controllable_anatomy_size:
        if controllable_anatomy_size_pair[0] not in available_controllable_anatomy:
            raise ValueError(
                f"The controllable_anatomy have to be chosen from {available_controllable_anatomy}, yet got {controllable_anatomy_size_pair[0]}."
            )
        if controllable_anatomy_size_pair[0] in available_controllable_tumor:
            controllable_tumor += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[0] in available_controllable_organ:
            controllable_organ += [controllable_anatomy_size_pair[0]]
        if controllable_anatomy_size_pair[1] == -1:
            continue
        if controllable_anatomy_size_pair[1] < 0 or controllable_anatomy_size_pair[1] > 1.0:
            raise ValueError(
                f"The controllable size scale have to be between 0 and 1,0, or equal to -1, yet got {controllable_anatomy_size_pair[1]}."
            )
    if len(controllable_tumor + controllable_organ) != len(list(set(controllable_tumor + controllable_organ))):
        raise ValueError(f"Please do not repeat controllable_anatomy. Got {controllable_tumor + controllable_organ}.")
    if len(controllable_tumor) > 1:
        raise ValueError(f"Only one controllable tumor is supported. Yet got {controllable_tumor}.")

    if len(controllable_anatomy_size) > 0:
        logging.info(
            f"`controllable_anatomy_size` is not empty.\nWe will ignore `body_region` and `anatomy_list` and synthesize based on `controllable_anatomy_size`: ({controllable_anatomy_size})."
        )
    else:
        logging.info(
            f"`controllable_anatomy_size` is empty.\nWe will synthesize based on `body_region`: ({body_region}) and `anatomy_list`: ({anatomy_list})."
        )
        # check body_region format
        available_body_region = [
            "head",
            "chest",
            "thorax",
            "abdomen",
            "pelvis",
            "lower",
        ]
        for region in body_region:
            if region not in available_body_region:
                raise ValueError(f"The components in body_region have to be chosen from {available_body_region}, yet got {region}.")

        # check anatomy_list format
        with open(label_dict_json) as f:
            label_dict = json.load(f)
        for anatomy in anatomy_list:
            if anatomy not in label_dict.keys():
                raise ValueError(f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}.")
    logging.info(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")

    return


def check_input_mr(
    body_region,
    anatomy_list,
    label_dict_json,
    output_size,
    spacing,
    controllable_anatomy_size=[("pancreas", 0.5)],
):
    """
    Validate input parameters for image generation.

    Args:
        body_region (list): List of body regions.
        anatomy_list (list): List of anatomical structures.
        label_dict_json (str): Path to the label dictionary JSON file.
        output_size (tuple): Desired output size of the image.
        spacing (tuple): Desired voxel spacing.
        controllable_anatomy_size (list): List of tuples specifying controllable anatomy sizes.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # check output_size and spacing format
    if output_size[0] != output_size[1] and output_size[0] != output_size[2] and output_size[2] != output_size[1]:
        raise ValueError(f"At least two components of output_size need to be equal, yet got {output_size}.")
    if output_size[2] == 128:
        if output_size[0] != output_size[1]:
            raise ValueError(f"Two first components of output_size need to be equal when the third size is 128, yet got {output_size}.")
        if output_size[0] not in [128, 256, 384, 512]:
            raise ValueError(f"The output_size[0] have to be chosen from [128, 256, 384, 512] when output_size[2]=128, yet got {output_size}.")
    elif output_size[2] == 256:
        if (
            (output_size[0] == 128 and output_size[1] == 256)
            or (output_size[0] == 256 and output_size[1] == 128)
            or (output_size[0] == 256 and output_size[1] == 256)
        ):
            pass
        else:
            raise ValueError(
                f"The output_size can only be [128,256,256] or [256,128,256], or [256,256,256] when output_size[2]=256, yet got {output_size}."
            )
    else:
        raise ValueError(f"The output_size[2] have to be chosen from [128, 256], yet got {output_size}.")

    if any(s < 0.4 for s in spacing) or any(s > 5.0 for s in spacing):
        raise ValueError(f"spacing have to be between 0.4 and 5.0 mm, yet got {spacing}.")

    # check anatomy_list format
    with open(label_dict_json) as f:
        label_dict = json.load(f)
    for anatomy in anatomy_list:
        if anatomy not in label_dict.keys():
            raise ValueError(f"The components in anatomy_list have to be chosen from {label_dict.keys()}, yet got {anatomy}.")
    logging.info(f"The generate results will have voxel size to be {spacing}mm, volume size to be {output_size}.")

    return
