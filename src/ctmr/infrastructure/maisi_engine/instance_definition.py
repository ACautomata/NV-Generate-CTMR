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
# - ``SUPPORT_MODALITIES``              from the retired scripts layer (git history; ``transforms``).py
# - ``define_fixed_intensity_transform`` from the retired scripts layer (git history; ``transforms``).py
# - ``define_instance``                 from the retired scripts layer (git history; ``utils``).py
#
# Only these extractions and this import block are new; everything below the
# imports is guarded by tests/infrastructure/maisi_engine/test_engine_smoke.py.
#
# One recorded deviation from the byte-for-byte rule (issue #251, series-② T4;
# the one recipe delta the retrain ticket pins): the mri arm's normalization
# flag is ``clip=True`` -- upstream shipped ``clip=False``. Job C measured the
# trade (deploy/experiments/20260829-P1根因甄别-作业C-t1c强度域甄别.md): the
# unclipped affine extrapolates the top ~0.5% of training t1c voxels above 1.0,
# out of the frozen autoencoder_v1's reconstruction domain (extrapolated-band
# self-eval MAE 0.8673 with intra-tumour negative-value artifacts vs 0.0062
# for truncated inputs); truncation aligns the encoding input domain with the
# frozen VAE. Bounded-output contract pinned by
# tests/infrastructure/maisi_engine/test_intensity_transform_factory.py; the
# pre-T4 behavior lives in git history (re-encode all training embeddings
# after this flag: clip=False-encoded embeddings are not reusable in the
# clip=True world). The P3 inference anchor rides the same verdict --
# AnchorLatentEncoder's preprocessing matched to clip=True in series-③ T3
# (issue #313, tests/application/generation/cross_modal/test_anchor_clip.py).
# ---------------------------------------------------------------------------

import warnings
from argparse import Namespace
from typing import Any

from monai.bundle import ConfigParser
from monai.transforms import (
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)

SUPPORT_MODALITIES = ["ct", "mri"]


def define_fixed_intensity_transform(modality: str, image_keys: list[str] = ["image"]) -> list:
    """
    Define fixed intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of intensity transforms.
    """
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    modality = modality.lower()  # Normalize modality to lowercase

    intensity_transforms = {
        # clip=True (issue #251): the one recorded deviation from the vendored
        # byte-for-byte rule -- see the module docstring block above.
        "mri": [ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=True)],
        "ct": [ScaleIntensityRanged(keys=image_keys, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)],
    }

    if modality not in intensity_transforms:
        return []

    return intensity_transforms[modality]


def define_instance(args: Namespace, instance_def_key: str) -> Any:
    """
    Define and instantiate an object based on the provided arguments and instance definition key.

    This function uses a ConfigParser to parse the arguments and instantiate an object
    defined by the instance_def_key.

    Args:
        args: An object containing the arguments to be parsed.
        instance_def_key (str): The key used to retrieve the instance definition from the parsed content.

    Returns:
        The instantiated object as defined by the instance_def_key in the parsed configuration.
    """
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)
