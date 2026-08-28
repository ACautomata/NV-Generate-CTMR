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
# Freeze-side support module (issue #134, ADR-0015 §2 maiisi_engine).
#
# Function bodies are copied byte-for-byte from their legacy homes so that
# vendored engine files keep upstream behavior with no numeric drift:
#
# - ``SUPPORT_MODALITIES``              from the retired scripts layer (git history; ``transforms``).py
# - ``define_fixed_intensity_transform`` from the retired scripts layer (git history; ``transforms``).py
# - ``define_instance``                 from the retired scripts layer (git history; ``utils``).py
#
# Only these extractions and this import block are new; everything below the
# imports is guarded by tests/infrastructure/maiisi_engine/test_vendored_parity.py.
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
        "mri": [ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False)],
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
