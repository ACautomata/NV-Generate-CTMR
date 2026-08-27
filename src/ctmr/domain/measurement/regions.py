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

"""BraTS region label sets -- the single source of truth (ADR-0010, issue #109).

The seven drifted ``REGIONS`` literals (calibration mother, synthetic-domain
eval, P1/P2 dev eval, terminal-acceptance judge tuple + labels dict) collapse
onto this one dict: instrumentation session masks are BraTS-2023 label arrays
(background 0, non-enhancing core 1, peritumoral oedema 2, enhancing tumour 3)
whose region projections are WT = {1,2,3}, TC = {1,3} and ET = {3}. The judge's
tuple-of-names form derives from this dict (``REGION_NAMES``), never a second
literal; ``LABEL_DOMAIN`` pins every value a well-formed mask may carry.
"""

import numpy as np

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
"""Per-region label tuples, keyed in canonical WT / TC / ET order (dict order)."""

REGION_NAMES = tuple(REGIONS)
"""The judge's tuple-of-names form, derived -- not a second literal (ADR-0010 decision 1)."""

LABEL_DOMAIN = (0, 1, 2, 3)
"""Every label value a well-formed instrument mask may carry (0 = background)."""

VOXEL_ML = 0.001
"""Millilitres per voxel: the instrument grid is 1 mm isotropic (ADR-0008), 1mm^3 = 0.001 mL."""

SPACING_MM = (1.0, 1.0, 1.0)
"""isotropic 1 mm sampling for physical-distance measurements (also fixed by the instrument grid)."""


class RegionMasks:
    """Per-region boolean extraction from one instrument label array.

    The only place ``np.isin(..., REGIONS[...])`` lives for mask extraction:
    every volume, centroid and Dice in this package draws its region masks
    here, so the region projection rule is defined exactly once.
    """

    def __init__(self, labels: np.ndarray):
        self._labels = labels

    def of(self, region: str) -> np.ndarray:
        """Boolean mask of one region's labels (``REGIONS[region]``)."""
        return np.isin(self._labels, REGIONS[region])
