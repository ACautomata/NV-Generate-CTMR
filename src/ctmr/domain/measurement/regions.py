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

"""BraTS region label sets -- re-exported from the stdlib-only leaf (ADR-0017 decision 3, issue #222).

``REGIONS`` / ``REGION_NAMES`` / ``LABEL_DOMAIN`` are defined once in
``ctmr.domain.vocabulary`` (stdlib-only, shared with the terminal-acceptance
judge) and re-exported here unchanged -- the measurement package's import
surface stays as ADR-0010 pinned it. What lives in this numpy-side module:
the instrument-grid constants (``VOXEL_ML`` / ``SPACING_MM``, ADR-0008) and
``RegionMasks``, the only place ``np.isin(..., REGIONS[...])`` lives for mask
extraction -- every volume, centroid and Dice in this package draws its
region masks there, so the region projection rule is applied exactly once
(its definition sits one import away, in the leaf).
"""

import numpy as np

from ctmr.domain.vocabulary import LABEL_DOMAIN, REGION_NAMES, REGIONS  # noqa: F401  (re-export, ADR-0017 decision 3)

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
