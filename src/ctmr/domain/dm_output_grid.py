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

"""The v1 base diffusion model's native output grid (issue #249, ruling #6).

``DmOutputGrid`` is the value object for the geometry the v1 DM natively samples
onto: a fixed 256x256x128 voxel grid at (0.94, 0.94, 1.36) mm (the #10 P1 recipe
pinning). It is the single definition point for the write-path affine: the
generated NIfTI sidecar must declare this real spacing, never the retired unit
1 mm convention that made the instrument chain's 1 mm resample a no-op and
produced the out-of-declared-domain / centroid-coordinate artefacts the geometry
re-audit quantified (146 cases / 431 ml above the z>=128 mm declared edge).

Pure value object, numpy-only closure -- no IO, no torch, no SimpleITK -- so both
the generation writers (the sidecar / diagnostic arms) and the acceptance
synthetic-domain report (which re-exports the size/spacing, consumer surface
unchanged) draw from this one home. File writing stays with the callers; this
module owns only the geometry value.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DmOutputGrid:
    """The v1 DM native output grid: size + spacing, both in xyz axis order."""

    size: tuple[int, int, int]
    spacing: tuple[float, float, float]

    def affine(self) -> np.ndarray:
        """The sidecar write affine ``diag(spacing, 1)`` -- a fresh array per call."""
        return np.diag([*self.spacing, 1.0])


V1_DM_OUTPUT_GRID = DmOutputGrid(size=(256, 256, 128), spacing=(0.94, 0.94, 1.36))
"""The v1 DM output grid (#10 P1 recipe pinning): 256x256x128 @ (0.94, 0.94, 1.36) mm."""
