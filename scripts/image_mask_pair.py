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

"""The two volumes MR-RATE publishes per series, loaded together (spec #13 section 3.2, ticket T6 #22).

Every study zip carries the defaced native-space image and its HD-BET brain mask, and both of
the pipeline's voxel-level stages consume the same pair: ``scripts.spine_filter`` measures the
mask against the image's field of view, and ``scripts.create_skull_stripped`` multiplies the
image by the mask.  Both are silently wrong if the two do not share a voxel grid -- nibabel
returns arrays of whatever shape each header claims, with no cross-check -- so the invariant
(and the one error message that reports it) is enforced here rather than restated per consumer.

The paths are MR-RATE's published layout; ``scripts.mrrate_series`` owns how those paths are
spelled, and this module owns what they hold once loaded.
"""

from pathlib import Path

import nibabel as nib


class ImageMaskPair:
    """An image and its HD-BET brain mask, loaded together and proven to share a voxel grid."""

    def __init__(self, image_path: Path, mask_path: Path) -> None:
        self._image_path = image_path
        self._image = nib.load(str(image_path))
        self._mask = nib.load(str(mask_path))
        if self._image.shape != self._mask.shape:
            raise ValueError(f"image {self._image.shape} and mask {self._mask.shape} differ in voxel grid ({image_path})")

    @property
    def image(self) -> nib.Nifti1Image:
        """The defaced native-space image; it owns the geometry both consumers preserve."""
        return self._image

    @property
    def mask(self) -> nib.Nifti1Image:
        """The binary HD-BET brain mask, on the image's grid."""
        return self._mask

    @property
    def image_path(self) -> Path:
        """Where the image was read from; the off-grid error names it, and so does any caller report."""
        return self._image_path
