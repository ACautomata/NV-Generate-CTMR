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

"""Skull-stripped derivation: ``image x brain-mask -> nii.gz`` (spec #13 section 3.2, ticket T6 #22,
new-code list item 3 "~20 行").

MR-RATE publishes only whole-brain volumes plus HD-BET brain masks, so the skull-stripped
twin of every replay volume is made here instead of downloaded (section 3.3 "双产": one
source volume becomes a whole-brain training entry and a skull-stripped one).  The mask is
binary and shares the image's voxel grid, so the derivation is a single elementwise
multiply -- no resampling, no interpolation, no intensity normalization: the intensity
transform of the original preprocessing pipeline runs later, on the encoded side.

Deliberately dependency-light (nibabel + numpy) and callable on its own, next to
``scripts.spine_filter``, which applies the same image/mask pair's verdict.

Usage (single pair, for inspection and manual repair)::

    python -m scripts.create_skull_stripped \\
        --image  mri/batch00/<study>/img/<study>_<series>.nii.gz \\
        --mask   mri/batch00/<study>/seg/<study>_<series>_brain-mask.nii.gz \\
        --output mri/batch00/<study>/img/<study>_<series>_skull_stripped.nii.gz

Programmatic use::

    from scripts.create_skull_stripped import SkullStrippedCreator
    SkullStrippedCreator().create(image_path, mask_path, output_path)
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


class SkullStrippedCreator:
    """Multiplies an image by its brain mask, preserving the image's geometry and dtype."""

    def create(self, image_path: Path, mask_path: Path, output_path: Path) -> Path:
        """Write ``image x mask`` to ``output_path``; raise ValueError if the pair is off-grid."""
        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))
        if image.shape != mask.shape:
            raise ValueError(f"image {image.shape} and mask {mask.shape} differ in voxel grid ({image_path})")
        image_data = np.asanyarray(image.dataobj)
        mask_data = np.asanyarray(mask.dataobj) != 0
        stripped = image_data * mask_data.astype(image_data.dtype)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(stripped, affine=image.affine, header=image.header), str(output_path))
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="defaced native-space image NIfTI")
    parser.add_argument("--mask", type=Path, required=True, help="HD-BET brain-mask NIfTI (same voxel grid)")
    parser.add_argument("--output", type=Path, required=True, help="path of the skull-stripped NIfTI to write")
    args = parser.parse_args()

    written = SkullStrippedCreator().create(args.image, args.mask, args.output)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
