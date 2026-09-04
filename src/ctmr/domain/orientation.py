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

"""The RAS direction world, the single direction-unification point (ADR-0020, #314).

The whole chain declares RAS as its direction world -- the NVIDIA upstream
convention, evidenced in the initial commit (``Orientationd(axcodes="RAS")``
across the training-data, transforms and sampling chains). Before ADR-0020 the
instrument chain straddled two worlds: the generated side was unconditionally
x/y-flipped onto LPS while the real side passed through in its native orientation
(BraTS 2023 is ~89% LPS / ~11% RAS), so every RAS-coded real case measured
against a mirrored instrument input. The flip compensation is retired; both
sides now enter the same RAS world and the misalignment class is eliminated by
construction.

Pure in-memory ``sitk.Image`` transforms -- no file IO, no paths (readers stay
with the callers, as in ``ctmr.domain.grid``). ``SimpleITK.DICOMOrient`` realises
the affine-driven permute/flip for axis-aligned volumes; the axis-aligned
boundary is asserted here explicitly because the sitk filter itself does not
raise on oblique input (verified against SimpleITK 2.5.6) -- an oblique volume
is a loud failure, never a silent approximation.

"""

import numpy as np
import SimpleITK as sitk


class NotRasWorldError(Exception):
    """A volume cannot be placed on (or is not already in) the RAS direction world."""


class RasOrientation:
    """Direction-world service: unify an axis-aligned volume onto RAS, or assert
    that a volume already is RAS.

    ``to_ras`` is the real-side/label entry: native orientation mixtures enter
    the RAS world with voxel-physical correspondence preserved (the affine-driven
    guarantee, machine-guarded over all 48 axis-aligned codings). ``require_ras``
    is the generated-side entry: the DM write protocol pins the generated NIfTI
    affine (``V1_DM_OUTPUT_GRID``, issue #249), so a non-RAS generated volume is
    an upstream protocol break -- asserted loudly, never silently corrected.
    """

    # sitk's physical frame is LPS, so the RAS world is the x/y-mirrored
    # identity: array +x -> R (physical -x), +y -> A, +z -> S.
    RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    TOLERANCE = 1e-3  # NIfTI stores float32 affines; leave room for their quantisation noise

    def to_ras(self, image: sitk.Image) -> sitk.Image:
        """Unifies an axis-aligned volume onto the RAS world (idempotent for RAS input)."""
        self._require_axis_aligned(image)
        oriented = sitk.DICOMOrient(image, "RAS")
        self.require_ras(oriented)
        return oriented

    def require_ras(self, image: sitk.Image) -> sitk.Image:
        """Asserts the volume already declares the RAS world; returns it unchanged."""
        if not np.allclose(image.GetDirection(), self.RAS_DIRECTION, atol=self.TOLERANCE):
            raise NotRasWorldError(
                f"volume direction {self._describe(image)} is not the RAS world "
                f"{self.RAS_DIRECTION}; the RAS write protocol (ADR-0020) expects RAS-declared volumes here"
            )
        return image

    def _require_axis_aligned(self, image: sitk.Image) -> None:
        matrix = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        absolute = np.abs(matrix)
        # direction cosines are orthonormal, so axis-aligned <=> the absolute matrix is a permutation
        if not (np.allclose(absolute.sum(axis=0), 1.0, atol=self.TOLERANCE) and np.allclose(absolute.sum(axis=1), 1.0, atol=self.TOLERANCE)):
            raise NotRasWorldError(
                f"volume direction {self._describe(image)} is not axis-aligned; "
                "the RAS unification covers permutation/flip codings only -- an oblique volume must be "
                "resampled onto an axis-aligned grid upstream"
            )

    @staticmethod
    def _describe(image: sitk.Image) -> tuple:
        """The direction tuple at display precision, for error messages."""
        return tuple(round(v, 4) for v in image.GetDirection())
