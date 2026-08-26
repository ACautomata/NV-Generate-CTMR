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

"""Generic volume-grid geometry engine (ADR-0008, issue #105).

Pure transforms on in-memory ``sitk.Image`` -- no file IO, no cluster or
controlled-path coupling (readers/writers stay with the callers). The frozen
terminal-acceptance geometry (the ``GeneratedVolumeResampler`` private methods
of ``scripts/nnunet_l2_final_acceptance_nifti.py``, pre-#105) is the
convergence standard this engine was extracted from, verbatim.
"""

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class TargetGrid:
    """Target grid value object: size + spacing, both in sitk xyz axis order."""

    size: tuple[int, int, int]
    spacing: tuple[float, float, float]


class GridResampler:
    """Resamples a volume onto the grid's spacing, with the interpolation
    strategy injected at construction (B-spline / nearest neighbour / linear).

    ``grid.spacing`` drives the output sampling; the output size is the rounded
    physical extent of the input. ``grid.size`` is realised by the paired
    CenterCropOrPad, not here.
    """

    def __init__(self, interpolator: int):
        self._interpolator = interpolator

    def resample(self, image: sitk.Image, grid: TargetGrid) -> sitk.Image:
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_spacing = list(grid.spacing)
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(self._interpolator)
        return resampler.Execute(image)


class CenterCropOrPad:
    """Centred crop/pad onto the grid, axis-order-correct for the zyx array
    layout: slices are built in xyz order and applied reversed (the #38
    original applied xyz slices to a zyx array -- the drift ADR-0008 unifies).
    """

    def crop_or_pad(self, image: sitk.Image, grid: TargetGrid) -> sitk.Image:
        size = image.GetSize()
        target_size = grid.size
        array = sitk.GetArrayFromImage(image)  # z, y, x
        cropped = np.zeros(tuple(reversed(target_size)), dtype=array.dtype)  # array axes are zyx
        src_slices, dst_slices = [], []
        for s, t in zip(size, target_size):  # slices built in xyz order, applied reversed below
            if s >= t:
                start = (s - t) // 2
                src_slices.append(slice(start, start + t))
                dst_slices.append(slice(0, t))
            else:
                start = (t - s) // 2
                src_slices.append(slice(0, s))
                dst_slices.append(slice(start, start + s))
        cropped[tuple(reversed(dst_slices))] = array[tuple(reversed(src_slices))]
        result = sitk.GetImageFromArray(cropped)
        result.SetSpacing(grid.spacing)
        result.SetOrigin(image.GetOrigin())
        result.SetDirection(np.eye(3).flatten().tolist())
        return result
