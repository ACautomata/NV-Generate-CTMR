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

"""Instrument input geometry (InstrumentGridGeometry): the instrument-grid
special case of the generic engine (ADR-0008, issue #105).

Pins the frozen parameter table of ADR-0008 decision 2: continuum volumes go
B-spline with the ``SetDefaultPixelValue(GetPixelIDValue())`` background, label
volumes go nearest neighbour so no label values are invented; both centred
crop/pad onto ``INSTRUMENT_GRID``. Terminal-acceptance-only concerns (the DM
RAS->LPS axis flip, file IO) stay with that caller, not here.
"""

import SimpleITK as sitk

from ctmr.grid.geometry import CenterCropOrPad, GridResampler, TargetGrid

INSTRUMENT_GRID = TargetGrid(size=(240, 240, 155), spacing=(1.0, 1.0, 1.0))


class InstrumentGridAdapter:
    """Aligns a volume onto the instrument grid: resample + centred crop/pad.

    The interpolation strategy is fixed by the two named factories -- this is
    the frozen instrument contract, not a per-caller choice.
    """

    def __init__(self, interpolator: int):
        if interpolator not in (sitk.sitkBSpline, sitk.sitkNearestNeighbor):  # the pinned parameter table
            raise ValueError(
                "the instrument contract pins B-spline (continuum) and nearest neighbour (label); other strategies belong to the generic GridResampler"
            )
        self._resampler = GridResampler(interpolator)
        self._crop_or_pad = CenterCropOrPad()

    def align(self, image: sitk.Image) -> sitk.Image:
        resampled = self._resampler.resample(image, INSTRUMENT_GRID)
        return self._crop_or_pad.crop_or_pad(resampled, INSTRUMENT_GRID)

    @classmethod
    def continuum(cls) -> "InstrumentGridAdapter":
        """Continuum (generated) volumes: B-spline, the frozen standard."""
        return cls(sitk.sitkBSpline)

    @classmethod
    def label(cls) -> "InstrumentGridAdapter":
        """Label / condition masks: nearest neighbour -- invents no label values."""
        return cls(sitk.sitkNearestNeighbor)
