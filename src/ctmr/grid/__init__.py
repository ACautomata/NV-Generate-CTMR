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

"""ctmr.grid -- instrument input geometry, the unique module (ADR-0008, #105).

``geometry`` holds the generic engine: pure transforms, in-memory
``sitk.Image`` in and out, no file IO. ``instrument`` pins the instrument-grid
special case: the ``INSTRUMENT_GRID`` constant and the continuum / label
adapter factories (ADR-0008 decision 2 parameter table).
"""

from ctmr.grid.geometry import CenterCropOrPad, GridResampler, TargetGrid
from ctmr.grid.instrument import INSTRUMENT_GRID, InstrumentGridAdapter

__all__ = [
    "INSTRUMENT_GRID",
    "CenterCropOrPad",
    "GridResampler",
    "InstrumentGridAdapter",
    "TargetGrid",
]
