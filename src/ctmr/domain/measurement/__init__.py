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

"""Instrument measurement, the unique module (ADR-0010, issue #109).

``regions`` is the single source of the WT / TC / ET label projection rules
(``REGIONS`` + the derived ``REGION_NAMES`` tuple form + ``RegionMasks``);
``metrics`` holds the shared primitives (``DiceScore`` with the single
empty-denominator ``None`` sentinel, ``WilsonUpper`` with the single ``n == 0``
guard); ``hierarchy`` separates the canonical containment
``HierarchyChecker.violates`` from the calibration case-usability gate
``CalibrationCaseUsability`` (two old ``hier_viol`` meanings, one per class);
``measurement`` is the canonical ``CaseMeasurement`` value object with the
calibration long / terminal-acceptance wide serializers; ``measurer`` is the
unique entry ``InstrumentMeasurer.measure(pred, *, gt, condition, brain)``
composing the collaborators. Pure transforms: numpy in, value object out, no
file IO, no cluster or path coupling -- readers stay with the callers.

The package re-exports the public surface; all modules are numpy/scipy-level
(deps of the light test stack, ADR-0013 §4) so the born-with convergence gates
run on any machine.
"""

from ctmr.domain.measurement.hierarchy import CalibrationCaseUsability, HierarchyChecker
from ctmr.domain.measurement.measurement import CALIBRATION_FIELDS, FINAL_ACCEPTANCE_FIELDS, CaseMeasurement, GtRegionMetrics
from ctmr.domain.measurement.measurer import InstrumentMeasurer
from ctmr.domain.measurement.metrics import DiceScore, WilsonUpper
from ctmr.domain.measurement.regions import LABEL_DOMAIN, REGION_NAMES, REGIONS, RegionMasks

__all__ = [
    "CALIBRATION_FIELDS",
    "CalibrationCaseUsability",
    "CaseMeasurement",
    "DiceScore",
    "FINAL_ACCEPTANCE_FIELDS",
    "GtRegionMetrics",
    "HierarchyChecker",
    "InstrumentMeasurer",
    "LABEL_DOMAIN",
    "REGION_NAMES",
    "REGIONS",
    "RegionMasks",
    "WilsonUpper",
]
