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

"""Pinned MR intensity protocol -- pure logic, no IO (ADR-0015 §2, ticket 08).

``MRIntensityNormalizer`` is the frozen per-volume 0-99.5 percentile -> [0, 1]
protocol (``percentile_0_99.5_to_0_1``, ADR-0002/ADR-0010 family) that the L1
paired metrics and the dev-side PSNR/SSIM trend share. Moved here from
``scripts/brats_l1_quantitative.py`` so the application layer can consume the
protocol without depending on the retiring scripts; the legacy module keeps a
thin forwarding shim. Verified dtype/shape/finiteness gating raises
``IntensityProtocolError`` (message text unchanged from the legacy
``L1QuantitativeError`` wording; the exception class name is the only
difference -- nothing in the legacy chain catches it).
"""

from __future__ import annotations

import numpy as np


class IntensityProtocolError(Exception):
    """Raised when a volume breaks the pinned MR [0, 1] intensity protocol."""


class MRIntensityNormalizer:
    """Applies the fixed per-volume MR 0–99.5 percentile intensity protocol."""

    def normalize(self, volume, label):
        data = np.asarray(volume, dtype=np.float64)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise IntensityProtocolError(f"{label} must be a finite 3D MR volume")
        lower, upper = np.percentile(data, (0.0, 99.5))
        if upper <= lower:
            raise IntensityProtocolError(f"{label} has no usable 0–99.5 percentile intensity range")
        return np.clip((data - lower) / (upper - lower), 0.0, 1.0)
