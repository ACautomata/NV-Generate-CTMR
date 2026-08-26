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

"""Convergence-gate tests for the region label source of truth (ADR-0010, #109).

Pins ``REGIONS`` to the literal shared by all six drifted copies (the drift
anchor: this exact dict was re-written, ever so slightly differently, seven
times), the derived judge tuple form ``REGION_NAMES``, the label domain, and
the ``RegionMasks`` per-region boolean extraction.
"""

import numpy as np

from ctmr.measure.regions import LABEL_DOMAIN, REGION_NAMES, REGIONS, RegionMasks

# The drift anchor: the literal of every pre-#109 copy, verbatim. Do not edit --
# drift here is exactly what this gate exists to catch.
FROZEN_REGIONS_LITERAL = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}


def test_regions_pins_the_frozen_literal():
    assert REGIONS == FROZEN_REGIONS_LITERAL


def test_region_names_derives_the_judge_tuple_form():
    assert REGION_NAMES == ("WT", "TC", "ET")
    assert len(REGION_NAMES) == len(REGIONS)  # derived, not a second literal


def test_label_domain_covers_background_and_all_region_labels():
    assert set(LABEL_DOMAIN) == {0, *FROZEN_REGIONS_LITERAL["WT"]}
    assert (0, 1, 2, 3) == LABEL_DOMAIN


def test_region_masks_extracts_each_region_boolean_mask():
    array = np.array(
        [
            [[0, 0], [0, 2]],
            [[1, 3], [0, 0]],
        ],
        dtype=np.uint8,
    )
    masks = RegionMasks(array)
    assert masks.of("WT").tolist() == [[[False, False], [False, True]], [[True, True], [False, False]]]
    assert masks.of("TC").tolist() == [[[False, False], [False, False]], [[True, True], [False, False]]]
    assert masks.of("ET").tolist() == [[[False, False], [False, False]], [[False, True], [False, False]]]
    assert all(mask.dtype == np.bool_ for mask in (masks.of("WT"), masks.of("TC"), masks.of("ET")))
