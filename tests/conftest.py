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

"""Marker skip rules for machine-dependent tiers (ADR-0015 §6).

``torch`` never skips around the mark itself: the CI full-dependency tier runs
those tests for real on CPU. ``gpu`` is the server tier: it additionally needs a
visible CUDA device AND an explicit opt-in (``CTMR_RUN_GPU_TESTS=1``), so
locally and in CI the gpu-marked tests auto-skip and execution rides the server
runbook instead.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CTMR_RUN_GPU_TESTS"):
        return
    skip = pytest.mark.skip(reason="gpu tests are opt-in server-tier checks: set CTMR_RUN_GPU_TESTS=1")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)
