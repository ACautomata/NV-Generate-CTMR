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

"""pytest option wiring for the two-tier test surface (ADR-0015 §6).

``torch``-marked tests are CPU-runnable and always execute for real (the CI
full-dependency tier installs torch + monai + nnunetv2). ``gpu``-marked tests
need a real GPU/cluster host: they stay auto-skipped locally and in CI and run
on servers explicitly via ``pytest --run-gpu``.
"""

from __future__ import annotations

import argparse

import pytest


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption("--run-gpu", action="store_true", default=False, help="execute gpu-marked tests (DCU/cluster hosts only)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="gpu-marked test: needs a real GPU/cluster host; execute there with pytest --run-gpu (ADR-0015 §6)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
