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
on servers explicitly via ``pytest --run-gpu`` or ``CTMR_RUN_GPU_TESTS=1``
(the two opt-in spellings are interchangeable).

Also home of the shared light-stack import probe: the stdlib-only purity
gates (``tests/test_cli_entry``, ``tests/test_wiring``) run one import
statement in a fresh interpreter and assert none of the heavy needles load
(ADR-0013 §4 light sci-stack CI job).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The heavy needles: none of these stacks may load merely by importing a
# gated surface (cli, the composition root). One list, so the gates cannot
# silently diverge in what they consider "heavy".
_LIGHT_STACK_NEEDLES = [
    "torch",
    "monai",
    "numpy",
    "scipy",
    "skimage",
    "nibabel",
    "SimpleITK",
    "PIL",
    "matplotlib",
    "einops",
    "huggingface_hub",
    "tqdm",
    "fire",
    "tensorboard",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def light_import_probe():
    """Run one import statement in a fresh interpreter (PYTHONPATH=src) and
    return the sorted heavy needles it loaded -- ``"[]"`` is clean."""

    def probe(import_statement):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {import_statement}, sys\nprint(sorted(name for name in sys.argv[1:] if name in sys.modules))\n",
                *_LIGHT_STACK_NEEDLES,
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")},
        )
        assert result.returncode == 0, f"import probe failed for {import_statement!r}:\n{result.stderr}"
        return result.stdout.strip()

    return probe


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption("--run-gpu", action="store_true", default=False, help="execute gpu-marked tests (DCU/cluster hosts only)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu") or os.environ.get("CTMR_RUN_GPU_TESTS"):
        return
    skip_gpu = pytest.mark.skip(
        reason="gpu-marked test: needs a real GPU/cluster host; opt in there with pytest --run-gpu or CTMR_RUN_GPU_TESTS=1 (ADR-0015 §6)"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
