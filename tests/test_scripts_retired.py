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

"""Negative probe: the legacy ``scripts/`` namespace is gone by design (issue #143).

ADR-0015 M5 retires the top-level ``scripts/`` package outright -- git history is
the reproduction anchor for everything it held. These are the guard-suite probes
for acceptance criterion 2: the directory must be physically absent, and a fresh
interpreter launched from the repo root (so the root is on ``sys.path``) must not
resolve ``import scripts``.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scripts_directory_is_physically_gone():
    assert not (REPO_ROOT / "scripts").exists()


def test_scripts_namespace_is_not_importable():
    """A clean interpreter with the repo root on the path must fail to import scripts."""
    proc = subprocess.run([sys.executable, "-c", "import scripts"], cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "No module named 'scripts'" in proc.stderr
