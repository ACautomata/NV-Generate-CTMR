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

"""Package metadata guards (issue #130 / ADR-0015 §3).

The terminal-state facts read straight from pyproject.toml: metadata name
``ctmr``, requires-python >= 3.11, empty runtime dependencies (requirements.txt
owns the version pins), and the ``ctmr`` console entry point. The legacy
pytest pythonpath double-track must stay untouched until the migration
batches close it out.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def test_project_metadata_name_python_range_and_empty_runtime_deps():
    project = _pyproject()["project"]
    assert project["name"] == "ctmr"
    assert project["requires-python"] == ">=3.11"
    # Runtime deps stay empty; requirements.txt owns version locks (ADR-0015 §3).
    assert project["dependencies"] == []


def test_console_entry_point_registered():
    assert _pyproject()["project"]["scripts"]["ctmr"] == "ctmr.cli:main"


def test_legacy_pytest_pythonpath_track_untouched():
    ini_options = _pyproject()["tool"]["pytest"]["ini_options"]
    assert ini_options["pythonpath"] == ["src", "."]
