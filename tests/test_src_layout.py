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

"""Test-surface infrastructure self-checks (issue #103 / ADR-0013).

Proves the pytest wiring: ``pythonpath = ["src", "."]`` in pyproject puts the
source root on ``sys.path``, so ``import ctmr`` works on any machine without
installing a package and without conftest ``sys.path`` hacks.
"""

import ctmr


def test_ctmr_importable_from_src_without_install():
    assert ctmr.__file__.endswith("src/ctmr/__init__.py")
