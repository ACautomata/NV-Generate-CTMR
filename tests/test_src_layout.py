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

"""Test-surface infrastructure self-check (issue #103 / ADR-0013 §6, ADR-0015 §3).

Proves the install wiring: ``pip install -e .`` (editable, ADR-0015 §3) maps the
``ctmr`` package onto ``src/``, so ``import ctmr`` resolves to the live source
tree -- no pytest ``pythonpath`` key and no conftest ``sys.path`` hacks. Because
the install is editable, the module file is the real source, not a site-packages
snapshot.
"""

import ctmr


def test_ctmr_resolves_to_the_live_src_tree():
    assert ctmr.__file__.endswith("src/ctmr/__init__.py")
