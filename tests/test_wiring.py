# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composition-root gates (issue #270 / ADR-0019 §2).

``ctmr.wiring`` is the terminal composition root -- the one home of concrete
implementation knowledge, one module per subcommand family, outside the three
layers and parallel to ``cli.py``. Its family modules compose lazily
(importlib on dispatch, the ``cli.py`` discipline): importing the package
must pull no third-party dependency, so the light sci-stack CI job can always
import the composition root, and a family module that starts eagerly pulling
its adapters turns this probe red (``ctmr.wiring.measure`` is only reached
through the dispatch registry, a gap the cli purity gate cannot see). The
behavioral face of the train dispatch (spawn vs in-process worker entry) is
pinned through the CLI seam in
``tests/application/generation/modality_label/test_spawn.py`` -- these gates
pin the structure only.
"""


def test_wiring_imports_pull_no_third_party_dependency(light_import_probe):
    assert light_import_probe("ctmr.wiring, ctmr.wiring.generate, ctmr.wiring.measure, ctmr.wiring.contract, ctmr.wiring.distribution") == "[]"
