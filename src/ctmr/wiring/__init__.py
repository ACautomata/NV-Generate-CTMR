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

"""Composition root of the ctmr package (issue #270 / ADR-0019 §2).

The one home of concrete implementation knowledge, outside the three layers
and parallel to ``cli.py``: which adapter stands behind which verb and how
the runtime topology is assembled is decided here and nowhere else. One
module per subcommand family (``generate`` / ``measure`` today; ``accept``
arrives with its family's migration ticket) -- ``cli.py`` stays a pure
dispatcher that imports no infrastructure and assembles nothing, it only
calls into this package. The layer gate (ADR-0019 §1) exempts
``ctmr.wiring`` from the direction rules because the composition root is
admitted to everything.

Family modules compose lazily (importlib on dispatch -- the ``cli.py``
discipline): torch / monai / nnunetv2 load only when a verb actually runs,
and the dispatch-fake seeding in the tests relies on ``sys.modules`` being
consulted first. Only stdlib-light collaborator classes (the launcher) are
imported at module top. The family assemblies land with their migration
tickets (#271-#275): this skeleton hosts exactly the concrete knowledge that
used to live at the interface layer, called in the existing construction
manner -- behavior unchanged.
"""
