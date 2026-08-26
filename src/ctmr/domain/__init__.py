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

"""``ctmr.domain`` -- the pure-logic layer (ADR-0015 section 2): no IO, no
process spawning; torch tensor math is allowed.

Born with the M2 batch of ADR-0015 section 10 (issue #133):

- ``ctmr.domain.grid``            -- instrument-grid geometry        (ADR-0008, #105; moved #133)
- ``ctmr.domain.instrument_spec`` -- frozen instrument command       (ADR-0009, #107; moved #133)
- ``ctmr.domain.measurement``     -- instrument measurement          (ADR-0010, #109; moved #133)
- ``ctmr.domain.recipe``          -- pinned-recipe guards            (ADR-0011; moved up from the shell, #133)
- ``ctmr.domain.identity``        -- sha256 weight-lineage entities  (ADR-0015 section 2; born with #133)

This package re-exports nothing: the modules have different dependency floors
(``grid`` needs SimpleITK, ``measurement`` needs numpy/scipy, ``recipe`` /
``identity`` are stdlib-only -- ADR-0013 section 4 light stack), so importing
one must not pay for another. Import submodules directly.
"""
