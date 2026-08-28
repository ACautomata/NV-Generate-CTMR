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

"""ctmr.domain -- the pure-logic layer (ADR-0015 §2).

Pure transforms and value objects only: no IO, no process spawning -- torch /
numpy / SimpleITK / scipy computation is allowed. Readers, writers and every
subprocess stay with the callers (application / infrastructure). The deep
modules live here as of the migration batches (#133):

- ``grid``            -- instrument input geometry        (ADR-0008, #105)
- ``measurement``     -- instrument measurement           (ADR-0010, #109)
- ``instrument_spec`` -- frozen instrument command        (ADR-0009, #107)
- ``recipe``          -- pinned-recipe guards             (ADR-0011, #111)
- ``identity``        -- weight lineage, sha256-addressed (#133)
- ``generation``      -- generation behaviour entities incl. VAE objective (ADR-0016)

The package re-exports nothing: import the submodules directly.
"""
