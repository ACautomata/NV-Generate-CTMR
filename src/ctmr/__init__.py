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

"""ctmr — src layout package (issue #103 / ADR-0013; installed since #130).

Installable via ``pip install -e . --no-deps`` (ADR-0015 §3, console entry
``ctmr``); that editable install is the single import track for CI and local
dev -- the pytest ``pythonpath`` double-track is retired (issue #143). The deep
modules land with their
convergence-gate tests ("born with tests", ADR-0013 §5), layered per
ADR-0015 §2 since #133:

- ``ctmr.domain``     — pure logic: grid / measurement / instrument_spec /
                        recipe / identity / generation entities      (#133, #170-174)
- ``ctmr.application`` — use-case orchestration: generation families, shells,
                        acceptance layers                             (ticket 08-10)
- ``ctmr.infrastructure`` — IO/framework adapters: checkpoints, nnunet_runner,
                        vendored maisi_engine, dataio                 (#134, #140)

The legacy phase-shell and instrument-execution forwarding shims retired
with issue #175 (ADR-0016 M5); their canonical homes are
``ctmr.application`` (train_cli / shell) and
``ctmr.infrastructure.nnunet_runner`` (``ctmr measure predict``).
"""
