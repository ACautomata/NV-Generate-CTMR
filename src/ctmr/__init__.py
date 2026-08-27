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
``ctmr``); the pytest ``pythonpath = ["src", "."]`` track stays until the
migration batches close it out. The deep modules land with their
convergence-gate tests ("born with tests", ADR-0013 §5):

- ``ctmr.grid``       — instrument input geometry      (ADR-0008, #105)
- ``ctmr.instrument`` — frozen instrument command       (ADR-0009, #107)
- ``ctmr.measure``    — instrument measurement          (ADR-0010, #109)
- ``ctmr.harness``    — phase script shells             (ADR-0011, #111)
"""
