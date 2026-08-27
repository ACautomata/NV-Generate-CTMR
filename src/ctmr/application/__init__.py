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

"""``ctmr.application`` -- the orchestration layer (ADR-0015 section 2):
use-case families, shells, generation-chain drivers; functional naming only,
stage codes never enter code.

Born ahead of the M4 batch with the acceptance base (#136): the run-contract
identity wiring -- the acceptance-layer registry and the frozen-run five-key
binding -- moves into ``ctmr.application.acceptance`` so generation-family
slices can adopt them before the whole judgement chain migrates.

This package re-exports nothing (same policy as ``ctmr.domain``): import
submodules directly. The remaining residents land batch by batch --
``generation`` / ``shell`` / ``vae_train`` plus the three acceptance layers
(quantitative / distribution / expert_review) with the contract face under
``acceptance.contract``.
"""
