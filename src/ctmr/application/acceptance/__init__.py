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

"""``ctmr.application.acceptance`` -- the acceptance base (#136): the
run-contract identity wiring of the three-layer acceptance.

- ``registry`` -- the acceptance-layer registry (ADR-0012 decision 2): one
  declaration point for layer kind / name / report schema, with every kind
  relation derived; judgement stays out by design (ADR-0006 dual-source).
- ``binding`` -- the frozen-run five-key binding (ADR-0012 decision 4): the
  single extraction point of a run record's candidate identity, require-frozen
  gate built in.

Layer-name subpackages (quantitative / distribution / expert_review) and the
contract face move in with their own tickets (#140/#142/#139 family) -- do not
pre-create them. Stdlib-only, importable on any machine.
"""
