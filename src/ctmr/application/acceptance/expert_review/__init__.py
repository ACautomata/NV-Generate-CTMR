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

"""Expert-review acceptance layer: the blinded-package and judgment-aggregation chain.

Migrated verbatim from ``brats_l3_blind_eval.py`` (retired scripts layer, git history) (#141 / ADR-0015
§2): ``catalog`` (blinding protocol constants and the controlled image
catalog), ``package`` (the deterministic blinding-package renderer) and
``aggregate`` (visual-Turing / Likert / Fleiss aggregation into the
candidate-bound ``brats-l3-report/1`` conclusion). Stdlib only; the gate
constants mirror the contract-side validator from a different source on
purpose (ADR-0006 referee independence).
"""
