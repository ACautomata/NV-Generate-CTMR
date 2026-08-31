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

"""Distribution-alignment acceptance (formerly L2): the frozen-instrument judge chain.

Migrated verbatim from the retired scripts layer along its seven internal
import edges (issue
#140 / ADR-0015 §4): ``final_acceptance`` (assemble/predict/evaluate/verify)
with ``measurement_run`` (NIfTI execution side), ``closing`` /
``freeze_audit`` (frozen-artifact gates over ``instrument_training``), the
html report pair, and the calibration pair + synthetic-domain evaluator.
Every domain quantity (envelopes, pass lines, quotas) is frozen by
ADR-0002/0003/0004 -- this package changes addresses only.

The shared vocabulary the judge no longer hosts lives here too
(ADR-0017 decision 1, issue #229): ``measurement_table`` (measurement-face
words + the wide 27-column CSV protocol), ``statistics`` (the rel-diff
primitive ``RelativeDifference``, the quantile/mean read-out + cluster
bootstrap) and ``challenge_registry`` (challenges / holdout quotas /
unified seed band -- judge bootstrap band + diagnostic namespace, ADR-0017
decision 5 #232 -- / ADR-0002 envelope literals), plus the diagnostic support
pieces (``diagnostic_support``: the one DiagnosticError, the variant=
diagnostic report writer, the seed allocator; ADR-0017 decision 6). All four
are stdlib-only -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py`` -- and the
execution/diagnostic modules import them directly, never through the judge.
"""
