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
"""
