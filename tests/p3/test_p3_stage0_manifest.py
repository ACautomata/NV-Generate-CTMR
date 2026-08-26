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

"""Pytest wrapper for the P3 stage-0 manifest selftest (issue #104 / ADR-0013 §2).

Calls the resident ``Stage0PlanSelfTest`` directly (the implementation stays in
the production script; the ``selftest`` subcommand remains the sugon-side
integration-gate entry and must not forward pytest).
"""

from scripts.brats_p3_stage0_manifest import Stage0PlanSelfTest


def test_p3_stage0_manifest_selftest(tmp_path):
    failures = Stage0PlanSelfTest(tmp_path).run()

    assert failures == []
