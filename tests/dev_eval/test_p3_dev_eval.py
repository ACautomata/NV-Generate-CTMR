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

"""Pytest wrapper for the P3 dev-eval selftest (issue #104 / ADR-0013 §2).

Calls the resident ``P3DevEvalSelfTest`` directly (the implementation stays in
the production script; the ``selftest`` subcommand remains the sugon-side
integration-gate entry and must not forward pytest). Torch-level: runs without
a GPU but imports torch, so it skips itself on light stacks via
``pytest.importorskip`` (ADR-0013 §4).
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")  # transitively imported at module level via diff_model_setting / utils_infer

from scripts.brats_p3_dev_eval import P3DevEvalSelfTest  # noqa: E402  (importorskip must precede the torch-dependent import)


@pytest.mark.torch
def test_p3_dev_eval_selftest(tmp_path):
    failures = P3DevEvalSelfTest(tmp_path).run()

    assert failures == []
