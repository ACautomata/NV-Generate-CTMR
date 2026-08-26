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

"""Pytest wrapper for the L2 HTML renderer selftest (issue #104 / ADR-0013 §2).

Calls the resident ``RendererSelfTest`` directly (stdlib + Pillow renderer; the
implementation stays in the production script and the ``selftest`` subcommand
remains the sugon-side integration-gate entry).
"""

from scripts.brats_p1_l2_html import RendererSelfTest


def test_p1_l2_html_renderer_selftest():
    assert RendererSelfTest().run() == 0
