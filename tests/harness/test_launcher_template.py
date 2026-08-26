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

"""Structure gates for the parameterized launcher template (ADR-0011 decision 3, #111).

The template must keep the P1/P2 launch shape (train + dev-eval sidecar, nohup,
pid files) behind one ``PHASE`` dispatch, and the environment-json idempotence
guard (the P2-only pre-#111 fix that became a drift) must be built in -- the
single template is what makes the guard two-sided. A syntax check plus the
guard/structure assertions; the sugon execution itself is the #__T12__ window.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "scripts" / "brats_phase_launch_train.sh"


def test_template_exists_and_is_syntactically_valid():
    assert TEMPLATE.is_file()
    # bash -n catches syntax errors without executing; the template must not
    # require cluster paths for the syntax pass.
    result = subprocess.run(["bash", "-n", str(TEMPLATE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_unknown_phase_is_rejected():
    result = subprocess.run(
        ["bash", str(TEMPLATE)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PHASE": "p9", "REPO": str(REPO)},
    )
    assert result.returncode != 0
    assert "unknown PHASE" in result.stderr


def test_phase_dispatch_covers_p1_p2_p3():
    text = TEMPLATE.read_text()
    assert "p1)" in text and "p2)" in text and "p3)" in text
    assert text.count('PHASE="${PHASE:?set PHASE to p1|p2|p3}"') == 1


def test_idempotence_guard_is_built_into_the_template():
    text = TEMPLATE.read_text()
    # the pre-#111 P2-only guard must surround the environment-json write once
    # for the whole dispatch, not per copy
    assert text.count('if [ ! -f "$ENV_JSON" ]; then') == 1
    assert 'cat > "$ENV_JSON"' in text


def test_replay_and_dm_source_hooks_for_p1_p2():
    text = TEMPLATE.read_text()
    assert "--replay-list" in text  # P1 train extra
    assert 'DM_SOURCE_CKPT="${DM_SOURCE_CKPT:?set DM_SOURCE_CKPT' in text  # P2/P3 hard prerequisite


def test_old_per_stage_launchers_are_gone():
    for old in ("scripts/brats_p1_launch_train.sh", "scripts/brats_p2_launch_train.sh"):
        assert not (REPO / old).exists(), f"{old} must be deleted (merged into the template)"


def test_sidecar_instrument_results_and_p3_phase_root_are_in_the_template():
    text = TEMPLATE.read_text()
    # the five frozen L2 instrument result flags (p1/p2 sidecars) and the P3
    # dev-eval phase-root flag land in the dispatch
    assert '--instrument-results "GLI=' in text
    assert "--phase-root" in text
