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

"""Convergence-gate tests for the stdlib-only vocabulary leaf (ADR-0017 decision 3, #222).

The leaf is the single definition point of the region/label vocabulary
(``REGIONS`` / ``REGION_NAMES`` / ``LABEL_DOMAIN``) and the frozen Wilson
constants (``WilsonUpper`` / ``Z95``); ``ctmr.domain.measurement`` re-exports
them so every existing consumer import is unchanged. Gates: the leaf's
dependency closure stays third-party-free (numpy / scipy / torch unreachable),
the measurement re-exports are the leaf's objects (not copies), and the frozen
values stay pinned at their new home. Wilson bit-for-bit parity keeps living
in ``tests/domain/measurement/test_metrics.py`` (unchanged; it runs against
the re-exported class).
"""

import subprocess
import sys

import ctmr.domain.measurement
import ctmr.domain.measurement.metrics
import ctmr.domain.measurement.regions
import ctmr.domain.vocabulary as vocabulary

# The drift anchor: the literal of every pre-#109 copy, verbatim (ADR-0010 #109).
FROZEN_REGIONS_LITERAL = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

# The stdlib-only closure probe: a fresh interpreter whose meta-path blocker
# makes every non-stdlib (and non-ctmr) top-level import fail, then imports
# the module under test. Prints IMPORTED_OK when the import survives,
# BLOCKED when the blocker fired -- a guard that cannot fail is not a guard,
# so the blocker itself gets a negative control (numpy must be blocked).
CLOSURE_PROBE = """
import sys

preexisting = set(sys.modules)  # interpreter-startup imports (editable-install hooks) are not the leaf's doing
allowed = preexisting | sys.stdlib_module_names | {"ctmr"}


class ThirdPartyBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in allowed:
            raise ImportError(f"stdlib-only closure violated: {fullname}")
        return None  # fall through to the normal finders


sys.meta_path.insert(0, ThirdPartyBlocker())

try:
    __import__(sys.argv[1])
except ImportError:
    print("BLOCKED:", sys.exc_info()[1])
else:
    print("IMPORTED_OK")
"""


def run_closure_probe(module_name):
    return subprocess.run([sys.executable, "-c", CLOSURE_PROBE, module_name], capture_output=True, text=True)


def test_vocabulary_dependency_closure_is_third_party_free():
    probe = run_closure_probe("ctmr.domain.vocabulary")
    assert probe.returncode == 0, probe.stderr
    assert "IMPORTED_OK" in probe.stdout  # numpy / scipy / torch all unreachable


def test_closure_probe_blocker_catches_third_party_imports():
    probe = run_closure_probe("numpy")
    assert probe.returncode == 0
    assert "BLOCKED" in probe.stdout and "IMPORTED_OK" not in probe.stdout


def test_measurement_reexports_are_the_leaf_objects_not_copies():
    for name in ("REGIONS", "REGION_NAMES", "LABEL_DOMAIN", "WilsonUpper"):
        assert getattr(ctmr.domain.measurement, name) is getattr(vocabulary, name)
    for name in ("REGIONS", "REGION_NAMES", "LABEL_DOMAIN"):
        assert getattr(ctmr.domain.measurement.regions, name) is getattr(vocabulary, name)
    assert ctmr.domain.measurement.metrics.WilsonUpper is vocabulary.WilsonUpper


def test_leaf_pins_the_frozen_vocabulary_values():
    assert vocabulary.REGIONS == FROZEN_REGIONS_LITERAL
    assert vocabulary.REGION_NAMES == ("WT", "TC", "ET")
    assert vocabulary.LABEL_DOMAIN == (0, 1, 2, 3)
    assert vocabulary.WilsonUpper.Z95 == 1.959963984540054
