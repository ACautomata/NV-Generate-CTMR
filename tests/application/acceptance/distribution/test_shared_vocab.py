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

"""Convergence-gate tests for the L2 shared vocabulary (ADR-0017 decisions 1/2/4, #229 + #231).

The judge's shared vocabulary lives in three stdlib-only modules inside the
distribution package -- ``measurement_table`` (measurement-face vocabulary +
the wide 27-column CSV protocol), ``statistics`` (the rel-diff primitive,
quantile read-out + cluster bootstrap) and ``challenge_registry`` (challenges /
holdout quotas / unified seed band / ADR-0002 envelope literals) -- and
``final_acceptance`` imports them instead of hosting them. Gates: each module's
dependency closure stays third-party-free (the #222 meta-path probe, with its
negative control) -- the judge's own closure included (ADR-0017 decision 2:
any machine can render the verdict) --, the judge's names are the shared
modules' objects (imports, not copies), the judge hosts no private statistics
primitive any more (the Wilson formula copy and the inlined rel-diff branch
converged onto the shared definitions, #231), the judge's ctmr import face
stays registered (the judgement chain draws only on the vocabulary leaf and
the shared modules; the stdlib-only assembly/CLI pieces are enumerated), the
frozen values stay pinned at their new home, the judge's region tuple derives
from the vocabulary leaf, and no module outside the judge and the shared
vocabulary imports the judge any more (an AST-based package-wide guard: the
judge is no longer the shared-vocabulary host).
"""

import ast
import subprocess
import sys
from pathlib import Path

import ctmr.application.acceptance.distribution.challenge_registry as challenge_registry
import ctmr.application.acceptance.distribution.diagnostic_support as diagnostic_support
import ctmr.application.acceptance.distribution.et_discrimination as et_discrimination
import ctmr.application.acceptance.distribution.final_acceptance as final_acceptance
import ctmr.application.acceptance.distribution.measurement_table as measurement_table
import ctmr.application.acceptance.distribution.statistics as statistics
import ctmr.application.acceptance.distribution.zcrop_compensation as zcrop_compensation
import ctmr.domain.vocabulary as vocabulary

PACKAGE_DIR = Path(final_acceptance.__file__).parent
JUDGE_MODULE = final_acceptance.__name__
JUDGE_PACKAGE = "ctmr.application.acceptance.distribution"
# The judge itself and the shared-vocabulary modules it draws its imports from;
# every other module in the package must not import the judge. Scanning the
# whole package (minus this list) keeps new diagnostic/execution modules
# covered without registering them by hand.
JUDGE_FREE_WHITELIST = frozenset({"__init__", "challenge_registry", "final_acceptance", "measurement_table", "statistics"})

# The stdlib-only closure probe: a fresh interpreter whose meta-path blocker
# makes every non-stdlib (and non-ctmr) top-level import fail, then imports
# the module under test. Prints IMPORTED_OK when the import survives,
# BLOCKED when the blocker fired -- a guard that cannot fail is not a guard,
# so the blocker itself gets a negative control (numpy must be blocked).
CLOSURE_PROBE = """
import sys

preexisting = set(sys.modules)  # interpreter-startup imports (editable-install hooks) are not the module's doing
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


def test_shared_vocabulary_dependency_closures_are_third_party_free():
    for module_name in (
        "ctmr.application.acceptance.distribution.measurement_table",
        "ctmr.application.acceptance.distribution.statistics",
        "ctmr.application.acceptance.distribution.challenge_registry",
        "ctmr.application.acceptance.distribution.final_acceptance",  # the judge itself (ADR-0017 decision 2)
        "ctmr.application.acceptance.distribution.diagnostic_support",  # the diagnostic support pieces (#232)
    ):
        probe = run_closure_probe(module_name)
        assert probe.returncode == 0, probe.stderr
        assert "IMPORTED_OK" in probe.stdout, f"{module_name}: {probe.stdout}"  # numpy / scipy / torch unreachable


def test_closure_probe_blocker_catches_third_party_imports():
    probe = run_closure_probe("numpy")
    assert probe.returncode == 0
    assert "BLOCKED" in probe.stdout and "IMPORTED_OK" not in probe.stdout


def test_judge_names_are_the_shared_modules_objects_not_copies():
    # The judge imports exactly the shared names its judgement chain consumes
    # (MEASUREMENT_FIELDS has no judge-side consumer since the protocol moved,
    # so it is deliberately absent from the judge's import face).
    assert final_acceptance.AcceptanceError is measurement_table.AcceptanceError
    assert final_acceptance.MeasurementTable is measurement_table.MeasurementTable
    assert final_acceptance.MODALITIES is measurement_table.MODALITIES
    assert final_acceptance.CHANNEL_SUFFIXES is measurement_table.CHANNEL_SUFFIXES
    assert not hasattr(final_acceptance, "MEASUREMENT_FIELDS")
    assert final_acceptance.ClusterBootstrap is statistics.ClusterBootstrap
    assert final_acceptance.RelativeDifference is statistics.RelativeDifference
    assert final_acceptance.WilsonUpper is vocabulary.WilsonUpper
    assert final_acceptance.CHALLENGES is challenge_registry.CHALLENGES
    assert final_acceptance.HOLDOUT_QUOTAS is challenge_registry.HOLDOUT_QUOTAS
    assert final_acceptance.BOOTSTRAP_B is challenge_registry.BOOTSTRAP_B
    assert final_acceptance.GLOBAL_SEED is challenge_registry.GLOBAL_SEED
    assert final_acceptance.CHALLENGE_SEED_OFFSET is challenge_registry.CHALLENGE_SEED_OFFSET
    assert final_acceptance.FROZEN_ENVELOPES is challenge_registry.FROZEN_ENVELOPES


def test_shared_registry_pins_the_frozen_values():
    assert challenge_registry.CHALLENGES == ("GLI", "SSA", "MEN", "METS", "PED")
    assert challenge_registry.HOLDOUT_QUOTAS == {"GLI": 250, "SSA": 12, "MEN": 200, "METS": 48, "PED": 20}
    assert challenge_registry.BOOTSTRAP_B == 10_000
    assert challenge_registry.GLOBAL_SEED == 20260821
    assert challenge_registry.CHALLENGE_SEED_OFFSET == {"GLI": 1, "SSA": 2, "MEN": 3, "METS": 4, "PED": 5}
    # the diagnostic seed namespace (ADR-0017 decision 5, #232): base, band
    # width and the job A/B slot table, byte-exact against the pre-#232 modules;
    # the dev monitor (#253) joins at the next free block 400
    assert challenge_registry.DIAGNOSTIC_SEED_BASE == 900_000_000
    assert challenge_registry.DIAGNOSTIC_SEED_BAND == 1000
    assert challenge_registry.DIAGNOSTIC_SEED_SLOTS == {
        "zcrop_vol_uncomp": 0,
        "zcrop_centroid_uncomp": 1,
        "zcrop_vol_comp": 100,
        "zcrop_centroid_comp": 101,
        "et_rel_diff": 200,
        "dev_monitor_wt_rel_diff": 400,
    }
    # the ADR-0002 published 4-dp literals, verbatim (spot anchors; the full
    # table is pinned by the judge's own envelope tests)
    assert challenge_registry.FROZEN_ENVELOPES["GLI"]["WT"] == (0.8053, 0.2802, 5.38)
    assert challenge_registry.FROZEN_ENVELOPES["METS"]["TC"] == (0.0000, 1.0000, 35.08)
    assert challenge_registry.FROZEN_ENVELOPES["PED"]["r_fail_upper"] == 0.0507


def test_judge_region_tuple_derives_from_the_vocabulary_leaf():
    assert final_acceptance.REGIONS is vocabulary.REGION_NAMES
    assert not hasattr(final_acceptance, "REGION_LABELS")  # the mirror literal lost its reason to exist


def test_diagnostic_jobs_import_the_support_pieces_instead_of_copies():
    """ADR-0017 decision 6 / issue #232: the diagnostic jobs hold no local
    DiagnosticError or seed constants any more -- the support module and the
    registry are the single homes."""
    assert zcrop_compensation.DiagnosticError is diagnostic_support.DiagnosticError
    assert et_discrimination.DiagnosticError is diagnostic_support.DiagnosticError
    assert not hasattr(zcrop_compensation, "DIAGNOSTIC_SEED_BASE")
    assert not hasattr(zcrop_compensation, "COMPENSATED_SEED_STRIDE")
    assert not hasattr(et_discrimination, "DIAGNOSTIC_SEED_BASE")
    assert not hasattr(et_discrimination, "JOB_B_SEED_SLOT")


# ── the judge hosts no private statistics primitive (#231, ADR-0017 decision 4) ──

# The ctmr modules the judge may import: the judgement chain draws ONLY on the
# vocabulary leaf and the shared modules; the stdlib-only assembly/CLI pieces it
# binds to (instrument spec, contract artifacts/binding) are registered here --
# anything else in ctmr must stay out of the judge's import face until a guard
# change registers it deliberately.
JUDGE_IMPORT_FACE = frozenset(
    {
        "ctmr.domain.vocabulary",
        "ctmr.domain.instrument_spec",
        "ctmr.application.acceptance.contract.artifacts",
        "ctmr.application.acceptance.contract.binding",
        "ctmr.application.acceptance.distribution.measurement_table",
        "ctmr.application.acceptance.distribution.statistics",
        "ctmr.application.acceptance.distribution.challenge_registry",
    }
)


# AST-level markers of the judge's former private statistics primitives: the
# Wilson formula copy (its frozen z-value and its function) and the inlined
# rel-diff branch. Their single homes are the vocabulary leaf's ``WilsonUpper``
# and ``statistics.RelativeDifference``. Parsing (not text matching) keeps
# docstring or comment mentions of the names from tripping the guard.
def private_statistics_copies(source):
    """AST markers of private statistics-primitive copies present in one source text."""
    markers = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "Z95" for target in node.targets):
            markers.append("assignment to Z95 (the frozen Wilson z-value lives only in the vocabulary leaf)")
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "wilson_upper":
            markers.append("def wilson_upper (the Wilson formula copy lives only in the vocabulary leaf's WilsonUpper.of)")
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.BinOp)
            and isinstance(node.left.op, ast.Sub)
            and isinstance(node.right, ast.Name)
            and node.right.id == "real"
        ):
            markers.append("inlined (gen - real) / real (the rel-diff primitive lives in statistics.RelativeDifference.of)")
    return markers


def test_judge_hosts_no_private_statistics_primitive_copies():
    source = Path(final_acceptance.__file__).read_text(encoding="utf-8")
    assert private_statistics_copies(source) == []


def test_no_private_primitives_guard_detects_a_seeded_copy():
    seeded = "class FailureGate:\n    Z95 = 1.959963984540054\n\n    def wilson_upper(k, n):\n        return (gen - real) / real\n"
    markers = private_statistics_copies(seeded)
    assert len(markers) == 3 and all("vocabulary leaf" in marker or "RelativeDifference" in marker for marker in markers)


def judge_ctmr_import_roots(source):
    """Every ``ctmr``-rooted module path the source imports (absolute import shapes)."""
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and not node.level:
            module = node.module or ""
            if module == "ctmr" or module.startswith("ctmr."):
                roots.add(module)
        elif isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names if alias.name == "ctmr" or alias.name.startswith("ctmr."))
    return roots


def test_judge_import_face_stays_registered():
    source = Path(final_acceptance.__file__).read_text(encoding="utf-8")
    unregistered = judge_ctmr_import_roots(source) - JUDGE_IMPORT_FACE
    assert not unregistered, f"the judge imports ctmr modules outside its registered face: {sorted(unregistered)}"


def test_judge_import_face_guard_detects_a_seeded_violation():
    seeded = "from ctmr.application.acceptance.distribution.html_report import HtmlReport\n"
    assert judge_ctmr_import_roots(seeded) - JUDGE_IMPORT_FACE == {"ctmr.application.acceptance.distribution.html_report"}


def test_relative_difference_primitive_is_the_single_rel_diff_definition():
    of = statistics.RelativeDifference.of
    assert of(2.0, 4.0) == -0.5
    assert of(0.0, 4.0) == -1.0  # a generated-side empty prediction stays in the distribution (protocol §4)
    assert of(2.0, 0.0) is None  # a non-positive real denominator leaves the quantity undefined
    assert of(None, 4.0) is None and of(2.0, None) is None  # undefined sides never reach the arithmetic


def imports_the_judge(node):
    """Whether one AST node is any import shape reaching the judge module."""
    if isinstance(node, ast.ImportFrom):
        if (node.module or "").startswith(JUDGE_MODULE):
            return True
        if node.module == JUDGE_PACKAGE and any(alias.name == "final_acceptance" for alias in node.names):
            return True
        return bool(node.level) and (node.module or "") == "final_acceptance"  # from .final_acceptance import ...
    if isinstance(node, ast.Import):
        return any(alias.name.startswith(JUDGE_MODULE) for alias in node.names)
    return False


def judge_import_offenders(package_dir):
    """Every module in the distribution package (bar the judge and the shared
    modules it draws its imports from) whose AST still imports the judge."""
    offenders = []
    for path in sorted(Path(package_dir).glob("*.py")):
        if path.stem in JUDGE_FREE_WHITELIST:
            continue
        if any(imports_the_judge(node) for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))):
            offenders.append(path.stem)
    return offenders


def test_no_module_outside_the_judge_imports_it():
    offenders = judge_import_offenders(PACKAGE_DIR)
    assert not offenders, f"modules still import the judge: {offenders}"


def test_judge_free_guard_detects_seeded_violations(tmp_path):
    seeds = {
        "diagnostic_a.py": "from ctmr.application.acceptance.distribution.final_acceptance import CHALLENGES\n",
        "diagnostic_b.py": "from ctmr.application.acceptance.distribution import final_acceptance\n",
        "diagnostic_c.py": "from .final_acceptance import MeasurementTable\n",
    }
    for name, line in seeds.items():
        (tmp_path / name).write_text(line, encoding="utf-8")
    assert judge_import_offenders(tmp_path) == ["diagnostic_a", "diagnostic_b", "diagnostic_c"]
