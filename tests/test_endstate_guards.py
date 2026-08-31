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

"""End-state guard suite (issue #144 / ADR-0015 §9-§10, batch M7; issue #175 / ADR-0016 M4-M5; issue #230 / ADR-0018 gate 5; issue #268 / ADR-0019 §1).

Six terminal-state gates pin the post-migration repository shape:

1. live code and live docs carry zero references into the retired scripts
   layer (git history is the reproduction anchor); frozen historical corpora
   (adr / research / calibration), the deploy operations surface and the
   out-of-scope data directory are exempt;
2. shell scripts survive only under deploy/jobs and deploy/data;
3. no notebook sits at the repo root, and the package tree carries no
   codename / anonymous-library / hyphenated identifiers (ADR-0015 §7:
   P1/P2/P3 + L1/L2/L3 codes stay out of code names, utils-style names stay
   dead, hyphen compounds live only at the CLI verb layer);
4. the legacy generation addresses deleted with issue #175 — the harness /
   instrument forwarding shims and the domain-replaced engine drivers — stay
   out of live surfaces (ADR-0016 M4-M5);
5. every module under src/ctmr is reached by some in-repo reference, or is
   explicitly registered on the orphan whitelist (issue #230 / ADR-0018
   decision 5: the eleven provisioning/dataio retirees were alive in name --
   tested, green, yet called by nothing; orphan status becomes a declared
   state, never a silent one);
6. imports between the three layers run only in the ADR-0019 §1 admitted
   directions; the cross-layer edges frozen alive at guard birth are pinned
   by a violation ratchet that may only shrink during the B1 migration
   (issue #268).

Each gate runs two ways: a positive probe over the real repository (must be
clean) and a negative probe over a synthetic tree (a seeded violation must be
detected -- a guard that cannot fail is not a guard). This module is itself
exempt from gates 1 and 4's scan: it must spell the forbidden needles to find
them.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_MODULE_NAME = Path(__file__).name

# Gate 1 scan surface: live code + live docs (ADR-0015 §9). Frozen corpora
# (docs/adr, docs/research, docs/calibration), deploy/ (§5) and data/
# (out-of-scope) stay outside by construction -- only these globs are scanned.
LIVE_GLOBS = (
    "src/**/*.py",
    "tests/**/*.py",
    "docs/*.md",
    "docs/agents/*.md",
    ".claude/skills/**/SKILL.md",
)
LIVE_TOP_LEVEL_DOCS = ("README.md", "CONTEXT.md", "CLAUDE.md")

# Gate 2: the only operational homes for .sh recipes.
SHELL_HOMES = ("deploy/jobs", "deploy/data")

# Gate 3: banned name shapes inside the package tree -- stage/acceptance codes
# (P1/P2/P3, L1/L2/L3) and anonymous-library names (utils and friends), each
# as a full name or a name prefix; hyphens are banned outright (they cannot
# be imported and §7 confines hyphen compounds to the CLI verb layer).
BANNED_NAME_PATTERN = re.compile(r"(?i)^(p[123]|l[123]|utils|util|helpers|helper|common|misc|tools)(?![a-z0-9])")


def live_paths():
    """Every live-code / live-doc file covered by the scripts-reference gate."""
    paths = set()
    for pattern in LIVE_GLOBS:
        paths.update(REPO_ROOT.glob(pattern))
    paths.update(REPO_ROOT / name for name in LIVE_TOP_LEVEL_DOCS)
    return sorted(p for p in paths if p.is_file() and p.name != GUARD_MODULE_NAME)


def scan_for_needle(paths, needle):
    """(path, lineno, line) for every needle occurrence in the given files."""
    hits = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in line:
                hits.append((path, lineno, line))
    return hits


def stray_shell_scripts(root):
    """.sh files under root that live outside the two deploy operational homes."""
    strays = []
    for path in sorted(root.rglob("*.sh")):
        parts = path.relative_to(root).parts
        if any(part in (".git", ".venv", "__pycache__") for part in parts):
            continue
        rel = path.relative_to(root).as_posix()
        if not any(rel.startswith(home + "/") for home in SHELL_HOMES):
            strays.append(rel)
    return strays


def package_name_violations(root):
    """Name-level violations in a package tree: hyphens, codenames, jargon names.

    ``maisi_engine/`` is exempt: it is the vendored upstream engine kept as
    frozen copies (ADR-0015 §2), where even file names are pinned by the
    frozen-copy requirement -- the §7 naming rules govern names we chose, not
    upstream's.
    """
    violations = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if "maisi_engine" in rel.parts:
            continue
        if "__pycache__" in rel.parts:
            continue
        for part in rel.parts:
            if "-" in part:
                violations.append((rel.as_posix(), part, "hyphen in package path"))
            if BANNED_NAME_PATTERN.match(part.removesuffix(".py")):
                violations.append((rel.as_posix(), part, "codename/anonymous-library name"))
    return violations


def test_live_code_and_docs_carry_no_scripts_references():
    paths = live_paths()
    needles = ("scripts/", "`scripts.", "python -m scripts.", "python3 -m scripts.")
    hits = [hit for needle in needles for hit in scan_for_needle(paths, needle)]
    detail = "\n".join(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}" for path, lineno, line in hits)
    assert not hits, f"retired scripts-layer references survive in live surfaces:\n{detail}"


def test_scripts_reference_gate_detects_a_seeded_violation(tmp_path):
    stale = tmp_path / "notes.md"
    stale.write_text("see `scripts/inference.py` for the entry\n", encoding="utf-8")
    hits = scan_for_needle([stale], "scripts/")
    assert len(hits) == 1 and hits[0][1] == 1


# Gate 4: the legacy generation addresses deleted with issue #175 (the
# harness/instrument shim packages and the domain-replaced engine drivers)
# must not appear in any live surface (ADR-0016 M4-M5).  The instrument
# needles carry a trailing dot / space so the canonical
# ``ctmr.instrument_spec`` module and ``ctmr.domain.instrument_spec`` package
# are not flagged: the dot catches attribute/module-run spellings, the space
# catches ``from ctmr.instrument import ...``.
RETIRED_GENERATION_ADDRESSES = (
    "ctmr.harness",
    "ctmr.instrument.",
    "ctmr.instrument ",
    "maisi_engine.diff_model_train",
    "maisi_engine.img2img_infer",
)


def test_live_code_and_docs_carry_no_retired_generation_addresses():
    paths = live_paths()
    hits = [hit for needle in RETIRED_GENERATION_ADDRESSES for hit in scan_for_needle(paths, needle)]
    detail = "\n".join(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}" for path, lineno, line in hits)
    assert not hits, f"issue-#175-retired generation addresses survive in live surfaces:\n{detail}"


@pytest.mark.parametrize(
    ("needle", "seeded_line"),
    [
        ("ctmr.harness", "from ctmr.harness.train_shell import PhaseHarness\n"),
        ("ctmr.instrument.", "python -m ctmr.instrument.predict\n"),
        ("ctmr.instrument ", "from ctmr.instrument import predict\n"),
        ("maisi_engine.diff_model_train", "python -m ctmr.infrastructure.maisi_engine.diff_model_train\n"),
        ("maisi_engine.img2img_infer", "python -m ctmr.infrastructure.maisi_engine.img2img_infer\n"),
    ],
)
def test_retired_generation_address_gate_detects_a_seeded_violation(tmp_path, needle, seeded_line):
    stale = tmp_path / "notes.md"
    stale.write_text(seeded_line, encoding="utf-8")
    hits = scan_for_needle([stale], needle)
    assert len(hits) == 1 and hits[0][1] == 1


def test_shell_scripts_live_only_in_the_deploy_operational_homes():
    strays = stray_shell_scripts(REPO_ROOT)
    assert not strays, f".sh files outside {SHELL_HOMES}: {strays}"


def test_shell_gate_detects_a_seeded_stray(tmp_path):
    homes = tmp_path / "deploy" / "jobs"
    homes.mkdir(parents=True)
    (homes / "recipe.sh").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "stray.sh").write_text("# stray\n", encoding="utf-8")
    assert stray_shell_scripts(tmp_path) == ["stray.sh"]


def test_no_notebook_at_the_repo_root():
    notebooks = sorted(REPO_ROOT.glob("*.ipynb"))
    assert notebooks == [], f"notebooks at repo root: {notebooks}"


def test_notebook_gate_detects_a_seeded_notebook(tmp_path):
    (tmp_path / "tutorial.ipynb").write_text("{}", encoding="utf-8")
    assert sorted(tmp_path.glob("*.ipynb")) == [tmp_path / "tutorial.ipynb"]


def test_package_tree_carries_no_codename_jargon_or_hyphen_names():
    violations = package_name_violations(REPO_ROOT / "src" / "ctmr")
    detail = "\n".join(f"  {rel}: {part} ({why})" for rel, part, why in violations)
    assert not violations, f"codename/jargon/hyphen identifiers in the package tree:\n{detail}"


def test_name_gate_detects_seeded_violations(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "maisi_engine").mkdir(parents=True)
    (pkg / "p1_utils-foo.py").write_text("", encoding="utf-8")
    (pkg / "l2_measurement.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text("", encoding="utf-8")
    (pkg / "clean_name.py").write_text("", encoding="utf-8")
    (pkg / "maisi_engine" / "utils_frozen.py").write_text("", encoding="utf-8")
    flagged = {part for _rel, part, _why in package_name_violations(pkg)}
    assert flagged == {"p1_utils-foo.py", "l2_measurement.py", "utils.py"}


# Gate 5: the orphan-module whitelist (issue #230 / ADR-0018 decision 5). A
# module under src/ctmr that no in-repo reference reaches is an orphan: the
# provisioning/dataio retirees were exactly that -- born with green tests,
# yet called by nothing. The cleanup retired them together with their tests;
# any orphan left after the sweep must be registered here explicitly (orphan
# status becomes a declaration, not a silence) or retired by its own ruling.
# A reference is any in-repo reach: an import in src/ or tests/ (AST-parsed,
# relative imports resolved, ``from pkg import name`` counting only when that
# submodule spelling exists), a VERBS registry handler module, the
# console-script / ``python -m`` entry points, or a ``python -m ctmr...``
# module invocation in a deploy shell recipe -- the deploy surface is the
# only scanned non-Python reference home, because frozen corpora (adr /
# research / calibration) carry historical commands that are records, not
# reaches. Tests count as references because a module whose only reach is
# its own test file is exactly the "retired module still carried by its
# tests" state this gate polices: retiring a module means retiring its test
# with it, which drops the module to zero reaches and makes the gate fire
# until both are gone.
#
# First sweep (#230): the provisioning/dataio retirees left the tree, and the
# probe surfaced three legacy run modules the ADR had not listed -- each a
# one-shot controlled-execution runner whose historical run record is the
# reproduction anchor (CONTEXT.md "历史运行器"), none carried by its caller
# anymore. Retiring them needs its own ruling (ADR-0018's decision 1 list is
# closed); until then their orphan status is declared here and in the ADR's
# implementation note.
ORPHAN_WHITELIST: frozenset[str] = frozenset(
    {
        "ctmr.application.acceptance.distribution.calibration_prep",  # #36 calibration-set assembly, protocol frozen (ADR-0002)
        "ctmr.application.acceptance.distribution.freeze_audit",  # #37 frozen-artifact audit, verdict recorded in controlled storage
        "ctmr.application.acceptance.quantitative.fid_2d5",  # one-shot 2.5D FID calculator (dev-trend machinery uses quantitative.fid)
    }
)

PACKAGE_ROOT = REPO_ROOT / "src" / "ctmr"

SHELL_MODULE_INVOCATION = re.compile(r"(?:python3?)\s+-m\s+([\w.]+)")


def prefix_reaches(name, modules):
    """The spellings of one dotted name that exist as modules: ``import
    a.b.c`` needs a, a.b and a.b.c to exist, so every prefix hits."""
    parts = name.split(".")
    return {".".join(parts[:size]) for size in range(1, len(parts) + 1)} & set(modules)


def package_modules(package_root):
    """Every importable module name under the package -> its file."""
    modules = {}
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root.parent).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def import_bases(tree, containing_package):
    """(absolute base, statement) for every import statement in the AST,
    relative imports resolved against ``containing_package`` (the parent
    package for a module file, the package itself for ``__init__``)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            base = [] if node.module is None else node.module.split(".")
            if node.level:
                parts = containing_package.split(".")
                for _ in range(node.level - 1):
                    parts = parts[:-1]
                base = [*parts, *base]
            if base:
                yield ".".join(base), node


def import_reaches(tree, containing_package, modules):
    """Module names in ``modules`` that the AST's import statements reach.

    ``from pkg import name`` hits ``pkg.name`` only when that submodule
    spelling exists in the tree -- an attribute import is not a module reach.
    Relative imports resolve against ``containing_package`` (see
    ``import_bases``).
    """
    reaches = set()

    def hit(name):
        reaches.update(prefix_reaches(name, modules))

    for base, statement in import_bases(tree, containing_package):
        hit(base)
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                hit(".".join([base, alias.name]))
    return reaches


def file_references(path, module_name, modules):
    """The module names one .py file's imports reach (``module_name=None``
    scans a file outside the package -- tests imports are absolute)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if module_name is None or path.name == "__init__.py":
        containing_package = module_name or ""
    else:
        containing_package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    return import_reaches(tree, containing_package, modules)


def deploy_shell_references(modules, repo_root):
    """Module names invoked as ``python -m <module>`` inside deploy recipes."""
    reaches = set()
    for path in (repo_root / "deploy").rglob("*.sh"):
        for match in SHELL_MODULE_INVOCATION.finditer(path.read_text(encoding="utf-8")):
            reaches.update(prefix_reaches(match.group(1), modules))
    return reaches


def registry_references(modules):
    """The VERBS handler modules (the CLI registry's lazy-import strings)."""
    from ctmr.cli import VERBS

    reaches = set()
    for route in VERBS.values():
        reaches.update(prefix_reaches(route.module, modules))
    return reaches


def orphan_modules(package_root, repo_root=None):
    """Modules under ``package_root`` that no in-repo reference reaches."""
    modules = package_modules(package_root)
    referenced = set()
    for name, path in modules.items():
        referenced |= file_references(path, name, modules)
    # entry points: the pyproject console script (ctmr = ctmr.cli:main) and
    # the ``python -m ctmr`` execution form live outside the import graph
    for name in ("ctmr.cli", "ctmr.__main__"):
        if name in modules:
            referenced.add(name)
    if repo_root is not None:
        for path in repo_root.glob("tests/**/*.py"):
            referenced |= file_references(path, None, modules)
        referenced |= deploy_shell_references(modules, repo_root)
        referenced |= registry_references(modules)
    return set(modules) - referenced


def test_no_orphan_modules_outside_the_whitelist():
    orphans = orphan_modules(PACKAGE_ROOT, REPO_ROOT) - ORPHAN_WHITELIST
    assert not orphans, f"zero-caller modules outside the ADR-0018 whitelist: {sorted(orphans)}"


def test_whitelist_entries_are_current_orphans():
    stale = ORPHAN_WHITELIST - orphan_modules(PACKAGE_ROOT, REPO_ROOT)
    assert not stale, f"whitelist entries that are no longer orphans (remove them): {sorted(stale)}"


def test_orphan_gate_detects_a_seeded_orphan(tmp_path):
    pkg = tmp_path / "src" / "ctmr"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("import ctmr.live_mod\n", encoding="utf-8")
    (pkg / "imported_mod.py").write_text("THING = 1\n", encoding="utf-8")
    # both absolute and relative spellings reach imported_mod
    (pkg / "live_mod.py").write_text(
        "import ctmr.imported_mod\nfrom ctmr.imported_mod import THING\nfrom . import imported_mod as _\n",
        encoding="utf-8",
    )
    (pkg / "orphan_mod.py").write_text("UNUSED = 2\n", encoding="utf-8")
    assert orphan_modules(tmp_path / "src" / "ctmr") == {"ctmr.orphan_mod"}


# Gate 6: the layer-direction guard and its violation ratchet (issue #268 /
# ADR-0019 §1).  Between the three layers exactly one direction is admitted:
# application -> domain (ports), infrastructure -> domain (implementing the
# ports), every layer -> itself.  Forbidden: application -> infrastructure,
# domain -> any upper layer (application / wiring / interface),
# infrastructure -> application.  The composition root (``ctmr.wiring``, the
# address the frozen edges' construction is being hoisted to) and the tests
# are exempt by construction -- the scan surface is the package tree only.
# During the B1 migration the gate runs as a ratchet:
# FROZEN_VIOLATION_RATCHET pins the application -> infrastructure edges alive
# at guard birth (2026-08 audit: 46 edges across 21 files, every one
# application -> infrastructure).  A violation outside the frozen list goes
# red immediately; an entry that no longer violates is stale and must be
# removed -- the list may only shrink, never grow (same self-stabilizing
# pair as the orphan whitelist above).  When the ratchet reaches zero the
# list is deleted and the gate turns purely terminal-state (issue #10).
FROZEN_VIOLATION_RATCHET: frozenset[str] = frozenset(
    {
        "ctmr.application.acceptance.contract.artifacts -> ctmr.infrastructure.dmsource",
        "ctmr.application.acceptance.contract.conclude -> ctmr.infrastructure.dmsource",
        "ctmr.application.acceptance.contract.lifecycle -> ctmr.infrastructure.dmsource",
        "ctmr.application.acceptance.contract.record -> ctmr.infrastructure.dmsource",
        "ctmr.application.acceptance.contract.verify -> ctmr.infrastructure.dmsource",
        "ctmr.application.acceptance.distribution.closing -> ctmr.infrastructure.nnunet_runner",
        "ctmr.application.acceptance.distribution.instrument_training -> ctmr.infrastructure.nnunet_runner",
        "ctmr.application.acceptance.distribution.intensity_domain -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.acceptance.distribution.intensity_domain -> ctmr.infrastructure.maisi_engine.instance_definition",
        "ctmr.application.generation.cross_modal.anchor -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.cross_modal.baseline -> ctmr.infrastructure.maisi_engine.diff_model_infer",
        "ctmr.application.generation.cross_modal.baseline -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.cross_modal.baseline -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.cross_modal.baseline -> ctmr.infrastructure.maisi_engine.utils_infer",
        "ctmr.application.generation.cross_modal.baseline -> ctmr.infrastructure.weightsref",
        "ctmr.application.generation.cross_modal.candidate -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.cross_modal.candidate -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.cross_modal.candidate -> ctmr.infrastructure.maisi_engine.utils_infer",
        "ctmr.application.generation.cross_modal.candidate -> ctmr.infrastructure.weightsref",
        "ctmr.application.generation.cross_modal.monitor -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.cross_modal.monitor -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.cross_modal.monitor -> ctmr.infrastructure.maisi_engine.utils_infer",
        "ctmr.application.generation.cross_modal.train -> ctmr.infrastructure.bypass_mounting",
        "ctmr.application.generation.cross_modal.train -> ctmr.infrastructure.gradient_executors",
        "ctmr.application.generation.cross_modal.train -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.mask.monitor -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.mask.sample -> ctmr.infrastructure.dataio.augmentation",
        "ctmr.application.generation.mask.sample -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.mask.sample -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.mask.sample -> ctmr.infrastructure.maisi_engine.instance_definition",
        "ctmr.application.generation.mask.sample -> ctmr.infrastructure.maisi_engine.utils_infer",
        "ctmr.application.generation.mask.train -> ctmr.infrastructure.bypass_mounting",
        "ctmr.application.generation.mask.train -> ctmr.infrastructure.gradient_executors",
        "ctmr.application.generation.mask.train -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.modality_label.monitor -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.modality_label.monitor -> ctmr.infrastructure.maisi_engine.inference_primitives",
        "ctmr.application.generation.modality_label.monitor -> ctmr.infrastructure.maisi_engine.instance_definition",
        "ctmr.application.generation.modality_label.monitor -> ctmr.infrastructure.maisi_engine.utils_infer",
        "ctmr.application.generation.modality_label.token_swap_sampling -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.modality_label.train -> ctmr.infrastructure.bypass_mounting",
        "ctmr.application.generation.modality_label.train -> ctmr.infrastructure.gradient_executors",
        "ctmr.application.generation.modality_label.train -> ctmr.infrastructure.maisi_engine.diff_model_setting",
        "ctmr.application.generation.modality_label.train -> ctmr.infrastructure.maisi_engine.instance_definition",
        "ctmr.application.generation.train_loader -> ctmr.infrastructure.dataio.list_assembly",
        "ctmr.application.shell -> ctmr.infrastructure.checkpoints",
        "ctmr.application.shell -> ctmr.infrastructure.gradient_executors",
    }
)

# The layers above domain -- the targets a domain import is forbidden to
# reach (ADR-0019 §1: domain -> any upper layer).
DOMAIN_UPPER_LAYERS = ("application", "infrastructure", "wiring", "interface")


def layer_of(module_name):
    """The architecture layer of a dotted ctmr module name (``None`` for
    stdlib / third-party names).  The layer set is a closed enumeration
    (ADR-0019 §1): an unknown top-level name is an architecture change and
    must be registered here with its admitted directions."""
    parts = module_name.split(".")
    if parts[0] != "ctmr":
        return None
    if len(parts) == 1:
        return "root"
    if parts[1] in ("application", "domain", "infrastructure", "wiring"):
        return parts[1]
    # the remaining top-level names are the interface layer (cli / __main__).
    # §1 lays no forbidden family on an interface source -- its pure-dispatch
    # discipline is §2's, landing with the B1 composition root, not this gate.
    return "interface"


def is_forbidden_direction(source_layer, target_layer):
    """True when the (source, target) layer pair breaks an ADR-0019 §1
    direction."""
    return (
        (source_layer == "application" and target_layer == "infrastructure")
        or (source_layer == "domain" and target_layer in DOMAIN_UPPER_LAYERS)
        or (source_layer == "infrastructure" and target_layer == "application")
    )


def layered_import_violations(package_root):
    """``"source -> target"`` edges in the package tree that break the
    ADR-0019 §1 directions.  The target is spelled as the imported base
    module exactly as written: a pure respelling of an unchanged edge reads
    as one stale ratchet entry plus one fresh violation and is resolved by
    the only-shrink rule, never silently."""
    violations = set()
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root.parent).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join(parts)
        containing = module_name if path.name == "__init__.py" else module_name.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for base, _statement in import_bases(tree, containing):
            if base.split(".")[0] != "ctmr":
                continue
            if is_forbidden_direction(layer_of(module_name), layer_of(base)):
                violations.add(f"{module_name} -> {base}")
    return violations


def test_layered_imports_stay_within_the_frozen_ratchet():
    fresh = layered_import_violations(PACKAGE_ROOT) - FROZEN_VIOLATION_RATCHET
    assert not fresh, f"new layer-direction violations outside the frozen ratchet: {sorted(fresh)}"


def test_ratchet_entries_are_current_violations():
    stale = FROZEN_VIOLATION_RATCHET - layered_import_violations(PACKAGE_ROOT)
    assert not stale, f"ratchet entries that no longer violate (shrink the ratchet): {sorted(stale)}"


def test_layer_gate_detects_seeded_violations_and_admits_the_legal_directions(tmp_path):
    pkg = tmp_path / "src" / "ctmr"
    application = pkg / "application"
    domain = pkg / "domain"
    infrastructure = pkg / "infrastructure"
    wiring = pkg / "wiring"
    for package in (pkg, application, domain, infrastructure, wiring):
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    (infrastructure / "checkpoint_adapter.py").write_text(
        "import ctmr.domain.vocabulary\n", encoding="utf-8"
    )  # infrastructure -> domain: the one admitted cross-layer direction
    (application / "use_case.py").write_text(
        "import ctmr.domain.vocabulary\nimport ctmr.infrastructure.checkpoints\n",
        encoding="utf-8",
    )  # application -> domain fine, application -> infrastructure forbidden
    (domain / "entity.py").write_text(
        "import ctmr.application.shell\nimport ctmr.infrastructure.checkpoints\nimport ctmr.cli\nimport ctmr.wiring\n",
        encoding="utf-8",
    )  # domain -> any upper layer, forbidden in all three spellings
    (infrastructure / "ledger.py").write_text("import ctmr.application.shell\n", encoding="utf-8")  # infrastructure -> application forbidden
    (wiring / "composition.py").write_text(
        "import ctmr.application.shell\nimport ctmr.infrastructure.checkpoints\n",
        encoding="utf-8",
    )  # the composition root reaches everything
    (pkg / "cli.py").write_text(
        "import ctmr.application.shell\nimport ctmr.infrastructure.checkpoints\n",
        encoding="utf-8",
    )  # an interface source carries no §1 forbidden family (its pure-dispatch
    # discipline is §2's, landing at the B1 composition root) -- seeded here
    # so the gate's silence on it is pinned, not accidental
    seeded = layered_import_violations(pkg)
    assert seeded == {
        "ctmr.application.use_case -> ctmr.infrastructure.checkpoints",
        "ctmr.domain.entity -> ctmr.application.shell",
        "ctmr.domain.entity -> ctmr.infrastructure.checkpoints",
        "ctmr.domain.entity -> ctmr.cli",
        "ctmr.domain.entity -> ctmr.wiring",
        "ctmr.infrastructure.ledger -> ctmr.application.shell",
    }
    # and every seed sits outside the frozen ratchet: growth of exactly this
    # shape is what turns the real-repository probe red
    assert seeded - FROZEN_VIOLATION_RATCHET == seeded
