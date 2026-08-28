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

"""End-state guard suite (issue #144 / ADR-0015 §9-§10, batch M7; issue #175 / ADR-0016 M4-M5).

Four terminal-state gates pin the post-migration repository shape:

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
   out of live surfaces (ADR-0016 M4-M5).

Each gate runs two ways: a positive probe over the real repository (must be
clean) and a negative probe over a synthetic tree (a seeded violation must be
detected -- a guard that cannot fail is not a guard). This module is itself
exempt from gates 1 and 4's scan: it must spell the forbidden needles to find
them.
"""

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
