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

"""Run-record vocabulary, violation type, provenance snapshot, and storage.

Migrated verbatim from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141): the
``brats-phase-run/1`` record vocabulary (phases, the P3 variant markers, the
one-way ``open -> frozen`` status, data-list sides, the upstream phase, and
the final-acceptance verdict schema), the contract-violation error type, the
best-effort git provenance snapshot, and the controlled record-root store.
Records live in controlled storage; a record root inside a git work tree is
a verification failure (DUA-constrained outputs must not enter the repo).
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ctmr.application.acceptance.contract.artifacts import ArtifactFingerprinter
from ctmr.infrastructure.dmsource import DmSourceViolationError

SCHEMA = "brats-phase-run/1"
PHASES = ("P1", "P2", "P3")
P3_VARIANTS = ("controlnet-candidate", "stage0-baseline")
STAGE0_BASELINE = "stage0-baseline"  # zero-training img2img P3 baseline (issue #60 / spec #51 decision 8)
CONTROLNET_CANDIDATE = "controlnet-candidate"  # trained image-conditioned P3 ControlNet candidate (issue #61)
STATUS_OPEN = "open"
LIST_SIDES = ("train", "dev", "replay")  # holdout is never a data-list side; replay is the
# P1-only external MR-RATE cohort (spec #51 decision 6): its entries carry the
# replay study identity and must NOT collide with the BraTS split manifest.
UPSTREAM_PHASE = "P1"  # P2 and P3 both hang off the same frozen P1-DM

FINAL_ACCEPTANCE_SCHEMA = "brats-final-acceptance/1"

# Issue #135: the ledger's violation type IS the contract-violation type, so
# ``except ContractViolationError`` keeps catching ledger gates unchanged.
ContractViolationError = DmSourceViolationError


class CodeVersion:
    """Best-effort provenance of the contract code itself (git is optional on sugon copies)."""

    def __init__(self, script_path):
        self._script_path = Path(script_path)

    def snapshot(self):
        record = {
            "git_commit": None,
            "git_dirty": None,
            "script_sha256": ArtifactFingerprinter().file_sha256(self._script_path),
        }
        probe = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self._script_path.parent, capture_output=True, text=True)
        if probe.returncode != 0:
            return record
        record["git_commit"] = probe.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=self._script_path.parent, capture_output=True, text=True)
        record["git_dirty"] = bool(status.stdout.strip())
        return record


class RunRecordStore:
    """Persistence and addressing for run records under a controlled record root."""

    def __init__(self, record_root):
        self._root = Path(record_root)

    @classmethod
    def for_run(cls, run_path):
        """Store rooted at the record root of a run.json at <root>/runs/<id>/run.json."""
        # parents: [0]=<id>, [1]=runs, [2]=<root>
        return cls(Path(run_path).parents[2])

    def now_utc(self):
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def record_path(self, run_id):
        return self._root / "runs" / run_id / "run.json"

    def runs_dir(self):
        return self._root / "runs"

    def root(self):
        return self._root

    def write(self, record):
        path = self.record_path(record["run_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return path

    def load_by_path(self, path):
        record_path = Path(path)
        if not record_path.is_file():
            raise ContractViolationError(f"run record not found: {record_path}")
        record = json.loads(record_path.read_text())
        if record.get("schema") != SCHEMA:
            raise ContractViolationError(f"{record_path}: schema {record.get('schema')!r} != {SCHEMA!r}")
        return record

    def all_record_paths(self):
        runs_dir = self.runs_dir()
        if not runs_dir.is_dir():
            raise ContractViolationError(f"record root has no runs/ directory: {self._root}")
        paths = sorted(runs_dir.glob("*/run.json"))
        if not paths:
            raise ContractViolationError(f"no run records under {runs_dir}")
        return paths
