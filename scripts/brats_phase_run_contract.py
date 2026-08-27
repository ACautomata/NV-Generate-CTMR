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

"""Phase run contract for the BraTS2023 rflow fine-tuning recipe (issue #53, spec #51).

One auditable P1/P2/P3 run equals one run record under a controlled record
root. The record correlates every artifact the spec's testing decisions name
(the single top-level seam): input split manifest, frozen training configs,
the candidate checkpoint, the generated-sample manifest, the dev-side
selection basis, and post-freeze L1/L2/L3 reports.

Record layout (``<record-root>/runs/<run_id>/run.json``)::

    {
      "schema": "brats-phase-run/1",
      "run_id": "p1-20260821T093000Z+9f2a1c",
      "phase": "P1",                      # P1 | P2 | P3
      "variant": null,                    # P3 only: controlnet-candidate | stage0-baseline
      "status": "open",                   # open -> frozen (one way)
      "created_utc": "...", "frozen_utc": null,
      "code_version": {"git_commit": ..., "git_dirty": ..., "script_sha256": ...},
      "manifest": {"path": ..., "sha256": ...},            # issue #52 phase manifest
      "configs": [{"role": "env", "path": ..., "sha256": ...}],        # frozen configs
      "data_lists": [{"side": "train", "path": ..., "sha256": ...}],  # guarded below
      "base_ckpt": {"path": ..., "sha256": ...},           # P1 only: rflow-mr-brain v1
      "upstream": {                                        # P2/P3 only
        "run_id": ..., "run_record": ..., "phase": "P1", "status_at_init": "frozen",
        "checkpoint": {"path": ..., "sha256": ..., "epoch": ...}},    # the frozen P1-DM
      "platform": {...},                                   # e.g. DCU smoke provenance JSON
      "selection": {"rule": ..., "checkpoint": {...}, "evidence": [...], "recorded_utc": ...},
      "samples": {"path": ..., "sha256": ...},
      "attachments": [{"kind": "l1_report", ...}]          # post-freeze only
    }

Contract rules enforced at every mutation:

- Holdout guard (spec decision 3): while a run is open, no data list and no
  selection evidence may reference a final-holdout case of the pinned
  manifest; selection evidence may only reference dev-side cases. L1/L2/L3
  reports attach only after the candidate is frozen.
- Phase chain (spec decision 2): a P2/P3 run must pin, at init, a frozen P1
  run and derive its DM checkpoint from that run's selection — P3 can never
  pin a P2 ControlNet through this contract. Retraining the DM means a new
  P1 run; existing records keep their pinned DM identity.
- Stage-0 baseline (issue #60): a P3 run may open as ``stage0-baseline`` —
  the zero-training img2img comparison floor run off the registered P1-DM.
  It selects exactly the upstream DM checkpoint (nothing is trained), and it
  can never attach formal L1/L2/L3 reports or conclude final acceptance: a
  baseline is only the floor P3 candidates are compared against, never a
  trained or accepted candidate itself.
- Final acceptance (issue #58): ``conclude`` judges a frozen run's three
  formal layer reports (l1/l2/l3, each exactly one, each revalidated) with a
  non-compensatory AND — every layer must read ``pass``; any L1/L3 fail or any
  L2 fail/undecided writes an immutable blocked verdict with traceable
  per-layer reasons that no other layer's score can offset. Missing or invalid
  layer attachments refuse the judgement (no verdict record) so the run stays
  conclusible once its evidence is completed. Formal L2 evidence must bind the
  run and cover all five challenges at their frozen quotas (provisional runs
  are smoke, never acceptance).
- DM source (issue #58): only a P1 run whose final acceptance passed is
  registered (``dm_source.json``) as the single DM source P2/P3 bypasses may
  hang off; a later passing P1 supersedes it, and every bypass pinned to the
  superseded DM then fails verification with an explicit mismatch.
- P1 records its full-param continuation base checkpoint; P2/P3 take none
  (ControlNet initializes from the frozen DM encoder, not a checkpoint file).
- Records live in controlled storage: a record root inside a git work tree
  is a verification failure (DUA-constrained outputs must not enter the repo).

Usage (each subcommand standalone, init first):
    python -m scripts.brats_phase_run_contract init --phase P1 --record-root DIR \
        --manifest phase_manifest.json --config env=configs/environment_maisi_diff_model_rflow-mr-brain.json \
        --data-list train=lists/p1_image_only.json --base-ckpt rflow_mr_brain_v1.pt
    python -m scripts.brats_phase_run_contract select --run runs/<id>/run.json \
        --checkpoint ckpt.pt --rule "dev FID trend + early stop p=10" --evidence dev_metrics.json
    python -m scripts.brats_phase_run_contract freeze --run runs/<id>/run.json --samples samples.json
    python -m scripts.brats_phase_run_contract attach --run runs/<id>/run.json --kind l1_report --path l1.json
    python -m scripts.brats_phase_run_contract conclude --run runs/<id>/run.json
    python -m scripts.brats_phase_run_contract verify --record-root DIR
    python -m scripts.brats_phase_run_contract selftest --workdir TMP
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ctmr.application.acceptance.binding import BINDING_KEYS, expected_binding
from ctmr.application.acceptance.registry import (
    ATTACH_KINDS,
    FORMAL_LAYER_KINDS,
    L1_SCHEMA,
    L2_SCHEMA,
    L3_SCHEMA,
    LAYER_KINDS,
)

SCHEMA = "brats-phase-run/1"
PHASES = ("P1", "P2", "P3")
P3_VARIANTS = ("controlnet-candidate", "stage0-baseline")
STAGE0_BASELINE = "stage0-baseline"  # zero-training img2img P3 baseline (issue #60 / spec #51 decision 8)
CONTROLNET_CANDIDATE = "controlnet-candidate"  # trained image-conditioned P3 ControlNet candidate (issue #61)
STATUS_OPEN = "open"
STATUS_FROZEN = "frozen"
LIST_SIDES = ("train", "dev", "replay")  # holdout is never a data-list side; replay is the
# P1-only external MR-RATE cohort (spec #51 decision 6): its entries carry the
# replay study identity and must NOT collide with the BraTS split manifest.
UPSTREAM_PHASE = "P1"  # P2 and P3 both hang off the same frozen P1-DM
L1_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
L1_PLANES = ("xy", "yz", "zx")
L1_VERDICTS = ("pass", "fail", "undecided")
L1_T1N_TO_T1C = ("t1n", "t1c")
L1_FEATURE_EXTRACTOR = "radimagenet_resnet50"
L1_MR_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad"

L3_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
L3_DIMENSIONS = (
    "overall_realism",
    "anatomical_plausibility",
    "tumor_authenticity",
    "artifact_slice_consistency",
)
L3_TURING_WINDOW = (0.40, 0.60)
L3_LIKERT_BOUND = 4.0
L3_VERDICTS = ("pass", "fail")

L2_CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")  # the frozen five; formal L2 evidence covers all
L2_VERDICTS = ("pass", "fail", "undecided")

FINAL_ACCEPTANCE_SCHEMA = "brats-final-acceptance/1"
DM_SOURCE_SCHEMA = "brats-dm-source/1"


class ContractViolationError(Exception):
    """Raised when a mutation or verification breaks the phase run contract."""


class ArtifactFingerprinter:
    """SHA-256 fingerprints linking record entries to bytes on disk."""

    def file_sha256(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def must_fingerprint(self, path, label):
        resolved = Path(path)
        if not resolved.is_file():
            raise ContractViolationError(f"{label} not found: {resolved}")
        return {"path": str(resolved.resolve()), "sha256": self.file_sha256(resolved)}

    def content_sha256(self, text):
        return hashlib.sha256(text.encode()).hexdigest()


class ManifestSides:
    """The pinned phase manifest viewed as a (challenge, case) -> side map."""

    def __init__(self, manifest):
        self._manifest = manifest
        self._side_of = {}
        for ch, info in manifest["challenges"].items():
            for side in ("train", "dev", "holdout"):
                for case in info["cases"][side]:
                    self._side_of[(ch, case)] = side

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def side_of(self, challenge, case):
        return self._side_of.get((challenge, case))

    def all_case_keys(self):
        return {case for (_challenge, case) in self._side_of}

    def holdout_keys(self):
        return {key for key, side in self._side_of.items() if side == "holdout"}


class HoldoutGuard:
    """Rejects final-holdout cases in run inputs and dev-selection evidence (spec decision 3)."""

    def __init__(self, sides):
        self._sides = sides

    def scan_case_pairs(self, payload):
        """Recursively collects (sub, case) pairs from any parsed-JSON structure."""
        pairs = []
        if isinstance(payload, dict):
            if "sub" in payload and "case" in payload:
                pairs.append((payload["sub"], payload["case"]))
            for value in payload.values():
                pairs += self.scan_case_pairs(value)
        elif isinstance(payload, list):
            for item in payload:
                pairs += self.scan_case_pairs(item)
        return pairs

    def guard_data_list(self, list_entry, phase=None):
        """A labelled list must exist, carry cases, and match its side label with no holdout.

        A ``replay`` list (P1 only, spec #51 decision 6) inverts the membership
        check: every entry must be an external replay-cohort study that is NOT
        in the pinned BraTS manifest — by pair identity or by bare case id, so
        a BraTS case cannot re-enter training under the replay label.

        A P2/P3 ControlNet run (spec #51 decision 8) uses a single fold-split
        list labelled ``train`` that carries both train (fold=1) and dev
        (fold=0) entries — dev is a legitimate light-acceptance run input, not
        holdout — so a ``train`` list in those phases may hold train or dev
        cases. P1 (full-param continuation) keeps the strict side match."""
        path = Path(list_entry["path"])
        if not path.is_file():
            raise ContractViolationError(f"data list not found: {path}")
        pairs = self.scan_case_pairs(json.loads(path.read_text()))
        if not pairs:
            raise ContractViolationError(f"data list carries no (sub, case) entries: {path}")
        label = list_entry["side"]
        manifest_cases = self._sides.all_case_keys() if label == "replay" else None
        allowed_sides = {label}
        if phase in {"P2", "P3"} and label == "train":
            allowed_sides = {"train", "dev"}
        for challenge, case in pairs:
            side = self._sides.side_of(challenge, case)
            if label == "replay":
                if side is not None or case in manifest_cases:
                    raise ContractViolationError(
                        f"{path}: replay list carries manifest case ({challenge}, {case}); "
                        "the replay cohort is external to the BraTS split and must not shadow a split case"
                    )
                continue
            if side is None:
                raise ContractViolationError(f"{path}: ({challenge}, {case}) is not in the pinned manifest")
            if side == "holdout":
                raise ContractViolationError(
                    f"{path}: final-holdout case ({challenge}, {case}) must not enter a run input (holdout runs only after candidate freeze)"
                )
            if side not in allowed_sides:
                raise ContractViolationError(f"{path}: ({challenge}, {case}) is {side}-side but the list is labelled {label}")

    def guard_evidence(self, path):
        """Selection evidence may cite dev-side cases only (dev light acceptance, spec decision 3).

        An evidence file must carry at least one (sub, case) reference: a vacuous
        file cannot substantiate a dev-side selection basis."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ContractViolationError(f"selection evidence not found: {file_path}")
        pairs = self.scan_case_pairs(json.loads(file_path.read_text()))
        if not pairs:
            raise ContractViolationError(f"{file_path}: selection evidence cites no (sub, case) at all")
        for challenge, case in pairs:
            side = self._sides.side_of(challenge, case)
            if side is None:
                raise ContractViolationError(f"{file_path}: ({challenge}, {case}) is not in the pinned manifest")
            if side != "dev":
                raise ContractViolationError(
                    f"{file_path}: selection evidence cites {side}-side case ({challenge}, {case}); "
                    "checkpoint selection may reference dev light acceptance only"
                )

    def guard_config(self, path):
        """A frozen config may reference train/dev cases but never final-holdout ones (tuning input)."""
        file_path = Path(path)
        pairs = self.scan_case_pairs(json.loads(file_path.read_text()))
        for challenge, case in pairs:
            if self._sides.side_of(challenge, case) == "holdout":
                raise ContractViolationError(
                    f"{file_path}: config references final-holdout case ({challenge}, {case}); holdout must not enter tuning inputs"
                )


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


class RunInitializer:
    """Opens a run record, fingerprinting inputs and enforcing the phase chain."""

    def __init__(self, store, fingerprinter, sides):
        self._store = store
        self._fingerprinter = fingerprinter
        self._sides = sides

    def derive_upstream(self, upstream_run_path):
        """The P2/P3 DM comes from a frozen P1 run's selection, never from a free-form path."""
        upstream = self._store.load_by_path(upstream_run_path)
        if upstream["phase"] != UPSTREAM_PHASE:
            raise ContractViolationError(
                f"upstream run {upstream['run_id']} is phase {upstream['phase']}; "
                f"{'P3' if upstream['phase'] == 'P2' else 'bypasses'} must pin the frozen {UPSTREAM_PHASE}-DM "
                "(P3 must not warm-start from a P2 ControlNet)"
            )
        if upstream["status"] != STATUS_FROZEN:
            raise ContractViolationError(f"upstream run {upstream['run_id']} is {upstream['status']}; P2/P3 may only hang off a frozen P1 candidate")
        selection = upstream.get("selection")
        if not selection:
            raise ContractViolationError(f"upstream run {upstream['run_id']} has no candidate checkpoint selection")
        checkpoint = dict(selection["checkpoint"])
        if "epoch" not in checkpoint:
            checkpoint["epoch"] = None
        on_disk = self._fingerprinter.must_fingerprint(checkpoint["path"], "upstream candidate checkpoint")
        if on_disk["sha256"] != checkpoint["sha256"]:
            raise ContractViolationError(f"upstream checkpoint changed on disk: {checkpoint['path']}")
        DmSourceLedger(self._store).check_upstream(upstream["run_id"], checkpoint["sha256"])
        return {
            "run_id": upstream["run_id"],
            "run_record": str(Path(upstream_run_path).resolve()),
            "phase": upstream["phase"],
            "status_at_init": upstream["status"],
            "checkpoint": checkpoint,
        }

    def new_run_id(self, phase, run_id):
        if run_id is not None:
            if not re.fullmatch(r"[a-z0-9][a-z0-9+~-]*", run_id):
                raise ContractViolationError(f"run id must match [a-z0-9][a-z0-9+~-]*: {run_id!r}")
            return run_id
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{phase.lower()}-{stamp}"

    def resolve_variant(self, phase, variant):
        """P3-only variant marker (issue #60): a stage-0 baseline is the zero-training
        img2img comparison floor; every other P3 run is a trained ControlNet candidate."""
        if phase != "P3":
            if variant is not None:
                raise ContractViolationError(f"--variant is P3-only (a {phase} run has no stage-0 baseline variant): {variant!r}")
            return None
        resolved = P3_VARIANTS[0] if variant is None else variant
        if resolved not in P3_VARIANTS:
            raise ContractViolationError(f"P3 variant must be one of {P3_VARIANTS}: {variant!r}")
        return resolved

    def init(self, phase, run_id, manifest_path, configs, data_lists, base_ckpt, upstream_run, platform_json, variant=None):
        if phase not in PHASES:
            raise ContractViolationError(f"phase must be one of {PHASES}: {phase!r}")
        resolved_variant = self.resolve_variant(phase, variant)
        manifest_entry = self._fingerprinter.must_fingerprint(manifest_path, "phase manifest")
        config_entries = [{**self._fingerprinter.must_fingerprint(path, f"config {role}"), "role": role} for role, path in configs]
        if not config_entries:
            raise ContractViolationError("at least one --config ROLE=PATH is required")
        if resolved_variant is not None:
            inference_entries = [entry for entry in config_entries if entry["role"] == "inference"]
            if len(inference_entries) != 1:
                raise ContractViolationError(
                    "a P3 run must pin exactly one --config inference=<official P3 inference config> "
                    "(the recorded inference provenance: stage-0 baseline issue #60, ControlNet candidate issue #61)"
                )
        list_entries = []
        for side, path in data_lists:
            if side not in LIST_SIDES:
                raise ContractViolationError(f"data list side must be one of {LIST_SIDES}: {side!r}")
            if side == "replay" and phase != "P1":
                raise ContractViolationError(
                    "replay data lists are P1-only (spec #51 decision 6: MR-RATE replay mixes into "
                    "the full-parameter DM continuation; ControlNet-only P2/P3 take no replay)"
                )
            list_entries.append({**self._fingerprinter.must_fingerprint(path, f"{side} data list"), "side": side})
        if not list_entries:
            raise ContractViolationError("at least one --data-list SIDE=PATH is required")

        resolved_id = self.new_run_id(phase, run_id)
        if self._store.record_path(resolved_id).exists():
            raise ContractViolationError(f"run id already exists: {resolved_id}")

        base_entry = None
        upstream_entry = None
        if phase == "P1":
            if upstream_run is not None:
                raise ContractViolationError("P1 continues the rflow-mr-brain v1 base; it takes no --upstream-run")
            if base_ckpt is None:
                raise ContractViolationError("P1 requires --base-ckpt (the frozen rflow-mr-brain v1 checkpoint)")
            base_entry = self._fingerprinter.must_fingerprint(base_ckpt, "P1 base checkpoint")
        else:
            if base_ckpt is not None:
                raise ContractViolationError("P2/P3 ControlNets initialize from the frozen P1-DM encoder/mid, not a --base-ckpt file")
            if upstream_run is None:
                raise ContractViolationError(f"{phase} requires --upstream-run pointing at a frozen P1 run record")
            upstream_entry = self.derive_upstream(upstream_run)

        platform = None
        if platform_json is not None:
            try:
                platform = json.loads(Path(platform_json).read_text())
            except ValueError as error:
                raise ContractViolationError(f"platform json is not valid JSON: {platform_json} ({error})")

        # The guard runs on the pinned manifest bytes, after every path is resolved.
        pinned_sides = ManifestSides.from_path(manifest_entry["path"])
        guard = HoldoutGuard(pinned_sides)
        for entry in list_entries:
            guard.guard_data_list(entry, phase)
        for entry in config_entries:
            guard.guard_config(entry["path"])

        record = {
            "schema": SCHEMA,
            "run_id": resolved_id,
            "phase": phase,
            "variant": resolved_variant,
            "status": STATUS_OPEN,
            "created_utc": self._store.now_utc(),
            "frozen_utc": None,
            "code_version": CodeVersion(Path(__file__)).snapshot(),
            "manifest": manifest_entry,
            "configs": config_entries,
            "data_lists": list_entries,
            "base_ckpt": base_entry,
            "upstream": upstream_entry,
            "platform": platform,
            "selection": None,
            "samples": None,
            "attachments": [],
        }
        return self._store.write(record)


class SelectionRecorder:
    """Records the dev-side checkpoint selection basis (spec: no training-loss best, no holdout)."""

    def __init__(self, store, fingerprinter):
        self._store = store
        self._fingerprinter = fingerprinter

    def select(self, run_path, checkpoint, rule, evidence_paths, epoch):
        record = self._store.load_by_path(run_path)
        if record["status"] != STATUS_OPEN:
            raise ContractViolationError(f"run {record['run_id']} is {record['status']}; the candidate is locked")
        if not rule.strip():
            raise ContractViolationError("--rule (the pre-registered selection/early-stop rule) must not be empty")
        if not evidence_paths:
            raise ContractViolationError("at least one --evidence file is required (dev light acceptance)")
        guard = HoldoutGuard(ManifestSides.from_path(record["manifest"]["path"]))
        evidence_entries = []
        for path in evidence_paths:
            guard.guard_evidence(path)
            evidence_entries.append(self._fingerprinter.must_fingerprint(path, "selection evidence"))
        checkpoint_entry = self._fingerprinter.must_fingerprint(checkpoint, "candidate checkpoint")
        if record.get("variant") == STAGE0_BASELINE:
            upstream = record.get("upstream") or {}
            if checkpoint_entry["sha256"] != upstream.get("checkpoint", {}).get("sha256"):
                raise ContractViolationError(
                    "a stage-0 baseline trains nothing: its selection must pin the upstream P1-DM checkpoint "
                    "(the zero-training baseline never selects a different DM)"
                )
        if epoch is not None:
            checkpoint_entry["epoch"] = epoch
        else:
            checkpoint_entry["epoch"] = None
        record["selection"] = {
            "rule": rule,
            "checkpoint": checkpoint_entry,
            "evidence": evidence_entries,
            "recorded_utc": self._store.now_utc(),
        }
        return self._store.write(record)


class CandidateFreezer:
    """Freezes a candidate after selection + samples are recorded (one-way transition)."""

    def __init__(self, store, fingerprinter):
        self._store = store
        self._fingerprinter = fingerprinter

    def freeze(self, run_path, samples_path):
        record = self._store.load_by_path(run_path)
        if record["status"] != STATUS_OPEN:
            raise ContractViolationError(f"run {record['run_id']} is already {record['status']}")
        if not record.get("selection"):
            raise ContractViolationError(f"run {record['run_id']} has no dev-side selection; freeze requires the selection basis first")
        record["samples"] = self._fingerprinter.must_fingerprint(samples_path, "sample manifest")
        record["status"] = STATUS_FROZEN
        record["frozen_utc"] = self._store.now_utc()
        return self._store.write(record)


class L1ReportValidator:
    """Validates the versioned L1 evidence schema and its frozen-candidate binding."""

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L1 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L1 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L1 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        challenges = self._challenges(record, failures)
        self._protocol(report, failures)
        self._fid_results(record["phase"], report, challenges, failures)
        self._p3_results(record["phase"], report, challenges, failures)
        self._summary(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        expected = expected_binding(record)
        if report.get("schema") != L1_SCHEMA:
            failures.append(f"L1 report schema != {L1_SCHEMA}")
        if not isinstance(binding, dict):
            failures.append("L1 report binding must be an object")
            return
        for key in BINDING_KEYS:
            if binding.get(key) != expected[key]:
                failures.append(f"L1 report binding {key} does not match frozen run")

    def _challenges(self, record, failures):
        try:
            manifest = json.loads(Path(record["manifest"]["path"]).read_text())
            return tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot read pinned manifest for L1 coverage: {error}")
            return ()

    def _protocol(self, report, failures):
        protocol = report.get("protocol")
        if not isinstance(protocol, dict):
            failures.append("L1 report protocol must be an object")
            return
        extractor = protocol.get("feature_extractor")
        if not isinstance(extractor, dict) or extractor.get("name") != L1_FEATURE_EXTRACTOR or not self._sha256(extractor.get("weights_sha256")):
            failures.append(f"L1 report must record {L1_FEATURE_EXTRACTOR} and a SHA-256 weights hash")
        if protocol.get("mr_preprocessing") != L1_MR_PREPROCESSING:
            failures.append(f"L1 report mr_preprocessing must be {L1_MR_PREPROCESSING}")
        if tuple(protocol.get("planes", ())) != L1_PLANES:
            failures.append(f"L1 report planes must be {L1_PLANES}")
        bootstrap = protocol.get("bootstrap")
        if not isinstance(bootstrap, dict) or bootstrap.get("method") != "case_level_percentile_pcg64":
            failures.append("L1 report bootstrap must record case_level_percentile_pcg64")
        elif bootstrap.get("confidence_level") != 0.95 or not isinstance(bootstrap.get("resamples"), int):
            failures.append("L1 report bootstrap must record 95% CI and integer resamples")
        if protocol.get("fid_multiplier") != 2.5:
            failures.append("L1 report FID multiplier must be 2.5")

    def _fid_results(self, phase, report, challenges, failures):
        results = report.get("fid_results")
        if not isinstance(results, list):
            failures.append("L1 report fid_results must be a list")
            return
        expected = {(challenge, modality) for challenge in challenges for modality in L1_MODALITIES}
        actual = {(item.get("challenge"), item.get("target_modality")) for item in results if isinstance(item, dict)}
        if len(results) != len(expected) or actual != expected:
            failures.append("L1 report must cover each pinned challenge and all four target modalities exactly once")
        for result in results:
            if isinstance(result, dict):
                self._fid_result(phase, result, failures)

    def _fid_result(self, phase, result, failures):
        sources = result.get("generated_source_modalities")
        if phase == "P3":
            expected_sources = {source for source in L1_MODALITIES if source != result.get("target_modality")}
            if not isinstance(sources, list) or set(sources) != expected_sources:
                failures.append("P3 target-modality FID must record all src!=tgt generated_source_modalities")
        elif sources != []:
            failures.append(f"{phase} target-modality FID must not record P3 source modalities")
        verdict = result.get("verdict")
        if verdict not in L1_VERDICTS:
            failures.append("L1 FID verdict must be pass, fail, or undecided")
            return
        if verdict == "undecided":
            failures.append("formal L1 report cannot attach an undecided FID result")
            return
        generated = result.get("generated_vs_holdout")
        baseline = result.get("train_vs_holdout_baseline")
        threshold = result.get("threshold")
        self._fid_bundle(generated, "generated-vs-holdout", failures)
        self._fid_bundle(baseline, "train-vs-holdout baseline", failures)
        if not self._number(threshold):
            failures.append("L1 FID threshold must be finite")
            return
        try:
            bootstrap_median = baseline["mean_bootstrap_median"]
            expected_threshold = 2.5 * bootstrap_median
            upper = generated["mean"]["ci95"][1]
        except (KeyError, TypeError, IndexError):
            failures.append("L1 FID baseline must record a finite mean_bootstrap_median")
            return
        if not self._number(bootstrap_median):
            failures.append("L1 FID baseline mean_bootstrap_median must be finite")
            return
        if not math.isclose(threshold, expected_threshold, rel_tol=1e-9, abs_tol=1e-12):
            failures.append("L1 FID threshold must equal 2.5 times real train-vs-holdout bootstrap median")
        expected_verdict = "pass" if upper <= threshold else "fail"
        if verdict != expected_verdict:
            failures.append("L1 FID verdict disagrees with its CI upper-bound gate")

    def _fid_bundle(self, bundle, label, failures):
        if not isinstance(bundle, dict):
            failures.append(f"L1 {label} FID bundle must be an object")
            return
        planes = bundle.get("planes")
        if not isinstance(planes, dict) or tuple(sorted(planes)) != L1_PLANES:
            failures.append(f"L1 {label} FID bundle must contain xy/yz/zx")
        elif all(plane in planes for plane in L1_PLANES):
            for plane in L1_PLANES:
                self._interval(planes[plane], f"L1 {label} {plane} FID", failures)
        self._interval(bundle.get("mean"), f"L1 {label} mean FID", failures)

    def _p3_results(self, phase, report, challenges, failures):
        results = report.get("p3_paired_results")
        if phase != "P3":
            if results != []:
                failures.append(f"{phase} L1 report must not carry P3 paired results")
            return
        if not isinstance(results, list):
            failures.append("P3 L1 report p3_paired_results must be a list")
            return
        expected = {
            (challenge, source, target) for challenge in challenges for source in L1_MODALITIES for target in L1_MODALITIES if source != target
        }
        actual = {(item.get("challenge"), item.get("src_modality"), item.get("target_modality")) for item in results if isinstance(item, dict)}
        if len(results) != len(expected) or actual != expected:
            failures.append("P3 L1 report must cover all 12 ordered directions for every pinned challenge")
        for result in results:
            if isinstance(result, dict):
                self._p3_result(result, failures)

    def _p3_result(self, result, failures):
        source, target = result.get("src_modality"), result.get("target_modality")
        applicable = (source, target) != L1_T1N_TO_T1C
        if result.get("gate_applicable") != applicable:
            failures.append("P3 L1 gate_applicable must preserve the t1n->t1c exception")
            return
        verdict = result.get("verdict")
        mae = result.get("mae_relative_reduction")
        ssim = result.get("ssim_increase")
        if not applicable:
            if verdict != "not_applicable_known_unobservable":
                failures.append("P3 t1n->t1c must be explicitly not_applicable_known_unobservable")
            self._interval(mae, "P3 t1n->t1c MAE diagnostic", failures)
            self._interval(ssim, "P3 t1n->t1c SSIM diagnostic", failures)
            return
        if verdict not in L1_VERDICTS:
            failures.append("P3 paired verdict must be pass, fail, or undecided")
            return
        if verdict == "undecided":
            failures.append("formal L1 report cannot attach an undecided P3 paired result")
            return
        self._interval(mae, "P3 MAE relative reduction", failures)
        self._interval(ssim, "P3 SSIM increase", failures)
        try:
            expected = "pass" if mae["point"] >= 0.10 and ssim["point"] >= 0.02 and mae["ci95"][0] > 0.0 and ssim["ci95"][0] > 0.0 else "fail"
        except (KeyError, TypeError, IndexError):
            return
        if verdict != expected:
            failures.append("P3 paired verdict disagrees with the pre-registered MAE/SSIM gate")

    def _summary(self, report, failures):
        summary = report.get("summary")
        if not isinstance(summary, dict) or summary.get("verdict") not in L1_VERDICTS:
            failures.append("L1 report summary verdict must be pass, fail, or undecided")
            return
        verdicts = [result.get("verdict") for result in report.get("fid_results", []) if isinstance(result, dict)]
        verdicts += [
            result.get("verdict")
            for result in report.get("p3_paired_results", [])
            if isinstance(result, dict) and result.get("gate_applicable") is True
        ]
        expected = "undecided" if "undecided" in verdicts else "fail" if "fail" in verdicts else "pass"
        if summary["verdict"] != expected:
            failures.append("L1 report summary verdict disagrees with its applicable FID/P3 results")

    def _interval(self, interval, label, failures):
        if not isinstance(interval, dict) or not self._number(interval.get("point")):
            failures.append(f"{label} must record a finite point estimate")
            return
        ci = interval.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2 or not all(self._number(value) for value in ci) or ci[0] > ci[1]:
            failures.append(f"{label} must record an ordered finite 95% CI")

    def _number(self, value):
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)

    def _sha256(self, value):
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class L3ReportValidator:
    """Validates the versioned L3 evidence schema and its frozen-candidate binding."""

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L3 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L3 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L3 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        challenges = self._challenges(record, failures)
        protocol = self._protocol(report, failures)
        self._coverage(challenges, report, protocol, failures)
        self._visual_turing(report, protocol, failures)
        self._likert(report, protocol, failures)
        self._verdict(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        expected = expected_binding(record)
        if report.get("schema") != L3_SCHEMA:
            failures.append(f"L3 report schema != {L3_SCHEMA}")
        if not isinstance(binding, dict):
            failures.append("L3 report binding must be an object")
            return
        for key in BINDING_KEYS:
            if binding.get(key) != expected[key]:
                failures.append(f"L3 report binding {key} does not match frozen run")

    def _challenges(self, record, failures):
        try:
            manifest = json.loads(Path(record["manifest"]["path"]).read_text())
            return tuple(sorted(manifest["challenges"]))
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
            failures.append(f"cannot read pinned manifest for L3 coverage: {error}")
            return ()

    def _protocol(self, report, failures):
        protocol = report.get("protocol")
        if not isinstance(protocol, dict):
            failures.append("L3 report protocol must be an object")
            return None
        reviewers = protocol.get("reviewers")
        if not isinstance(reviewers, int) or reviewers < 2:
            failures.append("L3 report must record at least two independent reviewers")
        dimensions = protocol.get("dimensions")
        if not isinstance(dimensions, list) or tuple(dimensions) != L3_DIMENSIONS:
            failures.append(f"L3 report dimensions must be {L3_DIMENSIONS}")
        modalities = protocol.get("target_modalities")
        if not isinstance(modalities, list) or tuple(modalities) != L3_MODALITIES:
            failures.append(f"L3 report target modalities must be {L3_MODALITIES}")
        window = protocol.get("visual_turing_ci_window")
        if not isinstance(window, list) or len(window) != 2 or window != list(L3_TURING_WINDOW):
            failures.append(f"L3 report visual-Turing CI window must be {list(L3_TURING_WINDOW)}")
        if protocol.get("likert_minimum") != L3_LIKERT_BOUND:
            failures.append(f"L3 report Likert bound must be {L3_LIKERT_BOUND}")
        if protocol.get("confidence_level") != 0.95:
            failures.append("L3 report must record 95% confidence")
        bootstrap = protocol.get("bootstrap")
        if not isinstance(bootstrap, dict) or not bootstrap.get("method") or not isinstance(bootstrap.get("resamples"), int):
            failures.append("L3 report bootstrap must record a method and integer resamples")
        elif not isinstance(bootstrap.get("seed"), int):
            failures.append("L3 report bootstrap must record an integer seed")
        if not isinstance(protocol.get("per_cell"), int) or protocol.get("per_cell", 0) < 1:
            failures.append("L3 report per_cell must be a positive integer")
        if not isinstance(protocol.get("total_entries"), int):
            failures.append("L3 report total_entries must be an integer")
        return protocol

    def _coverage(self, challenges, report, protocol, failures):
        coverage = report.get("coverage")
        if not isinstance(coverage, list):
            failures.append("L3 report coverage must be a list")
            return
        per_cell = (protocol or {}).get("per_cell")
        expected = {(challenge, modality) for challenge in challenges for modality in L3_MODALITIES}
        actual = {(row.get("challenge"), row.get("target_modality")) for row in coverage if isinstance(row, dict)}
        if not expected or len(coverage) != len(expected) or actual != expected:
            failures.append("L3 report must cover each pinned challenge and all four target modalities exactly once")
            return
        if per_cell is None:
            return
        total = 0
        for row in coverage:
            if isinstance(row, dict):
                real_count, synth_count = row.get("real"), row.get("synth")
                if real_count != per_cell or synth_count != per_cell:
                    failures.append(f"L3 coverage {row.get('challenge')}/{row.get('target_modality')} must be {per_cell} real + {per_cell} synth")
                if isinstance(real_count, int) and isinstance(synth_count, int):
                    total += real_count + synth_count
        if protocol.get("total_entries") != total:
            failures.append("L3 report total_entries must equal the sum of the per-cell coverage")

    def _visual_turing(self, report, protocol, failures):
        vt = report.get("visual_turing")
        if not isinstance(vt, dict):
            failures.append("L3 report visual_turing must be an object")
            return
        expected_reviewers = (protocol or {}).get("reviewers")
        per_reviewer = vt.get("per_reviewer")
        if not isinstance(per_reviewer, list) or len(per_reviewer) != expected_reviewers:
            failures.append("L3 report visual_turing per_reviewer must match the recorded reviewer count")
            return
        for result in per_reviewer:
            if isinstance(result, dict):
                self._vt_result(result, failures)
        pooled = vt.get("pooled")
        if isinstance(pooled, dict):
            self._vt_result(pooled, failures, pooled=True)
        else:
            failures.append("L3 report must record the pooled visual-Turing result")
        if isinstance(pooled, dict):
            recorded = vt.get("verdict")
            expected = "pass" if all(isinstance(item, dict) and item.get("verdict") == "pass" for item in per_reviewer + [pooled]) else "fail"
            if recorded != expected:
                failures.append("L3 report visual_turing verdict disagrees with its per-reviewer/pooled CI window gates")

    def _vt_result(self, result, failures, pooled=False):
        verdict = result.get("verdict")
        if verdict not in L3_VERDICTS:
            failures.append("L3 visual-Turing verdict must be pass or fail")
            return
        if not self._number(result.get("balanced_accuracy")):
            failures.append("L3 visual-Turing balanced accuracy must be finite")
            return
        ci = result.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2 or not all(self._number(value) for value in ci) or ci[0] > ci[1]:
            failures.append("L3 visual-Turing CI must be an ordered finite 95% CI")
            return
        if not (ci[0] <= result["balanced_accuracy"] <= ci[1]):
            failures.append("L3 visual-Turing CI must contain its balanced-accuracy point estimate")
        expected = "pass" if ci[0] >= L3_TURING_WINDOW[0] and ci[1] <= L3_TURING_WINDOW[1] else "fail"
        if verdict != expected:
            failures.append("L3 visual-Turing verdict disagrees with its CI window gate")
        if pooled:
            return
        confusion = result.get("confusion")
        if not isinstance(confusion, dict):
            failures.append("L3 per-reviewer visual-Turing must record a confusion matrix")
            return
        try:
            real_total = confusion.get("real_said_real", 0) + confusion.get("real_said_synth", 0)
            synth_total = confusion.get("synth_said_real", 0) + confusion.get("synth_said_synth", 0)
            if real_total <= 0 or synth_total <= 0:
                failures.append("L3 per-reviewer visual-Turing confusion must have both real and synth entries")
                return
            if result.get("n") != real_total + synth_total:
                failures.append("L3 per-reviewer visual-Turing n must equal the confusion total")
            rederived = 0.5 * (confusion["real_said_real"] / real_total + confusion["synth_said_synth"] / synth_total)
        except (KeyError, TypeError):
            failures.append("L3 per-reviewer visual-Turing confusion must carry integer counts")
            return
        if not math.isclose(rederived, result["balanced_accuracy"], rel_tol=1e-9, abs_tol=1e-12):
            failures.append("L3 per-reviewer visual-Turing balanced accuracy disagrees with its confusion matrix")

    def _likert(self, report, protocol, failures):
        likert = report.get("likert")
        if not isinstance(likert, list):
            failures.append("L3 report likert must be a list")
            return
        dimensions = {item.get("dimension") for item in likert if isinstance(item, dict)}
        if len(likert) != len(L3_DIMENSIONS) or dimensions != set(L3_DIMENSIONS):
            failures.append(f"L3 report likert must cover each of {L3_DIMENSIONS} exactly once")
            return
        for item in likert:
            if isinstance(item, dict):
                self._likert_item(item, failures)

    def _likert_item(self, item, failures):
        dimension = item.get("dimension")
        phase = item.get("phase")
        self._likert_bundle(phase, f"L3 Likert {dimension} phase", failures)
        per_modality = item.get("per_modality")
        if not isinstance(per_modality, dict) or set(per_modality) != set(L3_MODALITIES):
            failures.append(f"L3 Likert {dimension} per_modality must cover all four target modalities")
            return
        for modality in L3_MODALITIES:
            self._likert_bundle(per_modality.get(modality), f"L3 Likert {dimension} {modality}", failures)
        if not (item.get("fleiss_kappa") is None or self._number(item["fleiss_kappa"])):
            failures.append(f"L3 Likert {dimension} Fleiss' kappa must be finite or null")

    def _likert_bundle(self, bundle, label, failures):
        if not isinstance(bundle, dict):
            failures.append(f"{label} must be an object")
            return
        point = bundle.get("point")
        lower = bundle.get("ci95_lower")
        if not self._number(point) or not self._number(lower) or lower > point:
            failures.append(f"{label} must record a finite point and a one-sided lower CI not above the mean")
            return
        if not isinstance(bundle.get("n"), int) or bundle["n"] < 1:
            failures.append(f"{label} must record a positive integer n")
            return
        if not isinstance(bundle.get("na"), int) or bundle["na"] < 0:
            failures.append(f"{label} must record a non-negative integer NA count")
            return
        verdict = bundle.get("verdict")
        if verdict not in L3_VERDICTS:
            failures.append(f"{label} verdict must be pass or fail")
            return
        expected = "pass" if lower >= L3_LIKERT_BOUND else "fail"
        if verdict != expected:
            failures.append(f"{label} verdict disagrees with its {L3_LIKERT_BOUND} lower-bound gate")

    def _verdict(self, report, failures):
        verdict = report.get("verdict")
        if not isinstance(verdict, dict):
            failures.append("L3 report verdict must be an object")
            return
        for key in ("visual_turing", "likert", "overall"):
            if verdict.get(key) not in L3_VERDICTS:
                failures.append(f"L3 report verdict {key} must be pass or fail")
        likert = report.get("likert") or []
        likert_expected = (
            "pass"
            if all(
                isinstance(item, dict)
                and (item.get("phase") or {}).get("verdict") == "pass"
                and all(isinstance(modality, dict) and modality.get("verdict") == "pass" for modality in (item.get("per_modality") or {}).values())
                for item in likert
            )
            else "fail"
        )
        overall_expected = "pass" if (report.get("visual_turing") or {}).get("verdict") == "pass" and likert_expected == "pass" else "fail"
        if verdict.get("likert") != likert_expected:
            failures.append("L3 report verdict likert disagrees with its dimension lower-bound gates")
        if verdict.get("overall") != overall_expected:
            failures.append("L3 report verdict overall must be the non-compensatory AND of visual-Turing and Likert")

    def _number(self, value):
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


class L2ReportValidator:
    """Validates the versioned L2 evidence schema and its frozen-candidate binding.

    Formal L2 evidence (``l2-final-acceptance-report/1``, issue #55) attaches
    only as a complete five-challenge report: ``challenges_missing`` empty, no
    provisional challenge, ``complete_coverage`` true (spec Further Notes --
    a run over a subset of the five challenges is provisional smoke, never
    full-spec acceptance evidence). Verdict consistency mirrors the issue #55
    judgement chain: any failure-audit count > 0 forces that challenge
    ``undecided``; otherwise all TOST (and, for P2, round-trip) checks passing
    forces ``pass``; the overall verdict is undecided > fail > pass.
    """

    def validate(self, record, path):
        report_path = Path(path)
        if not report_path.is_file():
            return [f"L2 report not found: {report_path}"]
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as error:
            return [f"L2 report is not valid JSON: {report_path} ({error})"]
        if not isinstance(report, dict):
            return ["L2 report root must be a JSON object"]
        failures = []
        self._binding(record, report, failures)
        self._coverage(report, failures)
        self._per_challenge(record, report, failures)
        self._overall(report, failures)
        return failures

    def _binding(self, record, report, failures):
        binding = report.get("binding")
        expected = expected_binding(record)
        if report.get("schema") != L2_SCHEMA:
            failures.append(f"L2 report schema != {L2_SCHEMA}")
        if not isinstance(binding, dict):
            failures.append("L2 report binding must be an object (evaluate with --run to bind the frozen candidate)")
            return
        for key in BINDING_KEYS:
            if binding.get(key) != expected[key]:
                failures.append(f"L2 report binding {key} does not match frozen run")

    def _coverage(self, report, failures):
        if report.get("challenges_missing") != []:
            failures.append("formal L2 evidence must cover all five challenges (challenges_missing must be empty)")
        if report.get("provisional_challenges") != []:
            failures.append("formal L2 evidence must meet every frozen holdout quota (no provisional challenge)")
        if report.get("complete_coverage") is not True:
            failures.append("formal L2 evidence must record complete_coverage true")

    def _per_challenge(self, record, report, failures):
        per_challenge = report.get("per_challenge")
        if not isinstance(per_challenge, dict):
            failures.append("L2 report per_challenge must be an object")
            return
        if set(per_challenge) != set(L2_CHALLENGES):
            failures.append(f"L2 report per_challenge must cover exactly {L2_CHALLENGES}")
            return
        for challenge, verdict in per_challenge.items():
            if not isinstance(verdict, dict):
                failures.append(f"L2 per_challenge {challenge} must be an object")
            else:
                self._challenge_verdict(challenge, verdict, record.get("phase"), failures)

    def _challenge_verdict(self, challenge, verdict, phase, failures):
        recorded = verdict.get("verdict")
        if recorded not in L2_VERDICTS:
            failures.append(f"L2 {challenge} verdict must be pass, fail, or undecided")
            return
        audit = verdict.get("failure_audit")
        n_failed = audit.get("n_failed") if isinstance(audit, dict) else None
        if not isinstance(n_failed, int) or n_failed < 0:
            failures.append(f"L2 {challenge} failure_audit must record a non-negative integer n_failed")
            return
        checks = [item.get("passed") for item in verdict.get("tost") or []]
        if verdict.get("round_trip") is not None:
            if phase != "P2":
                failures.append(f"L2 {challenge} round_trip evidence is P2-only; {phase} must not carry it")
                return
            checks += [item.get("passed") for item in verdict["round_trip"] or []]
        elif phase == "P2":
            failures.append(f"L2 {challenge} P2 evidence must carry the condition round-trip results")
            return
        if not checks:
            failures.append(f"L2 {challenge} carries no TOST checks")
            return
        if n_failed:
            expected = "undecided"
        elif all(checks):
            expected = "pass"
        else:
            expected = "fail"
        if recorded != expected:
            failures.append(f"L2 {challenge} verdict {recorded!r} disagrees with its failure gate/TOST/round-trip evidence")

    def _overall(self, report, failures):
        overall = report.get("overall_verdict")
        if overall not in L2_VERDICTS:
            failures.append("L2 report overall_verdict must be pass, fail, or undecided")
            return
        per_challenge = report.get("per_challenge")
        verdicts = (
            [verdict.get("verdict") for verdict in per_challenge.values() if isinstance(verdict, dict)] if isinstance(per_challenge, dict) else []
        )
        if len(verdicts) != len(L2_CHALLENGES) or any(v not in L2_VERDICTS for v in verdicts):
            return  # already reported by _per_challenge
        expected = "undecided" if "undecided" in verdicts else "pass" if all(v == "pass" for v in verdicts) else "fail"
        if overall != expected:
            failures.append("L2 report overall_verdict disagrees with its per-challenge verdicts")


class ReportAttacher:
    """Attaches post-freeze L1/L2/L3/env reports (the only mutation allowed after freezing)."""

    def __init__(self, store, fingerprinter):
        self._store = store
        self._fingerprinter = fingerprinter

    def _assert_controlled_report(self, path, kind):
        for parent in Path(path).resolve().parents:
            if (parent / ".git").exists():
                raise ContractViolationError(f"{kind} lives inside a git work tree ({parent}); controlled reports must stay outside the repo")

    def attach(self, run_path, kind, path):
        record = self._store.load_by_path(run_path)
        if record["status"] != STATUS_FROZEN:
            raise ContractViolationError(
                f"run {record['run_id']} is {record['status']}; L1/L2/L3 holdout evidence attaches only to frozen candidates"
            )
        if kind not in ATTACH_KINDS:
            raise ContractViolationError(f"attachment kind must be one of {ATTACH_KINDS}: {kind!r}")
        if kind in FORMAL_LAYER_KINDS and record.get("variant") == STAGE0_BASELINE:
            raise ContractViolationError(
                f"a stage-0 baseline (run {record['run_id']}) is the P3 comparison floor, not a trained candidate; "
                f"formal {kind} evidence would mislabel zero-training img2img output as an accepted candidate"
            )
        if kind == "l1_report":
            self._assert_controlled_report(path, "l1_report")
            if any(attachment["kind"] == "l1_report" for attachment in record["attachments"]):
                raise ContractViolationError(f"run {record['run_id']} already has a formal l1_report attachment")
            failures = L1ReportValidator().validate(record, path)
            if failures:
                raise ContractViolationError("invalid l1_report: " + "; ".join(failures))
        if kind == "l2_report":
            self._assert_controlled_report(path, "l2_report")
            if any(attachment["kind"] == "l2_report" for attachment in record["attachments"]):
                raise ContractViolationError(f"run {record['run_id']} already has a formal l2_report attachment")
            failures = L2ReportValidator().validate(record, path)
            if failures:
                raise ContractViolationError("invalid l2_report: " + "; ".join(failures))
        if kind == "l3_report":
            self._assert_controlled_report(path, "l3_report")
            if any(attachment["kind"] == "l3_report" for attachment in record["attachments"]):
                raise ContractViolationError(f"run {record['run_id']} already has a formal l3_report attachment")
            failures = L3ReportValidator().validate(record, path)
            if failures:
                raise ContractViolationError("invalid l3_report: " + "; ".join(failures))
        entry = {**self._fingerprinter.must_fingerprint(path, f"{kind} attachment"), "kind": kind}
        entry["attached_utc"] = self._store.now_utc()
        record["attachments"].append(entry)
        return self._store.write(record)


class DmSourceLedger:
    """The single registered P1-DM source that P2/P3 bypasses may hang off (issue #58).

    Registering is the freeze of a final-acceptance-passing P1 candidate's DM
    identity, configs and provenance. Replacement is explicit: a later P1
    candidate that passes final acceptance supersedes the previous source, and
    every bypass pinned to the superseded DM fails verification with a
    mismatch -- a retrained DM never silently keeps old bypasses comparable
    (spec #51 user story 22 / CONTEXT.md 产物链).
    """

    def __init__(self, store):
        self._store = store

    def path(self):
        return self._store.root() / "dm_source.json"

    def current(self):
        ledger_path = self.path()
        if not ledger_path.is_file():
            return None
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("schema") != DM_SOURCE_SCHEMA:
            raise ContractViolationError(f"dm_source ledger {ledger_path} has schema {ledger.get('schema')!r} != {DM_SOURCE_SCHEMA!r}")
        return ledger

    def register(self, record, run_record_path):
        """Freezes the passing P1 candidate as the current DM source (superseding any previous)."""
        if record["phase"] != "P1":
            raise ContractViolationError("only a P1 candidate can be registered as the DM source (P2/P3 are bypasses, not sources)")
        current = self.current()
        if current is not None and current["run_id"] == record["run_id"]:
            return current  # idempotent re-register of the same candidate
        entry = {
            "schema": DM_SOURCE_SCHEMA,
            "run_id": record["run_id"],
            "run_record": str(Path(run_record_path).resolve()),
            "run_record_sha256": ArtifactFingerprinter().file_sha256(run_record_path),
            "checkpoint": record["selection"]["checkpoint"],
            "configs": record["configs"],
            "manifest": record["manifest"],
            "base_ckpt": record.get("base_ckpt"),
            "code_version": record.get("code_version"),
            "registered_utc": self._store.now_utc(),
            "superseded_run_id": None,
        }
        if current is not None:
            entry["superseded_run_id"] = current["run_id"]
        self.path().write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return entry

    def check_upstream(self, upstream_run_id, checkpoint_sha256):
        """Init-time gate: a P2/P3 bypass may only pin the registered DM source."""
        current = self.current()
        if current is None:
            raise ContractViolationError(
                "no P1 candidate has passed final acceptance yet; P2/P3 must hang off the " "registered DM source (conclude a passing P1 run first)"
            )
        if upstream_run_id != current["run_id"] or checkpoint_sha256 != current["checkpoint"]["sha256"]:
            raise ContractViolationError(
                f"upstream run {upstream_run_id} is not the registered DM source {current['run_id']}; "
                "P2/P3 may only hang off the final-acceptance-passing P1-DM"
            )

    def check_record(self, record):
        """Verify-time mismatch detection against the current DM source."""
        current = self.current()
        if current is None:
            return []
        if record.get("phase") == "P1":
            if (
                record["run_id"] == current["run_id"]
                and record.get("selection", {}).get("checkpoint", {}).get("sha256") != current["checkpoint"]["sha256"]
            ):
                return ["registered DM source checkpoint no longer matches its P1 run record"]
            return []
        upstream = record.get("upstream")
        if upstream and (upstream["checkpoint"]["sha256"] != current["checkpoint"]["sha256"] or upstream["run_id"] != current["run_id"]):
            return [
                f"DM was retrained: this bypass is pinned to superseded DM {upstream['run_id']} "
                f"while the registered DM source is {current['run_id']}"
            ]
        return []


class FinalAcceptanceJudge:
    """Non-compensatory L1 ∧ L2 ∧ L3 final acceptance over a frozen candidate (issue #58).

    The verdict is an AND with no compensation: a pass requires every layer's
    own verdict to be ``pass``; any L1/L3 fail, any L2 fail or undecided, or
    any missing layer blocks the conclusion with a traceable per-layer reason
    list (spec #51 decision 15 / CONTEXT.md 数据划分角色). Only a P1 pass
    registers the DM source for P2/P3 (issue #58 acceptance criterion 3).
    """

    def __init__(self, store, fingerprinter):
        self._store = store
        self._fingerprinter = fingerprinter

    @staticmethod
    def verdict_path_for(record_path):
        # parents: [0]=<id>, [1]=runs|final_acceptance, [2]=<root>
        return Path(record_path).parents[2] / "final_acceptance" / f"{Path(record_path).parent.name}.json"

    def conclude(self, run_path):
        record = self._store.load_by_path(run_path)
        if record["status"] != STATUS_FROZEN:
            raise ContractViolationError(
                f"run {record['run_id']} is {record['status']}; final acceptance concludes only frozen candidates "
                "(holdout evidence attaches after the freeze)"
            )
        if record.get("variant") == STAGE0_BASELINE:
            raise ContractViolationError(
                f"run {record['run_id']} is the stage-0 zero-training img2img baseline; it is the P3 comparison "
                "floor and never takes final acceptance (issue #60: a baseline must not be mislabelled as a "
                "ControlNet-trained or final-acceptance-passing candidate)"
            )
        verdict_path = self.verdict_path_for(run_path)
        if verdict_path.is_file():
            raise ContractViolationError(f"final acceptance for {record['run_id']} is already concluded at {verdict_path}; the verdict is immutable")
        layers, problems = self._collect_layers(record)
        if problems:
            raise ContractViolationError("final acceptance blocked before judgement: " + "; ".join(problems))
        blocked_reasons = []
        for layer_name in LAYER_KINDS:
            blocked_reasons += self._layer_reasons(layer_name, layers[layer_name])
        verdict = "pass" if not blocked_reasons else "blocked"
        entry = {
            "schema": FINAL_ACCEPTANCE_SCHEMA,
            "run_id": record["run_id"],
            "phase": record["phase"],
            "decided_utc": self._store.now_utc(),
            "layers": {name: {"attachment": layer["attachment"], "verdict": layer["verdict"]} for name, layer in layers.items()},
            "verdict": verdict,
            "blocked_reasons": blocked_reasons,
            "dm_source_registered": False,
        }
        if verdict == "pass" and record["phase"] == "P1":
            entry["dm_source_registered"] = True
            DmSourceLedger(self._store).register(record, run_path)
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return entry, verdict_path

    def _collect_layers(self, record):
        """One formal attachment per layer, each revalidated against the frozen run."""
        validators = {
            "l1_report": L1ReportValidator(),
            "l2_report": L2ReportValidator(),
            "l3_report": L3ReportValidator(),
        }
        layers = {}
        problems = []
        for layer_name, kind in LAYER_KINDS.items():
            attachments = [a for a in record.get("attachments", []) if a.get("kind") == kind]
            if len(attachments) > 1:
                problems.append(f"run has more than one formal {kind} attachment")
                continue
            if not attachments:
                problems.append(f"final acceptance requires a formal {kind} attachment (candidate freeze is not enough)")
                continue
            failures = validators[kind].validate(record, attachments[0]["path"])
            if failures:
                problems.append(f"invalid {kind}: " + "; ".join(failures))
                continue
            report = json.loads(Path(attachments[0]["path"]).read_text())
            layers[layer_name] = {
                "attachment": {"path": attachments[0]["path"], "sha256": attachments[0]["sha256"]},
                "verdict": self._layer_verdict(kind, report),
                "report": report,
            }
        return layers, problems

    @staticmethod
    def _layer_verdict(kind, report):
        if kind == "l1_report":
            return report.get("summary", {}).get("verdict")
        if kind == "l2_report":
            return report.get("overall_verdict")
        return (report.get("verdict") or {}).get("overall")

    def _layer_reasons(self, layer_name, layer):
        """Traceable blockers: layer + failing criterion, never offset by other layers' scores."""
        if layer["verdict"] == "pass":
            return []
        report = layer["report"]
        reasons = []
        if layer_name == "L1":
            for result in report.get("fid_results", []):
                if result.get("verdict") != "pass":
                    reasons.append(f"L1 FID {result.get('challenge')}/{result.get('target_modality')}: {result.get('verdict')}")
            for result in report.get("p3_paired_results", []):
                if result.get("gate_applicable") and result.get("verdict") != "pass":
                    reasons.append(
                        f"L1 P3 paired {result.get('challenge')}/{result.get('src_modality')}->{result.get('target_modality')}: {result.get('verdict')}"
                    )
        elif layer_name == "L2":
            for challenge, verdict in (report.get("per_challenge") or {}).items():
                if verdict.get("verdict") != "pass":
                    reason = verdict.get("reason") or self._l2_challenge_reason(verdict)
                    reasons.append(f"L2 {challenge}: {verdict.get('verdict')} ({reason})")
        elif layer_name == "L3":
            if (report.get("visual_turing") or {}).get("verdict") != "pass":
                reasons.append("L3 visual-Turing: CI window gate not met")
            for item in report.get("likert") or []:
                failing = [m for m, b in (item.get("per_modality") or {}).items() if b.get("verdict") != "pass"]
                if (item.get("phase") or {}).get("verdict") != "pass" or failing:
                    detail = f"; per-modality fail: {', '.join(sorted(failing))}" if failing else ""
                    reasons.append(f"L3 Likert {item.get('dimension')}: lower-bound gate not met{detail}")
        if not reasons:
            reasons.append(f"{layer_name} verdict is {layer['verdict']}")
        return reasons

    @staticmethod
    def _l2_challenge_reason(verdict):
        if verdict.get("verdict") == "undecided":
            return "instrument failure gate; fix direction is the instrument or a re-run"
        tost_failed = sum(0 if verdict.get("tost") is None else (not item.get("passed")) for item in (verdict.get("tost") or []))
        rt_failed = sum(0 if verdict.get("round_trip") is None else (not item.get("passed")) for item in (verdict.get("round_trip") or []))
        return f"{tost_failed} TOST and {rt_failed} round-trip checks failed"

    def revalidate_verdict(self, record, verdict_path):
        """Verify-time reconciliation of an existing verdict record against the run."""
        verdict_record = json.loads(Path(verdict_path).read_text())
        problems = []
        if verdict_record.get("schema") != FINAL_ACCEPTANCE_SCHEMA:
            return [f"verdict record schema != {FINAL_ACCEPTANCE_SCHEMA}"]
        if verdict_record.get("run_id") != record.get("run_id") or verdict_record.get("phase") != record.get("phase"):
            problems.append("verdict record does not bind this run")
        layers = verdict_record.get("layers")
        if not isinstance(layers, dict) or set(layers) != set(LAYER_KINDS):
            problems.append("verdict record must carry exactly the L1/L2/L3 layer entries")
            return problems
        for layer_name, layer in layers.items():
            attachment = layer.get("attachment") or {}
            current = [a for a in record.get("attachments", []) if a.get("kind") == LAYER_KINDS[layer_name]]
            if len(current) != 1 or current[0]["sha256"] != attachment.get("sha256"):
                problems.append(f"verdict record {layer_name} attachment no longer matches the run record")
        # Re-derive the AND: the recorded verdict must follow from its layer verdicts
        # (an edited/flipped verdict file fails verification even with intact attachments).
        layer_verdicts = [layers[name].get("verdict") for name in LAYER_KINDS]
        if all(verdict == "pass" for verdict in layer_verdicts):
            expected = "pass" if verdict_record.get("blocked_reasons") == [] else "blocked"
        else:
            expected = "blocked"
        if verdict_record.get("verdict") != expected:
            problems.append("verdict record disagrees with the non-compensatory AND of its layer verdicts")
        if (verdict_record.get("verdict") == "blocked") != bool(verdict_record.get("blocked_reasons")):
            problems.append("verdict record blocked state and blocked_reasons are inconsistent")
        return problems


class RunVerifier:
    """Reconciles one run record against the contract: hashes, guard, phase chain, storage."""

    def __init__(self, fingerprinter):
        self._fingerprinter = fingerprinter
        self.failures = []

    def check(self, cond, msg):
        if not cond:
            self.failures.append(msg)

    def verify_hashes(self, record):
        entries = (
            [("manifest", record["manifest"])]
            + [(f"config {e['role']}", e) for e in record["configs"]]
            + [(f"{e['side']} data list", e) for e in record["data_lists"]]
        )
        if record.get("selection"):
            entries += [("selection evidence", e) for e in record["selection"]["evidence"]]
            entries.append(("candidate checkpoint", record["selection"]["checkpoint"]))
        if record.get("base_ckpt"):
            entries.append(("base checkpoint", record["base_ckpt"]))
        if record.get("upstream"):
            entries.append(("upstream checkpoint", record["upstream"]["checkpoint"]))
        if record.get("samples"):
            entries.append(("sample manifest", record["samples"]))
        for attachment in record.get("attachments", []):
            entries.append((f"{attachment['kind']} attachment", attachment))
        for label, entry in entries:
            path = Path(entry["path"])
            if not path.is_file():
                self.check(False, f"{label} missing on disk: {path}")
                continue
            self.check(
                self._fingerprinter.file_sha256(path) == entry["sha256"],
                f"{label} sha256 changed on disk: {path}",
            )

    def verify_l1_reports(self, record):
        attachments = [attachment for attachment in record.get("attachments", []) if attachment.get("kind") == "l1_report"]
        if len(attachments) > 1:
            self.failures.append("run has more than one formal l1_report attachment")
        for attachment in attachments:
            public_root = self.work_tree_ancestor(attachment["path"])
            if public_root is not None:
                self.failures.append(f"l1 report lives inside a git work tree ({public_root}); controlled reports must stay outside the repo")
            for failure in L1ReportValidator().validate(record, attachment["path"]):
                self.failures.append(f"l1 report: {failure}")

    def verify_l3_reports(self, record):
        attachments = [attachment for attachment in record.get("attachments", []) if attachment.get("kind") == "l3_report"]
        if len(attachments) > 1:
            self.failures.append("run has more than one formal l3_report attachment")
        for attachment in attachments:
            public_root = self.work_tree_ancestor(attachment["path"])
            if public_root is not None:
                self.failures.append(f"l3 report lives inside a git work tree ({public_root}); controlled reports must stay outside the repo")
            for failure in L3ReportValidator().validate(record, attachment["path"]):
                self.failures.append(f"l3 report: {failure}")

    def verify_l2_reports(self, record):
        attachments = [attachment for attachment in record.get("attachments", []) if attachment.get("kind") == "l2_report"]
        if len(attachments) > 1:
            self.failures.append("run has more than one formal l2_report attachment")
        for attachment in attachments:
            public_root = self.work_tree_ancestor(attachment["path"])
            if public_root is not None:
                self.failures.append(f"l2 report lives inside a git work tree ({public_root}); controlled reports must stay outside the repo")
            for failure in L2ReportValidator().validate(record, attachment["path"]):
                self.failures.append(f"l2 report: {failure}")

    def verify_final_acceptance(self, record, record_path):
        """A concluded verdict record, when present, must still match the run's attachments."""
        verdict_path = FinalAcceptanceJudge.verdict_path_for(record_path)
        if not verdict_path.is_file():
            return
        judge = FinalAcceptanceJudge(RunRecordStore.for_run(record_path), ArtifactFingerprinter())
        for failure in judge.revalidate_verdict(record, verdict_path):
            self.failures.append(f"final acceptance: {failure}")

    def verify_phase_shape(self, record):
        phase, status = record["phase"], record["status"]
        self.check(record["schema"] == SCHEMA, f"schema != {SCHEMA}")
        self.check(phase in PHASES, f"phase {phase!r} not in {PHASES}")
        self.check(record.get("run_id"), "run_id missing")
        variant = record.get("variant")
        if phase != "P3":
            self.check(variant is None, f"a {phase} run must not carry the P3-only variant marker: {variant!r}")
        elif variant is not None and variant not in P3_VARIANTS:
            # legacy P3 records predate the variant marker and read as controlnet candidates
            self.failures.append(f"P3 variant {variant!r} not in {P3_VARIANTS}")
        if phase == "P1":
            self.check(record.get("base_ckpt") is not None, "P1 must record its full-param continuation base")
            self.check(record.get("upstream") is None, "P1 must not carry an upstream run")
        else:
            self.check(record.get("upstream") is not None, f"{phase} must pin its frozen P1-DM upstream")
            self.check(record.get("base_ckpt") is None, f"{phase} must not carry a base checkpoint")
        if status == STATUS_FROZEN:
            self.check(record.get("selection") is not None, "frozen run has no selection basis")
            self.check(record.get("samples") is not None, "frozen run has no sample manifest")
            self.check(record.get("frozen_utc"), "frozen run has no frozen_utc")
        if variant is not None:
            inference_entries = [entry for entry in record.get("configs", []) if entry.get("role") == "inference"]
            self.check(
                len(inference_entries) == 1,
                "a P3 run must pin exactly one role=inference config (recorded official inference provenance: "
                "stage-0 baseline issue #60, ControlNet candidate issue #61)",
            )
        if variant == STAGE0_BASELINE and record.get("selection"):
            upstream = record.get("upstream") or {}
            self.check(
                record["selection"]["checkpoint"]["sha256"] == upstream.get("checkpoint", {}).get("sha256"),
                "stage-0 baseline selection must pin the upstream P1-DM checkpoint (zero training, no DM of its own)",
            )
            if any(attachment.get("kind") in FORMAL_LAYER_KINDS for attachment in record.get("attachments", [])):
                self.failures.append("a stage-0 baseline must not carry formal L1/L2/L3 report attachments (comparison floor, not a candidate)")
        if variant == CONTROLNET_CANDIDATE and record.get("selection"):
            # issue #61 acceptance criterion 3: a trained candidate pins its own ControlNet checkpoint,
            # never the upstream P1-DM (that would be a stage-0 baseline in disguise)
            upstream = record.get("upstream") or {}
            self.check(
                record["selection"]["checkpoint"]["sha256"] != upstream.get("checkpoint", {}).get("sha256"),
                "P3 ControlNet candidate selection must pin its own trained checkpoint, not the upstream P1-DM",
            )

    def verify_guard(self, record):
        guard = HoldoutGuard(ManifestSides.from_path(record["manifest"]["path"]))
        for entry in record["data_lists"]:
            try:
                guard.guard_data_list(entry, record["phase"])
            except ContractViolationError as violation:
                self.failures.append(f"data list guard: {violation}")
        for entry in record["configs"]:
            try:
                guard.guard_config(entry["path"])
            except ContractViolationError as violation:
                self.failures.append(f"config guard: {violation}")
        if record.get("selection"):
            for entry in record["selection"]["evidence"]:
                try:
                    guard.guard_evidence(entry["path"])
                except ContractViolationError as violation:
                    self.failures.append(f"selection evidence guard: {violation}")

    def verify_chain(self, record):
        upstream = record.get("upstream")
        if not upstream:
            return
        run_record = Path(upstream["run_record"])
        if not run_record.is_file():
            self.failures.append(f"upstream run record missing: {run_record}")
            return
        upstream_record = json.loads(run_record.read_text())
        expected = (
            upstream_record.get("phase") == UPSTREAM_PHASE
            and upstream_record.get("status") == STATUS_FROZEN
            and upstream_record.get("selection") is not None
            and upstream_record["selection"]["checkpoint"]["sha256"] == upstream["checkpoint"]["sha256"]
        )
        self.check(
            expected,
            f"upstream run {upstream['run_id']} is no longer a frozen {UPSTREAM_PHASE} candidate "
            "with the pinned DM checkpoint (a retrained DM invalidates this bypass)",
        )
        self.verify(upstream_record, record_path=run_record, chain_depth=1)

    @staticmethod
    def work_tree_ancestor(path):
        """The first ancestor of path carrying a .git directory (the public repo), or None."""
        for parent in Path(path).resolve().parents:
            if (parent / ".git").exists():
                return parent
        return None

    def verify_storage(self, record_path):
        # DUA rule: run records (weights, samples, per-case evidence) live in
        # controlled storage, never inside the public repository work tree.
        ancestor = self.work_tree_ancestor(record_path)
        if ancestor is not None:
            self.failures.append(f"record lives inside a git work tree ({ancestor}); controlled artifacts must stay outside the repo")

    def verify(self, record, record_path=None, chain_depth=0):
        self.verify_phase_shape(record)
        self.verify_hashes(record)
        self.verify_l1_reports(record)
        self.verify_l2_reports(record)
        self.verify_l3_reports(record)
        self.verify_guard(record)
        resolved_path = record_path or Path(record.get("run_id", "."))
        self.verify_storage(resolved_path)
        for failure in DmSourceLedger(RunRecordStore.for_run(resolved_path)).check_record(record):
            self.failures.append(f"dm source: {failure}")
        self.verify_final_acceptance(record, resolved_path)
        if chain_depth == 0:
            self.verify_chain(record)
        return self.failures


class ContractSelfTest:
    """Fixture-driven end-to-end check with synthetic non-subject ids (stdlib only)."""

    QUOTAS = {
        "GLI": {"train": ["FIXGLI-0000-000", "FIXGLI-0001-000"], "dev": ["FIXGLI-0100-000"], "holdout": ["FIXGLI-0200-000"]},
        "SSA": {"train": ["FIXSSA-0000-000"], "dev": ["FIXSSA-0100-000"], "holdout": ["FIXSSA-0200-000"]},
    }

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def write_fixture(self):
        root = self._workdir / "fixture"
        manifest = {"split_id": "selftest", "challenges": {}}
        for ch, sides in self.QUOTAS.items():
            manifest["challenges"][ch] = {"cases": dict(sides)}
        manifest_path = root / "phase_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        lists_dir = root / "lists"
        lists_dir.mkdir(parents=True)
        train_list = {"training": [{"sub": "GLI", "case": "FIXGLI-0000-000"}, {"sub": "SSA", "case": "FIXSSA-0000-000"}]}
        dev_list = {"training": [{"sub": "GLI", "case": "FIXGLI-0100-000"}]}
        holdout_list = {"training": [{"sub": "GLI", "case": "FIXGLI-0200-000"}]}
        mislabelled_list = {"training": [{"sub": "GLI", "case": "FIXGLI-0100-000"}]}
        combined_sided_list = {"training": [{"sub": "GLI", "case": "FIXGLI-0000-000"}, {"sub": "GLI", "case": "FIXGLI-0100-000"}]}
        replay_list = {"training": [{"sub": "MRRATE", "case": "AB12CD34EF"}, {"sub": "MRRATE", "case": "FG56HI78JK"}]}
        replay_collision_list = {"training": [{"sub": "MRRATE", "case": "FIXGLI-0000-000"}]}
        (lists_dir / "train.json").write_text(json.dumps(train_list))
        (lists_dir / "dev.json").write_text(json.dumps(dev_list))
        (lists_dir / "holdout.json").write_text(json.dumps(holdout_list))
        (lists_dir / "mislabelled.json").write_text(json.dumps(mislabelled_list))
        (lists_dir / "combined_sided.json").write_text(json.dumps(combined_sided_list))
        (lists_dir / "replay.json").write_text(json.dumps(replay_list))
        (lists_dir / "replay_collision.json").write_text(json.dumps(replay_collision_list))

        (root / "env_config.json").write_text('{"lr": 2e-06, "n_epochs": 100}\n')
        (root / "model_config.json").write_text('{"batch_size": 1}\n')
        (root / "infer_config.json").write_text(
            '{"schema": "brats-p3-stage0-infer/1", "scheduler": "RFlowScheduler", "num_inference_steps": 30, '
            '"cfg_guidance_scale": 10.0, "strength": 0.9, "modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": 34}, '
            '"seed_rule": "sha256"}\n'
        )
        (root / "base_ckpt.pt").write_bytes(b"rflow-mr-brain-v1-fixture")
        (root / "candidate.pt").write_bytes(b"candidate-fixture")
        (root / "controlnet_candidate.pt").write_bytes(b"controlnet-candidate-fixture")
        dev_evidence = {"metrics": [{"sub": "GLI", "case": "FIXGLI-0100-000", "fid": 0.42}]}
        (root / "dev_metrics.json").write_text(json.dumps(dev_evidence))
        holdout_evidence = {"metrics": [{"sub": "GLI", "case": "FIXGLI-0200-000", "fid": 0.1}]}
        (root / "holdout_metrics.json").write_text(json.dumps(holdout_evidence))
        train_evidence = {"metrics": [{"sub": "SSA", "case": "FIXSSA-0000-000", "loss": 0.1}]}
        (root / "train_metrics.json").write_text(json.dumps(train_evidence))
        (root / "empty_metrics.json").write_text('{"summary": {"aggregate_fid": 0.5}}\n')
        (root / "samples.json").write_text('{"samples": ["sample-t1n-000.nii.gz"]}\n')
        (root / "l1_report.json").write_text('{"fid": {"t1n": 0.05}}\n')
        (root / "invalid_l1_report.json").write_text('{"schema": "brats-l1-report/1", "binding": {"run_id": "wrong-run"}}\n')
        (root / "platform.json").write_text('{"world_size": 1, "amp_dtype": "bf16"}\n')
        return root

    def write_l1_report(self, path, record, passing=False):
        interval = {"point": 0.4, "ci95": [0.3, 0.5]}
        baseline = {"planes": {plane: interval for plane in L1_PLANES}, "mean": interval, "mean_bootstrap_median": 0.4}
        generated_interval = {"point": 0.8, "ci95": [0.7, 0.9 if passing else 1.1]}
        generated = {"planes": {plane: generated_interval for plane in L1_PLANES}, "mean": generated_interval}
        fid_results = []
        for challenge in self.QUOTAS:
            for modality in L1_MODALITIES:
                fid_results.append(
                    {
                        "challenge": challenge,
                        "target_modality": modality,
                        "generated_source_modalities": [source for source in L1_MODALITIES if source != modality] if record["phase"] == "P3" else [],
                        "generated_vs_holdout": generated,
                        "train_vs_holdout_baseline": baseline,
                        "threshold": 1.0,
                        "verdict": "pass" if passing else "fail",
                    }
                )
        p3_results = []
        if record["phase"] == "P3":
            paired_interval = {"point": 0.11, "ci95": [0.01, 0.20]}
            ssim_interval = {"point": 0.03, "ci95": [0.01, 0.04]}
            for challenge in self.QUOTAS:
                for source in L1_MODALITIES:
                    for target in L1_MODALITIES:
                        if source == target:
                            continue
                        exceptional = (source, target) == L1_T1N_TO_T1C
                        p3_results.append(
                            {
                                "challenge": challenge,
                                "src_modality": source,
                                "target_modality": target,
                                "case_count": 2,
                                "mae_relative_reduction": paired_interval,
                                "ssim_increase": ssim_interval,
                                "gate_applicable": not exceptional,
                                "verdict": "not_applicable_known_unobservable" if exceptional else "pass",
                            }
                        )
        report = {
            "schema": L1_SCHEMA,
            "binding": {
                "run_id": record["run_id"],
                "phase": record["phase"],
                "manifest_sha256": record["manifest"]["sha256"],
                "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
                "samples_sha256": record["samples"]["sha256"],
            },
            "protocol": {
                "feature_extractor": {"name": L1_FEATURE_EXTRACTOR, "weights_sha256": "f" * 64},
                "mr_preprocessing": L1_MR_PREPROCESSING,
                "planes": list(L1_PLANES),
                "bootstrap": {"method": "case_level_percentile_pcg64", "confidence_level": 0.95, "resamples": 32},
                "fid_multiplier": 2.5,
            },
            "fid_results": fid_results,
            "p3_paired_results": p3_results,
            "summary": {"verdict": "pass" if passing else "fail"},
        }
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def write_l3_report(self, path, record):
        challenges = tuple(sorted(self.QUOTAS))
        per_cell = 5
        coverage = []
        for challenge in challenges:
            for modality in L3_MODALITIES:
                coverage.append({"challenge": challenge, "target_modality": modality, "real": per_cell, "synth": per_cell})
        real_total = per_cell * len(challenges) * len(L3_MODALITIES)  # 5 * 2 * 4 = 40
        per_reviewer = []
        for reviewer in ("R1", "R2"):
            per_reviewer.append(
                {
                    "reviewer": reviewer,
                    "n": real_total + real_total,
                    "balanced_accuracy": 0.5,
                    "confusion": {
                        "real_said_real": real_total // 2,
                        "real_said_synth": real_total // 2,
                        "synth_said_real": real_total // 2,
                        "synth_said_synth": real_total // 2,
                    },
                    "ci95": [0.42, 0.58],
                    "verdict": "pass",
                }
            )
        pooled = {"reviewers": 2, "n": per_reviewer[0]["n"] * 2, "balanced_accuracy": 0.5, "ci95": [0.44, 0.56], "verdict": "pass"}
        likert = []
        for dimension in L3_DIMENSIONS:
            phase = {"point": 4.2, "ci95_lower": 4.1, "n": per_reviewer[0]["n"] * 2, "na": 0, "verdict": "pass"}
            per_modality = {
                modality: {"point": 4.2, "ci95_lower": 4.1, "n": per_cell * 2 * len(challenges), "na": 0, "verdict": "pass"}
                for modality in L3_MODALITIES
            }
            likert.append({"dimension": dimension, "phase": phase, "per_modality": per_modality, "fleiss_kappa": 0.4})
        report = {
            "schema": L3_SCHEMA,
            "binding": {
                "run_id": record["run_id"],
                "phase": record["phase"],
                "manifest_sha256": record["manifest"]["sha256"],
                "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
                "samples_sha256": record["samples"]["sha256"],
            },
            "protocol": {
                "reviewers": 2,
                "dimensions": list(L3_DIMENSIONS),
                "target_modalities": list(L3_MODALITIES),
                "visual_turing_ci_window": [0.40, 0.60],
                "likert_minimum": 4.0,
                "likert_scale": {"min": 1, "max": 5},
                "confidence_level": 0.95,
                "bootstrap": {"method": "entry_level_stratified_percentile_mt19937", "resamples": 100, "seed": 20260821},
                "per_cell": per_cell,
                "total_entries": per_cell * 2 * len(challenges) * len(L3_MODALITIES),
            },
            "coverage": coverage,
            "provenance": {"catalog_sha256": "c" * 64, "blind_map_sha256": "b" * 64},
            "visual_turing": {"per_reviewer": per_reviewer, "pooled": pooled, "verdict": "pass", "fleiss_kappa": 0.35},
            "likert": likert,
            "verdict": {"visual_turing": "pass", "likert": "pass", "overall": "pass"},
        }
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def write_l2_report(self, path, record, failing_challenges=(), undecided_challenges=()):
        """A five-challenge L2 fixture; failing/undecided challenge names override their verdicts."""
        per_challenge = {}
        for challenge in L2_CHALLENGES:
            n_failed = 1 if challenge in undecided_challenges else 0
            passed = challenge not in failing_challenges and challenge not in undecided_challenges
            if challenge in undecided_challenges:
                verdict = "undecided"
            elif challenge in failing_challenges:
                verdict = "fail"
            else:
                verdict = "pass"
            tost = [
                {
                    "quantity": "vol_wt_rel",
                    "margin": 0.2802,
                    "ci90_low": -0.02,
                    "ci90_high": 0.02,
                    "n_cases": 6,
                    "n_excluded": 0,
                    "exclusion_reasons": {},
                    "passed": passed if challenge not in undecided_challenges else True,
                }
            ]
            round_trip = None
            if record["phase"] == "P2":
                round_trip = [
                    {
                        "region": region,
                        "floor": 0.0,
                        "bound": 0.9,
                        "n_cases": 6,
                        "n_excluded": 0,
                        "vacuous_pass": False,
                        "passed": passed,
                    }
                    for region in ("WT", "TC", "ET")
                ]
            per_challenge[challenge] = {
                "challenge": challenge,
                "n_observations": 12,
                "failure_audit": {
                    "n_failed": n_failed,
                    "breakdown": {"input_fail": 0, "run_fail": n_failed, "hier_viol": 0},
                    "n_failed_by_side": {"gen": n_failed, "real": 0},
                    "wilson_95_upper": 0.08 if n_failed else 0.0,
                },
                "r_fail_point": 0.0,
                "tost": tost,
                "round_trip": round_trip,
                "verdict": verdict,
            }
            if verdict == "undecided":
                per_challenge[challenge]["reason"] = (
                    "instrument failure on tested samples (input/run/hierarchy); blocks final acceptance -- "
                    "fix direction is the instrument or a re-run, not the candidate"
                )
        verdicts = [info["verdict"] for info in per_challenge.values()]
        overall = "undecided" if "undecided" in verdicts else "pass" if all(v == "pass" for v in verdicts) else "fail"
        report = {
            "schema": L2_SCHEMA,
            "title": "L2 冻结仪器最终验收报告",
            "phase": record["phase"],
            "run_id": record["run_id"],
            "binding": {
                "run_id": record["run_id"],
                "phase": record["phase"],
                "manifest_sha256": record["manifest"]["sha256"],
                "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
                "samples_sha256": record["samples"]["sha256"],
            },
            "provisional_challenges": [],
            "challenges_missing": [],
            "complete_coverage": True,
            "overall_verdict": overall,
            "per_challenge": per_challenge,
        }
        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report

    def expect_reject(self, action, label):
        try:
            action()
        except ContractViolationError:
            return
        self.failures.append(f"expected rejection but succeeded: {label}")

    def store_at(self, path):
        return RunRecordStore(path)

    def _open_passing_candidate(self, store, fingerprinter, fixture, run_id, checkpoint=None):
        """A frozen P1 candidate shell (init -> select -> freeze) ready for report attachments."""
        run_path = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P1",
            run_id,
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json")],
            fixture / "base_ckpt.pt",
            None,
            None,
        )
        SelectionRecorder(store, fingerprinter).select(
            run_path, checkpoint or fixture / "candidate.pt", "dev FID trend, early stop patience 10", [fixture / "dev_metrics.json"], epoch=7
        )
        CandidateFreezer(store, fingerprinter).freeze(run_path, fixture / "samples.json")
        return run_path

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        fixture = self.write_fixture()
        fingerprinter = ArtifactFingerprinter()

        # --- P1 positive path: init -> select -> freeze -> attach -> verify
        records = self._workdir / "records"
        store = self.store_at(records)
        initializer = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json"))
        p1_path = initializer.init(
            "P1",
            "p1-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json"), ("model", fixture / "model_config.json")],
            [("train", fixture / "lists/train.json")],
            fixture / "base_ckpt.pt",
            None,
            fixture / "platform.json",
        )
        SelectionRecorder(store, fingerprinter).select(
            p1_path,
            fixture / "candidate.pt",
            "dev FID trend, early stop patience 10",
            [fixture / "dev_metrics.json"],
            epoch=5,
        )
        CandidateFreezer(store, fingerprinter).freeze(p1_path, fixture / "samples.json")
        self.write_l1_report(fixture / "l1_report.json", store.load_by_path(p1_path))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", fixture / "invalid_l1_report.json"),
            "unbound L1 report",
        )
        invalid_summary = json.loads((fixture / "l1_report.json").read_text())
        invalid_summary["summary"] = {"verdict": "pass"}
        (fixture / "invalid_l1_summary.json").write_text(json.dumps(invalid_summary))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", fixture / "invalid_l1_summary.json"),
            "L1 summary inconsistent with results",
        )
        incomplete = json.loads((fixture / "l1_report.json").read_text())
        incomplete["fid_results"][0]["verdict"] = "undecided"
        incomplete["summary"] = {"verdict": "undecided"}
        (fixture / "incomplete_l1_report.json").write_text(json.dumps(incomplete))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", fixture / "incomplete_l1_report.json"),
            "incomplete L1 metrics",
        )
        invalid_provenance = json.loads((fixture / "l1_report.json").read_text())
        invalid_provenance["protocol"]["mr_preprocessing"] = "unverified"
        (fixture / "invalid_l1_provenance.json").write_text(json.dumps(invalid_provenance))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", fixture / "invalid_l1_provenance.json"),
            "unverified L1 feature provenance",
        )
        public_report = self._workdir / "l1-public-repo" / "l1_report.json"
        (public_report.parent / ".git").mkdir(parents=True)
        public_report.write_text((fixture / "l1_report.json").read_text())
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", public_report),
            "L1 report in public work tree",
        )
        ReportAttacher(store, fingerprinter).attach(p1_path, "l1_report", fixture / "l1_report.json")
        self.write_l3_report(fixture / "l3_report.json", store.load_by_path(p1_path))
        invalid_l3 = json.loads((fixture / "l3_report.json").read_text())
        invalid_l3["visual_turing"]["verdict"] = "fail"
        invalid_l3["verdict"] = {"visual_turing": "fail", "likert": "pass", "overall": "pass"}
        (fixture / "invalid_l3_verdict.json").write_text(json.dumps(invalid_l3))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l3_report", fixture / "invalid_l3_verdict.json"),
            "L3 non-compensatory AND mismatch",
        )
        malformed_l3 = json.loads((fixture / "l3_report.json").read_text())
        del malformed_l3["protocol"]["bootstrap"]
        malformed_l3["visual_turing"]["pooled"] = None
        (fixture / "malformed_l3_report.json").write_text(json.dumps(malformed_l3))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l3_report", fixture / "malformed_l3_report.json"),
            "malformed L3 report rejected (no crash)",
        )
        ReportAttacher(store, fingerprinter).attach(p1_path, "l3_report", fixture / "l3_report.json")
        verifier = RunVerifier(fingerprinter)
        verifier.verify(store.load_by_path(p1_path), record_path=p1_path)
        self.failures += [f"p1 positive path: {f}" for f in verifier.failures]

        # --- guards: holdout and mislabelled lists, holdout/train selection evidence
        for label, data_lists in (
            ("holdout data list", [("train", fixture / "lists/holdout.json")]),
            ("mislabelled side list", [("train", fixture / "lists/mislabelled.json")]),
            ("replay collision list", [("train", fixture / "lists/train.json"), ("replay", fixture / "lists/replay_collision.json")]),
        ):
            fresh = self.store_at(self._workdir / "records_reject")
            self.expect_reject(
                lambda fresh=fresh, data_lists=data_lists: RunInitializer(
                    fresh, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")
                ).init(
                    "P1",
                    f"p1-bad-{label.replace(' ', '-')}",
                    fixture / "phase_manifest.json",
                    [("env", fixture / "env_config.json")],
                    data_lists,
                    fixture / "base_ckpt.pt",
                    None,
                    None,
                ),
                label,
            )

        # P1 replay positive path: train + external replay list opens and verifies.
        replay_store = self.store_at(self._workdir / "records_replay")
        replay_path = RunInitializer(replay_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P1",
            "p1-replay-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json"), ("replay", fixture / "lists/replay.json")],
            fixture / "base_ckpt.pt",
            None,
            None,
        )
        replay_verifier = RunVerifier(fingerprinter)
        replay_verifier.verify(replay_store.load_by_path(replay_path), record_path=replay_path)
        self.failures += [f"p1 replay positive path: {f}" for f in replay_verifier.failures]

        frozen_run = store.load_by_path(p1_path)
        if frozen_run["status"] != "frozen" or frozen_run.get("samples") is None:
            self.failures.append("p1 freeze: status or sample manifest not recorded")
        self.expect_reject(
            lambda: SelectionRecorder(store, fingerprinter).select(
                p1_path, fixture / "candidate.pt", "re-select", [fixture / "dev_metrics.json"], None
            ),
            "selection on frozen run",
        )
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l4_report", fixture / "l1_report.json"),
            "unknown attachment kind",
        )

        # Evidence guards on an open run: holdout and train evidence must be rejected.
        p1_open = self.store_at(self._workdir / "records_open")
        open_path = RunInitializer(p1_open, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P1",
            "p1-open",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json")],
            fixture / "base_ckpt.pt",
            None,
            None,
        )
        open_store = p1_open
        for label, evidence in (
            ("holdout evidence", fixture / "holdout_metrics.json"),
            ("train evidence", fixture / "train_metrics.json"),
            ("case-free evidence", fixture / "empty_metrics.json"),
        ):
            self.expect_reject(
                lambda evidence=evidence: SelectionRecorder(open_store, fingerprinter).select(
                    open_path, fixture / "candidate.pt", "rule", [evidence], None
                ),
                label,
            )
        self.expect_reject(
            lambda: CandidateFreezer(open_store, fingerprinter).freeze(open_path, fixture / "samples.json"),
            "freeze without selection",
        )

        # --- L2 attachment (issue #58): binding, coverage, verdict consistency
        p1_record = store.load_by_path(p1_path)
        self.write_l2_report(fixture / "l2_report.json", p1_record)
        l2_mutations = (
            ("unbound L2 report", lambda r: r["binding"].update(run_id="wrong-run")),
            (
                "provisional L2 coverage",
                lambda r: (r.update(provisional_challenges=["GLI"], complete_coverage=False), r.update(challenges_missing=["PED"]))[0],
            ),
            (
                "L2 overall-verdict mismatch",
                lambda r: (
                    r["per_challenge"]["SSA"].update(verdict="fail", tost=[dict(r["per_challenge"]["SSA"]["tost"][0], passed=False)]),
                    r,
                )[1],
            ),
            ("L2 P1 carrying round-trip evidence", lambda r: r["per_challenge"]["GLI"].update(round_trip=[{"region": "WT", "passed": True}])),
            (
                "L2 challenge verdict disagreeing with its evidence",
                lambda r: r["per_challenge"]["MEN"].update(verdict="pass", tost=[dict(r["per_challenge"]["MEN"]["tost"][0], passed=False)]),
            ),
        )
        for label, mutate in l2_mutations:
            report = json.loads((fixture / "l2_report.json").read_text())
            mutate(report)
            bad_path = fixture / f"bad_l2_{label.replace(' ', '_')}.json"
            bad_path.write_text(json.dumps(report))
            self.expect_reject(
                lambda bad_path=bad_path: ReportAttacher(store, fingerprinter).attach(p1_path, "l2_report", bad_path),
                label,
            )
        malformed_l2 = json.loads((fixture / "l2_report.json").read_text())
        malformed_l2["per_challenge"]["GLI"] = "not-an-object"
        (fixture / "malformed_l2_report.json").write_text(json.dumps(malformed_l2))
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(p1_path, "l2_report", fixture / "malformed_l2_report.json"),
            "malformed L2 report rejected (no crash)",
        )
        ReportAttacher(store, fingerprinter).attach(p1_path, "l2_report", fixture / "l2_report.json")

        # --- final acceptance: non-compensatory AND, traceable blockers, DM-source freeze
        judge = FinalAcceptanceJudge(store, fingerprinter)
        blocked_entry, _blocked_path = judge.conclude(p1_path)  # L1 fail + L2/L3 pass -> blocked
        if blocked_entry["verdict"] != "blocked" or not blocked_entry["blocked_reasons"]:
            self.failures.append("L1-failing run must conclude blocked with traceable reasons")
        if not any(reason.startswith("L1 FID") for reason in blocked_entry["blocked_reasons"]):
            self.failures.append("blocked reasons must cite the failing L1 criteria (no offset by passing layers)")
        if DmSourceLedger(store).current() is not None:
            self.failures.append("a blocked conclusion must not register a DM source")
        self.expect_reject(lambda: judge.conclude(p1_path), "double conclude (immutable verdict)")
        flipped_path = FinalAcceptanceJudge.verdict_path_for(p1_path)
        flipped = json.loads(flipped_path.read_text())
        flipped["verdict"] = "pass"  # a hand-edited flip must not survive verification
        flipped["blocked_reasons"] = []
        flipped_path.write_text(json.dumps(flipped))
        flip_verifier = RunVerifier(fingerprinter)
        flip_verifier.verify(store.load_by_path(p1_path), record_path=p1_path)
        if not any("non-compensatory AND" in failure for failure in flip_verifier.failures):
            self.failures.append(f"flipped verdict record must fail verification, got {flip_verifier.failures}")

        p1_undecided_path = self._open_passing_candidate(store, fingerprinter, fixture, "p1-undecided")
        self.write_l1_report(fixture / "l1_pass_undecided_run.json", store.load_by_path(p1_undecided_path), passing=True)
        ReportAttacher(store, fingerprinter).attach(p1_undecided_path, "l1_report", fixture / "l1_pass_undecided_run.json")
        self.write_l2_report(fixture / "l2_undecided_report.json", store.load_by_path(p1_undecided_path), undecided_challenges=("SSA",))
        ReportAttacher(store, fingerprinter).attach(p1_undecided_path, "l2_report", fixture / "l2_undecided_report.json")
        self.write_l3_report(fixture / "l3_undecided_run_report.json", store.load_by_path(p1_undecided_path))
        ReportAttacher(store, fingerprinter).attach(p1_undecided_path, "l3_report", fixture / "l3_undecided_run_report.json")
        undecided_entry, _ = FinalAcceptanceJudge(store, fingerprinter).conclude(p1_undecided_path)
        if undecided_entry["verdict"] != "blocked" or not any("L2 SSA: undecided" in r for r in undecided_entry["blocked_reasons"]):
            self.failures.append(f"L2 undecided must block final acceptance traceably, got {undecided_entry['blocked_reasons']}")
        if DmSourceLedger(store).current() is not None:
            self.failures.append("an L2-undecided conclusion must not register a DM source")

        p1_final_path = self._open_passing_candidate(store, fingerprinter, fixture, "p1-final")
        p1_final_record = store.load_by_path(p1_final_path)
        self.write_l1_report(fixture / "l1_pass_report.json", p1_final_record, passing=True)
        ReportAttacher(store, fingerprinter).attach(p1_final_path, "l1_report", fixture / "l1_pass_report.json")
        self.write_l2_report(fixture / "l2_pass_report.json", p1_final_record)
        ReportAttacher(store, fingerprinter).attach(p1_final_path, "l2_report", fixture / "l2_pass_report.json")
        self.write_l3_report(fixture / "l3_pass_report.json", p1_final_record)
        ReportAttacher(store, fingerprinter).attach(p1_final_path, "l3_report", fixture / "l3_pass_report.json")
        pass_entry, _ = FinalAcceptanceJudge(store, fingerprinter).conclude(p1_final_path)
        if pass_entry["verdict"] != "pass" or pass_entry["blocked_reasons"] != []:
            self.failures.append("all-pass run must conclude pass with no blockers")
        if pass_entry["dm_source_registered"] is not True:
            self.failures.append("a passing P1 conclusion must register the DM source")
        source = DmSourceLedger(store).current()
        if source is None or source["run_id"] != "p1-final":
            self.failures.append("dm source ledger does not point at the passing candidate")
        elif source["checkpoint"]["sha256"] != p1_final_record["selection"]["checkpoint"]["sha256"]:
            self.failures.append("dm source froze the wrong DM checkpoint")
        self.expect_reject(
            lambda: FinalAcceptanceJudge(store, fingerprinter).conclude(p1_final_path),
            "double conclude after registration",
        )
        p1_final_verifier = RunVerifier(fingerprinter)
        p1_final_verifier.verify(store.load_by_path(p1_final_path), record_path=p1_final_path)
        self.failures += [f"p1-final verify: {f}" for f in p1_final_verifier.failures]

        # --- phase chain: P2 needs the frozen *registered* P1-DM; P3 must not pin a P2 run
        self.expect_reject(
            lambda: RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P2",
                "p2-replay",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json"), ("replay", fixture / "lists/replay.json")],
                None,
                p1_path,
                None,
            ),
            "P2 with a replay list",
        )
        self.expect_reject(
            lambda: RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P2",
                "p2-early",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                None,
                open_path,
                None,
            ),
            "P2 pinned to an open P1",
        )
        self.expect_reject(
            lambda: RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P2",
                "p2-offsource",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                None,
                p1_path,
                None,
            ),
            "P2 pinned to a frozen P1 that is not the registered DM source",
        )
        p2_path = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P2",
            "p2-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json")],
            None,
            p1_final_path,
            None,
        )
        # P2 fold-split combined list (train+dev under one train label) opens and
        # verifies; P1 must reject the same list (no dev leak into a full-param
        # continuation train list). spec #51 decision 8.
        p2_combined_path = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P2",
            "p2-combined-split",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/combined_sided.json")],
            None,
            p1_final_path,
            None,
        )
        comb_verifier = RunVerifier(fingerprinter)
        comb_verifier.verify(store.load_by_path(p2_combined_path), record_path=p2_combined_path)
        self.failures += [f"p2 combined-split list: {f}" for f in comb_verifier.failures]
        comb_reject_store = self.store_at(self._workdir / "records_comb_reject")
        self.expect_reject(
            lambda: RunInitializer(comb_reject_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P1",
                "p1-combined-split",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/combined_sided.json")],
                fixture / "base_ckpt.pt",
                None,
                None,
            ),
            "P1 rejects a fold-split combined train+dev list",
        )
        SelectionRecorder(store, fingerprinter).select(
            p2_path, fixture / "candidate.pt", "dev light acceptance", [fixture / "dev_metrics.json"], None
        )
        CandidateFreezer(store, fingerprinter).freeze(p2_path, fixture / "samples.json")
        self.expect_reject(
            lambda: RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P3",
                "p3-warm",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                None,
                p2_path,
                None,
            ),
            "P3 warm-started from a P2 run",
        )
        self.expect_reject(
            lambda: RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
                "P1",
                "p1-with-upstream",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                fixture / "base_ckpt.pt",
                p1_path,
                None,
            ),
            "P1 with an upstream run",
        )
        # P3 positive path: pins the same registered P1-DM (independent init), full record verifies.
        p3_path = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P3",
            "p3-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json"), ("inference", fixture / "infer_config.json")],
            [("train", fixture / "lists/train.json")],
            None,
            p1_final_path,
            None,
        )
        SelectionRecorder(store, fingerprinter).select(
            p3_path, fixture / "controlnet_candidate.pt", "dev light acceptance", [fixture / "dev_metrics.json"], None
        )
        CandidateFreezer(store, fingerprinter).freeze(p3_path, fixture / "samples.json")
        self.write_l1_report(fixture / "p3_l1_report.json", store.load_by_path(p3_path))
        ReportAttacher(store, fingerprinter).attach(p3_path, "l1_report", fixture / "p3_l1_report.json")

        chain_verifier = RunVerifier(fingerprinter)
        chain_verifier.verify(store.load_by_path(p2_path), record_path=p2_path)
        self.failures += [f"p2 chain: {f}" for f in chain_verifier.failures]
        p3_verifier = RunVerifier(fingerprinter)
        p3_verifier.verify(store.load_by_path(p3_path), record_path=p3_path)
        self.failures += [f"p3 chain: {f}" for f in p3_verifier.failures]

        # --- stage-0 baseline (issue #60): zero-training P3 variant, comparison floor only
        stage0_initializer = RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json"))
        self.expect_reject(
            lambda: stage0_initializer.init(
                "P3",
                "p3-stage0",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                None,
                p1_final_path,
                None,
                variant="bogus-variant",
            ),
            "P3 init with an unknown variant",
        )
        self.expect_reject(
            lambda: stage0_initializer.init(
                "P3",
                "p3-stage0-noinfer",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                None,
                p1_final_path,
                None,
                variant=STAGE0_BASELINE,
            ),
            "stage-0 without the pinned inference config",
        )
        self.expect_reject(
            lambda: stage0_initializer.init(
                "P1",
                "p1-stage0",
                fixture / "phase_manifest.json",
                [("env", fixture / "env_config.json")],
                [("train", fixture / "lists/train.json")],
                fixture / "base_ckpt.pt",
                None,
                None,
                variant=STAGE0_BASELINE,
            ),
            "variant on a P1 run",
        )
        stage0_path = stage0_initializer.init(
            "P3",
            "p3-stage0",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json"), ("inference", fixture / "infer_config.json")],
            [("train", fixture / "lists/train.json")],
            None,
            p1_final_path,
            None,
            variant=STAGE0_BASELINE,
        )
        stage0_record = store.load_by_path(stage0_path)
        if stage0_record.get("variant") != STAGE0_BASELINE:
            self.failures.append("stage-0 run record must carry the stage0-baseline variant marker")
        self.expect_reject(
            lambda: SelectionRecorder(store, fingerprinter).select(
                stage0_path, fixture / "base_ckpt.pt", "zero-training baseline", [fixture / "dev_metrics.json"], None
            ),
            "stage-0 selecting a checkpoint that is not the upstream DM",
        )
        upstream_ckpt = Path(stage0_record["upstream"]["checkpoint"]["path"])
        SelectionRecorder(store, fingerprinter).select(
            stage0_path, upstream_ckpt, "zero-training stage-0 baseline: DM is the upstream P1-DM selection", [fixture / "dev_metrics.json"], None
        )
        CandidateFreezer(store, fingerprinter).freeze(stage0_path, fixture / "samples.json")
        stage0_verifier = RunVerifier(fingerprinter)
        stage0_verifier.verify(store.load_by_path(stage0_path), record_path=stage0_path)
        self.failures += [f"p3-stage0 verify: {f}" for f in stage0_verifier.failures]
        self.expect_reject(
            lambda: ReportAttacher(store, fingerprinter).attach(stage0_path, "l1_report", fixture / "l1_report.json"),
            "stage-0 attaching a formal L1 report",
        )
        self.expect_reject(
            lambda: FinalAcceptanceJudge(store, fingerprinter).conclude(stage0_path),
            "stage-0 concluding final acceptance",
        )
        # A P1 record carrying the P3-only variant marker must fail verification.
        tainted = json.loads(Path(p1_path).read_text())
        tainted["variant"] = STAGE0_BASELINE
        tainted_path = Path(p1_path).parent.parent / "runs" / "p1-tainted-variant" / "run.json"
        tainted_path.parent.mkdir(parents=True, exist_ok=True)
        tainted_path.write_text(json.dumps(tainted))
        tainted_verifier = RunVerifier(fingerprinter)
        tainted_verifier.verify(store.load_by_path(tainted_path), record_path=tainted_path)
        if not any("variant" in failure for failure in tainted_verifier.failures):
            self.failures.append(f"a P1 record carrying a variant marker must fail verification, got {tainted_verifier.failures}")

        # A hand-edited stage-0 record with a formal report attached must fail verification.
        tainted_stage0 = json.loads(Path(stage0_path).read_text())
        tainted_stage0["attachments"] = [{"kind": "l1_report", "path": str(fixture / "l1_report.json"), "sha256": "0" * 64}]
        tainted_stage0_path = Path(stage0_path).parent.parent / "runs" / "p3-stage0-tainted" / "run.json"
        tainted_stage0_path.parent.mkdir(parents=True, exist_ok=True)
        tainted_stage0_path.write_text(json.dumps(tainted_stage0))
        tainted_stage0_verifier = RunVerifier(fingerprinter)
        tainted_stage0_verifier.verify(store.load_by_path(tainted_stage0_path), record_path=tainted_stage0_path)
        if not any("stage-0" in failure for failure in tainted_stage0_verifier.failures):
            self.failures.append(f"a stage-0 record carrying a formal report must fail verification, got {tainted_stage0_verifier.failures}")

        # --- DM retrain (issue #58): a later passing P1 supersedes the source; old bypasses mismatch explicitly
        retrained_ckpt = fixture / "candidate_retrained.pt"
        retrained_ckpt.write_bytes(b"candidate-retrained-fixture")
        p1_retrained_path = self._open_passing_candidate(store, fingerprinter, fixture, "p1-retrained", checkpoint=retrained_ckpt)
        retrained_record = store.load_by_path(p1_retrained_path)
        self.write_l1_report(fixture / "l1_retrained_report.json", retrained_record, passing=True)
        ReportAttacher(store, fingerprinter).attach(p1_retrained_path, "l1_report", fixture / "l1_retrained_report.json")
        self.write_l2_report(fixture / "l2_retrained_report.json", retrained_record)
        ReportAttacher(store, fingerprinter).attach(p1_retrained_path, "l2_report", fixture / "l2_retrained_report.json")
        self.write_l3_report(fixture / "l3_retrained_report.json", retrained_record)
        ReportAttacher(store, fingerprinter).attach(p1_retrained_path, "l3_report", fixture / "l3_retrained_report.json")
        retrained_entry, _ = FinalAcceptanceJudge(store, fingerprinter).conclude(p1_retrained_path)
        if retrained_entry["verdict"] != "pass":
            self.failures.append("retrained P1 fixture must conclude pass")
        superseded = DmSourceLedger(store).current()
        if superseded["run_id"] != "p1-retrained" or superseded["superseded_run_id"] != "p1-final":
            self.failures.append(f"dm source supersession not recorded: {superseded.get('run_id')} <- {superseded.get('superseded_run_id')}")
        stale_verifier = RunVerifier(fingerprinter)
        stale_verifier.verify(store.load_by_path(p2_path), record_path=p2_path)
        if not any("DM was retrained" in failure for failure in stale_verifier.failures):
            self.failures.append(f"retrained DM must explicitly mismatch the old bypass, got {stale_verifier.failures}")
        retrained_verifier = RunVerifier(fingerprinter)
        retrained_verifier.verify(store.load_by_path(p1_retrained_path), record_path=p1_retrained_path)
        self.failures += [f"p1-retrained verify: {f}" for f in retrained_verifier.failures]

        # --- tamper detection: a changed candidate checkpoint must fail verify
        tampered = fixture / "candidate.pt"
        original = tampered.read_bytes()
        tampered.write_bytes(b"tampered")
        tamper_verifier = RunVerifier(fingerprinter)
        tamper_verifier.verify(store.load_by_path(p1_path), record_path=p1_path)
        if not any("sha256 changed" in f or "missing on disk" in f for f in tamper_verifier.failures):
            self.failures.append("tamper detection: verify did not flag a changed candidate checkpoint")
        tampered.write_bytes(original)

        # --- public-output guard: a record root inside a git work tree must fail
        fake_repo = self._workdir / "fakerepo"
        (fake_repo / ".git").mkdir(parents=True, exist_ok=True)
        repo_records = fake_repo / "records" / "runs" / "p1-fixture" / "run.json"
        repo_records.parent.mkdir(parents=True, exist_ok=True)
        repo_records.write_text(p1_path.read_text())
        repo_verifier = RunVerifier(fingerprinter)
        repo_verifier.verify(store.load_by_path(p1_path), record_path=repo_records)
        if not any("git work tree" in f for f in repo_verifier.failures):
            self.failures.append("storage guard: verify did not flag a record inside a git work tree")

        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def parse_role(value):
        role, sep, path = value.partition("=")
        if not sep or not role or not path:
            raise argparse.ArgumentTypeError(f"expected ROLE=PATH, got {value!r}")
        return role, path

    p = sub.add_parser("init", help="open a run record (fingerprint inputs, enforce the phase chain)")
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--record-root", required=True)
    p.add_argument("--manifest", required=True, help="issue #52 phase manifest (split sides source)")
    p.add_argument("--config", dest="configs", action="append", type=parse_role, required=True)
    p.add_argument(
        "--data-list", dest="data_lists", action="append", type=parse_role, required=True, metavar="SIDE=PATH", help=f"side is one of {LIST_SIDES}"
    )
    p.add_argument("--run-id", help="stable id (default: <phase>-<utc stamp>)")
    p.add_argument("--base-ckpt", help="P1 only: the frozen rflow-mr-brain v1 checkpoint")
    p.add_argument("--upstream-run", help="P2/P3 only: run.json of the frozen P1 candidate")
    p.add_argument(
        "--variant",
        choices=P3_VARIANTS,
        help="P3 only: controlnet-candidate (default) or stage0-baseline (issue #60 zero-training img2img comparison floor)",
    )
    p.add_argument("--platform-json", help="embedded verbatim (e.g. the DCU smoke provenance)")
    p.set_defaults(handler="init")

    p = sub.add_parser("select", help="record the dev-side checkpoint selection basis")
    p.add_argument("--run", required=True, help="path to run.json (must be open)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rule", required=True, help="the pre-registered selection / early-stop rule")
    p.add_argument(
        "--evidence", dest="evidence", action="append", required=True, help="dev light-acceptance evidence file (JSON, may cite dev cases only)"
    )
    p.add_argument("--epoch", type=int)
    p.set_defaults(handler="select")

    p = sub.add_parser("freeze", help="freeze the candidate (requires selection + sample manifest)")
    p.add_argument("--run", required=True)
    p.add_argument("--samples", required=True, help="generated-sample manifest of this candidate")
    p.set_defaults(handler="freeze")

    p = sub.add_parser("attach", help=f"attach a post-freeze report ({'/'.join(ATTACH_KINDS)})")
    p.add_argument("--run", required=True)
    p.add_argument("--kind", required=True, choices=ATTACH_KINDS)
    p.add_argument("--path", required=True)
    p.set_defaults(handler="attach")

    p = sub.add_parser("conclude", help="non-compensatory L1∧L2∧L3 final acceptance over a frozen run (issue #58)")
    p.add_argument("--run", required=True, help="path to run.json (must be frozen with all three layer reports attached)")
    p.set_defaults(handler="conclude")

    p = sub.add_parser("verify", help="verify one run record or every record under a record root")
    p.add_argument("--run", help="path to run.json")
    p.add_argument("--record-root", help="verify every runs/*/run.json under this root")
    p.set_defaults(handler="verify")

    p = sub.add_parser("selftest", help="fixture-driven end-to-end check (synthetic ids, stdlib only)")
    p.add_argument("--workdir", required=True)
    p.set_defaults(handler="selftest")

    args = parser.parse_args(argv)
    fingerprinter = ArtifactFingerprinter()

    try:
        if args.handler == "init":
            sides = ManifestSides.from_path(args.manifest)
            path = RunInitializer(RunRecordStore(args.record_root), fingerprinter, sides).init(
                args.phase,
                args.run_id,
                args.manifest,
                args.configs,
                args.data_lists,
                args.base_ckpt,
                args.upstream_run,
                args.platform_json,
                args.variant,
            )
            print(f"opened {path} (status=open)")
            return 0
        if args.handler == "select":
            path = SelectionRecorder(RunRecordStore.for_run(args.run), fingerprinter).select(
                args.run, args.checkpoint, args.rule, args.evidence, args.epoch
            )
            print(f"selection recorded -> {path}")
            return 0
        if args.handler == "freeze":
            path = CandidateFreezer(RunRecordStore.for_run(args.run), fingerprinter).freeze(args.run, args.samples)
            print(f"candidate frozen -> {path}")
            return 0
        if args.handler == "attach":
            path = ReportAttacher(RunRecordStore.for_run(args.run), fingerprinter).attach(args.run, args.kind, args.path)
            print(f"{args.kind} attached -> {path}")
            return 0
        if args.handler == "conclude":
            entry, verdict_path = FinalAcceptanceJudge(RunRecordStore.for_run(args.run), fingerprinter).conclude(args.run)
            layers = ", ".join(f"{name}={layer['verdict']}" for name, layer in entry["layers"].items())
            print(f"FINAL ACCEPTANCE {entry['verdict'].upper()} ({layers}) -> {verdict_path}")
            for reason in entry["blocked_reasons"]:
                print(f"  blocked: {reason}", file=sys.stderr)
            return 0 if entry["verdict"] == "pass" else 1
        if args.handler == "verify":
            paths = [Path(args.run)] if args.run else RunRecordStore(args.record_root).all_record_paths()
            failures = []
            for record_path in paths:
                record = RunRecordStore(record_path.parent.parent).load_by_path(record_path)
                run_failures = RunVerifier(fingerprinter).verify(record, record_path=record_path)
                failures += [f"{record.get('run_id', record_path.parent.name)}: {f}" for f in run_failures]
            for failure in failures:
                print("FAIL " + failure, file=sys.stderr)
            if failures:
                return 1
            print(f"VERIFY PASS ({len(paths)} run{'s' if len(paths) > 1 else ''})")
            return 0
        failures = ContractSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0
    except ContractViolationError as violation:
        print(f"CONTRACT VIOLATION: {violation}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
