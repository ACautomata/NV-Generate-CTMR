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

SCHEMA = "brats-phase-run/1"
PHASES = ("P1", "P2", "P3")
STATUS_OPEN = "open"
STATUS_FROZEN = "frozen"
LIST_SIDES = ("train", "dev")  # holdout is never a data-list side
ATTACH_KINDS = ("l1_report", "l2_report", "l3_report", "env")
UPSTREAM_PHASE = "P1"  # P2 and P3 both hang off the same frozen P1-DM
L1_SCHEMA = "brats-l1-report/1"
L1_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
L1_PLANES = ("xy", "yz", "zx")
L1_VERDICTS = ("pass", "fail", "undecided")
L1_T1N_TO_T1C = ("t1n", "t1c")
L1_FEATURE_EXTRACTOR = "radimagenet_resnet50"
L1_MR_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad"


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

    def guard_data_list(self, list_entry):
        """A labelled list must exist, carry cases, and match its side label with no holdout."""
        path = Path(list_entry["path"])
        if not path.is_file():
            raise ContractViolationError(f"data list not found: {path}")
        pairs = self.scan_case_pairs(json.loads(path.read_text()))
        if not pairs:
            raise ContractViolationError(f"data list carries no (sub, case) entries: {path}")
        label = list_entry["side"]
        for challenge, case in pairs:
            side = self._sides.side_of(challenge, case)
            if side is None:
                raise ContractViolationError(f"{path}: ({challenge}, {case}) is not in the pinned manifest")
            if side == "holdout":
                raise ContractViolationError(
                    f"{path}: final-holdout case ({challenge}, {case}) must not enter a run input (holdout runs only after candidate freeze)"
                )
            if side != label:
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

    def init(self, phase, run_id, manifest_path, configs, data_lists, base_ckpt, upstream_run, platform_json):
        if phase not in PHASES:
            raise ContractViolationError(f"phase must be one of {PHASES}: {phase!r}")
        manifest_entry = self._fingerprinter.must_fingerprint(manifest_path, "phase manifest")
        config_entries = [{**self._fingerprinter.must_fingerprint(path, f"config {role}"), "role": role} for role, path in configs]
        if not config_entries:
            raise ContractViolationError("at least one --config ROLE=PATH is required")
        list_entries = []
        for side, path in data_lists:
            if side not in LIST_SIDES:
                raise ContractViolationError(f"data list side must be one of {LIST_SIDES}: {side!r}")
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
            guard.guard_data_list(entry)
        for entry in config_entries:
            guard.guard_config(entry["path"])

        record = {
            "schema": SCHEMA,
            "run_id": resolved_id,
            "phase": phase,
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
        expected = {
            "run_id": record.get("run_id"),
            "phase": record.get("phase"),
            "manifest_sha256": record.get("manifest", {}).get("sha256"),
            "candidate_checkpoint_sha256": record.get("selection", {}).get("checkpoint", {}).get("sha256"),
            "samples_sha256": record.get("samples", {}).get("sha256"),
        }
        if report.get("schema") != L1_SCHEMA:
            failures.append(f"L1 report schema != {L1_SCHEMA}")
        if not isinstance(binding, dict):
            failures.append("L1 report binding must be an object")
            return
        for key, value in expected.items():
            if binding.get(key) != value:
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


class ReportAttacher:
    """Attaches post-freeze L1/L2/L3/env reports (the only mutation allowed after freezing)."""

    def __init__(self, store, fingerprinter):
        self._store = store
        self._fingerprinter = fingerprinter

    def _assert_controlled_l1_report(self, path):
        for parent in Path(path).resolve().parents:
            if (parent / ".git").exists():
                raise ContractViolationError(f"l1_report lives inside a git work tree ({parent}); controlled reports must stay outside the repo")

    def attach(self, run_path, kind, path):
        record = self._store.load_by_path(run_path)
        if record["status"] != STATUS_FROZEN:
            raise ContractViolationError(
                f"run {record['run_id']} is {record['status']}; L1/L2/L3 holdout evidence attaches only to frozen candidates"
            )
        if kind not in ATTACH_KINDS:
            raise ContractViolationError(f"attachment kind must be one of {ATTACH_KINDS}: {kind!r}")
        if kind == "l1_report":
            self._assert_controlled_l1_report(path)
            if any(attachment["kind"] == "l1_report" for attachment in record["attachments"]):
                raise ContractViolationError(f"run {record['run_id']} already has a formal l1_report attachment")
            failures = L1ReportValidator().validate(record, path)
            if failures:
                raise ContractViolationError("invalid l1_report: " + "; ".join(failures))
        entry = {**self._fingerprinter.must_fingerprint(path, f"{kind} attachment"), "kind": kind}
        entry["attached_utc"] = self._store.now_utc()
        record["attachments"].append(entry)
        return self._store.write(record)


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

    def verify_phase_shape(self, record):
        phase, status = record["phase"], record["status"]
        self.check(record["schema"] == SCHEMA, f"schema != {SCHEMA}")
        self.check(phase in PHASES, f"phase {phase!r} not in {PHASES}")
        self.check(record.get("run_id"), "run_id missing")
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

    def verify_guard(self, record):
        guard = HoldoutGuard(ManifestSides.from_path(record["manifest"]["path"]))
        for entry in record["data_lists"]:
            try:
                guard.guard_data_list(entry)
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
        self.verify_guard(record)
        self.verify_storage(record_path or Path(record.get("run_id", ".")))
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
        (lists_dir / "train.json").write_text(json.dumps(train_list))
        (lists_dir / "dev.json").write_text(json.dumps(dev_list))
        (lists_dir / "holdout.json").write_text(json.dumps(holdout_list))
        (lists_dir / "mislabelled.json").write_text(json.dumps(mislabelled_list))

        (root / "env_config.json").write_text('{"lr": 2e-06, "n_epochs": 100}\n')
        (root / "model_config.json").write_text('{"batch_size": 1}\n')
        (root / "base_ckpt.pt").write_bytes(b"rflow-mr-brain-v1-fixture")
        (root / "candidate.pt").write_bytes(b"candidate-fixture")
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

    def write_l1_report(self, path, record):
        interval = {"point": 0.4, "ci95": [0.3, 0.5]}
        baseline = {"planes": {plane: interval for plane in L1_PLANES}, "mean": interval, "mean_bootstrap_median": 0.4}
        generated_interval = {"point": 0.8, "ci95": [0.7, 1.1]}
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
                        "verdict": "fail",
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
            "summary": {"verdict": "fail"},
        }
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def expect_reject(self, action, label):
        try:
            action()
        except ContractViolationError:
            return
        self.failures.append(f"expected rejection but succeeded: {label}")

    def store_at(self, path):
        return RunRecordStore(path)

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
        verifier = RunVerifier(fingerprinter)
        verifier.verify(store.load_by_path(p1_path), record_path=p1_path)
        self.failures += [f"p1 positive path: {f}" for f in verifier.failures]

        # --- guards: holdout and mislabelled lists, holdout/train selection evidence
        for label, data_lists in (
            ("holdout data list", [("train", fixture / "lists/holdout.json")]),
            ("mislabelled side list", [("train", fixture / "lists/mislabelled.json")]),
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

        # --- phase chain: P2 needs a frozen P1; P3 must not pin a P2 run
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
        p2_path = RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P2",
            "p2-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json")],
            None,
            p1_path,
            None,
        )
        SelectionRecorder(open_store, fingerprinter).select(
            p2_path, fixture / "candidate.pt", "dev light acceptance", [fixture / "dev_metrics.json"], None
        )
        CandidateFreezer(open_store, fingerprinter).freeze(p2_path, fixture / "samples.json")
        self.expect_reject(
            lambda: RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
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
        # P3 positive path: pins the same frozen P1-DM (independent init), full record verifies.
        p3_path = RunInitializer(open_store, fingerprinter, ManifestSides.from_path(fixture / "phase_manifest.json")).init(
            "P3",
            "p3-fixture",
            fixture / "phase_manifest.json",
            [("env", fixture / "env_config.json")],
            [("train", fixture / "lists/train.json")],
            None,
            p1_path,
            None,
        )
        SelectionRecorder(open_store, fingerprinter).select(
            p3_path, fixture / "candidate.pt", "dev light acceptance", [fixture / "dev_metrics.json"], None
        )
        CandidateFreezer(open_store, fingerprinter).freeze(p3_path, fixture / "samples.json")
        self.write_l1_report(fixture / "p3_l1_report.json", open_store.load_by_path(p3_path))
        ReportAttacher(open_store, fingerprinter).attach(p3_path, "l1_report", fixture / "p3_l1_report.json")

        chain_verifier = RunVerifier(fingerprinter)
        chain_verifier.verify(open_store.load_by_path(p2_path), record_path=p2_path)
        self.failures += [f"p2 chain: {f}" for f in chain_verifier.failures]
        p3_verifier = RunVerifier(fingerprinter)
        p3_verifier.verify(open_store.load_by_path(p3_path), record_path=p3_path)
        self.failures += [f"p3 chain: {f}" for f in p3_verifier.failures]

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
