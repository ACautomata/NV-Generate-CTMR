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

"""Run-lifecycle mutations: open, select, freeze, attach.

Migrated verbatim from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141). Every
mutation enforces its contract rule at write time:

- init fingerprints inputs and enforces the phase chain (a P2/P3 run pins a
  frozen P1 run's selected DM checkpoint -- the registered DM source; P3 can
  never pin a P2 ControlNet) and the P3 variant marker (issue #60);
- select records the dev-side checkpoint selection basis (no training-loss
  best, no holdout; a stage-0 baseline pins exactly the upstream DM);
- freeze is the one-way ``open -> frozen`` transition (selection required);
- attach is the only post-freeze mutation (formal layers validate against the
  frozen run; a stage-0 baseline never takes formal layer evidence).
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from ctmr.application.acceptance.contract.artifacts import ManifestSides
from ctmr.application.acceptance.contract.binding import STATUS_FROZEN
from ctmr.application.acceptance.contract.guard import HoldoutGuard
from ctmr.application.acceptance.contract.record import (
    LIST_SIDES,
    P3_VARIANTS,
    PHASES,
    SCHEMA,
    STAGE0_BASELINE,
    STATUS_OPEN,
    UPSTREAM_PHASE,
    CodeVersion,
    ContractViolationError,
)
from ctmr.application.acceptance.contract.registry import ATTACH_KINDS, FORMAL_LAYER_KINDS, LAYER_BY_KIND
from ctmr.domain.dmsource import DmSourceLedgerFactory, DmSourceViolationError
from ctmr.domain.identity import WeightsRef


class RunInitializer:
    """Opens a run record, fingerprinting inputs and enforcing the phase chain.

    ``ledger_factory`` is the injected ``(record_root) -> DmSourceLedger`` port
    (ADR-0019 §2/§3): the phase-chain gate consults the DM-source ledger
    through it, and a domain ledger violation translates into the contract's
    own violation type at this boundary. The concrete adapter is the
    composition root's choice (``ctmr.wiring.contract``), never this module's.
    """

    def __init__(self, store, fingerprinter, sides, ledger_factory: DmSourceLedgerFactory):
        self._store = store
        self._fingerprinter = fingerprinter
        self._sides = sides
        self._ledger_factory = ledger_factory

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
        try:
            self._ledger_factory(self._store.root()).check_upstream(upstream["run_id"], WeightsRef(sha256=checkpoint["sha256"]))
        except DmSourceViolationError as violation:
            raise ContractViolationError(str(violation)) from violation
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
            "code_version": CodeVersion(Path(__file__), self._fingerprinter).snapshot(),
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
        layer = LAYER_BY_KIND.get(kind)
        if layer is not None:
            self._assert_controlled_report(path, kind)
            if any(attachment["kind"] == kind for attachment in record["attachments"]):
                raise ContractViolationError(f"run {record['run_id']} already has a formal {kind} attachment")
            failures = layer.validator_factory().validate(record, path)
            if failures:
                raise ContractViolationError("invalid " + kind + ": " + "; ".join(failures))
        entry = {**self._fingerprinter.must_fingerprint(path, f"{kind} attachment"), "kind": kind}
        entry["attached_utc"] = self._store.now_utc()
        record["attachments"].append(entry)
        return self._store.write(record)
