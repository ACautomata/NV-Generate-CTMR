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

"""Run-record verification: hashes, guard, phase shape, chain, storage, verdict.

Migrated verbatim from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141).
Reconciles one run record against the contract: every fingerprinted entry
against the bytes on disk, every formal layer attachment revalidated through
the acceptance-layer registry, the holdout guard re-run, the phase chain
re-walked (a retrained DM explicitly mismatches old bypasses), the DM-source
ledger consulted, the concluded verdict record reconciled through the shared
domain kernel, and controlled-storage placement enforced.
"""

import json
from pathlib import Path

from ctmr.application.acceptance.contract.artifacts import ManifestSides
from ctmr.application.acceptance.contract.binding import STATUS_FROZEN
from ctmr.application.acceptance.contract.conclude import FinalAcceptanceJudge
from ctmr.application.acceptance.contract.guard import HoldoutGuard
from ctmr.application.acceptance.contract.record import (
    CONTROLNET_CANDIDATE,
    P3_VARIANTS,
    PHASES,
    SCHEMA,
    STAGE0_BASELINE,
    UPSTREAM_PHASE,
    ContractViolationError,
    RunRecordStore,
)
from ctmr.application.acceptance.contract.registry import ACCEPTANCE_LAYERS, FORMAL_LAYER_KINDS
from ctmr.domain.dmsource import DmSourceLedgerFactory, DmSourceViolationError


class RunVerifier:
    """Reconciles one run record against the contract: hashes, guard, phase chain, storage.

    ``ledger_factory`` is the injected ``(record_root) -> DmSourceLedger`` port
    (ADR-0019 §2/§3): the DM-source consultation draws one ledger per record
    root -- the chain recursion reaches the upstream record under its own root,
    so a factory over roots is the injection shape, not a single instance.
    """

    def __init__(self, fingerprinter, ledger_factory: DmSourceLedgerFactory):
        self._fingerprinter = fingerprinter
        self._ledger_factory = ledger_factory
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

    def verify_layer_reports(self, record):
        """Revalidates every formal layer attachment against the frozen run (registry-driven)."""
        for layer in ACCEPTANCE_LAYERS:
            kind = layer.kind
            attachments = [attachment for attachment in record.get("attachments", []) if attachment.get("kind") == kind]
            if len(attachments) > 1:
                self.failures.append(f"run has more than one formal {kind} attachment")
            for attachment in attachments:
                public_root = self.work_tree_ancestor(attachment["path"])
                if public_root is not None:
                    self.failures.append(f"{kind} report lives inside a git work tree ({public_root}); controlled reports must stay outside the repo")
                for failure in layer.validator_factory().validate(record, attachment["path"]):
                    self.failures.append(f"{kind} report: {failure}")

    def verify_final_acceptance(self, record, record_path):
        """A concluded verdict record, when present, must still match the run's attachments."""
        verdict_path = FinalAcceptanceJudge.verdict_path_for(record_path)
        if not verdict_path.is_file():
            return
        judge = FinalAcceptanceJudge(RunRecordStore.for_run(record_path), self._fingerprinter, self._ledger_factory)
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
        self.verify_layer_reports(record)
        self.verify_guard(record)
        resolved_path = record_path or Path(record.get("run_id", "."))
        self.verify_storage(resolved_path)
        try:
            dm_failures = self._ledger_factory(RunRecordStore.for_run(resolved_path).root()).check_record(record)
        except DmSourceViolationError as violation:
            raise ContractViolationError(str(violation)) from violation
        for failure in dm_failures:
            self.failures.append(f"dm source: {failure}")
        self.verify_final_acceptance(record, resolved_path)
        if chain_depth == 0:
            self.verify_chain(record)
        return self.failures
