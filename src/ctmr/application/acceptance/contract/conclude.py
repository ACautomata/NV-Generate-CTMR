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

"""Final-acceptance orchestration: evidence collection, judgement, immutability.

Migrated from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141). Collection and
persistence stay here; the cross-layer verdict itself is the pure domain
kernel ``ctmr.domain.acceptance.FinalAcceptanceRule`` -- the non-compensatory
L1∧L2∧L3 AND where any L2 ``undecided`` blocks exactly like a fail (issue
#58 / spec #51 decision 15). Missing or invalid layer attachments refuse the
judgement (no verdict record) so the run stays conclusible once its evidence
is completed; a written verdict record is immutable, and verify-time
reconciliation re-derives the AND through the same kernel. Only a P1 pass
registers the DM source for P2/P3.
"""

import json
from pathlib import Path

from ctmr.application.acceptance.contract.binding import STATUS_FROZEN
from ctmr.application.acceptance.contract.record import (
    FINAL_ACCEPTANCE_SCHEMA,
    STAGE0_BASELINE,
    ContractViolationError,
)
from ctmr.application.acceptance.contract.registry import LAYER_BY_KIND, LAYER_KINDS
from ctmr.domain.acceptance import FinalAcceptanceRule
from ctmr.domain.dmsource import DmSourceLedgerFactory, DmSourceViolationError


class FinalAcceptanceJudge:
    """Non-compensatory L1 ∧ L2 ∧ L3 final acceptance over a frozen candidate (issue #58).

    ``LAYER_KINDS`` comes from the acceptance-layer registry (single source);
    the AND itself comes from the domain kernel, shared with verify-time
    reconciliation so the two sides cannot drift. A P1 pass registers the DM
    source through the injected ``(record_root) -> DmSourceLedger`` port
    (ADR-0019 §2/§3); the concrete adapter is the composition root's choice.
    """

    def __init__(self, store, fingerprinter, ledger_factory: DmSourceLedgerFactory):
        self._store = store
        self._fingerprinter = fingerprinter
        self._ledger_factory = ledger_factory
        self._rule = FinalAcceptanceRule()

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
        verdict, blocked_reasons = self._rule.judge(
            {name: layer_data["verdict"] for name, layer_data in layers.items()},
            {
                name: LAYER_BY_KIND[LAYER_KINDS[name]].reasons_builder(layer_data["report"])
                for name, layer_data in layers.items()
                if layer_data["verdict"] != "pass"
            },
        )
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
            try:
                self._ledger_factory(self._store.root()).register(record, run_path)
            except DmSourceViolationError as violation:
                raise ContractViolationError(str(violation)) from violation
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return entry, verdict_path

    def _collect_layers(self, record):
        """One formal attachment per layer, each revalidated against the frozen run."""
        layers = {}
        problems = []
        for layer_name, kind in LAYER_KINDS.items():
            layer = LAYER_BY_KIND[kind]
            attachments = [a for a in record.get("attachments", []) if a.get("kind") == kind]
            if len(attachments) > 1:
                problems.append(f"run has more than one formal {kind} attachment")
                continue
            if not attachments:
                problems.append(f"final acceptance requires a formal {kind} attachment (candidate freeze is not enough)")
                continue
            failures = layer.validator_factory().validate(record, attachments[0]["path"])
            if failures:
                problems.append(f"invalid {kind}: " + "; ".join(failures))
                continue
            report = json.loads(Path(attachments[0]["path"]).read_text())
            layers[layer_name] = {
                "attachment": {"path": attachments[0]["path"], "sha256": attachments[0]["sha256"]},
                "verdict": layer.verdict_reader(report),
                "report": report,
            }
        return layers, problems

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
        # Re-derive the AND through the shared kernel: the recorded verdict must
        # follow from its layer verdicts (an edited/flipped verdict file fails
        # verification even with intact attachments).
        expected = self._rule.expected_overall({name: layers[name].get("verdict") for name in LAYER_KINDS})
        if expected == "pass" and verdict_record.get("blocked_reasons") != []:
            expected = "blocked"
        if verdict_record.get("verdict") != expected:
            problems.append("verdict record disagrees with the non-compensatory AND of its layer verdicts")
        if (verdict_record.get("verdict") == "blocked") != bool(verdict_record.get("blocked_reasons")):
            problems.append("verdict record blocked state and blocked_reasons are inconsistent")
        return problems
