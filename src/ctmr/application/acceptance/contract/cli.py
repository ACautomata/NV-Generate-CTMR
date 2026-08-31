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

"""Run-contract verb entry: init / select / freeze / attach / conclude / verify.

Migrated verbatim from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141; the
legacy ``selftest`` verb died in the move -- its assertions live as pytest
functions under tests/application/acceptance/contract). Reached as
``ctmr accept contract <verb> ...``; the argparse surface and every exit code
(0 ok / 1 blocked-or-failed / 2 contract violation) are the legacy ones.

Since #271 (ADR-0019 §2) the composition root routes this face and injects
the ``ledger_factory`` -- the concrete DM-source ledger is wiring's choice,
not this module's; a bare ``python -m`` here is not an entry.
"""

import argparse
import sys
from pathlib import Path

from ctmr.application.acceptance.contract.artifacts import ArtifactFingerprinter, ManifestSides
from ctmr.application.acceptance.contract.conclude import FinalAcceptanceJudge
from ctmr.application.acceptance.contract.lifecycle import CandidateFreezer, ReportAttacher, RunInitializer, SelectionRecorder
from ctmr.application.acceptance.contract.record import (
    LIST_SIDES,
    P3_VARIANTS,
    PHASES,
    ContractViolationError,
    RunRecordStore,
)
from ctmr.application.acceptance.contract.registry import ATTACH_KINDS
from ctmr.application.acceptance.contract.verify import RunVerifier


def main(argv=None, *, ledger_factory):
    """Run one contract verb; returns the process exit code. The composition
    root injects the ``(record_root) -> DmSourceLedger`` port factory."""
    parser = argparse.ArgumentParser(
        prog="ctmr accept contract",
        description="Run-contract orchestration: open/select/freeze a candidate, attach evidence, conclude and verify.",
    )
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

    args = parser.parse_args(argv)
    fingerprinter = ArtifactFingerprinter()

    try:
        if args.handler == "init":
            sides = ManifestSides.from_path(args.manifest)
            path = RunInitializer(RunRecordStore(args.record_root), fingerprinter, sides, ledger_factory).init(
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
            entry, verdict_path = FinalAcceptanceJudge(RunRecordStore.for_run(args.run), fingerprinter, ledger_factory).conclude(args.run)
            layers = ", ".join(f"{name}={layer['verdict']}" for name, layer in entry["layers"].items())
            print(f"FINAL ACCEPTANCE {entry['verdict'].upper()} ({layers}) -> {verdict_path}")
            for reason in entry["blocked_reasons"]:
                print(f"  blocked: {reason}", file=sys.stderr)
            return 0 if entry["verdict"] == "pass" else 1
        paths = [Path(args.run)] if args.run else RunRecordStore(args.record_root).all_record_paths()
        failures = []
        for record_path in paths:
            record = RunRecordStore(record_path.parent.parent).load_by_path(record_path)
            run_failures = RunVerifier(fingerprinter, ledger_factory).verify(record, record_path=record_path)
            failures += [f"{record.get('run_id', record_path.parent.name)}: {f}" for f in run_failures]
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print(f"VERIFY PASS ({len(paths)} run{'s' if len(paths) > 1 else ''})")
        return 0
    except ContractViolationError as violation:
        print(f"CONTRACT VIOLATION: {violation}", file=sys.stderr)
        return 2
