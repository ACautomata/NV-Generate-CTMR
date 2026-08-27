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

"""Unified ``ctmr`` console entry point (issues #130/#137/#140 / ADR-0015 §3).

Five command families pinned as the terminal CLI face; verbs land family by
family with the migration batches. Two doors are live:

``ctmr measure predict`` (issue #140) is the canonical frozen-instrument
execution entry, replacing ``python -m ctmr.instrument.predict``.

``ctmr generate cross-modal`` (ticket 08) routes train/dev-eval/generate to the
cross-modal family module; the verb grammar is deliberately thin -- each verb
forwards its remaining argv verbatim to the family module entry
(``train`` / ``dev-eval`` / ``generate baseline|candidate``), whose full
argparse surface (and the argv↔namespace equivalence gate) lives in the module
itself. Because that entry argv is an arbitrary argparse surface of its own,
the CLI cannot pre-parse it -- ``run`` peels the fixed verb prefix off the raw
argv and hands the remainder verbatim to the entry (argparse REMAINDER would
reject the unknown options). The parser tree exists for ``--help`` and for
invalid invocations.

Both doors reach their handlers lazily (importlib on dispatch / in-function
imports): torch / monai / nnunetv2 are only loaded when a verb actually runs,
so this module stays importable on any machine without them -- pinned by the
CLI purity gate in tests/test_cli_entry.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

TRAIN_MODULE = "ctmr.application.generation.cross_modal.train"

_VARIANTS = ("baseline", "candidate")


class CtmrCli:
    """The unified command-line application (ADR-0015 §3)."""

    # Command families in ADR order; generate rides its ``gen`` alias.
    FAMILIES = (
        ("generate", ("gen",), "use-case chains modality-label|mask|cross-modal (train/dev-eval/generate/manifest)"),
        ("measure", (), "instrument-side prediction/calibration/FID"),
        ("accept", (), "quantitative/distribution/expert-review chains plus contract orchestration"),
        ("data", (), "data preparation/encoding/download tooling"),
        ("experiment", (), "experiment-record repository"),
    )

    def __init__(self):
        # argparse stores the invoked spelling ("gen" or "generate") in dest;
        # this map restores the canonical family name for messages.
        self._alias_to_family = {alias: family for family, aliases, _ in self.FAMILIES for alias in aliases}
        self._parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            prog="ctmr",
            description="Unified CLI for NV-Generate-CTMR (ADR-0015); verb families migrate in batch under issue #129.",
        )
        subparsers = parser.add_subparsers(dest="family", metavar="<family>", required=True)
        for family, aliases, blurb in self.FAMILIES:
            if family == "generate":
                self._add_generate(subparsers, aliases, blurb)
                continue
            if family == "measure":  # its predict verb landed with #140; unknown verbs are argparse errors
                fam_parser = subparsers.add_parser(family, aliases=list(aliases), help=blurb)
            else:
                fam_parser = subparsers.add_parser(family, aliases=list(aliases), help=f"{blurb} -- not migrated yet")
            fam_parser.add_argument("rest", nargs="*", metavar="verb", help="verbs/flags of this family")
            fam_parser.set_defaults(run=self._not_migrated)
            verb_parsers = {
                "measure": self._build_measure_verbs,
            }.get(family)
            if verb_parsers is not None:
                verb_parsers(fam_parser)
        return parser

    @staticmethod
    def _build_measure_verbs(fam_parser):
        """Instrument-side verbs (ADR-0009/#140). Only the spelling lives in
        argparse (help/usage/errors); flags after ``measure predict`` must reach
        nnUNetv2's own parser verbatim, so routing happens in :meth:`run` --
        argparse cannot hold a trailing star-slug of unknown dash-tokens."""
        measure_subparsers = fam_parser.add_subparsers(dest="verb", metavar="<verb>")
        measure_subparsers.add_parser(
            "predict",
            help="run the native nnUNetv2 predictor inside the weights_only allowlist scope (flags pass through to nnUNetv2)",
            description=(
                "Canonical frozen-instrument execution entry (ADR-0009 decision 3). "
                "Everything after 'measure predict' goes straight to nnUNetv2's predictor parser, e.g. "
                "-i IN -o OUT -d DATASET -c CONFIG -p PLANS -tr TRAINER -f FOLD."
            ),
        )

    def _add_generate(self, subparsers, aliases, blurb):
        fam_parser = subparsers.add_parser("generate", aliases=list(aliases), help=blurb)
        cases = fam_parser.add_subparsers(dest="case", metavar="<case>", required=True)
        for case in ("modality-label", "mask"):
            case_parser = cases.add_parser(case, help=f"{case} chain -- not migrated yet")
            case_parser.add_argument("rest", nargs="*", metavar="verb", help="future verbs/flags of this case")
            case_parser.set_defaults(run=self._not_migrated)
        cases.add_parser("cross-modal", help="image-conditioned cross-modality chain (train/dev-eval/generate)")

    def _not_migrated(self, args):
        """Answer any concrete call on a not-yet-migrated family with a pointer, not a traceback."""
        family = self._alias_to_family.get(args.family, args.family)
        verbs = " " + " ".join(args.rest) if getattr(args, "rest", None) else ""
        print(
            f"ctmr {family}{verbs}: not migrated yet -- this command family lands with the ADR-0015 migration batches (issue #129).",
            file=sys.stderr,
        )
        return 2

    def _invoke_verb(self, handler_module, handler_name, pass_through=None):
        """Dispatch a routed verb; imports its handler module lazily so the CLI face stays stdlib-only."""
        module = importlib.import_module(handler_module)
        return getattr(module, handler_name)().run(list(pass_through or []))

    @staticmethod
    def _peel_generate(argv):
        """Peel the fixed ``generate cross-modal <verb> [variant]`` prefix; (handler, rest) or None.

        ``None`` means the argv does not fit a migrated verb -- fall back to the
        parser tree (help / argparse error). This deliberately does not parse
        options: the entry argv belongs to the entry's own parser.
        """
        if len(argv) < 2:
            return None
        if argv[0] not in ("generate", "gen"):
            return None
        case = argv[1]
        if case in ("modality-label", "mask"):
            args = argparse.Namespace(family=argv[0], case=case, rest=argv[1:])
            return (CtmrCli._not_migrated, args)
        if case != "cross-modal":
            return None
        if len(argv) < 3:
            return None
        verb = argv[2]
        rest = argv[3:]
        if verb == "train":
            return (CtmrCli._run_cross_modal_train, rest)
        if verb == "dev-eval":
            return (CtmrCli._run_cross_modal_dev_eval, rest)
        if verb == "generate":
            if len(rest) >= 1 and rest[0] in _VARIANTS:
                return (CtmrCli._run_cross_modal_generate, rest[0], rest[1:])
            return None
        return None

    def run(self, argv=None):
        """Parse argv and dispatch; argparse errors/help exit inside parse_args."""
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv[:2] == ["measure", "predict"]:  # the frozen instrument door: everything after is nnUNetv2's business
            return self._invoke_verb("ctmr.infrastructure.nnunet_runner", "MeasurePredictVerb", argv[2:])
        peeled = self._peel_generate(argv)
        if peeled is not None:
            handler, *payload = peeled
            return handler(self, *payload)
        args = self._parser.parse_args(argv)
        return args.run(args)

    def _run_cross_modal_train(self, rest):
        """Dispatch the finetune entry; outside torchrun, derive the torchrun child."""
        from ctmr.application.generation.cross_modal import train
        from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

        if os.environ.get("WORLD_SIZE"):
            return train.main(rest)
        return TorchrunLauncher(TRAIN_MODULE, rest, num_gpus_of(rest)).run()

    def _run_cross_modal_dev_eval(self, rest):
        from ctmr.application.generation.cross_modal import monitor

        return monitor.main(rest)

    def _run_cross_modal_generate(self, variant, rest):
        from ctmr.application.generation.cross_modal import baseline, candidate

        entry = baseline if variant == "baseline" else candidate
        return entry.main(rest)


def main(argv=None):
    """Console-script target (``ctmr = ctmr.cli:main``); returns the process exit code."""
    return CtmrCli().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
