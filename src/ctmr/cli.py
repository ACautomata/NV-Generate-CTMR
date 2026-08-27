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

"""Unified ``ctmr`` console entry point (issue #130 / ADR-0015 §3).

The five command families are pinned as the terminal CLI face; real verbs land
family by family with the migration batches (M4 application layer). The
``generate`` family is live for ``cross-modal`` (ticket 08): the verb grammar
is deliberately thin -- each verb forwards its remaining argv verbatim to the
family module entry (``train`` / ``dev-eval`` / ``generate baseline|candidate``),
whose full argparse surface (and the argv↔namespace equivalence gate) lives in
the module itself. ``ctmr generate cross-modal train`` derives torchrun itself
(no WORLD_SIZE -> spawn the trainer child); everything else dispatches in-process.

Because the entry argv is an arbitrary argparse surface of its own (``-e``,
``-g``, ...), the CLI cannot pre-parse it -- ``run`` peels the fixed verb prefix
off the raw argv and hands the remainder verbatim to the entry (argparse
REMAINDER would reject the unknown options). The parser tree exists for
``--help`` and for invalid invocations.

Stdlib-only like the rest of src/ctmr: importable on any machine without
torch / monai (ADR-0013 §4) -- the heavy modules are imported lazily inside
the handlers.
"""

from __future__ import annotations

import argparse
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
            fam_parser = subparsers.add_parser(family, aliases=list(aliases), help=f"{blurb} -- not migrated yet")
            fam_parser.add_argument("rest", nargs="*", metavar="verb", help="future verbs/flags of this family")
            fam_parser.set_defaults(run=self._not_migrated)
        return parser

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
        rest = getattr(args, "rest", ())
        verbs = " " + " ".join(rest) if rest else ""
        print(
            f"ctmr {family}{verbs}: not migrated yet -- " "this command family lands with the ADR-0015 migration batches (issue #129).",
            file=sys.stderr,
        )
        return 2

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
