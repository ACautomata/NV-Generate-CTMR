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

Five command families pinned as the terminal CLI face. Families whose verbs
have landed route to their implementations; every other concrete invocation
answers a friendly "not migrated yet" message instead of an error traceback.

``ctmr measure predict`` (issue #140) is the canonical frozen-instrument
execution entry, replacing ``python -m ctmr.instrument.predict``. Its handler
is reached lazily (importlib on dispatch): torch / monai / nnunetv2 are only
loaded when the verb actually runs, so this module stays importable on any
machine without them -- pinned by the CLI purity gate in tests/test_cli_entry.
"""

from __future__ import annotations

import argparse
import importlib
import sys


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
            fam_parser = subparsers.add_parser(family, aliases=list(aliases), help=blurb)
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

    def run(self, argv=None):
        """Parse argv and dispatch; argparse errors/help exit inside parse_args."""
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv[:2] == ["measure", "predict"]:  # the frozen instrument door: everything after is nnUNetv2's business
            return self._invoke_verb("ctmr.infrastructure.nnunet_runner", "MeasurePredictVerb", argv[2:])
        args = self._parser.parse_args(argv)
        return args.run(args)


def main(argv=None):
    """Console-script target (``ctmr = ctmr.cli:main``); returns the process exit code."""
    return CtmrCli().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
