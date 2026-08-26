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

"""ctmr unified CLI — command-family skeleton (issue #130 / ADR-0015 §3).

Registers the five command-family nouns — ``generate`` (alias ``gen``),
``measure``, ``accept``, ``data``, ``experiment``; their verbs migrate here
batch by batch (M4/M5). Until a family's verbs land, invoking it prints an
explicit not-yet-migrated notice (with its planned verbs) and exits 2 — no
traceback, nothing executed. Bare ``ctmr`` prints help and exits 0; unknown
families stay ordinary argparse usage errors (exit 2).

Stdlib-only: importable on any machine, no torch / monai (ADR-0013 §4) --
``pip install -e . --no-deps`` yields a working ``ctmr`` in any Python 3.11
environment; version locking stays with requirements.txt.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandFamily:
    """One registered CLI noun plus the verb set that migration will bring (ADR-0015 §3)."""

    name: str
    summary: str
    aliases: tuple[str, ...] = ()
    verbs: tuple[str, ...] = ()


FAMILIES = (
    CommandFamily(
        name="generate",
        summary="stage generation chains (train / dev-eval / generate / manifest)",
        aliases=("gen",),
        verbs=("train", "dev-eval", "generate", "manifest"),
    ),
    CommandFamily(name="measure", summary="instrument measurement runs (predict / calibration / FID)", verbs=("predict", "calibrate", "fid")),
    CommandFamily(
        name="accept",
        summary="acceptance-layer chains (quantitative / distribution / expert-review + run contract)",
        verbs=("quantitative", "distribution", "expert-review", "contract"),
    ),
    CommandFamily(name="data", summary="data prep / encode / download", verbs=("prep", "encode", "download")),
    CommandFamily(name="experiment", summary="experiment-record repository front (deploy/experiments, ADR-0015 decision 11)"),
)


class CtmrCli:
    """Console dispatcher over the five command families."""

    def __init__(self, families=FAMILIES):
        self._families = tuple(families)
        self._parser = self._build_parser()

    def _build_parser(self):
        parser = argparse.ArgumentParser(
            prog="ctmr", description="NV-Generate-CTMR unified CLI (ADR-0015 §3): five command families; verbs migrate in M4/M5"
        )
        subparsers = parser.add_subparsers(dest="family", metavar="<family>")
        for family in self._families:
            subparser = subparsers.add_parser(family.name, aliases=list(family.aliases), help=family.summary)
            planned = ", ".join(family.verbs) if family.verbs else "not fixed yet (ADR-0015 decision 11)"
            subparser.add_argument("verb", nargs="*", metavar="VERB", help=f"not migrated yet; planned: {planned}")
            # Carry the declaration itself so aliases resolve to the same family.
            subparser.set_defaults(_family=family)
        return parser

    def run(self, argv=None):
        """Parse argv and dispatch; returns the process exit code."""
        args = self._parser.parse_args(argv)
        family = getattr(args, "_family", None)
        if family is None:
            self._parser.print_help()
            return 0
        received = f" (received: {' '.join(args.verb)})" if args.verb else ""
        print(f"ctmr {family.name}{received}: registered but no verb is migrated yet (尚未迁移); nothing was executed.", file=sys.stderr)
        planned = ", ".join(family.verbs) if family.verbs else "not fixed yet"
        print(f"planned verbs (ADR-0015 §3): {planned}", file=sys.stderr)
        return 2


def main():
    """Console-script entry point: ``ctmr = ctmr.cli:main`` (the setuptools wrapper applies sys.exit)."""
    return CtmrCli().run()
