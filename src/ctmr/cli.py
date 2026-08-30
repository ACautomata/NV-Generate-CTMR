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

"""Unified ``ctmr`` console entry point (issues #130/#137/#138/#139/#140 / ADR-0015 §3; registry issue #228).

The five command families are pinned as the terminal CLI face. Every live verb
is one row of the single ``(family, case, verb)``→handler registry (``VERBS``):
the argparse spelling tree and the dispatch router both read that one table, so
adding a verb is adding one entry -- spelling, help text, and routing arrive
together. A routed verb is matched by longest fixed-prefix lookup on the raw
argv (:meth:`CtmrCli.route`) and its handler receives the remainder verbatim,
because each entry's flags belong to the entry's own argparse surface
(argparse REMAINDER would reject the unknown dash-tokens); the spelling tree
exists for ``--help`` and for invalid invocations only. ``peel_verb=False``
rows (the distribution/contract modules keep their own verb argparse) leave
the verb spelling in the handler argv.

``ctmr measure predict`` (issue #140) is the canonical frozen-instrument
execution entry (ADR-0009 decision 3) -- in the registry it is simply the
passthrough verb of the measure family: everything after the verb reaches the
nnUNetv2 predictor parser untouched. The superseded reverse shim retired with
issue #175.

``ctmr generate <case> train`` derives torchrun itself (no WORLD_SIZE -> spawn
the trainer child); everything else dispatches in-process. All verbs reach
their handlers lazily (importlib on dispatch): torch / monai / nnunetv2 are
only loaded when a verb actually runs, so this module stays importable on any
machine without them -- pinned by the CLI purity gate in tests/test_cli_entry.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Literal, NamedTuple


class VerbRoute(NamedTuple):
    """One verb row of the single CLI registry (issue #228).

    ``module`` is the handler module, imported lazily on dispatch and invoked
    as ``module.main(rest)``. ``help``/``description`` are the argparse
    spelling's texts. ``peel_verb=False`` marks a module that keeps its own
    verb argparse -- the verb spelling stays in the handler argv; the default
    peels the fixed prefix off and hands the flags only. ``kind="train"``
    routes through the torchrun derivation (ADR-0015 §3) instead of a direct
    ``main`` call.
    """

    module: str
    help: str | None = None
    description: str | None = None
    peel_verb: bool = True
    kind: Literal["main", "train"] = "main"


# The one (family, case, verb)→handler registry: both the argparse spelling
# tree (CtmrCli._build_parser) and the dispatch router (CtmrCli.route) read
# this single table -- adding a verb is one entry here.
VERBS: dict[tuple[str, ...], VerbRoute] = {
    # -- generate: the three use-case chains (tickets 08/09/10) --------------
    ("generate", "modality-label", "train"): VerbRoute("ctmr.application.generation.modality_label.train", kind="train"),
    ("generate", "modality-label", "dev-eval"): VerbRoute("ctmr.application.generation.modality_label.monitor"),
    ("generate", "mask", "train"): VerbRoute("ctmr.application.generation.mask.train", kind="train"),
    ("generate", "mask", "dev-eval"): VerbRoute("ctmr.application.generation.mask.monitor"),
    ("generate", "mask", "generate"): VerbRoute("ctmr.application.generation.mask.sample"),
    ("generate", "cross-modal", "train"): VerbRoute("ctmr.application.generation.cross_modal.train", kind="train"),
    ("generate", "cross-modal", "dev-eval"): VerbRoute("ctmr.application.generation.cross_modal.monitor"),
    ("generate", "cross-modal", "generate", "baseline"): VerbRoute("ctmr.application.generation.cross_modal.baseline"),
    ("generate", "cross-modal", "generate", "candidate"): VerbRoute("ctmr.application.generation.cross_modal.candidate"),
    # -- measure: the frozen-instrument door (issue #140 / ADR-0009 decision 3)
    ("measure", "predict"): VerbRoute(
        "ctmr.infrastructure.nnunet_runner",
        help="run the native nnUNetv2 predictor inside the weights_only allowlist scope (flags pass through to nnUNetv2)",
        description=(
            "Canonical frozen-instrument execution entry (ADR-0009 decision 3). "
            "Everything after 'measure predict' goes straight to the nnUNetv2 predictor parser, e.g. "
            "-i IN -o OUT -d DATASET -c CONFIG -p PLANS -tr TRAINER -f FOLD."
        ),
    ),
    # -- accept: the three acceptance layers plus the run contract (issue #141)
    ("accept", "quantitative", "evaluate"): VerbRoute(
        "ctmr.application.acceptance.quantitative.evaluate",
        help="write a candidate-bound L1 report from controlled evidence",
    ),
    ("accept", "distribution", "assemble"): VerbRoute("ctmr.application.acceptance.distribution.final_acceptance", peel_verb=False),
    ("accept", "distribution", "predict"): VerbRoute("ctmr.application.acceptance.distribution.final_acceptance", peel_verb=False),
    ("accept", "distribution", "evaluate"): VerbRoute("ctmr.application.acceptance.distribution.final_acceptance", peel_verb=False),
    ("accept", "distribution", "verify-frozen"): VerbRoute("ctmr.application.acceptance.distribution.final_acceptance", peel_verb=False),
    ("accept", "expert-review", "build-package"): VerbRoute(
        "ctmr.application.acceptance.expert_review.package",
        help="sample and blind the per-cell reviewer package",
    ),
    ("accept", "expert-review", "aggregate"): VerbRoute(
        "ctmr.application.acceptance.expert_review.aggregate",
        help="aggregate blinded judgments into the candidate-bound L3 report",
    ),
    ("accept", "contract", "init"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
    ("accept", "contract", "select"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
    ("accept", "contract", "freeze"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
    ("accept", "contract", "attach"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
    ("accept", "contract", "conclude"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
    ("accept", "contract", "verify"): VerbRoute("ctmr.application.acceptance.contract.cli", peel_verb=False),
}

# Spelling-tree help for the intermediate case/layer nodes. Not a routing
# surface -- the verb rows above are; this only texts the argparse tree.
CASE_BLURBS = {
    ("generate", "modality-label"): "modality-label-conditioned chain (train/dev-eval)",
    ("generate", "mask"): "mask-conditioned chain (train/dev-eval/generate)",
    ("generate", "cross-modal"): "image-conditioned cross-modality chain (train/dev-eval/generate)",
    ("accept", "quantitative"): "L1 quantitative evidence",
    ("accept", "distribution"): "L2 frozen-instrument judge chain",
    ("accept", "expert-review"): "L3 blinded-package and judgment aggregation",
    ("accept", "contract"): "run contract",
}


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
        # this map restores the canonical family name for messages and routing.
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
            fam_parser = subparsers.add_parser(
                family,
                aliases=list(aliases),
                # only the not-yet-migrated families advertise the migration banner
                help=blurb if family in ("generate", "measure", "accept") else f"{blurb} -- not migrated yet",
            )
            fam_parser.add_argument("rest", nargs="*", metavar="verb", help="verbs/flags of this family")
            fam_parser.set_defaults(run=self._not_migrated)
            if family == "measure":  # its predict verb landed with #140; unknown verbs are argparse errors
                self._build_measure_verbs(fam_parser)
            elif family == "accept":  # its four layer doors landed with #141; unknown layers are argparse errors
                self._build_accept_verbs(fam_parser)
        return parser

    def _build_measure_verbs(self, fam_parser):
        """The frozen-instrument spellings, read off VERBS (ADR-0009/#140). Only
        the spelling lives in argparse (help/usage/errors); flags after ``measure
        predict`` must reach the nnUNetv2's own parser verbatim, so routing
        happens in :meth:`run` via :meth:`route` -- argparse cannot hold a
        trailing star-slug of unknown dash-tokens."""
        verbs = fam_parser.add_subparsers(dest="verb", metavar="<verb>")
        for key, route in VERBS.items():
            if key[0] == "measure":
                verbs.add_parser(key[1], help=route.help, description=route.description)

    def _build_accept_verbs(self, fam_parser):
        """The acceptance-layer spellings, read off VERBS (issue #141). Only the
        spellings live in argparse (help/usage/errors); each layer's flags belong
        to the layer module's own parser, so routing happens in :meth:`run` via
        :meth:`route` -- both read the one table."""
        layers = fam_parser.add_subparsers(dest="layer", metavar="<layer>", required=True)
        seen = set()
        for key in VERBS:
            if key[0] != "accept" or key[1] in seen:
                continue
            seen.add(key[1])
            layer_parser = layers.add_parser(key[1], help=CASE_BLURBS[("accept", key[1])])
            verb_subparsers = layer_parser.add_subparsers(dest="verb", metavar="<verb>")
            for layer_key, route in VERBS.items():
                if layer_key[0] == "accept" and layer_key[1] == key[1]:
                    verb_subparsers.add_parser(layer_key[2], help=route.help)

    def _add_generate(self, subparsers, aliases, blurb):
        fam_parser = subparsers.add_parser("generate", aliases=list(aliases), help=blurb)
        cases = fam_parser.add_subparsers(dest="case", metavar="<case>", required=True)
        seen = set()
        for key in VERBS:
            if key[0] != "generate" or key[1] in seen:
                continue
            seen.add(key[1])
            # a bare case (no verb) answers a usage pointer, not a traceback
            case_parser = cases.add_parser(key[1], help=CASE_BLURBS[("generate", key[1])])
            case_parser.set_defaults(run=self._bare_generate_case)
            self._add_generate_verbs(case_parser, key[1])

    def _add_generate_verbs(self, case_parser, case):
        """The verbs of one generate case, read off VERBS; a variant row
        (``cross-modal generate baseline|candidate``) hangs off its verb."""
        verbs = case_parser.add_subparsers(dest="verb", metavar="<verb>")
        seen = set()
        for key in VERBS:
            if key[0] != "generate" or key[1] != case or key[2] in seen:
                continue
            seen.add(key[2])
            verb_parser = verbs.add_parser(key[2], help=VERBS[key].help)
            variant_rows = [(variant_key, VERBS[variant_key]) for variant_key in VERBS if variant_key[:3] == key[:3] and len(variant_key) == 4]
            if variant_rows:
                variants = verb_parser.add_subparsers(dest="variant", metavar="<variant>", required=True)
                for variant_key, variant_route in variant_rows:
                    variants.add_parser(variant_key[3], help=variant_route.help)

    def _bare_generate_case(self, args):
        """Answer a bare ``generate <case>`` (no verb) with a usage pointer, not a traceback."""
        family = self._alias_to_family.get(args.family, args.family)
        print(f"ctmr {family} {args.case}: needs a verb -- see `ctmr {family} {args.case} --help`.", file=sys.stderr)
        return 2

    def _not_migrated(self, args):
        """Answer any concrete call on a not-yet-migrated family with a pointer, not a traceback."""
        family = self._alias_to_family.get(args.family, args.family)
        verbs = " " + " ".join(args.rest) if getattr(args, "rest", None) else ""
        print(
            f"ctmr {family}{verbs}: not migrated yet -- this command family lands with the ADR-0015 migration batches (issue #129).",
            file=sys.stderr,
        )
        return 2

    def route(self, argv):
        """Longest-prefix lookup of the raw argv in the VERBS registry -- the
        dispatch half of the one table (the spelling tree is the other). Returns
        ``(verb route, handler argv)`` or ``None``; ``None`` falls back to the
        argparse tree (help / usage errors / not-migrated pointers). The handler
        argv excludes the fixed prefix; ``peel_verb=False`` rows keep their verb
        spelling for the module's own verb parser. This deliberately does not
        parse options: the handler argv belongs to the handler's own parser.
        """
        if not argv:
            return None
        head = self._alias_to_family.get(argv[0], argv[0])
        for size in (4, 3, 2):
            if size > len(argv):
                continue
            route = VERBS.get((head, *argv[1:size]))
            if route is not None:
                return route, list(argv[size if route.peel_verb else size - 1 :])
        return None

    def run(self, argv=None):
        """Parse argv and dispatch; argparse errors/help exit inside parse_args."""
        argv = list(sys.argv[1:] if argv is None else argv)
        hit = self.route(argv)
        if hit is not None:
            route, rest = hit
            if route.kind == "train":
                return self._run_train(route.module, rest)
            return importlib.import_module(route.module).main(rest)
        args = self._parser.parse_args(argv)
        return args.run(args)

    def _run_train(self, module, rest):
        """Dispatch a train verb; outside torchrun, derive the torchrun child."""
        from ctmr.application.generation.launcher import TorchrunLauncher, num_gpus_of

        if os.environ.get("WORLD_SIZE"):
            return importlib.import_module(module).main(rest)
        return TorchrunLauncher(module, rest, num_gpus_of(rest)).run()


def main(argv=None):
    """Console-script target (``ctmr = ctmr.cli:main``); returns the process exit code."""
    return CtmrCli().run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
