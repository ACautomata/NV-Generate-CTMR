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

"""The blinding-package renderer: deterministic sampling and opaque entry ids.

Migrated verbatim from ``brats_l3_blind_eval.py`` (retired scripts layer, git history) (#141). For the
frozen candidate's target modalities x sub-challenges it draws ``per-cell``
real + ``per-cell`` synthetic images, blinds them into opaque entry ids
(``L3-0001``...) in a seeded presentation order, and emits the reviewer-facing
``brats-l3-package/1`` (no source or subject id) plus the controlled
``brats-l3-blind-map/1`` that records the unblinded source, case and image
path. A cell with fewer than ``per-cell`` available images of either source
refuses to build (no silent partial sub-sampling). Stdlib only.

Reached as ``ctmr accept expert-review build-package ...``.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from ctmr.application.acceptance.expert_review.catalog import (
    BLIND_MAP_SCHEMA,
    MODALITIES,
    PACKAGE_SCHEMA,
    Catalog,
    L3Error,
)


class BlindSampler:
    """Deterministic, auditable sampling and blinding of the 200-entry package."""

    def __init__(self, seed, per_cell):
        self._seed = seed
        self._per_cell = per_cell

    def _draw(self, candidates, count, rng):
        ordered = sorted(candidates, key=lambda e: (e.challenge, e.case, e.target_modality, e.source, e.path))
        if len(ordered) < count:
            raise L3Error(f"cell has only {len(ordered)} available images of one source; need {count}")
        return rng.sample(ordered, count)

    def sample(self, catalog):
        """Draw per-cell real+synth entries, then blind them into a presentation order.

        Returns ``(package_entries, blind_map_entries)``, each a list of dicts in the
        blinded presentation order."""
        rng = random.Random(self._seed)
        drawn = []
        for challenge in catalog.challenges():
            for modality in MODALITIES:
                real, synth = catalog.cell(challenge, modality)
                drawn.extend(self._draw(real, self._per_cell, rng))
                drawn.extend(self._draw(synth, self._per_cell, rng))
        rng.shuffle(drawn)
        package, blind_map = [], []
        for index, entry in enumerate(drawn, start=1):
            entry_id = f"L3-{index:04d}"
            package.append({"entry_id": entry_id, "challenge": entry.challenge, "target_modality": entry.target_modality, "image_path": entry.path})
            blind_entry = {
                "entry_id": entry_id,
                "challenge": entry.challenge,
                "case": entry.case,
                "target_modality": entry.target_modality,
                "source": entry.source,
                "image_path": entry.path,
            }
            if entry.src_modality is not None:
                blind_entry["src_modality"] = entry.src_modality
            blind_map.append(blind_entry)
        return package, blind_map


class BlindPackageBuilder:
    """Assembles and writes the blinded package and the controlled blind map."""

    def __init__(self, seed, per_cell):
        self._seed = seed
        self._per_cell = per_cell

    def build(self, run_record, catalog):
        package, blind_map = BlindSampler(self._seed, self._per_cell).sample(catalog)
        sampling = {
            "seed": self._seed,
            "per_cell": {"real": self._per_cell, "synth": self._per_cell},
            "total_entries": len(package),
        }
        package_doc = {
            "schema": PACKAGE_SCHEMA,
            "phase": run_record["phase"],
            "run_id": run_record["run_id"],
            "sampling": sampling,
            "entries": package,
        }
        blind_map_doc = {
            "schema": BLIND_MAP_SCHEMA,
            "phase": run_record["phase"],
            "run_id": run_record["run_id"],
            "sampling": sampling,
            "entries": blind_map,
        }
        return package_doc, blind_map_doc


def main(argv=None):
    """Run the build-package verb (``ctmr accept expert-review build-package``)."""
    parser = argparse.ArgumentParser(
        prog="ctmr accept expert-review build-package",
        description="Sample and blind the per-cell reviewer package from a controlled catalog.",
    )
    parser.add_argument("--run", required=True, help="frozen run.json record (binding + freeze guard)")
    parser.add_argument("--catalog", required=True, help="catalog manifest (schema brats-l3-catalog/1)")
    parser.add_argument("--output", required=True, help="output directory for package.json + blind_map.json")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--per-cell", type=int, default=5, help="real and synthetic images drawn per challenge x modality cell")
    args = parser.parse_args(argv)
    try:
        run_record = json.loads(Path(args.run).read_text())
        package_doc, blind_map_doc = BlindPackageBuilder(args.seed, args.per_cell).build(run_record, Catalog.from_path(args.catalog))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "package.json").write_text(json.dumps(package_doc, indent=2) + "\n")
        (out / "blind_map.json").write_text(json.dumps(blind_map_doc, indent=2) + "\n")
        print(f"package + blind map -> {out}")
        return 0
    except L3Error as error:
        print(f"L3 ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
