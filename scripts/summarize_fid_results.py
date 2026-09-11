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

"""FID freeze summarizer: per-run result jsons -> the frozen json/csv pair (ticket T8, issue #24).

The acceptance protocol (spec #13 section 5.5) freezes the reference numbers once
and reuses them for every comparison -- never recomputed. Each
``compute_fid_2-5d_ct`` invocation with ``--result_json`` writes one self-describing
record; this script aggregates those records into the frozen artifacts:

- ``baseline_gen_real``: the pretrained baseline, one entry per (label, seed),
  tags ``label<L>_seed<S>`` -- the FID_pre side of the forgetting check (section 5.1).
- ``real_real_floor``: the BRATS real-real halves comparison, tags ``floor_<label>``
  -- the sanity magnitude anchor (section 5.3).

Tag conventions are strict: anything that matches neither pattern is refused, so a
mis-tagged run cannot silently land in the wrong table.

Usage::

    python -m scripts.summarize_fid_results \
        --results runs/t8-freeze-20260911/fid/*.json \
        --output-json runs/t8-freeze-20260911/fid_freeze.json \
        --output-csv runs/t8-freeze-20260911/fid_freeze.csv
"""

import argparse
import csv
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")
FidResult = compute_fid.FidResult

BASELINE_PATTERN = re.compile(r"^label(?P<label>\d+)_seed(?P<seed>\d+)$")
FLOOR_PATTERN = re.compile(r"^floor_(?P<label>.+)$")
CSV_FIELDS = ("kind", "tag", "fid_xy", "fid_yz", "fid_zx", "fid_avg")


class ComparisonTag:
    """The strict tag vocabulary separating baseline gen-real records from floor records."""

    @staticmethod
    def classify(tag: str) -> tuple[str, str, str]:
        """(kind, primary key, secondary key) for one tag; refuses anything outside the vocabulary."""
        baseline = BASELINE_PATTERN.match(tag)
        if baseline:
            return "baseline_gen_real", baseline.group("label"), baseline.group("seed")
        floor = FLOOR_PATTERN.match(tag)
        if floor:
            return "real_real_floor", floor.group("label"), ""
        raise ValueError(f"unrecognized comparison tag {tag!r} (expected 'label<L>_seed<S>' or 'floor_<label>')")


@dataclass(frozen=True)
class FidSummary:
    """The frozen aggregation: baseline table + floor table, renderable as json and csv."""

    records: list[FidResult]
    sources: list[str]

    def summarize(self) -> dict:
        """Nest records by their tag classification; duplicate tags are refused."""
        rows = self._sorted_rows()
        baseline: dict[str, dict[str, dict]] = {}
        floor: dict[str, dict] = {}
        for kind, tag, _primary, _secondary, values in rows:
            if kind == "baseline_gen_real":
                baseline.setdefault(_primary, {})[_secondary] = values
            else:
                floor[_primary] = values
        return {"generated_from": self.sources, "baseline_gen_real": baseline, "real_real_floor": floor}

    def save(self, json_path: Path, csv_path: Path) -> None:
        """Write the json freeze and its flat csv rendering."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as file:
            json.dump(self.summarize(), file, indent=2)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for kind, tag, _primary, _secondary, values in self._sorted_rows():
                writer.writerow({"kind": kind, "tag": tag, **values})

    def _sorted_rows(self) -> list[tuple[str, str, str, str, dict]]:
        """(kind, tag, primary, secondary, fid values) per record, tag-sorted, duplicates refused."""
        if not self.records:
            raise ValueError("no records to summarize")
        rows = []
        seen: set[str] = set()
        for record in sorted(self.records, key=lambda item: item.comparison_tag):
            if record.comparison_tag in seen:
                raise ValueError(f"duplicate comparison tag across result files: {record.comparison_tag}")
            seen.add(record.comparison_tag)
            kind, primary, secondary = ComparisonTag.classify(record.comparison_tag)
            values = {"fid_xy": record.fid_xy, "fid_yz": record.fid_yz, "fid_zx": record.fid_zx, "fid_avg": record.fid_avg}
            rows.append((kind, record.comparison_tag, primary, secondary, values))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", type=Path, required=True, help="FidResult json files written by compute_fid_2-5d_ct")
    parser.add_argument("--output-json", type=Path, required=True, help="path of the frozen summary json")
    parser.add_argument("--output-csv", type=Path, required=True, help="path of the frozen summary csv")
    args = parser.parse_args()

    records = [FidResult.load(path) for path in sorted(args.results)]
    sources = [path.name for path in sorted(args.results)]
    FidSummary(records, sources=sources).save(args.output_json, args.output_csv)
    print(f"summarized {len(records)} FID records -> {args.output_json}, {args.output_csv}")


if __name__ == "__main__":
    main()
