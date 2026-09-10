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

"""Top-up: refill post-spine-filter shortages along each modality's seeded ordering
(spec #13 section 3.3 "缺额补抽", ticket T6 #22).

The spine filter runs after download and rejects a few series (the manifest carries no
body-part label, so spine acquisitions can only be caught on the voxels).  A rejected
series must not silently shrink a tier, and it must not be replaced by an arbitrary
other one either: ``scripts.create_replay_manifest`` guarantees a capped draw is the
prefix of each modality's seeded shuffle, so the replacement is deterministic --
skip the rejects and continue down the same ordering.

This module does exactly that, per tier:

1. Take the tier's **full ordering** (``--n-per-label`` at or above the tier's cap, so
   every tier's roster is a prefix of the same sequence) and drop the rejected rows.
2. Split what remains by modality.
3. Take the first ``min(cap, available)`` of each modality's remainder, where ``cap`` is
   the layered-cap rule of the tier (head labels N, T2w ``min(N, 669)``, MRA all).

Nesting is preserved by construction: tiers of increasing N consume the same ordering and
the same reject set, so a smaller tier's rows are a prefix of a larger tier's.  Fewer than
``cap`` survivors means the modality's candidate pool is exhausted -- the shortfall is
reported, never padded.

**The ordering must carry headroom.**  ``--ordering`` is what the generator produced at some
``--n-per-label``; if that cap equals the largest tier's cap, the ordering ends exactly where
the tier does and a rejected series has nothing to be replaced by -- the tier comes up short
instead of being refilled.  Generate the ordering above the tiers it feeds, e.g. for the
N=1000 tier::

    python -m scripts.create_replay_manifest ... --n-per-label 1100 \\
        --output .../manifests/replay_ordering_N1100.csv

Head-label availability in the Train split is 48k-75k subjects, so the extra rows cost
nothing but a longer CSV.

**T2w and MRA cannot be refilled at all, whatever N.**  Raising ``--n-per-label`` does not
help them: T2w is capped at ``min(N, 669)`` by the layered-cap rule, so the ordering stops at
669 no matter how large N is, and MRA is uncapped -- it takes the whole 141-subject Train
pool, so there is nothing beyond it.  A rejected T2w or MRA series therefore leaves its tier
genuinely one short, and the roster is written short with a warning rather than padded.  This
is the "prefer false rejects over false accepts" bias (spec section 3.3) resolving in the
direction it was told to.

Usage::

    uv run python -m scripts.refill_replay_manifest \\
        --ordering    $RUN_ROOT/datasets/MR-RATE/manifests/replay_ordering_N1100.csv \\
        --rejected    $RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000_rejected.csv \\
        --tier N300=300 --tier N500=500 --tier N1000=1000 \\
        --output-dir  $RUN_ROOT/datasets/MR-RATE/manifests

Each tier writes ``replay_manifest_<N>_final.csv``: the same manifest columns as the
input, ready for ``scripts.create_replay_latent_dataset`` and the merge generator.
"""

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .create_replay_manifest import MANIFEST_COLUMNS, CapPolicy
from .download_replay_subset import VerdictLog

TIER_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9_-]+)=(?P<n>\d+)")
OUTPUT_TEMPLATE = "replay_manifest_{tier}_final.csv"


@dataclass(frozen=True)
class TierTarget:
    """One experiment point: the tier's name and its N (the layered-cap input)."""

    name: str
    n_per_label: int

    @property
    def output_filename(self) -> str:
        return OUTPUT_TEMPLATE.format(tier=self.name)

    @property
    def policy(self) -> CapPolicy:
        """The layered-cap rule (section 3.3), owned by the generator so the two cannot drift."""
        return CapPolicy(n_per_label=self.n_per_label)

    @classmethod
    def parse(cls, specification: str) -> "TierTarget":
        match = TIER_PATTERN.fullmatch(specification)
        if match is None:
            raise ValueError(f"unparseable tier {specification!r} (expected NAME=N)")
        return cls(name=match["name"], n_per_label=int(match["n"]))


class ModalityOrdering:
    """One modality's seeded candidate ordering with the rejected series taken out."""

    def __init__(self, rows: list[dict[str, str]], rejected: set[tuple[str, str]]) -> None:
        self._rows = [row for row in rows if (row["study_uid"], row["series_id"]) not in rejected]

    @property
    def available(self) -> int:
        return len(self._rows)

    def take(self, cap: int | None) -> list[dict[str, str]]:
        """The first ``cap`` survivors -- the top-up continuation of the seeded ordering.

        ``cap=None`` (MRA) keeps everything available.
        """
        return list(self._rows if cap is None else self._rows[:cap])


class ReplayOrderingIndex:
    """The full ordering of one manifest, viewable per modality, minus the rejected series."""

    def __init__(self, ordering_path: Path, rejected_path: Path | None) -> None:
        self._rows = self._read(ordering_path, MANIFEST_COLUMNS)
        self._rejected = self._reject_keys(rejected_path)

    @property
    def rows(self) -> list[dict[str, str]]:
        return list(self._rows)

    @property
    def rejected(self) -> set[tuple[str, str]]:
        return set(self._rejected)

    def modality(self, modality: str) -> ModalityOrdering:
        """The ordering restricted to one modality, with rejected series removed."""
        return ModalityOrdering([row for row in self._rows if row["modality"] == modality], self._rejected)

    @staticmethod
    def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
        with path.open(newline="") as file:
            return [{name: row[name] for name in columns} for row in csv.DictReader(file)]

    @staticmethod
    def _reject_keys(rejected_path: Path | None) -> set[tuple[str, str]]:
        if rejected_path is None or not rejected_path.is_file():
            return set()
        return {(row["study_uid"], row["series_id"]) for row in VerdictLog(rejected_path).read_rows()}


class TierRoster:
    """Assembles one tier's final roster and writes it, next to the shortages it could not fill."""

    def __init__(self, target: TierTarget, index: ReplayOrderingIndex) -> None:
        self._target = target
        self._index = index

    def rows(self) -> list[dict[str, str]]:
        """This tier's rows: each modality's first ``cap`` survivors of the ordering."""
        selected = []
        for modality in sorted({row["modality"] for row in self._index.rows}):
            ordering = self._index.modality(modality)
            selected.extend(ordering.take(self._target.policy.cap_for(modality)))
        return selected

    def shortages(self) -> dict[str, int]:
        """``modality -> how many rows short of cap``; uncapped modalities are never short.

        Only a capped modality can fall short (it wanted N subjects and the pool ran out).
        MRA takes whatever exists, so reporting it as short would be noise.
        """
        short = {}
        for modality in sorted({row["modality"] for row in self._index.rows}):
            cap = self._target.policy.cap_for(modality)
            if cap is None:
                continue
            deficit = cap - self._index.modality(modality).available
            if deficit > 0:
                short[modality] = deficit
        return short

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row["label"] for row in self.rows()).items()))

    def write(self, output_dir: Path) -> dict:
        """Write the tier's roster CSV; return a summary with counts and shortages."""
        rows = self.rows()
        destination = output_dir / self._target.output_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        shortages = self.shortages()
        print(f"{self._target.name}: rows={len(rows)} per-label={self.counts()} -> {destination}")
        if shortages:
            print(f"{self._target.name}: WARNING short of cap for {shortages} (candidate pool exhausted)")
        return {"tier": self._target.name, "rows": len(rows), "counts": self.counts(), "shortages": shortages, "output": str(destination)}


class RefilledReplayManifests:
    """Every tier of one experiment matrix, refilled from one ordering and one reject set."""

    def __init__(self, ordering_path: Path, rejected_path: Path | None) -> None:
        self._index = ReplayOrderingIndex(ordering_path, rejected_path)

    @property
    def rejected(self) -> set[tuple[str, str]]:
        return self._index.rejected

    def write_all(self, targets: list[TierTarget], output_dir: Path) -> list[dict]:
        return [TierRoster(target, self._index).write(output_dir) for target in targets]

    def accepted_union(self, targets: list[TierTarget]) -> dict[tuple[str, str], dict[str, str]]:
        """Every row any tier needs, keyed by ``(study_uid, series_id)`` -- the download top-up list.

        The tiers nest, so this is the largest tier's rows as long as its N is the maximum;
        computing the union keeps the answer correct even for a non-monotone matrix.
        """
        rows = {}
        for target in targets:
            for row in TierRoster(target, self._index).rows():
                rows[(row["study_uid"], row["series_id"])] = row
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ordering", type=Path, required=True, help="the full-ordering manifest (largest cap)")
    parser.add_argument("--rejected", type=Path, default=None, help="the spine filter's rejected-series manifest")
    parser.add_argument("--tier", action="append", required=True, metavar="NAME=N", help="one experiment point, e.g. N300=300 (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True, help="where the *_final.csv rosters go")
    parser.add_argument("--topup-manifest", type=Path, default=None, help="also write the union of rows every tier needs")
    args = parser.parse_args()

    targets = [TierTarget.parse(specification) for specification in args.tier]
    refilled = RefilledReplayManifests(args.ordering, args.rejected)
    summaries = refilled.write_all(targets, args.output_dir)

    if args.topup_manifest is not None:
        union = refilled.accepted_union(targets)
        with args.topup_manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(union.values())
        print(f"top-up manifest: {len(union)} rows -> {args.topup_manifest}")

    print(f"rejected={len(refilled.rejected)} wrote {len(summaries)} tiers to {args.output_dir}")


if __name__ == "__main__":
    main()
