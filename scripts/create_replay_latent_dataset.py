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

"""Replay latent dataset.json + sidecars: accepted manifest -> training entries (spec #13
sections 3.3 / 3.4, ticket T6 #22, stage C2).

For every accepted replay series this writes **two** dataset.json training entries (section
3.3 "双产"): the whole-brain source image under its v1 label (``mri_t1`` ... ``mri_mra``) and
its skull-stripped twin under the matching ``_skull_stripped`` label (29-33).  Both latents
are produced by the original, unmodified encoding pipeline from the same downloaded volume,
so the training set sees the two conditions the v1 model was trained on without downloading
anything twice.

The entries carry the same two fields the BRATS stage-1 dataset.json uses -- ``image``
(relative to the training env's ``data_base_dir``) and ``modality`` -- because the training
code turns them into the latent path (``.nii.gz`` -> ``_emb.nii.gz``) and the sidecar path
(``+ .json``) by convention alone.  The sidecars are written by ``scripts.latent_sidecars``,
so the replay sidecar is byte-for-byte the BRATS sidecar format of section 3.4.

Tiering: the replay subsets are nested (N=300 subset of N=500 subset of N=1000, per each
modality's seeded ordering), so the pipeline downloads and encodes the N=1000 superset once
and writes one dataset.json per tier from the accepted rows of that tier's manifest.  The
latents are shared across tiers; nothing is re-encoded.

Latent presence is checked before any file is written: the training run silently skips
entries whose latent is missing (``diff_model_train.load_filenames`` filters on
``os.path.exists``), which would quietly shrink the training set instead of failing, so a
gap must abort here.

The script runs in the two moments the pipeline actually has, because the encoder's input
and the training output have opposite preconditions:

- ``--stage encode`` (before encoding) writes ``replay_source_<tier>.json``: the source
  volumes of both halves of the dual derivation, latents not yet required.  This is the
  list ``scripts.diff_model_create_training_data`` consumes.
- ``--stage finalize`` (after encoding) requires every latent, writes the sidecars, and
  writes ``dataset_replay_rflow-mr-brain_<tier>.json`` for the merge generator.

Both path roots are the replay run's ``data`` directory: ``--data-base-dir`` is where the
downloaded volumes sit (the entries' ``image`` strings resolve under it) and
``--embedding-base-dir`` is where the encoder wrote the ``_emb.nii.gz`` twins.  The merge
generator (ticket T7 #23) rebuilds the merged training root as one directory holding
``brats2023-gli/`` and a ``mri/`` symlink onto the replay run's data tree, so a single
``data_base_dir`` / ``embedding_base_dir`` pair reaches every merged entry.

Usage::

    # before encoding: the encoder's input list
    uv run python -m scripts.create_replay_latent_dataset --stage encode \\
        --data-base-dir $RUN_ROOT/runs/replay-<yyyymmdd>/data \\
        --embedding-base-dir $RUN_ROOT/runs/replay-<yyyymmdd>/data \\
        --tier N1000=$RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000_final.csv \\
        --output-dir $RUN_ROOT/runs/replay-<yyyymmdd>

    # after encoding: sidecars + the per-tier training dataset.json
    uv run python -m scripts.create_replay_latent_dataset --stage finalize \\
        --data-base-dir $RUN_ROOT/runs/replay-<yyyymmdd>/data \\
        --embedding-base-dir $RUN_ROOT/runs/replay-<yyyymmdd>/data \\
        --tier N300=$RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N300_final.csv \\
        --tier N500=$RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N500_final.csv \\
        --tier N1000=$RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N1000_final.csv \\
        --output-dir $RUN_ROOT/runs/replay-<yyyymmdd>
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .download_replay_subset import MANIFEST_COLUMNS, ManifestCandidate
from .latent_sidecars import LatentEntry, LatentSidecarWriter

STAGES = ("encode", "finalize")
DATASET_FILENAME_TEMPLATE = "dataset_replay_rflow-mr-brain_{tier}.json"
SOURCE_FILENAME_TEMPLATE = "replay_source_{tier}.json"
TIER_ARGUMENT_PATTERN = re.compile(r"(?P<name>[A-Za-z0-9_-]+)=(?P<manifest>.+)")


@dataclass(frozen=True)
class ReplayTier:
    """One experiment point: a name (``N300``) and the manifest that defines it."""

    name: str
    manifest_path: Path

    @property
    def dataset_filename(self) -> str:
        """The training dataset.json for this tier.

        Deliberately *not* named ``dataset_rflow-mr-brain_<N>.json``: that is the merged
        BRATS+replay file the training env configs consume (ticket T7 #23), and two artifacts
        answering to one name would let a replay-only file stand in for a merged one.
        """
        return DATASET_FILENAME_TEMPLATE.format(tier=self.name)

    @property
    def source_filename(self) -> str:
        """The encoder's input list for this tier: source volumes only, no latents yet."""
        return SOURCE_FILENAME_TEMPLATE.format(tier=self.name)

    @classmethod
    def parse(cls, specification: str) -> "ReplayTier":
        """Parse one ``NAME=path/to/accepted.csv`` command-line specification."""
        match = TIER_ARGUMENT_PATTERN.fullmatch(specification)
        if match is None:
            raise ValueError(f"unparseable tier {specification!r} (expected NAME=manifest.csv)")
        return cls(name=match["name"], manifest_path=Path(match["manifest"]))


@dataclass(frozen=True)
class ReplayDatasetEntry:
    """The training entries one accepted replay series contributes: whole-brain plus its twin."""

    candidate: ManifestCandidate

    def training_entries(self) -> list[dict[str, str]]:
        """The two dataset.json records, in the order the dual derivation was decided (source first)."""
        return [
            {"image": self.candidate.image_path, "modality": self.candidate.variant.label},
            {"image": self.candidate.variant.skull_stripped_path, "modality": self.candidate.variant.skull_stripped_label},
        ]

    def latent_entries(self) -> list[LatentEntry]:
        """The same two entries as path adapters, for the shared sidecar writer."""
        return [LatentEntry(image=record["image"], modality=record["modality"]) for record in self.training_entries()]


class ReplayLatentDataset:
    """Assembles and writes one tier's dataset.json plus the sidecars every entry needs."""

    def __init__(self, tier: ReplayTier, data_base_dir: Path, sidecar_writer: LatentSidecarWriter) -> None:
        self._tier = tier
        self._data_base_dir = data_base_dir
        self._sidecar_writer = sidecar_writer

    def candidates(self) -> list[ManifestCandidate]:
        """The tier's accepted series, in manifest order."""
        with self._tier.manifest_path.open(newline="") as file:
            return [ManifestCandidate.from_row({name: row[name] for name in MANIFEST_COLUMNS}) for row in csv.DictReader(file)]

    def training_entries(self) -> list[dict[str, str]]:
        """Every training record for the tier: two per accepted series."""
        entries = []
        for candidate in self.candidates():
            entries.extend(ReplayDatasetEntry(candidate).training_entries())
        return entries

    def write_source_list(self, output_dir: Path) -> dict:
        """Write the encoder's input list for this tier; return a summary.

        This is the ``--stage encode`` artifact.  It holds the entries the tier needs encoded
        -- both halves of the dual derivation -- and checks that their source volumes are on
        disk.  It must come *before* encoding: the training dataset.json of
        ``--stage finalize`` refuses to be written while latents are missing, so without this
        list nothing could drive the encoder at all.

        Volumes that have not been downloaded yet are **left out and counted**: the encoder
        aborts on the first unreadable path (``diff_model_create_training_data`` probes the
        volume's shape before its per-file try/except), so a list naming a file that is not
        there cannot be handed to it at all.

        Already-encoded entries are left out too.  The encoder's shape probe runs *before* its
        "latent exists, skip" check, so a list that repeats finished work pays a full volume
        read per entry for nothing; naming only the outstanding work makes an incremental
        rerun cheap.  Rerunning after the download advances picks up whatever arrived, and
        ``--stage finalize`` still refuses to finish until every latent is on disk.
        """
        entries = self.training_entries()
        pending = sum(1 for record in entries if not (self._data_base_dir / record["image"]).is_file())
        outstanding = [
            record
            for record in entries
            if (self._data_base_dir / record["image"]).is_file()
            and not (self._sidecar_writer.base_dir / LatentEntry(record["image"], record["modality"]).embedding_relative_path).is_file()
        ]
        if not outstanding:
            # Distinguish "the tier is finished" from "everything that has arrived is encoded but
            # the download is still running".  The second looks identical from the file system and
            # is why the counts are reported either way: a caller driving the encoder in a loop
            # must not stop just because it caught up with a download that is still going.
            state = "complete" if pending == 0 else f"caught up, {pending} still downloading"
            print(f"{self._tier.name}: nothing outstanding ({state}); entries={len(entries)}")
            return {"tier": self._tier.name, "stage": "encode", "source_entries": 0, "not_yet_downloaded": pending, "output": None}
        output_path = self._write_json(output_dir / self._tier.source_filename, outstanding)
        print(f"{self._tier.name}: source_entries={len(outstanding)} not_yet_downloaded={pending} -> {output_path}")
        return {
            "tier": self._tier.name,
            "stage": "encode",
            "source_entries": len(outstanding),
            "not_yet_downloaded": pending,
            "output": str(output_path),
        }

    def write(self, output_dir: Path) -> dict:
        """``--stage finalize``: write the sidecars, then the training dataset.json; return a summary."""
        entries = self.training_entries()
        self._require_source_volumes(entries)
        latent_entries = [LatentEntry(image=record["image"], modality=record["modality"]) for record in entries]
        self._sidecar_writer.require_all(latent_entries, f"{self._tier.name} latents")
        sidecars = self._sidecar_writer.write(latent_entries)

        output_path = self._write_json(output_dir / self._tier.dataset_filename, entries)
        counts = self._counts(entries)
        print(f"{self._tier.name}: series={len(entries) // 2} training_entries={len(entries)} sidecars={sidecars} -> {output_path}")
        print(f"{self._tier.name}: per-label {counts}")
        return {"tier": self._tier.name, "training_entries": len(entries), "sidecars": sidecars, "output": str(output_path), "counts": counts}

    @staticmethod
    def _write_json(path: Path, entries: list[dict[str, str]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump({"training": entries}, file, indent=2)
            file.write("\n")
        return path

    def _require_source_volumes(self, entries: list[dict[str, str]]) -> None:
        """Fail loudly when the accepted manifest does not line up with the downloaded tree.

        A path typo here would otherwise surface as "training just sees fewer samples", which
        the training loop cannot distinguish from an intended subset.
        """
        missing = [record["image"] for record in entries if not (self._data_base_dir / record["image"]).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{self._tier.name}: {len(missing)} of {len(entries)} source volumes missing under {self._data_base_dir}, e.g.: {preview}"
            )

    @staticmethod
    def _counts(entries: list[dict[str, str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in entries:
            counts[record["modality"]] = counts.get(record["modality"], 0) + 1
        return dict(sorted(counts.items()))


class ReplayLatentDatasetSet:
    """Every tier of one replay run, sharing one encoded latent tree."""

    def __init__(self, tiers: list[ReplayTier], data_base_dir: Path, sidecar_writer: LatentSidecarWriter) -> None:
        self._datasets = [ReplayLatentDataset(tier, data_base_dir, sidecar_writer) for tier in tiers]

    def write_all(self, output_dir: Path, stage: str) -> list[dict]:
        """Run ``stage`` (``encode`` or ``finalize``) for every tier."""
        if stage == "encode":
            return [dataset.write_source_list(output_dir) for dataset in self._datasets]
        return [dataset.write(output_dir) for dataset in self._datasets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-base-dir", type=Path, required=True, help="root the entries' image paths are relative to")
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="root the latents live under")
    parser.add_argument(
        "--tier",
        action="append",
        required=True,
        metavar="NAME=ACCEPTED_CSV",
        help="one experiment point, e.g. N300=.../replay_manifest_N300_accepted.csv (repeatable)",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        required=True,
        help="'encode' writes the encoder's input lists (before encoding); 'finalize' writes sidecars + dataset.json (after)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="where the tier JSON files go")
    args = parser.parse_args()

    tiers = [ReplayTier.parse(specification) for specification in args.tier]
    for tier in tiers:
        if not tier.manifest_path.is_file():
            raise FileNotFoundError(f"tier {tier.name}: accepted manifest not found: {tier.manifest_path}")

    dataset_set = ReplayLatentDatasetSet(
        tiers=tiers,
        data_base_dir=args.data_base_dir,
        sidecar_writer=LatentSidecarWriter(args.embedding_base_dir),
    )
    summaries = dataset_set.write_all(args.output_dir, stage=args.stage)
    if args.stage == "encode":
        total = sum(summary["source_entries"] for summary in summaries)
        print(f"wrote {len(summaries)} encoder input lists, {total} source entries total")
    else:
        total = sum(summary["training_entries"] for summary in summaries)
        print(f"wrote {len(summaries)} dataset.json files, {total} training entries total")


if __name__ == "__main__":
    main()
