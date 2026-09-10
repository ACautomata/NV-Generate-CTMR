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

"""Replay manifest generator: MR-RATE metadata CSVs -> stratified replay subset roster
(spec #13 sections 3.3 / 3.4, ticket T2 #18, new-code list item 1).

Samples a subject (``patient_uid``) level replay subset from the MR-RATE Train split and
writes a series-level manifest CSV -- the frozen, seed-reproducible roster that the
replay download/encoding ticket and the dataset.json merge generator both consume.

Selection rules (spec #13 section 3.3, decisions D1/D3 of issue #7):

- **Train split only**: candidates are metadata rows whose ``patient_uid`` is a
  ``train`` patient in ``splits.csv``; the manifest ``split`` column records ``Train``.
- **Subject-level sampling**: one series per (subject, modality) -- when a subject has
  several series of a modality, the one with the smallest ``SeriesNumber`` wins (the
  official pre-contrast heuristic).  Sampling by subject avoids the bias towards
  series-rich subjects of scan-level sampling.
- **Layered caps** (replay exists to hold the latent distribution against forgetting,
  not to rebalance v1's historical imbalance): head labels T1w/FLAIR/SWI sample N
  subjects each; T2w samples ``min(N, 669)`` (669 = v1's usable brain-T2w source-volume
  count, ``data/README.md`` section 3.4); MRA takes every available subject (141 in the
  current Train split -- the spec's "all 157" predates the split-restricted count).
- **Data-quality exclusion**: rows flagged ``is_derived`` / ``is_localizer`` /
  ``is_subtraction`` are never candidates (scouts, post-processed and subtraction
  volumes are unusable for replay).  Availability counts quoted anywhere are
  post-exclusion, so they sit slightly below the spec section 3.3 anchors.
- **Fixed seed**: each modality draws with ``random.Random`` seeded from
  ``sha256(f"{seed}:{modality}")``, so the manifest reproduces line by line on rerun.
  Selection is the first ``cap`` entries of a seeded shuffle of the sorted candidate
  subjects, and a capped draw is always the prefix of the full ordering -- the download
  ticket can refill post-spine-filter shortages by continuing down the same ordering.

Manifest columns (spec section 3.4): ``patient_uid, study_uid, series_id, modality,
label, split, image_path``.  ``modality`` is the lowercase MR-RATE modality string,
``label`` the whole-brain v1 label (``mri_t1`` ...); the skull-stripped twin of every
row (``mri_*_skull_stripped``, dual derivation) is derived downstream by the merge
generator.  ``image_path`` is relative to the MR-RATE data root and follows the
official unzip layout (zip root == study directory):
``mri/<batch>/<study_uid>/img/<study_uid>_<series_id>.nii.gz``; the HD-BET brain mask
sits at the mirrored ``seg/`` path (``MrRateSeries.mask_path``).  Consumers rebase the
path onto their ``data_base_dir``.

Usage::

    python -m scripts.create_replay_manifest \
        --splits-csv data/MR-RATE/splits.csv \
        --metadata-dir data/MR-RATE/metadata \
        --n-per-label 300 \
        --output data/MR-RATE/manifests/replay_manifest_N300.csv

Spine filtering is deliberately NOT applied here (the metadata carries no body-part
label); it happens after download via ``scripts.spine_filter``, followed by a top-up
re-run of this script.
"""

import argparse
import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

METADATA_GLOB = "batch*_metadata.csv"
TRAIN_SPLIT = "train"
MANIFEST_SPLIT_VALUE = "Train"
MODALITIES = ("t1w", "t2w", "flair", "swi", "mra")
MODALITY_TO_LABEL = {"t1w": "mri_t1", "t2w": "mri_t2", "flair": "mri_flair", "swi": "mri_swi", "mra": "mri_mra"}
EXCLUDED_FLAGS = ("is_derived", "is_localizer", "is_subtraction")
T2W_CAP = 669
MANIFEST_COLUMNS = ("patient_uid", "study_uid", "series_id", "modality", "label", "split", "image_path")
DEFAULT_SEED = 42

# Metadata columns this script reads; the CSVs carry ~100 columns, everything else is ignored.
SERIES_COLUMNS = ("patient_uid", "study_uid", "series_id", "classified_modality", "SeriesNumber", *EXCLUDED_FLAGS)


@dataclass(frozen=True)
class MrRateSeries:
    """One candidate metadata row: a series of one study of one patient in one batch."""

    batch: str
    patient_uid: str
    study_uid: str
    series_id: str
    modality: str
    series_number: float

    @property
    def label(self) -> str:
        """Whole-brain v1 label string (the skull-stripped twin is derived downstream)."""
        return MODALITY_TO_LABEL[self.modality]

    @property
    def image_path(self) -> str:
        """Path relative to the MR-RATE data root, official unzip layout (zip root == study dir)."""
        return f"mri/{self.batch}/{self.study_uid}/img/{self.study_uid}_{self.series_id}.nii.gz"

    @property
    def mask_path(self) -> str:
        """HD-BET brain mask path, same voxel grid as the image."""
        return f"mri/{self.batch}/{self.study_uid}/seg/{self.study_uid}_{self.series_id}_brain-mask.nii.gz"


class MrRateCatalog:
    """Repository over ``splits.csv`` and the 28 batch metadata CSVs.

    Loads and validates the raw rows; answers with the series that qualify as replay
    candidates (Train-split patients, brain modalities, data-quality flags excluded).
    """

    def __init__(self, splits_csv: Path, metadata_paths: list[Path]) -> None:
        self.splits_csv = splits_csv
        self.metadata_paths = metadata_paths
        if not metadata_paths:
            raise FileNotFoundError(f"no metadata CSVs provided (expected files like {METADATA_GLOB})")
        self._train_patients = self._load_train_patients()

    def train_brain_series(self) -> list[MrRateSeries]:
        """All qualifying series rows across every batch, in stable load order."""
        series: list[MrRateSeries] = []
        seen: set[tuple[str, str]] = set()
        for path in self.metadata_paths:
            for row in self._read_series_rows(path):
                key = (row.study_uid, row.series_id)
                if key in seen:
                    raise ValueError(f"duplicate (study_uid, series_id) across metadata CSVs: {key}")
                seen.add(key)
                series.append(row)
        return series

    def _load_train_patients(self) -> frozenset[str]:
        with self.splits_csv.open(newline="") as file:
            patients = {row["patient_uid"] for row in csv.DictReader(file) if row["split"] == TRAIN_SPLIT}
        if not patients:
            raise ValueError(f"no '{TRAIN_SPLIT}' patients found in {self.splits_csv}")
        return frozenset(patients)

    def _read_series_rows(self, path: Path) -> list[MrRateSeries]:
        rows = []
        with path.open(newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            index = self._column_index(header, path)
            for fields in reader:
                record = {name: fields[position] for name, position in index.items()}
                series = self._to_series(record, path)
                if series is not None:
                    rows.append(series)
        return rows

    @staticmethod
    def _column_index(header: list[str], path: Path) -> dict[str, int]:
        missing = sorted(name for name in SERIES_COLUMNS if name not in header)
        if missing:
            raise ValueError(f"{path}: metadata is missing columns {missing}")
        return {name: header.index(name) for name in SERIES_COLUMNS}

    def _to_series(self, record: dict[str, str], path: Path) -> MrRateSeries | None:
        if record["patient_uid"] not in self._train_patients:
            return None
        raw_modality = record["classified_modality"]
        modality = raw_modality.lower()
        if modality not in MODALITY_TO_LABEL:
            raise ValueError(f"{path}: unexpected classified_modality {raw_modality!r}")
        # The pull writes Python-style booleans; anything else would silently flip the
        # exclusion, so fail loudly instead (verified "True"/"False" on the 2026-09 pull).
        for flag in EXCLUDED_FLAGS:
            if record[flag] not in ("True", "False"):
                raise ValueError(f"{path}: unexpected {flag} value {record[flag]!r} (expected 'True'/'False')")
        if any(record[flag] == "True" for flag in EXCLUDED_FLAGS):
            return None
        batch = path.name.split("_")[0]
        return MrRateSeries(
            batch=batch,
            patient_uid=record["patient_uid"],
            study_uid=record["study_uid"],
            series_id=record["series_id"],
            modality=modality,
            # Missing SeriesNumber can only win when it is the subject's sole series.
            series_number=float(record["SeriesNumber"]) if record["SeriesNumber"] else float("inf"),
        )


class ModalitySubjectIndex:
    """Subject-level view of the candidate series: one entry per (subject, modality),
    the min-``SeriesNumber`` series winning; entries per modality sorted by ``patient_uid``."""

    def __init__(self, series: list[MrRateSeries]) -> None:
        self._entries = self._reduce_to_subjects(series)

    def entries(self, modality: str) -> list[MrRateSeries]:
        """Candidate series of one modality, one per subject, sorted by ``patient_uid``."""
        return list(self._entries[modality])

    @staticmethod
    def _reduce_to_subjects(series: list[MrRateSeries]) -> dict[str, list[MrRateSeries]]:
        best: dict[tuple[str, str], MrRateSeries] = {}
        for item in series:
            key = (item.modality, item.patient_uid)
            if key not in best or (item.series_number, item.series_id) < (best[key].series_number, best[key].series_id):
                best[key] = item
        grouped: dict[str, list[MrRateSeries]] = {modality: [] for modality in MODALITY_TO_LABEL}
        for (modality, _patient), item in best.items():
            grouped[modality].append(item)
        for entries in grouped.values():
            entries.sort(key=lambda item: item.patient_uid)
        return grouped


@dataclass(frozen=True)
class CapPolicy:
    """Layered-cap rule (spec section 3.3): head labels N each, T2w min(N, 669), MRA uncapped."""

    n_per_label: int
    t2w_cap: int = T2W_CAP

    def cap_for(self, modality: str) -> int | None:
        """Sample cap for one modality; ``None`` means take every available subject."""
        if modality == "mra":
            return None
        if modality == "t2w":
            return min(self.n_per_label, self.t2w_cap)
        return self.n_per_label


class StratifiedSubjectSampler:
    """Seeded subject sampler: a capped draw is the prefix of a seeded shuffle of the
    sorted candidates, so reruns reproduce line by line and top-ups continue the order."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def select(self, modality: str, candidates: list[MrRateSeries], cap: int | None) -> list[MrRateSeries]:
        """First ``cap`` entries of the modality's seeded shuffle over ``candidates``;
        ``cap=None`` keeps the full ordering (the top-up continuation contract)."""
        shuffled = sorted(candidates, key=lambda item: item.patient_uid)
        random.Random(self._modality_seed(modality)).shuffle(shuffled)
        return shuffled[: len(shuffled) if cap is None else cap]

    def _modality_seed(self, modality: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{modality}".encode()).digest()
        return int.from_bytes(digest[:8], "big")


class ReplayManifest:
    """Assembles the selected series into the manifest CSV (spec section 3.4 columns)."""

    def __init__(self, selected: list[MrRateSeries]) -> None:
        self._selected = sorted(selected, key=lambda item: (MODALITIES.index(item.modality), item.patient_uid, item.study_uid, item.series_id))

    def save(self, output_path: Path) -> dict:
        """Write the manifest CSV; return a summary dict with "rows" and per-label "counts"."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(MANIFEST_COLUMNS)
            for item in self._selected:
                writer.writerow([item.patient_uid, item.study_uid, item.series_id, item.modality, item.label, MANIFEST_SPLIT_VALUE, item.image_path])
        return {"rows": len(self._selected), "counts": self._counts()}

    def _counts(self) -> dict[str, int]:
        counts = {label: 0 for label in MODALITY_TO_LABEL.values()}
        for item in self._selected:
            counts[item.label] += 1
        return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits-csv", type=Path, required=True, help="MR-RATE splits.csv (patient-level train/val/test)")
    parser.add_argument("--metadata-dir", type=Path, required=True, help="directory holding the 28 batchXX_metadata.csv files")
    parser.add_argument(
        "--n-per-label",
        type=int,
        required=True,
        help="N: head labels sample N subjects each, T2w min(N, 669), MRA all available",
    )
    parser.add_argument("--output", type=Path, required=True, help="path of the replay manifest CSV to write")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (fixed => reproducible manifest)")
    args = parser.parse_args()

    catalog = MrRateCatalog(
        splits_csv=args.splits_csv,
        metadata_paths=sorted(args.metadata_dir.glob(METADATA_GLOB)),
    )
    index = ModalitySubjectIndex(catalog.train_brain_series())
    sampler = StratifiedSubjectSampler(seed=args.seed)
    policy = CapPolicy(n_per_label=args.n_per_label)

    selected = []
    below_cap = {}  # modality -> (sampled, requested cap) where availability fell short
    for modality in MODALITIES:
        candidates = index.entries(modality)
        cap = policy.cap_for(modality)
        chosen = sampler.select(modality, candidates, cap=cap)
        selected.extend(chosen)
        requested = "all" if cap is None else cap
        if cap is not None and len(chosen) < cap:
            below_cap[modality] = (len(chosen), cap)
        print(f"{modality}: availability={len(candidates)} requested={requested} sampled={len(chosen)}")

    manifest = ReplayManifest(selected)
    summary = manifest.save(args.output)
    counts = ", ".join(f"{label}={count}" for label, count in summary["counts"].items())
    print(f"sampled per modality: {counts}")
    print(f"manifest rows={summary['rows']} (dual derivation doubles this into training entries)")
    if below_cap:
        print(f"WARNING: availability below cap for {below_cap}; top-up after spine filtering can refill")
    print(f"wrote {args.output} (seed={args.seed}, n_per_label={args.n_per_label})")


if __name__ == "__main__":
    main()
