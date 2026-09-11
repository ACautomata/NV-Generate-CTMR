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

"""Merged dataset.json generator: BRATS stage-1 dataset.json + replay manifest -> the training
data of one experiment point (spec #13 sections 3.4 / 3.5, ticket T7 #23, new-code list item 5).

The merged file is what the finetune env configs consume (``json_data_list``):
``dataset_rflow-mr-brain_N{N}.json``.  Its ``training`` list is the BRATS stage-1 entries
(4,748 on the current data, verbatim and in file order) followed by the replay entries the
accepted manifest expands into -- the dual derivation of section 3.3, two entries per series
(``mri_t1`` ... ``mri_mra`` whole-brain plus the ``_skull_stripped`` twins).  Its ``validation``
list is the BRATS roster, verbatim; there is **no replay-side validation** (section 3.4: the
FID reference set never enters a dataset.json).

``--n-per-label N`` is the experiment matrix's only knob (section 3.5).  It is *not* a sampler
here -- the manifest already froze its tier's roster -- it re-checks the roster against the
layered-cap contract (head labels N, T2w ``min(N, 669)``, MRA all, ``CapPolicy``): a manifest
holding more series of a modality than its cap allows is refused, because the tiers are nested
prefixes of one ordering and a bigger tier cannot be truncated back into a smaller one -- pass
the tier's own ``*_final.csv``.  A roster *short* of its cap is kept and reported (the
spine-filter/availability shortfall T6 already accepted: never padded).

Before anything is written every merged entry is audited against the training roots it must
resolve under, because the training loop silently skips entries whose files are missing
(``diff_model_train.load_filenames`` filters on ``os.path.exists``) and a quiet shrink is
indistinguishable from an intended subset:

- the source image under ``--data-base-dir``;
- the latent and its sidecar under ``--embedding-base-dir``;
- the sidecar's ``modality`` equal to the entry's -- the ticket's "模态字符串与 manifest 一致",
  checked against the written sidecar rather than assumed from the writer's contract.

The same audit reaches the BRATS half, so "training-ready" means every path in the file
resolves.  The roots themselves are deployment: one merged directory per tier holding a
``brats2023-gli/`` and an ``mri/`` link under both ``data/`` and ``embeddings/``::

    RUN=runs/merge-dataset-<yyyymmdd>/N300        # per tier
    mkdir -p $RUN/data $RUN/embeddings
    ln -s $RUN_ROOT/datasets/brats2023-gli                          $RUN/data/brats2023-gli
    ln -s $RUN_ROOT/runs/replay-latent-20260910/data/mri            $RUN/data/mri
    ln -s $RUN_ROOT/runs/brats-latent-20260910/embeddings/brats2023-gli $RUN/embeddings/brats2023-gli
    ln -s $RUN_ROOT/runs/replay-latent-20260910/data/mri            $RUN/embeddings/mri
    uv run python -m scripts.create_merged_dataset_json \\
        --brats-json $RUN_ROOT/datasets/brats2023-gli/dataset.json \\
        --replay-manifest $RUN_ROOT/datasets/MR-RATE/manifests/replay_manifest_N300_final.csv \\
        --n-per-label 300 \\
        --data-base-dir $RUN/data \\
        --embedding-base-dir $RUN/embeddings \\
        --output $RUN/dataset_rflow-mr-brain_N300.json

Writing is a pure function of the inputs -- BRATS entries in file order, replay entries grouped
by modality in the manifest's own order, ``json.dump(indent=2)`` -- so a same-argument rerun
reproduces the file byte for byte.
"""

import argparse
import csv
import json
from pathlib import Path

from scripts.create_replay_manifest import MANIFEST_COLUMNS, CapPolicy
from scripts.download_replay_subset import ManifestCandidate
from scripts.latent_sidecars import LatentEntry, LatentSidecarWriter

SPLIT_VALUE_TRAIN = "Train"
MODALITY_ORDER = ("t1w", "t2w", "flair", "swi", "mra")


class BratsDatasetJson:
    """Reads the stage-1 BRATS dataset.json back: its training records and validation roster, verbatim."""

    def __init__(self, path: Path) -> None:
        with path.open() as file:
            payload = json.load(file)
        self.training: list[dict[str, str]] = payload["training"]
        self.validation: list[str] = payload["validation"]

    def entries(self) -> list[LatentEntry]:
        """The training records as latent entries, in file order."""
        return [LatentEntry(image=item["image"], modality=item["modality"]) for item in self.training]


class AcceptedReplayRoster:
    """The tier's accepted replay series, read back from its final manifest and re-checked
    against the layered-cap contract before anything is merged.

    The roster is a frozen artifact (spec section 3.4); this class does not sample.  It states
    what the merge needs the roster to be: all-Train, within cap per modality, every modality
    present (a label silently absent from the mixed training set is a forgetting risk, not a
    smaller experiment), and each series carrying its dual derivation.
    """

    def __init__(self, manifest_path: Path, n_per_label: int) -> None:
        self._candidates = self._read(manifest_path)
        self._policy = CapPolicy(n_per_label=n_per_label)
        self._require_training_split(self._candidates, manifest_path)
        self._grouped = self._group_by_modality(manifest_path)

    def entries(self) -> list[LatentEntry]:
        """Both halves of the dual derivation per series, grouped by modality in manifest order."""
        return [entry for candidate in self._ordered_candidates() for entry in candidate.variant.training_entries()]

    def series_count(self) -> int:
        return len(self._candidates)

    def per_modality(self) -> dict[str, int]:
        """Accepted series per modality (whole-brain halves; the twins double into entries)."""
        return {modality: len(candidates) for modality, candidates in self._grouped.items()}

    def shortfalls(self) -> dict[str, tuple[int, int]]:
        """Capped modalities that came back short of their cap: modality -> (accepted, cap)."""
        return {
            modality: (len(candidates), cap)
            for modality, candidates in self._grouped.items()
            if (cap := self._policy.cap_for(modality)) is not None and len(candidates) < cap
        }

    def _ordered_candidates(self) -> list[ManifestCandidate]:
        return [candidate for modality in MODALITY_ORDER for candidate in self._grouped[modality]]

    @staticmethod
    def _read(manifest_path: Path) -> list[ManifestCandidate]:
        """Every manifest row as a validated candidate, in row order.

        ``ManifestCandidate.from_row`` re-checks each row's label against its series id, so a
        hand-edited manifest cannot smuggle a mislabelled series into the training set.
        """
        with manifest_path.open(newline="") as file:
            reader = csv.reader(file)
            header = next(reader)
            if header != list(MANIFEST_COLUMNS):
                raise ValueError(f"{manifest_path}: expected columns {list(MANIFEST_COLUMNS)}, found {header}")
            return [ManifestCandidate.from_row(dict(zip(MANIFEST_COLUMNS, row))) for row in reader]

    @staticmethod
    def _require_training_split(candidates: list[ManifestCandidate], manifest_path: Path) -> None:
        """Refuse to merge anything but the Train split (the forgetting reference set is Val)."""
        strangers = sorted({candidate.split for candidate in candidates} - {SPLIT_VALUE_TRAIN})
        if strangers:
            raise ValueError(f"{manifest_path}: roster holds non-{SPLIT_VALUE_TRAIN} rows ({', '.join(strangers)})")

    def _group_by_modality(self, manifest_path: Path) -> dict[str, list[ManifestCandidate]]:
        """Candidates per modality in manifest order, after the cap and presence checks."""
        grouped: dict[str, list[ManifestCandidate]] = {}
        for candidate in self._candidates:
            grouped.setdefault(candidate.modality, []).append(candidate)
        for modality, candidates in grouped.items():
            cap = self._policy.cap_for(modality)
            if cap is not None and len(candidates) > cap:
                raise ValueError(
                    f"{manifest_path}: {modality} holds {len(candidates)} series, above the cap {cap} for "
                    f"n-per-label={self._policy.n_per_label}; the tiers are nested prefixes of one ordering, so pass "
                    f"this tier's own *_final.csv -- a larger tier cannot be truncated back into a smaller one"
                )
        missing = [modality for modality in MODALITY_ORDER if modality not in grouped]
        if missing:
            raise ValueError(f"{manifest_path}: modality absent from the roster: {', '.join(missing)} (a label silently leaving the training set)")
        return {modality: grouped[modality] for modality in MODALITY_ORDER}


class SidecarModalityAudit:
    """Reads each written sidecar and refuses one whose ``modality`` drifts from its entry.

    The sidecar string is what the condition embedding looks up and the dataset.json string is
    what the intensity dispatch keys on; the ticket's "模态字符串与 manifest 一致" is checked here
    against the bytes on disk rather than assumed from the sidecar writer's contract.
    """

    def __init__(self, embedding_base_dir: Path) -> None:
        self._embedding_base_dir = embedding_base_dir

    def require_consistent(self, entries: list[LatentEntry]) -> None:
        """Raise when a sidecar is unreadable or names another modality than its entry."""
        mismatches = []
        for entry in entries:
            sidecar_path = self._embedding_base_dir / entry.sidecar_relative_path
            try:
                payload = json.loads(sidecar_path.read_text())
            except json.JSONDecodeError as error:
                raise ValueError(f"sidecar {sidecar_path} is not valid JSON: {error}") from error
            if payload["modality"] != entry.modality:
                mismatches.append(f"{entry.sidecar_relative_path}: sidecar says {payload['modality']!r}, entry says {entry.modality!r}")
        if mismatches:
            preview = "; ".join(mismatches[:5])
            raise ValueError(f"{len(mismatches)} of {len(entries)} sidecars disagree with their entry's modality, e.g.: {preview}")


class MergedDatasetJson:
    """Assembles, audits and writes one experiment point's merged dataset.json.

    ``training`` = the BRATS stage-1 records (verbatim order) followed by the replay dual
    derivation; ``validation`` = the BRATS roster alone.  Saving audits every entry against the
    two roots first, so a file that exists is a file the training run can consume.
    """

    def __init__(self, brats: BratsDatasetJson, replay: AcceptedReplayRoster, data_base_dir: Path, embedding_base_dir: Path) -> None:
        self._brats = brats
        self._replay = replay
        self._data_base_dir = data_base_dir
        self._writer = LatentSidecarWriter(embedding_base_dir)
        self._sidecar_audit = SidecarModalityAudit(embedding_base_dir)

    def to_dict(self) -> dict:
        """The merged payload: replay entries after the BRATS block, validation untouched."""
        return {
            "training": [entry.to_record() for entry in self._all_entries()],
            "validation": list(self._brats.validation),
        }

    def save(self, output_path: Path) -> dict:
        """Audit, write, and return the summary the run log quotes (counts, ratio, roster shape)."""
        entries = self._all_entries()
        self._require_source_volumes(entries)
        self._writer.require_all(entries, str(output_path))
        self._sidecar_audit.require_consistent(entries)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.write("\n")

        counts = self._counts(entries)
        total = len(entries)
        summary = {
            "training_entries": total,
            "brats_entries": len(self._brats.training),
            "replay_entries": total - len(self._brats.training),
            "validation_cases": len(self._brats.validation),
            "counts": counts,
        }
        print(f"brats: training={summary['brats_entries']} validation={summary['validation_cases']}")
        print(f"replay: series={self._replay.series_count()} entries={summary['replay_entries']} per-modality {self._replay.per_modality()}")
        for modality, (accepted, cap) in self._replay.shortfalls().items():
            print(f"NOTE: {modality} came back {accepted} of cap {cap} (spine-filter/availability shortfall; kept, never padded)")
        brats_share = round(100 * summary["brats_entries"] / total)
        print(
            f"merged: training={total} (brats:replay = {brats_share}:{100 - brats_share}) "
            f"validation={summary['validation_cases']} (brats only, no replay-side val)"
        )
        print(f"wrote {output_path} (same arguments reproduce this file byte for byte)")
        return summary

    def _all_entries(self) -> list[LatentEntry]:
        return self._brats.entries() + self._replay.entries()

    def _require_source_volumes(self, entries: list[LatentEntry]) -> None:
        """Refuse to write when a source image does not resolve under the merged data root."""
        missing = [entry.image for entry in entries if not (self._data_base_dir / entry.image).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{len(missing)} of {len(entries)} source volumes missing under {self._data_base_dir}, e.g.: {preview}")

    @staticmethod
    def _counts(entries: list[LatentEntry]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.modality] = counts.get(entry.modality, 0) + 1
        return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brats-json", type=Path, required=True, help="the BRATS stage-1 dataset.json (training + validation)")
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        required=True,
        help="this tier's accepted replay manifest (*_final.csv, post spine filter)",
    )
    parser.add_argument("--n-per-label", type=int, required=True, help="N: the experiment matrix knob the manifest's tier was drawn with")
    parser.add_argument("--data-base-dir", type=Path, required=True, help="merged training root the image paths resolve under")
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="merged embedding root the latents and sidecars resolve under")
    parser.add_argument("--output", type=Path, required=True, help="path of the merged dataset.json (convention: dataset_rflow-mr-brain_N<N>.json)")
    args = parser.parse_args()

    merged = MergedDatasetJson(
        brats=BratsDatasetJson(args.brats_json),
        replay=AcceptedReplayRoster(args.replay_manifest, n_per_label=args.n_per_label),
        data_base_dir=args.data_base_dir,
        embedding_base_dir=args.embedding_base_dir,
    )
    merged.save(args.output)


if __name__ == "__main__":
    main()
