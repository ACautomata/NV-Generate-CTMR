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

"""Smoke-run subset builder: merged dataset.json -> small mixed training list (ticket T8, issue #24).

The training smoke test (spec #13 stage D3) consumes a small slice of one experiment
point's merged dataset.json. The slice must still exercise what makes the real data
hard: every requested label (new BRATS labels 40-43 plus old replay labels) and
**mixed latent sizes** (BRATS 256x256x128 alongside MR-RATE native round-to-128
volumes) flowing through batch_size=1.

The training loop silently skips entries whose latent is missing, so this builder
audits every sampled entry before writing -- a smoke subset whose latents or
sidecars are absent is refused rather than silently shrunk. Sampling is seeded per
label, so the same arguments reproduce the subset line for line.

Usage::

    python -m scripts.create_smoke_dataset_json \
        --dataset-json runs/merge-dataset-20260911/N300/dataset_rflow-mr-brain_N300.json \
        --embedding-base-dir runs/merge-dataset-20260911/N300/embeddings \
        --per-label 2 \
        --output runs/t8-smoke-20260911/configs/dataset_smoke.json
"""

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

BRATS_LABELS = ("mri_t1n", "mri_t1ce", "mri_t2w", "mri_t2f")
REPLAY_LABELS = (
    "mri_t1",
    "mri_t2",
    "mri_flair",
    "mri_mra",
    "mri_swi",
    "mri_t1_skull_stripped",
    "mri_t2_skull_stripped",
    "mri_flair_skull_stripped",
    "mri_mra_skull_stripped",
    "mri_swi_skull_stripped",
)
DEFAULT_LABELS = BRATS_LABELS + REPLAY_LABELS
DEFAULT_SEED = 42
DEFAULT_PER_LABEL = 2


@dataclass(frozen=True)
class SmokeQuota:
    """How many entries per label the smoke subset carries, and which labels are required."""

    per_label: int
    labels: tuple[str, ...] = DEFAULT_LABELS


class LatentPaths:
    """Derives the on-disk latent and sidecar paths of a dataset.json entry and reads latent headers.

    Mirrors the training-side derivation (``image.replace(".nii.gz", "_emb.nii.gz")``
    under the embedding base dir; sidecar = latent path + ".json").
    """

    def __init__(self, embedding_base_dir: Path) -> None:
        self._embedding_base_dir = embedding_base_dir

    def latent(self, image_relative: str) -> Path:
        return self._embedding_base_dir / image_relative.replace(".nii.gz", "_emb.nii.gz")

    def sidecar(self, image_relative: str) -> Path:
        latent = self.latent(image_relative)
        return latent.parent / (latent.name + ".json")

    def shape(self, image_relative: str) -> tuple[int, ...]:
        """The latent's voxel shape, from its NIfTI header."""
        return tuple(int(value) for value in nib.load(self.latent(image_relative)).shape)


class SmokeSubsetBuilder:
    """Samples a seeded, audited, size-mixed subset from a merged dataset.json entry list."""

    def __init__(self, entries: list[dict], quota: SmokeQuota, embedding_base_dir: Path, seed: int) -> None:
        self._entries = entries
        self._quota = quota
        self._paths = LatentPaths(embedding_base_dir)
        self._seed = seed

    def build(self) -> list[dict]:
        """The subset entries, ordered by (modality, image); raises rather than emitting a hollow smoke."""
        selected: list[dict] = []
        for label in self._quota.labels:
            candidates = sorted(entry["image"] for entry in self._entries if entry["modality"] == label)
            if not candidates:
                raise ValueError(f"requested smoke label {label!r} has no entries in the dataset")
            shuffled = list(candidates)
            random.Random(self._modality_seed(label)).shuffle(shuffled)
            chosen = shuffled[: self._quota.per_label]
            selected += [{"image": image, "modality": label} for image in chosen]

        self._audit(selected)
        return sorted(selected, key=lambda entry: (entry["modality"], entry["image"]))

    def _audit(self, selected: list[dict]) -> None:
        shapes: set[tuple[int, ...]] = set()
        for entry in selected:
            latent = self._paths.latent(entry["image"])
            if not latent.is_file():
                raise FileNotFoundError(f"smoke latent missing (training would silently skip it): {latent}")
            if not self._paths.sidecar(entry["image"]).is_file():
                raise FileNotFoundError(f"smoke sidecar missing: {self._paths.sidecar(entry['image'])}")
            shapes.add(self._paths.shape(entry["image"]))
        if len(shapes) < 2:
            raise ValueError(
                f"smoke subset carries a single latent shape {sorted(shapes)[0]}; "
                "the mixed-size consumption property is not exercised -- check the quota covers BRATS and replay"
            )

    def _modality_seed(self, label: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", type=Path, required=True, help="merged dataset.json to sample from")
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="root the entry image paths resolve against")
    parser.add_argument("--per-label", type=int, default=DEFAULT_PER_LABEL, help="entries sampled per label (default 2)")
    parser.add_argument("--labels", nargs="+", choices=DEFAULT_LABELS, default=list(DEFAULT_LABELS), help="labels the smoke must cover")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed (fixed => reproducible subset)")
    parser.add_argument("--output", type=Path, required=True, help="path of the smoke dataset.json to write")
    args = parser.parse_args()

    with args.dataset_json.open() as file:
        entries = json.load(file)["training"]

    subset = SmokeSubsetBuilder(
        entries, SmokeQuota(per_label=args.per_label, labels=tuple(args.labels)), args.embedding_base_dir, seed=args.seed
    ).build()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        json.dump({"training": subset}, file, indent=2)

    per_label: dict[str, int] = {}
    for entry in subset:
        per_label[entry["modality"]] = per_label.get(entry["modality"], 0) + 1
    counts = ", ".join(f"{label}={count}" for label, count in sorted(per_label.items()))
    print(f"smoke subset: {len(subset)} entries ({counts}); wrote {args.output}")


if __name__ == "__main__":
    main()
