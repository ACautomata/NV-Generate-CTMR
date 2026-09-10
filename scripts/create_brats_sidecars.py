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

"""Stage-2 generator for the BRATS dataset.json sidecars (spec #13 section 2.3, ticket T5 #21).

Stage 2 (after latent encoding): for every ``training`` entry of the stage-1 dataset.json,
locate the latent under the embedding base dir, read the spacing measured in its NIfTI
header (post-resize physical spacing, for BRATS [0.9375, 0.9375, 1.2109]), and write a
sidecar next to the latent:

    {"spacing": [x, y, z], "modality": "<label string>"}

The modality string is copied verbatim from the dataset.json entry, so the two agree
entry by entry by construction (section 2.3: the dataset.json string drives the intensity
dispatch, the sidecar string drives the condition lookup). Sidecar paths follow the
training code's convention (``scripts/diff_model_train.py``): the image path with
``.nii.gz`` replaced by ``_emb.nii.gz``, plus ``.json``. Rerunning overwrites the
sidecars, keeping them in lockstep with the dataset.json they were generated from.

Entries are enumerated from the dataset.json rather than by scanning the embedding
base dir (section 2.3 says "scan"); the two are equivalent because every training
entry has exactly one latent and validation cases are never encoded (section 2.4) —
a stray latent outside the dataset.json is not a training sample and gets no sidecar.

Missing latents abort the run before anything is written: the stage-2 contract is that
the training data is directly consumable afterwards (section 2.5), so a partial latent
set must not yield a partial set of sidecars.

Usage (on gauss, after ``diff_model_create_training_data`` has encoded every entry)::

    uv run python -m scripts.create_brats_sidecars \\
        --dataset-json $RUN_ROOT/datasets/brats2023-gli/dataset.json \\
        --embedding-base-dir $RUN_ROOT/runs/brats-latent-<yyyymmdd>/embeddings
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

SIDECAR_EXTENSION = ".json"


@dataclass(frozen=True)
class TrainingEntry:
    """One dataset.json training record; owns the image->latent path convention the training code relies on."""

    image: str
    modality: str

    def embedding_relative_path(self) -> str:
        """Latent path relative to the embedding base dir (same replacement as ``diff_model_train.load_filenames``)."""
        return self.image.replace(".nii.gz", "_emb.nii.gz")


class BratsSidecarGenerator:
    """Writes one spacing/modality sidecar per training latent, reading the spacing from the latent NIfTI header."""

    def __init__(self, dataset_json_path: Path, embedding_base_dir: Path) -> None:
        self._dataset_json_path = dataset_json_path
        self._embedding_base_dir = embedding_base_dir

    def write_sidecars(self) -> dict:
        """Write every sidecar and return the count; abort without writing if any latent is missing."""
        entries = self.entries()
        latents = [self._embedding_base_dir / entry.embedding_relative_path() for entry in entries]
        missing = [latent for latent in latents if not latent.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise ValueError(f"{len(missing)} of {len(latents)} latents missing under {self._embedding_base_dir}, e.g.: {preview}")

        for entry, latent in zip(entries, latents, strict=True):
            self._write_sidecar(entry, latent)
        return {"written": len(entries)}

    def entries(self) -> list[TrainingEntry]:
        """The dataset.json training records, in file order."""
        with self._dataset_json_path.open() as file:
            payload = json.load(file)
        return [TrainingEntry(image=item["image"], modality=item["modality"]) for item in payload["training"]]

    def _write_sidecar(self, entry: TrainingEntry, latent: Path) -> None:
        spacing = [float(value) for value in nib.load(str(latent)).header.get_zooms()[:3]]
        sidecar = latent.with_name(latent.name + SIDECAR_EXTENSION)
        with sidecar.open("w") as file:
            json.dump({"spacing": spacing, "modality": entry.modality}, file, indent=2)
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", type=Path, required=True, help="the stage-1 dataset.json to read entries from")
    parser.add_argument(
        "--embedding-base-dir",
        type=Path,
        required=True,
        help="root directory the latents live under (the env's embedding_base_dir)",
    )
    args = parser.parse_args()

    generator = BratsSidecarGenerator(dataset_json_path=args.dataset_json, embedding_base_dir=args.embedding_base_dir)
    stats = generator.write_sidecars()
    print(f"sidecars={stats['written']}")


if __name__ == "__main__":
    main()
