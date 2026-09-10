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

"""Latent path conventions and sidecar writing, shared by the BRATS and replay pipelines
(spec #13 sections 2.3 / 3.4, tickets T5 #21 and T6 #22).

Both pipelines encode their source images with ``scripts.diff_model_create_training_data``
and then hand the training run a sidecar next to each latent::

    {"spacing": [x, y, z], "modality": "<label string>"}

The training code (``scripts/diff_model_train.py``) derives every path by convention, so
this module owns the two derivations in one place:

- the latent sits beside the image with ``.nii.gz`` replaced by ``_emb.nii.gz``;
- the sidecar is the latent path plus ``.json``.

The modality string is the label the dataset.json entry carries -- in BRATS the new
``mri_t1n``/``mri_t1ce``/``mri_t2w``/``mri_t2f`` (40-43), in replay the v1 labels
(``mri_t1``... ``mri_mra_skull_stripped``), which is what the condition embedding and the
intensity dispatch key on.  Reading the spacing from the latent header (rather than
copying the source volume's) is deliberate: it is the post-resize physical spacing the
condition actually sees.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

SIDECAR_EXTENSION = ".json"
LATENT_SUFFIX = "_emb.nii.gz"


@dataclass(frozen=True)
class LatentEntry:
    """One training entry: a source image path plus the label its sidecar must carry."""

    image: str
    modality: str

    @property
    def embedding_relative_path(self) -> str:
        """Latent path relative to the embedding base dir (the same replacement the training code does)."""
        return self.image.replace(".nii.gz", LATENT_SUFFIX)

    @property
    def sidecar_relative_path(self) -> str:
        """Sidecar path relative to the embedding base dir: the latent path plus the sidecar extension."""
        return self.embedding_relative_path + SIDECAR_EXTENSION

    def to_record(self) -> dict[str, str]:
        """The dataset.json record this entry stands for.

        The two fields are the whole record, and the training loader reads the latent and
        sidecar paths back out of ``image`` by convention -- so a record and an entry cannot
        be allowed to drift apart.  Deriving one from the other keeps them the same fact.
        """
        return {"image": self.image, "modality": self.modality}


class LatentSidecarWriter:
    """Writes one spacing/modality sidecar per latent, reading the spacing from the latent header."""

    def __init__(self, embedding_base_dir: Path) -> None:
        self._embedding_base_dir = embedding_base_dir

    def has(self, entry: LatentEntry) -> bool:
        """Whether this entry's latent is already on disk -- the encoder's own skip condition."""
        return (self._embedding_base_dir / entry.embedding_relative_path).is_file()

    def missing_latents(self, entries: list[LatentEntry]) -> list[Path]:
        """The latents that are not on disk yet; the stage-2 contract is that none may be."""
        return [self._embedding_base_dir / entry.embedding_relative_path for entry in entries if not self.has(entry)]

    def require_all(self, entries: list[LatentEntry], stage: str) -> None:
        """Raise before anything is written when a latent is missing, so no partial sidecar set can exist."""
        missing = self.missing_latents(entries)
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise ValueError(f"{stage}: {len(missing)} of {len(entries)} latents missing under {self._embedding_base_dir}, e.g.: {preview}")

    def write(self, entries: list[LatentEntry]) -> int:
        """Write every sidecar and return the count; abort without writing if any latent is missing."""
        self.require_all(entries, "sidecars")
        for entry in entries:
            self._write_one(entry)
        return len(entries)

    def _write_one(self, entry: LatentEntry) -> None:
        latent = self._embedding_base_dir / entry.embedding_relative_path
        spacing = [float(value) for value in nib.load(str(latent)).header.get_zooms()[:3]]
        sidecar = self._embedding_base_dir / entry.sidecar_relative_path
        with sidecar.open("w") as file:
            json.dump({"spacing": spacing, "modality": entry.modality}, file, indent=2)
            file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-json", type=Path, required=True, help="dataset.json whose training entries name the latents")
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="root directory the latents live under")
    args = parser.parse_args()

    with args.dataset_json.open() as file:
        payload = json.load(file)
    entries = [LatentEntry(image=item["image"], modality=item["modality"]) for item in payload["training"]]
    writer = LatentSidecarWriter(args.embedding_base_dir)
    writer.require_all(entries, str(args.dataset_json))
    print(f"sidecars={writer.write(entries)}")


if __name__ == "__main__":
    main()
