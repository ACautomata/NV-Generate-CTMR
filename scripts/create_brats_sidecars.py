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
locate the latent under the embedding base dir and write a sidecar next to it::

    {"spacing": [x, y, z], "modality": "<label string>"}

For BRATS the spacing is the post-resize physical spacing measured in the latent header
([0.9375, 0.9375, 1.2109]).  The modality string is copied verbatim from the dataset.json
entry, so the two agree entry by entry by construction (section 2.3: the dataset.json
string drives the intensity dispatch, the sidecar string drives the condition lookup).

Sidecar paths follow the training code's convention; both the path derivation and the
spacing/modality payload live in ``scripts.latent_sidecars``, which the replay pipeline
(``scripts.create_replay_latent_dataset``) shares.  Rerunning overwrites the sidecars,
keeping them in lockstep with the dataset.json they were generated from.

Entries are enumerated from the dataset.json rather than by scanning the embedding base
dir (section 2.3 says "scan"); the two are equivalent because every training entry has
exactly one latent and validation cases are never encoded (section 2.4) -- a stray latent
outside the dataset.json is not a training sample and gets no sidecar.

Missing latents abort the run before anything is written: the stage-2 contract is that the
training data is directly consumable afterwards (section 2.5), so a partial latent set must
not yield a partial set of sidecars.

Usage (on gauss, after ``diff_model_create_training_data`` has encoded every entry)::

    uv run python -m scripts.create_brats_sidecars \\
        --dataset-json $RUN_ROOT/datasets/brats2023-gli/dataset.json \\
        --embedding-base-dir $RUN_ROOT/runs/brats-latent-<yyyymmdd>/embeddings
"""

import argparse
import json
from pathlib import Path

from .latent_sidecars import LatentEntry, LatentSidecarWriter


class BratsSidecarGenerator:
    """Writes one spacing/modality sidecar per BRATS training latent, from the stage-1 dataset.json."""

    def __init__(self, dataset_json_path: Path, embedding_base_dir: Path) -> None:
        self._dataset_json_path = dataset_json_path
        self._writer = LatentSidecarWriter(embedding_base_dir)

    def write_sidecars(self) -> int:
        """Write every sidecar and return the count; abort without writing if any latent is missing."""
        return self._writer.write(self.entries())

    def entries(self) -> list[LatentEntry]:
        """The dataset.json training records, in file order."""
        with self._dataset_json_path.open() as file:
            payload = json.load(file)
        return [LatentEntry(image=item["image"], modality=item["modality"]) for item in payload["training"]]


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
    print(f"sidecars={generator.write_sidecars()}")


if __name__ == "__main__":
    main()
