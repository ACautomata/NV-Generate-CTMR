# PROTOTYPE (throwaway, wayfinder #15) — write the companion metadata JSON the
# training loader needs.
#
# Gap found during the smoke: diff_model_train.py reads, for every embedding,
# a companion `<image>_emb.nii.gz.json` holding {"spacing", "modality"} (see
# prepare_data's _load_data_from_file). But neither this repo's nor upstream
# NVIDIA's diff_model_create_training_data.py writes that file — only the
# _emb.nii.gz is saved. So the smoke generates the companions here, after the
# VAE-encode step.
#
#   spacing  — read back from the written embedding's own affine (pixdim), i.e.
#              the resampled image-level spacing the DM conditions on.
#   modality — the modality_mapping key from the smoke data list (e.g.
#              "mri_t1_skull_stripped" -> 29). body-region indices are omitted
#              because config_network_rflow.json sets include_body_region=false.
#
# Run on the DCU node, after diff_model_create_training_data.py:
#   python prototype/dcu_smoke/write_emb_metadata.py \
#       --data-list prototype/dcu_smoke/dataset_dcu_smoke.json \
#       --embedding-base-dir /root/private_data/nv-dcu-smoke/embeddings

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib


class EmbMetadataWriter:
    """Writes the <emb>.json companion (spacing + modality) for each embedding."""

    def __init__(self, data_list_path: Path, embedding_base_dir: Path) -> None:
        self._data_list_path = data_list_path
        self._embedding_base_dir = embedding_base_dir

    def _entries(self) -> list[dict]:
        return json.loads(self._data_list_path.read_text())["training"]

    def _write_one(self, entry: dict) -> Path:
        emb_rel = entry["image"].replace(".nii.gz", "_emb.nii.gz")
        emb_path = self._embedding_base_dir / emb_rel
        if not emb_path.is_file():
            raise FileNotFoundError(f"embedding not found (run data prep first): {emb_path}")
        spacing = [float(v) for v in nib.load(str(emb_path)).header["pixdim"][1:4]]
        companion = {"spacing": spacing, "modality": entry["modality"]}
        out_path = emb_path.with_name(emb_path.name + ".json")
        out_path.write_text(json.dumps(companion))
        return out_path

    def write_all(self) -> int:
        paths = [self._write_one(entry) for entry in self._entries()]
        return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write <emb>.json companion metadata for the DCU smoke")
    parser.add_argument("--data-list", type=Path, required=True, help="Smoke data list JSON (image + modality)")
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="Dir holding the *_emb.nii.gz files")
    args = parser.parse_args()

    writer = EmbMetadataWriter(args.data_list, args.embedding_base_dir)
    count = writer.write_all()
    print(f"wrote {count} companion .json files under {args.embedding_base_dir}")


if __name__ == "__main__":
    main()
