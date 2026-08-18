# PROTOTYPE (throwaway, wayfinder #15) — build the DCU smoke data list.
#
# Picks a handful of BraTS GLI cases already on the cluster and emits a
# MONAI-style {"training": [...]} JSON for diff_model_create_training_data.py
# (VAE encode) and diff_model_train.py.
#
# Only the three v1-DM-*trained* skull-stripped modalities are used, so the
# class label stays a valid embedding row for the frozen rflow-mr-brain v1 DM:
#   t1n -> mri_t1_skull_stripped    (index 29)
#   t2w -> mri_t2_skull_stripped    (index 30)
#   t2f -> mri_flair_skull_stripped (index 31)
# t1c is deliberately skipped: its skull-stripped index 34 is a P1-planned
# addition the v1 DM never trained (would be an untrained embedding row).
#
# Each "image" path is relative to the env config's data_base_dir. P1 is
# image-only, so entries carry just image + modality.
#
# Run on the DCU node:
#   python prototype/dcu_smoke/make_smoke_datalist.py \
#       --data-base-dir /root/private_data/datasets/ASNR-MICCAI-BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
#       --out /root/private_data/nv-dcu-smoke/NV-Generate-CTMR/prototype/dcu_smoke/dataset_dcu_smoke.json \
#       --n-cases 6

from __future__ import annotations

import argparse
import json
from pathlib import Path

# BraTS suffix -> modality_mapping.json key (v1-trained skull-stripped only).
MODALITY_FOR_SUFFIX = {
    "t1n": "mri_t1_skull_stripped",
    "t2w": "mri_t2_skull_stripped",
    "t2f": "mri_flair_skull_stripped",
}


class SmokeDataListBuilder:
    """Builds a minimal P1 image-only data list from on-disk BraTS GLI cases."""

    def __init__(self, data_base_dir: Path, out_path: Path, n_cases: int) -> None:
        self._data_base_dir = data_base_dir
        self._out_path = out_path
        self._n_cases = n_cases

    def _case_dirs(self) -> list[Path]:
        cases = sorted(path for path in self._data_base_dir.iterdir() if path.is_dir())
        if not cases:
            raise FileNotFoundError(f"no case dirs under {self._data_base_dir}")
        return cases[: self._n_cases]

    def build(self) -> list[dict]:
        entries = []
        for case_dir in self._case_dirs():
            case = case_dir.name
            for suffix, modality in MODALITY_FOR_SUFFIX.items():
                image = case_dir / f"{case}-{suffix}.nii.gz"
                if not image.is_file():
                    raise FileNotFoundError(f"missing modality file: {image}")
                entries.append({"image": f"{case}/{image.name}", "modality": modality})
        return entries

    def write(self) -> Path:
        entries = self.build()
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_path.write_text(json.dumps({"training": entries}, indent=1))
        return self._out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DCU smoke data list")
    parser.add_argument("--data-base-dir", type=Path, required=True, help="BraTS GLI training-data dir (case dirs)")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset JSON path")
    parser.add_argument("--n-cases", type=int, default=6, help="Number of GLI cases to include")
    args = parser.parse_args()

    builder = SmokeDataListBuilder(args.data_base_dir, args.out, args.n_cases)
    out = builder.write()
    entries = json.loads(out.read_text())["training"]
    print(f"wrote {len(entries)} entries ({args.n_cases} cases x {len(MODALITY_FOR_SUFFIX)} modalities) -> {out}")


if __name__ == "__main__":
    main()
