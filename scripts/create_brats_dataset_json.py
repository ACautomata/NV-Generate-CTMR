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

"""Stage-1 generator for the BRATS dataset.json (spec #13 sections 2.3 / 2.4, ticket T1 #17).

Stage 1 (before latent encoding): enumerate the BraTS 2023 GLI training cases x 4
modalities (seg stays on disk and never enters dataset.json), split the subjects 95/5
with a frozen seed (timepoints of one subject never cross the split), and emit a single
dataset.json:

- ``"training"``:   [{"image": <relative to data-base-dir>, "modality": <label>}, ...]
  Validation cases are absent from training (section 2.4: validation is only the
  image-domain FID reference and is never latent-encoded).
- ``"validation"``: [case directory name, ...] -- the FID reference roster (the training
  code reads ``training`` only).

Modality is mapped from the file suffix: t1n->``mri_t1n``, t1c->``mri_t1ce``,
t2w->``mri_t2w``, t2f->``mri_t2f``.
The number of validation subjects is floor(total subjects x val_fraction); with a fixed
seed the output reproduces line by line. The section 2.1 spot-check assertions (t1c shape
(240, 240, 155), seg in {0, 1, 2, 3}) run on the lexicographically first case on every
invocation.

Usage (on gauss, with the dataset linked from the shared pool)::

    uv run python -m scripts.create_brats_dataset_json \\
        --training-data-dir $RUN_ROOT/datasets/brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \\
        --data-base-dir $RUN_ROOT/datasets \\
        --output $RUN_ROOT/datasets/brats2023-gli/dataset.json

Stage 2 (scanning embedding headers to write the sidecar after latent encoding) is a
separate extension outside this script.
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

SCAN_NAME_PATTERN = re.compile(r"BraTS-GLI-\d{5}-\d{3}")
NIFTI_EXTENSION = ".nii.gz"
SEG_SUFFIX = "seg"
SUFFIX_TO_MODALITY = {"t1n": "mri_t1n", "t1c": "mri_t1ce", "t2w": "mri_t2w", "t2f": "mri_t2f"}
REQUIRED_SUFFIXES = (SEG_SUFFIX, *SUFFIX_TO_MODALITY)
EXPECTED_SHAPE = (240, 240, 155)
SEG_LABELS = {0, 1, 2, 3}


@dataclass(frozen=True)
class BraTSScan:
    """One BraTS case (a single-timepoint scan); the directory name looks like ``BraTS-GLI-00000-000``."""

    directory: str

    @property
    def subject(self) -> str:
        """Longitudinal subject (the ``BraTS-GLI-XXXXX`` segment); timepoints of one subject never cross train/val."""
        return self.directory.rsplit("-", 1)[0]

    def path(self, root: Path, suffix: str) -> Path:
        """Full path of a case file under ``root``; the one place production code builds ``<directory>-<suffix>.nii.gz``."""
        return root / self.directory / f"{self.directory}-{suffix}{NIFTI_EXTENSION}"


@dataclass(frozen=True)
class SubjectSplit:
    """Result of a subject-level holdout split."""

    train: frozenset[str]
    val: frozenset[str]


class BraTSScanIndex:
    """Case index for a training-data directory: scan the directory names and check every case has all five files."""

    def __init__(self, training_data_dir: Path) -> None:
        self._training_data_dir = training_data_dir
        if not training_data_dir.is_dir():
            raise FileNotFoundError(f"training data dir not found: {training_data_dir}")
        self._scans = self._scan_directories()

    @property
    def scans(self) -> list[BraTSScan]:
        """All cases, sorted by directory name."""
        return list(self._scans)

    @property
    def subjects(self) -> list[str]:
        """Deduplicated subject segments, sorted."""
        return sorted({scan.subject for scan in self._scans})

    def _scan_directories(self) -> list[BraTSScan]:
        scans = []
        for case_dir in sorted(path for path in self._training_data_dir.iterdir() if path.is_dir()):
            if not SCAN_NAME_PATTERN.fullmatch(case_dir.name):
                raise ValueError(f"unexpected case directory name: {case_dir.name}")
            scan = BraTSScan(directory=case_dir.name)
            missing = sorted(suffix for suffix in REQUIRED_SUFFIXES if not scan.path(self._training_data_dir, suffix).is_file())
            if missing:
                raise ValueError(f"case {scan.directory} is missing suffixes: {missing}")
            scans.append(scan)
        return scans


class HoldoutSplitter:
    """Subject-level 95/5 split: shuffle with a fixed seed, then take the first floor(n x val_fraction) subjects as validation."""

    def __init__(self, val_fraction: float = 0.05, seed: int = 42) -> None:
        self._val_fraction = val_fraction
        self._seed = seed

    def split(self, subjects: list[str]) -> SubjectSplit:
        shuffled = sorted(subjects)
        random.Random(self._seed).shuffle(shuffled)
        n_val = int(len(shuffled) * self._val_fraction)
        return SubjectSplit(train=frozenset(shuffled[n_val:]), val=frozenset(shuffled[:n_val]))


class BraTSDatasetList:
    """Stage-1 dataset.json assembly: ``training`` entries (validation cases excluded) plus the ``validation`` roster."""

    def __init__(
        self,
        training_data_dir: Path,
        data_base_dir: Path,
        scans: list[BraTSScan],
        split: SubjectSplit,
    ) -> None:
        self._training_data_dir = training_data_dir
        self._data_base_dir = data_base_dir
        self._scans = scans
        self._split = split

    def to_dict(self) -> dict:
        return {"training": self._training_entries(), "validation": self._validation_roster()}

    def save(self, output_path: Path) -> dict:
        """Write dataset.json and return the payload written, so callers reuse it instead of re-assembling."""
        payload = self.to_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        return payload

    def _training_entries(self) -> list[dict]:
        entries = []
        for scan in sorted(self._scans, key=lambda item: item.directory):
            if scan.subject in self._split.val:
                continue
            for suffix in sorted(SUFFIX_TO_MODALITY):
                entries.append({"image": self._image_path(scan, suffix), "modality": SUFFIX_TO_MODALITY[suffix]})
        return entries

    def _validation_roster(self) -> list[str]:
        return sorted(scan.directory for scan in self._scans if scan.subject in self._split.val)

    def _image_path(self, scan: BraTSScan, suffix: str) -> str:
        return scan.path(self._training_data_dir, suffix).relative_to(self._data_base_dir).as_posix()


class NiftiSpotCheck:
    """Section 2.1 spot-check assertions: t1c shape == (240, 240, 155), seg labels subset of {0, 1, 2, 3}."""

    def __init__(self, training_data_dir: Path) -> None:
        self._training_data_dir = training_data_dir

    def run(self, scan: BraTSScan) -> None:
        """Raise ValueError on failure (not assert: ``python -O`` strips asserts and would silently disable the check)."""
        t1c = nib.load(str(scan.path(self._training_data_dir, "t1c")))
        if t1c.shape != EXPECTED_SHAPE:
            raise ValueError(f"{scan.directory}: t1c shape {t1c.shape} != {EXPECTED_SHAPE}")
        seg = np.asarray(nib.load(str(scan.path(self._training_data_dir, SEG_SUFFIX))).dataobj)
        labels = set(np.unique(seg).tolist())
        if not labels <= SEG_LABELS:
            raise ValueError(f"{scan.directory}: seg labels {sorted(labels)} not within {sorted(SEG_LABELS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        required=True,
        help="the ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData directory",
    )
    parser.add_argument(
        "--data-base-dir",
        type=Path,
        required=True,
        help="base directory of the relative image paths (the training env's data_base_dir)",
    )
    parser.add_argument("--output", type=Path, required=True, help="path of the dataset.json to write")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split (fixed => reproducible)")
    args = parser.parse_args()

    index = BraTSScanIndex(args.training_data_dir)
    scans = index.scans
    NiftiSpotCheck(args.training_data_dir).run(scans[0])
    print(f"spot check passed for {scans[0].directory}")
    split = HoldoutSplitter(seed=args.seed).split(index.subjects)
    dataset_list = BraTSDatasetList(
        training_data_dir=args.training_data_dir,
        data_base_dir=args.data_base_dir,
        scans=scans,
        split=split,
    )
    payload = dataset_list.save(args.output)
    print(
        f"scans={len(scans)} subjects={len(index.subjects)} "
        f"training={len(payload['training'])} validation_cases={len(payload['validation'])} "
        f"validation_subjects={len(split.val)} seed={args.seed}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
