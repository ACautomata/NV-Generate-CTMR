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

"""Tests for scripts.create_brats_sidecars — the stage-2 BRATS sidecar generator (ticket T5, issue #21).

Fake data layout: a small dataset.json (two cases x four modalities) next to a
latent tree that mirrors the image paths, one NIfTI per entry with a
distinctive spacing written into its affine header.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_brats_dataset_json import SUFFIX_TO_MODALITY
from scripts.create_brats_sidecars import BratsSidecarGenerator, TrainingEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_DIRNAME = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
FAKE_CASES = ("BraTS-GLI-00000-000", "BraTS-GLI-00001-000")
LATENT_SPACING = (0.9375, 0.9375, 1.2109375)


@pytest.fixture
def dataset_entries() -> list[TrainingEntry]:
    """The dataset.json training entries for the fake cases, one per suffix."""
    entries = []
    for case in FAKE_CASES:
        for suffix, modality in sorted(SUFFIX_TO_MODALITY.items()):
            image = f"brats2023-gli/{TRAINING_DATA_DIRNAME}/{case}/{case}-{suffix}.nii.gz"
            entries.append(TrainingEntry(image=image, modality=modality))
    return entries


@pytest.fixture
def write_latent() -> Callable[[Path, str, tuple[float, float, float]], Path]:
    """Factory: save one latent NIfTI carrying ``spacing`` in its affine header; return its path."""

    def _write(embedding_base_dir: Path, relative_path: str, spacing: tuple[float, float, float]) -> Path:
        path = embedding_base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        affine = np.diag((*spacing, 1.0))
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), dtype=np.float32), affine=affine), path)
        return path

    return _write


@pytest.fixture
def embedding_base_dir(
    tmp_path: Path, write_latent: Callable[[Path, str, tuple[float, float, float]], Path], dataset_entries: list[TrainingEntry]
) -> Path:
    """An embedding base dir holding one latent per fake dataset.json entry, all with LATENT_SPACING."""
    base = tmp_path / "embeddings"
    for entry in dataset_entries:
        write_latent(base, entry.embedding_relative_path(), LATENT_SPACING)
    return base


@pytest.fixture
def dataset_json(tmp_path: Path, dataset_entries: list[TrainingEntry]) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"training": [asdict(entry) for entry in dataset_entries], "validation": []}))
    return path


@pytest.fixture
def build_generator(dataset_json: Path, embedding_base_dir: Path) -> Callable[[], BratsSidecarGenerator]:
    """Factory: a generator over the fake dataset.json and latent tree."""

    def _build() -> BratsSidecarGenerator:
        return BratsSidecarGenerator(dataset_json_path=dataset_json, embedding_base_dir=embedding_base_dir)

    return _build


class TestTrainingEntry:
    def test_embedding_relative_path_follows_training_code_convention(self) -> None:
        entry = TrainingEntry(image="brats2023-gli/data/case-t1n.nii.gz", modality="mri_t1n")
        assert entry.embedding_relative_path() == "brats2023-gli/data/case-t1n_emb.nii.gz"


class TestBratsSidecarGenerator:
    def test_writes_one_sidecar_per_training_entry(
        self, build_generator: Callable[[], BratsSidecarGenerator], dataset_entries: list[TrainingEntry]
    ) -> None:
        assert build_generator().write_sidecars() == len(dataset_entries)

    def test_sidecar_lies_next_to_latent_with_json_extension(
        self, build_generator: Callable[[], BratsSidecarGenerator], embedding_base_dir: Path, dataset_entries: list[TrainingEntry]
    ) -> None:
        build_generator().write_sidecars()

        # Independent literal: pins the next-to-latent + ".json" convention without the code under test.
        entry = dataset_entries[0]
        expected = embedding_base_dir / (entry.image.replace(".nii.gz", "_emb.nii.gz") + ".json")
        assert expected.is_file()

    def test_sidecar_records_header_spacing_and_modality(
        self, build_generator: Callable[[], BratsSidecarGenerator], embedding_base_dir: Path, dataset_entries: list[TrainingEntry]
    ) -> None:
        build_generator().write_sidecars()

        entry = dataset_entries[0]
        sidecar = json.loads((embedding_base_dir / entry.sidecar_relative_path()).read_text())
        assert set(sidecar) == {"spacing", "modality"}
        assert sidecar["spacing"] == pytest.approx(list(LATENT_SPACING))
        assert sidecar["modality"] == entry.modality

    def test_sidecar_spacing_tracks_replaced_latent(
        self,
        build_generator: Callable[[], BratsSidecarGenerator],
        embedding_base_dir: Path,
        dataset_entries: list[TrainingEntry],
        write_latent: Callable[[Path, str, tuple[float, float, float]], Path],
    ) -> None:
        odd_spacing = (2.0, 1.5, 3.25)
        entry = dataset_entries[-1]
        (embedding_base_dir / entry.embedding_relative_path()).unlink()
        write_latent(embedding_base_dir, entry.embedding_relative_path(), odd_spacing)
        build_generator().write_sidecars()

        sidecar = json.loads((embedding_base_dir / entry.sidecar_relative_path()).read_text())
        assert sidecar["spacing"] == pytest.approx(list(odd_spacing))

    def test_sidecar_modality_matches_dataset_json_entry_wise(
        self, build_generator: Callable[[], BratsSidecarGenerator], embedding_base_dir: Path, dataset_entries: list[TrainingEntry]
    ) -> None:
        build_generator().write_sidecars()

        for entry in dataset_entries:
            sidecar = json.loads((embedding_base_dir / entry.sidecar_relative_path()).read_text())
            assert sidecar["modality"] == entry.modality, entry.image

    def test_missing_latent_raises_and_writes_no_sidecars(
        self, build_generator: Callable[[], BratsSidecarGenerator], embedding_base_dir: Path, dataset_entries: list[TrainingEntry]
    ) -> None:
        (embedding_base_dir / dataset_entries[0].embedding_relative_path()).unlink()

        with pytest.raises(ValueError, match="missing"):
            build_generator().write_sidecars()

        assert not list(embedding_base_dir.rglob("*.json"))

    def test_rerun_overwrites_sidecars_to_match_updated_dataset_json(
        self, build_generator: Callable[[], BratsSidecarGenerator], embedding_base_dir: Path, dataset_json: Path, dataset_entries: list[TrainingEntry]
    ) -> None:
        build_generator().write_sidecars()
        payload = json.loads(dataset_json.read_text())
        payload["training"][0]["modality"] = "mri_t1ce"
        dataset_json.write_text(json.dumps(payload))
        build_generator().write_sidecars()

        sidecar = json.loads((embedding_base_dir / dataset_entries[0].sidecar_relative_path()).read_text())
        assert sidecar["modality"] == "mri_t1ce"


class TestCommandLine:
    def test_end_to_end_generation(self, embedding_base_dir: Path, dataset_json: Path, dataset_entries: list[TrainingEntry]) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.create_brats_sidecars",
                "--dataset-json",
                str(dataset_json),
                "--embedding-base-dir",
                str(embedding_base_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert f"sidecars={len(dataset_entries)}" in result.stdout
        sidecars = sorted(embedding_base_dir.rglob("*.json"))
        assert len(sidecars) == len(dataset_entries)
        assert all(json.loads(path.read_text())["spacing"] == pytest.approx(list(LATENT_SPACING)) for path in sidecars)
