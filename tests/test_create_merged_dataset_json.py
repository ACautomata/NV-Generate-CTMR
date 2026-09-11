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

"""Tests for scripts.create_merged_dataset_json — the merged dataset.json generator (ticket T7 #23)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_merged_dataset_json import AcceptedReplayRoster, BratsDatasetJson, MergedDatasetJson, main
from scripts.latent_sidecars import LatentEntry, LatentSidecarWriter
from scripts.mrrate_series import MrRateVariant

REPO_ROOT = Path(__file__).resolve().parents[1]

BRATS_IMAGES = [
    ("brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz", "mri_t1n"),
    ("brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz", "mri_t1ce"),
    ("brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t1n.nii.gz", "mri_t1n"),
    ("brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00001-000/BraTS-GLI-00001-000-t2w.nii.gz", "mri_t2w"),
]
BRATS_VALIDATION = ["BraTS-GLI-00002-000", "BraTS-GLI-00003-000"]

REPLAY_SERIES = [
    ("STUDYA", "t1w-raw-axi", "mri_t1"),
    ("STUDYB", "t2w-raw-sag", "mri_t2"),
    ("STUDYC", "flair-raw-cor", "mri_flair"),
    ("STUDYD", "swi-raw-axi", "mri_swi"),
    ("STUDYE", "mra-raw-cor", "mri_mra"),
]


def write_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)


@pytest.fixture
def brats_json(tmp_path: Path) -> Path:
    """A stage-1 BRATS dataset.json plus the source images its entries name."""
    payload = {
        "training": [{"image": image, "modality": modality} for image, modality in BRATS_IMAGES],
        "validation": BRATS_VALIDATION,
    }
    path = tmp_path / "brats" / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    for image, _modality in BRATS_IMAGES:
        write_nifti(tmp_path / "brats" / image)
    return path


@pytest.fixture
def data_base_dir(tmp_path: Path, brats_json: Path) -> Path:
    """The merged training root: BRATS images plus the replay tree the manifest rows name."""
    root = tmp_path / "merged"
    for image, _modality in BRATS_IMAGES:
        write_nifti(root / image)
    for study, series_id, _label in REPLAY_SERIES:
        variant = MrRateVariant(f"mri/batch00/{study}/img/{study}_{series_id}.nii.gz")
        write_nifti(root / variant.whole_brain_path)
        write_nifti(root / variant.skull_stripped_path)
    return root


@pytest.fixture
def embedding_base_dir(tmp_path: Path, brats_json: Path) -> Path:
    """The merged embedding root: a latent + sidecar for every entry the merge will produce."""
    entries = [LatentEntry(image=image, modality=modality) for image, modality in BRATS_IMAGES]
    entries += [
        entry
        for study, series_id, _label in REPLAY_SERIES
        for entry in MrRateVariant(f"mri/batch00/{study}/img/{study}_{series_id}.nii.gz").training_entries()
    ]
    root = tmp_path / "merged" / "embeddings"
    for entry in entries:
        write_nifti(root / entry.embedding_relative_path)
    writer = LatentSidecarWriter(root)
    writer.write(entries)
    return root


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("patient_uid", "study_uid", "series_id", "modality", "label", "split", "image_path")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def manifest_row(study: str, series_id: str, label: str, split: str = "Train") -> dict[str, str]:
    variant = MrRateVariant(f"mri/batch00/{study}/img/{study}_{series_id}.nii.gz")
    return {
        "patient_uid": f"PAT-{study}",
        "study_uid": study,
        "series_id": series_id,
        "modality": variant.modality,
        "label": label,
        "split": split,
        "image_path": variant.whole_brain_path,
    }


@pytest.fixture
def replay_manifest(tmp_path: Path) -> Path:
    """One accepted series per modality -- every cap is satisfied at any N >= 1."""
    rows = [manifest_row(study, series_id, label) for study, series_id, label in REPLAY_SERIES]
    return write_manifest(tmp_path / "manifests" / "replay_manifest_final.csv", rows)


@pytest.fixture
def merged(brats_json: Path, replay_manifest: Path, data_base_dir: Path, embedding_base_dir: Path) -> MergedDatasetJson:
    return MergedDatasetJson(
        brats=BratsDatasetJson(brats_json),
        replay=AcceptedReplayRoster(replay_manifest, n_per_label=300),
        data_base_dir=data_base_dir,
        embedding_base_dir=embedding_base_dir,
    )


class TestBratsDatasetJson:
    def test_training_entries_are_read_verbatim(self, brats_json: Path) -> None:
        brats = BratsDatasetJson(brats_json)
        assert brats.training == [{"image": image, "modality": modality} for image, modality in BRATS_IMAGES]

    def test_validation_roster_is_read_verbatim(self, brats_json: Path) -> None:
        assert BratsDatasetJson(brats_json).validation == BRATS_VALIDATION


class TestAcceptedReplayRoster:
    def test_dual_derivation_yields_two_entries_per_series(self, replay_manifest: Path) -> None:
        roster = AcceptedReplayRoster(replay_manifest, n_per_label=300)
        assert len(roster.entries()) == 2 * len(REPLAY_SERIES)

    def test_entries_are_grouped_by_modality_then_dually_derived(self, replay_manifest: Path) -> None:
        roster = AcceptedReplayRoster(replay_manifest, n_per_label=300)
        entries = roster.entries()
        labels = [entry.modality for entry in entries]
        assert labels == [
            "mri_t1",
            "mri_t1_skull_stripped",
            "mri_t2",
            "mri_t2_skull_stripped",
            "mri_flair",
            "mri_flair_skull_stripped",
            "mri_swi",
            "mri_swi_skull_stripped",
            "mri_mra",
            "mri_mra_skull_stripped",
        ]

    def test_per_modality_counts(self, replay_manifest: Path) -> None:
        roster = AcceptedReplayRoster(replay_manifest, n_per_label=300)
        assert roster.per_modality() == {"t1w": 1, "t2w": 1, "flair": 1, "swi": 1, "mra": 1}

    def test_a_manifest_row_exceeding_the_cap_is_refused(self, tmp_path: Path) -> None:
        rows = [manifest_row(f"STUDY{i}", "t1w-raw-axi", "mri_t1") for i in range(3)]
        manifest = write_manifest(tmp_path / "manifests" / "over.csv", rows)
        with pytest.raises(ValueError, match="cap"):
            AcceptedReplayRoster(manifest, n_per_label=2)

    def test_a_shortfall_below_the_cap_is_kept_and_reported(self, tmp_path: Path) -> None:
        rows = [manifest_row(f"STUDY{i}", "t1w-raw-axi", "mri_t1") for i in range(2)]
        rows += [manifest_row(study, series_id, label) for study, series_id, label in REPLAY_SERIES[1:]]
        manifest = write_manifest(tmp_path / "manifests" / "short.csv", rows)
        roster = AcceptedReplayRoster(manifest, n_per_label=3)
        # Every capped modality below its cap is a shortfall; mra is uncapped and never appears.
        assert roster.shortfalls() == {"t1w": (2, 3), "t2w": (1, 3), "flair": (1, 3), "swi": (1, 3)}

    def test_a_modality_missing_entirely_is_refused(self, tmp_path: Path) -> None:
        rows = [manifest_row("STUDYA", "t1w-raw-axi", "mri_t1")]
        manifest = write_manifest(tmp_path / "manifests" / "mraless.csv", rows)
        with pytest.raises(ValueError, match="mra"):
            AcceptedReplayRoster(manifest, n_per_label=300)

    def test_a_non_train_roster_cannot_be_merged(self, replay_manifest: Path, tmp_path: Path) -> None:
        rows = [manifest_row("STUDYA", "t1w-raw-axi", "mri_t1", split="Val")]
        manifest = write_manifest(tmp_path / "manifests" / "val.csv", rows)
        with pytest.raises(ValueError, match="Train"):
            AcceptedReplayRoster(manifest, n_per_label=300)


class TestMergedDatasetJson:
    def test_training_leads_with_brats_then_dually_derived_replay(self, merged: MergedDatasetJson) -> None:
        payload = merged.to_dict()
        assert payload["training"][: len(BRATS_IMAGES)] == [{"image": image, "modality": modality} for image, modality in BRATS_IMAGES]
        assert payload["training"][len(BRATS_IMAGES) :][:2] == [
            {"image": "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi.nii.gz", "modality": "mri_t1"},
            {"image": "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi_skull_stripped.nii.gz", "modality": "mri_t1_skull_stripped"},
        ]

    def test_validation_is_the_brats_roster_and_nothing_else(self, merged: MergedDatasetJson) -> None:
        assert merged.to_dict()["validation"] == BRATS_VALIDATION

    def test_entry_counts_follow_the_experiment_matrix(self, merged: MergedDatasetJson, brats_json: Path) -> None:
        summary = merged.to_dict()
        replay_entries = 2 * len(REPLAY_SERIES)
        assert len(summary["training"]) == 4 + replay_entries

    def test_saving_twice_reproduces_byte_for_byte(self, merged: MergedDatasetJson, tmp_path: Path) -> None:
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        merged.save(first)
        merged.save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_save_reports_the_summary(self, merged: MergedDatasetJson, tmp_path: Path) -> None:
        summary = merged.save(tmp_path / "merged.json")
        assert summary["training_entries"] == 4 + 2 * len(REPLAY_SERIES)
        assert summary["validation_cases"] == len(BRATS_VALIDATION)
        assert summary["counts"]["mri_t1n"] == 2

    def test_a_missing_latent_aborts_before_anything_is_written(
        self, brats_json: Path, replay_manifest: Path, data_base_dir: Path, embedding_base_dir: Path, tmp_path: Path
    ) -> None:
        (embedding_base_dir / "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi_emb.nii.gz").unlink()
        merged = MergedDatasetJson(
            brats=BratsDatasetJson(brats_json),
            replay=AcceptedReplayRoster(replay_manifest, n_per_label=300),
            data_base_dir=data_base_dir,
            embedding_base_dir=embedding_base_dir,
        )
        output = tmp_path / "merged.json"
        with pytest.raises(ValueError, match="latent"):
            merged.save(output)
        assert not output.exists()

    def test_a_sidecar_whose_modality_drifts_is_refused(
        self, brats_json: Path, replay_manifest: Path, data_base_dir: Path, embedding_base_dir: Path, tmp_path: Path
    ) -> None:
        sidecar = embedding_base_dir / "mri/batch00/STUDYB/img/STUDYB_t2w-raw-sag_emb.nii.gz.json"
        payload = json.loads(sidecar.read_text())
        payload["modality"] = "mri_t2w"
        sidecar.write_text(json.dumps(payload, indent=2) + "\n")
        merged = MergedDatasetJson(
            brats=BratsDatasetJson(brats_json),
            replay=AcceptedReplayRoster(replay_manifest, n_per_label=300),
            data_base_dir=data_base_dir,
            embedding_base_dir=embedding_base_dir,
        )
        with pytest.raises(ValueError, match="modality"):
            merged.save(tmp_path / "merged.json")

    def test_a_missing_source_volume_aborts(
        self, brats_json: Path, replay_manifest: Path, data_base_dir: Path, embedding_base_dir: Path, tmp_path: Path
    ) -> None:
        (data_base_dir / "mri/batch00/STUDYC/img/STUDYC_flair-raw-cor.nii.gz").unlink()
        merged = MergedDatasetJson(
            brats=BratsDatasetJson(brats_json),
            replay=AcceptedReplayRoster(replay_manifest, n_per_label=300),
            data_base_dir=data_base_dir,
            embedding_base_dir=embedding_base_dir,
        )
        with pytest.raises(ValueError, match="source"):
            merged.save(tmp_path / "merged.json")


class TestMain:
    def test_end_to_end_write(
        self, brats_json: Path, replay_manifest: Path, data_base_dir: Path, embedding_base_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        output = tmp_path / "out" / "dataset_rflow-mr-brain_N300.json"
        sys.argv = [
            "create_merged_dataset_json",
            "--brats-json",
            str(brats_json),
            "--replay-manifest",
            str(replay_manifest),
            "--n-per-label",
            "300",
            "--data-base-dir",
            str(data_base_dir),
            "--embedding-base-dir",
            str(embedding_base_dir),
            "--output",
            str(output),
        ]
        main()
        payload = json.loads(output.read_text())
        assert len(payload["training"]) == 4 + 2 * len(REPLAY_SERIES)
        assert payload["validation"] == BRATS_VALIDATION
        printed = capsys.readouterr().out
        assert f"training={4 + 2 * len(REPLAY_SERIES)}" in printed

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.create_merged_dataset_json", "--help"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        assert "--brats-json" in result.stdout
