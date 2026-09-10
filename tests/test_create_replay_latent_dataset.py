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

"""Tests for scripts.create_replay_latent_dataset — replay dataset.json + sidecars (ticket T6 #22)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_replay_latent_dataset import (
    DATASET_FILENAME_TEMPLATE,
    ReplayDatasetEntry,
    ReplayLatentDataset,
    ReplayLatentDatasetSet,
    ReplayTier,
    main,
)
from scripts.download_replay_subset import ManifestCandidate
from scripts.latent_sidecars import LatentEntry, LatentSidecarWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_SPACING = (1.5, 1.5, 2.0)
SERIES = [
    ("STUDYA", "t1w-raw-axi", "mri_t1"),
    ("STUDYB", "flair-raw-sag", "mri_flair"),
    ("STUDYC", "mra-raw-cor", "mri_mra"),
]


@pytest.fixture
def data_base_dir(tmp_path: Path) -> Path:
    """A data tree holding the three accepted series' whole-brain images (and their twins)."""
    for study, series_id, _label in SERIES:
        for suffix in ("", "_skull_stripped"):
            image = f"mri/batch00/{study}/img/{study}_{series_id}{suffix}.nii.gz"
            path = tmp_path / "data" / image
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), path)
    return tmp_path / "data"


@pytest.fixture
def accepted_manifest(tmp_path: Path) -> Path:
    """The accepted-series manifest the download stage writes for one tier."""
    path = tmp_path / "manifests" / "replay_manifest_N300_accepted.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient_uid", "study_uid", "series_id", "modality", "label", "split", "image_path"])
        for index, (study, series_id, label) in enumerate(SERIES):
            writer.writerow(
                [f"{index}", study, series_id, series_id.split("-")[0], label, "Train", f"mri/batch00/{study}/img/{study}_{series_id}.nii.gz"]
            )
    return path


@pytest.fixture
def embedding_base_dir(data_base_dir: Path) -> Path:
    """A latent tree mirroring the data tree, with one latent per training entry (both twins)."""
    base = data_base_dir.parent / "embeddings"
    for study, series_id, _label in SERIES:
        for suffix in ("", "_skull_stripped"):
            relative = f"mri/batch00/{study}/img/{study}_{series_id}{suffix}_emb.nii.gz"
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.diag((*LATENT_SPACING, 1.0))), path)
    return base


@pytest.fixture
def tier(accepted_manifest: Path) -> ReplayTier:
    return ReplayTier(name="N300", manifest_path=accepted_manifest)


@pytest.fixture
def dataset(tier: ReplayTier, data_base_dir: Path, embedding_base_dir: Path) -> ReplayLatentDataset:
    return ReplayLatentDataset(tier=tier, data_base_dir=data_base_dir, sidecar_writer=LatentSidecarWriter(embedding_base_dir))


class TestReplayTier:
    def test_dataset_filename_is_distinct_from_the_merged_training_file(self) -> None:
        """The merged BRATS+replay dataset the env configs read is dataset_rflow-mr-brain_<N>.json (T7)."""
        tier = ReplayTier(name="N500", manifest_path=Path("m.csv"))
        assert tier.dataset_filename == "dataset_replay_rflow-mr-brain_N500.json"
        assert tier.dataset_filename == DATASET_FILENAME_TEMPLATE.format(tier="N500")
        assert tier.dataset_filename != "dataset_rflow-mr-brain_N500.json"

    def test_source_filename_names_the_encoder_input(self) -> None:
        assert ReplayTier(name="N300", manifest_path=Path("m.csv")).source_filename == "replay_source_N300.json"

    def test_parses_a_cli_specification(self) -> None:
        tier = ReplayTier.parse("N1000=/tmp/manifests/replay_manifest_N1000_accepted.csv")
        assert tier.name == "N1000"
        assert tier.manifest_path == Path("/tmp/manifests/replay_manifest_N1000_accepted.csv")

    def test_rejects_an_unparseable_specification(self) -> None:
        with pytest.raises(ValueError, match="unparseable tier"):
            ReplayTier.parse("N300")


class TestReplayDatasetEntry:
    def test_one_series_becomes_two_training_entries_whole_brain_first(self) -> None:
        candidate = ManifestCandidate.from_row(
            {
                "patient_uid": "1",
                "study_uid": "STUDYA",
                "series_id": "t1w-raw-axi",
                "modality": "t1w",
                "label": "mri_t1",
                "split": "Train",
                "image_path": "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi.nii.gz",
            }
        )
        entries = ReplayDatasetEntry(candidate).training_entries()
        assert entries == [
            {"image": "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi.nii.gz", "modality": "mri_t1"},
            {
                "image": "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi_skull_stripped.nii.gz",
                "modality": "mri_t1_skull_stripped",
            },
        ]

    def test_every_modality_pairs_with_its_own_skull_stripped_label(self) -> None:
        for modality, whole_brain, skull_stripped in (
            ("t2w", "mri_t2", "mri_t2_skull_stripped"),
            ("swi", "mri_swi", "mri_swi_skull_stripped"),
            ("mra", "mri_mra", "mri_mra_skull_stripped"),
        ):
            candidate = ManifestCandidate.from_row(
                {
                    "patient_uid": "1",
                    "study_uid": "STUDYA",
                    "series_id": f"{modality}-raw-axi",
                    "modality": modality,
                    "label": whole_brain,
                    "split": "Train",
                    "image_path": f"mri/batch00/STUDYA/img/STUDYA_{modality}-raw-axi.nii.gz",
                }
            )
            entries = ReplayDatasetEntry(candidate).training_entries()
            assert [record["modality"] for record in entries] == [whole_brain, skull_stripped]

    def test_latent_entries_match_the_training_entries(self, dataset: ReplayLatentDataset) -> None:
        candidate = dataset.candidates()[0]
        dataset_entry = ReplayDatasetEntry(candidate)
        assert [(item.image, item.modality) for item in dataset_entry.latent_entries()] == [
            (record["image"], record["modality"]) for record in dataset_entry.training_entries()
        ]


class TestReplayLatentDataset:
    def test_writes_two_entries_per_accepted_series(self, dataset: ReplayLatentDataset, tmp_path: Path) -> None:
        summary = dataset.write(tmp_path)
        assert summary["training_entries"] == len(SERIES) * 2

        payload = json.loads((tmp_path / dataset._tier.dataset_filename).read_text())
        assert set(payload) == {"training"}
        labels = [record["modality"] for record in payload["training"]]
        assert labels == [
            "mri_t1",
            "mri_t1_skull_stripped",
            "mri_flair",
            "mri_flair_skull_stripped",
            "mri_mra",
            "mri_mra_skull_stripped",
        ]

    def test_entries_carry_exactly_the_training_fields(self, dataset: ReplayLatentDataset, tmp_path: Path) -> None:
        """The training code reads ``image`` and ``modality`` only; anything else would be dead weight."""
        dataset.write(tmp_path)
        payload = json.loads((tmp_path / dataset._tier.dataset_filename).read_text())
        assert all(set(record) == {"image", "modality"} for record in payload["training"])

    def test_sidecar_lands_next_to_the_latent_with_the_label(self, dataset: ReplayLatentDataset, embedding_base_dir: Path, tmp_path: Path) -> None:
        dataset.write(tmp_path)
        sidecar_path = embedding_base_dir / "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi_skull_stripped_emb.nii.gz.json"
        sidecar = json.loads(sidecar_path.read_text())
        assert sidecar["modality"] == "mri_t1_skull_stripped"
        assert sidecar["spacing"] == pytest.approx(list(LATENT_SPACING))

    def test_sidecar_count_equals_the_training_entry_count(self, dataset: ReplayLatentDataset, embedding_base_dir: Path, tmp_path: Path) -> None:
        summary = dataset.write(tmp_path)
        assert summary["sidecars"] == summary["training_entries"] == 6
        assert [path.name for path in sorted(embedding_base_dir.rglob("*.json"))].count("STUDYA_t1w-raw-axi_emb.nii.gz.json") == 1

    def test_a_missing_latent_aborts_before_anything_is_written(self, dataset: ReplayLatentDataset, embedding_base_dir: Path, tmp_path: Path) -> None:
        (embedding_base_dir / "mri/batch00/STUDYB/img/STUDYB_flair-raw-sag_emb.nii.gz").unlink()
        with pytest.raises(ValueError, match="latents missing"):
            dataset.write(tmp_path)
        assert not list(tmp_path.glob("dataset_*.json"))

    def test_a_missing_source_volume_aborts_with_the_path(self, dataset: ReplayLatentDataset, data_base_dir: Path, tmp_path: Path) -> None:
        """A manifest that does not line up with the downloaded tree must fail loudly, not shrink the set."""
        (data_base_dir / "mri/batch00/STUDYC/img/STUDYC_mra-raw-cor.nii.gz").unlink()
        with pytest.raises(FileNotFoundError, match="source volumes missing"):
            dataset.write(tmp_path)

    def test_latent_paths_follow_the_training_code_replacement(self, dataset: ReplayLatentDataset) -> None:
        entry = LatentEntry(image=dataset.candidates()[0].image_path, modality="mri_t1")
        assert entry.embedding_relative_path.endswith("_emb.nii.gz")
        assert entry.sidecar_relative_path.endswith("_emb.nii.gz.json")

    def test_a_manifest_label_that_contradicts_the_series_id_is_refused(self, dataset: ReplayLatentDataset, accepted_manifest: Path) -> None:
        """The sidecar's modality must equal the manifest's label, so the mismatch is refused here too.

        The downloader checks this before it starts, but the roster it accepts is not the file
        this stage reads -- finalize re-reads a refilled roster, so the guarantee has to hold
        at the point the artifact is actually written.
        """
        accepted_manifest.write_text(accepted_manifest.read_text().replace(",t1w-raw-axi,t1w,mri_t1,", ",t1w-raw-axi,t1w,mri_flair,"))

        with pytest.raises(ValueError, match="contradicts the series id"):
            dataset.candidates()


class TestReplayLatentDatasetSet:
    def test_nested_tiers_share_one_latent_tree(self, data_base_dir: Path, embedding_base_dir: Path, tmp_path: Path) -> None:
        """A smaller tier is a prefix of the larger one, so both encode the same files."""
        small = tmp_path / "small.csv"
        large = tmp_path / "large.csv"
        header = ["patient_uid", "study_uid", "series_id", "modality", "label", "split", "image_path"]

        def rows(subset: list[tuple[str, str, str]]) -> list[list[str]]:
            return [
                [f"{index}", study, series_id, series_id.split("-")[0], label, "Train", f"mri/batch00/{study}/img/{study}_{series_id}.nii.gz"]
                for index, (study, series_id, label) in enumerate(subset)
            ]

        for path, subset in ((small, SERIES[:1]), (large, SERIES)):
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(rows(subset))

        dataset_set = ReplayLatentDatasetSet(
            tiers=[ReplayTier("N300", small), ReplayTier("N1000", large)],
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(embedding_base_dir),
        )
        summaries = dataset_set.write_all(tmp_path, stage="finalize")
        assert [summary["training_entries"] for summary in summaries] == [2, 6]

        small_payload = json.loads((tmp_path / "dataset_replay_rflow-mr-brain_N300.json").read_text())
        large_payload = json.loads((tmp_path / "dataset_replay_rflow-mr-brain_N1000.json").read_text())
        assert large_payload["training"][:2] == small_payload["training"]
        # Both tiers point into the same embedding tree: no tier-specific latent root.
        assert all("mri/batch00/STUDYA/" in record["image"] for record in small_payload["training"])


class TestEncodeStage:
    """The encoder's input must exist before any latent does -- finalize cannot supply it."""

    def test_source_list_is_written_without_latents(self, data_base_dir: Path, accepted_manifest: Path, tmp_path: Path) -> None:
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(tmp_path / "empty_embeddings"),
        )
        summary = dataset.write_source_list(tmp_path)

        assert summary["source_entries"] == len(SERIES) * 2
        payload = json.loads((tmp_path / "replay_source_N300.json").read_text())
        assert set(payload) == {"training"}
        assert len(payload["training"]) == len(SERIES) * 2

    def test_source_list_carries_both_halves_of_the_dual_derivation(self, data_base_dir: Path, accepted_manifest: Path, tmp_path: Path) -> None:
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(tmp_path / "empty_embeddings"),
        )
        dataset.write_source_list(tmp_path)

        payload = json.loads((tmp_path / "replay_source_N300.json").read_text())
        assert [record["modality"] for record in payload["training"]] == [
            "mri_t1",
            "mri_t1_skull_stripped",
            "mri_flair",
            "mri_flair_skull_stripped",
            "mri_mra",
            "mri_mra_skull_stripped",
        ]
        assert any(record["image"].endswith("_skull_stripped.nii.gz") for record in payload["training"])

    def test_source_list_omits_volumes_that_have_not_arrived_yet(self, data_base_dir: Path, accepted_manifest: Path, tmp_path: Path) -> None:
        """The encoder aborts on an unreadable path, so a partial download yields a partial list."""
        (data_base_dir / "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi.nii.gz").unlink()
        (data_base_dir / "mri/batch00/STUDYA/img/STUDYA_t1w-raw-axi_skull_stripped.nii.gz").unlink()
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(tmp_path / "empty_embeddings"),
        )
        summary = dataset.write_source_list(tmp_path)

        assert summary["source_entries"] == len(SERIES) * 2 - 2
        assert summary["not_yet_downloaded"] == 2
        payload = json.loads((tmp_path / "replay_source_N300.json").read_text())
        assert all("STUDYA_t1w-raw-axi" not in record["image"] for record in payload["training"])

    def test_already_encoded_entries_are_left_out(
        self, data_base_dir: Path, embedding_base_dir: Path, accepted_manifest: Path, tmp_path: Path
    ) -> None:
        """The encoder probes a volume's shape before checking for its latent, so repeats cost real reads."""
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(embedding_base_dir),
        )
        summary = dataset.write_source_list(tmp_path)

        # Every fixture entry already has a latent: there is no outstanding work.
        assert summary["source_entries"] == 0
        assert summary["not_yet_downloaded"] == 0
        assert summary["output"] is None

    def test_caught_up_reports_what_is_still_downloading(
        self, data_base_dir: Path, embedding_base_dir: Path, accepted_manifest: Path, tmp_path: Path
    ) -> None:
        """Caught up and finished look identical to the file system; a driver loop must tell them apart."""
        for suffix in ("", "_skull_stripped"):
            (data_base_dir / f"mri/batch00/STUDYC/img/STUDYC_mra-raw-cor{suffix}.nii.gz").unlink()
            (embedding_base_dir / f"mri/batch00/STUDYC/img/STUDYC_mra-raw-cor{suffix}_emb.nii.gz").unlink()
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(embedding_base_dir),
        )
        summary = dataset.write_source_list(tmp_path)

        # STUDYC is absent from both trees, so nothing is outstanding -- but two volumes are
        # still on their way, and a driver that stopped here would idle the GPUs until the
        # download finished.
        assert summary["source_entries"] == 0
        assert summary["not_yet_downloaded"] == 2

    def test_only_the_outstanding_half_is_listed(
        self, data_base_dir: Path, embedding_base_dir: Path, accepted_manifest: Path, tmp_path: Path
    ) -> None:
        """The skull-stripped twins are not encoded yet, so they are what remains."""
        for study, series_id, _label in SERIES:
            (embedding_base_dir / f"mri/batch00/{study}/img/{study}_{series_id}_emb.nii.gz").unlink()
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(embedding_base_dir),
        )
        summary = dataset.write_source_list(tmp_path)

        assert summary["source_entries"] == len(SERIES)
        payload = json.loads((tmp_path / "replay_source_N300.json").read_text())
        assert [record["modality"] for record in payload["training"]] == ["mri_t1", "mri_flair", "mri_mra"]

    def test_finalize_would_refuse_the_same_tree(self, data_base_dir: Path, accepted_manifest: Path, tmp_path: Path) -> None:
        """The two stages have opposite preconditions; this is why both exist."""
        dataset = ReplayLatentDataset(
            tier=ReplayTier("N300", accepted_manifest),
            data_base_dir=data_base_dir,
            sidecar_writer=LatentSidecarWriter(tmp_path / "empty_embeddings"),
        )
        with pytest.raises(ValueError, match="latents missing"):
            dataset.write(tmp_path)


class TestCommandLine:
    def test_end_to_end_finalize_writes_every_tier(
        self, data_base_dir: Path, embedding_base_dir: Path, accepted_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "create_replay_latent_dataset",
                "--stage",
                "finalize",
                "--data-base-dir",
                str(data_base_dir),
                "--embedding-base-dir",
                str(embedding_base_dir),
                "--tier",
                f"N300={accepted_manifest}",
                "--output-dir",
                str(tmp_path),
            ],
        )
        main()

        payload = json.loads((tmp_path / "dataset_replay_rflow-mr-brain_N300.json").read_text())
        assert len(payload["training"]) == 6

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.create_replay_latent_dataset", "--help"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--tier" in result.stdout
        assert "--stage" in result.stdout
