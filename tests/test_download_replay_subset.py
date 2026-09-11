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

"""Tests for scripts.download_replay_subset — the manifest-driven replay download (ticket T6 #22).

Fake data is the real thing in miniature: study zips whose members follow the official
layout, holding genuine NIfTI image/mask pairs whose mask coverage decides the spine
filter's verdict.  A recording zip source stands in for HuggingFace, so the tests observe
exactly which zips were fetched, when, and how often.
"""

import csv
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.download_replay_subset import (
    MANIFEST_COLUMNS,
    VERDICT_COLUMNS,
    HuggingFaceZipSource,
    ManifestCandidate,
    ReplayDownloader,
    SeriesVerdict,
    VerdictLog,
)
from scripts.spine_filter import SpineFilter, SpineFilterThresholds

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_SHAPE = (64, 64, 64)
AFFINE = np.diag((1.0, 1.0, 1.0, 1.0))


@pytest.fixture
def volume_writer(tmp_path: Path) -> Callable[[str, float], Path]:
    """Write one NIfTI carrying a brain-mask-shaped blob covering ``mask_ratio`` of the volume."""

    def build(name: str, mask_ratio: float) -> Path:
        path = tmp_path / name
        data = np.zeros(BRAIN_SHAPE, dtype=np.uint8)
        if mask_ratio > 0:
            filled = int(np.prod(BRAIN_SHAPE) * mask_ratio)
            data.reshape(-1)[:filled] = 1
        nib.save(nib.Nifti1Image(data, AFFINE), path)
        return path

    return build


class RecordingZipSource:
    """A ``ZipSource`` serving real study zips out of ``store``; records every fetch call.

    Mimics ``snapshot_download``: the requested entries are materialized under a fresh
    root with their repository-relative layout, and that root is returned.
    """

    def __init__(self, store: dict[str, Path], root: Path) -> None:
        self._store = store
        self._root = root
        self.fetches: list[list[str]] = []
        self._fetch_index = 0
        self._materialized: list[Path] = []

    def fetch(self, repo_paths: list[str]) -> Path:
        """Copy the requested zips into a fresh root laid out like the repository, and return it."""
        self.fetches.append(list(repo_paths))
        self._fetch_index += 1
        download_root = self._root / f"fetch{self._fetch_index}"
        for repo_path in repo_paths:
            destination = download_root / repo_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self._store[repo_path], destination)
            self._materialized.append(destination)
        return download_root

    @property
    def fetched_paths(self) -> list[str]:
        return [path for batch in self.fetches for path in batch]

    @property
    def materialized(self) -> list[Path]:
        """Every zip copy handed out, for asserting they were deleted after processing."""
        return list(self._materialized)


@pytest.fixture
def replay_data(tmp_path: Path, volume_writer: Callable[[str, float], Path]) -> dict[str, Path]:
    """Build one study zip per study; map repo path -> stored zip so tests can assert deletion."""

    def study(study_uid: str, series: list[tuple[str, float]]) -> Path:
        """Add one study zip to the store; ``series`` is ``(series_id, mask_ratio)``."""
        store_root = tmp_path / "store"
        store_root.mkdir(parents=True, exist_ok=True)
        archive = store_root / f"{study_uid}.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for series_id, mask_ratio in series:
                image = volume_writer(f"{study_uid}_{series_id}.nii.gz", mask_ratio=0.30)
                mask = volume_writer(f"{study_uid}_{series_id}_brain-mask.nii.gz", mask_ratio=mask_ratio)
                zipped.write(image, f"{study_uid}/img/{image.name}")
                zipped.write(mask, f"{study_uid}/seg/{mask.name}")
        return archive

    return {
        "mri/batch00/STUDYB01.zip": study("STUDYB01", [("t1w-raw-axi", 0.30)]),
        "mri/batch00/STUDYB02.zip": study("STUDYB02", [("flair-raw-sag", 0.30), ("t1w-raw-axi", 0.005)]),
        "mri/batch02/STUDYB03.zip": study("STUDYB03", [("mra-raw-cor", 0.30)]),
    }


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """A manifest naming four series: three brain-like, one spine-like (mask ratio 0.5%)."""
    rows = [
        ("1", "STUDYB01", "t1w-raw-axi", "t1w", "mri_t1", "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz"),
        ("2", "STUDYB02", "flair-raw-sag", "flair", "mri_flair", "mri/batch00/STUDYB02/img/STUDYB02_flair-raw-sag.nii.gz"),
        ("2", "STUDYB02", "t1w-raw-axi", "t1w", "mri_t1", "mri/batch00/STUDYB02/img/STUDYB02_t1w-raw-axi.nii.gz"),
        ("3", "STUDYB03", "mra-raw-cor", "mra", "mri_mra", "mri/batch02/STUDYB03/img/STUDYB03_mra-raw-cor.nii.gz"),
    ]
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(MANIFEST_COLUMNS)
        for patient, study_uid, series_id, modality, label, image_path in rows:
            writer.writerow([patient, study_uid, series_id, modality, label, "Train", image_path])
    return path


@pytest.fixture
def downloader_factory(
    manifest_path: Path,
    tmp_path: Path,
    replay_data: dict[str, Path],
) -> Callable[..., tuple[ReplayDownloader, RecordingZipSource, VerdictLog]]:
    """Factory building a downloader over the fake store."""

    def build(workers: int = 16, verdicts_name: str = "verdicts.csv") -> tuple[ReplayDownloader, RecordingZipSource, VerdictLog]:
        source = RecordingZipSource(replay_data, tmp_path / "downloads")
        log = VerdictLog(tmp_path / verdicts_name)
        downloader = ReplayDownloader(
            manifest_path=manifest_path,
            download_dir=tmp_path / "data",
            source=source,
            verdict_log=log,
            workers=workers,
        )
        return downloader, source, log

    return build


class TestManifestCandidate:
    def test_row_round_trips_through_the_manifest_columns(self) -> None:
        row = {
            "patient_uid": "1",
            "study_uid": "STUDYB01",
            "series_id": "t1w-raw-axi",
            "modality": "t1w",
            "label": "mri_t1",
            "split": "Train",
            "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
        }
        candidate = ManifestCandidate.from_row(row)
        assert candidate.batch == "batch00"
        assert candidate.repo_zip_path == "mri/batch00/STUDYB01.zip"
        assert candidate.key == ("STUDYB01", "t1w-raw-axi")
        assert candidate.variant.mask_path == "mri/batch00/STUDYB01/seg/STUDYB01_t1w-raw-axi_brain-mask.nii.gz"

    def test_zip_member_strips_the_batch_prefix(self) -> None:
        candidate = ManifestCandidate.from_row(
            {
                "patient_uid": "1",
                "study_uid": "STUDYB01",
                "series_id": "t1w-raw-axi",
                "modality": "t1w",
                "label": "mri_t1",
                "split": "Train",
                "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
            }
        )
        assert candidate.zip_member(candidate.image_path) == "STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz"

    def test_a_label_that_contradicts_the_series_id_is_refused_at_construction(self) -> None:
        """The contract lives in the factory, so no consumer can forget to check it."""
        with pytest.raises(ValueError, match="contradicts the series id"):
            ManifestCandidate.from_row(
                {
                    "patient_uid": "1",
                    "study_uid": "STUDYB01",
                    "series_id": "t1w-raw-axi",
                    "modality": "t1w",
                    "label": "mri_flair",
                    "split": "Train",
                    "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
                }
            )


class TestSeriesVerdict:
    def test_accept_row_carries_the_manifest_columns_and_measurements(self) -> None:
        candidate = ManifestCandidate.from_row(
            {
                "patient_uid": "1",
                "study_uid": "STUDYB01",
                "series_id": "t1w-raw-axi",
                "modality": "t1w",
                "label": "mri_t1",
                "split": "Train",
                "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
            }
        )
        verdict = SeriesVerdict(status="accept", mask_voxel_ratio=0.3, fov_mm=(64.0, 64.0, 64.0), reasons=())
        row = verdict.to_row(candidate)
        assert set(row) == set(VERDICT_COLUMNS)
        assert row["verdict"] == "accept"
        assert row["study_uid"] == "STUDYB01"
        assert row["image_path"] == candidate.image_path
        assert row["mask_voxel_ratio"] == "0.300000"
        assert row["fov_mm"] == "64.00 64.00 64.00"
        assert row["reasons"] == ""

    def test_reject_row_records_the_failed_conditions(self) -> None:
        candidate = ManifestCandidate.from_row(
            {
                "patient_uid": "1",
                "study_uid": "STUDYB01",
                "series_id": "t1w-raw-axi",
                "modality": "t1w",
                "label": "mri_t1",
                "split": "Train",
                "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
            }
        )
        verdict = SeriesVerdict(status="reject", mask_voxel_ratio=0.005, fov_mm=(64.0, 64.0, 64.0), reasons=("too small", "too long"))
        row = verdict.to_row(candidate)
        assert row["verdict"] == "reject"
        assert row["reasons"] == "too small; too long"

    def test_unusable_leaves_the_measurements_empty_rather_than_zeroed(self) -> None:
        """An off-grid pair was never measured; 0.0 % and 0 mm would read as measurements."""
        candidate = ManifestCandidate.from_row(
            {
                "patient_uid": "1",
                "study_uid": "STUDYB01",
                "series_id": "t1w-raw-axi",
                "modality": "t1w",
                "label": "mri_t1",
                "split": "Train",
                "image_path": "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz",
            }
        )
        row = SeriesVerdict.unusable("differ in voxel grid").to_row(candidate)
        assert row["verdict"] == "reject"
        assert row["fov_mm"] == ""
        assert row["mask_voxel_ratio"] != "0.000000"
        assert row["reasons"] == "differ in voxel grid"


class TestReplayDownloader:
    def test_accepted_series_land_with_their_skull_stripped_twin(self, downloader_factory) -> None:
        downloader, _, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        data = downloader._download_dir
        assert (data / "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz").is_file()
        assert (data / "mri/batch00/STUDYB01/seg/STUDYB01_t1w-raw-axi_brain-mask.nii.gz").is_file()
        assert (data / "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi_skull_stripped.nii.gz").is_file()

    def test_skull_stripped_twin_is_the_masked_image(self, downloader_factory) -> None:
        downloader, _, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        data = downloader._download_dir
        image = np.asanyarray(nib.load(str(data / "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi.nii.gz")).dataobj)
        mask = np.asanyarray(nib.load(str(data / "mri/batch00/STUDYB01/seg/STUDYB01_t1w-raw-axi_brain-mask.nii.gz")).dataobj) != 0
        stripped = np.asanyarray(nib.load(str(data / "mri/batch00/STUDYB01/img/STUDYB01_t1w-raw-axi_skull_stripped.nii.gz")).dataobj)
        assert np.array_equal(stripped, image * mask.astype(image.dtype))

    def test_spine_like_series_is_rejected_and_leaves_nothing_behind(self, downloader_factory) -> None:
        downloader, _, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        data = downloader._download_dir
        assert not (data / "mri/batch00/STUDYB02/img/STUDYB02_t1w-raw-axi.nii.gz").exists()
        assert not (data / "mri/batch00/STUDYB02/img/STUDYB02_t1w-raw-axi_skull_stripped.nii.gz").exists()

    def test_the_other_series_of_a_rejected_study_survives(self, downloader_factory) -> None:
        """STUDYB02 lists two series; only the spine-like one is dropped, and its zip stays until both are done."""
        downloader, _, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        data = downloader._download_dir
        assert (data / "mri/batch00/STUDYB02/img/STUDYB02_flair-raw-sag.nii.gz").is_file()
        assert (data / "mri/batch00/STUDYB02/img/STUDYB02_flair-raw-sag_skull_stripped.nii.gz").is_file()

    def test_each_study_zip_is_fetched_exactly_once(self, downloader_factory) -> None:
        downloader, source, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        assert sorted(source.fetched_paths) == ["mri/batch00/STUDYB01.zip", "mri/batch00/STUDYB02.zip", "mri/batch02/STUDYB03.zip"]
        assert len(source.fetched_paths) == len(set(source.fetched_paths))

    def test_a_row_whose_label_contradicts_its_series_id_fails_fast(self, manifest_path: Path, tmp_path: Path, replay_data: dict[str, Path]) -> None:
        """A contradiction would mislabel the training set (or KeyError hours in), so it is refused up front."""
        rows = manifest_path.read_text().replace(",t1w,mri_t1,", ",t1w,mri_flair,")
        manifest_path.write_text(rows)
        downloader = ReplayDownloader(
            manifest_path=manifest_path,
            download_dir=tmp_path / "data",
            source=RecordingZipSource(replay_data, tmp_path / "downloads"),
            verdict_log=VerdictLog(tmp_path / "verdicts.csv"),
        )

        with pytest.raises(ValueError, match="contradicts the series id"):
            downloader.candidates()

    def test_a_known_modality_passes_the_check(self, downloader_factory) -> None:
        downloader, _, _ = downloader_factory()
        assert len(downloader.candidates()) == 4

    def test_verdict_log_records_one_row_per_candidate_in_manifest_order(self, downloader_factory) -> None:
        downloader, _, log = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        rows = log.read_rows()
        assert list(rows[0]) == list(VERDICT_COLUMNS)
        assert [(row["study_uid"], row["series_id"]) for row in rows] == [
            ("STUDYB01", "t1w-raw-axi"),
            ("STUDYB02", "flair-raw-sag"),
            ("STUDYB02", "t1w-raw-axi"),
            ("STUDYB03", "mra-raw-cor"),
        ]
        assert [row["verdict"] for row in rows] == ["accept", "accept", "reject", "accept"]

    def test_accepted_manifest_is_a_manifest_the_top_up_can_consume(self, downloader_factory, tmp_path: Path) -> None:
        downloader, _, _ = downloader_factory()
        accepted_csv = tmp_path / "accepted.csv"
        rejected_csv = tmp_path / "rejected.csv"
        summary = downloader.run(accepted_csv=accepted_csv, rejected_csv=rejected_csv)

        assert summary == {"candidates": 4, "accepted": 3, "rejected": 1, "failed": 0}
        with accepted_csv.open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert list(rows[0]) == list(MANIFEST_COLUMNS)
        assert [(row["study_uid"], row["series_id"]) for row in rows] == [
            ("STUDYB01", "t1w-raw-axi"),
            ("STUDYB02", "flair-raw-sag"),
            ("STUDYB03", "mra-raw-cor"),
        ]
        with rejected_csv.open(newline="") as file:
            rejected = list(csv.DictReader(file))
        assert [(row["study_uid"], row["series_id"]) for row in rejected] == [("STUDYB02", "t1w-raw-axi")]

    def test_rerun_downloads_nothing_and_keeps_the_verdicts(self, downloader_factory) -> None:
        downloader, source, log = downloader_factory()
        accepted_csv = downloader._download_dir / "accepted.csv"
        rejected_csv = downloader._download_dir / "rejected.csv"
        downloader.run(accepted_csv=accepted_csv, rejected_csv=rejected_csv)
        before = log.read_rows()

        downloader, resumed_source, resumed_log = downloader_factory(verdicts_name="verdicts.csv")
        summary = downloader.run(accepted_csv=accepted_csv, rejected_csv=rejected_csv)

        assert resumed_source.fetches == []
        assert summary["accepted"] == 3
        assert resumed_log.read_rows() == before

    def test_interrupted_run_resumes_the_missing_studies_only(self, downloader_factory) -> None:
        """A crash mid-manifest leaves verdicts behind; the rerun picks up exactly the rest."""
        downloader, first_source, log = downloader_factory(workers=1)
        accepted_csv = downloader._download_dir / "accepted.csv"
        rejected_csv = downloader._download_dir / "rejected.csv"

        # First pass processes only the first study, as a crash after chunk one would.
        chunk, root = next(iter(downloader._chunks(downloader.candidates())))
        downloader._process_study(downloader._group_by_study(chunk)[0], root)
        assert len(log.read_rows()) == 1

        downloader, resumed_source, _ = downloader_factory(workers=1)
        downloader.run(accepted_csv=accepted_csv, rejected_csv=rejected_csv)

        refetched = [path for path in resumed_source.fetched_paths if "STUDYB01" not in path]
        assert sorted(refetched) == ["mri/batch00/STUDYB02.zip", "mri/batch02/STUDYB03.zip"]
        assert len(log.read_rows()) == 4

    def test_study_zip_is_deleted_after_processing(self, downloader_factory) -> None:
        downloader, source, _ = downloader_factory()
        downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        assert all(not path.exists() for path in source.materialized)
        assert not (downloader._download_dir / "staging").exists()

    def test_unreadable_zip_is_reported_as_failed_and_does_not_stop_the_run(self, downloader_factory, replay_data: dict[str, Path]) -> None:
        replay_data["mri/batch00/STUDYB01.zip"].write_bytes(b"not a zip")
        downloader, _, log = downloader_factory()
        summary = downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

        assert summary["failed"] == 1
        assert summary["accepted"] == 2
        assert len(log.read_rows()) == 3


class TestOffGridPair:
    """An image/mask pair that cannot be combined must be rejected, not left to wedge the run."""

    @pytest.fixture
    def off_grid_downloader(self, tmp_path: Path) -> tuple[ReplayDownloader, VerdictLog]:
        """One study whose mask sits on a different voxel grid than its image."""
        image = tmp_path / "OFFGRID_t1w-raw-axi.nii.gz"
        mask = tmp_path / "OFFGRID_t1w-raw-axi_brain-mask.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.uint8), AFFINE), image)
        nib.save(nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.uint8), AFFINE), mask)

        archive = tmp_path / "store" / "OFFGRID.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(image, "OFFGRID/img/OFFGRID_t1w-raw-axi.nii.gz")
            zipped.write(mask, "OFFGRID/seg/OFFGRID_t1w-raw-axi_brain-mask.nii.gz")

        manifest = tmp_path / "offgrid_manifest.csv"
        with manifest.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(MANIFEST_COLUMNS)
            writer.writerow(["1", "OFFGRID", "t1w-raw-axi", "t1w", "mri_t1", "Train", "mri/batch00/OFFGRID/img/OFFGRID_t1w-raw-axi.nii.gz"])

        log = VerdictLog(tmp_path / "offgrid_verdicts.csv")
        downloader = ReplayDownloader(
            manifest_path=manifest,
            download_dir=tmp_path / "data",
            source=RecordingZipSource({"mri/batch00/OFFGRID.zip": archive}, tmp_path / "downloads"),
            verdict_log=log,
        )
        return downloader, log

    @staticmethod
    def run(downloader: ReplayDownloader) -> dict:
        return downloader.run(accepted_csv=downloader._download_dir / "accepted.csv", rejected_csv=downloader._download_dir / "rejected.csv")

    def test_the_pair_is_recorded_as_a_reject_rather_than_escaping(self, off_grid_downloader) -> None:
        """Letting the ValueError out would kill an hours-long run on one unusable series."""
        downloader, log = off_grid_downloader
        summary = self.run(downloader)

        assert summary["failed"] == 0
        assert summary["accepted"] == 0
        assert summary["rejected"] == 1
        row = log.read_rows()[0]
        assert row["verdict"] == "reject"
        assert "differ in voxel grid" in row["reasons"]

    def test_a_rerun_does_not_pick_the_series_up_again(self, off_grid_downloader) -> None:
        """The whole point of recording it: an unrecorded series would be retried forever."""
        downloader, log = off_grid_downloader
        self.run(downloader)
        assert log.completed() == {("OFFGRID", "t1w-raw-axi")}

        self.run(downloader)
        assert len(log.read_rows()) == 1

    def test_the_unusable_pair_is_dropped_from_the_data_tree(self, off_grid_downloader) -> None:
        """Nothing downstream should be able to pick up a volume that was rejected."""
        downloader, _ = off_grid_downloader
        self.run(downloader)

        assert not (downloader._download_dir / "mri/batch00/OFFGRID/img/OFFGRID_t1w-raw-axi.nii.gz").exists()
        assert not (downloader._download_dir / "mri/batch00/OFFGRID/seg/OFFGRID_t1w-raw-axi_brain-mask.nii.gz").exists()


class TestUnreadableVolume:
    """A member nibabel cannot parse reaches the same reject path as an off-grid pair."""

    def test_a_corrupt_member_is_recorded_as_a_reject_rather_than_escaping(self, tmp_path: Path) -> None:
        """Letting it out would end the run; leaving it unrecorded would retry the series forever."""
        image = tmp_path / "CORRUPT_t1w-raw-axi.nii.gz"
        image.write_bytes(b"\x1f\x8b" + bytes(range(40)) * 3)
        mask = tmp_path / "CORRUPT_t1w-raw-axi_brain-mask.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(BRAIN_SHAPE, dtype=np.uint8), AFFINE), mask)

        archive = tmp_path / "store" / "CORRUPT.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(image, "CORRUPT/img/CORRUPT_t1w-raw-axi.nii.gz")
            zipped.write(mask, "CORRUPT/seg/CORRUPT_t1w-raw-axi_brain-mask.nii.gz")

        manifest = tmp_path / "corrupt_manifest.csv"
        with manifest.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(MANIFEST_COLUMNS)
            writer.writerow(["1", "CORRUPT", "t1w-raw-axi", "t1w", "mri_t1", "Train", "mri/batch00/CORRUPT/img/CORRUPT_t1w-raw-axi.nii.gz"])

        log = VerdictLog(tmp_path / "corrupt_verdicts.csv")
        downloader = ReplayDownloader(
            manifest_path=manifest,
            download_dir=tmp_path / "data",
            source=RecordingZipSource({"mri/batch00/CORRUPT.zip": archive}, tmp_path / "downloads"),
            verdict_log=log,
        )

        summary = downloader.run(accepted_csv=tmp_path / "accepted.csv", rejected_csv=tmp_path / "rejected.csv")

        assert summary["failed"] == 0
        assert summary["rejected"] == 1
        row = log.read_rows()[0]
        assert row["verdict"] == "reject"
        assert "could not be read" in row["reasons"]


class TestThresholdOverridesReachTheFilter:
    def test_tightening_the_floor_rejects_every_brain_like_series(self, manifest_path: Path, tmp_path: Path, replay_data: dict[str, Path]) -> None:
        """The verdicts are the filter's own, so a threshold override must change them."""
        log = VerdictLog(tmp_path / "verdicts.csv")
        downloader = ReplayDownloader(
            manifest_path=manifest_path,
            download_dir=tmp_path / "data",
            source=RecordingZipSource(replay_data, tmp_path / "downloads"),
            verdict_log=log,
            filter_service=SpineFilter(thresholds=SpineFilterThresholds(mask_ratio_min=0.5)),
        )
        summary = downloader.run(accepted_csv=tmp_path / "a.csv", rejected_csv=tmp_path / "r.csv")

        assert summary["accepted"] == 0
        assert summary["rejected"] == 4
        assert all(row["reasons"].startswith("mask voxel ratio") for row in log.read_rows())


class TestHuggingFaceCredentials:
    """The run sets ``HF_HOME`` on the scratch disk, so the client's own lookup misses the token."""

    def test_defaults_point_at_the_mr_rate_dataset(self) -> None:
        assert HuggingFaceZipSource()._repo_id == "Forithmus/MR-RATE"

    def test_bare_token_file_is_read(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text("hf_baretoken\n")
        assert HuggingFaceZipSource._parse_credential_file(token_file) == "hf_baretoken"

    def test_ini_style_stored_tokens_are_read(self, tmp_path: Path) -> None:
        stored = tmp_path / "stored_tokens"
        stored.write_text("[Dataset]\nhf_token = hf_oldone\n\n[gauss]\nhf_token = hf_newone\n")
        assert HuggingFaceZipSource._parse_credential_file(stored) == "hf_oldone"

    def test_unrecognized_content_yields_nothing(self, tmp_path: Path) -> None:
        empty = tmp_path / "token"
        empty.write_text("")
        assert HuggingFaceZipSource._parse_credential_file(empty) is None

    def test_environment_token_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_fromenv")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        (tmp_path / "token").write_text("hf_fromfile\n")
        assert HuggingFaceZipSource._resolve_token() == "hf_fromenv"


class ScriptedDownloader:
    """A ``snapshot_download`` stand-in following a script of outcomes, so retries are observable.

    Injected into ``HuggingFaceZipSource`` as its download collaborator -- no network, and no
    inheritance from the class under test.
    """

    def __init__(self, outcomes: list[object], root: Path) -> None:
        self._outcomes = outcomes
        self._root = root
        self.calls: list[list[str]] = []

    def __call__(self, repo_paths: list[str]) -> str:
        self.calls.append(list(repo_paths))
        outcome = self._outcomes[min(len(self.calls), len(self._outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return str(self._root)


class TestFetchRetries:
    """A multi-hour run cannot be ended by one transient network blip."""

    @pytest.fixture(autouse=True)
    def no_real_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Capture the backoff durations instead of living through them."""
        waits: list[float] = []
        monkeypatch.setattr("scripts.download_replay_subset.time.sleep", waits.append)
        return waits

    def test_a_transient_failure_is_retried(self, no_real_sleeping: list[float], tmp_path: Path) -> None:
        downloader = ScriptedDownloader([ConnectionError("temporary"), ConnectionError("temporary"), "ok"], tmp_path / "root")
        source = HuggingFaceZipSource(downloader=downloader)

        assert source.fetch(["mri/batch00/A.zip"]) == tmp_path / "root"
        assert len(downloader.calls) == 3

    def test_an_exhausted_retry_budget_propagates(self, no_real_sleeping: list[float], tmp_path: Path) -> None:
        downloader = ScriptedDownloader([ConnectionError("still down")], tmp_path / "root")
        source = HuggingFaceZipSource(downloader=downloader)

        with pytest.raises(ConnectionError, match="still down"):
            source.fetch(["mri/batch00/A.zip"])
        assert len(downloader.calls) == HuggingFaceZipSource.ATTEMPTS

    def test_the_backoff_widens_between_attempts(self, no_real_sleeping: list[float], tmp_path: Path) -> None:
        downloader = ScriptedDownloader([ConnectionError("x"), ConnectionError("x"), "ok"], tmp_path / "root")
        source = HuggingFaceZipSource(downloader=downloader)

        assert source.fetch(["mri/batch00/A.zip"]) == tmp_path / "root"
        assert no_real_sleeping == [30.0, 60.0]


class TestCommandLine:
    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run([sys.executable, "-m", "scripts.download_replay_subset", "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0
        assert "--accepted-csv" in result.stdout
        assert "--rejected-csv" in result.stdout
