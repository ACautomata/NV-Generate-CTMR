"""Tests for scripts.create_brats_dataset_json — BRATS dataset.json 阶段①生成器（ticket T1, issue #17）.

Fake data layout: 5 scans across 4 subjects; subject BraTS-GLI-00002 has two
timepoints (-000/-001) to exercise the same-subject-stays-together invariant.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_brats_dataset_json import (
    SEG_SUFFIX,
    SUFFIX_TO_MODALITY,
    BraTSDatasetList,
    BraTSScan,
    BraTSScanIndex,
    HoldoutSplitter,
    NiftiSpotCheck,
    SubjectSplit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_DIRNAME = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
ALL_SUFFIXES = ("seg", "t1c", "t1n", "t2f", "t2w")
FAKE_SCANS = (
    "BraTS-GLI-00000-000",
    "BraTS-GLI-00001-000",
    "BraTS-GLI-00002-000",
    "BraTS-GLI-00002-001",
    "BraTS-GLI-00003-000",
)


@pytest.fixture
def write_scan_files() -> Callable[[Path, str], None]:
    """Factory: create a case directory holding all five expected files under ``training_data``."""

    def _write(training_data: Path, scan: str) -> None:
        bra_scan = BraTSScan(directory=scan)
        for suffix in ALL_SUFFIXES:
            path = bra_scan.path(training_data, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    return _write


@pytest.fixture
def brats_root(tmp_path: Path, write_scan_files: Callable[[Path, str], None]) -> Path:
    """A fake brats2023-gli root whose TrainingData holds 5 scans / 4 subjects."""
    training_data = tmp_path / "brats2023-gli" / TRAINING_DATA_DIRNAME
    for scan in FAKE_SCANS:
        write_scan_files(training_data, scan)
    return tmp_path / "brats2023-gli"


@pytest.fixture
def training_data_dir(brats_root: Path) -> Path:
    return brats_root / TRAINING_DATA_DIRNAME


@pytest.fixture
def build_dataset_list(brats_root: Path, training_data_dir: Path) -> Callable[[float], tuple[BraTSDatasetList, BraTSScanIndex, SubjectSplit]]:
    """Factory: build the generator's collaborators for a given val fraction."""

    def _build(val_fraction: float) -> tuple[BraTSDatasetList, BraTSScanIndex, SubjectSplit]:
        index = BraTSScanIndex(training_data_dir)
        split = HoldoutSplitter(val_fraction=val_fraction, seed=42).split(index.subjects)
        dataset_list = BraTSDatasetList(
            training_data_dir=training_data_dir,
            data_base_dir=brats_root.parent,
            scans=index.scans,
            split=split,
        )
        return dataset_list, index, split

    return _build


@pytest.fixture
def write_scan_volumes(tmp_path: Path) -> Callable[..., BraTSScan]:
    """Factory: write one case's t1c/seg NIfTI volumes under ``root`` and return its ``BraTSScan``."""

    def _write(root: Path, t1c: np.ndarray, seg: np.ndarray, scan: str = "BraTS-GLI-00000-000") -> BraTSScan:
        bra_scan = BraTSScan(directory=scan)
        for suffix, array in (("t1c", t1c), (SEG_SUFFIX, seg)):
            path = bra_scan.path(root, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(array, affine=np.eye(4)), path)
        return bra_scan

    return _write


class TestBraTSScanIndex:
    def test_finds_all_scans_sorted(self, training_data_dir: Path) -> None:
        index = BraTSScanIndex(training_data_dir)

        assert [scan.directory for scan in index.scans] == sorted(FAKE_SCANS)
        assert index.subjects == sorted({BraTSScan(directory=scan).subject for scan in FAKE_SCANS})

    def test_scan_subject_strips_timepoint(self) -> None:
        assert BraTSScan(directory="BraTS-GLI-00002-001").subject == "BraTS-GLI-00002"

    def test_rejects_scan_missing_modality_files(self, training_data_dir: Path) -> None:
        victim = training_data_dir / "BraTS-GLI-00001-000"
        (victim / "BraTS-GLI-00001-000-t2w.nii.gz").unlink()

        with pytest.raises(ValueError, match="BraTS-GLI-00001-000"):
            BraTSScanIndex(training_data_dir)

    def test_rejects_scan_missing_seg(self, training_data_dir: Path) -> None:
        victim = training_data_dir / "BraTS-GLI-00001-000"
        (victim / "BraTS-GLI-00001-000-seg.nii.gz").unlink()

        with pytest.raises(ValueError, match="seg"):
            BraTSScanIndex(training_data_dir)

    def test_rejects_directory_with_unexpected_name(self, training_data_dir: Path, write_scan_files: Callable[[Path, str], None]) -> None:
        write_scan_files(training_data_dir, "BraTS-MEN-99999-000")

        with pytest.raises(ValueError, match="BraTS-MEN-99999-000"):
            BraTSScanIndex(training_data_dir)


class TestHoldoutSplitter:
    def test_partition_covers_all_subjects_disjointly(self) -> None:
        subjects = ["BraTS-GLI-00000", "BraTS-GLI-00001", "BraTS-GLI-00002", "BraTS-GLI-00003"]
        split = HoldoutSplitter(val_fraction=0.5, seed=42).split(subjects)

        assert sorted(split.train | split.val) == sorted(subjects)
        assert not (split.train & split.val)

    def test_val_size_rounds_down_to_fraction(self) -> None:
        subjects = [f"BraTS-GLI-{i:05d}" for i in range(100)]

        split = HoldoutSplitter(val_fraction=0.05, seed=42).split(subjects)

        assert len(split.val) == 5

    def test_same_seed_reproduces_split(self) -> None:
        subjects = [f"BraTS-GLI-{i:05d}" for i in range(100)]

        first = HoldoutSplitter(val_fraction=0.05, seed=42).split(subjects)
        second = HoldoutSplitter(val_fraction=0.05, seed=42).split(subjects)

        assert first == second

    def test_different_seed_changes_split(self) -> None:
        subjects = [f"BraTS-GLI-{i:05d}" for i in range(100)]

        first = HoldoutSplitter(val_fraction=0.05, seed=42).split(subjects)
        second = HoldoutSplitter(val_fraction=0.05, seed=7).split(subjects)

        assert first != second


class TestBraTSDatasetList:
    def test_training_entry_count_is_cases_times_modalities(self, build_dataset_list) -> None:
        dataset_list, index, split = build_dataset_list(0.25)

        entries = dataset_list.to_dict()["training"]

        val_cases = sum(1 for scan in index.scans if scan.subject in split.val)
        assert len(entries) == (len(FAKE_SCANS) - val_cases) * len(SUFFIX_TO_MODALITY)

    def test_training_entries_map_suffix_to_modality(self, build_dataset_list) -> None:
        dataset_list, _, _ = build_dataset_list(0.0)

        entries = dataset_list.to_dict()["training"]
        by_image = {entry["image"]: entry["modality"] for entry in entries}

        for scan in FAKE_SCANS:
            for suffix, modality in SUFFIX_TO_MODALITY.items():
                image = f"brats2023-gli/{TRAINING_DATA_DIRNAME}/{scan}/{scan}-{suffix}.nii.gz"
                assert by_image[image] == modality

    def test_seg_never_enters_training_entries(self, build_dataset_list) -> None:
        dataset_list, _, _ = build_dataset_list(0.0)

        assert all("-seg" not in entry["image"] for entry in dataset_list.to_dict()["training"])

    def test_validation_roster_keeps_timepoints_together(self, build_dataset_list) -> None:
        dataset_list, _, _ = build_dataset_list(0.5)

        roster = dataset_list.to_dict()["validation"]
        roster_subjects = {BraTSScan(directory=case).subject for case in roster}
        for subject in roster_subjects:
            expected = [scan for scan in FAKE_SCANS if BraTSScan(directory=scan).subject == subject]
            in_roster = [case for case in roster if BraTSScan(directory=case).subject == subject]
            assert sorted(expected) == sorted(in_roster)

    def test_validation_cases_are_excluded_from_training(self, build_dataset_list) -> None:
        dataset_list, _, _ = build_dataset_list(0.25)

        payload = dataset_list.to_dict()

        assert set(payload) == {"training", "validation"}
        trained_scans = {Path(entry["image"]).parent.name for entry in payload["training"]}
        assert trained_scans.isdisjoint(set(payload["validation"]))

    def test_save_is_reproducible_byte_for_byte(self, build_dataset_list, tmp_path: Path) -> None:
        outputs = []
        for name in ("first.json", "second.json"):
            dataset_list, _, _ = build_dataset_list(0.25)
            output = tmp_path / name
            dataset_list.save(output)
            outputs.append(output.read_bytes())

        assert outputs[0] == outputs[1]


class TestNiftiSpotCheck:
    def test_passes_on_wellformed_scan(self, write_scan_volumes: Callable[..., BraTSScan], tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        scan = write_scan_volumes(
            tmp_path,
            np.zeros((240, 240, 155), dtype=np.float32),
            rng.integers(0, 4, size=(240, 240, 155)).astype(np.uint8),
        )

        NiftiSpotCheck(training_data_dir=tmp_path).run(scan)

    def test_rejects_wrong_shape(self, write_scan_volumes: Callable[..., BraTSScan], tmp_path: Path) -> None:
        scan = write_scan_volumes(tmp_path, np.zeros((240, 240, 154), dtype=np.float32), np.zeros((240, 240, 154), dtype=np.uint8))

        with pytest.raises(ValueError, match="shape"):
            NiftiSpotCheck(training_data_dir=tmp_path).run(scan)

    def test_rejects_seg_labels_outside_domain(self, write_scan_volumes: Callable[..., BraTSScan], tmp_path: Path) -> None:
        scan = write_scan_volumes(
            tmp_path,
            np.zeros((240, 240, 155), dtype=np.float32),
            np.full((240, 240, 155), 4, dtype=np.uint8),
        )

        with pytest.raises(ValueError, match="seg"):
            NiftiSpotCheck(training_data_dir=tmp_path).run(scan)


class TestCommandLine:
    def test_end_to_end_generation(self, brats_root: Path, write_scan_volumes: Callable[..., BraTSScan], tmp_path: Path) -> None:
        training_data = brats_root / TRAINING_DATA_DIRNAME
        write_scan_volumes(
            training_data,
            np.zeros((240, 240, 155), dtype=np.float32),
            np.zeros((240, 240, 155), dtype=np.uint8),
        )
        output = tmp_path / "dataset.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.create_brats_dataset_json",
                "--training-data-dir",
                str(training_data),
                "--data-base-dir",
                str(brats_root.parent),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "spot check passed" in result.stdout
        payload = json.loads(output.read_text())
        assert set(payload) == {"training", "validation"}
        assert len(payload["training"]) == len(FAKE_SCANS) * 4
        assert all(entry["modality"] in set(SUFFIX_TO_MODALITY.values()) for entry in payload["training"])
