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

"""Tests for the replay manifest generator (spec #13 section 3, ticket T2 #18)."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.create_replay_manifest import (
    MANIFEST_COLUMNS,
    MODALITIES,
    CapPolicy,
    ModalitySubjectIndex,
    MrRateCatalog,
    MrRateSeries,
    ReplayManifest,
    StratifiedSubjectSampler,
    main,
)

METADATA_COLUMNS = [
    "patient_uid",
    "study_uid",
    "series_id",
    "classified_modality",
    "is_derived",
    "acquisition_plane",
    "SeriesNumber",
    "is_localizer",
    "is_subtraction",
]


def series_row(
    patient: str,
    study: str,
    series_id: str,
    modality: str,
    series_number: str,
    is_derived: str = "False",
    is_localizer: str = "False",
    is_subtraction: str = "False",
) -> dict:
    return {
        "patient_uid": patient,
        "study_uid": study,
        "series_id": series_id,
        "classified_modality": modality,
        "is_derived": is_derived,
        "acquisition_plane": "AXIAL",
        "SeriesNumber": series_number,
        "is_localizer": is_localizer,
        "is_subtraction": is_subtraction,
    }


def splits_row(patient: str, study: str, split: str, batch: str = "batch00") -> dict:
    return {"batch_id": batch, "patient_uid": patient, "study_uid": study, "split": split}


def write_csv(path: Path, columns: list, rows: list) -> Path:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    """Fixture metadata: multi-series subjects, flag-excluded rows, non-train patients, two batches."""
    write_csv(
        tmp_path / "splits.csv",
        ["batch_id", "patient_uid", "study_uid", "split"],
        [
            splits_row("1", "AAA111AAA", "train"),
            splits_row("2", "BBB222BBB", "train", batch="batch01"),
            splits_row("3", "CCC333CCC", "train"),
        ]
        + [splits_row(str(number), f"SUB{number:03d}XYZ", "train") for number in range(6, 16)],
    )
    write_csv(
        tmp_path / "batch00_metadata.csv",
        METADATA_COLUMNS,
        [
            # subject 1: three t1w series with shuffled SeriesNumbers; the min-SeriesNumber
            # one must win; one derived and one localizer row must never win
            series_row("1", "AAA111AAA", "t1w-raw-axi-2", "T1w", "7.0"),
            series_row("1", "AAA111AAA", "t1w-raw-axi", "T1w", "3.0"),
            series_row("1", "AAA111AAA", "t1w-raw-sag", "T1w", "5.0"),
            series_row("1", "AAA111AAA", "t1w-derived", "T1w", "1.0", is_derived="True"),
            series_row("1", "AAA111AAA", "t1w-localizer", "T1w", "0.5", is_localizer="True"),
            series_row("1", "AAA111AAA", "flair-raw-sag", "FLAIR", "4.0"),
            series_row("1", "AAA111AAA", "swi-raw-axi", "SWI", "13.0"),
            series_row("1", "AAA111AAA", "t2w-raw-axi", "T2w", "2.0"),
            series_row("1", "AAA111AAA", "mra-raw-sag", "MRA", "21.0"),
            # subject 3: train, t1w + t2w
            series_row("3", "CCC333CCC", "t1w-raw-axi", "T1w", "2.0"),
            series_row("3", "CCC333CCC", "t2w-raw-sag", "T2w", "6.0"),
            # val/test patients never enter the candidate pool
            series_row("4", "DDD444DDD", "t1w-raw-axi", "T1w", "1.0"),
            series_row("5", "EEE555EEE", "t1w-raw-axi", "T1w", "1.0"),
        ]
        + [
            # ten more t1w-only train subjects so sampler tests have a wide draw pool
            series_row(str(number), f"SUB{number:03d}XYZ", "t1w-raw-axi", "T1w", "2.0")
            for number in range(6, 16)
        ],
    )
    write_csv(
        tmp_path / "batch01_metadata.csv",
        METADATA_COLUMNS,
        [
            # subject 2 lives in batch01 (its splits.csv batch is batch01): subject-level
            # candidates must be found regardless of batch
            series_row("2", "BBB222BBB", "t2w-raw-axi", "T2w", "2.0"),
            series_row("2", "BBB222BBB", "t2w-raw-axi-2", "T2w", "9.0"),
            series_row("2", "BBB222BBB", "mra-raw-cor", "MRA", "30.0", is_subtraction="True"),
        ],
    )
    return tmp_path


@pytest.fixture
def catalog(metadata_dir: Path) -> MrRateCatalog:
    return MrRateCatalog(
        splits_csv=metadata_dir / "splits.csv",
        metadata_paths=[metadata_dir / "batch00_metadata.csv", metadata_dir / "batch01_metadata.csv"],
    )


class TestMrRateCatalog:
    def test_train_brain_series_excludes_non_train_and_flagged_rows(self, catalog: MrRateCatalog) -> None:
        series = catalog.train_brain_series()
        keys = {(item.patient_uid, item.series_id) for item in series}
        assert ("4", "t1w-raw-axi") not in keys  # val patient
        assert ("5", "t1w-raw-axi") not in keys  # test patient
        assert ("1", "t1w-derived") not in keys
        assert ("1", "t1w-localizer") not in keys
        assert ("2", "mra-raw-cor") not in keys  # subtraction
        assert ("1", "t1w-raw-axi") in keys

    def test_batch_and_modality_are_carried_through(self, catalog: MrRateCatalog) -> None:
        series = {(item.patient_uid, item.series_id): item for item in catalog.train_brain_series()}
        assert series[("1", "t1w-raw-axi")].batch == "batch00"
        assert series[("2", "t2w-raw-axi")].batch == "batch01"
        assert series[("1", "flair-raw-sag")].modality == "flair"

    def test_duplicate_study_series_raises(self, tmp_path: Path) -> None:
        splits = write_csv(
            tmp_path / "splits.csv",
            ["batch_id", "patient_uid", "study_uid", "split"],
            [splits_row("1", "AAA111AAA", "train")],
        )
        rows = [series_row("1", "AAA111AAA", "t1w-raw-axi", "T1w", "1.0")]
        first = write_csv(tmp_path / "batch00_metadata.csv", METADATA_COLUMNS, rows)
        second = write_csv(tmp_path / "batch01_metadata.csv", METADATA_COLUMNS, rows)
        with pytest.raises(ValueError, match="duplicate"):
            MrRateCatalog(splits_csv=splits, metadata_paths=[first, second]).train_brain_series()

    def test_empty_metadata_paths_raise(self, tmp_path: Path) -> None:
        splits = write_csv(
            tmp_path / "splits.csv",
            ["batch_id", "patient_uid", "study_uid", "split"],
            [splits_row("1", "AAA111AAA", "train")],
        )
        with pytest.raises(FileNotFoundError, match="no metadata CSVs"):
            MrRateCatalog(splits_csv=splits, metadata_paths=[])

    def test_unexpected_boolean_encoding_raises(self, tmp_path: Path) -> None:
        """A future pull writing 'true'/'1' must fail loudly instead of silently keeping flagged rows."""
        splits = write_csv(
            tmp_path / "splits.csv",
            ["batch_id", "patient_uid", "study_uid", "split"],
            [splits_row("1", "AAA111AAA", "train")],
        )
        metadata = write_csv(
            tmp_path / "batch00_metadata.csv",
            METADATA_COLUMNS,
            [series_row("1", "AAA111AAA", "t1w-raw-axi", "T1w", "1.0", is_derived="true")],
        )
        catalog = MrRateCatalog(splits_csv=splits, metadata_paths=[metadata])
        with pytest.raises(ValueError, match="unexpected is_derived"):
            catalog.train_brain_series()


class TestModalitySubjectIndex:
    def test_min_series_number_wins_per_subject_and_modality(self, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        t1w = {entry.patient_uid: entry for entry in index.entries("t1w")}
        assert set(t1w) == {"1", "3"} | {str(number) for number in range(6, 16)}
        assert t1w["1"].series_id == "t1w-raw-axi"  # SeriesNumber 3.0 beats 5.0 and 7.0
        assert t1w["3"].series_id == "t1w-raw-axi"

    def test_subjects_across_batches_are_indexed(self, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        t2w = {entry.patient_uid: entry for entry in index.entries("t2w")}
        assert t2w["2"].series_id == "t2w-raw-axi"  # 2.0 beats 9.0; study lives in batch01
        assert t2w["2"].batch == "batch01"

    def test_entries_are_sorted_by_patient_uid(self, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        for modality in MODALITIES:
            patients = [entry.patient_uid for entry in index.entries(modality)]
            assert patients == sorted(patients)


class TestStratifiedSubjectSampler:
    def test_same_seed_selects_the_same_subjects(self, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        t1w = index.entries("t1w")
        first = StratifiedSubjectSampler(seed=42).select("t1w", t1w, cap=6)
        second = StratifiedSubjectSampler(seed=42).select("t1w", t1w, cap=6)
        assert [entry.patient_uid for entry in first] == [entry.patient_uid for entry in second]

    def test_the_seed_actually_feeds_the_draw(self, catalog: MrRateCatalog) -> None:
        """Across many seeds a capped draw from 12 candidates cannot always return the same set."""
        index = ModalitySubjectIndex(catalog.train_brain_series())
        t1w = index.entries("t1w")
        selections = {tuple(entry.patient_uid for entry in StratifiedSubjectSampler(seed=seed).select("t1w", t1w, cap=6)) for seed in range(20)}
        assert len(selections) >= 2

    def test_capped_selection_is_the_prefix_of_the_full_ordering(self, catalog: MrRateCatalog) -> None:
        """Top-up contract (ticket 6): continue down the same ordering when refilling shortages."""
        index = ModalitySubjectIndex(catalog.train_brain_series())
        t1w = index.entries("t1w")
        sampler = StratifiedSubjectSampler(seed=7)
        full = sampler.select("t1w", t1w, cap=None)
        for cap in (0, 1, 5, len(full)):
            assert sampler.select("t1w", t1w, cap=cap) == full[:cap]

    def test_cap_none_takes_all(self, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        mra = StratifiedSubjectSampler(seed=42).select("mra", index.entries("mra"), cap=None)
        assert [entry.patient_uid for entry in mra] == ["1"]


class TestCapPolicy:
    def test_head_label_caps_at_n(self) -> None:
        policy = CapPolicy(n_per_label=300)
        assert policy.cap_for("t1w") == 300
        assert policy.cap_for("flair") == 300
        assert policy.cap_for("swi") == 300

    def test_t2w_caps_at_min_of_n_and_669(self) -> None:
        assert CapPolicy(n_per_label=300).cap_for("t2w") == 300
        assert CapPolicy(n_per_label=1000).cap_for("t2w") == 669

    def test_mra_is_uncapped(self) -> None:
        assert CapPolicy(n_per_label=300).cap_for("mra") is None
        assert CapPolicy(n_per_label=1000).cap_for("mra") is None


class TestMrRateSeries:
    def test_label_mapping(self) -> None:
        series = MrRateSeries(batch="batch00", patient_uid="1", study_uid="AAA111AAA", series_id="t1w-raw-axi", modality="t1w", series_number=3.0)
        assert series.label == "mri_t1"

    def test_image_path_follows_the_official_unzip_layout(self) -> None:
        series = MrRateSeries(batch="batch07", patient_uid="1", study_uid="AAA111AAA", series_id="t2w-raw-axi", modality="t2w", series_number=2.0)
        assert series.image_path == "mri/batch07/AAA111AAA/img/AAA111AAA_t2w-raw-axi.nii.gz"
        assert series.mask_path == "mri/batch07/AAA111AAA/seg/AAA111AAA_t2w-raw-axi_brain-mask.nii.gz"


class TestReplayManifest:
    def test_columns_are_exactly_the_spec_fields(self) -> None:
        assert MANIFEST_COLUMNS == ("patient_uid", "study_uid", "series_id", "modality", "label", "split", "image_path")

    def test_save_writes_rows_and_reports_counts(self, tmp_path: Path, catalog: MrRateCatalog) -> None:
        index = ModalitySubjectIndex(catalog.train_brain_series())
        sampler = StratifiedSubjectSampler(seed=42)
        policy = CapPolicy(n_per_label=1)
        selected = []
        for modality in MODALITIES:
            selected.extend(sampler.select(modality, index.entries(modality), cap=policy.cap_for(modality)))
        manifest = ReplayManifest(selected)
        output = tmp_path / "manifest.csv"
        summary = manifest.save(output)
        with output.open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 5  # t1w 1 + t2w 1 + flair 1 + swi 1 + mra 1
        assert list(rows[0]) == list(MANIFEST_COLUMNS)
        assert all(row["split"] == "Train" for row in rows)
        assert all(row["image_path"].startswith("mri/batch0") for row in rows)
        assert summary["counts"] == {"mri_t1": 1, "mri_t2": 1, "mri_flair": 1, "mri_swi": 1, "mri_mra": 1}


class TestCommandLine:
    def test_end_to_end_manifest_is_byte_identical_across_reruns(
        self, tmp_path: Path, metadata_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def run(output_name: str) -> None:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "create_replay_manifest",
                    "--splits-csv",
                    str(metadata_dir / "splits.csv"),
                    "--metadata-dir",
                    str(metadata_dir),
                    "--n-per-label",
                    "1",
                    "--output",
                    str(tmp_path / output_name),
                    "--seed",
                    "42",
                ],
            )
            main()

        run("manifest.csv")
        first = (tmp_path / "manifest.csv").read_bytes()
        run("manifest-rerun.csv")
        assert (tmp_path / "manifest-rerun.csv").read_bytes() == first

        lines = first.decode().splitlines()
        assert lines[0] == ",".join(MANIFEST_COLUMNS)
        assert len(lines) == 6  # header + one row per modality (t1w/t2w/flair/swi/mra at n=1)
        with (tmp_path / "manifest.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert {row["modality"] for row in rows} == {"t1w", "t2w", "flair", "swi", "mra"}
        assert {row["label"] for row in rows} == {"mri_t1", "mri_t2", "mri_flair", "mri_swi", "mri_mra"}
        stdout = capsys.readouterr().out
        assert "sampled per modality" in stdout

    def test_cli_is_invocable_as_a_module(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.create_replay_manifest", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--n-per-label" in result.stdout
