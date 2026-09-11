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

"""Tests for the BRATS real-real floor split (ticket T8, issue #24; spec #13 section 5.3/5.5).

The floor anchor is one real-real FID on the BRATS validation cases split into two
disjoint halves of (near-)equal size. The split is at **case** level so all four
modality volumes of one subject land in the same half; per-label filelists are then
expanded from the case halves with the dataset's file-suffix mapping. The split must
be seed-reproducible (the frozen record is reused for every acceptance point, never
recomputed) and must cover every case exactly once.
"""

import json
from pathlib import Path

import pytest

from scripts.split_real_real_halves import (
    BRATS_LABEL_SUFFIX,
    CaseFilelistExpander,
    HalvesSplitter,
    SplitFreezeRecord,
)


class TestHalvesSplitter:
    def test_split_is_deterministic_for_a_seed(self) -> None:
        cases = [f"BraTS-GLI-{i:05d}-000" for i in range(64)]
        first = HalvesSplitter(seed=42).split(cases)
        second = HalvesSplitter(seed=42).split(cases)

        assert first == second

    def test_different_seed_can_give_different_split(self) -> None:
        cases = [f"BraTS-GLI-{i:05d}-000" for i in range(64)]
        first = HalvesSplitter(seed=42).split(cases)
        second = HalvesSplitter(seed=1337).split(cases)

        assert first != second

    def test_halves_partition_the_cases(self) -> None:
        cases = [f"BraTS-GLI-{i:05d}-000" for i in range(64)]
        half_a, half_b = HalvesSplitter(seed=42).split(cases)

        assert len(half_a) == 32
        assert len(half_b) == 32
        assert set(half_a) & set(half_b) == set()
        assert set(half_a) | set(half_b) == set(cases)

    def test_odd_case_count_keeps_remainder_in_first_half(self) -> None:
        cases = [f"BraTS-GLI-{i:05d}-000" for i in range(5)]
        half_a, half_b = HalvesSplitter(seed=42).split(cases)

        assert len(half_a) == 3
        assert len(half_b) == 2

    def test_duplicate_case_is_refused(self) -> None:
        cases = ["BraTS-GLI-00000-000", "BraTS-GLI-00000-000"]

        with pytest.raises(ValueError, match="duplicate"):
            HalvesSplitter(seed=42).split(cases)


class TestCaseFilelistExpander:
    def test_label_suffix_mapping_covers_the_four_brats_labels(self) -> None:
        assert BRATS_LABEL_SUFFIX == {
            "mri_t1n": "t1n",
            "mri_t1ce": "t1c",
            "mri_t2w": "t2w",
            "mri_t2f": "t2f",
        }

    def test_expands_cases_to_volume_paths_for_one_label(self) -> None:
        cases = ["BraTS-GLI-00018-000", "BraTS-GLI-00024-001"]
        expander = CaseFilelistExpander(data_prefix="brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

        paths = expander.expand(cases, label="mri_t1ce")

        assert paths == [
            "brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00018-000/BraTS-GLI-00018-000-t1c.nii.gz",
            "brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00024-001/BraTS-GLI-00024-001-t1c.nii.gz",
        ]

    def test_unknown_label_is_refused(self) -> None:
        expander = CaseFilelistExpander(data_prefix="x")

        with pytest.raises(ValueError, match="unknown BRATS label"):
            expander.expand(["BraTS-GLI-00000-000"], label="mri_t1")

    def test_all_four_label_filelists_have_equal_length(self) -> None:
        cases = [f"BraTS-GLI-{i:05d}-000" for i in range(64)]
        expander = CaseFilelistExpander(data_prefix="p")

        per_label = {label: expander.expand(cases, label=label) for label in BRATS_LABEL_SUFFIX}

        assert all(len(paths) == 64 for paths in per_label.values())


class TestSplitFreezeRecord:
    def test_record_round_trips_through_json(self, tmp_path: Path) -> None:
        record = SplitFreezeRecord(
            seed=42,
            half_a=["BraTS-GLI-00000-000"],
            half_b=["BraTS-GLI-00001-000"],
            validation_source="brats2023-gli/dataset.json",
        )
        path = tmp_path / "record.json"

        record.save(path)
        restored = SplitFreezeRecord.load(path)

        assert restored == record

    def test_record_json_contains_halves_and_seed(self, tmp_path: Path) -> None:
        record = SplitFreezeRecord(seed=7, half_a=["a"], half_b=["b"], validation_source="s")
        path = tmp_path / "record.json"

        record.save(path)

        payload = json.loads(path.read_text())
        assert payload["seed"] == 7
        assert payload["half_a"] == ["a"]
        assert payload["half_b"] == ["b"]
        assert payload["validation_source"] == "s"
