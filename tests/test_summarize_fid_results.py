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

"""Tests for the FID freeze summarizer (ticket T8, issue #24; spec #13 section 5.5).

One FID invocation writes one result json (``compute_fid_2-5d_ct.FidResult``); the
freeze is the aggregation of those records into a single json/csv pair that every
acceptance point compares against -- the pretrained baseline table (10 old labels x
2 seeds, tag ``label<L>_seed<S>``) and the BRATS real-real floor (tag
``floor_<label>``). Tag conventions are strict: an unparseable or duplicated tag is
refused rather than silently mis-filed into the frozen table.
"""

import csv
import importlib
import json
from pathlib import Path

import pytest

from scripts.summarize_fid_results import FidSummary

# Module name carries a hyphen, so import it programmatically.
compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")
FidResult = compute_fid.FidResult


def write_result(path: Path, tag: str, fid_avg: float) -> Path:
    payload = {
        "comparison_tag": tag,
        "fid_xy": fid_avg,
        "fid_yz": fid_avg + 1.0,
        "fid_zx": fid_avg + 2.0,
        "fid_avg": fid_avg,
        "modality": "mr",
        "model_name": "radimagenet_resnet50",
        "num_images": 50,
        "real_filelist": f"ref/{tag}.txt",
        "synth_filelist": f"synth/{tag}.txt",
        "target_shape": "256x256x128",
        "center_slices_ratio": 0.4,
    }
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def records(tmp_path: Path) -> list[FidResult]:
    paths = [
        write_result(tmp_path / "b9s42.json", "label9_seed42", 10.0),
        write_result(tmp_path / "b9s1337.json", "label9_seed1337", 12.0),
        write_result(tmp_path / "b10s42.json", "label10_seed42", 14.0),
        write_result(tmp_path / "floor.json", "floor_mri_t1n", 3.0),
    ]
    return [FidResult.load(path) for path in paths]


class TestFidSummary:
    def test_baseline_and_floor_land_in_separate_tables(self, records: list[FidResult]) -> None:
        summary = FidSummary(records, sources=["a.json"]).summarize()

        assert set(summary["baseline_gen_real"]) == {"9", "10"}
        assert set(summary["baseline_gen_real"]["9"]) == {"42", "1337"}
        assert summary["baseline_gen_real"]["9"]["42"]["fid_avg"] == 10.0
        assert set(summary["real_real_floor"]) == {"mri_t1n"}
        assert summary["real_real_floor"]["mri_t1n"]["fid_avg"] == 3.0

    def test_summary_records_its_sources(self, records: list[FidResult]) -> None:
        summary = FidSummary(records, sources=["a.json", "b.json"]).summarize()

        assert summary["generated_from"] == ["a.json", "b.json"]

    def test_csv_has_one_row_per_record(self, records: list[FidResult], tmp_path: Path) -> None:
        csv_path = tmp_path / "fid_freeze.csv"

        FidSummary(records, sources=[]).save(tmp_path / "fid_freeze.json", csv_path)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 4
        assert {row["kind"] for row in rows} == {"baseline_gen_real", "real_real_floor"}
        assert {row["fid_avg"] for row in rows} == {"10.0", "12.0", "14.0", "3.0"}

    def test_a_duplicate_tag_is_refused(self, records: list[FidResult]) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            FidSummary([*records, records[0]], sources=[]).summarize()

    def test_an_unknown_tag_pattern_is_refused(self, tmp_path: Path) -> None:
        stray = FidResult.load(write_result(tmp_path / "stray.json", "mystery_tag", 1.0))

        with pytest.raises(ValueError, match="unrecognized comparison tag"):
            FidSummary([stray], sources=[]).summarize()

    def test_an_empty_record_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no records"):
            FidSummary([], sources=[]).summarize()
