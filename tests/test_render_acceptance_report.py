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

"""Tests for scripts.render_acceptance_report (ticket T10, issue #28).

The report is what gets pasted into the experiment-point issues (#14/#15/#16),
so the tests pin the record format of spec #13 §⑤.6: the forgetting FID table
(10 labels x pre/ft/Δ/CI/verdict x 2 seeds), the BRATS table (4 labels + the
real-real floor), the montage link, and the HITL section listing exactly the
labels whose verdict is quickscan.
"""

import json
from pathlib import Path

import pytest

from scripts.render_acceptance_report import AcceptanceReportRenderer


@pytest.fixture
def adjudication(tmp_path: Path) -> Path:
    labels = {}
    for label in (9, 10, 11, 16, 20, 29, 30, 31, 32, 33):
        seeds = {}
        for seed in (42, 1337):
            seeds[f"seed{seed}"] = {
                "fid_pre": 5.0,
                "fid_ft": 5.2,
                "delta": 0.04,
                "delta_pct": 4.0,
                "ci_pre": [4.8, 5.2],
                "ci_ft": [4.9, 5.5],
                "ci_overlap": True,
                "verdict": "pass",
            }
        labels[str(label)] = {**seeds, "verdict": "pass"}
    # label 16 seed42 lands in the quickscan band
    labels["16"]["seed42"] = {
        "fid_pre": 5.0,
        "fid_ft": 5.7,
        "delta": 0.14,
        "delta_pct": 14.0,
        "ci_pre": [4.8, 5.2],
        "ci_ft": [5.1, 6.3],
        "ci_overlap": True,
        "verdict": "quickscan",
    }
    labels["16"]["verdict"] = "quickscan"

    payload = {
        "run_id": "N300e300",
        "labels": labels,
        "forgetting_verdict": "quickscan_pending",
        "quickscan_labels": ["16"],
        "brats": {
            "40": {"per_seed": {"42": 10.0, "1337": 12.0}, "mean": 11.0},
            "41": {"per_seed": {"42": 20.0, "1337": 24.0}, "mean": 22.0},
            "42": {"per_seed": {"42": 15.0, "1337": 17.0}, "mean": 16.0},
            "43": {"per_seed": {"42": 18.0, "1337": 22.0}, "mean": 20.0},
        },
        "brats_fid_mean": 17.25,
    }
    path = tmp_path / "adjudication.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.fixture
def freeze(tmp_path: Path) -> Path:
    path = tmp_path / "fid_freeze.json"
    path.write_text(json.dumps({"real_real_floor": {"floor_mri_t1n": {"fid_avg": 3.1}, "floor_mri_t1ce": {"fid_avg": 3.2}, "floor_mri_t2w": {"fid_avg": 3.3}, "floor_mri_t2f": {"fid_avg": 3.4}}}))
    return path


class TestAcceptanceReportRenderer:
    def test_renders_forgotten_table_with_all_labels_and_seeds(self, tmp_path: Path, adjudication: Path, freeze: Path) -> None:
        report = AcceptanceReportRenderer(adjudication_json=adjudication, freeze_json=freeze, release_url="https://example.com/release").render()

        assert report.count("| 9 |") >= 1 and report.count("| 33 |") >= 1
        assert "seed42" in report and "seed1337" in report
        assert "FID_pre" in report and "Δ%" in report and "CI95" in report

    def test_marks_quickscan_labels_and_hitsl_section(self, tmp_path: Path, adjudication: Path, freeze: Path) -> None:
        report = AcceptanceReportRenderer(adjudication_json=adjudication, freeze_json=freeze, release_url="https://example.com/release").render()

        assert "quickscan" in report
        assert "16" in report
        assert "HITL" in report
        for check in ("解剖", "对比度", "颅骨", "label 判别", "伪影"):
            assert check in report

    def test_renders_brats_table_with_floor(self, tmp_path: Path, adjudication: Path, freeze: Path) -> None:
        report = AcceptanceReportRenderer(adjudication_json=adjudication, freeze_json=freeze, release_url="https://example.com/release").render()

        assert "| 40 |" in report and "| 43 |" in report
        assert "3.1" in report  # floor value present
        assert "17.25" in report

    def test_includes_release_link_and_verdict(self, tmp_path: Path, adjudication: Path, freeze: Path) -> None:
        report = AcceptanceReportRenderer(adjudication_json=adjudication, freeze_json=freeze, release_url="https://example.com/release").render()

        assert "https://example.com/release" in report
        assert "quickscan_pending" in report

    def test_writes_report_file(self, tmp_path: Path, adjudication: Path, freeze: Path) -> None:
        output = tmp_path / "report.md"
        AcceptanceReportRenderer(adjudication_json=adjudication, freeze_json=freeze, release_url="https://x").write(output)

        assert output.exists() and "N300e300" in output.read_text()
