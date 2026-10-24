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

"""Tests for scripts.adjudicate_forgetting (ticket T10, issue #28).

The adjudication table of spec #13 §⑤.2 is the ticket's hard gate, so every
row of it gets a test: pass band, quickscan band split by CI overlap, the
hard-fail rows including the "large delta but overlapping CI" hole the table
exists to close, the two-seed conjunction, and the snapshot-level aggregation.
Baseline data comes from the T8 freeze plus its backfilled baseline CI.
"""

import json
from pathlib import Path

import pytest

from scripts.adjudicate_forgetting import FORGETTING_LABELS, ForgettingAdjudicator, MatrixCandidateRanker

BASELINE_FID = 5.0
PASSING = (5.2, (4.9, 5.5))  # Δ=4%, overlapping -> pass


def write_freeze(path: Path) -> None:
    baseline = {str(label): {str(seed): {"fid_avg": BASELINE_FID} for seed in (42, 1337)} for label in FORGETTING_LABELS}
    path.write_text(json.dumps({"baseline_gen_real": baseline, "real_real_floor": {"floor_mri_t1n": {"fid_avg": 3.0}}}))


def write_baseline_ci(directory: Path) -> None:
    for label in FORGETTING_LABELS:
        for seed in (42, 1337):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"label{label}_seed{seed}.json").write_text(
                json.dumps({"comparison_tag": f"label{label}_seed{seed}", "fid_avg": {"point": BASELINE_FID, "ci95": [4.8, 5.2]}})
            )


def write_snapshot(directory: Path, label: int, seed: int, fid_ft: float, ci_ft: tuple[float, float]) -> None:
    fid_dir = directory / "fid"
    ci_dir = directory / "ci"
    fid_dir.mkdir(parents=True, exist_ok=True)
    ci_dir.mkdir(parents=True, exist_ok=True)
    (fid_dir / f"genreal_label{label}_seed{seed}.json").write_text(json.dumps({"comparison_tag": f"genreal_label{label}_seed{seed}", "fid_avg": fid_ft}))
    (ci_dir / f"genreal_label{label}_seed{seed}.json").write_text(
        json.dumps({"comparison_tag": f"genreal_label{label}_seed{seed}", "fid_avg": {"point": fid_ft, "ci95": list(ci_ft)}})
    )


def write_full_snapshot(directory: Path, overrides: dict[tuple[int, int], tuple[float, tuple[float, float]]] | None = None) -> Path:
    """A full 10-label x 2-seed snapshot, all passing, with per-(label, seed) overrides."""
    overrides = overrides or {}
    for label in FORGETTING_LABELS:
        for seed in (42, 1337):
            fid_ft, ci_ft = overrides.get((label, seed), PASSING)
            write_snapshot(directory, label, seed, fid_ft, ci_ft)
    return directory


def make_adjudicator(tmp_path: Path, snapshot_dir: Path) -> ForgettingAdjudicator:
    freeze = tmp_path / "fid_freeze.json"
    write_freeze(freeze)
    baseline_ci = tmp_path / "ci_baseline"
    write_baseline_ci(baseline_ci)
    return ForgettingAdjudicator(snapshot_dir=snapshot_dir, freeze_json=freeze, baseline_ci_dir=baseline_ci)


class TestAdjudicationTable:
    def test_delta_within_10pct_passes_regardless_of_ci(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (5.4, (4.9, 5.9))})  # Δ=8%

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "pass"

    def test_delta_10_to_20_with_overlapping_ci_is_quickscan(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (5.7, (5.1, 6.3))})  # Δ=14%, ft CI [5.1,6.3] overlaps [4.8,5.2]

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "quickscan"

    def test_delta_10_to_20_with_separated_ci_is_hard_fail(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (5.7, (5.3, 6.1))})  # Δ=14%, separated

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "hard_fail"

    def test_delta_over_20_is_hard_fail_even_with_overlap(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (6.3, (5.0, 7.6))})  # Δ=26%, overlapping

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "hard_fail"

    def test_delta_exactly_10pct_is_pass(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (5.5, (5.4, 5.6))})  # Δ=10.0%

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "pass"

    def test_delta_exactly_20pct_with_overlap_is_quickscan(self, tmp_path: Path) -> None:
        # The table's hard-fail row reads "> 20%": at exactly 20% the quickscan
        # band still applies ("快扫带封顶" — 20% is the band's cap, not past it).
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 42): (6.0, (5.0, 7.0))})

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["seed42"]["verdict"] == "quickscan"


class TestSeedAndSnapshotAggregation:
    def test_label_fails_if_either_seed_hard_fails(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 1337): (5.7, (5.3, 6.1))})  # hard fail (separated, 14%)

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["verdict"] == "hard_fail"

    def test_label_is_quickscan_when_one_seed_passes_one_needs_scan(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(9, 1337): (5.7, (5.1, 6.3))})  # quickscan

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["labels"]["9"]["verdict"] == "quickscan"

    def test_snapshot_passes_only_when_all_labels_pass(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot")

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["forgetting_verdict"] == "pass"

    def test_any_hard_fail_fails_the_snapshot(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot", {(10, 1337): (6.3, (5.0, 7.6))})  # hard fail

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["forgetting_verdict"] == "fail"

    def test_missing_baseline_ci_is_a_hard_error(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot")
        freeze = tmp_path / "fid_freeze.json"
        write_freeze(freeze)

        with pytest.raises(FileNotFoundError, match="label9_seed42"):
            ForgettingAdjudicator(snapshot_dir=snapshot_dir, freeze_json=freeze, baseline_ci_dir=tmp_path / "ci_baseline").adjudicate()


class TestBratsSummary:
    def test_brats_fid_summary_carries_per_seed_and_mean(self, tmp_path: Path) -> None:
        snapshot_dir = write_full_snapshot(tmp_path / "snapshot")
        for label, seed, value in ((40, 42, 10.0), (40, 1337, 12.0), (41, 42, 20.0), (41, 1337, 30.0)):
            (snapshot_dir / "fid" / f"brats_label{label}_seed{seed}.json").write_text(
                json.dumps({"comparison_tag": f"brats_label{label}_seed{seed}", "fid_avg": value})
            )

        payload = make_adjudicator(tmp_path, snapshot_dir).adjudicate()

        assert payload["brats"]["40"]["per_seed"] == {"42": 10.0, "1337": 12.0}
        assert payload["brats"]["40"]["mean"] == pytest.approx(11.0)
        assert payload["brats"]["41"]["mean"] == pytest.approx(25.0)


class TestMatrixCandidateRanker:
    def test_ranks_forgotten_pass_snapshots_by_brats_fid(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        self._write_adjudication(results / "N300e300.json", forgetting="pass", brats_mean=12.0)
        self._write_adjudication(results / "N500e300.json", forgetting="pass", brats_mean=8.0)
        self._write_adjudication(results / "N1000e300.json", forgetting="fail", brats_mean=2.0)

        ranking = MatrixCandidateRanker(adjudication_dir=results).rank()

        assert [c["run_id"] for c in ranking["candidates"]] == ["N500e300", "N300e300"]

    def test_reports_retest_branch_when_no_snapshot_passes_both(self, tmp_path: Path) -> None:
        results = tmp_path / "results"
        self._write_adjudication(results / "N300e300.json", forgetting="fail", brats_mean=12.0)
        self._write_adjudication(results / "N500e300.json", forgetting="fail", brats_mean=8.0)

        ranking = MatrixCandidateRanker(adjudication_dir=results).rank()

        assert ranking["retest"] == {"branch": "scale_up_replay", "reason": "no snapshot passes forgetting"}
        assert ranking["candidates"] == []

    @staticmethod
    def _write_adjudication(path: Path, forgetting: str, brats_mean: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"forgetting_verdict": forgetting, "brats_fid_mean": brats_mean, "run_id": path.stem}))
