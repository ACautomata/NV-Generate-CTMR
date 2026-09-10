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

"""Tests for scripts.refill_replay_manifest — the post-spine-filter top-up (ticket T6 #22).

The contract under test is the one ticket T2 froze: a capped draw is a prefix of each
modality's seeded ordering, so refilling means dropping the rejected rows and continuing
down the same sequence -- never re-drawing.
"""

import csv
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.create_replay_manifest import MANIFEST_COLUMNS
from scripts.mrrate_series import WHOLE_BRAIN_LABEL
from scripts.refill_replay_manifest import (
    RefilledReplayManifests,
    ReplayOrderingIndex,
    TierRoster,
    TierTarget,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest_row() -> Callable[[int, str], dict[str, str]]:
    """Factory for one manifest row; ``index`` is its position, so the study uid is unique."""

    def build(index: int, modality: str) -> dict[str, str]:
        study = f"STUDY{index:04d}"
        series = f"{modality}-raw-axi"
        return {
            "patient_uid": f"{index}",
            "study_uid": study,
            "series_id": series,
            "modality": modality,
            "label": WHOLE_BRAIN_LABEL[modality],
            "split": "Train",
            "image_path": f"mri/batch00/{study}/img/{study}_{series}.nii.gz",
        }

    return build


@pytest.fixture
def write_manifest() -> Callable[[Path, list[dict[str, str]]], Path]:
    """Factory writing a manifest CSV in the frozen column order."""

    def build(path: Path, rows: list[dict[str, str]]) -> Path:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(MANIFEST_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        return path

    return build


@pytest.fixture
def ordering(tmp_path: Path, manifest_row: Callable[[int, str], dict[str, str]], write_manifest: Callable) -> Path:
    """Orderings per modality, interleaved so position in the file never equals position per modality."""
    rows = []
    for index in range(12):
        rows.append(manifest_row(index * 10, "t1w"))
        rows.append(manifest_row(index * 10 + 1, "flair"))
        rows.append(manifest_row(index * 10 + 2, "mra"))
    return write_manifest(tmp_path / "ordering.csv", rows)


@pytest.fixture
def rejected(tmp_path: Path, ordering: Path) -> Path:
    """The spine filter's record: the 1st and 4th t1w candidates and the 2nd flair one are spine."""
    rows = list(csv.DictReader(ordering.open()))
    t1w = [row for row in rows if row["modality"] == "t1w"]
    flair = [row for row in rows if row["modality"] == "flair"]
    rejected_rows = [
        {**t1w[0], "verdict": "reject", "mask_voxel_ratio": "0.001", "fov_mm": "300 300 300", "reasons": "too small"},
        {**t1w[3], "verdict": "reject", "mask_voxel_ratio": "0.001", "fov_mm": "300 300 300", "reasons": "too small"},
        {**flair[1], "verdict": "reject", "mask_voxel_ratio": "0.001", "fov_mm": "300 300 300", "reasons": "too small"},
    ]
    path = tmp_path / "rejected.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*MANIFEST_COLUMNS, "verdict", "mask_voxel_ratio", "fov_mm", "reasons"])
        writer.writeheader()
        writer.writerows(rejected_rows)
    return path


class TestTierTarget:
    def test_cap_rules_follow_the_layered_policy(self) -> None:
        target = TierTarget("N1000", 1000)
        assert target.policy.cap_for("t1w") == 1000
        assert target.policy.cap_for("flair") == 1000
        assert target.policy.cap_for("swi") == 1000
        assert target.policy.cap_for("t2w") == 669
        assert TierTarget("N300", 300).policy.cap_for("t2w") == 300

    def test_mra_is_uncapped_like_the_generator(self) -> None:
        """MRA takes every available subject, so its N never bounds the roster (section 3.3)."""
        for n in (300, 500, 1000):
            assert TierTarget(f"N{n}", n).policy.cap_for("mra") is None

    def test_output_filename_names_the_tier(self) -> None:
        assert TierTarget("N500", 500).output_filename == "replay_manifest_N500_final.csv"

    def test_parses_and_rejects_specifications(self) -> None:
        assert TierTarget.parse("N1000=1000") == TierTarget("N1000", 1000)
        with pytest.raises(ValueError, match="unparseable tier"):
            TierTarget.parse("N1000")


class TestReplayOrderingIndex:
    def test_rejected_rows_leave_every_modality(self, ordering: Path, rejected: Path) -> None:
        index = ReplayOrderingIndex(ordering, rejected)
        assert len(index.rejected) == 3
        assert index.modality("t1w").available == 10
        assert index.modality("flair").available == 11
        assert index.modality("mra").available == 12

    def test_a_missing_reject_file_means_nothing_was_rejected(self, ordering: Path) -> None:
        assert ReplayOrderingIndex(ordering, None).modality("t1w").available == 12


class TestTierRoster:
    def test_refill_skips_rejects_and_continues_the_ordering(self, ordering: Path, rejected: Path) -> None:
        index = ReplayOrderingIndex(ordering, rejected)
        rows = [row for row in TierRoster(TierTarget("N3", 3), index).rows() if row["modality"] == "t1w"]

        # Candidate 1 (STUDY0000) and candidate 4 (STUDY0030) are spine; the cap-3 roster is the
        # first three survivors, so it reaches one candidate further down the same ordering.
        assert [row["study_uid"] for row in rows] == ["STUDY0010", "STUDY0020", "STUDY0040"]

    def test_each_tier_takes_a_separate_cap_from_the_same_ordering(self, ordering: Path, rejected: Path) -> None:
        index = ReplayOrderingIndex(ordering, rejected)
        small = [row["study_uid"] for row in TierRoster(TierTarget("N2", 2), index).rows()]
        large = [row["study_uid"] for row in TierRoster(TierTarget("N5", 5), index).rows()]
        assert len(small) == 16  # (2 t1w + 2 flair) + all 12 MRA
        assert len(large) == 22  # (5 t1w + 5 flair) + all 12 MRA
        assert set(small) < set(large)

    def test_shortage_is_reported_not_padded(self, tmp_path: Path, ordering: Path) -> None:
        index = ReplayOrderingIndex(ordering, None)
        roster = TierRoster(TierTarget("N50", 50), index)
        assert roster.shortages() == {"t1w": 38, "flair": 38}  # MRA is uncapped, never short
        assert len(roster.rows()) == 36  # everything available, nothing invented

    def test_mra_takes_the_whole_pool_in_every_tier(self, ordering: Path) -> None:
        index = ReplayOrderingIndex(ordering, None)
        for n in (2, 5, 50):
            mra = [row for row in TierRoster(TierTarget(f"N{n}", n), index).rows() if row["modality"] == "mra"]
            assert len(mra) == 12

    def test_written_roster_carries_the_manifest_columns(self, tmp_path: Path, ordering: Path, rejected: Path) -> None:
        index = ReplayOrderingIndex(ordering, rejected)
        summary = TierRoster(TierTarget("N4", 4), index).write(tmp_path)

        assert summary["rows"] == 20  # (4 t1w + 4 flair) + all 12 MRA
        assert summary["shortages"] == {}
        with (tmp_path / "replay_manifest_N4_final.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert list(rows[0]) == list(MANIFEST_COLUMNS)
        assert {row["modality"] for row in rows} == {"t1w", "flair", "mra"}


class TestRefilledReplayManifests:
    def test_accepted_union_covers_every_tier_once(self, ordering: Path, rejected: Path) -> None:
        refilled = RefilledReplayManifests(ordering, rejected)
        union = refilled.accepted_union([TierTarget("N2", 2), TierTarget("N5", 5)])
        assert len(union) == 22  # the larger tier's rows; the smaller adds nothing new
        assert all("STUDY" in row["study_uid"] for row in union.values())

    def test_union_never_contains_a_rejected_series(self, ordering: Path, rejected: Path) -> None:
        refilled = RefilledReplayManifests(ordering, rejected)
        union = refilled.accepted_union([TierTarget("N12", 12)])
        assert refilled.rejected.isdisjoint(union)


class TestUnrefillableModalities:
    """T2w and MRA cannot be refilled from a bigger N, so their shortfalls are permanent."""

    def test_t2w_is_capped_so_raising_n_adds_no_candidates(self) -> None:
        assert TierTarget("N1000", 1000).policy.cap_for("t2w") == 669
        assert TierTarget("N5000", 5000).policy.cap_for("t2w") == 669

    def test_a_rejected_t2w_leaves_its_tier_one_short(
        self, tmp_path: Path, manifest_row: Callable[[int, str], dict[str, str]], write_manifest: Callable
    ) -> None:
        rows = [manifest_row(index, "t2w") for index in range(4)]
        ordering = write_manifest(tmp_path / "ordering.csv", rows)
        rejected_path = tmp_path / "rejected.csv"
        with rejected_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[*MANIFEST_COLUMNS, "verdict"])
            writer.writeheader()
            writer.writerow({**rows[0], "verdict": "reject"})

        roster = TierRoster(TierTarget("N4", 4), ReplayOrderingIndex(ordering, rejected_path))

        assert len(roster.rows()) == 3  # one short, never padded from outside the ordering
        assert roster.shortages() == {"t2w": 1}


class TestCommandLine:
    def test_end_to_end_writes_each_tier(self, tmp_path: Path, ordering: Path, rejected: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "refill_replay_manifest",
                "--ordering",
                str(ordering),
                "--rejected",
                str(rejected),
                "--tier",
                "N2=2",
                "--tier",
                "N5=5",
                "--output-dir",
                str(tmp_path),
                "--topup-manifest",
                str(tmp_path / "topup.csv"),
            ],
        )
        main()

        assert (tmp_path / "replay_manifest_N2_final.csv").is_file()
        assert (tmp_path / "replay_manifest_N5_final.csv").is_file()
        with (tmp_path / "topup.csv").open(newline="") as file:
            assert len(list(csv.DictReader(file))) == 22

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run([sys.executable, "-m", "scripts.refill_replay_manifest", "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0
        assert "--ordering" in result.stdout
