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

"""Tests for the BRATS finetune run audit (ticket T9c, issue #27).

The tool has two halves and they are tested separately: ``freeze`` turns a launch's
configs into a baseline record, ``audit`` compares a finished run's disk against that
record.  Every test builds its own tiny checkpoints -- the audit reads two scalars out
of each one, so a four-element state dict is as good as a 722 MB one.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.audit_finetune_run import (
    CheckpointRecord,
    FindingKind,
    LaunchBaseline,
    RunAudit,
    RunAuditor,
    Verdict,
    WeightFingerprint,
)

N_EPOCHS = 300
SAVE_INTERVAL = 50
EXPECTED_SNAPSHOT_EPOCHS = (50, 100, 150, 200, 250, 300)
V1_SCALE_FACTOR = 0.969678
RUN_SCALE_FACTOR = 0.9915065765380859
MAIN_FILENAME = "diff_unet_3d_rflow-mr-brain_N1000.pt"


def write_checkpoint(path: Path, epoch: int, scale_factor: float = RUN_SCALE_FACTOR, with_scale_factor: bool = True) -> None:
    """Write a checkpoint shaped like ``diff_model_train.save_checkpoint``'s output."""
    payload = {
        "epoch": epoch,
        "loss": 0.91,
        "num_train_timesteps": 1000,
        "unet_state_dict": {"weight": torch.zeros(4)},
    }
    if with_scale_factor:
        payload["scale_factor"] = torch.tensor(scale_factor)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class Tier:
    """One tier's on-disk shape, built under ``root`` and freely breakable by a test."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_dir = root / "runs" / "brats-finetune-N1000-sugon-20260911"
        self.model_dir = root / "models" / "brats_finetune_N1000"
        self.run_dir.mkdir(parents=True)
        self.model_dir.mkdir(parents=True)

        self.v1_ckpt = root / "models" / "diff_unet_3d_rflow-mr-brain_v1.pt"
        write_checkpoint(self.v1_ckpt, epoch=0, scale_factor=V1_SCALE_FACTOR)

        self.train_config = root / "config_maisi_diff_model_rflow-mr-brain-brats.json"
        self.train_config.write_text(
            json.dumps({"diffusion_unet_train": {"batch_size": 1, "lr": 1e-5, "n_epochs": N_EPOCHS, "save_interval": SAVE_INTERVAL}})
        )

        self.env_config = root / "environment_N1000_sugon.json"
        self.env_config.write_text(
            json.dumps(
                {
                    "json_data_list": "./dataset_rflow-mr-brain_N1000.json",
                    "model_dir": str(self.model_dir),
                    "model_filename": MAIN_FILENAME,
                    "existing_ckpt_filepath": str(self.v1_ckpt),
                }
            )
        )

        self.snapshot_epochs = list(EXPECTED_SNAPSHOT_EPOCHS)
        for epoch in self.snapshot_epochs:
            # The trainer's own sequence: save the main checkpoint for the epoch, then copy
            # it to the snapshot name.
            write_checkpoint(self.main_path, epoch=epoch)
            shutil.copyfile(self.main_path, self.snapshot_path(epoch))

    def audit(self, baseline: LaunchBaseline | None = None) -> RunAudit:
        return RunAuditor(baseline if baseline is not None else self.freeze(), self.baseline_path()).run()

    def snapshot_path(self, epoch: int) -> Path:
        return self.model_dir / f"ckpt_epoch{epoch}.pt"

    @property
    def main_path(self) -> Path:
        return self.model_dir / MAIN_FILENAME

    def freeze(self) -> LaunchBaseline:
        return LaunchBaseline.freeze(self.train_config, self.env_config)

    def baseline_path(self) -> Path:
        return self.run_dir / "launch_baseline.json"


@pytest.fixture
def tier(tmp_path: Path) -> Tier:
    return Tier(tmp_path)


class TestWeightFingerprint:
    def test_captures_size_mtime_and_sha256(self, tmp_path: Path) -> None:
        target = tmp_path / "weights.pt"
        target.write_bytes(b"v1-weights")

        fingerprint = WeightFingerprint.capture(target)

        assert fingerprint.size == len(b"v1-weights")
        assert fingerprint.mtime_ns == target.stat().st_mtime_ns
        assert fingerprint.sha256 == hashlib.sha256(b"v1-weights").hexdigest()

    def test_an_untouched_file_differs_by_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "weights.pt"
        target.write_bytes(b"v1-weights")

        assert WeightFingerprint.capture(target).differences(WeightFingerprint.capture(target)) == ()

    def test_the_same_bytes_under_another_path_are_not_a_difference(self, tmp_path: Path) -> None:
        # The cluster mounts its private disk at two paths, so one untouched v1 checkpoint is
        # legitimately spelled two ways; only the content fields decide.
        first, second = tmp_path / "a.pt", tmp_path / "b.pt"
        first.write_bytes(b"v1-weights")
        shutil.copyfile(first, second, follow_symlinks=True)
        second.chmod(first.stat().st_mode)
        os.utime(second, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))

        assert WeightFingerprint.capture(first).differences(WeightFingerprint.capture(second)) == ()

    def test_a_touched_file_differs_in_mtime_only(self, tmp_path: Path) -> None:
        target = tmp_path / "weights.pt"
        target.write_bytes(b"v1-weights")
        before = WeightFingerprint.capture(target)

        target.touch()

        assert before.differences(WeightFingerprint.capture(target)) == ("mtime_ns",)

    def test_rewritten_content_differs_in_sha256_and_mtime(self, tmp_path: Path) -> None:
        target = tmp_path / "weights.pt"
        target.write_bytes(b"v1-weights")
        before = WeightFingerprint.capture(target)

        target.write_bytes(b"finetuned!")

        differences = before.differences(WeightFingerprint.capture(target))
        assert "sha256" in differences
        assert "mtime_ns" in differences

    def test_round_trips_through_dict(self, tmp_path: Path) -> None:
        target = tmp_path / "weights.pt"
        target.write_bytes(b"v1-weights")

        fingerprint = WeightFingerprint.capture(target)

        assert WeightFingerprint.from_dict(fingerprint.to_dict()) == fingerprint


class TestFreezeLaunchBaseline:
    def test_derives_expected_snapshot_epochs_from_the_train_config(self, tier: Tier) -> None:
        assert tier.freeze().expected_snapshot_epochs == EXPECTED_SNAPSHOT_EPOCHS

    def test_takes_the_three_paths_from_the_env_config(self, tier: Tier) -> None:
        baseline = tier.freeze()

        assert Path(baseline.model_dir) == tier.model_dir
        assert baseline.model_filename == MAIN_FILENAME
        assert Path(baseline.v1_weights.path) == tier.v1_ckpt

    def test_records_v1_declared_scale_factor(self, tier: Tier) -> None:
        assert tier.freeze().v1_declared_scale_factor == pytest.approx(V1_SCALE_FACTOR)

    def test_a_save_interval_of_zero_expects_no_snapshots(self, tier: Tier) -> None:
        tier.train_config.write_text(json.dumps({"diffusion_unet_train": {"n_epochs": N_EPOCHS, "save_interval": 0}}))

        assert tier.freeze().expected_snapshot_epochs == ()

    def test_save_interval_must_divide_into_the_epoch_count(self, tier: Tier) -> None:
        tier.train_config.write_text(json.dumps({"diffusion_unet_train": {"n_epochs": N_EPOCHS, "save_interval": 70}}))

        with pytest.raises(ValueError, match="save_interval"):
            tier.freeze()

    def test_saves_and_reloads(self, tier: Tier) -> None:
        baseline = tier.freeze()

        baseline.save(tier.baseline_path())

        assert LaunchBaseline.load(tier.baseline_path()) == baseline


class TestReadCheckpointRecord:
    def test_reads_epoch_and_scale_factor(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt_epoch50.pt"
        write_checkpoint(path, epoch=50, scale_factor=1.25)

        record = CheckpointRecord.read(path)

        assert record.epoch == 50
        assert record.scale_factor == pytest.approx(1.25)
        assert record.size == path.stat().st_size

    def test_a_checkpoint_without_scale_factor_reads_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt_epoch50.pt"
        write_checkpoint(path, epoch=50, with_scale_factor=False)

        assert CheckpointRecord.read(path).scale_factor is None


class TestAuditRun:
    def test_a_finished_untouched_tier_passes(self, tier: Tier) -> None:
        baseline = tier.freeze()
        baseline.save(tier.baseline_path())

        report = tier.audit(baseline)

        assert report.verdict == Verdict.PASS
        assert report.findings == ()
        assert report.missing_snapshot_epochs == ()

    def test_inventories_every_expected_snapshot(self, tier: Tier) -> None:
        report = tier.audit()

        assert tuple(record.epoch for record in report.snapshots) == EXPECTED_SNAPSHOT_EPOCHS

    def test_records_each_snapshot_scale_factor_against_v1s_declared_value(self, tier: Tier) -> None:
        report = tier.audit()

        deviations = {snapshot["epoch"]: snapshot["scale_factor_relative_deviation"] for snapshot in report.to_dict()["snapshots"]}
        expected = (RUN_SCALE_FACTOR - V1_SCALE_FACTOR) / V1_SCALE_FACTOR
        assert deviations == {epoch: pytest.approx(expected) for epoch in EXPECTED_SNAPSHOT_EPOCHS}

    def test_proves_the_main_checkpoint_is_the_final_snapshot(self, tier: Tier) -> None:
        report = tier.audit()

        assert report.main_matches_final_snapshot is True

    def test_a_rewritten_main_checkpoint_is_caught(self, tier: Tier) -> None:
        baseline = tier.freeze()
        write_checkpoint(tier.main_path, epoch=N_EPOCHS, scale_factor=0.5)

        report = tier.audit(baseline)

        assert FindingKind.MAIN_CHECKPOINT_STALE in report.finding_kinds
        assert report.verdict == Verdict.FAIL

    def test_detects_v1_weights_touched_after_the_freeze(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.v1_ckpt.write_bytes(b"someone overwrote v1")

        report = tier.audit(baseline)

        assert FindingKind.WEIGHTS_MODIFIED in report.finding_kinds
        assert report.verdict == Verdict.FAIL
        assert report.to_dict()["v1_weights"]["modified_fields"]

    def test_detects_v1_weights_touched_without_content_change(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.v1_ckpt.touch()

        report = tier.audit(baseline)

        assert FindingKind.WEIGHTS_MODIFIED in report.finding_kinds

    def test_a_vanished_v1_checkpoint_is_a_finding_not_a_crash(self, tier: Tier) -> None:
        # Nothing the run produces can be compared against a baseline that is gone, and an
        # audit that raises instead of reporting would leave the operator with a traceback.
        baseline = tier.freeze()
        tier.v1_ckpt.unlink()

        report = tier.audit(baseline)
        payload = json.dumps(report.to_dict(), allow_nan=False)

        assert FindingKind.WEIGHTS_MODIFIED in report.finding_kinds
        assert report.verdict == Verdict.FAIL
        assert json.loads(payload)["v1_weights"] is None

    def test_a_truncated_snapshot_is_named_rather_than_raised(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.snapshot_path(150).write_bytes(b"half a checkpo")

        report = tier.audit(baseline)
        payload = json.dumps(report.to_dict(), allow_nan=False)

        assert FindingKind.CHECKPOINT_UNREADABLE in report.finding_kinds
        assert report.verdict == Verdict.FAIL
        assert [record["epoch"] for record in json.loads(payload)["snapshots"]][2] is None

    def test_a_truncated_main_checkpoint_is_named_rather_than_raised(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.main_path.write_bytes(b"half a checkpo")

        report = tier.audit(baseline)

        assert FindingKind.CHECKPOINT_UNREADABLE in report.finding_kinds
        assert report.verdict == Verdict.FAIL

    def test_detects_a_missing_snapshot(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.snapshot_path(150).unlink()

        report = tier.audit(baseline)

        assert FindingKind.SNAPSHOT_MISSING in report.finding_kinds
        assert report.missing_snapshot_epochs == (150,)
        assert report.verdict == Verdict.FAIL

    def test_detects_a_missing_main_checkpoint(self, tier: Tier) -> None:
        baseline = tier.freeze()
        tier.main_path.unlink()

        report = tier.audit(baseline)

        assert FindingKind.MAIN_CHECKPOINT_MISSING in report.finding_kinds
        assert report.main_checkpoint is None

    def test_an_unfinished_run_reports_the_epochs_still_missing(self, tier: Tier) -> None:
        # Mid-run shape: the main checkpoint is at the current epoch and only the
        # snapshots already passed are on disk.
        baseline = tier.freeze()
        for epoch in (150, 200, 250, 300):
            tier.snapshot_path(epoch).unlink()
        write_checkpoint(tier.main_path, epoch=137)

        report = tier.audit(baseline)

        assert report.missing_snapshot_epochs == (150, 200, 250, 300)
        assert FindingKind.MAIN_CHECKPOINT_INCOMPLETE in report.finding_kinds
        assert FindingKind.MAIN_CHECKPOINT_STALE not in report.finding_kinds

    def test_detects_a_snapshot_carrying_the_wrong_epoch(self, tier: Tier) -> None:
        baseline = tier.freeze()
        write_checkpoint(tier.snapshot_path(150), epoch=151)

        report = tier.audit(baseline)

        assert FindingKind.EPOCH_MISMATCH in report.finding_kinds

    def test_detects_a_snapshot_that_lost_its_scale_factor(self, tier: Tier) -> None:
        baseline = tier.freeze()
        write_checkpoint(tier.snapshot_path(100), epoch=100, with_scale_factor=False)

        report = tier.audit(baseline)

        assert FindingKind.SCALE_FACTOR_MISSING in report.finding_kinds
        assert report.verdict == Verdict.FAIL

    def test_a_non_finite_scale_factor_is_json_safe_and_flagged(self, tier: Tier) -> None:
        baseline = tier.freeze()
        write_checkpoint(tier.snapshot_path(100), epoch=100, scale_factor=float("inf"))

        report = tier.audit(baseline)
        payload = json.dumps(report.to_dict(), allow_nan=False)

        assert FindingKind.SCALE_FACTOR_MISSING in report.finding_kinds
        assert json.loads(payload)["snapshots"][1]["scale_factor"] is None

    def test_stamps_the_report_with_when_it_ran(self, tier: Tier) -> None:
        report = tier.audit()

        assert report.generated_at_utc.endswith("+00:00")

    def test_a_tier_frozen_before_its_snapshots_exist_still_audits(self, tier: Tier) -> None:
        # The freeze runs at launch, when model_dir is empty -- it must not need the
        # outputs it will later be checked against.
        baseline = tier.freeze()
        for epoch in tier.snapshot_epochs:
            tier.snapshot_path(epoch).unlink()
        tier.main_path.unlink()

        report = tier.audit(baseline)

        assert report.missing_snapshot_epochs == EXPECTED_SNAPSHOT_EPOCHS


class TestCli:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "scripts.audit_finetune_run", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

    def test_freeze_then_audit_exits_zero_on_a_clean_tier(self, tier: Tier) -> None:
        freeze = self._run("freeze", "-c", str(tier.train_config), "-e", str(tier.env_config), "--report", str(tier.baseline_path()))
        assert freeze.returncode == 0, freeze.stderr

        audit = self._run("audit", "--baseline", str(tier.baseline_path()), "--report", str(tier.run_dir / "run_audit.json"))

        assert audit.returncode == 0, audit.stdout + audit.stderr
        assert "PASS" in audit.stdout

    def test_audit_exits_one_and_names_the_finding_on_a_broken_tier(self, tier: Tier) -> None:
        self._run("freeze", "-c", str(tier.train_config), "-e", str(tier.env_config), "--report", str(tier.baseline_path()))
        tier.snapshot_path(250).unlink()

        audit = self._run("audit", "--baseline", str(tier.baseline_path()))

        assert audit.returncode == 1
        assert "snapshot_missing" in audit.stdout
