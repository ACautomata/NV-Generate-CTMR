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

"""Tests for the training-log watcher (ticket T9a, issue #25; spec #13 section 4.7).

The training discipline is "no resume, no interruption": once a tier is launched it runs to
completion or is killed and retrained from scratch, and the decision to kill is the human's.
An agent therefore only has to notice the two failure modes the spec names -- a loss that stops
being finite, and a loss that climbs monotonically for ~20 epochs -- and raise them. The watcher
parses the training log into step/epoch records, applies each health criterion, and writes a
report whose verdict a polling shell can act on via the exit code.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.watch_training_log import (
    ALARM,
    HEALTHY,
    EpochRecord,
    MonotonicRiseCriterion,
    NonFiniteLossCriterion,
    StepRecord,
    TrainingLogParser,
    TrainingLogWatcher,
)

STEP_LINE = "[{ts}][ INFO](training) - [{day} {clock}] epoch {epoch}, iter {iter}/{total}, loss: {loss}, lr: {lr}."
EPOCH_LINE = "[{ts}][ INFO](training) - epoch {epoch} average loss: {loss}."
SNAPSHOT_LINE = "[{ts}][ INFO](training) - Snapshot saved to {path}."


def step_line(epoch: int, iter_: int, loss: float, ts: str = "2026-09-11 17:02:08.979", total: int = 3714) -> str:
    # The logger prints the step timestamp with milliseconds but the inner epoch stamp without.
    return STEP_LINE.format(ts=ts, day=ts[:10], clock=ts[11:19], epoch=epoch, iter=iter_, total=total, loss=loss, lr="0.000010000000")


def epoch_line(epoch: int, loss: float, ts: str = "2026-09-11 17:04:30.847") -> str:
    return EPOCH_LINE.format(ts=ts, epoch=epoch, loss=loss)


def snapshot_line(epoch: int, path: str = "/models/brats_finetune_N300/ckpt_epoch50.pt") -> str:
    return SNAPSHOT_LINE.format(ts="2026-09-11 17:16:47.138", path=path)


class TestTrainingLogParser:
    def test_parses_step_records_with_epoch_iter_loss_and_lr(self) -> None:
        log = "\n".join([step_line(1, 1, 1.0373), step_line(2, 7, 0.8992, ts="2026-09-11 17:02:25.793")])

        steps = TrainingLogParser().parse(log).steps

        assert [s.epoch for s in steps] == [1, 2]
        assert [s.iter for s in steps] == [1, 7]
        assert steps[0].loss == 1.0373
        assert steps[0].iterations_per_epoch == 3714

    def test_parses_epoch_average_loss_records(self) -> None:
        log = "\n".join([step_line(1, 1, 1.0), epoch_line(1, 0.9352)])

        parsed = TrainingLogParser().parse(log)

        assert len(parsed.epochs) == 1
        assert parsed.epochs[0] == EpochRecord(epoch=1, average_loss=0.9352)

    def test_parses_snapshot_paths(self) -> None:
        log = snapshot_line(50, "/models/brats_finetune_N300/ckpt_epoch50.pt")

        assert TrainingLogParser().parse(log).snapshots == ["/models/brats_finetune_N300/ckpt_epoch50.pt"]

    def test_ignores_unrelated_log_lines(self) -> None:
        log = "\n".join(
            [
                "W0911 17:01:58.825000 3599076 torch/distributed/run.py:982] Setting OMP_NUM_THREADS",
                "[2026-09-11 17:02:03.498][ INFO](training) - [config] num_epochs -> 300.",
                step_line(1, 1, 1.0),
            ]
        )

        parsed = TrainingLogParser().parse(log)

        assert len(parsed.steps) == 1
        assert parsed.epochs == []

    def test_captures_non_finite_loss_verbatim(self) -> None:
        # The logger prints whatever float it holds: `nan`, `inf`, `-inf`. A parser that
        # drops these lines would report a healthy run right up to the point it is not.
        log = "\n".join([step_line(3, 10, float("nan")), step_line(3, 11, float("inf"))])

        losses = [s.loss for s in TrainingLogParser().parse(log).steps]

        assert losses[0] != losses[0]  # nan
        assert losses[1] == float("inf")

    def test_parses_scientific_notation_losses(self) -> None:
        log = step_line(1, 1, 1e-05)

        assert TrainingLogParser().parse(log).steps[0].loss == 1e-05

    def test_empty_log_parses_to_empty_records(self) -> None:
        parsed = TrainingLogParser().parse("")

        assert parsed.steps == []
        assert parsed.epochs == []
        assert parsed.snapshots == []


class TestNonFiniteLossCriterion:
    def test_no_finding_on_finite_losses(self) -> None:
        log = "\n".join([step_line(1, 1, 1.0), epoch_line(1, 0.9)])

        assert NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log)) == []

    def test_flags_a_nan_step_loss(self) -> None:
        log = step_line(7, 12, float("nan"))

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].kind == "non_finite_loss"
        assert findings[0].epoch == 7

    def test_flags_a_non_finite_epoch_average(self) -> None:
        log = epoch_line(9, float("inf"))

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].epoch == 9

    def test_reports_only_the_first_non_finite_step_and_counts_the_rest(self) -> None:
        # Once a run diverges, every subsequent step logs a non-finite loss: one report per step
        # would make the status file a copy of the log (a nan-every-step run is ~1.1M lines) and
        # would be rewritten in full every poll. The operator needs where the break started and
        # how far it has spread, not each line of it.
        log = "\n".join([step_line(3, 10, float("nan")), step_line(3, 11, float("nan")), step_line(3, 12, float("-inf"))])

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].epoch == 3
        assert "(first of 3 non-finite steps)" in findings[0].detail

    def test_reports_only_the_first_non_finite_epoch_average(self) -> None:
        log = "\n".join([epoch_line(9, float("inf")), epoch_line(10, float("nan"))])

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].epoch == 9
        assert "(first of 2 non-finite epochs)" in findings[0].detail


class TestMonotonicRiseCriterion:
    @staticmethod
    def _log_with_epochs(losses: list[float]) -> str:
        return "\n".join(epoch_line(index + 1, loss) for index, loss in enumerate(losses))

    def test_no_finding_on_a_shorter_run_than_the_window(self) -> None:
        # Fewer epochs than the window cannot establish the ~20-epoch trend; staying silent
        # is correct, not a missed alarm.
        log = self._log_with_epochs([1.0, 1.1, 1.2])

        assert MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(log)) == []

    def test_no_finding_when_the_window_is_not_strictly_rising(self) -> None:
        losses = [1.0] * 19 + [0.9]

        assert MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(self._log_with_epochs(losses))) == []

    def test_flags_a_strictly_rising_window(self) -> None:
        losses = [1.0 + 0.01 * index for index in range(20)]

        findings = MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(self._log_with_epochs(losses)))

        assert len(findings) == 1
        assert findings[0].kind == "monotonic_rise"
        assert findings[0].epoch == 20

    def test_a_rise_that_has_since_stabilised_is_still_reported(self) -> None:
        # Every window is examined, not just the trailing one: a run that climbed for 20 epochs
        # and then sat at the higher loss is exactly the diverged-then-stuck shape the operator
        # needs to see, even once the climb itself is long past.
        losses = [1.0 + 0.01 * index for index in range(20)] + [5.0] * 40

        findings = MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(self._log_with_epochs(losses)))

        assert len(findings) == 1
        assert findings[0].epoch == 20

    def test_recovers_after_a_fall_breaks_the_rise(self) -> None:
        losses = [1.0 + 0.01 * index for index in range(19)] + [0.2]

        assert MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(self._log_with_epochs(losses))) == []

    def test_one_finding_per_sustained_rise_not_one_per_window(self) -> None:
        # 30 rising epochs contain 11 windows of 20; the operator needs one alarm naming the
        # epoch the trend was established at, not eleven restatements of it.
        losses = [1.0 + 0.01 * index for index in range(30)]

        findings = MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse(self._log_with_epochs(losses)))

        assert len(findings) == 1
        assert findings[0].epoch == 20

    def test_a_gap_in_epoch_numbering_does_not_count_as_consecutive(self) -> None:
        # A rotated or truncated log can leave epoch numbers non-adjacent; a rise across the
        # gap is not the consecutive-epoch trend the criterion exists to catch.
        lines = [epoch_line(index + 1, 1.0 + 0.01 * index) for index in range(10)]
        lines += [epoch_line(100 + index, 2.0 + 0.01 * index) for index in range(10)]

        assert MonotonicRiseCriterion(window=20).inspect(TrainingLogParser().parse("\n".join(lines))) == []


class TestTrainingLogWatcher:
    def test_healthy_run_reports_ok_and_advances_progress(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([step_line(1, 1, 1.0), epoch_line(1, 0.9)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == HEALTHY
        assert report["progress"]["epochs_completed"] == 1
        assert report["progress"]["total_epochs"] == 300
        assert report["findings"] == []

    def test_progress_carries_how_far_into_the_current_epoch_the_run_is(self, tmp_path: Path) -> None:
        # Epochs take ~20 minutes here, so `epochs_completed` alone leaves the reader unsure
        # whether the next snapshot is imminent or 20 minutes out.
        log_path = tmp_path / "train.log"
        log_path.write_text(step_line(4, 1000, 1.0, total=3714))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["progress"]["epoch_fraction"] == pytest.approx(1000 / 3714)

    def test_alarm_on_non_finite_loss(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text(step_line(4, 9, float("nan")))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert [f["kind"] for f in report["findings"]] == ["non_finite_loss"]

    def test_alarm_on_monotonic_rise(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join(epoch_line(index + 1, 1.0 + 0.01 * index) for index in range(20)))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert [f["kind"] for f in report["findings"]] == ["monotonic_rise"]

    def test_report_records_step_timing_from_the_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "\n".join(
                [
                    step_line(1, 1, 1.0, ts="2026-09-11 17:02:08.979"),
                    step_line(1, 101, 0.9, ts="2026-09-11 17:02:42.479"),
                ]
            )
        )

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["progress"]["steps_observed"] == 2
        assert report["timing"]["seconds_per_step"] == pytest.approx(0.335, abs=1e-3)

    def test_report_records_snapshots_and_remaining_epochs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([epoch_line(50, 0.9), snapshot_line(50)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["progress"]["epochs_remaining"] == 250
        assert report["snapshots"] == ["/models/brats_finetune_N300/ckpt_epoch50.pt"]

    def test_writes_the_report_to_disk_when_a_path_is_given(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text(epoch_line(1, 0.9))
        report_path = tmp_path / "status.json"

        TrainingLogWatcher(log_path, total_epochs=300, report_path=report_path).run()

        assert json.loads(report_path.read_text())["verdict"] == HEALTHY

    def test_missing_log_file_reports_alarm_rather_than_raising(self, tmp_path: Path) -> None:
        # A poller that crashes on a rotated/absent log stops watching entirely; an ALARM
        # report is the honest signal -- the run's health is unknown.
        report = TrainingLogWatcher(tmp_path / "absent.log", total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert report["findings"][0]["kind"] == "log_unavailable"

    def test_records_the_step_domain_average_loss_per_epoch(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([step_line(1, 1, 1.0), step_line(1, 2, 0.5)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert [(point["epoch"], point["step_mean_loss"]) for point in report["loss_curve"]] == [(1, 0.75)]

    def test_records_the_reported_epoch_average_alongside_the_step_mean(self, tmp_path: Path) -> None:
        # The training script's own `epoch N average loss` is the all-reduced number the run
        # is judged on; the step-domain mean is recomputed here and agrees only approximately.
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([step_line(1, 1, 1.0), epoch_line(1, 0.9352)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["loss_curve"] == [{"epoch": 1, "step_mean_loss": 1.0, "reported_average_loss": 0.9352}]


class TestEpochRecord:
    def test_epoch_record_carries_its_average_loss(self) -> None:
        assert EpochRecord(epoch=3, average_loss=0.25).average_loss == 0.25


class TestStepRecord:
    def test_step_record_reports_its_iteration_progress(self) -> None:
        record = StepRecord(
            epoch=2,
            iter=15,
            iterations_per_epoch=100,
            loss=0.5,
            lr=1e-05,
            timestamp=datetime(2026, 9, 11, 17, 2, 8, 979000),
        )

        assert record.fractional_epoch_position == pytest.approx(0.15)
