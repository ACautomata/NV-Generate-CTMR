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
import os
import time
from pathlib import Path

import pytest

from scripts.watch_training_log import (
    ALARM,
    DEFAULT_STARTUP_GRACE_SECONDS,
    HEALTHY,
    LOG_UNAVAILABLE,
    MONOTONIC_RISE,
    NO_TRAINING_STATE,
    NON_FINITE_LOSS,
    EpochRecord,
    MonotonicRiseCriterion,
    NonFiniteLossCriterion,
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


def snapshot_line(path: str = "/models/brats_finetune_N300/ckpt_epoch50.pt") -> str:
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
        log = snapshot_line("/models/brats_finetune_N300/ckpt_epoch50.pt")

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

    def test_a_single_transient_non_finite_step_does_not_alarm(self) -> None:
        # Under AMP a one-off inf loss is expected: GradScaler skips that step, backs the scale
        # off, and training continues. This is the tool's highest false-positive surface, and a
        # false alarm costs the whole tier -- no resume, so spec section 4.7 loosens the criteria
        # precisely to avoid that.
        log = "\n".join([step_line(3, 10, 1.0), step_line(3, 11, float("nan")), step_line(3, 12, 0.9)])

        assert NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log)) == []

    def test_a_sustained_run_of_non_finite_steps_alarms(self) -> None:
        log = "\n".join(step_line(7, 100 + index, float("nan")) for index in range(20))

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].kind == NON_FINITE_LOSS
        assert findings[0].epoch == 7
        assert "20 consecutive steps" in findings[0].detail

    def test_a_finite_step_resets_the_streak(self) -> None:
        # Recovery is not divergence: two short bursts either side of a healthy step must not
        # add up to one long one.
        log = "\n".join(
            [step_line(3, 10 + index, float("nan")) for index in range(19)]
            + [step_line(3, 29, 0.5)]
            + [step_line(3, 30 + index, float("nan")) for index in range(19)]
        )

        assert NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log)) == []

    def test_the_streak_threshold_is_configurable(self) -> None:
        log = "\n".join(step_line(2, 5 + index, float("inf")) for index in range(3))

        findings = NonFiniteLossCriterion(consecutive_steps=3).inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].epoch == 2

    def test_a_non_positive_streak_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="consecutive_steps"):
            NonFiniteLossCriterion(consecutive_steps=0)

    def test_a_sustained_run_of_non_finite_epoch_averages_alarms(self) -> None:
        # The per-iter line is guarded by `local_rank == 0` in `diff_model_train.py` while the
        # epoch average is all-reduced across ranks, so a nonzero rank that diverges never appears
        # in the step records -- the average is the only place it surfaces, and a watcher reading
        # steps alone calls such a run healthy for the rest of its life.
        log = "\n".join(epoch_line(epoch, float("inf")) for epoch in range(5, 8))

        findings = NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log))

        assert len(findings) == 1
        assert findings[0].kind == NON_FINITE_LOSS
        assert "3 consecutive epochs" in findings[0].detail
        assert findings[0].epoch == 7

    def test_a_lone_non_finite_epoch_average_does_not_alarm(self) -> None:
        # The tolerance is the step criterion's, for the same reason: one bad step on any rank
        # makes its whole epoch average non-finite, so a single one is a transient, not a
        # divergence. Full-epoch coverage makes a repeat over consecutive epochs the real signal.
        log = "\n".join([epoch_line(5, 0.93), epoch_line(6, float("inf")), epoch_line(7, 0.92)])

        assert NonFiniteLossCriterion().inspect(TrainingLogParser().parse(log)) == []


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
        assert findings[0].kind == MONOTONIC_RISE
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
        log_path.write_text("\n".join(step_line(4, 9 + index, float("nan")) for index in range(20)))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert [f["kind"] for f in report["findings"]] == [NON_FINITE_LOSS]

    def test_alarm_on_monotonic_rise(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join(epoch_line(index + 1, 1.0 + 0.01 * index) for index in range(20)))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert [f["kind"] for f in report["findings"]] == [MONOTONIC_RISE]

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
        log_path.write_text("\n".join([epoch_line(50, 0.9), snapshot_line()]))

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
        assert report["findings"][0]["kind"] == LOG_UNAVAILABLE

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

    def test_the_written_report_is_valid_json_after_a_transient_non_finite_loss(self, tmp_path: Path) -> None:
        # A transient NaN is expected and permitted, but `json.dump` writes it as a bare `NaN`
        # token, which is not JSON. Strict consumers refuse the whole report and `jq` silently
        # rewrites it (`Infinity` becomes 1.797e308); either way every later poll stays unreadable.
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([step_line(2, 10, 1.0), step_line(2, 11, float("nan")), step_line(2, 12, 0.9)]))
        report_path = tmp_path / "status.json"

        TrainingLogWatcher(log_path, total_epochs=300, report_path=report_path).run()

        report = json.loads(report_path.read_text(), parse_constant=lambda constant: pytest.fail(f"bare {constant} token"))
        assert report["verdict"] == HEALTHY
        assert report["loss_curve"][0]["step_mean_loss"] is None

    def test_eta_credits_the_part_of_the_current_epoch_already_done(self, tmp_path: Path) -> None:
        # `epochs_completed` counts whole epochs, so charging the epoch in flight in full overstates
        # the time left by every step already invested in it -- near the end of the final epoch,
        # by almost a whole one.
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "\n".join(
                [
                    epoch_line(1, 0.9),
                    epoch_line(2, 0.9),
                    step_line(3, 1, 1.0, ts="2026-09-11 17:00:00.000", total=2000),
                    step_line(3, 1001, 1.0, ts="2026-09-11 17:00:20.000", total=2000),
                ]
            )
        )

        report = TrainingLogWatcher(log_path, total_epochs=10).run()

        remaining_epochs = 10 - 2 - 1001 / 2000
        assert report["progress"]["epochs_remaining"] == pytest.approx(remaining_epochs)
        assert report["timing"]["eta_seconds"] == pytest.approx(remaining_epochs * 40.0)

    def test_progress_survives_a_log_rotated_mid_epoch(self, tmp_path: Path) -> None:
        # A log rotated mid-epoch opens on that epoch's step lines, and its average-loss line may
        # not be written yet. Reading completion from the average lines alone reports such a run as
        # barely started and hands the whole run back as remaining.
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([step_line(5, 10, 1.0), step_line(5, 11, 0.9)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["progress"]["epochs_completed"] == 4
        assert report["progress"]["epochs_remaining"] == pytest.approx(300 - 4 - 11 / 3714)

    def test_loss_curve_keeps_epochs_known_only_from_their_average_line(self, tmp_path: Path) -> None:
        # Rotation can keep the `epoch N average loss` line without N's step lines. The curve is
        # what the acceptance record exports, so a finished epoch must not vanish from it.
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([epoch_line(1, 0.9), epoch_line(2, 0.85), step_line(3, 1, 0.8)]))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert [(point["epoch"], point["step_mean_loss"], point["reported_average_loss"]) for point in report["loss_curve"]] == [
            (1, None, 0.9),
            (2, None, 0.85),
            (3, 0.8, None),
        ]

    def test_a_log_that_never_logged_training_state_alarms_once_it_stops_growing(self, tmp_path: Path) -> None:
        # The launcher `tee`s the log into existence before the trainer writes a line, so a log
        # holding only torchrun warnings and config echoes is normal for the first minutes. A
        # startup that failed leaves that same log behind forever, and the poller would report
        # HEALTHY for the rest of the day -- the parser ignores every line such a log contains.
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "W0911 17:01:58.825000 3599076 torch/distributed/run.py:982] Setting OMP_NUM_THREADS\n"
            "[2026-09-11 17:02:03.498][ INFO](training) - [config] num_epochs -> 300.\n"
        )
        stale = time.time() - DEFAULT_STARTUP_GRACE_SECONDS - 60
        os.utime(log_path, (stale, stale))

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == ALARM
        assert [finding["kind"] for finding in report["findings"]] == [NO_TRAINING_STATE]

    def test_a_log_that_has_just_started_is_not_alarmed_on(self, tmp_path: Path) -> None:
        log_path = tmp_path / "train.log"
        log_path.write_text("[2026-09-11 17:02:03.498][ INFO](training) - [config] num_epochs -> 300.\n")

        report = TrainingLogWatcher(log_path, total_epochs=300).run()

        assert report["verdict"] == HEALTHY
