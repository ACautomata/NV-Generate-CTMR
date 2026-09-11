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

"""Training-log watcher: health verdict, progress and loss curve for a running finetune
(spec #13 section 4.7, ticket T9a #25).

The execution discipline for every tier is "no resume, training must not be interrupted": a
killed tier restarts from epoch 1, so the only decision an agent may take is to *notice* and
*escalate*.  Two failure modes are named by the spec, and this watcher is how they are caught:

- a loss that stays non-finite across a sustained run -- counted both over the per-step records
  and over the all-reduced per-epoch averages, since each resolution is blind where the other
  sees (a one-off inf is GradScaler's normal business under AMP), and
- a loss that rises monotonically for ~20 consecutive epochs.

A third failure is the absence of any of those.  A trainer that died -- an OOM, a node reboot, a
killed session -- writes nothing, and nothing it already wrote contradicts any criterion, so a log
that stops growing while epochs remain is its own alarm, judged against a tolerance scaled to the
epoch length the log itself measured.

Both alarm criteria are loose on purpose -- a false alarm here costs a whole tier, since the
only available response is to kill the run and retrain it from epoch 1.

Both are advisory: the report says ALARM and ``main`` exits 1 so a polling shell can react,
but killing the run stays a human decision (spec section 4.7).  The same report carries the
numbers the acceptance record needs -- per-epoch loss curve, measured seconds per step, ETA,
and the snapshot files seen so far -- so one parser serves both jobs.  It is strict JSON: a
bare ``NaN`` token would leave every later poll unreadable to strict consumers.

Cost: a 300-epoch run logs one line per optimizer step (~1.1M lines, ~170 MB), and this
reads all of it on every poll.  That is seconds against a poll interval of minutes; the
alternative (tail-following state) would make every invocation depend on its predecessor.

Usage (on gauss; poll from a shell loop or cron, --report makes the verdict durable)::

    python -m scripts.watch_training_log \\
        --log $RUN/logs/train.log --total-epochs 300 \\
        --report $RUN/logs/status.json || echo "ALARM: inspect $RUN/logs/status.json"
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class Verdict(StrEnum):
    """The report's headline: whether anything about the run needs a human."""

    HEALTHY = "HEALTHY"
    ALARM = "ALARM"


class FindingKind(StrEnum):
    """Why a report says ALARM.

    A closed set, so a kind that is not in the vocabulary cannot reach the report by typo, and
    so an operator reading ``ALARM.txt`` learns from one word which failure to go and look at.
    """

    NON_FINITE_LOSS = "non_finite_loss"
    MONOTONIC_RISE = "monotonic_rise"
    LOG_STALLED = "log_stalled"
    LOG_UNAVAILABLE = "log_unavailable"
    NO_TRAINING_STATE = "no_training_state"


DEFAULT_RISE_WINDOW = 20
DEFAULT_NON_FINITE_STREAK = 20
DEFAULT_NON_FINITE_EPOCH_STREAK = 3
DEFAULT_STARTUP_GRACE_SECONDS = 1800
DEFAULT_STALE_AFTER_EPOCHS = 3.0
MIN_SILENCE_SECONDS = 300
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# `[2026-09-11 17:02:08.979][ INFO](training) - [2026-09-11 17:02:08] epoch 1, iter 1/3714, loss: 1.0373, lr: 0.000010000000.`
_PREFIX = r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]\[ INFO\]\(training\) - "
_STEP_PATTERN = re.compile(
    _PREFIX + r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] epoch (?P<epoch>\d+), iter (?P<iteration>\d+)/(?P<total>\d+), "
    r"loss: (?P<loss>[^,]+), lr: \S+\.$"
)
_EPOCH_PATTERN = re.compile(_PREFIX + r"epoch (?P<epoch>\d+) average loss: (?P<loss>\S+)\.$")
_SNAPSHOT_PATTERN = re.compile(_PREFIX + r"Snapshot saved to (?P<path>.+)\.$")


@dataclass(frozen=True)
class StepRecord:
    """One logged optimizer step."""

    epoch: int
    iter: int
    iterations_per_epoch: int
    loss: float
    timestamp: datetime

    @property
    def fractional_epoch_position(self) -> float:
        return self.iter / self.iterations_per_epoch


@dataclass(frozen=True)
class EpochRecord:
    """One logged ``epoch N average loss`` line -- the all-reduced per-epoch number."""

    epoch: int
    average_loss: float


@dataclass(frozen=True)
class LossCurvePoint:
    """One epoch of the loss curve.

    Either half can be absent, and which one is which says something about the log: a rotated log
    may keep the step lines without the epoch's average line, or the other way round.  The report
    spells this type out as JSON keys in one place, so the records stay a reading of the log
    rather than a knowledge of the wire format.
    """

    epoch: int
    step_mean_loss: float | None
    reported_average_loss: float | None


@dataclass(frozen=True)
class TrainingLog:
    """Everything the watcher reads out of one training log.

    The report's aggregations are derived here rather than in the watcher: they are facts about
    the log, not about watching, and this repo already puts them on the record object
    (``summarize_fid_results.FidSummary.summarize``).
    """

    steps: list[StepRecord]
    epochs: list[EpochRecord]
    snapshots: list[str]

    @property
    def epochs_completed(self) -> int:
        """The highest epoch the log shows as finished.

        Two record types witness completion and a rotated log may keep only one of them.  An
        ``epoch N average loss`` line means N is done; step records inside epoch N mean N-1 is
        done, since the trainer can only be in epoch N once N-1 has finished.  Reading the
        average lines alone would report a log rotated mid-epoch as barely started.
        """
        from_epochs = max((epoch.epoch for epoch in self.epochs), default=0)
        from_steps = self.steps[-1].epoch - 1 if self.steps else 0
        return max(from_epochs, from_steps)

    @property
    def iterations_per_epoch(self) -> int | None:
        return self.steps[-1].iterations_per_epoch if self.steps else None

    @property
    def epoch_fraction(self) -> float | None:
        """How far into the epoch the run is currently working on, or None before the first step.

        An epoch here runs ~20 minutes, so the completed count alone leaves a reader guessing
        whether the next snapshot is imminent or a whole epoch away.  Zero once that epoch's
        average-loss line has landed: the steps then belong to a finished epoch, and counting
        their fraction again would charge the ETA for an epoch already paid for.
        """
        if not self.steps:
            return None
        newest = self.steps[-1]
        if any(epoch.epoch == newest.epoch for epoch in self.epochs):
            return 0.0
        return newest.fractional_epoch_position

    @property
    def seconds_per_step(self) -> float | None:
        """Measured cost of one optimizer step, from the log's own timestamps.

        Steps are logged in order, so the first-to-last span over the optimizer steps between
        them excludes the one-off startup cost (checkpoint load, first CUDA kernels) that a
        first-step delta would fold in.  The step count comes from the epoch/iteration pair --
        the iteration counter restarts each epoch, so the row count alone is only right while
        the span stays inside one epoch.
        """
        if len(self.steps) < 2:
            return None
        first, last = self.steps[0], self.steps[-1]
        elapsed_steps = (last.epoch - first.epoch) * last.iterations_per_epoch + (last.iter - first.iter)
        if elapsed_steps <= 0:
            return None
        return (last.timestamp - first.timestamp).total_seconds() / elapsed_steps

    @property
    def loss_curve(self) -> list[LossCurvePoint]:
        """Per-epoch loss, from the step lines and from the script's own average-loss lines.

        The two agree only approximately: the reported average is the all-reduced number across
        ranks, while ``step_mean_loss`` is recomputed here from this rank's log.  Both are kept,
        and an epoch appears if either record type mentions it: a rotated log can retain one
        without the other, and an epoch missing from the curve is an epoch missing from the
        acceptance record this report feeds.
        """
        reported = {epoch.epoch: epoch.average_loss for epoch in self.epochs}
        per_epoch_steps: dict[int, list[float]] = {}
        for step in self.steps:
            per_epoch_steps.setdefault(step.epoch, []).append(step.loss)
        means = {epoch: sum(losses) / len(losses) for epoch, losses in per_epoch_steps.items()}
        return [
            LossCurvePoint(epoch=epoch, step_mean_loss=means.get(epoch), reported_average_loss=reported.get(epoch))
            for epoch in sorted(means.keys() | reported.keys())
        ]


@dataclass(frozen=True)
class HealthFinding:
    """One reason the run needs a human look."""

    kind: FindingKind
    epoch: int
    detail: str


class HealthCriterion(Protocol):
    """A single, independently testable reason to alarm."""

    def inspect(self, log: TrainingLog) -> list[HealthFinding]: ...


class TrainingLogParser:
    """Adapter from the training script's log text to structured records.

    Unrecognized lines are ignored rather than reported: the log interleaves torchrun/DDP
    warnings, config echoes and snapshot copies with the step lines, and only the three
    patterns above carry run state.
    """

    def parse(self, text: str) -> TrainingLog:
        steps: list[StepRecord] = []
        epochs: list[EpochRecord] = []
        snapshots: list[str] = []
        for line in text.splitlines():
            step = _STEP_PATTERN.match(line)
            if step is not None:
                steps.append(
                    StepRecord(
                        epoch=int(step["epoch"]),
                        iter=int(step["iteration"]),
                        iterations_per_epoch=int(step["total"]),
                        loss=float(step["loss"]),
                        timestamp=datetime.strptime(step["timestamp"], TIMESTAMP_FORMAT),
                    )
                )
                continue
            epoch = _EPOCH_PATTERN.match(line)
            if epoch is not None:
                epochs.append(EpochRecord(epoch=int(epoch["epoch"]), average_loss=float(epoch["loss"])))
                continue
            snapshot = _SNAPSHOT_PATTERN.match(line)
            if snapshot is not None:
                snapshots.append(snapshot["path"])
        return TrainingLog(steps=steps, epochs=epochs, snapshots=snapshots)


class NonFiniteLossCriterion:
    """Alarms when the loss stops being finite and stays that way.

    One NaN/inf loss is not a failure.  Under AMP ``GradScaler`` skips that step and backs the
    scale off, and training continues; alarming on a single one would be this tool's highest
    false-positive surface, and a false alarm here costs the whole tier -- there is no resume,
    so spec section 4.7 loosens the criteria precisely to keep the operator from killing a run
    that is fine.  A run that has actually diverged stops producing finite losses altogether, so
    a streak is what this criterion counts, at both resolutions the log offers.  Only a streak
    still standing at the newest record is reported: a burst the scaler recovered from stays in
    the log forever, and re-reporting it on every poll would leave the verdict permanently red,
    and a permanently red poller's next real alarm goes unread -- the argument
    ``MonotonicRiseCriterion`` makes for a climb the loss has given back.

    Two resolutions, because each is blind where the other sees.  The step records come from
    rank 0 alone -- ``diff_model_train.py`` logs the per-iter line under ``local_rank == 0`` --
    so a nonzero rank that diverges never shows up there; the epoch average is all-reduced, so
    it does.  That average is correspondingly coarser: one bad step on any rank makes its whole
    epoch non-finite, which is why a lone non-finite epoch is still not an alarm.  Every epoch
    here covers the whole dataset exactly once, so a flaky sample lands in a single epoch and
    only a persistent fault can reach consecutive ones.
    """

    def __init__(
        self,
        consecutive_steps: int = DEFAULT_NON_FINITE_STREAK,
        consecutive_epochs: int = DEFAULT_NON_FINITE_EPOCH_STREAK,
    ) -> None:
        if consecutive_steps < 1:
            raise ValueError(f"consecutive_steps must be at least one step, got {consecutive_steps}")
        if consecutive_epochs < 1:
            raise ValueError(f"consecutive_epochs must be at least one epoch, got {consecutive_epochs}")
        self._consecutive_steps = consecutive_steps
        self._consecutive_epochs = consecutive_epochs

    def inspect(self, log: TrainingLog) -> list[HealthFinding]:
        findings = []
        steps = self._sustained_step_run(log)
        if steps is not None:
            first, last = steps[0], steps[-1]
            findings.append(
                HealthFinding(
                    FindingKind.NON_FINITE_LOSS,
                    last.epoch,
                    f"the latest {len(steps)} steps are non-finite, from epoch {first.epoch} step "
                    f"{first.iter} through epoch {last.epoch} step {last.iter} (latest {last.loss})",
                )
            )
        epochs = self._sustained_epoch_run(log)
        if epochs is not None:
            first, last = epochs[0], epochs[-1]
            findings.append(
                HealthFinding(
                    FindingKind.NON_FINITE_LOSS,
                    last.epoch,
                    f"the latest {len(epochs)} epoch averages are non-finite after the all-reduce, epochs "
                    f"{first.epoch}-{last.epoch} (latest {last.average_loss}); step records come from "
                    f"rank 0 alone, so a nonzero rank is where to look",
                )
            )
        return findings

    def _sustained_step_run(self, log: TrainingLog) -> list[StepRecord] | None:
        """The run of non-finite step losses still standing at the newest record, if long enough."""
        streak: list[StepRecord] = []
        for step in reversed(log.steps):
            if math.isfinite(step.loss):
                break
            streak.append(step)
        streak.reverse()
        return streak if len(streak) >= self._consecutive_steps else None

    def _sustained_epoch_run(self, log: TrainingLog) -> list[EpochRecord] | None:
        """The run of non-finite epoch averages still standing at the newest record, if long enough."""
        streak: list[EpochRecord] = []
        for epoch in reversed(log.epochs):
            if math.isfinite(epoch.average_loss):
                break
            streak.append(epoch)
        streak.reverse()
        return streak if len(streak) >= self._consecutive_epochs else None


class MonotonicRiseCriterion:
    """Alarms when ``window`` consecutive epochs each raise the average loss.

    Strict monotonicity over ~20 epochs is the spec's tripwire for divergence: per-epoch
    averages over thousands of steps are smooth enough that 20 rises in a row is a trend, not
    noise, while a run that merely plateaus or oscillates stays silent.

    A rise is reported only while the run has not given it back.  Reporting every historical
    window would leave the verdict stuck at ALARM for the rest of the run after a climb the loss
    has since recovered from, and a poller that is permanently red is one whose next real alarm
    goes unread.  A climb the loss has stayed above is still reported -- that is the
    diverged-and-stuck shape.  Only the first window that both rises and still stands is
    reported: 30 rising epochs contain 11 windows of 20, and the operator needs one alarm naming
    where the trend was established, not eleven restatements of it.
    """

    def __init__(self, window: int = DEFAULT_RISE_WINDOW) -> None:
        if window < 2:
            raise ValueError(f"window must span at least two epochs to express a rise, got {window}")
        self._window = window

    def inspect(self, log: TrainingLog) -> list[HealthFinding]:
        epochs = [epoch for epoch in log.epochs if math.isfinite(epoch.average_loss)]
        if not epochs:
            return []
        latest = epochs[-1].average_loss
        for start in range(len(epochs) - self._window + 1):
            candidate = epochs[start : start + self._window]
            if self._is_consecutive_rise(candidate) and latest >= candidate[-1].average_loss:
                first, last = candidate[0], candidate[-1]
                return [
                    HealthFinding(
                        FindingKind.MONOTONIC_RISE,
                        last.epoch,
                        f"epochs {first.epoch}-{last.epoch} rose monotonically "
                        f"({first.average_loss:.4f} -> {last.average_loss:.4f}) over {self._window} consecutive epochs, "
                        f"and the loss is still at or above that level (latest {latest:.4f})",
                    )
                ]
        return []

    @staticmethod
    def _is_consecutive_rise(candidate: list[EpochRecord]) -> bool:
        return all(earlier.epoch + 1 == later.epoch and earlier.average_loss < later.average_loss for earlier, later in zip(candidate, candidate[1:]))


class StalledLogCriterion:
    """Alarms when the log itself goes quiet while the run still has epochs to go.

    Two shapes, one symptom, and neither is visible in the text: a stopped trainer writes
    exactly as much as a healthy one.  The launcher creates the log through ``tee`` before the
    trainer writes a line, so no records at all is normal for the first minutes -- the grace
    period is that window.  A log that logged thousands of steps and then went silent is a
    trainer that died: an OOM, a node reboot, a killed session.  Every other criterion reads
    records, and records that stop arriving cannot contradict any of them, so without this the
    report reads HEALTHY for a run that is no longer doing anything.

    Unlike the loss criteria this one reads the log file, not just its parsed records -- the
    file's mtime is the only witness that growth has stopped.  A missing file is the same
    failure taken to its limit and reported as ``LOG_UNAVAILABLE``.
    """

    def __init__(
        self,
        log_path: Path,
        total_epochs: int,
        startup_grace_seconds: float = DEFAULT_STARTUP_GRACE_SECONDS,
    ) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")
        self._log_path = log_path
        self._total_epochs = total_epochs
        self._startup_grace_seconds = startup_grace_seconds

    def inspect(self, log: TrainingLog) -> list[HealthFinding]:
        if not self._log_path.exists():
            return [HealthFinding(FindingKind.LOG_UNAVAILABLE, 0, f"{self._log_path} does not exist")]
        return self._stalled_log_findings(log)

    def _stalled_log_findings(self, log: TrainingLog) -> list[HealthFinding]:
        if log.epochs_completed >= self._total_epochs:
            return []
        silent_seconds = datetime.now(UTC).timestamp() - self._log_path.stat().st_mtime
        if silent_seconds < self._silence_tolerance(log):
            return []
        minutes = silent_seconds / 60
        if log.steps or log.epochs:
            return [
                HealthFinding(
                    FindingKind.LOG_STALLED,
                    log.epochs_completed,
                    f"no new line for {minutes:.0f} minutes, with epoch {log.epochs_completed}/{self._total_epochs} the last one finished: "
                    f"{self._log_path} has stopped growing, so the trainer is gone or hung",
                )
            ]
        return [
            HealthFinding(
                FindingKind.NO_TRAINING_STATE,
                0,
                f"no step or epoch record after {minutes:.0f} minutes: {self._log_path} has not grown past its startup output",
            )
        ]

    def _silence_tolerance(self, log: TrainingLog) -> float:
        """How long the log may go unwritten before it counts as stopped.

        An epoch is the log's own heartbeat -- it prints at least its average-loss line once per
        epoch -- so the tolerance is a few of those, derived from the log's measured pace instead
        of being passed in.  Before the first step there is no pace to scale against, so the
        startup grace period stands in for it.

        Never below ``MIN_SILENCE_SECONDS``.  The measured pace is what the log *has* done, not a
        licence to alarm on a few seconds of quiet -- a checkpoint write can take that -- and a log
        whose timestamps do not resolve a pace would otherwise earn a tolerance of zero.
        """
        if log.seconds_per_step is not None and log.iterations_per_epoch:
            return max(MIN_SILENCE_SECONDS, DEFAULT_STALE_AFTER_EPOCHS * log.seconds_per_step * log.iterations_per_epoch)
        return self._startup_grace_seconds


class TrainingLogWatcher:
    """Parses one training log, applies every criterion, and writes the verdict report."""

    def __init__(
        self,
        log_path: Path,
        total_epochs: int,
        report_path: Path | None = None,
        non_finite_streak: int = DEFAULT_NON_FINITE_STREAK,
        rise_window: int = DEFAULT_RISE_WINDOW,
        startup_grace_seconds: float = DEFAULT_STARTUP_GRACE_SECONDS,
    ) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")
        self._log_path = log_path
        self._total_epochs = total_epochs
        self._report_path = report_path
        self._criteria: tuple[HealthCriterion, ...] = (
            NonFiniteLossCriterion(non_finite_streak),
            MonotonicRiseCriterion(rise_window),
            StalledLogCriterion(log_path, total_epochs, startup_grace_seconds),
        )
        self._parser = TrainingLogParser()

    def run(self) -> dict:
        """Read the log, decide, write the report; return it.

        A missing log parses to an empty record set rather than an exception: a poller that dies
        on a rotated log stops watching, and a stopped watcher looks exactly like a healthy run --
        the criteria then see the missing file for what it is.
        """
        if self._log_path.exists():
            log = self._parser.parse(self._log_path.read_text())
        else:
            log = TrainingLog([], [], [])
        findings = [finding for criterion in self._criteria for finding in criterion.inspect(log)]
        report = self._build_report(log, findings)
        if self._report_path is not None:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            with self._report_path.open("w") as file:
                # The curve carries None where a loss is non-finite, so this never fires today;
                # `allow_nan=False` is the guard that keeps a later source of non-finite report
                # values from silently writing tokens no strict JSON reader accepts.
                json.dump(report, file, indent=2, allow_nan=False)
                file.write("\n")
        return report

    def _build_report(self, log: TrainingLog, findings: list[HealthFinding]) -> dict:
        epochs_remaining = self._epochs_remaining(log)
        return {
            "verdict": Verdict.ALARM if findings else Verdict.HEALTHY,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "log_path": str(self._log_path),
            "progress": {
                "epochs_completed": log.epochs_completed,
                "total_epochs": self._total_epochs,
                "epochs_remaining": epochs_remaining,
                "steps_observed": len(log.steps),
                "iterations_per_epoch": log.iterations_per_epoch,
                "epoch_fraction": log.epoch_fraction,
            },
            "timing": self._timing(log, epochs_remaining),
            "loss_curve": [self._curve_point(point) for point in log.loss_curve],
            "snapshots": log.snapshots,
            "findings": [{"kind": finding.kind, "epoch": finding.epoch, "detail": finding.detail} for finding in findings],
        }

    def _epochs_remaining(self, log: TrainingLog) -> float:
        """Epochs of work left, counting the part of the epoch in flight already done.

        Charging that epoch in full would overstate the time left by every step already invested
        in it -- close to a whole epoch at the end of one, ~20 minutes here.
        """
        in_flight = log.epoch_fraction if log.epoch_fraction is not None else 0.0
        return max(0.0, self._total_epochs - log.epochs_completed - in_flight)

    def _curve_point(self, point: LossCurvePoint) -> dict:
        """One loss-curve point as the report holds it."""
        return {
            "epoch": point.epoch,
            "step_mean_loss": self._json_loss(point.step_mean_loss),
            "reported_average_loss": self._json_loss(point.reported_average_loss),
        }

    @staticmethod
    def _json_loss(loss: float | None) -> float | None:
        """A loss as JSON holds it.

        JSON has no NaN or Infinity literal: `json.dump` writes them as bare tokens that strict
        consumers refuse and that `jq` silently rewrites (``Infinity`` becomes 1.797e308).  A
        transient non-finite loss is expected and is not an alarm, so the curve carries it as
        null -- this report's existing spelling of "no value here".
        """
        return None if loss is None or not math.isfinite(loss) else loss

    def _timing(self, log: TrainingLog, epochs_remaining: float) -> dict:
        seconds_per_epoch = log.seconds_per_step * log.iterations_per_epoch if log.seconds_per_step is not None and log.iterations_per_epoch else None
        eta_seconds = seconds_per_epoch * epochs_remaining if seconds_per_epoch is not None else None
        return {
            "seconds_per_step": self._round_microseconds(log.seconds_per_step),
            "seconds_per_epoch": self._round_microseconds(seconds_per_epoch),
            "eta_seconds": self._round_microseconds(eta_seconds),
        }

    @staticmethod
    def _round_microseconds(value: float | None) -> float | None:
        """Durations resolve to the log's millisecond timestamps; more digits would be noise."""
        return None if value is None else round(value, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="the training log written by scripts.diff_model_train")
    parser.add_argument("--total-epochs", type=int, required=True, help="epochs the run was configured for (the training config's n_epochs)")
    parser.add_argument("--report", type=Path, help="write the JSON report here (default: print only)")
    parser.add_argument(
        "--rise-window",
        type=int,
        default=DEFAULT_RISE_WINDOW,
        help=f"consecutive rising epochs that raise the monotonic-rise alarm (default {DEFAULT_RISE_WINDOW})",
    )
    parser.add_argument(
        "--non-finite-streak",
        type=int,
        default=DEFAULT_NON_FINITE_STREAK,
        help=f"consecutive non-finite steps before the NaN/inf alarm (default {DEFAULT_NON_FINITE_STREAK}; one-off losses are normal under AMP)",
    )
    parser.add_argument(
        "--startup-grace-seconds",
        type=float,
        default=DEFAULT_STARTUP_GRACE_SECONDS,
        help=(
            "how long a log may go unwritten before the run counts as stopped, for as long as no step pace is known "
            f"(default {DEFAULT_STARTUP_GRACE_SECONDS}); once steps are logged the tolerance is "
            f"{DEFAULT_STALE_AFTER_EPOCHS} measured epochs instead, so this needs no tuning per workload"
        ),
    )
    args = parser.parse_args()

    report = TrainingLogWatcher(
        log_path=args.log,
        total_epochs=args.total_epochs,
        report_path=args.report,
        non_finite_streak=args.non_finite_streak,
        rise_window=args.rise_window,
        startup_grace_seconds=args.startup_grace_seconds,
    ).run()

    progress, timing = report["progress"], report["timing"]
    print(
        f"verdict {report['verdict']}: {progress['epochs_completed']}/{progress['total_epochs']} epochs, "
        f"{progress['steps_observed']} steps observed, "
        f"{timing['seconds_per_step']} s/step, ETA {timing['eta_seconds']} s"
    )
    for finding in report["findings"]:
        print(f"  [{finding['kind']}] epoch {finding['epoch']}: {finding['detail']}")
    if report["verdict"] == Verdict.ALARM:
        print("ALARM is advisory: spec section 4.7 leaves killing the run to the operator.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
