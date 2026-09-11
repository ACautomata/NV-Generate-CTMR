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

- a loss that stays non-finite across a sustained run of steps (a one-off inf is GradScaler's
  normal business under AMP), and
- a loss that rises monotonically for ~20 consecutive epochs.

Both alarm criteria are loose on purpose -- a false alarm here costs a whole tier, since the
only available response is to kill the run and retrain it from epoch 1.

Both are advisory: the report says ALARM and ``main`` exits 1 so a polling shell can react,
but killing the run stays a human decision (spec section 4.7).  The same report carries the
numbers the acceptance record needs -- per-epoch loss curve, measured seconds per step, ETA,
and the snapshot files seen so far -- so one parser serves both jobs.

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
from pathlib import Path
from typing import Protocol

HEALTHY = "HEALTHY"
ALARM = "ALARM"
NON_FINITE_LOSS = "non_finite_loss"
MONOTONIC_RISE = "monotonic_rise"
LOG_UNAVAILABLE = "log_unavailable"

DEFAULT_RISE_WINDOW = 20
DEFAULT_NON_FINITE_STREAK = 20
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# `[2026-09-11 17:02:08.979][ INFO](training) - [2026-09-11 17:02:08] epoch 1, iter 1/3714, loss: 1.0373, lr: 0.000010000000.`
_PREFIX = r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]\[ INFO\]\(training\) - "
_STEP_PATTERN = re.compile(
    _PREFIX + r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] epoch (?P<epoch>\d+), iter (?P<iteration>\d+)/(?P<total>\d+), "
    r"loss: (?P<loss>[^,]+), lr: (?P<lr>\S+)\.$"
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
    lr: float
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
        """The highest epoch the log reports -- not the number of average-loss lines seen.

        A log rotated mid-run keeps its epoch numbering, and counting lines would read such a
        run as barely started.
        """
        return max((epoch.epoch for epoch in self.epochs), default=0)

    @property
    def iterations_per_epoch(self) -> int | None:
        return self.steps[-1].iterations_per_epoch if self.steps else None

    @property
    def epoch_fraction(self) -> float | None:
        """How far into its current epoch the newest logged step is.

        An epoch here runs ~20 minutes, so the completed count alone leaves a reader guessing
        whether the next snapshot is imminent or a whole epoch away.
        """
        return self.steps[-1].fractional_epoch_position if self.steps else None

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
    def loss_curve(self) -> list[dict]:
        """Per-epoch loss, from the step lines and from the script's own average-loss lines.

        The two agree only approximately: the reported average is the all-reduced number across
        ranks, while ``step_mean_loss`` is recomputed here from this rank's log.  Both are kept
        because the curve must survive a log whose epoch lines have not been written yet.
        """
        reported = {epoch.epoch: epoch.average_loss for epoch in self.epochs}
        per_epoch_steps: dict[int, list[float]] = {}
        for step in self.steps:
            per_epoch_steps.setdefault(step.epoch, []).append(step.loss)
        return [
            {"epoch": epoch, "step_mean_loss": sum(losses) / len(losses), "reported_average_loss": reported.get(epoch)}
            for epoch, losses in sorted(per_epoch_steps.items())
        ]


@dataclass(frozen=True)
class HealthFinding:
    """One reason the run needs a human look."""

    kind: str
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
                        lr=float(step["lr"]),
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
    """Alarms when a loss stays non-finite across a sustained run of steps.

    One NaN/inf loss is not a failure.  Under AMP ``GradScaler`` skips that step and backs the
    scale off, and training continues; alarming on a single one would be this tool's highest
    false-positive surface, and a false alarm here costs the whole tier -- there is no resume,
    so spec section 4.7 loosens the criteria precisely to keep the operator from killing a run
    that is fine.  A run that has actually diverged stops producing finite losses altogether, so
    the streak is what the criterion counts.

    Epoch averages are deliberately not a separate trigger: a single non-finite step makes its
    epoch average non-finite too, so alarming there would smuggle the transient case back in.
    A non-finite average is still visible in the loss curve.
    """

    def __init__(self, consecutive_steps: int = DEFAULT_NON_FINITE_STREAK) -> None:
        if consecutive_steps < 1:
            raise ValueError(f"consecutive_steps must be at least one step, got {consecutive_steps}")
        self._consecutive_steps = consecutive_steps

    def inspect(self, log: TrainingLog) -> list[HealthFinding]:
        streak_start: StepRecord | None = None
        streak = 0
        for step in log.steps:
            if math.isfinite(step.loss):
                streak_start, streak = None, 0
                continue
            streak_start = streak_start if streak_start is not None else step
            streak += 1
            if streak >= self._consecutive_steps:
                return [
                    HealthFinding(
                        NON_FINITE_LOSS,
                        step.epoch,
                        f"{streak} consecutive steps are non-finite, from epoch {streak_start.epoch} step "
                        f"{streak_start.iter} through epoch {step.epoch} step {step.iter} (latest {step.loss})",
                    )
                ]
        return []


class MonotonicRiseCriterion:
    """Alarms when ``window`` consecutive epochs each raise the average loss.

    Strict monotonicity over ~20 epochs is the spec's tripwire for divergence: per-epoch
    averages over thousands of steps are smooth enough that 20 rises in a row is a trend, not
    noise, while a run that merely plateaus or oscillates stays silent.  Only the first such
    window is reported -- 30 rising epochs contain 11 windows of 20, and the operator needs one
    alarm naming where the trend was established, not eleven restatements of it.
    """

    def __init__(self, window: int = DEFAULT_RISE_WINDOW) -> None:
        if window < 2:
            raise ValueError(f"window must span at least two epochs to express a rise, got {window}")
        self._window = window

    @property
    def window(self) -> int:
        return self._window

    def inspect(self, log: TrainingLog) -> list[HealthFinding]:
        epochs = [epoch for epoch in log.epochs if math.isfinite(epoch.average_loss)]
        for start in range(len(epochs) - self._window + 1):
            candidate = epochs[start : start + self._window]
            if self._is_consecutive_rise(candidate):
                first, last = candidate[0], candidate[-1]
                return [
                    HealthFinding(
                        MONOTONIC_RISE,
                        last.epoch,
                        f"epochs {first.epoch}-{last.epoch} rose monotonically "
                        f"({first.average_loss:.4f} -> {last.average_loss:.4f}) over {self._window} consecutive epochs",
                    )
                ]
        return []

    @staticmethod
    def _is_consecutive_rise(candidate: list[EpochRecord]) -> bool:
        return all(earlier.epoch + 1 == later.epoch and earlier.average_loss < later.average_loss for earlier, later in zip(candidate, candidate[1:]))


class TrainingLogWatcher:
    """Parses one training log, applies every criterion, and writes the verdict report."""

    def __init__(
        self,
        log_path: Path,
        total_epochs: int,
        report_path: Path | None = None,
        criteria: tuple[HealthCriterion, ...] | None = None,
    ) -> None:
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be positive, got {total_epochs}")
        self._log_path = log_path
        self._total_epochs = total_epochs
        self._report_path = report_path
        self._criteria = criteria if criteria is not None else (NonFiniteLossCriterion(), MonotonicRiseCriterion())
        self._parser = TrainingLogParser()

    def run(self) -> dict:
        """Read the log, decide, write the report; return it.

        A missing log is an ALARM rather than an exception: a poller that dies on a rotated
        log stops watching, and a stopped watcher looks exactly like a healthy run.
        """
        if self._log_path.exists():
            log = self._parser.parse(self._log_path.read_text())
            findings = [finding for criterion in self._criteria for finding in criterion.inspect(log)]
        else:
            log = TrainingLog([], [], [])
            findings = [HealthFinding(LOG_UNAVAILABLE, 0, f"{self._log_path} does not exist")]
        report = self._build_report(log, findings)
        if self._report_path is not None:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            with self._report_path.open("w") as file:
                json.dump(report, file, indent=2)
                file.write("\n")
        return report

    def _build_report(self, log: TrainingLog, findings: list[HealthFinding]) -> dict:
        epochs_remaining = max(0, self._total_epochs - log.epochs_completed)
        return {
            "verdict": ALARM if findings else HEALTHY,
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
            "loss_curve": log.loss_curve,
            "snapshots": log.snapshots,
            "findings": [{"kind": finding.kind, "epoch": finding.epoch, "detail": finding.detail} for finding in findings],
        }

    def _timing(self, log: TrainingLog, epochs_remaining: int) -> dict:
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
    args = parser.parse_args()

    report = TrainingLogWatcher(
        log_path=args.log,
        total_epochs=args.total_epochs,
        report_path=args.report,
        criteria=(NonFiniteLossCriterion(args.non_finite_streak), MonotonicRiseCriterion(args.rise_window)),
    ).run()

    progress, timing = report["progress"], report["timing"]
    print(
        f"verdict {report['verdict']}: {progress['epochs_completed']}/{progress['total_epochs']} epochs, "
        f"{progress['steps_observed']} steps observed, "
        f"{timing['seconds_per_step']} s/step, ETA {timing['eta_seconds']} s"
    )
    for finding in report["findings"]:
        print(f"  [{finding['kind']}] epoch {finding['epoch']}: {finding['detail']}")
    if report["verdict"] == ALARM:
        print("ALARM is advisory: spec section 4.7 leaves killing the run to the operator.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
