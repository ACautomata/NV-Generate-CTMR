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

"""BRATS finetune run audit: v1 weight protection and checkpoint inventory (spec #13
sections 4.4 / 5.0, ticket T9c #27).

Most of a tier's acceptance record is read off the training log (``watch_training_log``),
but three of its criteria are not in the log at all -- they are facts about files:

- **the v1 weights were not written to** (section 4.4).  The run warm-starts from
  ``existing_ckpt_filepath`` and is supposed to only read it; nothing in the training loop
  enforces that, and the damage would be silent, since the finetuned checkpoints would keep
  looking fine while the baseline every FID comparison is measured against had moved.  Only
  a reading taken *before* the run can prove it, which is what ``freeze`` is for.
- **every expected snapshot landed**, and the main checkpoint is the final epoch's weights
  (section 4.3).  The trainer copies the main checkpoint to ``ckpt_epoch{N}.pt`` after saving
  it, so the two hold the same weights at every snapshot epoch -- checking it catches a
  truncated or half-written final save that the log would still call complete.
- **every checkpoint carries the scale_factor recomputed at launch** (section 4.5), with the
  deviation from v1's declared value on the record.  Inference reads that number back out of
  whichever checkpoint gets selected, so a checkpoint missing it is not selectable.

The two halves are one tool because they are one record.  ``freeze`` runs at launch, writes
the baseline next to the run's logs, and needs nothing from the run it is about to observe.
``audit`` runs when the tier finishes, compares the disk against that baseline, and writes the
report plus exit code 1 when anything is wrong -- the same advisory-to-a-shell shape as
``watch_training_log``, with the difference that here a FAIL means the run's outputs are not
what the ticket claims, not that a human must decide something.

Scale_factor *drift* is recorded, never judged: the spec puts the pre-training gate on
``check_scale_factor``, and this report is the after-the-fact record of what each checkpoint
ended up holding.  A checkpoint holding no finite value at all is a different matter, and fails.

Usage (on the cluster, freeze before the launch and audit after the last epoch)::

    python -m scripts.audit_finetune_run freeze \\
        -c $RUN/configs/config_maisi_diff_model_rflow-mr-brain-brats.json \\
        -e $RUN/configs/environment_N1000_sugon.json \\
        --report $RUN/logs/launch_baseline.json

    python -m scripts.audit_finetune_run audit \\
        --baseline $RUN/logs/launch_baseline.json \\
        --report $RUN/logs/run_audit.json
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import torch

HASH_CHUNK_BYTES = 1 << 22
SNAPSHOT_PREFIX = "ckpt_epoch"


class Verdict(StrEnum):
    """Whether the run's on-disk outputs match what the baseline says they should be."""

    PASS = "PASS"
    FAIL = "FAIL"


class FindingKind(StrEnum):
    """Why a report says FAIL.  A closed set, so a kind cannot reach the report by typo and an
    operator learns from one word which file to go and look at."""

    WEIGHTS_MODIFIED = "weights_modified"
    SNAPSHOT_MISSING = "snapshot_missing"
    MAIN_CHECKPOINT_MISSING = "main_checkpoint_missing"
    MAIN_CHECKPOINT_INCOMPLETE = "main_checkpoint_incomplete"
    MAIN_CHECKPOINT_STALE = "main_checkpoint_stale"
    EPOCH_MISMATCH = "epoch_mismatch"
    SCALE_FACTOR_MISSING = "scale_factor_missing"
    CHECKPOINT_UNREADABLE = "checkpoint_unreadable"


@dataclass(frozen=True)
class RunFinding:
    """One reason a report says FAIL."""

    kind: FindingKind
    detail: str


@dataclass(frozen=True)
class WeightFingerprint:
    """Identity of a weights file: what it takes to say a later reading saw the same file.

    ``mtime_ns`` is what the ticket's acceptance criterion asks for (section 4.4, "v1 权重文件
    mtime 未变"); ``sha256`` is what makes the claim mean something, since any copy that
    preserves times -- rsync ``-t``, ``cp -p``, an untar over the top -- moves the content while
    leaving the mtime alone.
    """

    path: str
    size: int
    mtime_ns: int
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> "WeightFingerprint":
        """Read a weights file's identity.  The digest streams: v1 is gigabytes, not a buffer."""
        stat = path.stat()
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return cls(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "WeightFingerprint":
        return cls(**payload)

    def differences(self, other: "WeightFingerprint") -> tuple[str, ...]:
        """Names of the compared fields that differ -- empty for an unchanged file.

        ``path`` is deliberately not compared: the cluster mounts the same export twice
        (``/root/private_data`` and ``/public/home/wang9691``), so the same untouched file is
        legitimately spellable two ways, and a report that cried "modified" over that would
        train its reader to ignore it.
        """
        return tuple(field for field in ("size", "mtime_ns", "sha256") if getattr(self, field) != getattr(other, field))

    def holds_the_same_weights_as(self, other: "WeightFingerprint") -> bool:
        """Content equality, ignoring when the two files were last written.

        A snapshot copy is a fresh file with its own mtime, so a comparison that included times
        would call every faithful copy stale.
        """
        return (self.size, self.sha256) == (other.size, other.sha256)

    def describe(self) -> str:
        return f"{self.path}: {self.size} bytes, mtime {self.mtime_ns}, sha256 {self.sha256[:16]}…"


@dataclass(frozen=True)
class CheckpointRecord:
    """One checkpoint file as the acceptance record needs it: which epoch it holds and the
    scale_factor it was trained with.  ``size`` rides along so a truncated write is visible
    without a second ``stat``.

    A checkpoint that names no epoch reads as ``None`` rather than raising, and one that cannot
    be read at all carries the reader's complaint in ``unreadable``: the caller's job is to
    compare what it got against the tier's expectation, and a raise would take the whole report
    down over one bad file instead of naming it.  A truncated write is a realistic failure here
    -- the checkpoint is 722 MB and the run is four days long.
    """

    path: str
    size: int
    epoch: int | None
    scale_factor: float | None
    unreadable: str | None = None

    @classmethod
    def read(cls, path: Path) -> "CheckpointRecord":
        """Read the two scalars the acceptance record needs out of a checkpoint.

        ``mmap=True`` keeps the state dict's storages mapped instead of copying them: this reads
        a 722 MB checkpoint per snapshot to learn two numbers, and seven of those per tier is
        real I/O on a shared filesystem.  ``weights_only=False`` follows ``check_scale_factor``'s
        reading of the same files.
        """
        size = path.stat().st_size
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        except Exception as error:  # torch.load's failures are many and none of them are ours to fix
            return cls(path=str(path), size=size, epoch=None, scale_factor=None, unreadable=f"{type(error).__name__}: {error}")
        return cls(
            path=str(path),
            size=size,
            epoch=cls._epoch_of(checkpoint),
            scale_factor=cls._finite_checkpoint_number(checkpoint.get("scale_factor")),
        )

    @staticmethod
    def _epoch_of(checkpoint: dict) -> int | None:
        epoch = checkpoint.get("epoch")
        return None if epoch is None else int(epoch)

    @staticmethod
    def _finite_checkpoint_number(value: object) -> float | None:
        """A number as the report holds it, or None.

        JSON has no NaN or Infinity literal: ``json.dump`` writes them as bare tokens the spec's
        strict consumers refuse.  A checkpoint the trainer wrote with a non-finite scale_factor
        is exactly the corruption this audit exists to surface, so it is reported as "no value
        here" and named as a finding rather than smuggled through as a token.
        """
        if value is None:
            return None
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else None


@dataclass(frozen=True)
class LaunchBaseline:
    """What the launch froze, read off the same two configs ``diff_model_train`` is launched with.

    Written once, before the run can have touched anything, and never rewritten -- the audit's
    whole claim is that it compares against a reading older than the run.
    """

    frozen_at_utc: str
    train_config: str
    env_config: str
    n_epochs: int
    save_interval: int
    expected_snapshot_epochs: tuple[int, ...]
    model_dir: str
    model_filename: str
    v1_weights: WeightFingerprint
    v1_declared_scale_factor: float | None

    @classmethod
    def freeze(cls, train_config_path: Path, env_config_path: Path) -> "LaunchBaseline":
        """Fingerprint what the launch is about to train from, and what it is expected to produce.

        The expected snapshot epochs are derived from the training config rather than passed in:
        ``n_epochs`` and ``save_interval`` are what the trainer itself reads, so a freeze that
        agreed with them by construction cannot disagree with the run it audits.

        Raises:
            ValueError: If ``save_interval`` does not divide into ``n_epochs``, which would leave
                the run without a snapshot at the final epoch -- no candidate checkpoint for the
                FID selection the whole matrix is built around.
            ValueError: If the env config names a v1 checkpoint that is not on disk.
        """
        with open(train_config_path) as file:
            training = json.load(file)["diffusion_unet_train"]
        with open(env_config_path) as file:
            env = json.load(file)

        n_epochs = int(training["n_epochs"])
        save_interval = int(training.get("save_interval", 0))
        if save_interval and n_epochs % save_interval:
            raise ValueError(
                f"save_interval {save_interval} does not divide into n_epochs {n_epochs}: the run would end without a "
                "snapshot at the final epoch, so no checkpoint could stand for the tier"
            )

        v1_ckpt = Path(env["existing_ckpt_filepath"])
        if not v1_ckpt.is_file():
            raise ValueError(f"v1 checkpoint named by the env config is not on disk: {v1_ckpt}")

        return cls(
            frozen_at_utc=datetime.now(UTC).isoformat(),
            train_config=str(train_config_path),
            env_config=str(env_config_path),
            n_epochs=n_epochs,
            save_interval=save_interval,
            expected_snapshot_epochs=tuple(range(save_interval, n_epochs + 1, save_interval)) if save_interval else (),
            model_dir=str(env["model_dir"]),
            model_filename=str(env["model_filename"]),
            v1_weights=WeightFingerprint.capture(v1_ckpt),
            v1_declared_scale_factor=CheckpointRecord.read(v1_ckpt).scale_factor,
        )

    @property
    def main_checkpoint_path(self) -> Path:
        return Path(self.model_dir) / self.model_filename

    def snapshot_path(self, epoch: int) -> Path:
        return Path(self.model_dir) / f"{SNAPSHOT_PREFIX}{epoch}.pt"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["expected_snapshot_epochs"] = list(self.expected_snapshot_epochs)
        return payload

    @classmethod
    def load(cls, path: Path) -> "LaunchBaseline":
        with open(path) as file:
            payload = json.load(file)
        payload["expected_snapshot_epochs"] = tuple(payload["expected_snapshot_epochs"])
        payload["v1_weights"] = WeightFingerprint.from_dict(payload["v1_weights"])
        return cls(**payload)

    def describe(self) -> str:
        epochs = ", ".join(map(str, self.expected_snapshot_epochs)) or "none"
        return f"froze {self.v1_weights.describe()}; expecting snapshots at {epochs} under {self.model_dir}"


class WeightProtection:
    """The v1 weights as the launch found them, against the same file now.

    A finding here means the run wrote to the file it was only supposed to read -- which no later
    reading of the finetuned checkpoints could reveal, since they would still look healthy.  A
    file that is simply gone is the same finding, because for anything downstream that measures
    against v1 it is the same loss.
    """

    def __init__(self, frozen: WeightFingerprint, current: WeightFingerprint | None) -> None:
        self.frozen = frozen
        self.current = current

    @classmethod
    def observe(cls, frozen: WeightFingerprint) -> "WeightProtection":
        path = Path(frozen.path)
        return cls(frozen=frozen, current=WeightFingerprint.capture(path) if path.is_file() else None)

    @property
    def modified_fields(self) -> tuple[str, ...]:
        """What differs, or ``("missing",)`` when there is no file left to compare."""
        return ("missing",) if self.current is None else self.frozen.differences(self.current)

    def findings(self) -> tuple[RunFinding, ...]:
        if self.current is None:
            return (RunFinding(FindingKind.WEIGHTS_MODIFIED, f"v1 weights are gone: {self.frozen.path} is not on disk"),)
        if not self.modified_fields:
            return ()
        return (
            RunFinding(
                FindingKind.WEIGHTS_MODIFIED,
                f"v1 weights changed since the freeze: {', '.join(self.modified_fields)} differ ({self.frozen.path})",
            ),
        )


@dataclass(frozen=True)
class InventoryReading:
    """What a run's model directory holds, judged against the baseline's expectations."""

    snapshots: tuple[CheckpointRecord, ...]
    missing_epochs: tuple[int, ...]
    main: CheckpointRecord | None
    main_matches_final_snapshot: bool | None
    findings: tuple[RunFinding, ...]


class CheckpointInventory:
    """The run's model directory read against the baseline's expected snapshots.

    The trainer's two outputs live here: the main checkpoint it rewrites every epoch, and the
    ``ckpt_epoch{N}.pt`` copies taken at each snapshot epoch.
    """

    def __init__(self, baseline: LaunchBaseline) -> None:
        self._baseline = baseline

    def read(self) -> InventoryReading:
        snapshots, missing = self._read_snapshots()
        main = self._read_main()
        matches = self._main_matches_final_snapshot(main)
        return InventoryReading(
            snapshots=snapshots,
            missing_epochs=missing,
            main=main,
            main_matches_final_snapshot=matches,
            findings=(*self._snapshot_findings(snapshots, missing), *self._main_findings(main, matches)),
        )

    def _read_snapshots(self) -> tuple[tuple[CheckpointRecord, ...], tuple[int, ...]]:
        records: list[CheckpointRecord] = []
        missing: list[int] = []
        for epoch in self._baseline.expected_snapshot_epochs:
            path = self._baseline.snapshot_path(epoch)
            if path.is_file():
                records.append(CheckpointRecord.read(path))
            else:
                missing.append(epoch)
        return tuple(records), tuple(missing)

    def _read_main(self) -> CheckpointRecord | None:
        path = self._baseline.main_checkpoint_path
        return CheckpointRecord.read(path) if path.is_file() else None

    def _main_matches_final_snapshot(self, main: CheckpointRecord | None) -> bool | None:
        """Whether the main checkpoint holds the final snapshot's weights.

        None when that cannot be asked -- no main checkpoint, no snapshots configured, or the
        final snapshot not written yet.  The comparison reads both files again, which is the cost
        of proving the copy happened rather than assuming it from the log line.
        """
        expected = self._baseline.expected_snapshot_epochs
        if main is None or not expected:
            return None
        final_snapshot = self._baseline.snapshot_path(expected[-1])
        if not final_snapshot.is_file():
            return None
        return WeightFingerprint.capture(Path(main.path)).holds_the_same_weights_as(WeightFingerprint.capture(final_snapshot))

    def _snapshot_findings(self, snapshots: tuple[CheckpointRecord, ...], missing: tuple[int, ...]) -> tuple[RunFinding, ...]:
        findings = [RunFinding(FindingKind.SNAPSHOT_MISSING, f"{self._baseline.snapshot_path(epoch)} is not on disk") for epoch in missing]
        missing_set = set(missing)
        present_epochs = [epoch for epoch in self._baseline.expected_snapshot_epochs if epoch not in missing_set]
        for expected_epoch, record in zip(present_epochs, snapshots):
            if record.unreadable is not None:
                findings.append(self._unreadable_finding(record))
            elif record.epoch != expected_epoch:
                findings.append(
                    RunFinding(FindingKind.EPOCH_MISMATCH, f"{record.path} is the epoch-{expected_epoch} snapshot but holds epoch {record.epoch}")
                )
        findings.extend(self._scale_factor_finding(record) for record in snapshots if record.scale_factor is None and record.unreadable is None)
        return tuple(findings)

    def _main_findings(self, main: CheckpointRecord | None, matches: bool | None) -> tuple[RunFinding, ...]:
        path = self._baseline.main_checkpoint_path
        if main is None:
            return (RunFinding(FindingKind.MAIN_CHECKPOINT_MISSING, f"{path} is not on disk"),)
        if main.unreadable is not None:
            return (self._unreadable_finding(main),)

        findings = []
        finished = main.epoch == self._baseline.n_epochs
        if not finished:
            findings.append(
                RunFinding(
                    FindingKind.MAIN_CHECKPOINT_INCOMPLETE,
                    f"main checkpoint holds epoch {main.epoch}, not the configured {self._baseline.n_epochs}: the run is not finished",
                )
            )
        if main.scale_factor is None:
            findings.append(self._scale_factor_finding(main))
        # Only meaningful once the run claims to be done: before that the main checkpoint is
        # simply newer than the last snapshot, which is what it is supposed to be.
        if finished and matches is False:
            findings.append(
                RunFinding(
                    FindingKind.MAIN_CHECKPOINT_STALE,
                    f"{path} differs from the final snapshot {self._baseline.snapshot_path(self._baseline.n_epochs)}, "
                    "which the trainer writes as a copy of it",
                )
            )
        return tuple(findings)

    @staticmethod
    def _scale_factor_finding(record: CheckpointRecord) -> RunFinding:
        """One complaint for both the snapshots and the main checkpoint.

        Inference reads the scale_factor back out of whichever checkpoint gets selected, so a
        checkpoint without one is unusable wherever it sits -- the two sites differ only in
        which file they are looking at, not in what is wrong with it.
        """
        return RunFinding(
            FindingKind.SCALE_FACTOR_MISSING,
            f"{record.path} holds no finite scale_factor, so it is not usable for inference checkpoint selection",
        )

    @staticmethod
    def _unreadable_finding(record: CheckpointRecord) -> RunFinding:
        """A file torch could not open names itself: a truncated or half-written checkpoint is
        the failure this audit exists to catch, and it must not be mistaken for a missing key."""
        return RunFinding(FindingKind.CHECKPOINT_UNREADABLE, f"{record.path} could not be read as a checkpoint: {record.unreadable}")


@dataclass(frozen=True)
class RunAudit:
    """The finished run's disk against its baseline -- the part of a tier's acceptance record
    that cannot be read off the log."""

    generated_at_utc: str
    baseline_path: str
    verdict: Verdict
    v1_weights: WeightFingerprint | None
    v1_modified_fields: tuple[str, ...]
    v1_declared_scale_factor: float | None
    snapshots: tuple[CheckpointRecord, ...]
    main_checkpoint: CheckpointRecord | None
    main_matches_final_snapshot: bool | None
    missing_snapshot_epochs: tuple[int, ...]
    findings: tuple[RunFinding, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)

    def to_dict(self) -> dict:
        return {
            "generated_at_utc": self.generated_at_utc,
            "baseline": self.baseline_path,
            "verdict": self.verdict,
            "v1_weights": None if self.v1_weights is None else {**self.v1_weights.to_dict(), "modified_fields": list(self.v1_modified_fields)},
            "v1_declared_scale_factor": self.v1_declared_scale_factor,
            "snapshots": [self._checkpoint_payload(record) for record in self.snapshots],
            "main_checkpoint": None if self.main_checkpoint is None else self._checkpoint_payload(self.main_checkpoint),
            "main_matches_final_snapshot": self.main_matches_final_snapshot,
            "missing_snapshot_epochs": list(self.missing_snapshot_epochs),
            "findings": [{"kind": finding.kind, "detail": finding.detail} for finding in self.findings],
        }

    @property
    def finding_kinds(self) -> frozenset[FindingKind]:
        """The kinds this report names -- for a reader who only asks whether one is present."""
        return frozenset(finding.kind for finding in self.findings)

    def describe(self) -> str:
        main_state = "present" if self.main_checkpoint is not None else "MISSING"
        v1_state = "MODIFIED" if self.v1_modified_fields else "unchanged"
        lines = [
            f"verdict {self.verdict}: {len(self.snapshots)} snapshot(s) on disk, "
            f"{len(self.missing_snapshot_epochs)} missing, main checkpoint {main_state}, v1 weights {v1_state}"
        ]
        lines.extend(f"  [{finding.kind}] {finding.detail}" for finding in self.findings)
        return "\n".join(lines)

    def _checkpoint_payload(self, record: CheckpointRecord) -> dict:
        """One checkpoint's row: what it holds, and how far its scale_factor sits from v1's."""
        deviation = None
        if record.scale_factor is not None and self.v1_declared_scale_factor:
            deviation = (record.scale_factor - self.v1_declared_scale_factor) / self.v1_declared_scale_factor
        return {**asdict(record), "scale_factor_relative_deviation": deviation}


class RunAuditor:
    """Compares a finished run's disk against the baseline frozen at its launch.

    A FAIL here is not advice: it says the tier's outputs are not what its acceptance record
    claims, which is why the exit code is the report's other half.
    """

    def __init__(self, baseline: LaunchBaseline, baseline_path: Path) -> None:
        self.baseline = baseline
        self.baseline_path = baseline_path

    def run(self) -> RunAudit:
        protection = WeightProtection.observe(self.baseline.v1_weights)
        inventory = CheckpointInventory(self.baseline).read()
        findings = (*protection.findings(), *inventory.findings)
        return RunAudit(
            generated_at_utc=datetime.now(UTC).isoformat(),
            baseline_path=str(self.baseline_path),
            verdict=Verdict.FAIL if findings else Verdict.PASS,
            v1_weights=protection.current,
            v1_modified_fields=protection.modified_fields,
            v1_declared_scale_factor=self.baseline.v1_declared_scale_factor,
            snapshots=inventory.snapshots,
            main_checkpoint=inventory.main,
            main_matches_final_snapshot=inventory.main_matches_final_snapshot,
            missing_snapshot_epochs=inventory.missing_epochs,
            findings=findings,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="record the launch's v1 weights and expected snapshots")
    freeze.add_argument("-c", "--train-config", type=Path, required=True, help="the training config (its n_epochs and save_interval)")
    freeze.add_argument("-e", "--env-config", type=Path, required=True, help="the env config the run is launched with")
    freeze.add_argument("--report", type=Path, required=True, help="write the baseline here")

    audit = subparsers.add_parser("audit", help="check a finished run against its baseline")
    audit.add_argument("--baseline", type=Path, required=True, help="the baseline written by freeze")
    audit.add_argument("--report", type=Path, help="write the audit report here (default: print only)")

    args = parser.parse_args()

    if args.command == "freeze":
        launch_baseline = LaunchBaseline.freeze(args.train_config, args.env_config)
        launch_baseline.save(args.report)
        print(launch_baseline.describe())
        return

    result = RunAuditor(LaunchBaseline.load(args.baseline), args.baseline).run()
    if args.report:
        result.save(args.report)
    print(result.describe())
    if result.verdict == Verdict.FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
