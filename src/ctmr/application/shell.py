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

"""Application-layer shell engines (ADR-0015 §2, ticket 08).

The two mechanical skeletons the use-case families share, moved out of the
retired harness/scripts layer (git history; the harness shim package was
deleted with issue #175):

- the **phase training shell**: ``PhaseHarness`` epoch loop with early-stop
  file polling (a sticky local observation -- the detecting rank trains the
  epoch tail, and start/end cross-rank consensus collectives make the exit
  unanimous, ADR-0019 §6 / #277), autocast + GradScaler mechanics and loss
  all_reduce for the pre-ADR-0016 kernel path, checkpoint publication via
  the injected ``CheckpointRepository`` port (the tmp atomic publish +
  ``latest.json`` protocol lives in the adapter), rank-0 gating for the
  recipe guard / mkdir / provenance — driven by an injected
  ``PhaseTrainKernel`` (composition, never implementation inheritance);
  kernels migrated per ADR-0016 replace ``train_batch`` with ``train_step`` and
  receive the injected ``GradientExecutor`` carrying the fp16 / bf16 /
  non-AMP strategy (the shell then only aggregates and polls);
  the common finetune argparse surface lives in the stdlib-only sibling
  ``ctmr.application.train_cli`` (``TrainCli``);
- the **dev-eval engine**: ``CheckpointWatcher`` / ``EarlyStopRule`` /
  ``TrendLedger`` plus the shared cohort constants, and the ``WatchEngine`` /
  ``SelectionEmitter`` watch/select skeletons every family's dev
  light-acceptance sidecar builds on (the stage sampler factory, scorer and
  optional post-score extension ride in as collaborators).

The shell holds no recipe value and no domain decision; the stage kernel and
the recipe guard ride in as collaborators. Torch-level: import only where
torch is present (like the retired harness shell it replaced).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist

from ctmr.domain.checkpoints import CheckpointRepository
from ctmr.domain.generation import GradientExecutor

STOP_FILE = ".early_stop"
DEV_EVAL_DIR = "dev_eval"

# Fixed dev cohort quotas per challenge (spec #51), shared by every family's
# dev sidecar; the 16-case cohort order is sha256((sub, case)) within quota.
COHORT_QUOTAS = {"GLI": 4, "SSA": 2, "MEN": 4, "METS": 3, "PED": 3}
MODALITY_TOKENS = {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}
TARGET_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


@dataclass
class TrainContext:
    """What ``kernel.load_models`` returns: the handles the shell's loop steps.

    ``scale`` is the scale_factor the shell hands back to ``checkpoint_payload``.
    """

    trainable: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scale: Any
    device: torch.device


class PhaseTrainKernel(Protocol):
    """The four-method kernel boundary every stage finetune injects (ADR-0011).

    No implementation inheritance: the shell calls these, the kernel owns the
    stage domain (data composition, model hook-up, optimizer/scheduler recipe
    values, the per-batch forward + loss, the checkpoint payload keys).

    A kernel implements exactly ONE of the two per-batch methods; the shell
    probes for ``train_step`` and downgrades to ``train_batch`` only for the
    still-migrating stages:

    - ``train_step`` (migrated stages, ADR-0016): the kernel hands the shell one
      closed single-batch update (its model entity drives loss → backward →
      optimizer step through the injected ``GradientExecutor``) and the shell
      only aggregates and polls.
    - ``train_batch`` (pre-migration stages): forward + loss only; the shell
      drives the update through the injected executor and steps
      ``ctx.scheduler`` itself.
    """

    def build_loader(self):
        """Build the training DataLoader (partitioned per local rank)."""
        ...

    def load_models(self, loader) -> TrainContext:
        """Hook up models + construct optimizer / lr_scheduler (recipe values live here)."""
        ...

    def train_batch(self, batch) -> torch.Tensor:
        """Single-batch forward + loss (pre-migration stages; the shell drives the update)."""
        ...

    def train_step(self, batch, gradient_executor) -> torch.Tensor:
        """Single-batch closed update (migrated stages): the kernel's model entity drives it."""
        ...

    def checkpoint_payload(self, epoch: int, avg_loss: float, scale) -> dict:
        """The per-stage checkpoint payload (key set kept: unet_state_dict / controlnet_state_dict)."""
        ...


class PhaseHarness:
    """The shared training shell: mechanical sequence only, no recipe or domain values.

    Every collaborator rides in injected (ADR-0019 §1 terminal state, #276):
    the per-batch precision strategy is always a composition-root-assembled
    ``GradientExecutor`` (migrated kernels, ``train_step``, receive it inside
    their closed update; still-migrating kernels, ``train_batch``, have the
    shell drive it) and the checkpoint publication goes through the injected
    ``CheckpointRepository`` port. The shell never re-implements autocast,
    GradScaler state or the tmp atomic publish itself, and both injections
    are validated at construction -- before any checkpoint loading or first
    batch. The early-stop file is polled locally at the epoch start and at
    every batch boundary, but an observation only sets a sticky flag: the
    detecting rank stays in the batch stream and trains the epoch tail (the
    DDP-wrapped kernels issue one gradient collective per batch, so no rank
    may leave the loop early), and start/end consensus collectives on every
    rank make the exit unanimous (ADR-0019 §6, #277).
    """

    ITER_LOG_EVERY = 50

    def __init__(
        self,
        kernel: PhaseTrainKernel,
        model_dir,
        n_epochs: int,
        local_rank: int,
        logger,
        gradient_executor: GradientExecutor | None = None,
        checkpoint_repository: CheckpointRepository | None = None,
        recipe_check: Callable[[], Any] | None = None,
        provenance: TrainProvenanceWriter | None = None,
        validation: ValidationPhase | None = None,
    ):
        self._kernel = kernel
        self._model_dir = model_dir
        self._n_epochs = n_epochs
        self._local_rank = local_rank
        self._logger = logger
        self._recipe_check = recipe_check
        self._provenance = provenance
        if gradient_executor is None:
            raise ValueError("no gradient_executor was injected (ADR-0016/ADR-0019: the composition root assembles the precision strategy)")
        if checkpoint_repository is None:
            raise ValueError("no checkpoint_repository was injected (ADR-0015 §4/ADR-0019: the composition root assembles the weight store)")
        if validation is not None and (validation.validator is None or validation.rule is None):
            raise ValueError("the validation phase requires both the validator and the early-stop rule (ADR-0019 §5)")
        self._gradient_executor = gradient_executor
        self._repository = checkpoint_repository
        self._validation = validation

    def run(self):
        """Drive one full training run: recipe guard -> provenance -> loop -> cleanup.

        The loop tail carries the embedded periodic validation stage (ADR-0019
        §5, #278): after an epoch has trained and published, every
        ``ValidationPhase.every`` epochs the shell runs the sharded validation
        and evaluates the early-stop rule on the accumulated trend -- a fired
        rule ends the run on every rank through a MAX consensus (never through
        file-polling timing).
        """
        if self._local_rank == 0:
            if self._recipe_check is not None:
                self._recipe_check()
            Path(self._model_dir).mkdir(parents=True, exist_ok=True)
            if self._provenance is not None:
                self._provenance.write(Path(self._model_dir) / "train_provenance.json")
        loader = self._kernel.build_loader()
        ctx = self._kernel.load_models(loader)
        torch.set_float32_matmul_precision("highest")
        for epoch in range(self._n_epochs):
            if self._train_one_epoch(epoch, loader, ctx):
                break
            if self._validation is not None and self._run_validation(epoch, ctx):
                break
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    def _stop_requested(self) -> bool:
        return (Path(self._model_dir) / STOP_FILE).is_file()

    def _stop_consensus(self, ctx, stop_seen: bool) -> bool:
        """Merge every rank's local stop observation into one verdict (ADR-0019 §6).

        A MAX all_reduce over the per-rank flags: whichever rank observed the
        stop file -- at whatever batch index -- makes the exit unanimous, and
        the rendezvous keeps the collective stream aligned (each call site is
        executed by every rank exactly once per epoch). Single-rank runs just
        take the local flag.
        """
        if not dist.is_initialized():
            return stop_seen
        flag = torch.tensor(1 if stop_seen else 0, dtype=torch.int64, device=ctx.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _validation_ok_consensus(self, ctx, ok: bool) -> bool:
        """Merge every rank's stage success into one verdict: ALL must succeed (MIN).

        The mirror of the MAX stop consensus: any rank whose validation stage
        raised folds the skip into a unanimous one, so no rank appends a trend
        point the others lack -- the rank-0 ledger never leads the in-memory
        trend, and a MAX stop verdict never fires on a record rank 0 did not
        append. Single-rank runs take the local flag.
        """
        if not dist.is_initialized():
            return ok
        flag = torch.tensor(1 if ok else 0, dtype=torch.int64, device=ctx.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _train_one_epoch(self, epoch, loader, ctx):
        """Run one epoch; returns True when the run must stop on every rank.

        The stop file is polled locally at the epoch start and at every batch
        boundary, but an observation only sets the sticky ``stop_seen`` flag:
        the detecting rank stays in the batch stream and trains the epoch
        tail, because the production kernels wrap their trainables in DDP and
        every batch's backward issues a gradient collective -- leaving the
        loop early would pair a later peer's gradient all_reduce with this
        rank's unrelated consensus collective and stall the communicator
        (ADR-0019 §6, #277). Two consensuses per epoch make the exit
        unanimous: the start consensus skips the epoch before any batch, the
        end consensus skips the loss all_reduce and the checkpoint publish on
        every rank together.
        """
        iteration = 0
        stop_seen = self._stop_requested()
        if self._stop_consensus(ctx, stop_seen):
            self._logger.info(f"early-stop file present; halting before epoch {epoch + 1}")
            return True
        if self._local_rank == 0:
            self._logger.info(f"Epoch {epoch + 1}, lr {ctx.optimizer.param_groups[0]['lr']}.")
        loss_totals = torch.zeros(2, dtype=torch.float, device=ctx.device)
        ctx.trainable.train()
        train_step = getattr(self._kernel, "train_step", None)
        for batch in loader:
            if not stop_seen:
                stop_seen = self._stop_requested()
            iteration += 1
            if train_step is None:
                # Pre-ADR-0016 mechanical path: the executor reproduces the shell's
                # former zero_grad → autocast-wrapped loss → backward → step order.
                loss = self._gradient_executor.run(lambda: self._kernel.train_batch(batch), ctx.trainable, ctx.optimizer)
                ctx.scheduler.step()
            else:
                # Migrated stages: the kernel's model entity drives one closed
                # update; the shell only aggregates and polls.
                loss = train_step(batch, self._gradient_executor)
            loss_totals[0] += loss.item()
            loss_totals[1] += 1.0
            if self._local_rank == 0 and iteration % self.ITER_LOG_EVERY == 0:
                self._logger.info(
                    f"[{str(datetime.now())[:19]}] epoch {epoch + 1}, iter {iteration}/{len(loader)}, "
                    f"loss: {loss.item():.4f}, lr: {ctx.optimizer.param_groups[0]['lr']:.12f}."
                )
        if self._stop_consensus(ctx, stop_seen):
            self._logger.info(f"early-stop file present; halting mid-epoch {epoch + 1}")
            return True
        if dist.is_initialized():
            dist.all_reduce(loss_totals, op=torch.distributed.ReduceOp.SUM)
        if self._local_rank == 0:
            self._publish_checkpoint(epoch, ctx, loss_totals)
        return False

    def _publish_checkpoint(self, epoch, ctx, loss_totals):
        average = (loss_totals[0] / loss_totals[1]).item()
        payload = self._kernel.checkpoint_payload(epoch + 1, average, ctx.scale)
        # The shell's single publication call point: the repository owns the
        # tmp atomic publish (the dev sidecar polls epoch_*.pt and must never
        # observe a partial write) + latest.json pointer protocol (ADR-0015 §4).
        path = self._repository.save(payload, epoch + 1)
        self._logger.info(f"epoch {epoch + 1} average loss: {average:.4f} -> {path}")

    def _run_validation(self, epoch, ctx):
        """The embedded periodic validation stage (ADR-0019 §5, #278); True ends the run.

        Called only on ``every``-epoch boundaries, after the epoch has trained
        and published (the trend point's ``checkpoint`` therefore exists, the
        same contract the retired sidecar's records kept). The mechanical
        sequence is the shell's: swap the model to eval, drive the injected
        validator (shard → sample → all_gather → score), append the WatchEngine
        record skeleton to the ledger (rank 0) and the in-memory trend (every
        rank), evaluate the pre-recorded rule, and merge the stop verdict with
        a MAX all_reduce so the exit is unanimous by construction -- never by
        file-visibility timing. A failing stage degrades to a logged skip:
        training is the main job and the next boundary retries. The skip is
        itself unanimous (a MIN consensus): any rank whose stage raised folds
        the whole world into skipping the point, so the rank-0 ledger and the
        in-memory trend advance in lockstep and no stop verdict fires on a
        record rank 0 did not append.
        """
        phase = self._validation
        epoch_number = epoch + 1
        if epoch_number % phase.every != 0:
            return False
        eval_root = Path(self._model_dir) / DEV_EVAL_DIR
        ctx.trainable.eval()
        ok = True
        fields, log_line = {}, ""
        # Fork the RNG around the stage: the live sampler reseeds per sample, and
        # that must never reach the training stream (shuffling, RF timesteps,
        # modality perturbation) -- enabling --val-every leaves the training math
        # bit-identical to a validation-free run (codex review, PR #301).
        rng_devices = [ctx.device] if ctx.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=rng_devices):
                fields, log_line = phase.validator.validate(ctx, epoch_number, eval_root)
        except Exception as error:
            ok = False
            self._logger.warning(f"periodic validation epoch {epoch_number} failed; training continues: {error}")
        finally:
            ctx.trainable.train()
        if not self._validation_ok_consensus(ctx, ok):
            return False
        record = {
            "eval_utc": datetime.now(UTC).isoformat(),
            "epoch": epoch_number,
            "checkpoint": str(Path(self._model_dir) / f"epoch_{epoch_number}.pt"),
            **fields,
            "cohort_file": phase.validator.cohort_file,
        }
        if self._local_rank == 0:
            TrendLedger(eval_root).append(record)
            epoch_dir = eval_root / f"epoch_{epoch_number}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "trend.json").write_text(json.dumps(record, indent=2) + "\n")
        # Every rank keeps the trend in memory: the boundary evaluation below
        # must not depend on when rank 0's append becomes visible on the share.
        phase.records.append(record)
        stop, reason = phase.rule.should_stop(phase.records)
        stop = self._stop_consensus(ctx, stop)
        self._logger.info(f"[eval] epoch {epoch_number}: {log_line} stop={stop} ({reason})")
        if stop:
            if self._local_rank == 0:
                (Path(self._model_dir) / STOP_FILE).write_text(json.dumps({"reason": reason, "epoch": epoch_number}) + "\n")
            self._logger.info(f"early-stop fired at the epoch {epoch_number} validation boundary")
        return stop


class TrainProvenanceWriter:
    """The shared provenance skeleton; stage domain fields inject via ``domain_fields``.

    Field order reproduces the pre-#111 writers: the six skeleton entries, then the
    stage domain block (``data_lists``/``base_ckpt`` vs ``data_list``/
    ``trained_diffusion_path``/``replay`` plus ``hyperparameters``), then the
    amp/dist/torch/git trailer. ``script`` / ``git_commit`` are self-referential
    metadata (the ADR-0011 gate exempts their values, not their presence).
    """

    def __init__(self, args, local_rank, logger, domain_fields: Callable[[], dict], script_path=None):
        self._args = args
        self._local_rank = local_rank
        self._logger = logger
        self._domain_fields = domain_fields
        self._script_path = Path(script_path) if script_path is not None else Path(__file__)

    def write(self, path):
        if self._local_rank != 0:
            return None
        provenance = {
            "written_utc": datetime.now(UTC).isoformat(),
            "script": str(self._script_path.resolve()),
            "env_config": str(Path(self._args.env_config_path).resolve()),
            "model_config": str(Path(self._args.model_config_path).resolve()),
            "model_def": str(Path(self._args.model_def_path).resolve()),
            **self._domain_fields(),
            "amp_dtype": self._args.amp_dtype,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "torch_version": torch.__version__,
            "git_commit": self._git_commit(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(provenance, indent=2) + "\n")
        self._logger.info(f"train provenance -> {out}")
        return out

    def _git_commit(self):
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(self._script_path.parent)
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


class CheckpointWatcher:
    """Polls the trainer's epoch checkpoints; yields un-evaluated eval points."""

    def __init__(self, ckpt_dir, eval_every, max_epoch, done_epochs=()):
        self._ckpt_dir = Path(ckpt_dir)
        self._eval_every = eval_every
        self._max_epoch = max_epoch
        # Seed from the ledger so a sidecar restart does not re-evaluate history
        # (re-appended trend points would corrupt the early-stop patience count).
        self._done = set(done_epochs)

    def pending(self):
        found = []
        for path in sorted(self._ckpt_dir.glob("epoch_*.pt")):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if epoch % self._eval_every == 0 and epoch <= self._max_epoch and epoch not in self._done:
                found.append((epoch, path))
        return sorted(found)

    def mark_done(self, epoch):
        self._done.add(epoch)


class EarlyStopRule:
    """Pre-recorded rule: patience on the mean dev trend (never before min_epoch).

    ``direction`` selects whether the trend metric is minimized (``min``, the FID
    rules the modality-label/mask families pre-registered) or maximized (``max``,
    the paired PSNR/SSIM rule the cross-modal family pre-registered); the default
    keeps modality-label/mask behavior byte-identical.
    """

    RULE_TEXT = (
        "metric m(N) = mean over t1n/t1c/t2w/t2f of plane-mean dev 2.5D RadImageNet FID on the "
        "fixed 16-case dev cohort (fixed seeds, cfg=10, 30 steps); stop when N >= {min_epoch} and "
        "the last {patience} consecutive evals set no new best m; hard cap = trainer n_epochs"
    )

    def __init__(self, patience, min_epoch, max_epoch, direction="min"):
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
        self.patience = patience
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.direction = direction

    def rule_text(self):
        return self.RULE_TEXT.format(min_epoch=self.min_epoch, patience=self.patience)

    def should_stop(self, trend):
        """trend: list of {epoch, m} in epoch order; returns (stop, reason)."""
        sign = -1 if self.direction == "max" else 1
        points = [point for point in trend if point["m"] is not None]
        if not points:
            return False, "no eval points yet"
        last_epoch = points[-1]["epoch"]
        if last_epoch < self.min_epoch:
            return False, f"before min_epoch {self.min_epoch}"
        best_index = min(range(len(points)), key=lambda i: (sign * points[i]["m"], i))
        best = points[best_index]["m"]
        since_best = len(points) - 1 - best_index
        if since_best >= self.patience:
            return True, f"no new best for {since_best} evals (best {best:.4f})"
        return False, f"best {best:.4f}, {since_best} evals since"

    @staticmethod
    def selection(trend, direction="min", metric_name="mean_fid"):
        points = [point for point in trend if point["m"] is not None]
        if not points:
            return None
        sign = -1 if direction == "max" else 1
        best = min(points, key=lambda point: (sign * point["m"], point["epoch"]))
        return {"epoch": best["epoch"], metric_name: best["m"], "checkpoint": best.get("checkpoint")}


class TrendLedger:
    """Appends eval records to dev_trend.jsonl and keeps the cohort + rule on disk.

    ``read`` is incremental under the append-only protocol: it parses only the
    bytes appended since the last call (the watch loop re-reads the ledger
    after every eval point, so a full re-parse per poll would be quadratic in
    the run length).  The accumulated view equals a full re-read -- pinned by
    test.
    """

    def __init__(self, eval_root):
        self._root = Path(eval_root)
        self._offset = 0
        self._records = []

    def path(self):
        return self._root / "dev_trend.jsonl"

    def read(self):
        path = self.path()
        if path.is_file():
            size = path.stat().st_size
            if size > self._offset:
                with open(path) as handle:
                    handle.seek(self._offset)
                    new_text = handle.read()
                self._offset = size
                self._records.extend(json.loads(line) for line in new_text.splitlines() if line.strip())
        return list(self._records)

    def append(self, record):
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self.path(), "a") as handle:
            handle.write(json.dumps(record) + "\n")


class WatchEngine:
    """The dev watch polling engine (ADR-0011): dedup, idle-exit, ledger, record, early-stop.

    The mechanical loop only -- the stage domain rides in through three
    collaborators: the ``sampler_factory`` (``(checkpoint_path, out_dir) ->
    samples``, the per-candidate sampling run), the ``scorer`` (``(samples) ->
    (record_fields, log_line)``, the stage trend metric) and the optional
    ``post_score`` extension (``(epoch, samples, epoch_dir) -> extra record
    fields``, e.g. the mask family's instrument + round-trip trends; it owns
    its own failure tolerance).  The engine holds no recipe value and no
    domain decision.
    """

    def __init__(
        self,
        ckpt_dir,
        eval_root,
        eval_every: int,
        max_epoch: int,
        rule: EarlyStopRule,
        sampler_factory: Callable,
        scorer: Callable,
        poll_seconds: float = 60.0,
        idle_exit_seconds: float = 0.0,
        post_score: Callable | None = None,
    ):
        self._ckpt_dir = Path(ckpt_dir)
        self._eval_root = Path(eval_root)
        self._eval_every = eval_every
        self._max_epoch = max_epoch
        self._rule = rule
        self._sampler_factory = sampler_factory
        self._scorer = scorer
        self._poll_seconds = poll_seconds
        self._idle_exit_seconds = idle_exit_seconds
        self._post_score = post_score

    def run(self, cohort_file=None):
        """Poll, dedup, score, append and early-stop; returns the process exit code."""
        ledger = TrendLedger(self._eval_root)
        watcher = CheckpointWatcher(self._ckpt_dir, self._eval_every, self._max_epoch, {record["epoch"] for record in ledger.read()})
        idle_since = None
        while True:
            pending = watcher.pending()
            if not pending:
                if self._idle_exit_seconds and idle_since is not None and time.time() - idle_since > self._idle_exit_seconds:
                    break
                if self._idle_exit_seconds and idle_since is None:
                    idle_since = time.time()
                time.sleep(self._poll_seconds)
                continue
            idle_since = None
            for epoch, path in pending:
                if any(record["epoch"] == epoch for record in ledger.read()):
                    watcher.mark_done(epoch)
                    continue
                epoch_dir = self._eval_root / f"epoch_{epoch}"
                try:
                    samples = self._sampler_factory(path, epoch_dir / "samples")
                    fields, log_line = self._scorer(samples)
                except Exception as error:
                    # A broken checkpoint, a transient network/model failure, or any
                    # single-epoch hiccup must not kill the sidecar: without it
                    # nobody writes .early_stop. Skip and retry on the next poll.
                    print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                    continue
                record = {
                    "eval_utc": datetime.now(UTC).isoformat(),
                    "epoch": epoch,
                    "checkpoint": str(path),
                    **fields,
                }
                if self._post_score is not None:
                    record.update(self._post_score(epoch, samples, epoch_dir))
                record["cohort_file"] = cohort_file
                epoch_dir.mkdir(parents=True, exist_ok=True)
                ledger.append(record)
                (epoch_dir / "trend.json").write_text(json.dumps(record, indent=2) + "\n")
                watcher.mark_done(epoch)
                stop, reason = self._rule.should_stop(ledger.read())
                print(f"[eval] epoch {epoch}: {log_line} stop={stop} ({reason})", flush=True)
                if stop:
                    (self._ckpt_dir / STOP_FILE).write_text(json.dumps({"reason": reason, "epoch": epoch}) + "\n")
                    print(f"early-stop fired ({reason}); wrote {self._ckpt_dir / STOP_FILE}", flush=True)
                    return 0
        return 0


class SelectionEmitter:
    """Emits the final dev-side checkpoint selection for the phase-run contract (argmin/argmax)."""

    def __init__(self, eval_root):
        self._eval_root = Path(eval_root)

    def emit(
        self,
        out,
        rule_text: str,
        direction: str = "min",
        metric_name: str = "mean_fid",
        extra_fields: Callable | None = None,
        summary_extra: Callable | None = None,
    ) -> int:
        """Writes the selection contract JSON; returns the process exit code.

        ``extra_fields(trend, selection) -> dict`` merges stage-specific trail
        fields after the base selection (the cross-modal PSNR trail); they land
        before the ``rule``/``trend``/``recorded_utc`` envelope keys.
        """
        trend = TrendLedger(self._eval_root).read()
        selection = EarlyStopRule.selection(trend, direction=direction, metric_name=metric_name)
        if selection is None:
            print("no eval points; nothing to select", file=sys.stderr)
            return 1
        if extra_fields is not None:
            selection.update(extra_fields(trend, selection))
        selection["rule"] = rule_text
        selection["trend"] = trend
        selection["recorded_utc"] = datetime.now(UTC).isoformat()
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(selection, indent=2) + "\n")
        summary = f"selection -> {out} (epoch {selection['epoch']}, {metric_name} {selection[metric_name]:.4f})"
        if summary_extra is not None:
            summary += summary_extra(selection)
        print(summary)
        return 0


class ValidationSkippedError(Exception):
    """The whole world agreed to skip a validation point: a shard's sampler failed
    before the gather, so no entry set exists to score. Raised on every rank by
    the pre-gather MIN consensus -- the shell's catch degrades it to a logged,
    unanimous skip (training is the main job)."""


class PeriodicValidator:
    """The embedded validation stage's domain (ADR-0019 §5, #278): shard, sample, all_gather, score.

    The injected ``sampler`` ``(ctx, shard_items, out_dir) -> entries`` renders
    this rank's cohort shard with the live training weights and returns one
    entry per item (the family carries its own entry shape -- e.g. the
    plane-mean feature vectors); the shell-level ``all_gather_object`` merges
    every rank's entries so the injected ``scorer`` ``(entries) -> (fields,
    log_line)`` always sees the FULL cohort no matter the world size -- the
    gathered view is the single-card full-cohort view up to floating-point
    summation order. The shell owns the boundary trigger, the eval/train swap
    and the ledger; this object owns only the sharding and reduction shape.
    """

    def __init__(self, items, sampler, scorer, local_rank, device, cohort_file="dev_cohort.json"):
        self._items = list(items)
        self._sampler = sampler
        self._scorer = scorer
        self._local_rank = local_rank
        self._device = device
        self.cohort_file = cohort_file

    def validate(self, ctx, epoch, eval_root=None):
        """Sample this rank's shard, all_gather the entries, score the full cohort.

        A shard-local sampling failure (OOM, a missing spacing file, output I/O)
        must not strand the healthy ranks inside ``all_gather_object``: the
        sampler's exception is caught into a flag and EVERY rank reaches the
        pre-gather MIN consensus, so the whole world agrees to skip the point
        before anyone gathers (codex review, PR #301). The scorer runs only on
        the gathered full cohort, after that rendezvous.
        """
        rank, world = self._rank_and_world()
        shard = self._items[rank::world]
        out_dir = None if eval_root is None else Path(eval_root) / f"epoch_{epoch}" / "samples"
        entries, ok, error = self._sampled(ctx, shard, out_dir)
        if not self._ranks_ok(ok):
            raise ValidationSkippedError(f"a shard failed to sample at epoch {epoch}: {error}")
        return self._scorer(self._gathered(entries))

    def _sampled(self, ctx, shard, out_dir):
        """Run the sampler, folding any failure into a flag so every rank (failed or
        not) reaches the pre-gather consensus instead of diverging at the gather."""
        try:
            return self._sampler(ctx, shard, out_dir), True, None
        except Exception as error:
            return [], False, error

    def _ranks_ok(self, ok):
        """MIN all_reduce over the per-rank sampler success: all succeed or all skip.

        The collective lives on the construction-injected device -- ``ctx`` is only
        the sampler's payload and may be None (validator-only call sites)."""
        if not dist.is_initialized():
            return ok
        flag = torch.tensor(1 if ok else 0, dtype=torch.int64, device=self._device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _rank_and_world(self):
        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return self._local_rank, 1

    @staticmethod
    def _gathered(entries):
        """The all-object gather: rank-interleaved shard inputs, rank-ordered outputs.

        Every rank contributes one entry list; the concatenation in rank order
        is the full cohort. Identical items never collide -- the shards are
        disjoint by construction (``items[rank::world]``).
        """
        if not dist.is_initialized():
            return entries
        world = dist.get_world_size()
        chunks = [None] * world
        dist.all_gather_object(chunks, entries)
        return [entry for chunk in chunks for entry in chunk]


@dataclass
class ValidationPhase:
    """The embedded periodic validation collaborators (ADR-0019 §5, #278).

    The shell consumes ``every``/``validator``/``rule``; ``records`` is the
    run's in-memory trend (one validated point per boundary, appended on every
    rank) -- the boundary evaluation's view, pinned equal to the rank-0
    ``dev_trend.jsonl`` ledger by test.
    """

    every: int
    validator: PeriodicValidator | None
    rule: EarlyStopRule | None
    records: list = field(default_factory=list)
