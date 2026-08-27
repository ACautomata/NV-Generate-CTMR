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

"""Phase training shell (ADR-0011 decision 2, issue #111).

``PhaseHarness`` owns the whole mechanical sequence the three symmetric stage
finetunes used to copy: the epoch loop with early-stop file polling at epoch
boundaries and mid-epoch, autocast + GradScaler mechanics, loss all_reduce,
checkpoint publication via ``CheckpointRepository`` (the tmp atomic publish +
``latest.json`` protocol lives there), rank-0 gating for the recipe guard /
mkdir / provenance. Stage-specific work
rides in through ``PhaseTrainKernel`` (composition, never implementation
inheritance) and a ``recipe_check`` first-class hook; the shell holds no
recipe value and no domain decision.

Torch-level: import only where torch is present; ``cli`` and ``recipe`` are the
stdlib-only harness pieces.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast

from ctmr.infrastructure.checkpoints import CheckpointRepository

STOP_FILE = ".early_stop"


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

    No implementation inheritance: the harness calls these, the kernel owns the
    stage domain (data composition, model hook-up, optimizer/scheduler recipe
    values, the per-batch forward + loss, the checkpoint payload keys).
    """

    def build_loader(self):
        """Build the training DataLoader (partitioned per local rank)."""
        ...

    def load_models(self, loader) -> TrainContext:
        """Hook up models + construct optimizer / lr_scheduler (recipe values live here)."""
        ...

    def train_batch(self, batch) -> torch.Tensor:
        """Single-batch forward + loss (already containing scale noise/scheduler usage)."""
        ...

    def checkpoint_payload(self, epoch: int, avg_loss: float, scale) -> dict:
        """The per-stage checkpoint payload (key set kept: unet_state_dict / controlnet_state_dict)."""
        ...


class PhaseHarness:
    """The shared training shell: mechanical sequence only, no recipe or domain values."""

    ITER_LOG_EVERY = 50

    def __init__(
        self,
        kernel: PhaseTrainKernel,
        model_dir,
        n_epochs: int,
        amp: bool,
        amp_dtype: str,
        local_rank: int,
        logger,
        recipe_check: Callable[[], Any] | None = None,
        provenance: TrainProvenanceWriter | None = None,
    ):
        self._kernel = kernel
        self._model_dir = model_dir
        self._n_epochs = n_epochs
        self._amp = amp
        self._amp_dtype = amp_dtype
        self._local_rank = local_rank
        self._logger = logger
        self._recipe_check = recipe_check
        self._provenance = provenance
        self._repository = CheckpointRepository(Path(model_dir))

    def run(self):
        """Drive one full training run: recipe guard -> provenance -> loop -> cleanup."""
        if self._local_rank == 0:
            if self._recipe_check is not None:
                self._recipe_check()
            Path(self._model_dir).mkdir(parents=True, exist_ok=True)
            if self._provenance is not None:
                self._provenance.write(Path(self._model_dir) / "train_provenance.json")
        loader = self._kernel.build_loader()
        ctx = self._kernel.load_models(loader)
        scaler = GradScaler("cuda")
        torch.set_float32_matmul_precision("highest")
        for epoch in range(self._n_epochs):
            if self._stop_requested():
                self._logger.info(f"early-stop file present; halting before epoch {epoch + 1}")
                break
            self._train_one_epoch(epoch, loader, ctx, scaler)
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    def _stop_requested(self) -> bool:
        return (Path(self._model_dir) / STOP_FILE).is_file()

    def _train_one_epoch(self, epoch, loader, ctx, scaler):
        if self._local_rank == 0:
            self._logger.info(f"Epoch {epoch + 1}, lr {ctx.optimizer.param_groups[0]['lr']}.")
        iteration = 0
        loss_totals = torch.zeros(2, dtype=torch.float, device=ctx.device)
        ctx.trainable.train()
        for batch in loader:
            if self._stop_requested():
                self._logger.info(f"early-stop file present; halting mid-epoch {epoch + 1}")
                return
            iteration += 1
            ctx.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16 if self._amp_dtype == "bf16" else torch.float16, enabled=self._amp):
                loss = self._kernel.train_batch(batch)
            if self._amp and self._amp_dtype == "fp16":
                scaler.scale(loss).backward()
                scaler.step(ctx.optimizer)
                scaler.update()
            else:
                loss.backward()
                ctx.optimizer.step()
            ctx.scheduler.step()
            loss_totals[0] += loss.item()
            loss_totals[1] += 1.0
            if self._local_rank == 0 and iteration % self.ITER_LOG_EVERY == 0:
                self._logger.info(
                    f"[{str(datetime.now())[:19]}] epoch {epoch + 1}, iter {iteration}/{len(loader)}, "
                    f"loss: {loss.item():.4f}, lr: {ctx.optimizer.param_groups[0]['lr']:.12f}."
                )
        if dist.is_initialized():
            dist.all_reduce(loss_totals, op=torch.distributed.ReduceOp.SUM)
        if self._local_rank == 0:
            self._publish_checkpoint(epoch, ctx, loss_totals)

    def _publish_checkpoint(self, epoch, ctx, loss_totals):
        average = (loss_totals[0] / loss_totals[1]).item()
        payload = self._kernel.checkpoint_payload(epoch + 1, average, ctx.scale)
        # The shell's single publication call point: the repository owns the
        # tmp atomic publish + latest.json pointer protocol (ADR-0015 §4).
        path = self._repository.save(payload, epoch + 1)
        self._logger.info(f"epoch {epoch + 1} average loss: {average:.4f} -> {path}")


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
