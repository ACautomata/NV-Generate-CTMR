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

"""Cross-rank early-stop gates for the phase training shell (issue #277, ADR-0019 §6).

Pre-#277 the mid-epoch halt made the detecting rank ``return`` straight out of
the epoch, skipping the epoch-end ``loss_totals`` all_reduce -- a rank that
had not observed the stop file yet kept training and then blocked forever on
that all_reduce (or failed on the process-group timeout), because ranks with
different detection timing no longer issued the same collectives.  The first
fix pinned here: the local observation only sets a sticky flag, the detecting
rank stays in the batch stream and trains the epoch tail (the production
kernels wrap their trainables in DDP, so every batch's backward issues a
gradient collective -- leaving the loop early pairs a peer's gradient
all_reduce with the detector's unrelated consensus collective), and start/end
MAX all_reduce consensuses make the exit unanimous: the start consensus skips
the epoch before any batch, the end consensus skips the loss all_reduce and
the checkpoint publish on every rank together.  The fake kernel below is
DDP-wrapped for exactly this reason -- a non-DDP fake would hide the desync.

Two tiers mirror ADR-0015 §6:

- the torch-marked CPU tier (gloo backend, file:// init) runs for real in CI
  and locally; its staggered arm is the deterministic regression -- only one
  rank ever observes the stop, the other must still exit through the
  consensus instead of hanging in the gradient stream.
- the gpu-marked tier (nccl backend, env:// init, one CUDA device per rank)
  is the issue's server-side gate (``pytest --run-gpu``): no NCCL error and
  a clean all-rank exit under the same observation patterns.

Each spawned worker drives the real ``PhaseHarness`` with the DDP fake kernel
and drops a ``worker_rank_<N>.json`` outcome file;
``torch.multiprocessing.spawn(join=True)`` re-raises any pre-outcome worker
failure, a failure inside the run is recorded as an ``error`` key the parent
asserts absent -- the clean return of every worker IS the "no NCCL error,
all ranks exited" assertion -- and the process-group timeout bounds a
regression hang.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ctmr.application.shell import STOP_FILE, PhaseHarness, TrainContext
from ctmr.infrastructure.checkpoints import CheckpointRepository
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

pytestmark = pytest.mark.torch

WORLD = 2
BATCHES_PER_EPOCH = 4
N_EPOCHS = 3
STOP_AFTER_BATCH = 6  # cumulative: two batches into epoch 2
GROUP_TIMEOUT_SECONDS = 60


class RankKernel:
    """Tiny fake PhaseTrainKernel per rank: a DDP-wrapped Linear.

    The DDP wrapper is the point: each batch's backward then issues the real
    per-batch gradient collective the production kernels issue (P1 DM + the
    P2/P3 ControlNets are DDP-wrapped), so the shell's stop handling is gated
    against the exact collective stream that the pre-fix code desynced.
    """

    def __init__(self, device, on_batch=None):
        self.device = device
        self.on_batch = on_batch
        self.batches = 0
        self.trainable = None
        self.optimizer = None
        self.scheduler = None
        self.batch_list = [{"x": float(i)} for i in range(BATCHES_PER_EPOCH)]

    def build_loader(self):
        return self.batch_list

    def load_models(self, loader):
        module = torch.nn.Linear(1, 1, bias=False).to(self.device)
        self.trainable = DistributedDataParallel(module)
        self.optimizer = torch.optim.SGD(self.trainable.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        return TrainContext(
            trainable=self.trainable,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scale=torch.tensor(1.0, device=self.device),
            device=self.device,
        )

    def train_batch(self, batch):
        self.batches += 1
        if self.on_batch is not None:
            self.on_batch(self.batches)
        x = torch.tensor([batch["x"]], device=self.device)
        return self.trainable(x).sum() ** 2

    def checkpoint_payload(self, epoch, avg_loss, scale):
        return {"epoch": epoch, "loss": avg_loss, "scale_factor": scale, "fake_state_dict": {}}


def _dist_worker(rank, world_size, model_dir, scenario, backend):
    """One real PhaseHarness.run() on one rank, then the per-rank outcome file.

    ``scenario`` selects the stop observation pattern:

    - ``full``: nobody ever observes a stop; the plain two-rank training arm.
    - ``boundary``: the stop file exists before the run; both ranks poll the
      real file and must halt before any batch.
    - ``uniform``: rank 1 writes the real stop file two batches into epoch 2;
      every rank keeps polling the real file, trains the epoch tail and the
      run halts at the epoch end without publishing it.
    - ``staggered``: only rank 1 ever observes the stop (rank 0's poll is
      blind) -- the deterministic form of the pre-#277 desync.

    A failure inside the run is recorded in the outcome file instead of dying
    with a bare traceback: the parent asserts the absence of an ``error``
    key, which turns an NCCL error or a process-group timeout into a precise
    per-rank assertion failure (a clean return of every worker IS the "no
    NCCL error, all ranks exited" gate).
    """
    if backend == "nccl":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://" if backend == "nccl" else f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kernel = RankKernel(device=device)
    if scenario == "uniform":

        def _write_stop(batches):
            if rank == 1 and batches == STOP_AFTER_BATCH:
                (Path(model_dir) / STOP_FILE).touch()

        kernel.on_batch = _write_stop
    root = Path(model_dir)
    harness = PhaseHarness(
        kernel=kernel,
        model_dir=root,
        n_epochs=N_EPOCHS,
        local_rank=rank,
        logger=logging.getLogger(f"rank-{rank}"),
        gradient_executor=PlainGradientExecutor(),
        checkpoint_repository=CheckpointRepository(root),
    )
    if scenario == "staggered":

        def _staggered_poll():
            return rank == 1 and kernel.batches >= STOP_AFTER_BATCH

        harness._stop_requested = _staggered_poll
    if scenario == "boundary":
        (root / STOP_FILE).touch()
    try:
        harness.run()
    except Exception as error:
        (root / f"worker_rank_{rank}.json").write_text(
            json.dumps({"rank": rank, "batches": kernel.batches, "error": f"{type(error).__name__}: {error}"}) + "\n"
        )
        return
    (root / f"worker_rank_{rank}.json").write_text(json.dumps({"rank": rank, "batches": kernel.batches}) + "\n")


def _spawn_ranks(monkeypatch, model_dir, scenario, backend, master_port=None):
    # spawn children re-import this module, so its directory must resolve in a
    # fresh interpreter (pytest's in-memory sys.path does not propagate).
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])))
    if backend == "nccl":
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", str(master_port))
    torch.multiprocessing.spawn(
        _dist_worker,
        args=(WORLD, str(model_dir), scenario, backend),
        nprocs=WORLD,
        join=True,
    )


def _outcomes(model_dir):
    return {rank: json.loads((Path(model_dir) / f"worker_rank_{rank}.json").read_text()) for rank in range(WORLD)}


def _published_epochs(model_dir):
    return sorted(int(path.stem.split("_")[1]) for path in Path(model_dir).glob("epoch_*.pt"))


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _assert_scenario(model_dir, scenario):
    outcomes = _outcomes(model_dir)
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed: {outcome.get('error')}"
    if scenario == "full":
        assert [outcomes[rank]["batches"] for rank in range(WORLD)] == [12, 12]
        assert _published_epochs(model_dir) == [1, 2, 3]  # rank 0 published every epoch
        assert json.loads((Path(model_dir) / "latest.json").read_text())["epoch"] == 3
    elif scenario == "boundary":
        assert [outcomes[rank]["batches"] for rank in range(WORLD)] == [0, 0]
        assert _published_epochs(model_dir) == []
    elif scenario == "uniform":
        assert [outcomes[rank]["batches"] for rank in range(WORLD)] == [8, 8]  # sticky flag: every rank trains the epoch tail
        assert _published_epochs(model_dir) == [1]  # epoch 2 ran to the end but was never published
    else:  # staggered: only rank 1 ever observes the stop
        assert [outcomes[rank]["batches"] for rank in range(WORLD)] == [8, 8]  # the observing rank stays in the DDP batch stream
        assert _published_epochs(model_dir) == [1]


@pytest.mark.parametrize("scenario", ["full", "boundary", "uniform", "staggered"])
def test_early_stop_all_ranks_exit_cleanly_on_gloo(tmp_path, monkeypatch, scenario):
    _spawn_ranks(monkeypatch, tmp_path, scenario, "gloo")
    _assert_scenario(tmp_path, scenario)


@pytest.mark.gpu
@pytest.mark.parametrize("scenario", ["full", "boundary", "uniform", "staggered"])
def test_early_stop_all_ranks_exit_cleanly_on_nccl(tmp_path, monkeypatch, scenario):
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        pytest.skip("needs >= 2 CUDA devices")
    _spawn_ranks(monkeypatch, tmp_path, scenario, "nccl", master_port=_free_port())
    _assert_scenario(tmp_path, scenario)
