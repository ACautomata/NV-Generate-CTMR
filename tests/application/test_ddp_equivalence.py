# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numerical equivalence gates for the DDP training path (issue #281, ADR-0019 §4/§9 B2).

The two gates of the four-category gpu suite the #277 (early-stop timing) and
#278 (periodic validation all_gather) files do not carry:

- **single-card vs multi-card loss tolerance**: under the per-GPU batch=1 /
  global batch = world_size contract (ADR-0019 §4), DDP's gradient averaging
  makes every global step exactly the single-card update on the W-item batch
  under a mean-loss reduction -- the mean of per-item gradients IS the
  gradient of the mean loss.  So the same deterministic item stream trains to
  the same trajectory single-card (one process, W-item batches, no DDP, no
  process group) and multi-card (W ranks, one item per rank per step,
  DDP-wrapped), up to floating-point summation order: every per-step loss,
  the shell's all_reduced epoch average and the published weights must agree
  within tolerance.
- **DDP checkpoint round-trip**: a checkpoint a multi-card run publishes must
  be lossless.  A run resumed from it (payload ``state_dict`` loaded into the
  module before the DDP wrapper wraps it, the production resume order) rejoins
  the uninterrupted trajectory: resumed and uninterrupted epoch-3 payloads
  carry the same average loss and the same weights, through the real
  ``CheckpointRepository`` tmp atomic publish + ``latest.json`` protocol
  (ADR-0015 §4).

Two tiers mirror ADR-0015 §6 (and the #277/#278 gates this file completes):

- the torch-marked CPU tier (gloo backend, file:// init) runs for real in CI
  and locally;
- the gpu-marked tier (nccl backend, env:// init, one CUDA device per rank)
  is the issue's server-side gate (``pytest --run-gpu``), parameterized over
  world sizes 2 and 8 -- the 8-rank arm is the production topology.

Everything is deterministic (constant initial weights, a fixed item stream, no
RNG, no shuffling): a divergence between the two sides is a real reduction or
publication defect, never sampling noise.  Each spawned worker drives the real
``PhaseHarness`` and drops a ``worker_rank_<N>.json`` outcome file; a failure
inside the run is recorded as an ``error`` key the parent asserts absent, and
the process-group timeout bounds a regression hang.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ctmr.application.shell import PhaseHarness, TrainContext
from ctmr.infrastructure.checkpoints import CheckpointRepository
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

pytestmark = pytest.mark.torch

N_ITEMS = 16
GROUP_TIMEOUT_SECONDS = 60
REL_TOL = 1e-4
ABS_TOL = 1e-7


def _items():
    """The deterministic item stream: 16 scalar x values (no RNG anywhere)."""
    return [0.1 * (index + 1) for index in range(N_ITEMS)]


def _item_loss(module, x, device):
    """The deterministic per-item loss ``(w * x - x)^2`` of the scalar Linear."""
    inp = torch.tensor([x], device=device)
    return (module(inp) - inp).pow(2).sum()


class GlobalBatchKernel:
    """The single-card reference kernel (the semantics the DDP path must reproduce).

    Each batch IS one global batch: ``items_per_step`` items consumed in a
    single mean-loss forward/backward, the trainable NOT DDP-wrapped.  The
    batch loss ``mean(f)`` has the mean of the per-item gradients as its
    gradient -- exactly what DDP's gradient averaging computes on the W ranks
    that share the global step; that identity is what the gate pins.
    """

    def __init__(self, items, items_per_step, device):
        self.device = device
        self.step_losses = []
        self.trainable = None
        self.optimizer = None
        self.scheduler = None
        self.batch_list = [items[k : k + items_per_step] for k in range(0, len(items), items_per_step)]

    def build_loader(self):
        return self.batch_list

    def load_models(self, loader):
        module = torch.nn.Linear(1, 1, bias=False).to(self.device)
        with torch.no_grad():
            module.weight.fill_(0.5)
        self.trainable = module
        self.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=10**9)
        return TrainContext(
            trainable=self.trainable,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scale=torch.tensor(1.0, device=self.device),
            device=self.device,
        )

    def train_batch(self, batch):
        loss = torch.stack([_item_loss(self.trainable, x, self.device) for x in batch]).mean()
        self.step_losses.append(loss.item())
        return loss

    def checkpoint_payload(self, epoch, avg_loss, scale):
        return {
            "epoch": epoch,
            "loss": avg_loss,
            "scale_factor": scale,
            "state_dict": {key: value.detach().cpu() for key, value in self.trainable.state_dict().items()},
        }


class ShardKernel:
    """The multi-card per-rank kernel: per-GPU batch=1, DDP-wrapped trainable.

    Rank ``rank``'s loader is its interleaved shard ``items[rank::world]`` (the
    sharding convention the periodic validator uses), one item per batch --
    so global step k of the W-rank run consumes the same items the single-card
    reference's k-th W-item batch does.  For the round-trip arm,
    ``resume_from`` names a published payload whose ``state_dict`` is loaded
    into the module BEFORE the DDP wrapper wraps it (every rank loads the same
    file, so the wrapper's initial broadcast is a no-op).
    """

    def __init__(self, rank, world, items, device, resume_from=None):
        self.device = device
        self.step_losses = []
        self.trainable = None
        self.optimizer = None
        self.scheduler = None
        self.resume_from = resume_from
        self.batch_list = [[x] for x in items[rank::world]]

    def build_loader(self):
        return self.batch_list

    def load_models(self, loader):
        module = torch.nn.Linear(1, 1, bias=False).to(self.device)
        with torch.no_grad():
            module.weight.fill_(0.5)
        if self.resume_from is not None:
            module.load_state_dict(torch.load(self.resume_from, weights_only=True)["state_dict"])
        self.trainable = DistributedDataParallel(module)
        self.optimizer = torch.optim.SGD(self.trainable.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=10**9)
        return TrainContext(
            trainable=self.trainable,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scale=torch.tensor(1.0, device=self.device),
            device=self.device,
        )

    def train_batch(self, batch):
        loss = _item_loss(self.trainable, batch[0], self.device)
        self.step_losses.append(loss.item())
        return loss

    def unwrapped(self):
        """The trainable with the DDP wrapper stripped (the publication and outcome face)."""
        return self.trainable.module if isinstance(self.trainable, DistributedDataParallel) else self.trainable

    def checkpoint_payload(self, epoch, avg_loss, scale):
        return {
            "epoch": epoch,
            "loss": avg_loss,
            "scale_factor": scale,
            "state_dict": {key: value.detach().cpu() for key, value in self.unwrapped().state_dict().items()},
        }


def _harness(kernel, model_dir, rank, n_epochs):
    """The real shell around a fake kernel -- the one mechanical path both tiers drive."""
    return PhaseHarness(
        kernel=kernel,
        model_dir=Path(model_dir),
        n_epochs=n_epochs,
        local_rank=rank,
        logger=logging.getLogger(f"rank-{rank}"),
        gradient_executor=PlainGradientExecutor(),
        checkpoint_repository=CheckpointRepository(model_dir),
    )


def _run_single_card_reference(model_dir, world_size):
    """The single-card reference arm: one process, no DDP, no process group.

    ``world_size`` fixes the global batch width (``items_per_step``), so the
    reference performs exactly the updates the W-rank run's gradient averaging
    computes, over the same item stream.  Runs on CPU even on the gpu tier:
    the gate is stronger for it -- the DDP trajectory must track an
    independent single-card computation, not just itself.
    """
    kernel = GlobalBatchKernel(_items(), world_size, torch.device("cpu"))
    _harness(kernel, model_dir, 0, 1).run()
    return kernel


def _init_worker_group(rank, world_size, model_dir, backend):
    """The shared per-worker setup: process group + device (the #277/#278 shape).

    Device selection follows the torchrun local_rank convention: no per-rank
    CUDA_VISIBLE_DEVICES writes (on the HIP/DCU stack they are ignored, every
    rank then lands on the same GPU and RCCL aborts with Duplicate GPU);
    instead each rank takes its own index in the caller-masked visible set."""
    dist.init_process_group(
        backend=backend,
        # FileStore rendezvous for BOTH backends: a fixed TCP port has no safe
        # range on a shared server (torchrun's own default is 29500), while a
        # per-model_dir file is collision-free by construction and NCCL then
        # binds its own kernel-assigned ephemeral ports (ADR-0019 §6 gates).
        init_method=f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    return device


def _write_outcome(model_dir, rank, outcome):
    (Path(model_dir) / f"worker_rank_{rank}.json").write_text(json.dumps(outcome) + "\n")


def _equivalence_worker(rank, world_size, model_dir, backend):
    """One W-rank arm of the loss-tolerance gate: 1 epoch over this rank's shard."""
    try:
        device = _init_worker_group(rank, world_size, model_dir, backend)
        kernel = ShardKernel(rank, world_size, _items(), device)
        _harness(kernel, model_dir, rank, 1).run()
        outcome = {
            "rank": rank,
            "step_losses": kernel.step_losses,
            "final_weight": float(kernel.unwrapped().weight.item()),
        }
    except Exception as error:
        _write_outcome(model_dir, rank, {"rank": rank, "error": f"{type(error).__name__}: {error}"})
        return
    _write_outcome(model_dir, rank, outcome)


def _roundtrip_worker(rank, world_size, model_dir, backend, n_epochs, resume_from=None):
    """One DDP arm of the round-trip gate: n_epochs over this rank's shard,
    optionally resumed from a published payload."""
    try:
        device = _init_worker_group(rank, world_size, model_dir, backend)
        kernel = ShardKernel(rank, world_size, _items(), device, resume_from=resume_from)
        _harness(kernel, model_dir, rank, n_epochs).run()
        outcome = {"rank": rank, "n_epochs": n_epochs}
    except Exception as error:
        _write_outcome(model_dir, rank, {"rank": rank, "error": f"{type(error).__name__}: {error}"})
        return
    _write_outcome(model_dir, rank, outcome)


def _spawn(monkeypatch, worker, model_dir, backend, world_size, *extra):
    """torch.multiprocessing.spawn with the re-import fix; the workers see
    ``(rank, world_size, model_dir, backend, *extra)`` and rendezvous through
    the per-model_dir FileStore (no fixed TCP port anywhere)."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])))
    torch.multiprocessing.spawn(worker, args=(world_size, str(model_dir), backend, *extra), nprocs=world_size, join=True)


def _outcomes(model_dir, world_size):
    return {rank: json.loads((Path(model_dir) / f"worker_rank_{rank}.json").read_text()) for rank in range(world_size)}


def _assert_no_worker_errors(outcomes, run):
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"{run} rank {rank} failed: {outcome.get('error')}"


def _assert_same_weights(state_a, state_b, context):
    assert set(state_a) == set(state_b), context
    for key in state_a:
        assert torch.allclose(state_a[key], state_b[key], rtol=REL_TOL, atol=ABS_TOL), f"{context}: {key}"


def _assert_loss_equivalence(reference, distributed_root, reference_root, outcomes, world_size):
    """The single-card reference and the W-rank run agree within float tolerance:
    every global step's losses, the shell's reduced epoch average, the published
    weights -- and the ranks stay mutually synchronized."""
    _assert_no_worker_errors(outcomes, "distributed")
    steps = len(reference.step_losses)
    for outcome in outcomes.values():
        assert len(outcome["step_losses"]) == steps  # per-GPU batch=1: equal shards, equal steps
    for step in range(steps):
        shard_mean = sum(outcome["step_losses"][step] for outcome in outcomes.values()) / world_size
        assert math.isclose(reference.step_losses[step], shard_mean, rel_tol=REL_TOL, abs_tol=ABS_TOL), f"step {step}"
    reference_payload = CheckpointRepository(reference_root).load(Path(reference_root) / "epoch_1.pt")
    distributed_payload = CheckpointRepository(distributed_root).load(Path(distributed_root) / "epoch_1.pt")
    assert math.isclose(reference_payload["loss"], distributed_payload["loss"], rel_tol=REL_TOL, abs_tol=ABS_TOL)
    _assert_same_weights(reference_payload["state_dict"], distributed_payload["state_dict"], "published weights")
    weights = [outcome["final_weight"] for outcome in outcomes.values()]
    assert all(math.isclose(weight, weights[0], rel_tol=REL_TOL, abs_tol=ABS_TOL) for weight in weights)


def _assert_roundtrip(tmp_path, world_size):
    """The published payload is lossless: an independent 2-epoch run and the
    uninterrupted 3-epoch run agree on the epoch-2 state, and the run resumed
    from that state rejoins the uninterrupted trajectory at epoch 3."""
    for run in ("run_a", "run_b", "run_c"):
        _assert_no_worker_errors(_outcomes(tmp_path / run, world_size), run)
    run_a, run_b, run_c = (tmp_path / run for run in ("run_a", "run_b", "run_c"))
    epoch_2_a = CheckpointRepository(run_a).load(run_a / "epoch_2.pt")
    epoch_2_c = CheckpointRepository(run_c).load(run_c / "epoch_2.pt")
    assert math.isclose(epoch_2_a["loss"], epoch_2_c["loss"], rel_tol=REL_TOL, abs_tol=ABS_TOL)
    _assert_same_weights(epoch_2_a["state_dict"], epoch_2_c["state_dict"], "independent epoch-2 states")
    # run_b resumed from run_a's epoch_2 and trained one epoch: the shell
    # numbers its publications from its own epoch counter, so the resumed
    # epoch-3 STATE lands in run_b/epoch_1.pt -- compare content, not name.
    resumed = CheckpointRepository(run_b).load(run_b / "epoch_1.pt")
    epoch_3_c = CheckpointRepository(run_c).load(run_c / "epoch_3.pt")
    assert math.isclose(resumed["loss"], epoch_3_c["loss"], rel_tol=REL_TOL, abs_tol=ABS_TOL)
    _assert_same_weights(resumed["state_dict"], epoch_3_c["state_dict"], "resumed vs uninterrupted epoch 3")
    # the pointer protocol directs at each run's newest published epoch
    assert json.loads((run_c / "latest.json").read_text())["epoch"] == 3
    assert json.loads((run_b / "latest.json").read_text())["epoch"] == 1


def _run_equivalence_scenario(tmp_path, monkeypatch, backend, world_size):
    """The loss-tolerance scenario: single-card reference vs the W-rank run,
    then the tolerance assertions."""
    reference = _run_single_card_reference(tmp_path / "reference", world_size)
    _spawn(monkeypatch, _equivalence_worker, tmp_path, backend, world_size)
    _assert_loss_equivalence(reference, tmp_path, tmp_path / "reference", _outcomes(tmp_path, world_size), world_size)


def _run_roundtrip_scenario(tmp_path, monkeypatch, backend, world_size):
    """The round-trip scenario over three W-rank runs: run_a trains 2 epochs,
    run_c trains 3 from scratch (the uninterrupted reference), run_b resumes
    from run_a's epoch-2 payload and trains one epoch; then the assertions."""
    run_a, run_b, run_c = (tmp_path / run for run in ("run_a", "run_b", "run_c"))
    for path in (run_a, run_b, run_c):
        path.mkdir()
    _spawn(monkeypatch, _roundtrip_worker, run_a, backend, world_size, 2)
    _spawn(monkeypatch, _roundtrip_worker, run_c, backend, world_size, 3)
    _spawn(monkeypatch, _roundtrip_worker, run_b, backend, world_size, 1, run_a / "epoch_2.pt")
    _assert_roundtrip(tmp_path, world_size)


def test_multicard_loss_trajectory_matches_the_single_card_global_batch_on_gloo(tmp_path, monkeypatch):
    _run_equivalence_scenario(tmp_path, monkeypatch, "gloo", world_size=2)


def test_checkpoint_roundtrip_rejoins_the_uninterrupted_trajectory_on_gloo(tmp_path, monkeypatch):
    _run_roundtrip_scenario(tmp_path, monkeypatch, "gloo", world_size=2)


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 8])
def test_multicard_loss_trajectory_matches_the_single_card_global_batch_on_nccl(tmp_path, monkeypatch, world_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices")
    _run_equivalence_scenario(tmp_path, monkeypatch, "nccl", world_size)


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 8])
def test_checkpoint_roundtrip_rejoins_the_uninterrupted_trajectory_on_nccl(tmp_path, monkeypatch, world_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices")
    _run_roundtrip_scenario(tmp_path, monkeypatch, "nccl", world_size)
