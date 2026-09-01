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

"""Cross-rank reduction gate for the embedded periodic validation (issue #278, ADR-0019 §5).

The sharded stage's equivalence contract: whatever the world size, the view
the scorer sees after the ``all_gather_object`` merge IS the single-card
full-cohort view -- the shards ``items[rank::world]`` are disjoint and the
merge concatenates them in rank order -- so the trend metric matches the
single-card reference within floating-point tolerance, summation order being
the only permitted difference.  The full ``PhaseHarness`` arm pins the two-rank
mechanics: rank 0 alone appends ``dev_trend.jsonl``/``trend.json``, every rank
keeps the in-memory trend for the boundary evaluation, and a fired rule ends
the run unanimously through the MAX consensus.

Two tiers mirror ADR-0015 §6 (and the #277 gate this file builds on):

- the torch-marked CPU tier (gloo backend, file:// init) runs for real in CI
  and locally;
- the gpu-marked tier (nccl backend, env:// init, one CUDA device per rank)
  is the issue's server-side gate (``pytest --run-gpu``), parameterized over
  world sizes 2 and 8 -- the 8-rank arm is the production topology.

Each spawned worker drives the real ``PeriodicValidator`` (and, in the harness
arm, the real ``PhaseHarness``) and drops a ``worker_rank_<N>.json`` outcome
file; a failure inside the run is recorded as an ``error`` key the parent
asserts absent -- the clean return of every worker IS the "all ranks exited"
assertion, and the process-group timeout bounds a regression hang.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ctmr.application.shell import STOP_FILE, EarlyStopRule, PeriodicValidator, PhaseHarness, TrainContext, ValidationPhase
from ctmr.infrastructure.checkpoints import CheckpointRepository
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

pytestmark = pytest.mark.torch

N_ITEMS = 16
FEATURE_DIM = 64
GROUP_TIMEOUT_SECONDS = 60


def _items():
    """The deterministic cohort: 16 items over 4 modalities (shard-able by any world size)."""
    modalities = ("t1n", "t1c", "t2w", "t2f")
    return [{"idx": index, "modality": modalities[index % 4]} for index in range(N_ITEMS)]


def _features_of(index):
    """Deterministic per-item features (no RNG: every rank and the reference agree bitwise)."""
    return np.linspace(0.0, 1.0, FEATURE_DIM, dtype=np.float64) * (index + 1)


class FeatureSampler:
    """The injected sampler seam: one entry per item carrying its plane-mean feature vector."""

    def __init__(self, device):
        self.device = device

    def __call__(self, ctx, shard_items, out_dir):
        return [{"idx": item["idx"], "modality": item["modality"], "features": _features_of(item["idx"])} for item in shard_items]


class SumScorer:
    """The injected scorer seam: one float over the gathered entries (summation-order sensitive)."""

    def __init__(self):
        self.merged = None

    def __call__(self, entries):
        self.merged = list(entries)
        total = float(np.sum([entry["features"] for entry in entries]))
        return {"m": total}, f"m={total}"


def _reference_m():
    """The single-card full-cohort trend metric: the scorer on the untouched original order."""
    fields, _log_line = SumScorer()([{"features": _features_of(item["idx"])} for item in _items()])
    return fields["m"]


class ValidatorKernel:
    """Tiny fake PhaseTrainKernel per rank: a DDP-wrapped Linear, as in the #277 gate.

    The DDP wrapper issues the per-batch gradient collective the production
    kernels issue, so the validation boundary runs against the real collective
    stream (the stage must not disturb it).
    """

    def __init__(self, device):
        self.device = device
        self.batches = 0
        self.trainable = None
        self.optimizer = None
        self.scheduler = None
        self.batch_list = [{"x": float(i)} for i in range(2)]

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
        x = torch.tensor([batch["x"]], device=self.device)
        return self.trainable(x).sum() ** 2

    def checkpoint_payload(self, epoch, avg_loss, scale):
        return {"epoch": epoch, "loss": avg_loss, "scale_factor": scale, "fake_state_dict": {}}


class EpochSampler:
    """Marks every entry with the validation round (one sampler call per boundary)."""

    def __init__(self):
        self.calls = 0
        self.counts = []
        self.dirs = []

    def __call__(self, ctx, shard_items, out_dir):
        self.calls += 1
        self.counts.append(len(shard_items))
        self.dirs.append(str(out_dir))
        return [{"epoch": self.calls, "idx": item["idx"]} for item in shard_items]


def _validator_worker(rank, world_size, model_dir, backend):
    """The validator-only arm: shard + all_gather + score against the single-card reference."""
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    # The torchrun local_rank convention: no per-rank CUDA_VISIBLE_DEVICES
    # writes (ignored on the HIP/DCU stack, where every rank then lands on
    # the same GPU and RCCL aborts with Duplicate GPU) -- each rank takes its
    # own index in the caller-masked visible set.
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    root = Path(model_dir)
    try:
        scorer = SumScorer()
        validator = PeriodicValidator(_items(), FeatureSampler(device), scorer, local_rank=rank, device=device, cohort_file="dev_cohort.json")
        fields, _log_line = validator.validate(ctx=None, epoch=2)
        outcome = {
            "rank": rank,
            "m": fields["m"],
            "merged_order": [entry["idx"] for entry in scorer.merged],
        }
        if rank == 0:
            outcome["reference_m"] = _reference_m()
    except Exception as error:
        (root / f"worker_rank_{rank}.json").write_text(json.dumps({"rank": rank, "error": f"{type(error).__name__}: {error}"}) + "\n")
        return
    (root / f"worker_rank_{rank}.json").write_text(json.dumps(outcome) + "\n")


def _harness_worker(rank, world_size, model_dir, backend):
    """The full-harness arm: two epochs, a validation boundary after each, patience-1 rule."""
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    # The torchrun local_rank convention: no per-rank CUDA_VISIBLE_DEVICES
    # writes (ignored on the HIP/DCU stack, where every rank then lands on
    # the same GPU and RCCL aborts with Duplicate GPU) -- each rank takes its
    # own index in the caller-masked visible set.
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    root = Path(model_dir)
    try:
        # The scripted trend worsens from the first boundary: patience 1 fires
        # the rule at epoch 2 and every rank must stop together.
        scripted_m = {1: 1.0, 2: 1.1}

        class ScriptedScorer:
            def __call__(self, entries):
                return {"m": scripted_m[entries[0]["epoch"]]}, f"m={scripted_m[entries[0]['epoch']]}"

        sampler = EpochSampler()
        validator = PeriodicValidator(_items(), sampler, ScriptedScorer(), local_rank=rank, device=device, cohort_file="dev_cohort.json")
        phase = ValidationPhase(every=1, validator=validator, rule=EarlyStopRule(patience=1, min_epoch=0, max_epoch=100))
        harness = PhaseHarness(
            kernel=ValidatorKernel(device),
            model_dir=root,
            n_epochs=4,
            local_rank=rank,
            logger=logging.getLogger(f"rank-{rank}"),
            gradient_executor=PlainGradientExecutor(),
            checkpoint_repository=CheckpointRepository(root),
            validation=phase,
        )
        harness.run()
        outcome = {
            "rank": rank,
            "shard_counts": sampler.counts,
            "sample_dirs": sampler.dirs,
            "trend_points": [(record["epoch"], record["m"]) for record in phase.records],
        }
    except Exception as error:
        (root / f"worker_rank_{rank}.json").write_text(json.dumps({"rank": rank, "error": f"{type(error).__name__}: {error}"}) + "\n")
        return
    (root / f"worker_rank_{rank}.json").write_text(json.dumps(outcome) + "\n")


def _harness_fail_worker(rank, world_size, model_dir, backend):
    """A rank-0-only scoring failure must skip the point on EVERY rank (MIN consensus)."""
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    # The torchrun local_rank convention: no per-rank CUDA_VISIBLE_DEVICES
    # writes (ignored on the HIP/DCU stack, where every rank then lands on
    # the same GPU and RCCL aborts with Duplicate GPU) -- each rank takes its
    # own index in the caller-masked visible set.
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    root = Path(model_dir)
    try:
        # The failure lands AFTER the all_gather (every rank has merged the
        # cohort), so every rank reaches the MIN consensus and skips together.
        class FlakyScorer:
            def __call__(self, entries):
                epoch = entries[0]["epoch"]
                if rank == 0 and epoch == 2:
                    raise RuntimeError("rank-0 scoring boom")
                return {"m": float(epoch)}, f"m={epoch}"

        validator = PeriodicValidator(_items(), EpochSampler(), FlakyScorer(), local_rank=rank, device=device, cohort_file="dev_cohort.json")
        phase = ValidationPhase(every=1, validator=validator, rule=EarlyStopRule(patience=5, min_epoch=0, max_epoch=100))
        harness = PhaseHarness(
            kernel=ValidatorKernel(device),
            model_dir=root,
            n_epochs=3,
            local_rank=rank,
            logger=logging.getLogger(f"rank-{rank}"),
            gradient_executor=PlainGradientExecutor(),
            checkpoint_repository=CheckpointRepository(root),
            validation=phase,
        )
        harness.run()
        outcome = {"rank": rank, "trend_points": [(record["epoch"], record["m"]) for record in phase.records]}
    except Exception as error:
        (root / f"worker_rank_{rank}.json").write_text(json.dumps({"rank": rank, "error": f"{type(error).__name__}: {error}"}) + "\n")
        return
    (root / f"worker_rank_{rank}.json").write_text(json.dumps(outcome) + "\n")


def _harness_sampler_fail_worker(rank, world_size, model_dir, backend):
    """A rank-0-only SAMPLER failure (BEFORE the gather) must skip on every rank, never hang."""
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{model_dir}/_init",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    # The torchrun local_rank convention: no per-rank CUDA_VISIBLE_DEVICES
    # writes (ignored on the HIP/DCU stack, where every rank then lands on
    # the same GPU and RCCL aborts with Duplicate GPU) -- each rank takes its
    # own index in the caller-masked visible set.
    device = torch.device("cuda" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    root = Path(model_dir)
    try:

        class FlakySampler:
            """Fails this rank's shard on the second boundary (epoch 2), before the gather."""

            def __init__(self):
                self.calls = 0

            def __call__(self, ctx, shard_items, out_dir):
                self.calls += 1
                if rank == 0 and self.calls == 2:
                    raise RuntimeError("rank-0 shard OOM")
                return [{"epoch": self.calls, "idx": item["idx"]} for item in shard_items]

        class EpochScorer:
            def __call__(self, entries):
                epoch = entries[0]["epoch"]
                return {"m": float(epoch)}, f"m={epoch}"

        validator = PeriodicValidator(_items(), FlakySampler(), EpochScorer(), local_rank=rank, device=device, cohort_file="dev_cohort.json")
        phase = ValidationPhase(every=1, validator=validator, rule=EarlyStopRule(patience=5, min_epoch=0, max_epoch=100))
        harness = PhaseHarness(
            kernel=ValidatorKernel(device),
            model_dir=root,
            n_epochs=3,
            local_rank=rank,
            logger=logging.getLogger(f"rank-{rank}"),
            gradient_executor=PlainGradientExecutor(),
            checkpoint_repository=CheckpointRepository(root),
            validation=phase,
        )
        harness.run()
        outcome = {"rank": rank, "trend_points": [(record["epoch"], record["m"]) for record in phase.records]}
    except Exception as error:
        (root / f"worker_rank_{rank}.json").write_text(json.dumps({"rank": rank, "error": f"{type(error).__name__}: {error}"}) + "\n")
        return
    (root / f"worker_rank_{rank}.json").write_text(json.dumps(outcome) + "\n")


def _spawn(monkeypatch, model_dir, worker, backend):
    # spawn children re-import this module, so its directory must resolve in a
    # fresh interpreter (pytest's in-memory sys.path does not propagate).
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])))
    torch.multiprocessing.spawn(worker, args=(2, str(model_dir), backend), nprocs=2, join=True)


def _outcomes(model_dir):
    return {rank: json.loads((Path(model_dir) / f"worker_rank_{rank}.json").read_text()) for rank in range(2)}


def test_gathered_view_matches_the_single_card_full_cohort_on_gloo(tmp_path, monkeypatch):
    _spawn(monkeypatch, tmp_path, _validator_worker, "gloo")
    outcomes = _outcomes(tmp_path)
    reference = outcomes[0]["reference_m"]
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed: {outcome.get('error')}"
        # the shard boundaries: 16 items over 2 ranks, interleaved then concatenated in rank order
        assert outcome["merged_order"] == [*range(0, N_ITEMS, 2), *range(1, N_ITEMS, 2)]
        assert sorted(outcome["merged_order"]) == list(range(N_ITEMS))  # disjoint, no loss, no duplication
        # the reduction equals the single-card full-cohort reference within floating-point tolerance
        assert math.isclose(outcome["m"], reference, rel_tol=1e-12), f"rank {rank}: {outcome['m']} vs {reference}"


def test_harness_validates_publishes_and_early_stops_on_two_ranks_on_gloo(tmp_path, monkeypatch):
    _spawn(monkeypatch, tmp_path, _harness_worker, "gloo")
    outcomes = _outcomes(tmp_path)
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed: {outcome.get('error')}"
        assert outcome["shard_counts"] == [8, 8]  # both boundaries, 16 items halved per rank
        # the samples land under the ledger root's per-epoch dir (the sidecar's layout)
        assert outcome["sample_dirs"] == [str(tmp_path / "dev_eval" / f"epoch_{epoch}" / "samples") for epoch in (1, 2)]
        assert outcome["trend_points"] == [[1, 1.0], [2, 1.1]]  # every rank's in-memory trend
    # rank 0 alone owns the on-disk ledger, and the fired rule wrote the stop file
    records = [json.loads(line) for line in (tmp_path / "dev_eval" / "dev_trend.jsonl").read_text().splitlines() if line.strip()]
    assert [(record["epoch"], record["m"]) for record in records] == [(1, 1.0), (2, 1.1)]
    assert json.loads((tmp_path / "dev_eval" / "epoch_2" / "trend.json").read_text()) == records[1]
    stop = json.loads((tmp_path / STOP_FILE).read_text())
    assert stop["epoch"] == 2
    assert "no new best" in stop["reason"]
    # epoch 3/4 never trained: the run broke at the epoch-2 validation boundary
    assert not (tmp_path / "epoch_3.pt").exists()
    assert not (tmp_path / "epoch_4.pt").exists()


def test_harness_skips_a_failed_validation_point_on_every_rank_on_gloo(tmp_path, monkeypatch):
    """A rank-0-only failure skips the point on EVERY rank (the MIN consensus):
    rank 1 scored it fine, yet neither rank keeps the trend point -- the rank-0
    ledger never leads the in-memory trend, and no stop verdict can fire on a
    record rank 0 did not append (the contract #278 keeps with the sidecar)."""
    _spawn(monkeypatch, tmp_path, _harness_fail_worker, "gloo")
    outcomes = _outcomes(tmp_path)
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed: {outcome.get('error')}"
        # epoch 2 unanimously skipped even though rank 1 scored it fine
        assert outcome["trend_points"] == [[1, 1.0], [3, 3.0]]
    records = [json.loads(line) for line in (tmp_path / "dev_eval" / "dev_trend.jsonl").read_text().splitlines() if line.strip()]
    assert [record["epoch"] for record in records] == [1, 3]  # the ledger matches the in-memory trend
    assert not (tmp_path / STOP_FILE).exists()
    for epoch in (1, 2, 3):  # training ran to completion: a failed stage never kills the run
        assert (tmp_path / f"epoch_{epoch}.pt").exists()


def test_harness_skips_a_shard_local_sampler_failure_without_hanging_on_gloo(tmp_path, monkeypatch):
    """codex P1 (PR #301): a shard-local sampling failure happens BEFORE the
    all_gather. The failed rank must not leave its peers stranded inside the
    gather -- the pre-gather MIN consensus makes the skip unanimous, so the run
    completes and no partial trend point lands (a hang would surface as a worker
    ``error`` after the process-group timeout)."""
    _spawn(monkeypatch, tmp_path, _harness_sampler_fail_worker, "gloo")
    outcomes = _outcomes(tmp_path)
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed/hung: {outcome.get('error')}"
        # epoch 2 unanimously skipped: rank 0's shard never reached the gather
        assert outcome["trend_points"] == [[1, 1.0], [3, 3.0]]
    records = [json.loads(line) for line in (tmp_path / "dev_eval" / "dev_trend.jsonl").read_text().splitlines() if line.strip()]
    assert [record["epoch"] for record in records] == [1, 3]  # no partial point from the failed epoch
    assert not (tmp_path / STOP_FILE).exists()
    for epoch in (1, 2, 3):  # training ran to completion despite the shard failure
        assert (tmp_path / f"epoch_{epoch}.pt").exists()


@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 8])
def test_gathered_view_matches_the_single_card_full_cohort_on_nccl(tmp_path, monkeypatch, world_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")])))
    # spawn children re-import this module: the worker is module-level
    # (picklable) with the topology passed as args (spawn precedent: no closures)
    torch.multiprocessing.spawn(_validator_worker, args=(world_size, str(tmp_path), "nccl"), nprocs=world_size, join=True)
    outcomes = {rank: json.loads((tmp_path / f"worker_rank_{rank}.json").read_text()) for rank in range(world_size)}
    reference = outcomes[0]["reference_m"]
    shard_size = N_ITEMS // world_size
    for rank, outcome in outcomes.items():
        assert "error" not in outcome, f"rank {rank} failed: {outcome.get('error')}"
        assert sorted(outcome["merged_order"]) == list(range(N_ITEMS))
        expected_order = [index for offset in range(world_size) for index in range(offset, N_ITEMS, world_size)]
        assert outcome["merged_order"] == expected_order
        assert math.isclose(outcome["m"], reference, rel_tol=1e-12), f"world {world_size} rank {rank}: {outcome['m']} vs {reference}"
    assert shard_size > 0
