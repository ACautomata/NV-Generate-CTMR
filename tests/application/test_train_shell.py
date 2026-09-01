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

"""Behaviour gates for the phase training shell (ADR-0011, #111; terminal form #276).

A fake ``PhaseTrainKernel`` drives ``PhaseHarness`` through the mechanical
sequence the shell owns (epoch loop, early-stop file polling at epoch
boundaries and mid-epoch, optimizer steps, atomic checkpoint publication +
latest.json, rank-0 gating) and the provenance writer's field sets are pinned
against the pre-#111 per-stage snapshots. Since #276 (ADR-0019 §1 terminal
state) the shell constructs nothing itself: the gradient executor and the
checkpoint repository ride in as injected collaborators (the tests inject the
real adapters -- tests are exempt from the layer rule). Since #278 the shell
also owns the embedded periodic validation stage (ADR-0019 §5): the N-epoch
boundary trigger, the eval/train swap, the ledger/trend.json append and the
early-stop boundary evaluation all stay mechanical, while the validation
domain rides in as the injected ``ValidationPhase`` (fake validator below).
Torch-level: runs on CPU in the CI full-dependency tier, which installs torch
(ADR-0015 §6).
"""

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from ctmr.application.shell import STOP_FILE, EarlyStopRule, PhaseHarness, TrainContext, TrainProvenanceWriter, ValidationPhase
from ctmr.infrastructure.checkpoints import CheckpointRepository
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor

# The pre-#111 provenance writer field sets, verbatim (do not edit -- the
# script/git_commit self-referential values are exempt, the key sets are not).
P1_PROVENANCE_KEYS = [
    "written_utc",
    "script",
    "env_config",
    "model_config",
    "model_def",
    "data_lists",
    "base_ckpt",
    "hyperparameters",
    "amp_dtype",
    "world_size",
    "torch_version",
    "git_commit",
]
P2_PROVENANCE_KEYS = [
    "written_utc",
    "script",
    "env_config",
    "model_config",
    "model_def",
    "data_list",
    "trained_diffusion_path",
    "replay",
    "hyperparameters",
    "amp_dtype",
    "world_size",
    "torch_version",
    "git_commit",
]
P3_PROVENANCE_KEYS = [
    "written_utc",
    "script",
    "env_config",
    "model_config",
    "model_def",
    "data_list",
    "trained_diffusion_path",
    "replay",
    "hyperparameters",
    "cfg_guidance_scale",
    "amp_dtype",
    "world_size",
    "torch_version",
    "git_commit",
]

CPU = "cpu"


class SpyKernel:
    """Fake PhaseTrainKernel: records the mechanical sequence the shell must drive."""

    def __init__(self, n_batches, on_train_batch=None):
        self.device = None
        self.calls = []
        self.on_train_batch = on_train_batch
        self.param = torch.nn.Parameter(torch.ones(()))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.batches = [{"x": float(i)} for i in range(n_batches)]
        self.trainable = None

    def build_loader(self):
        self.calls.append("build_loader")
        return self.batches

    def load_models(self, loader):
        self.calls.append(("load_models", len(loader)))
        self.device = torch.device(CPU)
        self.trainable = torch.nn.Linear(1, 1, bias=False)
        return TrainContext(
            trainable=self.trainable,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scale=torch.tensor(1.0),
            device=self.device,
        )

    def train_batch(self, batch):
        self.calls.append(("train_batch", batch["x"]))
        if self.on_train_batch is not None:
            self.on_train_batch(batch["x"])
        return self.param**2

    def checkpoint_payload(self, epoch, avg_loss, scale):
        self.calls.append(("payload", epoch, avg_loss))
        return {"epoch": epoch, "loss": avg_loss, "num_train_timesteps": 1000, "scale_factor": scale, "fake_state_dict": {}}


class ScriptedValidator:
    """Fake PeriodicValidator: records (epoch, training_mode) calls, scripted trend fields.

    ``fail_at`` epochs raise -- the embedded stage must degrade to a logged
    skip (training is the main job; a broken scorer must not kill the run).
    """

    def __init__(self, fields_by_epoch, fail_at=(), cohort_file="dev_cohort.json"):
        self.calls = []
        self._fields_by_epoch = fields_by_epoch
        self._fail_at = set(fail_at)
        self.cohort_file = cohort_file

    def validate(self, ctx, epoch, eval_root=None):
        self.calls.append((epoch, ctx.trainable.training))
        if epoch in self._fail_at:
            raise RuntimeError(f"validation boom at {epoch}")
        return dict(self._fields_by_epoch[epoch]), f"m={self._fields_by_epoch[epoch]['m']}"


def _build_harness(kernel, tmp_path, n_epochs=2, local_rank=0, recipe_check=None, provenance=None, gradient_executor=None, validation=None):
    return PhaseHarness(
        kernel=kernel,
        model_dir=tmp_path,
        n_epochs=n_epochs,
        local_rank=local_rank,
        logger=logging.getLogger("test-harness"),
        recipe_check=recipe_check,
        provenance=provenance,
        gradient_executor=gradient_executor if gradient_executor is not None else PlainGradientExecutor(),
        checkpoint_repository=CheckpointRepository(tmp_path),
        validation=validation,
    )


def test_epoch_loop_drives_the_mechanical_sequence_and_publishes_atomically(tmp_path):
    kernel = SpyKernel(n_batches=3)
    _build_harness(kernel, tmp_path, n_epochs=2).run()

    assert kernel.calls.count("build_loader") == 1
    assert ("load_models", 3) in kernel.calls
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 6  # 2 epochs x 3 batches
    assert len([c for c in kernel.calls if c[0] == "payload"]) == 2
    # checkpoint tmp 原子发布:只有发布后的 final,无 .tmp 残留
    for epoch in (1, 2):
        ckpt = tmp_path / f"epoch_{epoch}.pt"
        assert ckpt.is_file()
        assert not ckpt.with_name(ckpt.name + ".tmp").exists()
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest == {"epoch": 2, "checkpoint": str(tmp_path / "epoch_2.pt")}
    assert (tmp_path / "latest.json").read_text().endswith("\n")
    # optimizer 确实步进(参数从 1.0 减小)
    assert kernel.param.item() < 1.0


class MigratedSpyKernel(SpyKernel):
    """A kernel whose model entity drives the batch update (ADR-0016 train_step boundary).

    ``train_step`` owns the lr step inside the update (like the domain
    ``DiffusionModel.train_step``) -- the shell must not step ``ctx.scheduler``
    on top of it, or the PolynomialLR would advance twice per batch.
    """

    def train_step(self, batch, gradient_executor):
        self.calls.append(("train_step", batch["x"], type(gradient_executor).__name__))
        if self.on_train_batch is not None:
            self.on_train_batch(batch["x"])
        loss = gradient_executor.run(lambda: self.param**2, self.param, self.optimizer)
        self.scheduler.step()  # lr ownership stays in the kernel-side update
        return loss


def test_migrated_kernel_drives_batches_through_the_injected_executor(tmp_path):
    kernel = MigratedSpyKernel(n_batches=2)
    _build_harness(kernel, tmp_path, n_epochs=2, gradient_executor=PlainGradientExecutor()).run()

    assert len([c for c in kernel.calls if c[0] == "train_step"]) == 4  # every batch closed by the kernel
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 0  # shell no longer drives the update
    # the injected executor reached every kernel-side batch update
    assert all(step_call[2] == "PlainGradientExecutor" for step_call in kernel.calls if step_call[0] == "train_step")
    assert kernel.param.item() < 1.0  # the optimizer still stepped
    assert kernel.scheduler.last_epoch == 4  # lr stepped exactly once per batch (no shell double-step)
    assert (tmp_path / "epoch_2.pt").is_file()  # checkpoint publication unchanged


def test_missing_collaborators_are_refused_at_construction(tmp_path):
    # The injection guard fires at construction (before any checkpoint loading
    # or first batch), never only when the cluster reaches epoch 1. Both
    # collaborators are terminal-state mandatory (#276): the shell no longer
    # defaults the precision strategy or builds the weight store itself.
    with pytest.raises(ValueError, match="no gradient_executor was injected"):
        PhaseHarness(
            kernel=MigratedSpyKernel(n_batches=1),
            model_dir=tmp_path,
            n_epochs=1,
            local_rank=0,
            logger=logging.getLogger("test-harness"),
            checkpoint_repository=CheckpointRepository(tmp_path),
        )
    with pytest.raises(ValueError, match="no checkpoint_repository was injected"):
        PhaseHarness(
            kernel=MigratedSpyKernel(n_batches=1),
            model_dir=tmp_path,
            n_epochs=1,
            local_rank=0,
            logger=logging.getLogger("test-harness"),
            gradient_executor=PlainGradientExecutor(),
        )


def test_early_stop_file_halts_before_the_next_epoch(tmp_path):
    kernel = SpyKernel(n_batches=2)
    (tmp_path / STOP_FILE).touch()
    _build_harness(kernel, tmp_path, n_epochs=2).run()
    assert not (tmp_path / "epoch_1.pt").exists()
    assert [c for c in kernel.calls if c[0] == "train_batch"] == []


def test_early_stop_file_halts_mid_epoch_without_that_checkpoint(tmp_path):
    calls = []

    def _write_stop(midpoint):
        calls.append(midpoint)
        if len(calls) == 5:  # the first batch of epoch 2
            (tmp_path / STOP_FILE).touch()

    kernel = SpyKernel(n_batches=4, on_train_batch=_write_stop)
    _build_harness(kernel, tmp_path, n_epochs=2).run()
    # epoch 1 完整发布(4 batch);epoch 2 检测到早停后照跑至尾(跨 rank 集合流
    # 对齐要求检测 rank 不提前离开 batch 流,ADR-0019 §6),但不发布 epoch_2
    assert (tmp_path / "epoch_1.pt").is_file()
    assert not (tmp_path / "epoch_2.pt").exists()
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 8


def test_recipe_check_runs_first_on_rank0(tmp_path):
    order = []
    kernel = SpyKernel(n_batches=1)
    _build_harness(kernel, tmp_path, recipe_check=lambda: order.append("guard")).run()
    assert order[0] == "guard"


def test_non_rank0_skips_checkpoint_and_provenance(tmp_path):
    kernel = SpyKernel(n_batches=1)
    _build_harness(kernel, tmp_path, local_rank=1).run()
    assert not (tmp_path / "epoch_1.pt").exists()
    assert not (tmp_path / "latest.json").exists()


def _provenance_args(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    return SimpleNamespace(
        env_config_path=str(parent / "env.json"),
        model_config_path=str(parent / "train.json"),
        model_def_path=str(parent / "net.json"),
        amp_dtype="bf16",
    )


@pytest.mark.parametrize(
    "domain_fields,expected_keys",
    [
        (
            {
                "data_lists": {"brats_train": "lists/p1.json", "replay": ["lists/replay.json"]},
                "base_ckpt": "/base.pt",
                "hyperparameters": {"lr": 2e-06},
            },
            P1_PROVENANCE_KEYS,
        ),
        (
            {
                "data_list": "lists/p2.json",
                "trained_diffusion_path": "/dm.pt",
                "replay": None,
                "hyperparameters": {"lr": 1e-05},
            },
            P2_PROVENANCE_KEYS,
        ),
        (
            {
                "data_list": "lists/p3.json",
                "trained_diffusion_path": "/dm.pt",
                "replay": None,
                "hyperparameters": {"lr": 1e-05},
                "cfg_guidance_scale": 0.0,
            },
            P3_PROVENANCE_KEYS,
        ),
    ],
)
def test_provenance_field_set_matches_the_pre_consolidation_snapshot(tmp_path, domain_fields, expected_keys):
    writer = TrainProvenanceWriter(_provenance_args(tmp_path), 0, logging.getLogger("test-harness"), domain_fields=lambda: domain_fields)
    out = writer.write(tmp_path / "train_provenance.json")
    data = json.loads(out.read_text())
    assert list(data) == expected_keys  # key set AND order (sugon diff friendliness)
    # skeleton values are filled by the writer, domain values pass through verbatim
    for key, value in domain_fields.items():
        assert data[key] == value


def test_provenance_writer_is_rank0_only(tmp_path):
    writer = TrainProvenanceWriter(_provenance_args(tmp_path), 1, logging.getLogger("test-harness"), domain_fields=lambda: {})
    assert writer.write(tmp_path / "train_provenance.json") is None
    assert not (tmp_path / "train_provenance.json").exists()


def test_provenance_script_and_git_commit_are_self_referential(tmp_path):
    writer = TrainProvenanceWriter(_provenance_args(tmp_path), 0, logging.getLogger("test-harness"), domain_fields=lambda: {})
    data = json.loads(writer.write(tmp_path / "train_provenance.json").read_text())
    assert data["script"].endswith(("shell.py", "train_shell.py", "train.py"))
    assert data["git_commit"] is None or len(data["git_commit"]) == 40  # repo HEAD, or absent git


# --------------------------------------------------------------- embedded periodic validation (#278)

DEV_EVAL = "dev_eval"


def test_periodic_validation_runs_at_epoch_boundaries_and_leaves_training_math_untouched(tmp_path):
    """The stage fires only on the N-epoch boundary, after that epoch's train +
    publish, and the training math is byte-identical to a validation-free run."""
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({2: {"m": 1.0}, 4: {"m": 0.9}})
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=2, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=4, validation=phase).run()

    # both boundary calls saw the model in eval mode (the shell owns the swap)
    assert validator.calls == [(2, False), (4, False)]
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 8  # every batch still trained
    assert len([c for c in kernel.calls if c[0] == "payload"]) == 4  # every epoch still published

    plain = SpyKernel(n_batches=2)
    _build_harness(plain, tmp_path / "plain", n_epochs=4).run()
    assert kernel.param.item() == plain.param.item()  # zero training-math drift


def test_validation_swaps_the_model_back_to_train_mode_after_each_stage(tmp_path):
    modes = []
    kernel = SpyKernel(n_batches=2, on_train_batch=lambda x: modes.append(kernel.trainable.training))
    validator = ScriptedValidator({2: {"m": 1.0}})
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=5, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=2, validation=phase).run()

    assert modes == [True] * 4  # every training batch ran in train mode, validation never leaked


def test_validation_record_lands_in_the_ledger_with_the_existing_contract(tmp_path):
    """The record skeleton is the WatchEngine's: eval_utc/epoch/checkpoint open,
    the score fields sit between, cohort_file closes; trend.json mirrors the
    append; the in-memory trend equals the on-disk ledger."""
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({2: {"m": 0.8}})
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=5, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=2, validation=phase).run()

    eval_root = tmp_path / DEV_EVAL
    records = [json.loads(line) for line in (eval_root / "dev_trend.jsonl").read_text().splitlines() if line.strip()]
    assert [record["epoch"] for record in records] == [2]
    assert list(records[0]) == ["eval_utc", "epoch", "checkpoint", "m", "cohort_file"]
    assert records[0]["checkpoint"] == str(tmp_path / "epoch_2.pt")
    assert records[0]["cohort_file"] == "dev_cohort.json"
    assert json.loads((eval_root / "epoch_2" / "trend.json").read_text()) == records[0]
    assert [(record["epoch"], record["m"]) for record in phase.records] == [(record["epoch"], record["m"]) for record in records]


def test_validation_skips_non_boundary_epochs(tmp_path):
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({4: {"m": 0.9}})
    phase = ValidationPhase(every=4, validator=validator, rule=EarlyStopRule(patience=2, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=6, validation=phase).run()

    assert validator.calls == [(4, False)]  # epochs 2 and 6 stay unvalidated (6 is past the run end)


def test_validation_early_stop_writes_the_stop_file_and_halts_the_run(tmp_path):
    """The boundary evaluation is the pre-recorded rule on the accumulated
    trend; when it fires the stop file carries the same {"reason", "epoch"}
    contract the sidecar wrote, and the remaining epochs never train."""
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({2: {"m": 1.0}, 4: {"m": 1.1}})  # min rule: a worsening plateau
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=1, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=6, validation=phase).run()

    assert validator.calls == [(2, False), (4, False)]
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 8  # epochs 5-6 never ran
    stop_text = (tmp_path / STOP_FILE).read_text()
    stop = json.loads(stop_text)
    assert stop["epoch"] == 4
    assert "no new best" in stop["reason"]
    assert stop_text.endswith("\n")


def test_validation_respects_min_epoch_before_stopping(tmp_path):
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({2: {"m": 1.0}, 4: {"m": 1.1}, 6: {"m": 1.2}})
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=1, min_epoch=100, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=6, validation=phase).run()

    assert validator.calls == [(2, False), (4, False), (6, False)]  # the run completes despite the plateau
    assert not (tmp_path / STOP_FILE).exists()


def test_validation_failure_does_not_kill_the_run(tmp_path):
    """A broken scorer degrades to a logged skip: training is the main job, the
    train mode is restored, and no trend point lands."""
    kernel = SpyKernel(n_batches=2)
    validator = ScriptedValidator({2: {"m": 1.0}, 4: {"m": 0.9}}, fail_at={2})
    phase = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=2, min_epoch=0, max_epoch=100))

    _build_harness(kernel, tmp_path, n_epochs=4, validation=phase).run()

    assert validator.calls == [(2, False), (4, False)]
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 8
    records = [json.loads(line) for line in (tmp_path / DEV_EVAL / "dev_trend.jsonl").read_text().splitlines() if line.strip()]
    assert [record["epoch"] for record in records] == [4]  # only the healthy point landed
    assert not (tmp_path / STOP_FILE).exists()


def test_validation_phase_requires_both_collaborators(tmp_path):
    with pytest.raises(ValueError, match="validator and the early-stop rule"):
        _build_harness(SpyKernel(n_batches=1), tmp_path, validation=ValidationPhase(every=2, validator=None, rule=None))


class RngPollutingValidator:
    """Fake validator that resets and burns the global RNG, as the live sampler's
    per-sample ``torch.manual_seed`` does (codex review, PR #301)."""

    cohort_file = "dev_cohort.json"

    def __init__(self, fields_by_epoch):
        self._fields_by_epoch = fields_by_epoch

    def validate(self, ctx, epoch, eval_root=None):
        torch.manual_seed(12345)
        torch.randn(64)  # burn the stream, like the per-sample latent draws
        return dict(self._fields_by_epoch[epoch]), f"m={self._fields_by_epoch[epoch]['m']}"


def test_validation_isolates_its_rng_from_the_training_stream(tmp_path):
    """The stage's sampling randomness must not leak into training: the shell forks
    the RNG around the validation call, so enabling ``--val-every`` leaves the
    training random stream (shuffling, RF timesteps, modality perturbation)
    bit-identical to a validation-free run."""

    def training_draws(with_validation):
        torch.manual_seed(0)
        draws = []
        kernel = SpyKernel(n_batches=2, on_train_batch=lambda _x: draws.append(torch.rand(1).item()))
        validation = None
        if with_validation:
            validator = RngPollutingValidator({2: {"m": 1.0}, 4: {"m": 0.9}})
            validation = ValidationPhase(every=2, validator=validator, rule=EarlyStopRule(patience=5, min_epoch=0, max_epoch=100))
        _build_harness(kernel, tmp_path / ("with" if with_validation else "without"), n_epochs=4, validation=validation).run()
        return draws

    assert training_draws(True) == training_draws(False)  # the boundary's RNG pollution never reaches training
