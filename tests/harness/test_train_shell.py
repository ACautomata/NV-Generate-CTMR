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

"""Behaviour gates for the phase training shell (ADR-0011, #111).

A fake ``PhaseTrainKernel`` drives ``PhaseHarness`` through the mechanical
sequence the shell owns (epoch loop, early-stop file polling at epoch
boundaries and mid-epoch, optimizer steps, atomic checkpoint publication +
latest.json, rank-0 gating) and the provenance writer's field sets are pinned
against the pre-#111 per-stage snapshots. Torch-level: runs on CPU in the CI
full-dependency tier, which installs torch (ADR-0015 §6).
"""

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from ctmr.harness.train_shell import STOP_FILE, PhaseHarness, TrainContext, TrainProvenanceWriter
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

    def build_loader(self):
        self.calls.append("build_loader")
        return self.batches

    def load_models(self, loader):
        self.calls.append(("load_models", len(loader)))
        self.device = torch.device(CPU)
        return TrainContext(
            trainable=torch.nn.Linear(1, 1, bias=False),
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


def _build_harness(kernel, tmp_path, n_epochs=2, local_rank=0, recipe_check=None, provenance=None, gradient_executor=None):
    return PhaseHarness(
        kernel=kernel,
        model_dir=tmp_path,
        n_epochs=n_epochs,
        amp=False,
        amp_dtype="bf16",
        local_rank=local_rank,
        logger=logging.getLogger("test-harness"),
        recipe_check=recipe_check,
        provenance=provenance,
        gradient_executor=gradient_executor,
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


def test_migrated_kernel_without_an_injected_executor_refuses_early(tmp_path):
    # The injection guard fires at construction (before any checkpoint loading
    # or first batch), never only when the cluster reaches epoch 1.
    with pytest.raises(ValueError, match="no gradient_executor was injected"):
        PhaseHarness(
            kernel=MigratedSpyKernel(n_batches=1),
            model_dir=tmp_path,
            n_epochs=1,
            amp=False,
            amp_dtype="bf16",
            local_rank=0,
            logger=logging.getLogger("test-harness"),
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
    # epoch 1 完整发布(4 batch);epoch 2 的 batch 0 即中断,无 epoch_2 发布
    assert (tmp_path / "epoch_1.pt").is_file()
    assert not (tmp_path / "epoch_2.pt").exists()
    assert len([c for c in kernel.calls if c[0] == "train_batch"]) == 5


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
