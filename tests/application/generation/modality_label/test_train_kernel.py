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

"""Full-parameter continuation kernel of the modality_label family on CPU (ticket 10).

Acceptance criterion: the full-param continuation load, the 1:1 replay-list
mix, and the scale_factor reuse convention must execute for real on a
synthetic mini fixture -- the CI full-dependency tier (torch-marked) runs
them, never skipped around the torch mark. The network is the production
MAISI ``DiffusionModelUNetMaisi`` at toy width; the scheduler is the pinned
production ``RFlowScheduler`` shape; the base checkpoint is a real
``torch.save`` payload in the upstream key layout.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from ctmr.application.generation.modality_label.train import (
    SCALE_FACTOR_RELATIVE_TOLERANCE,
    DataCatalog,
    ScaleFactorPolicy,
    TrainKernel,
)
from ctmr.infrastructure.engine import MaisiEngine
from ctmr.infrastructure.gradient_executors import PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance
from ctmr.wiring.generate import MonaiCheckpointArchive

pytestmark = pytest.mark.torch

CPU = torch.device("cpu")
CKPT_SCALE_FACTOR = 0.87
MODALITY_MAPPING = {
    "mri_t1_skull_stripped": 29,
    "mri_t2_skull_stripped": 30,
    "mri_flair_skull_stripped": 31,
    "mri_t1c_skull_stripped": 34,
}


def _toy_unet_def():
    """The production config_network_rflow.json topology at toy width."""
    return {
        "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 4,
        "num_channels": [32, 64],
        "attention_levels": [False, False],
        "num_head_channels": [0, 32],
        "num_res_blocks": 1,
        "use_flash_attention": False,
        "include_top_region_index_input": False,
        "include_bottom_region_index_input": False,
        "include_spacing_input": True,
        "num_class_embeds": 40,
        "resblock_updown": True,
        "include_fc": True,
    }


def _write_embedding(path, spacing=(1.0, 1.2, 0.8), modality="mri_t1_skull_stripped"):
    """One synthetic training latent plus its companion json (the phase encode layout)."""
    latent = np.random.randn(4, 8, 8, 4).astype(np.float32)  # std ~ 1 -> recomputed 1/std(z) lands near the ckpt value
    nib.save(nib.Nifti1Image(latent, np.diag([1.0, 1.0, 1.0, 1.0])), str(path))
    info = str(path) + ".json"
    Path(info).write_text(json.dumps({"spacing": list(spacing), "modality": modality}))
    return info


def _list_file(path, entries):
    Path(path).write_text(json.dumps({"training": entries}))
    return str(path)


def _fixture(tmp_path, brats_entries=1, replay_entries=1):
    """The synthetic mini run root: embeddings, both lists, mapping, base checkpoint."""
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    _write_embedding(emb_dir / "caseA_emb.nii.gz", spacing=(1.0, 1.2, 0.8))
    _write_embedding(emb_dir / "caseB_emb.nii.gz", spacing=(0.9, 0.9, 1.1), modality="mri_t2_skull_stripped")
    names = ["caseA", "caseB"]

    def _entry(index):
        return {"image": f"{names[index % 2]}.nii.gz", "sub": "GLI", "case": f"SYNTH-{index:04d}"}

    brats = _list_file(tmp_path / "p1_image_only.json", [_entry(i) for i in range(brats_entries)])
    replay = _list_file(tmp_path / "p1_mrrate_replay.json", [_entry(i) for i in range(replay_entries)])
    mapping_path = tmp_path / "modality_mapping.json"
    mapping_path.write_text(json.dumps(MODALITY_MAPPING))

    args = SimpleNamespace(
        embedding_base_dir=str(emb_dir),
        json_data_list=brats,
        replay_list=[replay],
        diffusion_unet_def=_toy_unet_def(),
        noise_scheduler={
            "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": False,
            "use_timestep_transform": True,
            "sample_method": "uniform",
            "scale": 1.4,
        },
        modality_mapping_path=str(mapping_path),
        modality_mapping=dict(MODALITY_MAPPING),
        diffusion_unet_train={"batch_size": 1, "cache_rate": 0.0, "lr": 2e-06, "n_epochs": 2},
    )

    unet = define_instance(args, "diffusion_unet_def")
    ckpt_path = tmp_path / "base_ckpt.pt"
    torch.save({"unet_state_dict": unet.state_dict(), "scale_factor": CKPT_SCALE_FACTOR}, str(ckpt_path))
    args.existing_ckpt_filepath = str(ckpt_path)
    return args


def _kernel(args):
    # the ports the composition root injects in production (issue #272): the
    # engine adapter and the base-checkpoint archive, driven here for real
    return TrainKernel(
        args,
        device=CPU,
        logger=logging.getLogger("test-kernel"),
        local_rank=0,
        engine=MaisiEngine(),
        base_checkpoints=MonaiCheckpointArchive(CPU),
    )


def _loaded_kernel(tmp_path):
    kernel = _kernel(_fixture(tmp_path))
    loader = kernel.build_loader()
    ctx = kernel.load_models(loader)
    return kernel, loader, ctx


def test_build_loader_mixes_the_brats_and_replay_lists(tmp_path):
    kernel = _kernel(_fixture(tmp_path, brats_entries=2, replay_entries=2))
    loader = kernel.build_loader()
    assert len(loader.dataset) == 4  # 2 brats + 2 replay (the 1:1 mix)


def test_build_loader_guards_the_1_to_1_replay_mix(tmp_path):
    args = _fixture(tmp_path, brats_entries=2, replay_entries=1)
    with pytest.raises(ValueError, match="1:1 replay mix"):
        DataCatalog(args, logging.getLogger("test-kernel")).load_entries()


def test_missing_training_embedding_fails_loudly(tmp_path):
    args = _fixture(tmp_path)
    os.remove(os.path.join(args.embedding_base_dir, "caseA_emb.nii.gz"))
    with pytest.raises(FileNotFoundError, match="training embedding missing"):
        DataCatalog(args, logging.getLogger("test-kernel")).file_records()


def test_load_models_reuses_the_checkpoint_scale_factor(tmp_path):
    kernel, loader, ctx = _loaded_kernel(tmp_path)

    # the convention under test: scale_factor comes from the base checkpoint,
    # never recomputed (issue #10 §7) -- and the full-param load is real.
    assert float(ctx.scale) == pytest.approx(CKPT_SCALE_FACTOR)
    assert float(kernel._model.scale_factor) == pytest.approx(CKPT_SCALE_FACTOR)
    assert isinstance(ctx.trainable, torch.nn.Module)
    assert all(p.requires_grad for p in ctx.trainable.parameters())
    assert ctx.optimizer.param_groups[0]["lr"] == 2e-06
    assert isinstance(ctx.scheduler, torch.optim.lr_scheduler.LRScheduler)


def test_scale_factor_policy_sanity_asserts_only_against_large_drift():
    logger = logging.getLogger("test-scale")
    policy = ScaleFactorPolicy(CKPT_SCALE_FACTOR, logger)
    assert policy.value() == pytest.approx(CKPT_SCALE_FACTOR)
    policy.sanity_check(torch.tensor(1.0), CPU)  # relative ~0.15 < tolerance: sanity only, value stays reused
    with pytest.raises(ValueError, match="scale_factor sanity assert failed"):
        policy.sanity_check(torch.tensor(3.0), CPU)
    assert SCALE_FACTOR_RELATIVE_TOLERANCE == 0.5


def test_full_continuation_load_rejects_an_incompatible_checkpoint(tmp_path):
    args = _fixture(tmp_path)
    torch.save({"unet_state_dict": {}, "scale_factor": 1.0}, args.existing_ckpt_filepath)
    kernel = _kernel(args)
    loader = kernel.build_loader()
    with pytest.raises(ValueError, match="missing keys for full-parameter continuation"):
        kernel.load_models(loader)


def test_train_batch_executes_a_closed_training_step_with_the_production_rflow_shape(tmp_path):
    kernel, _loader, ctx = _loaded_kernel(tmp_path)
    batch = {
        "image": torch.randn(1, 4, 8, 8, 4),
        "spacing": torch.ones(1, 3),
        "modality": torch.tensor([29]),
    }
    loss = kernel.train_step(batch, PlainGradientExecutor())

    assert loss.dim() == 0  # a scalar loss
    assert torch.isfinite(loss)
    # the full-param training closure: gradients reach the trainable DM and the
    # closed update already applied one optimizer step (ADR-0016 train_step)
    grads = [p.grad for p in ctx.trainable.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert ctx.optimizer.param_groups[0]["lr"] < 2e-06  # PolynomialLR stepped already


def test_rflow_sampling_step_closes_on_cpu():
    """The rectified-flow inference step the dev-sidecar sampler drives (the
    pinned production scheduler shape), closed on CPU."""
    from monai.networks.schedulers import RFlowScheduler

    scheduler = RFlowScheduler(
        num_train_timesteps=1000, use_discrete_timesteps=False, use_timestep_transform=True, sample_method="uniform", scale=1.4
    )
    scheduler.set_timesteps(num_inference_steps=3, input_img_size_numel=int(np.prod((8, 8, 4))))
    sample = torch.randn(1, 4, 8, 8, 4)
    model_output = torch.randn(1, 4, 8, 8, 4)  # the predicted velocity (images - noise)
    prev = scheduler.step(model_output=model_output, timestep=scheduler.timesteps[0], sample=sample, next_timestep=scheduler.timesteps[1])
    prev_sample = prev[0] if isinstance(prev, tuple) else prev.prev_sample
    assert prev_sample.shape == sample.shape
    assert torch.isfinite(prev_sample).all()


def test_checkpoint_payload_keeps_the_upstream_key_layout(tmp_path):
    kernel, _loader, ctx = _loaded_kernel(tmp_path)
    payload = kernel.checkpoint_payload(epoch=3, avg_loss=0.25, scale=ctx.scale)

    assert list(payload) == ["epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict"]
    assert payload["epoch"] == 3
    assert payload["loss"] == 0.25
    assert payload["num_train_timesteps"] == 1000
    assert float(payload["scale_factor"]) == pytest.approx(CKPT_SCALE_FACTOR)
    assert set(payload["unet_state_dict"]) == set(ctx.trainable.state_dict())
