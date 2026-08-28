"""Execution smoke for the vendored maiisi_engine (issue #134; standalone as of #143).

Same synthetic config/argv in, expected observable out — CPU-only real execution
(torch-marked; the CI torch job runs this on plain CPU per ADR-0015 §6, which
requires torch to be installed, not skipped around).

Before #143 these gates ran as an equivalence check against the legacy
upstream originals. That package is retired (ADR-0015 M5: git history is the
reproduction anchor), so the freeze-vs-upstream guarantee no longer has a live
reference — these tests now pin the vendored engine's own observable behavior
directly. GPU-bound paths (``initialize_distributed``'s CUDA branch,
``run_torchrun``, DDP gathering) are intentionally NOT executed here.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import DDPMScheduler

import ctmr.infrastructure.maiisi_engine  # noqa: F401  (import = new-home resolution smoke)
from ctmr.infrastructure.maiisi_engine import diff_model_infer as engine_infer
from ctmr.infrastructure.maiisi_engine import diff_model_setting as engine_setting
from ctmr.infrastructure.maiisi_engine import diff_model_train as engine_train
from ctmr.infrastructure.maiisi_engine.create_training_data import create_transforms as engine_create_transforms
from ctmr.infrastructure.maiisi_engine.create_training_data import round_number as engine_round_number
from ctmr.infrastructure.maiisi_engine.diff_model_train import load_filenames as engine_load_filenames
from ctmr.infrastructure.maiisi_engine.inference_primitives import check_input_ct, check_input_mr, dynamic_infer, get_body_region_index_from_mask
from ctmr.infrastructure.maiisi_engine.instance_definition import define_instance
from ctmr.infrastructure.maiisi_engine.utils_infer import initialize_noise_latents

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[3]

VENDORED_ENTRY_MODULES = [
    "ctmr.infrastructure.maiisi_engine.diff_model_train",
    "ctmr.infrastructure.maiisi_engine.diff_model_infer",
    "ctmr.infrastructure.maiisi_engine.create_training_data",
]


@pytest.fixture()
def synthetic_configs(tmp_path):
    """Three-file MAISI-style config bundle; overlapping key proves merge order."""
    env = {"output_dir": str(tmp_path / "out"), "output_prefix": "smoke", "env_only": True}
    model_cfg = {
        "diffusion_unet_inference": {
            "dim": [256, 256, 128],
            "spacing": [1.5, 1.5, 2.0],
            "top_region_index": [0.02, 0.05, 0.93],
            "bottom_region_index": [0.91, 0.04, 0.05],
            "modality": 9,
            "num_inference_steps": 2,
            "cfg_guidance_scale": 0.0,
        },
        "noise_scheduler": {"_target_": "monai.networks.schedulers.DDPMScheduler", "num_train_timesteps": 5},
        "overlap_key": "model",
        "latent_channels": 4,
    }
    model_def = {"diffusion_unet_def": {"_target_": "monai.networks.nets.DiffusionModelUNet"}, "overlap_key": "def"}
    paths = []
    for name, payload in [("env.json", env), ("model.json", model_cfg), ("def.json", model_def)]:
        p = tmp_path / name
        p.write_text(json.dumps(payload))
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------- config layer


def test_load_config_merges_env_model_def_in_order(synthetic_configs):
    args = engine_setting.load_config(*synthetic_configs)
    assert vars(args)["overlap_key"] == "def"  # env -> model -> def precedence
    assert vars(args)["env_only"] is True  # env-only key survives the merge
    assert vars(args)["diffusion_unet_inference"]["modality"] == 9


def test_prepare_tensors_returns_region_spacing_modality(synthetic_configs):
    args = engine_setting.load_config(*synthetic_configs)
    out = engine_infer.prepare_tensors(args, torch.device("cpu"))
    assert len(out) == 4
    top, bottom, spacing, modality = out
    assert top.dtype == bottom.dtype == spacing.dtype == torch.float16
    assert modality.dtype == torch.long
    assert modality.item() == 9  # the fixture's modality label


# ----------------------------------------------------------------- argv layer


@pytest.mark.parametrize("vendored_module", VENDORED_ENTRY_MODULES)
def test_entry_module_help_exits_zero(vendored_module):
    """Each vendored engine entry is executable as ``__main__`` with an argparse surface."""
    env = os.environ | {"PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", vendored_module, "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


# ------------------------------------------------------------ engine pieces


def test_set_random_seed_returns_the_seed():
    assert engine_infer.set_random_seed(42) == 42


def test_save_image_writes_loadable_nifti(tmp_path):
    data = np.zeros((8, 8, 8), dtype=np.float32)
    out_path = tmp_path / "vol.nii.gz"
    logger = logging.getLogger("smoke")
    engine_infer.save_image(data, (8, 8, 8), (1.5, 1.5, 2.0), str(out_path), logger)
    assert out_path.exists()
    loaded = nib.load(str(out_path))
    assert loaded.shape == (8, 8, 8)
    assert tuple(np.diag(loaded.affine)[:3]) == (1.5, 1.5, 2.0)


# -------------------------------------------------------------- train pieces


def test_augment_modality_label_is_seed_deterministic():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[7.0]], [[8.0]], [[9.0]], [[12.0]], [[13.0]]]])
    torch.manual_seed(7)
    first = engine_train.augment_modality_label(base.clone(), prob=0.3)
    torch.manual_seed(7)
    second = engine_train.augment_modality_label(base.clone(), prob=0.3)
    assert torch.equal(first, second)


def test_augment_modality_label_zero_prob_is_identity():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[9.0]], [[13.0]]]])
    torch.manual_seed(0)
    out = engine_train.augment_modality_label(base.clone(), prob=0.0)
    assert torch.equal(out, base)


def test_load_filenames_maps_to_embedding_names(tmp_path):
    data_list = tmp_path / "data.json"
    data_list.write_text(json.dumps({"training": [{"image": str(tmp_path / "a.nii.gz")}, {"image": str(tmp_path / "b.nii.gz")}]}))
    assert engine_load_filenames(str(data_list)) == [str(tmp_path / "a_emb.nii.gz"), str(tmp_path / "b_emb.nii.gz")]


def test_create_optimizer_and_scheduler_types():
    model = torch.nn.Linear(2, 2)
    opt = engine_train.create_optimizer(model, 1e-4)
    assert type(opt) is torch.optim.Adam
    assert opt.param_groups[0]["lr"] == 1e-4

    sched = engine_train.create_lr_scheduler(opt, total_steps=10)
    assert type(sched) is torch.optim.lr_scheduler.PolynomialLR
    assert sched.total_iters == 10


# ------------------------------------------------- create_training_data pieces


def test_round_number_rounds_to_base_multiple_with_base_floor():
    assert engine_round_number(100, 128) == 128
    assert engine_round_number(200, 128) == 256
    assert engine_round_number(500, 128) == 512
    assert engine_round_number(0, 128) == 128  # clamped up to one base
    assert engine_round_number(77, 64) == 64


def test_create_transforms_pipelines_by_modality_and_dim():
    mri = engine_create_transforms((256, 256, 128), "mri")
    assert type(mri.transforms[0]).__name__ == "LoadImaged"
    assert type(mri.transforms[-1]).__name__ == "Resized"  # dim given -> resize appended

    unknown_no_dim = engine_create_transforms(None, "unknown")
    # no intensity transform for an unsupported modality, no resize without a dim
    assert [type(t).__name__ for t in unknown_no_dim.transforms] == ["LoadImaged", "EnsureChannelFirstd", "Orientationd"]


# -------------------------------------------------------------- input guards


def test_check_input_ct_valid_passes_and_invalid_size_raises():
    # controllable_anatomy_size non-empty -> label_dict_json is never read, so a
    # placeholder path is fine; (256,256,128)@(1.5,1.5,2.0) is a valid head grid.
    assert check_input_ct(["head"], ["liver"], "unused.json", (256, 256, 128), (1.5, 1.5, 2.0), [("pancreas", 0.5)]) is None

    with pytest.raises(ValueError):
        check_input_ct(["head"], ["liver"], "unused.json", (300, 300, 128), (1.5, 1.5, 2.0))


def test_check_input_mr_invalid_spacing_raises():
    # spacing[0]=0.3 < 0.4 floor -> ValueError raised before label_dict_json is read.
    with pytest.raises(ValueError):
        check_input_mr(["head"], ["liver"], "unused.json", (256, 256, 128), (0.3, 1.0, 1.0))


# -------------------------------------------------------------- primitives


def test_initialize_noise_latents_shape_and_dtype():
    latents = initialize_noise_latents((4, 8, 8), torch.device("cpu"))
    assert tuple(latents.shape) == (1, 4, 8, 8)
    assert latents.dtype == torch.float16


def test_get_body_region_index_from_mask():
    mask = torch.zeros((8, 8, 8))
    mask[:4] = 30  # thorax member -> region_1
    mask[4:] = 3  # abdomen member -> region_2

    top, bottom = get_body_region_index_from_mask(mask)
    assert top == [0, 1, 0, 0]
    assert bottom == [0, 0, 1, 0]


class _TwiceModel:
    def __call__(self, images):
        return images * 2.0


def test_dynamic_infer_direct_and_sliding_paths():
    # small input fits the roi -> model called directly
    inferer_small = SimpleNamespace(roi_size=[2, 2])
    images_small = torch.ones((1, 1, 2, 2)) * 0.25
    assert torch.equal(dynamic_infer(inferer_small, _TwiceModel(), images_small), images_small * 2.0)

    # larger volume -> sliding-window path
    inferer_sw = SlidingWindowInferer(roi_size=(2, 2, 2), sw_batch_size=1, progress=False)
    images_vol = torch.ones((1, 1, 4, 4, 4)) * 0.5
    out = dynamic_infer(inferer_sw, _TwiceModel(), images_vol)
    assert out.shape == images_vol.shape
    assert torch.allclose(out, images_vol * 2.0)


def test_define_instance_builds_monai_object():
    args = argparse.Namespace(noise_scheduler={"_target_": "monai.networks.schedulers.DDPMScheduler", "num_train_timesteps": 5})
    scheduler = define_instance(args, "noise_scheduler")
    assert isinstance(scheduler, DDPMScheduler)
    assert scheduler.num_train_timesteps == 5
