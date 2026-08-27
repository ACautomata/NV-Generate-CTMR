"""Equivalence smoke between the vendored maiisi_engine freeze and the legacy
``scripts/`` originals (issue #134).

Same synthetic config/argv in, same observable out — CPU-only real execution
(torch-marked; the CI torch job runs this on plain CPU per ADR-0015 §6, which
requires torch to be installed, not skipped around).
GPU-bound paths (``initialize_distributed``'s CUDA branch, ``run_torchrun``,
DDP gathering) are intentionally NOT executed here: their #123/spawn precedent
behavior stays registered and preserved via ``test_vendored_parity``, which
keeps those code paths byte-stable without running them.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from scripts.diff_model_create_training_data import create_transforms as legacy_create_transforms
from scripts.diff_model_create_training_data import round_number as legacy_round_number
from scripts.diff_model_infer import prepare_tensors as legacy_prepare_tensors
from scripts.diff_model_infer import save_image as legacy_save_image
from scripts.diff_model_infer import set_random_seed as legacy_set_random_seed
from scripts.diff_model_setting import load_config as legacy_load_config
from scripts.diff_model_train import augment_modality_label as legacy_augment_modality_label
from scripts.diff_model_train import create_lr_scheduler as legacy_create_lr_scheduler
from scripts.diff_model_train import create_optimizer as legacy_create_optimizer
from scripts.diff_model_train import load_filenames as legacy_load_filenames
from scripts.sample_mask import check_input_ct as legacy_check_ct
from scripts.sample_mask import check_input_mr as legacy_check_mr
from scripts.utils import dynamic_infer as legacy_dynamic_infer
from scripts.utils import get_body_region_index_from_mask as legacy_region_index

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_load_config_equivalence(synthetic_configs):
    legacy = legacy_load_config(*synthetic_configs)
    vendored = engine_setting.load_config(*synthetic_configs)
    assert vars(vendored) == vars(legacy)
    assert vars(vendored)["overlap_key"] == "def"  # env -> model -> def precedence intact


def test_prepare_tensors_equivalence(synthetic_configs):
    args = legacy_load_config(*synthetic_configs)
    device = torch.device("cpu")
    legacy_out = legacy_prepare_tensors(args, device)
    vendored_out = engine_infer.prepare_tensors(args, device)
    assert len(vendored_out) == len(legacy_out) == 4
    for v_new, v_old in zip(vendored_out, legacy_out):
        assert v_new.dtype == v_old.dtype
        assert torch.equal(v_new.cpu(), v_old.cpu())


# ----------------------------------------------------------------- argv layer


@pytest.mark.parametrize(
    ("legacy_module", "vendored_module"),
    [
        ("scripts.diff_model_train", "ctmr.infrastructure.maiisi_engine.diff_model_train"),
        ("scripts.diff_model_infer", "ctmr.infrastructure.maiisi_engine.diff_model_infer"),
        ("scripts.diff_model_create_training_data", "ctmr.infrastructure.maiisi_engine.create_training_data"),
    ],
)
def test_argparse_help_equivalent(legacy_module, vendored_module):
    """Same argv surface: --help must be identical up to the prog name."""

    def help_text(module: str) -> str:
        env = os.environ | {"PYTHONPATH": str(REPO_ROOT / "src")}
        proc = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return _normalize_help(proc.stdout, module)

    def _normalize_help(stdout: str, module: str) -> str:
        # argparse derives ``prog`` (and the usage reflow) from the module
        # basename, which legitimately differs where ADR-0015 §2 de-prefixed a
        # filename (create_training_data). The option surface lives below the
        # usage block; compare that.
        text = stdout.replace(module, "").replace(module.split(".")[-1], "MOD")
        return text.split("\n\n", 1)[1] if "\n\n" in text else text

    assert help_text(vendored_module) == help_text(legacy_module)


# ------------------------------------------------------------ engine pieces


def test_set_random_seed_equivalent():
    assert engine_infer.set_random_seed(42) == legacy_set_random_seed(42) == 42


def test_save_image_writes_equivalent_nifti(tmp_path):
    data = np.zeros((8, 8, 8), dtype=np.float32)
    legacy_path = tmp_path / "legacy.nii.gz"
    vendored_path = tmp_path / "vendored.nii.gz"
    logger = logging.getLogger("smoke")
    engine_infer.save_image(data, (8, 8, 8), (1.5, 1.5, 2.0), str(vendored_path), logger)
    legacy_save_image(data, (8, 8, 8), (1.5, 1.5, 2.0), str(legacy_path), logger)
    assert vendored_path.exists() and legacy_path.exists()
    assert vendored_path.read_bytes() == legacy_path.read_bytes()


# -------------------------------------------------------------- train pieces


def test_augment_modality_label_equivalent():
    base = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[7.0]], [[8.0]], [[9.0]], [[12.0]], [[13.0]]]])
    torch.manual_seed(7)
    vendored_out = engine_train.augment_modality_label(base.clone(), prob=0.3)
    torch.manual_seed(7)
    legacy_out = legacy_augment_modality_label(base.clone(), prob=0.3)
    assert torch.equal(vendored_out, legacy_out)


def test_load_filenames_equivalent(tmp_path):
    data_list = tmp_path / "data.json"
    data_list.write_text(json.dumps({"training": [{"image": str(tmp_path / "a.nii.gz")}, {"image": str(tmp_path / "b.nii.gz")}]}))
    assert engine_load_filenames(str(data_list)) == legacy_load_filenames(str(data_list))
    assert engine_load_filenames(str(data_list)) == [str(tmp_path / "a_emb.nii.gz"), str(tmp_path / "b_emb.nii.gz")]


def test_optimizer_and_scheduler_equivalent():
    model_new = torch.nn.Linear(2, 2)
    model_old = torch.nn.Linear(2, 2)
    model_new.load_state_dict(model_old.state_dict())

    opt_new = engine_train.create_optimizer(model_new, 1e-4)
    opt_old = legacy_create_optimizer(model_old, 1e-4)
    assert type(opt_new) is type(opt_old) is torch.optim.Adam
    assert opt_new.param_groups[0]["lr"] == opt_old.param_groups[0]["lr"] == 1e-4

    sched_new = engine_train.create_lr_scheduler(opt_new, total_steps=10)
    sched_old = legacy_create_lr_scheduler(opt_old, total_steps=10)
    assert type(sched_new) is type(sched_old) is torch.optim.lr_scheduler.PolynomialLR
    assert sched_new.total_iters == sched_old.total_iters == 10


# ------------------------------------------------- create_training_data pieces


def test_round_number_equivalent():
    for number, base in [(100, 128), (200, 128), (500, 128), (0, 128), (77, 64)]:
        assert engine_round_number(number, base) == legacy_round_number(number, base)


def _strip_memory_addresses(text: str) -> str:
    return re.sub(r" at 0x[0-9a-fA-F]+", "", text)


def test_create_transforms_equivalent():
    new_mri = engine_create_transforms((256, 256, 128), "mri")
    old_mri = legacy_create_transforms((256, 256, 128), "mri")
    assert len(new_mri.transforms) == len(old_mri.transforms)  # intensity transform inclusion parity
    assert _strip_memory_addresses(repr(new_mri)) == _strip_memory_addresses(repr(old_mri))


# -------------------------------------------------------------- input guards


def test_check_input_ct_valid_and_invalid_message_equivalent():
    body, anatomy, label_json = ["head"], ["liver"], "unused.json"
    size, spacing = (256, 256, 128), (1.5, 1.5, 2.0)
    assert check_input_ct(body, anatomy, label_json, size, spacing, [("pancreas", 0.5)]) is None

    with pytest.raises(ValueError) as exc_legacy:
        legacy_check_ct(body, anatomy, label_json, (300, 300, 128), spacing)
    with pytest.raises(ValueError) as exc_vendored:
        check_input_ct(body, anatomy, label_json, (300, 300, 128), spacing)
    assert str(exc_vendored.value) == str(exc_legacy.value)


def test_check_input_mr_invalid_message_equivalent():
    size, spacing = (256, 256, 128), (0.3, 1.0, 1.0)
    with pytest.raises(ValueError) as exc_legacy:
        legacy_check_mr(["head"], ["liver"], "unused.json", size, spacing)
    with pytest.raises(ValueError) as exc_vendored:
        check_input_mr(["head"], ["liver"], "unused.json", size, spacing)
    assert str(exc_vendored.value) == str(exc_legacy.value)


# -------------------------------------------------------------- primitives


def test_initialize_noise_latents_shape_and_dtype():
    latents = initialize_noise_latents((4, 8, 8), torch.device("cpu"))
    assert tuple(latents.shape) == (1, 4, 8, 8)
    assert latents.dtype == torch.float16


def test_get_body_region_index_from_mask_equivalence():
    mask = torch.zeros((8, 8, 8))
    mask[:4] = 30  # thorax member -> region_1
    mask[4:] = 3  # abdomen member -> region_2

    top_v, bottom_v = get_body_region_index_from_mask(mask)
    top_l, bottom_l = legacy_region_index(mask)
    assert top_v == top_l == [0, 1, 0, 0]
    assert bottom_v == bottom_l == [0, 0, 1, 0]


class _TwiceModel:
    def __call__(self, images):
        return images * 2.0


def test_dynamic_infer_direct_and_sliding_paths_equivalent():
    inferer_small = SimpleNamespace(roi_size=[2, 2])
    images_small = torch.ones((1, 1, 2, 2)) * 0.25
    assert torch.equal(legacy_dynamic_infer(inferer_small, _TwiceModel(), images_small), dynamic_infer(inferer_small, _TwiceModel(), images_small))

    inferer_sw = SlidingWindowInferer(roi_size=(2, 2, 2), sw_batch_size=1, progress=False)
    images_vol = torch.ones((1, 1, 4, 4, 4)) * 0.5
    new_out = dynamic_infer(inferer_sw, _TwiceModel(), images_vol)
    legacy_out = legacy_dynamic_infer(inferer_sw, _TwiceModel(), images_vol)
    assert new_out.shape == legacy_out.shape
    assert torch.allclose(new_out, legacy_out)


def test_define_instance_builds_monai_object():
    args = argparse.Namespace(noise_scheduler={"_target_": "monai.networks.schedulers.DDPMScheduler", "num_train_timesteps": 5})
    scheduler = define_instance(args, "noise_scheduler")
    assert isinstance(scheduler, DDPMScheduler)
    assert scheduler.num_train_timesteps == 5
