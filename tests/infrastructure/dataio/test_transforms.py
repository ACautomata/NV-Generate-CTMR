"""Config-level and end-to-end tests for ctmr.infrastructure.dataio.transforms (migrated from the retired scripts layer (git history; ``transforms``))."""

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import Compose, ScaleIntensityRanged, ScaleIntensityRangePercentilesd

from ctmr.infrastructure.dataio.transforms import (
    SUPPORT_MODALITIES,
    VAE_Transform,
    define_fixed_intensity_transform,
    define_random_intensity_transform,
    define_vae_transform,
)

pytestmark = pytest.mark.torch


def _write_nifti(tmp_path, shape=(10, 10, 10)):
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(np.random.default_rng(0).random(shape).astype(np.float32), np.eye(4)), str(path))
    return str(path)


def test_supported_modalities():
    assert SUPPORT_MODALITIES == ["ct", "mri"]


def test_fixed_intensity_transform_per_modality():
    ct = define_fixed_intensity_transform("ct")
    assert len(ct) == 1 and isinstance(ct[0], ScaleIntensityRanged)
    mri = define_fixed_intensity_transform("mri")
    assert len(mri) == 1 and isinstance(mri[0], ScaleIntensityRangePercentilesd)
    # unsupported modality warns and yields no transforms
    with pytest.warns(UserWarning, match="only support"):
        assert define_fixed_intensity_transform("pet") == []


def test_random_intensity_transform_per_modality():
    assert define_random_intensity_transform("ct") == []  # CT HU intensities are stable
    mri = define_random_intensity_transform("mri")
    assert len(mri) == 4
    with pytest.warns(UserWarning, match="only support"):
        assert define_random_intensity_transform("pet") == []


def test_define_vae_transform_rejects_unknown_spacing_type():
    with pytest.raises(ValueError, match="spacing_type"):
        define_vae_transform(is_train=False, modality="ct", random_aug=False, spacing_type="bogus")


def test_define_vae_transform_val_pipeline_runs_on_real_nifti(tmp_path):
    transform = define_vae_transform(is_train=False, modality="ct", random_aug=False)
    assert isinstance(transform, Compose)
    out = transform({"image": _write_nifti(tmp_path)})
    assert "image" in out
    assert tuple(out["image"].shape) == (1, 12, 12, 12)  # DivisiblePadd k=4 pads 10 -> 12
    assert out["image"].dtype == torch.float32


def test_define_vae_transform_train_pipeline_patches_to_requested_size(tmp_path):
    transform = define_vae_transform(is_train=True, modality="ct", random_aug=False, patch_size=[8, 8, 8])
    out = transform({"image": _write_nifti(tmp_path)})
    assert tuple(out["image"].shape) == (1, 8, 8, 8)  # SpatialPadd + RandSpatialCropd fix the patch


def test_vae_transform_class_switches_pipeline_by_modality(tmp_path):
    vaet = VAE_Transform(is_train=False, random_aug=False)
    assert set(vaet.transform_dict) == {"ct", "mri"}
    out = vaet({"image": _write_nifti(tmp_path)}, fixed_modality="ct")
    assert tuple(out["image"].shape) == (1, 12, 12, 12)
