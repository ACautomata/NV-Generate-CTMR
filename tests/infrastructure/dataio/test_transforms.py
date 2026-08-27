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


"""Behaviour gates for ctmr.infrastructure.dataio.transforms (#132).

Verbatim lift of scripts/transforms.py (the MAISI VAE pipeline factory):
per-modality fixed/random intensity recipes (asserted behaviourally on
in-memory arrays -- MONAI keeps its window attributes private), train-vs-val
crop structure, spacing-mode validation and modality routing of VAE_Transform.
The MRI branch's LoadImaged legs are not driven end-to-end here (they need a
real image file; that leg is exercised by the generate pipelines). monai+torch
level, CPU-only.
"""

import pytest

pytest.importorskip("monai")

import numpy as np  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose,
    DivisiblePadd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
)

import ctmr.infrastructure.dataio.transforms as dt  # noqa: E402


def _compose(xform):
    return list(xform.transforms) if isinstance(xform, Compose) else [xform]


def test_fixed_ct_transform_scales_hu_window_to_unit_range():
    xforms = dt.define_fixed_intensity_transform("ct")
    assert len(xforms) == 1 and isinstance(xforms[0], ScaleIntensityRanged)
    out = Compose(xforms)({"image": np.linspace(-1000, 1000, 64, dtype=np.float32).reshape(1, 4, 4, 4)})["image"]
    assert float(out.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-5)


def test_fixed_mri_transform_uses_percentile_scaling():
    tf = dt.define_fixed_intensity_transform("mri")[0]
    assert isinstance(tf, ScaleIntensityRangePercentilesd)
    ramp = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4) * 100.0
    out = Compose([tf])({"image": ramp})["image"]
    lo, hi = np.percentile(ramp, [0.0, 99.5])
    expected = (ramp - lo) / (hi - lo)
    assert np.allclose(out, expected, rtol=1e-3)


def test_unsupported_modality_warns_and_skips_intensity_transforms():
    for fn in (dt.define_fixed_intensity_transform, dt.define_random_intensity_transform):
        with pytest.warns(UserWarning, match="only support"):
            assert fn("pet") == []


def test_random_mri_recipe_composition():
    got = dt.define_random_intensity_transform("mri")
    assert [type(t) for t in got] == [RandBiasFieldd, RandGibbsNoised, RandAdjustContrastd, RandHistogramShiftd]
    assert dt.define_random_intensity_transform("ct") == []


def test_spacing_type_validation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="original.*fixed.*rand_zoom"):
        dt.define_vae_transform(is_train=False, modality="ct", random_aug=False, spacing_type="chaotic")


def test_train_pipeline_carries_flips_rotations_intensity_jitter_and_crop():
    tr = dt.define_vae_transform(is_train=True, modality="mri", random_aug=True, patch_size=[16, 16, 16])
    kinds = [type(t) for t in _compose(tr)]
    for expected in (
        RandSpatialCropd,
        SpatialPadd,
        RandFlipd,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandBiasFieldd,
        RandGibbsNoised,
        RandAdjustContrastd,
        RandHistogramShiftd,
    ):
        assert expected in kinds, f"missing {expected.__name__}"
    assert sum(k is RandFlipd for k in kinds) == 3  # one per spatial axis


def test_val_pipeline_central_crops_or_divisible_pads():
    whole = dt.define_vae_transform(is_train=False, modality="ct", random_aug=False)
    assert any(isinstance(t, DivisiblePadd) for t in _compose(whole))
    centred = dt.define_vae_transform(is_train=False, modality="ct", random_aug=False, val_patch_size=[32, 32, 32])
    assert any(isinstance(t, ResizeWithPadOrCropd) for t in _compose(centred))


def test_vae_transform_class_routes_modalities_by_name(monkeypatch):
    class _Recorder:
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, img):
            return {"applied": self.tag}

    class _StubVT(dt.VAE_Transform):
        def __init__(self):
            self.transform_dict = {key: _Recorder(key) for key in ("ct", "mri")}

    vt = _StubVT()
    assert set(vt.transform_dict) == {"ct", "mri"}
    assert vt({"class": "MRI"})["applied"] == "mri"  # falls back to the record's own key
    assert vt({}, fixed_modality="CT")["applied"] == "ct"  # explicit override wins
