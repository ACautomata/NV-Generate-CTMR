"""Tests for ctmr.infrastructure.dataio.sample_mask (migrated from scripts/sample_mask).

GPU-only full sampling (``ldm_conditional_sample_one_mask``) is out of the "any machine" line
(ADR-0015 §6); the CPU gate functions and the engine-side copies are exercised here.
"""

import json

import pytest
import torch

from ctmr.infrastructure.dataio.sample_mask import (
    ReconModel,
    check_input_ct,
    check_input_mr,
    dynamic_infer,
    filter_mask_with_organs,
    initialize_noise_latents,
)

pytestmark = pytest.mark.torch


def _label_dict(tmp_path):
    path = tmp_path / "label_dict.json"
    path.write_text(json.dumps({"liver": 1, "lung": 23}))
    return str(path)


def test_filter_mask_with_organs_keeps_only_requested_organs():
    combine = torch.tensor([[[3, 1, 4, 1]]], dtype=torch.long)  # [1,1,4]
    out = filter_mask_with_organs(combine, [4, 1])
    assert out.tolist() == [[[0, 2, 1, 2]]]  # 3->0 (not requested), 4->1, 1->2


def test_filter_mask_with_organs_all_absent_gives_all_zero():
    combine = torch.tensor([[[7, 8]]], dtype=torch.long)
    out = filter_mask_with_organs(combine, [42])
    assert out.tolist() == [[[0, 0]]]


def test_check_input_ct_accepts_valid_head_pipeline():
    check_input_ct(
        body_region=["head"],
        anatomy_list=["liver"],
        label_dict_json=None,  # not consulted: controllable list is non-empty
        output_size=[512, 512, 128],
        spacing=[1.0, 1.0, 1.0],
        controllable_anatomy_size=None,
    )


def test_check_input_ct_rejects_output_size_with_unequal_first_two():
    with pytest.raises(ValueError, match="first two components"):
        check_input_ct(["head"], ["liver"], None, [512, 256, 128], [1, 1, 1], controllable_anatomy_size=None)


def test_check_input_ct_rejects_unsupported_xy_size():
    with pytest.raises(ValueError, match="chosen from"):
        check_input_ct(["head"], ["liver"], None, [300, 300, 128], [1, 1, 1], controllable_anatomy_size=None)


def test_check_input_ct_rejects_unsupported_depth():
    with pytest.raises(ValueError, match="chosen from"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 100], [1, 1, 1], controllable_anatomy_size=None)


def test_check_input_ct_rejects_unequal_and_out_of_range_spacing():
    with pytest.raises(ValueError, match="need to be equal"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1.5, 1], controllable_anatomy_size=None)
    with pytest.raises(ValueError, match="between"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [0.2, 0.2, 0.2], controllable_anatomy_size=None)


def test_check_input_ct_rejects_small_fov():
    with pytest.raises(ValueError, match="field of view"):
        check_input_ct(["head"], ["liver"], None, [256, 256, 128], [0.75, 0.75, 5.0], controllable_anatomy_size=None)


def test_check_input_ct_validates_controllable_anatomy():
    ok = [("pancreas", 0.5)]
    check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=ok)
    with pytest.raises(ValueError, match="less than 10"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[("liver", 0.5)] * 11)
    with pytest.raises(ValueError, match="chosen from"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[("nose", 0.5)])
    with pytest.raises(ValueError, match="do not repeat"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[("liver", 0.5), ("liver", 0.6)])
    with pytest.raises(ValueError, match="Only one controllable tumor"):
        check_input_ct(
            ["head"],
            ["liver"],
            None,
            [512, 512, 128],
            [1, 1, 1],
            controllable_anatomy_size=[("lung tumor", 0.5), ("bone lesion", 0.5)],
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        check_input_ct(["head"], ["liver"], None, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[("liver", 1.5)])


def test_check_input_ct_checks_body_region_and_anatomy_when_no_controllable(tmp_path):
    path = _label_dict(tmp_path)
    check_input_ct(["head"], ["liver"], path, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="chosen from"):
        check_input_ct(["arm"], ["liver"], path, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="have to be chosen from"):
        check_input_ct(["head"], ["nose"], path, [512, 512, 128], [1, 1, 1], controllable_anatomy_size=[])


def test_check_input_mr_accepts_valid_combination(tmp_path):
    check_input_mr(["head"], ["liver"], _label_dict(tmp_path), [256, 256, 128], [1, 1, 1], controllable_anatomy_size=[])
    check_input_mr(["head"], ["liver"], _label_dict(tmp_path), [256, 128, 256], [1, 1, 1], controllable_anatomy_size=[])


def test_check_input_mr_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="At least two components"):
        check_input_mr(["head"], ["liver"], None, [128, 200, 256], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="need to be equal when the third size"):
        check_input_mr(["head"], ["liver"], None, [128, 200, 128], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="chosen from.*when output_size"):
        check_input_mr(["head"], ["liver"], None, [300, 300, 128], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="only be"):
        check_input_mr(["head"], ["liver"], None, [128, 128, 256], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="have to be chosen from"):
        check_input_mr(["head"], ["liver"], None, [512, 512, 300], [1, 1, 1], controllable_anatomy_size=[])
    with pytest.raises(ValueError, match="between"):
        check_input_mr(["head"], ["liver"], None, [256, 256, 128], [6, 6, 6], controllable_anatomy_size=[])


class _FakeAutoencoder:
    def __init__(self, factor):
        self.factor = factor

    def decode_stage_2_outputs(self, z):
        return z * self.factor


def test_recon_model_applies_scale_factor_before_decode():
    recon = ReconModel(_FakeAutoencoder(factor=2.0), scale_factor=4.0)
    z = torch.tensor([4.0, 8.0])
    out = recon(z)
    assert torch.allclose(out, z / 2.0)  # 2 * (z / 4)


def test_initialize_noise_latents_shape_and_dtype():
    torch.manual_seed(0)
    latents = initialize_noise_latents([4, 4, 4], torch.device("cpu"))
    assert latents.shape == (1, 4, 4, 4)
    assert latents.dtype == torch.float16
    assert torch.isfinite(latents).all()


class _CountingInferer:
    def __init__(self, roi_size):
        self.roi_size = roi_size
        self.calls = 0

    def __call__(self, network, inputs):
        self.calls += 1
        return network(inputs)


def test_dynamic_infer_uses_model_when_smaller_than_roi():
    inferer = _CountingInferer(roi_size=(2, 2, 2))
    images = torch.zeros(1, 1, 2, 2, 2)  # 8 elements == roi volume
    out = dynamic_infer(inferer, lambda x: x + 1, images)
    assert inferer.calls == 0
    assert torch.equal(out, images + 1)


def test_dynamic_infer_uses_sliding_window_when_larger_than_roi():
    inferer = _CountingInferer(roi_size=(2, 2, 2))
    images = torch.zeros(1, 1, 4, 4, 4)  # exceeds roi volume
    out = dynamic_infer(inferer, lambda x: x + 1, images)
    assert inferer.calls == 1
    assert torch.equal(out, images + 1)


def test_dynamic_infer_rejects_roi_ndim_mismatch():
    inferer = _CountingInferer(roi_size=(2, 2))
    images = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="ROI length"):
        dynamic_infer(inferer, lambda x: x, images)


class _BoomInferer:
    def __init__(self, roi_size):
        self.roi_size = roi_size

    def __call__(self, network, inputs):
        raise RuntimeError("simulated infer failure")


def test_dynamic_infer_restores_roi_size_when_inference_raises():
    """Regression (Codex P2, PR #155): when sliding-window inference raises after
    ``inferer.roi_size`` was clamped to the input, the original ROI must be
    restored — otherwise every retry or the next volume silently reuses the
    dimensions adjusted for the failed input."""
    inferer = _BoomInferer(roi_size=(2, 2, 2))
    images = torch.zeros(1, 1, 4, 4, 4)  # larger than roi -> sliding-window branch
    with pytest.raises(RuntimeError, match="simulated"):
        dynamic_infer(inferer, lambda x: x, images)
    assert inferer.roi_size == (2, 2, 2)
