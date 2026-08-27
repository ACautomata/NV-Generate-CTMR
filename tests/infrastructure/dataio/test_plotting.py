"""Tests for ctmr.infrastructure.dataio.plotting (migrated from scripts/utils_plot, renamed per ADR-0015 rule ③)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ctmr.infrastructure.dataio.plotting import (
    find_label_center_loc,
    get_xyz_plot,
    normalize_label_to_uint8,
    show_image,
    to_shape,
    visualize_one_slice_in_3d,
)

pytestmark = pytest.mark.torch


def test_find_label_center_loc_reports_middle_nonzero_index():
    mask = torch.zeros(8, 8, 8)
    mask[4, 2, 6] = 1
    assert find_label_center_loc(mask) == [4, 2, 6]


def test_find_label_center_loc_empty_mask_gives_none_per_axis():
    mask = torch.zeros(3, 3, 3)
    assert find_label_center_loc(mask) == [None, None, None]


def test_to_shape_pads_to_exact_shape():
    a = np.ones((2, 2, 2))
    padded = to_shape(a, (4, 6, 8))
    assert padded.shape == (4, 6, 8)
    assert np.array_equal(padded[1:3, 2:4, 3:5], a)
    assert np.array_equal(to_shape(a, (2, 2, 2)), a)  # already at target: identity


def test_normalize_label_to_uint8_pins_platform_drift():
    """Upstream drifted under monai>=1.5: AsDiscrete(to_onehot) drops a dim, so the
    permute(1,0,2,3) order crashes. Pinned as the current external behavior; the
    numerical fix belongs to the expand phase, not to this migration (verbatim copy policy)."""
    colorize = torch.zeros(3, 4, 1, 1)
    colorize[0, 1, 0, 0] = 1.0
    label = torch.tensor([[[1, 0, 2]]], dtype=torch.long)  # [1,1,3]
    with pytest.raises(RuntimeError, match="permute"):
        normalize_label_to_uint8(colorize, label, 4)


def test_visualize_one_slice_in_3d_invalid_axis_raises_index_error():
    # Upstream orders center computation before the axis check, so an invalid axis
    # surfaces as IndexError, not the documented ValueError. Pinned as-is.
    volume = torch.rand(1, 4, 4, 4)
    with pytest.raises(IndexError):
        visualize_one_slice_in_3d(volume, axis=3)


def test_visualize_one_slice_in_3d_image_branch_is_three_channel_float():
    volume = torch.rand(1, 4, 5, 6)
    slice_img = visualize_one_slice_in_3d(volume, axis=2, center=2, mask_bool=False)
    assert slice_img.dtype == np.float32
    assert slice_img.shape[-1] == 3


def test_get_xyz_plot_output_is_three_channel_image():
    volume = torch.rand(1, 6, 7, 8)
    plot = get_xyz_plot(volume, [3, 3, 4], mask_bool=False)
    assert plot.ndim == 3
    assert plot.shape[-1] == 3
    assert plot.shape[0] == 8  # max(H, W, D) of the flipped volume
    assert np.isfinite(plot).all()


def test_show_image_renders_without_error(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    show_image(np.zeros((4, 4)), title="smoke")
    assert shown == [True]
    plt.close("all")
