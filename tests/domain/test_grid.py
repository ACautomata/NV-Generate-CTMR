"""Convergence-gate unit tests for ctmr.domain.grid (ADR-0008, issue #105).

The generic engine's public surface -- TargetGrid / GridResampler /
CenterCropOrPad -- is the highest available seam (#102 testing decisions):
pure transforms, in-memory ``sitk.Image`` in and out, no file IO. Synthesis is
self-contained per test (ADR-0013); no external data, any machine can run.
"""

import dataclasses

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.domain.grid import CenterCropOrPad, GridResampler, TargetGrid


def test_target_grid_is_a_frozen_xyz_value_object():
    grid = TargetGrid(size=(240, 240, 155), spacing=(1.0, 1.0, 1.0))
    twin = TargetGrid(size=(240, 240, 155), spacing=(1.0, 1.0, 1.0))
    assert grid.size == (240, 240, 155)
    assert grid.spacing == (1.0, 1.0, 1.0)
    assert grid == twin
    assert hash(grid) == hash(twin)
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.size = (1, 1, 1)


def test_grid_resampler_targets_the_grid_spacing_with_rounded_extent():
    array = np.zeros((10, 20, 30), dtype=np.float32)  # zyx; xyz size is (30, 20, 10)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.75, 1.2, 2.0))  # xyz mm
    grid = TargetGrid(size=(16, 16, 16), spacing=(2.0, 2.0, 2.0))
    resampled = GridResampler(sitk.sitkLinear).resample(image, grid)
    assert resampled.GetSpacing() == (2.0, 2.0, 2.0)
    # output size = round(original_size * original_spacing / target_spacing), per xyz axis
    assert resampled.GetSize() == (11, 12, 10)


def test_grid_resampler_injects_the_interpolation_strategy():
    array = np.zeros((2, 2, 6), dtype=np.float32)
    array[:, :, 2:4] = 10.0  # x ramp: 0, 0, 10, 10, 0, 0
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((4.0, 4.0, 4.0))  # xyz mm -> 1mm quadruples the x axis
    grid = TargetGrid(size=(24, 8, 8), spacing=(1.0, 1.0, 1.0))
    nearest = GridResampler(sitk.sitkNearestNeighbor).resample(image, grid)
    spline = GridResampler(sitk.sitkBSpline).resample(image, grid)
    # trim the border: its outside-domain samples take the frozen default fill
    # (SetDefaultPixelValue(GetPixelIDValue()) == 8.0 for float32), not a strategy choice
    trim = (slice(2, -2), slice(2, -2), slice(2, -2))
    assert set(np.unique(sitk.GetArrayFromImage(nearest)[trim])) <= {0.0, 10.0}  # NN invents no values
    assert not set(np.unique(sitk.GetArrayFromImage(spline)[trim])) <= {0.0, 10.0}  # B-spline interpolates


def test_center_crop_or_pad_crops_the_centre_of_larger_axes():
    array = np.broadcast_to(np.arange(7, dtype=np.float32)[None, None, :], (3, 7, 7)).copy()  # value == x index
    image = sitk.GetImageFromArray(array)
    grid = TargetGrid(size=(4, 7, 3), spacing=(1.0, 1.0, 1.0))  # only x is larger than the target
    cropped = CenterCropOrPad().crop_or_pad(image, grid)
    assert sitk.GetArrayFromImage(cropped).shape == (3, 7, 4)  # zyx
    expected = np.broadcast_to(np.arange(1, 5, dtype=np.float32)[None, None, :], (3, 7, 4))
    assert np.array_equal(sitk.GetArrayFromImage(cropped), expected)  # x [1:5]


def test_center_crop_or_pad_pads_the_centre_of_smaller_axes():
    array = np.ones((3, 5, 4), dtype=np.float32)  # zyx; xyz size is (4, 5, 3)
    image = sitk.GetImageFromArray(array)
    grid = TargetGrid(size=(4, 5, 9), spacing=(1.0, 1.0, 1.0))  # only z is smaller than the target
    padded = CenterCropOrPad().crop_or_pad(image, grid)
    out = sitk.GetArrayFromImage(padded)
    assert out.shape == (9, 5, 4)
    assert (out[3:6] == 1.0).all()  # (9-3)//2 = 2 -> dst z [3:6]
    assert (out[:3] == 0.0).all() and (out[6:] == 0.0).all()


def test_center_crop_or_pad_maps_xyz_targets_onto_the_zyx_array():
    # The #38 drift this module replaces applied xyz sizes/slices to the zyx
    # array (ADR-0008); the xyz target (4, 3, 2) must land as array shape
    # (2, 3, 4), with the crop acting on the x axis.
    array = np.zeros((5, 3, 6), dtype=np.float32)  # zyx; xyz size is (6, 3, 5)
    array[:, :, 1:5] = 7.0  # the x crop keeps columns [1:5] of the six
    image = sitk.GetImageFromArray(array)
    grid = TargetGrid(size=(4, 3, 2), spacing=(1.0, 1.0, 1.0))
    cropped = CenterCropOrPad().crop_or_pad(image, grid)
    out = sitk.GetArrayFromImage(cropped)
    assert out.shape == (2, 3, 4)  # zyx reversal of the xyz target
    assert (out == 7.0).all()


def test_center_crop_or_pad_realises_the_grid_metadata():
    array = np.zeros((2, 2, 2), dtype=np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((3.0, 3.0, 3.0))
    image.SetOrigin((10.0, -5.0, 2.0))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))  # non-identity (e.g. RAS)
    grid = TargetGrid(size=(4, 4, 4), spacing=(2.0, 2.0, 2.0))
    cropped = CenterCropOrPad().crop_or_pad(image, grid)
    assert cropped.GetSpacing() == (2.0, 2.0, 2.0)  # the grid's spacing
    assert cropped.GetOrigin() == (10.0, -5.0, 2.0)  # terminal-acceptance behaviour: carried over
    assert cropped.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)  # reset to identity
