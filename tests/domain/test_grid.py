"""Convergence-gate unit tests for ctmr.domain.grid (ADR-0008, issues #105/#133).

The generic engine's public surface -- TargetGrid / GridResampler /
CenterCropOrPad -- is the highest available seam (#102 testing decisions):
pure transforms, in-memory ``sitk.Image`` in and out, no file IO. Synthesis is
self-contained per test (ADR-0013); no external data, any machine can run.

The instrument special case pins ``INSTRUMENT_GRID`` and the continuum/label
adapter factories, and proves the frozen terminal-acceptance geometry -- now
routed through ``ctmr.domain.grid`` -- is bit-identical to the pre-#105
implementation on synthetic inputs. That is the ADR-0008 convergence gate:
SimpleITK unit level, any machine, no cluster, no external data (ADR-0013);
the sugon byte-identical rerun stays with the frozen gate (issue #T12).
"""

import dataclasses

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.domain.grid import INSTRUMENT_GRID, CenterCropOrPad, GridResampler, InstrumentGridAdapter, TargetGrid


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


class FinalAcceptanceReferenceGeometry:
    """Byte-for-byte snapshot of the pre-#105 terminal-acceptance geometry.

    The frozen convergence standard of ADR-0008: the private methods of
    ``GeneratedVolumeResampler`` in ``scripts/nnunet_l2_final_acceptance_nifti.py``
    before the ``ctmr.domain.grid`` extraction. Do not edit -- drift here is exactly
    what this gate exists to catch.
    """

    @staticmethod
    def resample_to_1mm(image, interpolator):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_spacing = [1.0, 1.0, 1.0]
        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)

    @staticmethod
    def crop_or_pad(image, target_size):
        size = image.GetSize()
        array = sitk.GetArrayFromImage(image)  # z, y, x
        cropped = np.zeros(tuple(reversed(target_size)), dtype=array.dtype)  # array axes are zyx
        src_slices, dst_slices = [], []
        for s, t in zip(size, target_size):  # slices built in xyz order, applied reversed below
            if s >= t:
                start = (s - t) // 2
                src_slices.append(slice(start, start + t))
                dst_slices.append(slice(0, t))
            else:
                start = (t - s) // 2
                src_slices.append(slice(0, s))
                dst_slices.append(slice(start, start + s))
        cropped[tuple(reversed(dst_slices))] = array[tuple(reversed(src_slices))]
        for axis in (1, 2):  # RAS(DM grid) -> LPS(instrument grid); zyx array axes (y=1, x=2)
            cropped = np.flip(cropped, axis=axis)
        result = sitk.GetImageFromArray(cropped)
        result.SetSpacing((1.0, 1.0, 1.0))
        result.SetOrigin(image.GetOrigin())
        result.SetDirection(np.eye(3).flatten().tolist())
        return result


def test_instrument_grid_pins_the_frozen_shape():
    assert INSTRUMENT_GRID.size == (240, 240, 155)
    assert INSTRUMENT_GRID.spacing == (1.0, 1.0, 1.0)


def test_instrument_adapters_reject_non_frozen_strategies():
    # ADR-0008 decision 2 pins the parameter table; only the two named
    # factories may construct an adapter -- linear & friends belong to the
    # generic GridResampler, not the frozen instrument contract.
    with pytest.raises(ValueError):
        InstrumentGridAdapter(sitk.sitkLinear)


def test_continuum_adapter_aligns_onto_the_instrument_grid():
    array = np.zeros((80, 130, 300), dtype=np.float32)  # zyx
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 2.0, 1.5))  # xyz mm
    aligned = InstrumentGridAdapter.continuum().align(image)
    assert aligned.GetSize() == (240, 240, 155)
    assert aligned.GetSpacing() == (1.0, 1.0, 1.0)


def test_label_adapter_aligns_onto_the_instrument_grid_without_inventing_labels():
    array = np.zeros((200, 100, 120), dtype=np.uint8)  # zyx
    array[40:160, 20:80, 10:110] = 1
    array[60:120, 30:70, 20:100] = 2
    array[80:100, 40:60, 40:90] = 3
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 1.0, 1.0))  # xyz mm
    aligned = InstrumentGridAdapter.label().align(image)
    assert sitk.GetArrayFromImage(aligned).shape == (155, 240, 240)
    assert set(np.unique(sitk.GetArrayFromImage(aligned))) <= {0, 1, 2, 3}  # NN invents no label values


def test_terminal_acceptance_continuum_geometry_matches_the_frozen_reference(tmp_path):
    pytest.importorskip("scipy")  # the script imports scipy at module level (ADR-0013 skip pattern)
    from scripts.nnunet_l2_final_acceptance_nifti import GeneratedVolumeResampler

    zz, yy, xx = np.mgrid[0:80, 0:130, 0:300]  # zyx; xyz size (300, 130, 80) -> crop x/y, pad z
    image = sitk.GetImageFromArray((800.0 * np.exp(-(((zz - 40.0) ** 2 + (yy - 65.0) ** 2 + (xx - 150.0) ** 2) / 2.0e4))).astype(np.float32))
    image.SetSpacing((1.0, 2.0, 1.5))  # xyz mm -> resampled to (300, 260, 120)
    image.SetOrigin((12.0, -7.0, 3.0))  # non-trivial origin exercises metadata carry-over
    source = tmp_path / "continuum.nii.gz"
    sitk.WriteImage(image, str(source))

    destination = tmp_path / "aligned.nii.gz"
    GeneratedVolumeResampler().write(source, destination)

    reference = FinalAcceptanceReferenceGeometry.crop_or_pad(
        FinalAcceptanceReferenceGeometry.resample_to_1mm(sitk.ReadImage(str(source)), sitk.sitkBSpline),
        (240, 240, 155),
    )
    produced = sitk.ReadImage(str(destination))
    assert produced.GetSize() == (240, 240, 155)
    assert np.array_equal(sitk.GetArrayFromImage(produced), sitk.GetArrayFromImage(reference))
    assert produced.GetSpacing() == reference.GetSpacing()
    assert produced.GetOrigin() == reference.GetOrigin()
    assert produced.GetDirection() == reference.GetDirection()
    assert produced.GetPixelIDValue() == reference.GetPixelIDValue()


def test_terminal_acceptance_label_geometry_matches_the_frozen_reference(tmp_path):
    pytest.importorskip("scipy")  # the script imports scipy at module level (ADR-0013 skip pattern)
    from scripts.nnunet_l2_final_acceptance_nifti import GeneratedVolumeResampler

    array = np.zeros((200, 100, 120), dtype=np.uint8)  # zyx; xyz size (120, 100, 200) @ (0.5, 1.0, 1.0)
    array[40:160, 20:80, 10:110] = 1
    array[60:120, 30:70, 20:100] = 2
    array[80:100, 40:60, 40:90] = 3
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 1.0, 1.0))  # xyz mm -> resampled to (60, 100, 200): pad x/y, crop z
    image.SetOrigin((-4.0, 6.0, 9.0))
    source = tmp_path / "label.nii.gz"
    sitk.WriteImage(image, str(source))

    aligned = GeneratedVolumeResampler().label_to_grid(source)

    reference = FinalAcceptanceReferenceGeometry.crop_or_pad(
        FinalAcceptanceReferenceGeometry.resample_to_1mm(sitk.ReadImage(str(source)), sitk.sitkNearestNeighbor),
        (240, 240, 155),
    )
    assert aligned is not None
    assert aligned.dtype == np.uint8
    assert aligned.shape == (155, 240, 240)
    assert np.array_equal(aligned, sitk.GetArrayFromImage(reference).astype(np.uint8))
