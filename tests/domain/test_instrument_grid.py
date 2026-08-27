"""Convergence-gate tests for the instrument-grid special case (ADR-0008, #105).

Pins ``INSTRUMENT_GRID`` and the continuum/label adapter factories, and proves
the frozen terminal-acceptance geometry -- now routed through ``ctmr.domain.grid`` --
is bit-identical to the pre-#105 implementation on synthetic inputs. That is
the ADR-0008 convergence gate: SimpleITK unit level, any machine, no cluster,
no external data (ADR-0013); the sugon byte-identical rerun stays with the
frozen gate (issue #T12).
"""

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.domain.grid import INSTRUMENT_GRID, InstrumentGridAdapter


class FinalAcceptanceReferenceGeometry:
    """Byte-for-byte snapshot of the pre-#105 terminal-acceptance geometry.

    The frozen convergence standard of ADR-0008: the private methods of
    ``GeneratedVolumeResampler`` in the distribution ``measurement_run``
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
    from ctmr.application.acceptance.distribution.measurement_run import GeneratedVolumeResampler

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
    from ctmr.application.acceptance.distribution.measurement_run import GeneratedVolumeResampler

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
