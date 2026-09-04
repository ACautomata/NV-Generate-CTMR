"""Convergence-gate tests for the remaining #106 call-site adoptions (ADR-0008).

The #105 gate pinned the engine against the frozen terminal-acceptance geometry;
this file proves the *call sites* adopted in issue #106 now land on that same
standard: under identical synthetic inputs, each adopted path must produce
bit-identical output to ``ctmr.domain.grid`` (which the #105 tests already proved equal
to the frozen standard). The two #38-family scripts carry no RAS->LPS flip
(that flip is terminal-acceptance-only), so equality against the plain adapter
is the full convergence statement here. SimpleITK unit level, any machine, no
cluster, no external data (ADR-0008 decision 5 / ADR-0013).
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.domain.grid import CenterCropOrPad, GridResampler, InstrumentGridAdapter, TargetGrid
from ctmr.domain.orientation import RasOrientation

RAS_DIRECTION = RasOrientation.RAS_DIRECTION


def _smooth_volume(shape_zyx, spacing_xyz):
    """A smooth asymmetric float32 volume whose value encodes position, so any
    axis-order or centring mistake is visible in the output."""
    zz, yy, xx = np.mgrid[0 : shape_zyx[0], 0 : shape_zyx[1], 0 : shape_zyx[2]]
    array = 100.0 * np.exp(-(((xx - shape_zyx[2] / 3.0) ** 2 + (yy - shape_zyx[1] / 2.0) ** 2 + (zz - shape_zyx[0] / 4.0) ** 2) / 5.0e3))
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((11.0, -5.0, 3.0))  # non-trivial origin exercises metadata carry-over
    image.SetDirection(RAS_DIRECTION)  # the DM write protocol's world (ADR-0020)
    return image


def _label_volume(shape_zyx, spacing_xyz):
    """An asymmetric three-label volume (WT/TC/ET nesting) for nearest-neighbour paths."""
    array = np.zeros(shape_zyx, dtype=np.uint8)
    array[10:60, 20:100, 30:200] = 1
    array[20:50, 35:85, 60:170] = 2
    array[28:42, 48:72, 90:150] = 3
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin((-4.0, 6.0, 9.0))
    return image


# ── synthetic_domain input preparation (successor of the retired sugon copy,
#    whose ``prep_one_case`` gate this block carries over verbatim in spirit) ──────────────


def test_input_preparation_is_axis_order_correct_and_centred(tmp_path):
    """The pre-adoption bug applied the xyz target to the zyx array (pad z towards
    240, crop x down to 155); adoption must land every axis on its intended size."""
    from ctmr.application.acceptance.distribution.synthetic_domain import InputPreparator

    modality_paths = {}
    for mod_name in ("t1n", "t1c", "t2w", "t2f"):
        source = tmp_path / f"{mod_name}.nii.gz"
        sitk.WriteImage(_smooth_volume((174, 241, 241), (0.94, 0.94, 1.36)), str(source))  # v1-DM footprint
        modality_paths[mod_name] = source

    case_dir = InputPreparator().prepare_case(
        case_id="SYNTH-0002",
        challenge="MEN",
        modality_paths=modality_paths,
        output_dir=tmp_path,
    )

    array = sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir / "SYNTH-0002_0000.nii.gz")))
    assert array.shape == (155, 240, 240)  # every axis on its intended size
    assert array.any()  # the blob survives the centred z-crop (the bug's z-pad-to-240 buried it)


# ── synthetic_domain.InputPreparator (same-family axis fix) ─────────────


def test_synthetic_domain_eval_prepare_case_matches_the_instrument_adapter(tmp_path):
    from ctmr.application.acceptance.distribution.synthetic_domain import InputPreparator

    modality_paths = {}
    for mod_name in ("t1n", "t1c", "t2w", "t2f"):
        source = tmp_path / f"{mod_name}.nii.gz"
        sitk.WriteImage(_smooth_volume((174, 241, 241), (0.94, 0.94, 1.36)), str(source))
        modality_paths[mod_name] = source

    case_dir = InputPreparator().prepare_case(
        case_id="SYNTH-0003",
        challenge="METS",
        modality_paths=modality_paths,
        output_dir=tmp_path,
    )

    produced = sitk.ReadImage(str(case_dir / "SYNTH-0003_0002.nii.gz"))
    expected = InstrumentGridAdapter.continuum().align(sitk.ReadImage(str(modality_paths["t2w"])))
    assert np.array_equal(sitk.GetArrayFromImage(produced), sitk.GetArrayFromImage(expected))
    assert produced.GetSize() == (240, 240, 155)
    assert produced.GetSpacing() == (1.0, 1.0, 1.0)


def test_synthetic_domain_eval_prepare_case_handles_all_four_channels(tmp_path):
    from ctmr.application.acceptance.distribution.synthetic_domain import NNUNET_CHANNELS, InputPreparator

    modality_paths = {}
    for mod_name in NNUNET_CHANNELS.values():
        source = tmp_path / f"{mod_name}.nii.gz"
        sitk.WriteImage(_smooth_volume((100, 120, 140), (1.0, 1.25, 0.8)), str(source))
        modality_paths[mod_name] = source

    case_dir = InputPreparator().prepare_case(
        case_id="SYNTH-0004",
        challenge="PED",
        modality_paths=modality_paths,
        output_dir=tmp_path,
    )

    written = sorted(p.name for p in case_dir.glob("*"))
    assert written == [f"SYNTH-0004_{suffix}.nii.gz" for suffix in sorted(NNUNET_CHANNELS)]
    for path in case_dir.glob("*.nii.gz"):
        assert sitk.ReadImage(str(path)).GetSize() == (240, 240, 155)


# ── html_report_nifti.SliceScene (engine client, per-case TargetGrid) ─────────────


def _reference_grid():
    """The reconstructed synthetic display grid: GRID size, derived spacing, raw frame."""
    grid = sitk.Image((256, 256, 128), sitk.sitkFloat32)
    grid.SetSpacing((0.9375, 0.9375, 1.2109375))  # derived = raw_spacing * raw_size / GRID
    grid.SetOrigin((11.0, -5.0, 3.0))
    grid.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    return grid


def _direct_reference_resample(image, reference, interpolator):
    """The pre-adoption behaviour, verbatim: sitk resample onto the reference grid."""
    filter = sitk.ResampleImageFilter()
    filter.SetInterpolator(interpolator)
    filter.SetReferenceImage(reference)
    return sitk.GetArrayFromImage(filter.Execute(image))


def test_slice_scene_real_modality_matches_the_direct_reference_behaviour():
    """Frame-matched input (native BraTS volume in the raw t1n frame, size/spacing
    whose rounded extent lands exactly on GRID): the engine client output is
    bit-identical to the pre-adoption SetReferenceImage resample. This is the
    convergence-equivalence proof for the report side."""
    from ctmr.application.acceptance.distribution.html_report_nifti import SliceScene

    reference = _reference_grid()
    grid = TargetGrid(size=tuple(reference.GetSize()), spacing=tuple(reference.GetSpacing()))
    real = _smooth_volume((155, 240, 240), (1.0, 1.0, 1.0))  # native BraTS footprint, frame-matched
    real.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

    scene = SliceScene(Path("/nonexistent"), Path("/nonexistent"), {})
    assert np.array_equal(scene._to_grid(grid, real, label=False), _direct_reference_resample(real, reference, sitk.sitkBSpline))


def test_slice_scene_label_prediction_matches_the_direct_reference_behaviour():
    from ctmr.application.acceptance.distribution.html_report_nifti import SliceScene

    reference = _reference_grid()
    grid = TargetGrid(size=tuple(reference.GetSize()), spacing=tuple(reference.GetSpacing()))
    prediction = _label_volume((155, 240, 240), (1.0, 1.0, 1.0))
    prediction.SetOrigin(reference.GetOrigin())  # frame-matched: native LPS prediction
    prediction.SetDirection(reference.GetDirection())

    scene = SliceScene(Path("/nonexistent"), Path("/nonexistent"), {})
    assert np.array_equal(scene._to_grid(grid, prediction, label=True), _direct_reference_resample(prediction, reference, sitk.sitkNearestNeighbor))


def test_slice_scene_cross_frame_prediction_uses_the_engine_frame_semantics():
    """Gen-side predictions carry a different on-disk frame (identity origin from the
    DM's IDENTITY_AFFINE) than the reconstructed raw t1n display grid.

    Pre-adoption, ``SetReferenceImage`` projected such inputs by interpreting their
    own frame against the raw frame -- sampling at ``O_raw + D_raw . (s . i)``, a
    translation/reflection of the array content. ADR-0008 decision 3 makes the
    report side a plain ``GridResampler`` client (strategy injected, grid as input),
    whose frame semantics sample the *input's own* grid: the gen prediction is
    treated as the instrument-space array it is, and the overlay lands on the same
    array coordinates as the gen volume rows. The change is intentional and
    registered in the #106 PR (report display only; no measurement path, no frozen
    numbers).
    """
    from ctmr.application.acceptance.distribution.html_report_nifti import GRID, SliceScene

    prediction = _label_volume((155, 240, 240), (1.0, 1.0, 1.0))
    prediction.SetOrigin((0.0, 0.0, 0.0))  # the DM writes an identity affine: different frame
    prediction.SetDirection(np.eye(3).flatten().tolist())
    grid = TargetGrid(size=GRID, spacing=(0.9375, 0.9375, 1.2109375))

    scene = SliceScene(Path("/nonexistent"), Path("/nonexistent"), {})
    expected = sitk.GetArrayFromImage(GridResampler(sitk.sitkNearestNeighbor).resample(prediction, grid))
    assert np.array_equal(scene._to_grid(grid, prediction, label=True), expected)


def test_slice_scene_builds_the_per_case_target_grid_from_the_raw_t1n(tmp_path):
    """The display grid is a per-case TargetGrid: GRID size + derived spacing,
    reconstructed from the raw t1n spacing/size (#58's reconstruction, now as the
    ADR-0008 grid value object; the raw origin/direction no longer drives the
    sampling -- engine frame semantics sample the input's own grid)."""
    from ctmr.application.acceptance.distribution.html_report_nifti import GRID, SliceScene

    raw = _smooth_volume((155, 240, 240), (1.0, 1.0, 1.0))
    raw.SetOrigin((-23.0, 12.0, 5.0))
    raw.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(raw, str(tmp_path / "CASE-0001-t1n.nii.gz"))

    scene = SliceScene(Path("/nonexistent"), Path("/nonexistent"), {"CASE-0001": {"dir": str(tmp_path)}})
    grid = scene._synthetic_grid("CASE-0001")

    assert grid is not None
    assert isinstance(grid, TargetGrid)
    assert grid.size == GRID
    derived = tuple(raw.GetSpacing()[i] * raw.GetSize()[i] / GRID[i] for i in range(3))
    assert grid.spacing == derived


@pytest.mark.parametrize("view,centre", [("axial", 64), ("coronal", 128), ("sagittal", 128)])
def test_slice_scene_build_case_falls_back_to_the_middle_slice(tmp_path, view, centre):
    """No predictions -> the fallback slice centre must be the true middle slice of
    the display grid: VIEW_AXIS indexes the zyx array while GRID is xyz, so the
    grid axis is mirrored (the pre-fix version indexed GRID by the array axis and
    picked a wrong/wrapping slice on the empty-prediction path)."""
    from ctmr.application.acceptance.distribution.html_report_nifti import MODALITIES, SliceScene

    gen_dir = tmp_path / "gen" / "GLI" / "SYNTH-0005"
    gen_dir.mkdir(parents=True)
    for modality in MODALITIES:
        sitk.WriteImage(_smooth_volume((128, 256, 256), (1.0, 1.0, 1.0)), str(gen_dir / f"SYNTH-0005_{modality}_seed0.nii.gz"))
    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True)
    sitk.WriteImage(_smooth_volume((155, 240, 240), (1.0, 1.0, 1.0)), str(real_dir / "SYNTH-0005-t1n.nii.gz"))

    scene = SliceScene(tmp_path / "gen", tmp_path / "preds", {"SYNTH-0005": {"dir": str(real_dir), "challenge_dir": "GLI"}})
    card = scene.build_case("GLI", "SYNTH-0005", view)

    assert card["slice_label"] == f"{view}, centre={centre}"


# ── trend-feature grid (shared constant + engine composition) ──────────────────────────


def test_trend_feature_grid_is_the_pinned_240x240x160():
    """The trend-feature grid used by MrTrendFeatures: 240x240x160 @ 1mm (ADR-0008
    decisions 3-4: linear strategy injected, centred crop/pad, grid as TargetGrid).
    Lives in the torch-free engine module so the convergence gate runs on any machine."""
    from ctmr.domain.grid import TREND_FEATURE_GRID

    assert TREND_FEATURE_GRID == TargetGrid(size=(240, 240, 160), spacing=(1.0, 1.0, 1.0))


def test_generic_engine_align_produces_the_centred_linear_trend_grid():
    """The exact composition MrTrendFeatures.preprocess delegates to: linear resample
    onto the trend grid spacing, then centred crop/pad onto its size."""
    image = _smooth_volume((80, 300, 130), (1.5, 0.5, 2.0))  # mixed pad/crop axes
    grid = TargetGrid(size=(240, 240, 160), spacing=(1.0, 1.0, 1.0))
    aligned = CenterCropOrPad().crop_or_pad(GridResampler(sitk.sitkLinear).resample(image, grid), grid)
    assert aligned.GetSize() == (240, 240, 160)
    assert aligned.GetSpacing() == (1.0, 1.0, 1.0)
