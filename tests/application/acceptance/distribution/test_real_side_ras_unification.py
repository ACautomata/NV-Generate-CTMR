"""The instrument-chain real-side RAS unification gate (ADR-0020, #314).

The pre-#314 world straddled two direction worlds: the real side passed through
in its native orientation (BraTS 2023 mixes ~89% LPS with ~11% RAS codings)
while the generated side was unconditionally x/y-flipped onto LPS. Every
RAS-coded real case therefore measured against a mirrored instrument input --
the misalignment class #314 retires.

This gate assembles the SAME physical phantom three times -- real side coded in
LPS, real side coded in RAS, generated side on the DM output grid in RAS --
through the public ``ObservationInputWriter`` plan surface, and requires all
three instrument inputs to land on one array world: both real codings
bit-identical, and the generated tumour centroid aligned with the real one
(the old world puts them ~180 voxels apart on the x/y axes). A non-axis-aligned
real coding and a non-RAS generated coding are loud assembly failures.

SimpleITK unit level, any machine, no cluster, no external data (ADR-0013).
"""

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.measurement_run import ObservationInputWriter
from ctmr.domain.orientation import NotRasWorldError, RasOrientation

RAS_DIRECTION = RasOrientation.RAS_DIRECTION

CHANNELS = ("0000", "0001", "0002", "0003")
TUMOUR_CENTRE_MM = (-60.0, 30.0, 60.0)  # deliberately off-centre: a centred marker cannot see a flip
TUMOUR_SIGMA_MM = 8.0


def _phantom(world_xyz):
    """A brain blob + an off-centre tumour Gaussian, evaluated at world points (mm)."""
    brain = np.exp(-(((world_xyz[0] - 5.0) ** 2 + (world_xyz[1] - 5.0) ** 2 + (world_xyz[2] - 77.0) ** 2) / (2.0 * 55.0**2)))
    tumour = np.exp(-(sum((world_xyz[axis] - TUMOUR_CENTRE_MM[axis]) ** 2 for axis in range(3)) / (2.0 * TUMOUR_SIGMA_MM**2)))
    return 100.0 * brain + 50.0 * tumour


def _tumour_centroid(array, threshold_ratio=0.5):
    """The brightest-structure centroid in zyx array coordinates."""
    threshold = threshold_ratio * array.max()
    indices = np.argwhere(array > threshold).astype(float)
    return indices.mean(axis=0)


def _write_real_volume(path, direction, origin, array):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, str(path))


def _real_arrays(origin=(-115.0, -115.0, 0.0), size=(240, 240, 155)):
    """The physical phantom sampled on the instrument-domain grid, in world coordinates."""
    zz, yy, xx = np.mgrid[0 : size[2], 0 : size[1], 0 : size[0]]  # zyx
    world = (
        origin[0] + xx.astype(float),
        origin[1] + yy.astype(float),
        origin[2] + zz.astype(float),
    )
    return _phantom(world).astype(np.float32)


def _gen_volume():
    """The same physical phantom on the DM output grid (256^2x128 @ 0.94/0.94/1.36, RAS).

    The grid origin is chosen so the DM field of view is centred on the real-side
    grid: the real centre is (4.5, 4.5, 77) mm, and under the RAS direction the
    DM centre voxel sits at origin + (-119.85, -119.85, 86.36).
    """
    origin = (4.5 + 127.5 * 0.94, 4.5 + 127.5 * 0.94, 77.0 - 63.5 * 1.36)
    spacing = (0.94, 0.94, 1.36)
    zz, yy, xx = np.mgrid[0:128, 0:256, 0:256]  # zyx
    world = (
        origin[0] - xx.astype(float) * spacing[0],  # RAS: array +x runs toward physical -x
        origin[1] - yy.astype(float) * spacing[1],
        origin[2] + zz.astype(float) * spacing[2],
    )
    array = _phantom(world).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(RAS_DIRECTION)
    return image


def _plan(lps_path, ras_path, gen_path):
    def observation(obs_id, side, source):
        return {
            "obs_id": obs_id,
            "challenge": "GLI",
            "case": "CASE0001",
            "side": side,
            "anchor": None,
            "channels": {suffix: str(source) for suffix in CHANNELS},
            "condition_mask": None,
        }

    return {
        "observations": [
            observation("CASE0001__real_lps", "real", lps_path),
            observation("CASE0001__real_ras", "real", ras_path),
            observation("CASE0001__gen", "gen", gen_path),
        ]
    }


def test_assembled_real_codings_share_one_array_world_and_align_with_gen(tmp_path):
    array = _real_arrays()
    mirrored = np.ascontiguousarray(np.flip(np.flip(array, axis=2), axis=1))
    # The RAS coding of the same physical volume: x/y-mirrored array; the origin
    # moves to the mirrored low corner (-115-239, -115-239, 0) = (-354, -354, 0).
    lps_path = tmp_path / "real_lps.nii.gz"
    ras_path = tmp_path / "real_ras.nii.gz"
    gen_path = tmp_path / "gen.nii.gz"
    _write_real_volume(lps_path, np.eye(3).flatten().tolist(), (-115.0, -115.0, 0.0), array)
    _write_real_volume(ras_path, RAS_DIRECTION, (-354.0, -354.0, 0.0), mirrored)
    sitk.WriteImage(_gen_volume(), str(gen_path))

    inputs = ObservationInputWriter(tmp_path).write_all(_plan(lps_path, ras_path, gen_path))

    def read(obs_id):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(inputs / "GLI" / f"{obs_id}_0000.nii.gz")))

    assembled_lps, assembled_ras, assembled_gen = read("CASE0001__real_lps"), read("CASE0001__real_ras"), read("CASE0001__gen")
    # one array world: both real codings unify bit-identically...
    assert np.array_equal(assembled_lps, assembled_ras)
    # ...and the generated side lands on the same world: tumour centroids within
    # a resampling tolerance (the pre-#314 world puts them ~180 voxels apart).
    assert np.abs(_tumour_centroid(assembled_gen) - _tumour_centroid(assembled_lps)).max() < 3.0


def test_assembly_rejects_a_non_axis_aligned_real_coding_loudly(tmp_path):
    oblique = np.eye(3)
    oblique[0, 1] = 0.3
    oblique[1, 0] = -0.3
    source = tmp_path / "oblique.nii.gz"
    _write_real_volume(source, tuple(oblique.flatten().tolist()), (-115.0, -115.0, 0.0), _real_arrays())
    gen_source = tmp_path / "gen_ok.nii.gz"
    sitk.WriteImage(_gen_volume(), str(gen_source))  # the gen slot stays valid: only the real-side guard can fire

    with pytest.raises(NotRasWorldError, match="axis-aligned"):
        ObservationInputWriter(tmp_path).write_all(_plan(source, source, gen_source))


def test_assembly_rejects_a_non_ras_generated_coding_loudly(tmp_path):
    """The generated side asserts the RAS write protocol (#249): a non-RAS declared
    volume is an upstream protocol break, not something to silently re-orient."""
    lps_source = tmp_path / "gen_lps.nii.gz"
    array = _real_arrays()
    _write_real_volume(lps_source, np.eye(3).flatten().tolist(), (-115.0, -115.0, 0.0), array)

    with pytest.raises(NotRasWorldError):
        ObservationInputWriter(tmp_path).write_all(
            _plan(lps_source, lps_source, lps_source)  # the gen slot fed an LPS-declared volume
        )
