"""Direction-world gates for the RAS orientation service (ADR-0020, #314).

Pins the single direction-unification point: every instrument-chain input --
generated side, real side, condition labels -- lands on the RAS direction
world declared by the NVIDIA upstream convention (the initial commit's
``Orientationd(axcodes="RAS")`` chain). The service must

- accept already-RAS volumes unchanged (idempotent);
- unify any axis-aligned LPS-family coding onto RAS with voxel-physical
  correspondence preserved (the affine-driven resample guarantee);
- reject oblique directions loudly -- the axis-aligned boundary of the
  permute/flip semantics, never a silent approximation;
- expose the written-contract assertion for the generated side (require_ras).

SimpleITK unit level, any machine, no cluster, no external data (ADR-0013).
"""

import itertools

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.domain.orientation import NotRasWorldError, RasOrientation

RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
MARKER = (1, 2, 3)  # deliberately off-centre: centre-symmetric markers cannot see a flip
MARKER_VALUE = 7.0


def _marker_array():
    array = np.zeros((6, 8, 10), dtype=np.float32)  # zyx
    array[MARKER] = MARKER_VALUE
    return array


def _volume(direction, array=None, spacing=(1.0, 1.0, 2.0), origin=(-115.0, 120.0, 5.0)):
    image = sitk.GetImageFromArray(_marker_array() if array is None else array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def _marker_physical_point(image, array):
    """The marker's physical point, read off the array + the image's own frame."""
    z, y, x = (int(v) for v in np.argwhere(array == MARKER_VALUE)[0])
    return np.asarray(image.TransformIndexToPhysicalPoint((x, y, z)))


def _axis_aligned_directions():
    """All 48 axis-aligned direction codings: every axis permutation x flip combination."""
    directions = []
    for permutation in itertools.permutations(range(3)):
        for flips in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, column in enumerate(permutation):
                matrix[row, column] = flips[row]
            directions.append(tuple(matrix.flatten().tolist()))
    return directions


def test_to_ras_is_idempotent_for_a_ras_volume():
    service = RasOrientation()
    image = _volume(RAS_DIRECTION)
    array = sitk.GetArrayFromImage(image)

    oriented = service.to_ras(image)

    assert np.allclose(oriented.GetDirection(), RAS_DIRECTION)
    assert np.array_equal(sitk.GetArrayFromImage(oriented), array)  # no-op, not a copy world


def test_to_ras_unifies_an_lps_volume_onto_ras_with_physical_correspondence():
    """The real-side LPS coding (BraTS majority): same physical volume as the RAS
    expression of the same array -- the unification must land exactly there."""
    service = RasOrientation()
    ras_view = _volume(RAS_DIRECTION)
    mirrored = np.ascontiguousarray(np.flip(np.flip(_marker_array(), axis=2), axis=1))
    # Same physical volume expressed in LPS: the RAS view spans x in [-124,-115],
    # y in [113,120], z in [5,15] mm (origin + mirrored direction over the extent),
    # so the LPS expression's origin is the low corner (-124, 113, 5) and the array
    # is the x/y mirror of the RAS array.
    lps_view = _volume(np.eye(3).flatten().tolist(), array=mirrored, origin=(-124.0, 113.0, 5.0))
    reference_point = _marker_physical_point(ras_view, sitk.GetArrayFromImage(ras_view))
    source_point = _marker_physical_point(lps_view, sitk.GetArrayFromImage(lps_view))
    assert np.allclose(reference_point, source_point)  # fixture sanity: same physical volume

    oriented = service.to_ras(lps_view)
    array = sitk.GetArrayFromImage(oriented)

    assert np.allclose(oriented.GetDirection(), RAS_DIRECTION)
    assert array[MARKER] == MARKER_VALUE  # the array lands on the RAS expression of the volume
    assert np.allclose(_marker_physical_point(oriented, array), reference_point)


@pytest.mark.parametrize("direction", _axis_aligned_directions())
def test_to_ras_unifies_every_axis_aligned_coding_onto_ras(direction):
    service = RasOrientation()
    image = _volume(direction)
    source_point = _marker_physical_point(image, sitk.GetArrayFromImage(image))

    oriented = service.to_ras(image)
    array = sitk.GetArrayFromImage(oriented)

    assert np.allclose(oriented.GetDirection(), RAS_DIRECTION)
    assert np.argwhere(array == MARKER_VALUE).shape[0] == 1  # exactly one marker, the RAS expression
    assert np.allclose(_marker_physical_point(oriented, array), source_point)


def test_to_ras_rejects_an_oblique_direction_loudly():
    """The axis-aligned boundary of the permute/flip semantics: oblique input is a
    loud failure, never a silent approximation (SimpleITK's own filter would not raise)."""
    service = RasOrientation()
    oblique = np.eye(3)
    oblique[0, 1] = 0.3
    oblique[1, 0] = -0.3
    image = _volume(tuple(oblique.flatten().tolist()))

    with pytest.raises(NotRasWorldError, match="axis-aligned"):
        service.to_ras(image)


def test_require_ras_accepts_the_ras_world_and_rejects_everything_else():
    service = RasOrientation()
    image = _volume(RAS_DIRECTION)
    assert service.require_ras(image) is image  # assert, not re-orient: the volume passes through unchanged

    with pytest.raises(NotRasWorldError, match="RAS"):
        service.require_ras(_volume(np.eye(3).flatten().tolist()))  # LPS-declared volume
