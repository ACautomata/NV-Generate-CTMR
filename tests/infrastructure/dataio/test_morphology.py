"""CPU numerical behavior tests for ctmr.infrastructure.dataio.morphology (migrated verbatim from scripts/utils)."""

import pytest
import torch

from ctmr.infrastructure.dataio.morphology import dilate_one_img, erode_one_img

pytestmark = pytest.mark.torch


def test_constant_inputs_are_fixed_points():
    solid = torch.ones(8, 8, 8)
    empty = torch.zeros(8, 8, 8)
    for fn, pad in ((erode_one_img, 1.0), (dilate_one_img, 0.0)):
        out_solid = fn(solid.clone(), filter_size=3, pad_value=pad)
        out_empty = fn(empty.clone(), filter_size=3, pad_value=pad)
        assert torch.equal(out_solid, solid)
        assert torch.equal(out_empty, empty)


def test_dilate_then_erode_is_close_to_identity_for_centered_seed():
    seed = torch.zeros(16, 16, 16)
    seed[8, 8, 8] = 1.0
    grown = dilate_one_img(seed, filter_size=3, pad_value=0.0)
    # a cubic kernel of radius 1 grows the single voxel to a 3x3x3 cube
    assert int(grown.sum()) == 27
    shrunk = erode_one_img(grown, filter_size=3, pad_value=1.0)
    assert int(shrunk.sum()) >= 1


def test_erode_shrinks_and_keeps_shape():
    blob = torch.zeros(12, 12)
    blob[3:9, 3:9] = 1.0
    eroded = erode_one_img(blob, filter_size=3, pad_value=1.0)
    assert eroded.shape == blob.shape
    assert int(eroded.sum()) < int(blob.sum())
    assert eroded[5, 5] == 1.0  # core survives
    assert eroded[3, 3] == 0.0  # corner does not
