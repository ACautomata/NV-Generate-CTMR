"""Behavioral tests for ctmr.infrastructure.dataio.mask_postprocess (generated-mask refinement family)."""

import json

import numpy as np
import pytest
import torch

from ctmr.infrastructure.dataio.mask_postprocess import (
    MapLabelValue,
    general_mask_generation_post_process,
    get_index_arr,
    organ_fill_by_closing,
    organ_fill_by_removed_mask,
    remap_labels,
    supress_non_largest_components,
)

pytestmark = pytest.mark.torch


def test_get_index_arr_covers_every_voxel_exactly_once():
    size = 3
    arr = get_index_arr(np.zeros((size, size, size)))
    assert arr.shape[-1] == 3
    triples = [tuple(int(c) for c in voxel) for voxel in arr.reshape(-1, 3)]
    assert len(triples) == size**3
    assert len(set(triples)) == size**3
    assert all(all(0 <= coord < size for coord in t) for t in triples)


def test_supress_non_largest_components_removes_smaller_blob():
    img = np.zeros((4, 4, 4), dtype=np.int64)
    img[0:2, 0:2, 0:2] = 1  # largest component (8 voxels)
    img[3, 3, 3] = 1  # satellite component (1 voxel)
    kept, diff = supress_non_largest_components(img, [1])
    assert kept[3, 3, 3] == 0
    assert kept[0:2, 0:2, 0:2].sum() == 8
    assert diff == 1
    # non-default replacement value honors default_val
    replaced, _ = supress_non_largest_components(img, [1], default_val=200)
    assert replaced[3, 3, 3] == 200


def test_supress_single_component_is_noop():
    img = np.zeros((4, 4, 4), dtype=np.int64)
    img[1:3, 1:3, 1:3] = 1
    kept, diff = supress_non_largest_components(img, [1])
    assert diff == 0
    assert np.array_equal(kept, img)


def test_organ_fill_by_closing_fills_center_hole():
    data = np.zeros((7, 7, 7), dtype=np.int64)
    data[2:5, 2:5, 2:5] = 7
    data[3, 3, 3] = 0  # single-voxel hole inside the organ
    filled = organ_fill_by_closing(data, target_label=7, device="cpu", close_times=1, filter_size=3)
    assert filled.dtype == np.bool_
    assert filled[3, 3, 3]
    assert filled[2:5, 2:5, 2:5].all()
    assert filled.sum() == 27


def test_organ_fill_by_removed_mask_confines_fill_to_removed_region():
    data = np.zeros((5, 5, 5), dtype=np.int64)
    data[1:4, 1:4, 1:4] = 9
    remove_mask = np.zeros((5, 5, 5), dtype=bool)
    remove_mask[2, 2, 2] = True
    filled = organ_fill_by_removed_mask(data, target_label=9, remove_mask=remove_mask, device="cpu")
    assert filled.dtype == np.bool_
    assert filled[2, 2, 2]
    assert filled.sum() == 1  # nowhere else of the removed-region is touched


def test_remap_labels_reads_json_mapping(tmp_path):
    mapping_file = tmp_path / "remap.json"
    mapping_file.write_text(json.dumps({"tumor": [26, 100], "lung": [23, 90]}))
    mask = torch.tensor([[[[26.0]], [[23.0]], [[1.0]]]])  # shape [1,C,H,W]
    remapped = remap_labels(mask, str(mapping_file))
    assert remapped.dtype == torch.long
    assert remapped.shape == mask.shape
    assert remapped.flatten().tolist() == [100, 90, 1]


def test_map_label_value_numpy_backend():
    mapper = MapLabelValue(orig_labels=[3.0, 2.0], target_labels=[10.0, 11.0], dtype=np.float32)
    out = mapper(np.array([[3.0, 2.0, 7.0]]))
    assert out.tolist() == [[10.0, 11.0, 7.0]]
    assert out.dtype == np.float32


def test_map_label_value_torch_backend():
    mapper = MapLabelValue(orig_labels=[3, 2], target_labels=[10, 11], dtype=torch.long)
    out = mapper(torch.tensor([[3, 2, 7]]))
    assert out.tolist() == [[10, 11, 7]]
    assert out.dtype == torch.long


def test_map_label_value_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        MapLabelValue(orig_labels=[1, 2], target_labels=[3])


def test_general_mask_generation_post_process_preserves_organ_external_tumor_and_stays_nonnegative():
    # bone lesion (128) is not an organ-closing target and sticks out of nothing:
    # it must survive processing end to end.
    volume = np.full((12, 12, 12), 200, dtype=np.int64)
    volume[4:7, 4:7, 4:7] = 1  # liver
    volume[5, 5, 5] = 128  # bone lesion voxel inside the body
    processed = general_mask_generation_post_process(volume.copy(), target_tumor_label=128, device="cpu")
    assert processed.shape == volume.shape
    assert processed.min() >= 0
    assert (processed == 128).any(), "target tumor must survive post-processing"


def test_general_mask_generation_post_process_pins_upstream_organ_closing_reabsorption():
    """Upstream behavior pin: the organ-closing pass rewrites tumor voxels fully enclosed
    inside an organ back to the organ label (a voxel of liver tumor 26 sitting strictly
    inside liver 1 is absorbed into 1). This is the verbatim-migrated behavior; whether it
    is desirable is an expand-phase question, not part of this migration."""
    volume = np.full((12, 12, 12), 200, dtype=np.int64)
    volume[4:7, 4:7, 4:7] = 1
    volume[5, 5, 5] = 26
    processed = general_mask_generation_post_process(volume.copy(), target_tumor_label=26, device="cpu")
    assert (processed == 26).sum() == 0
    assert (processed == 1).sum() == 27
