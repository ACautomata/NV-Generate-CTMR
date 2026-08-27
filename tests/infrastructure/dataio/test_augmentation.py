# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Behaviour gates for ctmr.infrastructure.dataio.augmentation (#132).

Verbatim lift of scripts/augmentation.py (single-line import redirect to the
package's morphology module). Covers the torch-morphology primitives
(dilate3d/erode3d, pinned with their verbatim padding quirks), label remap
surgery, the two tumor-removal strategies, finalize/augment-only acceptance
loops including the no-fit error path and the uint8 closing cap (kept small so
the .to(torch.uint8) branch stays representative), the zoom-style body jitter,
and a dispatcher-routing gate for the BraTS 401-403 branch. The per-organ
tumor functions hardwire ``.cuda()`` and stay outside CPU gate scope.
Torch-level, CPU-only.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip must precede heavy imports)

from ctmr.infrastructure.dataio.augmentation import (  # noqa: E402
    MAX_COUNT,
    augmentation,
    augmentation_body,
    augmentation_tumor_only,
    dilate3d,
    erode3d,
    finalize_tumor_mask,
    remap_labels,
    remove_tumors,
    remove_tumors_majority_vote,
)


def _blob_tensor(labels):
    vol = torch.zeros(1, 8, 8, 8)
    for val, box in labels:
        x0, x1, y0, y1, z0, z1 = box
        vol[0, x0:x1, y0:y1, z0:z1] = val
    return vol


def test_erode3d_shrinks_box_by_full_window_interior():
    box = torch.zeros(8, 8, 8)
    box[1:7, 1:7, 1:7] = 1.0
    assert int(erode3d(box).sum()) == 64  # strict min-window: one shell per face gone


def test_dilate3d_inflates_by_padding_quirk_to_full_volume():
    seed = torch.zeros(5, 5, 5)
    seed[2, 2, 2] = 1.0
    # dilate3d pads with zeros and thresholds at any overlap, so the grown set
    # reaches the padded frame -- pinned as-is (verbatim oddity, not a bug fix).
    assert int(dilate3d(seed).sum()) == 125


def test_remap_labels_moves_only_mapped_ids_on_a_clone():
    vol = torch.tensor([[[1, 26], [23, 200]]])
    out = remap_labels(vol, {26: 1, 23: 62})
    assert torch.equal(out, torch.tensor([[[1, 1], [62, 200]]]))
    assert torch.equal(vol, torch.tensor([[[1, 26], [23, 200]]]))  # input untouched


def test_majority_vote_replaces_tumor_with_ring_organ_mode():
    vol = _blob_tensor([(28, (0, 8, 0, 4, 0, 8)), (401, (2, 6, 1, 3, 2, 6))])
    tumor = (vol == 401).long()
    out = remove_tumors_majority_vote(tumor, vol.clone())
    assert not bool((out == 401).any())
    assert bool((out[tumor.bool()] == 28).all())


def test_majority_vote_falls_back_when_ring_has_no_listed_organs():
    # tumor fully wrapped by an UNLISTED shell -> ring filter empties -> the
    # fallback votes the most frequent LISTED organ across the volume
    vol = _blob_tensor([(29, (0, 3, 0, 3, 0, 3)), (100, (0, 7, 4, 7, 0, 7)), (402, (2, 5, 5, 7, 2, 5))])
    tumor = (vol == 402).long()
    out = remove_tumors_majority_vote(tumor, vol.clone(), organ_label_lists=(28, 29))
    assert bool((out[tumor.bool()] == 29).all())


def test_remove_tumors_static_remap_and_pseudo_replacement():
    vol = _blob_tensor([(1, (0, 4, 0, 4, 0, 4)), (23, (4, 6, 4, 6, 4, 6)), (116, (6, 7, 6, 7, 6, 7))])
    pseudo = vol.clone()
    pseudo[vol == 23] = 200
    out = remove_tumors(vol.clone(), pseudo_labels=pseudo)
    assert not bool((out == 23).any())
    assert bool((out[pseudo == 200] == 200).all())  # lesion takes its pseudo label back
    assert bool((out[vol == 116] == 14).all())  # right kidney cyst statically folded into kidney 14


def test_finalize_accepts_and_closes_binary_mask_above_threshold():
    tumor = torch.zeros(1, 8, 8, 8)
    tumor[0, 3:5, 3:5, 3:5] = 1
    organ = torch.ones(1, 8, 8, 8)
    out = finalize_tumor_mask(tumor, organ, threshold_tumor_size=7.0)
    assert out is not None
    assert int(out.sum()) >= 8 and float(out.max()) == 1.0


def test_finalize_returns_none_below_threshold():
    tumor = torch.zeros(1, 8, 8, 8)
    tumor[0, 3:5, 3:5, 3:5] = 1
    organ = torch.zeros_like(tumor)  # veto the intersection entirely
    assert finalize_tumor_mask(tumor, organ, threshold_tumor_size=1.0) is None


class _ShellAug:
    """Stands in for a MONAI random transform: callable returning .as_tensor()."""

    def __call__(self, mask, spatial_size=None):
        return SimpleNamespace(as_tensor=lambda: mask)


def test_augmentation_tumor_only_keeps_labeled_tumor_in_organ():
    whole = _blob_tensor([(22, (0, 8, 0, 8, 0, 2)), (2, (3, 6, 3, 6, 3, 6))])  # label 2 fits the uint8 close
    organ = (whole > 0).long()
    out = augmentation_tumor_only(whole.clone(), organ, _ShellAug(), tumor_label=[2], min_tumor_size_ratio=0.5)
    assert int((out == 2).sum()) > 0  # identity aug keeps every tumor voxel


def test_augmentation_tumor_only_raises_when_no_fit_within_retry_budget():
    whole = _blob_tensor([(2, (3, 6, 3, 6, 3, 6))])
    empty_organ = torch.zeros_like(whole)  # veto every attempt -> exhaust MAX_COUNT retries
    with pytest.raises(ValueError, match="inside organ"):
        augmentation_tumor_only(whole, empty_organ, _ShellAug(), tumor_label=[2], min_tumor_size_ratio=1e9)
    assert MAX_COUNT >= 1


def test_body_zoom_is_a_near_identity_shape_preserving_jitter():
    vol = _blob_tensor([(22, (0, 8, 0, 8, 0, 8))])
    out = augmentation_body(vol.clone(), random_seed=0)
    assert out.shape == vol.shape
    assert set(out.unique().tolist()) <= {0, 22}


def test_dispatcher_routes_brats_branch_to_the_multi_label_augmenter(monkeypatch):
    # the dispatcher's overlay writes require an integer label volume (its own
    # .long() cast upstream of the writeback), so gate on the real contract shape
    pt_nda = _blob_tensor([(22, (0, 8, 0, 8, 0, 2)), (401, (3, 6, 3, 6, 3, 6))]).to(torch.long)

    import ctmr.infrastructure.dataio.augmentation as aug_mod

    calls = []

    def spy(tumor_mask_, organ_mask, aug_transform, spatial_size=None, tumor_label=None, min_tumor_size_ratio=0.8):
        calls.append(list(tumor_label))
        return tumor_mask_.clone()  # same integer dtype; no fit-fail loop under the stub

    monkeypatch.setattr(aug_mod, "augmentation_tumor_only", spy)

    out = augmentation(pt_nda.clone(), output_size=[8, 8, 8], random_seed=3)
    assert calls == [[401, 402, 403]]  # brats family routed to augmentation_tumor_only once
    assert out.shape == pt_nda.shape
