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


"""Behaviour gates for ctmr.infrastructure.dataio.plotting (#132).

Verbatim lift of scripts/utils_plot.py (renamed from utils_plot): center-slice
localisation on binary masks, label colorisation to uint8 RGB (asserted
permutation-invariant -- MONAI's AsDiscrete axis contract differs across
versions, so tests pin palette purity and distinct-colour counts rather than a
specific channel order), the 2D slice extractor with its axis validation at
the documented [B,C,H,W,D] rank, integer-centred zero padding, and the XYZ
triptych assembly. matplotlib.pyplot stays unexercised. monai+torch+matplotlib
level, CPU-only.
"""

import pytest

pytest.importorskip("torch")

import numpy as np  # noqa: E402  (importorskip must precede heavy imports)
import torch  # noqa: E402

from ctmr.infrastructure.dataio.plotting import (  # noqa: E402
    find_label_center_loc,
    get_xyz_plot,
    normalize_label_to_uint8,
    to_shape,
    visualize_one_slice_in_3d,
)

PRIMARIES = {(255, 0, 0), (0, 255, 0), (0, 0, 255)}


def _colorize(n_label):
    # identity palette: label k renders on channel k%3
    return torch.eye(3, n_label).reshape(3, n_label, 1, 1)


def test_find_label_center_loc_midpoints_per_dimension():
    mask = torch.zeros(6, 6, 6)
    mask[2, :, :] = 1
    centers = find_label_center_loc(mask)
    assert [int(c) for c in centers] == [2, 3, 3]


def test_find_label_center_loc_all_none_for_empty_mask():
    assert find_label_center_loc(torch.zeros(4, 4, 4)) == [None, None, None]


def test_normalize_label_to_uint8_identity_palette_purity():
    label = torch.tensor([[[[0.0, 1.0], [2.0, 0.0]]]])  # [N,C,H,W] as visualize_one_slice hands over
    img = normalize_label_to_uint8(_colorize(3), label, n_label=3)
    assert img.shape == (2, 2, 3) and img.dtype == np.uint8
    colors = {tuple(int(v) for v in px) for px in img.reshape(-1, 3)}
    # every pixel -- background included -- renders as a pure primary: value 0 is
    # still class index 0 on the identity palette, so no black exists by design.
    assert colors == PRIMARIES


def test_visualize_one_slice_returns_middle_by_default_and_validates_axis():
    vol = torch.zeros(1, 1, 4, 4, 6)  # [B,C,H,W,D] per the module's slice contract
    vol[..., :, :, 3] = 5.0  # distinctive middle slice along the last spatial axis
    grey = visualize_one_slice_in_3d(vol, axis=2, mask_bool=False)
    assert grey.shape == (4, 4, 3)
    assert float(grey[..., 0].max()) == 5.0
    flagged = visualize_one_slice_in_3d(vol, axis=2, center=0, mask_bool=False)
    assert float(flagged[..., 0].max()) == 0.0
    with pytest.raises(ValueError, match="axis should be"):
        visualize_one_slice_in_3d(vol, axis=5, center=2)


def test_get_xyz_triplex_pads_panels_to_common_extent():
    vol = torch.ones(1, 10, 8, 6)  # [C,H,W,D]; the triptych adds the batch dim itself
    tri = get_xyz_plot(vol, center_loc_axis=[5, 4, 3], mask_bool=True, n_label=2, colorize=_colorize(2))
    longest = max(10, 8, 6)
    assert tri.shape == (longest, 3 * longest, 3)


def test_to_shape_pads_centred_with_remainder_on_the_end():
    a = np.ones((2, 3, 4), dtype=np.int64)
    out = to_shape(a, (5, 3, 9))
    assert out.shape == (5, 3, 9)
    assert int(out.sum()) == a.size  # pure padding, no value drift
    nz = np.nonzero(out)[0]
    assert (int(nz.min()), int(nz.max())) == (1, 2)  # depth-2 core centred with remainder to the end
