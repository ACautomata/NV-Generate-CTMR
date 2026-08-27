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


"""Behaviour gates for ctmr.infrastructure.dataio.morphology (#132).

The two MONAI dilate/erode tensor wrappers lifted verbatim out of
scripts/utils.py, exercised at their documented shapes (plain [M,N,P] /
[M,N] masks -- the wrappers add the channel+batch dims themselves):
shrink/grow amounts, pad_value border behaviour and dtype passthrough.
Torch-level, CPU-only.
"""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402  (importorskip must precede the torch-dependent import)

from ctmr.infrastructure.dataio.morphology import (  # noqa: E402  (importorskip must precede the torch-dependent import)
    dilate_one_img,
    erode_one_img,
)


def _solid_box(n=7, inner=5):
    box = torch.zeros(n, n, n)
    o = (n - inner) // 2
    box[o : o + inner, o : o + inner, o : o + inner] = 1.0
    return box


def test_erode_shrinks_solid_box_by_one_shell_per_face():
    eroded = erode_one_img(_solid_box(inner=5), filter_size=3)
    assert int(eroded.sum()) == 27  # the strict interior survives


def test_dilate_grows_seed_to_full_neighborhood():
    seed = torch.zeros(5, 5, 5)
    seed[2, 2, 2] = 1.0
    grown = dilate_one_img(seed, filter_size=3)
    assert int(grown.sum()) == 27  # 3x3x3 block around the seed
    assert float(grown[2, 2, 2]) == 1.0 and float(grown[3, 3, 3]) == 1.0 and float(grown[0, 0, 0]) == 0.0


def test_pad_value_controls_border_behaviour_on_erode():
    # erode's padding semantic: voxels whose full window cannot be observed
    # take the pad value into account. A constant plane stays intact under
    # pad_value=1 and shrinks to its 2x2 interior under pad_value=0.
    plane = torch.ones(4, 4)
    assert float(erode_one_img(plane, filter_size=3, pad_value=1.0).sum()) == 16.0
    assert float(erode_one_img(plane, filter_size=3, pad_value=0.0).sum()) == 4.0


def test_output_preserves_shape_and_handles_uint8_input():
    cube = (_solid_box() * 255).to(torch.uint8)
    out = dilate_one_img(erode_one_img(cube))
    assert out.shape == cube.shape
    assert out.dtype == torch.float32
