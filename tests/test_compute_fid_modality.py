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

"""Tests for the ``--modality`` switch in scripts.compute_fid_2-5d_ct (ticket T4, issue #20).

The ``ct`` row anchors the pre-change constants (regression safety). The ``mr``
row has NO fixed clip — MR intensity is scanner/sequence dependent and real
volumes can far exceed 1000 — instead both datasets are mapped with the
training pipeline's dynamic (0, 99.5) percentile window into the
generator-side (0, 1000) output domain. Behavior tests replay the exact
transform calls from ``main()`` fed by the shared constant table, including
the intensity-before-padding ordering (padding voxels must not dilute the
percentile statistics). The same single Compose instance serves both the real
and the synth loader, which is what keeps FID comparable.
"""

import importlib

import pytest
import torch
from monai.transforms import CenterSpatialCropd, Compose, ScaleIntensityRanged, ScaleIntensityRangePercentilesd, SpatialPadd

# Module name carries a hyphen, so import it programmatically.
compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")


@pytest.fixture
def modality_preprocessing() -> dict:
    return compute_fid.MODALITY_PREPROCESSING


@pytest.fixture
def build_preprocessing(modality_preprocessing: dict) -> object:
    """Factory: build the crop → intensity → pad chain exactly as ``main()`` does, fed by the shared constant table."""

    def _build(modality: str, target_shape: tuple[int, int, int] = (16, 16, 16)) -> Compose:
        preprocessing = modality_preprocessing[modality]
        transform_list = [CenterSpatialCropd(keys=["image"], roi_size=target_shape)]
        if preprocessing.percentile_range is not None:
            lower, upper = preprocessing.percentile_range
            b_min, b_max = preprocessing.intensity_range
            transform_list.append(
                ScaleIntensityRangePercentilesd(keys=["image"], lower=lower, upper=upper, b_min=b_min, b_max=b_max, clip=False)
            )
        else:
            a_min, a_max = preprocessing.intensity_range
            transform_list.append(
                ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=a_min, b_max=a_max, clip=True)
            )
        transform_list.append(
            SpatialPadd(keys=["image"], spatial_size=target_shape, mode="constant", value=preprocessing.padding_value)
        )
        return Compose(transform_list)

    return _build


class TestModalityPreprocessingTable:
    def test_ct_row_matches_prechange_constants(self, modality_preprocessing: dict) -> None:
        ct = modality_preprocessing["ct"]

        assert ct.padding_value == -1000
        assert ct.intensity_range == (-1000, 1000)
        assert ct.percentile_range is None  # fixed HU clip, as before this ticket

    def test_mr_row_matches_spec_constants(self, modality_preprocessing: dict) -> None:
        mr = modality_preprocessing["mr"]

        assert mr.padding_value == 0
        assert mr.intensity_range == (0, 1000)  # generator-side output domain
        assert mr.percentile_range == (0.0, 99.5)  # training pipeline's dynamic window

    def test_table_knows_only_ct_and_mr(self, modality_preprocessing: dict) -> None:
        assert sorted(modality_preprocessing) == ["ct", "mr"]


class TestModalityPaddingBehavior:
    """The padding constant differs per modality and must come from the shared table."""

    def test_ct_pads_with_hu_air_value(self, modality_preprocessing: dict) -> None:
        pad = SpatialPadd(
            keys=["image"],
            spatial_size=(16, 16, 16),
            mode="constant",
            value=modality_preprocessing["ct"].padding_value,
        )

        out = pad({"image": torch.full((1, 8, 8, 8), 500.0)})["image"]

        # SpatialPadd pads symmetrically: the 8^3 volume lands at [4:12] per axis.
        assert out[0, 0, 0, 0] == -1000  # padding region filled with CT air
        assert out[0, 7, 7, 7] == 500  # original voxels untouched
        assert out[0, 15, 15, 15] == -1000

    def test_mr_pads_with_zero(self, modality_preprocessing: dict) -> None:
        pad = SpatialPadd(
            keys=["image"],
            spatial_size=(16, 16, 16),
            mode="constant",
            value=modality_preprocessing["mr"].padding_value,
        )

        out = pad({"image": torch.full((1, 8, 8, 8), 500.0)})["image"]

        # SpatialPadd pads symmetrically: the 8^3 volume lands at [4:12] per axis.
        assert out[0, 0, 0, 0] == 0  # padding region filled with MR background
        assert out[0, 7, 7, 7] == 500  # original voxels untouched
        assert out[0, 15, 15, 15] == 0


class TestModalityIntensityBehavior:
    """Intensity windowing per modality, both fed by the shared constant table."""

    def test_ct_clips_to_hu_range_keeping_negative_hu(self, modality_preprocessing: dict) -> None:
        a_min, a_max = modality_preprocessing["ct"].intensity_range
        clip = ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=a_min, b_max=a_max, clip=True)

        volume = torch.tensor([[[[-500.0, 500.0], [2000.0, -2000.0]]]])
        out = clip({"image": volume})["image"]

        assert out[0, 0, 0, 0] == -500  # legal HU preserved
        assert out[0, 0, 0, 1] == 500
        assert out[0, 0, 1, 0] == 1000  # over-range clipped to HU max
        assert out[0, 0, 1, 1] == -1000  # below-range clipped to HU min

    def test_mr_maps_percentile_window_without_clipping_high_intensities(self, modality_preprocessing: dict) -> None:
        """Real MR volumes can far exceed 1000 — the mapping must keep the dynamic tail."""
        mr = modality_preprocessing["mr"]
        lower, upper = mr.percentile_range
        b_min, b_max = mr.intensity_range
        mapping = ScaleIntensityRangePercentilesd(keys=["image"], lower=lower, upper=upper, b_min=b_min, b_max=b_max, clip=False)

        # 1000 voxels: one at 0 (percentile floor), 997 at 500 (the 99.5th
        # percentile), two at 2000 (above-window tail). With clip=False the
        # linear map (x - 0) / 500 * 1000 = 2x sends the tail to 4000, NOT 1000.
        volume = torch.full((1, 10, 10, 10), 500.0)
        volume[0, 0, 0, 0] = 0.0
        volume[0, 0, 0, 1] = 2000.0
        volume[0, 0, 0, 2] = 2000.0

        out = mapping({"image": volume})["image"]

        assert out[0, 0, 0, 0] == 0  # percentile floor maps to b_min
        assert out[0, 5, 5, 5] == 1000  # the 99.5th percentile maps to b_max
        assert out[0, 0, 0, 1] == 4000  # above-window intensity is preserved, not clipped
        assert out.max() == 4000


class TestIntensityRunsBeforePadding:
    """Padding voxels must not dilute the percentile statistics.

    An 8^3 content in a 16^3 target (12.5% content share) whose values make
    the content's 99.5th percentile (2000) differ from the padded volume's
    effective percentile (1000): the two orderings give different interior
    mappings, so the assertion pins the ordering.
    """

    def test_mr_percentile_ignores_padding_voxels(self, build_preprocessing: object) -> None:
        volume = torch.zeros(1, 8, 8, 8)
        volume[0, 0, 0, 0] = 0.0
        volume[0, 1:, ...] = 500.0  # 504 voxels at 500
        volume[0, 0, 0, 1:5] = 1000.0  # 4 voxels at 1000
        volume[0, 0, 1, 1:12] = 2000.0  # 11 voxels at 2000 (top tail)

        out = build_preprocessing("mr")({"image": volume})["image"]

        # Content-only p99.5 = 2000 → linear map x/2: interior 500 becomes 250.
        # Pad-first ordering would compute p99.5 = 1000 on the padded volume
        # (12.5% content share) and map interior 500 to 500.
        assert out[0, 4 + 4, 4 + 4, 4 + 4] == 250  # volume[0, 4, 4, 4] sits at 500
        assert out[0, 4, 4, 5] == 500  # a 1000 voxel at x/2 (pad-first would keep it at 1000)
        assert out[0, 4, 5, 5] == 1000  # a 2000 voxel at x/2 — dynamic range preserved
        assert out[0, 0, 0, 0] == 0  # padding region stays at MR background


class TestFullPreprocessingChain:
    """End-to-end shape of the crop → intensity → pad chain per modality."""

    def test_small_volume_is_mapped_then_padded_for_mr(self, build_preprocessing: object) -> None:
        volume = torch.full((1, 8, 8, 8), 500.0)
        volume[0, 0, 0, 0] = 0.0  # percentile floor
        volume[0, 0, 0, 1] = 2000.0  # above-window tail

        out = build_preprocessing("mr")({"image": volume})["image"]

        assert out.shape == (1, 16, 16, 16)
        assert out[0, 7, 7, 7] == 1000  # interior 500 maps to 2x = 1000 (p99.5 = 500)
        assert out[0, 4, 4, 5] == 4000  # the 2000 voxel is preserved, NOT clipped to 1000
        assert out.min() == 0
        assert out.max() == 4000

    def test_small_volume_is_clipped_then_padded_for_ct(self, build_preprocessing: object) -> None:
        volume = torch.full((1, 8, 8, 8), 500.0)
        volume[0, 0, 0, 0] = -500.0  # legal HU: must survive untouched
        volume[0, 0, 0, 1] = 2000.0  # over-range HU: clipped to 1000

        out = build_preprocessing("ct")({"image": volume})["image"]

        assert out.shape == (1, 16, 16, 16)
        assert out[0, 7, 7, 7] == 500  # interior voxel preserved
        assert out[0, 4, 4, 4] == -500  # negative HU preserved
        assert out[0, 4, 4, 5] == 1000  # over-range clipped to HU max
        assert out[0, 0, 0, 0] == -1000  # padding region uses CT air
        assert out[0, 15, 15, 15] == -1000
