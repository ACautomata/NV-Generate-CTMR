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
generator-side (0, 1000) output domain. Chain tests run volumes through the
real ``ModalityPreprocessing.compose()`` (the exact code ``main()`` uses) via
small on-disk NIfTI volumes, including the intensity-before-padding ordering
(padding voxels must not dilute the percentile statistics). The same single
Compose instance serves both the real and the synth loader, which is what
keeps FID comparable.
"""

import importlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import ScaleIntensityRanged, ScaleIntensityRangePercentilesd, SpatialPadd

# Module name carries a hyphen, so import it programmatically.
compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")

ModalityPreprocessing = compute_fid.ModalityPreprocessing


@pytest.fixture
def modality_preprocessing() -> dict[str, ModalityPreprocessing]:
    return compute_fid.MODALITY_PREPROCESSING


@pytest.fixture
def run_preprocessing(modality_preprocessing: dict[str, ModalityPreprocessing], tmp_path: Path) -> Callable[..., torch.Tensor]:
    """Factory: write an 8^3 volume to disk and run it through the modality's full ``compose()`` chain."""

    def _run(modality: str, volume: torch.Tensor, target_shape: tuple[int, int, int] = (16, 16, 16)) -> torch.Tensor:
        path = tmp_path / f"{modality}.nii.gz"
        nib.save(nib.Nifti1Image(volume[0].numpy().astype(np.float32), affine=np.eye(4)), path)
        transforms = modality_preprocessing[modality].compose(target_shape)
        return transforms({"image": str(path)})["image"].as_tensor()

    return _run


class TestModalityPreprocessingTable:
    def test_ct_row_matches_prechange_constants(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        ct = modality_preprocessing["ct"]

        assert ct.padding_value == -1000
        assert ct.output_range == (-1000, 1000)
        assert ct.percentile_range is None  # fixed HU clip, as before this ticket

    def test_mr_row_matches_spec_constants(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        mr = modality_preprocessing["mr"]

        assert mr.padding_value == 0
        assert mr.output_range == (0, 1000)  # generator-side output domain
        assert mr.percentile_range == (0.0, 99.5)  # training pipeline's dynamic window

    def test_table_knows_only_ct_and_mr(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        assert sorted(modality_preprocessing) == ["ct", "mr"]


class TestModalityPaddingBehavior:
    """The padding constant differs per modality and must come from the shared table."""

    @pytest.mark.parametrize("modality,expected_pad", [("ct", -1000), ("mr", 0)])
    def test_padding_value_fills_symmetric_border(
        self, modality_preprocessing: dict[str, ModalityPreprocessing], modality: str, expected_pad: float
    ) -> None:
        pad = SpatialPadd(
            keys=["image"],
            spatial_size=(16, 16, 16),
            mode="constant",
            value=modality_preprocessing[modality].padding_value,
        )

        out = pad({"image": torch.full((1, 8, 8, 8), 500.0)})["image"]

        # SpatialPadd pads symmetrically: the 8^3 volume lands at [4:12] per axis.
        assert out[0, 0, 0, 0] == expected_pad  # padding region filled with the modality's background value
        assert out[0, 7, 7, 7] == 500  # original voxels untouched
        assert out[0, 15, 15, 15] == expected_pad


class TestModalityIntensityBehavior:
    """Intensity windowing per modality, both fed by the shared constant table."""

    def test_ct_clips_to_hu_range_keeping_negative_hu(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        a_min, a_max = modality_preprocessing["ct"].output_range
        clip = ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=a_min, b_max=a_max, clip=True)

        volume = torch.tensor([[[[-500.0, 500.0], [2000.0, -2000.0]]]])
        out = clip({"image": volume})["image"]

        assert out[0, 0, 0, 0] == -500  # legal HU preserved
        assert out[0, 0, 0, 1] == 500
        assert out[0, 0, 1, 0] == 1000  # over-range clipped to HU max
        assert out[0, 0, 1, 1] == -1000  # below-range clipped to HU min

    def test_mr_maps_percentile_window_without_clipping_high_intensities(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        """Real MR volumes can far exceed 1000 — the mapping must keep the dynamic tail."""
        mr = modality_preprocessing["mr"]
        lower, upper = mr.percentile_range
        b_min, b_max = mr.output_range
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


class TestFeatureCacheNamespacing:
    """Regression (PR #33 review, codex P1): the .pt feature cache is reused across
    runs by default (``ignore_existing=False``), so its directory must be namespaced
    by modality. Otherwise re-running with a different ``--modality`` on the same
    ``--output_root`` silently serves features preprocessed for the other
    modality's intensity domain and produces wrong FID values.
    """

    def test_cache_dir_layout_includes_modality_segment(self, modality_preprocessing: dict[str, ModalityPreprocessing]) -> None:
        assert modality_preprocessing["ct"].feature_cache_dir("/out", "datasetA") == os.path.join("/out", "ct", "datasetA")
        assert modality_preprocessing["mr"].feature_cache_dir("/out", "datasetA") == os.path.join("/out", "mr", "datasetA")

    def test_mr_run_does_not_hit_ct_cache_in_reused_output_root(
        self, modality_preprocessing: dict[str, ModalityPreprocessing], tmp_path: Path
    ) -> None:
        # Run 1 (--modality ct): real features cached under the shared output root.
        ct_dir = Path(modality_preprocessing["ct"].feature_cache_dir(str(tmp_path), "features_real"))
        ct_dir.mkdir(parents=True)
        (ct_dir / "case_001.pt").touch()

        # Run 2 (--modality mr, same --output_root, default ignore_existing=False):
        # the stale CT cache must be invisible to the MR run.
        mr_dir = Path(modality_preprocessing["mr"].feature_cache_dir(str(tmp_path), "features_real"))
        assert not (mr_dir / "case_001.pt").exists()


class TestComposeChainOrdering:
    """Chains built by ``ModalityPreprocessing.compose()`` — the exact code ``main()`` runs."""

    def test_mr_percentile_ignores_padding_voxels(self, run_preprocessing: Callable[..., torch.Tensor]) -> None:
        """Padding voxels must not dilute the percentile statistics.

        An 8^3 content in a 16^3 target (12.5% content share) whose values
        make the content's 99.5th percentile (2000) differ from the padded
        volume's effective percentile (1000): the two orderings give
        different interior mappings, so the assertions pin the ordering.
        """
        volume = torch.zeros(1, 8, 8, 8)
        volume[0, 0, 0, 0] = 0.0
        volume[0, 1:, ...] = 500.0  # 504 voxels at 500
        volume[0, 0, 0, 1:5] = 1000.0  # 4 voxels at 1000
        volume[0, 0, 1, 1:12] = 2000.0  # 11 voxels at 2000 (top tail)

        out = run_preprocessing("mr", volume)

        # Content-only p99.5 = 2000 → linear map x/2: interior 500 becomes 250.
        # Pad-first ordering would compute p99.5 = 1000 on the padded volume
        # (12.5% content share) and map interior 500 to 500.
        assert out[0, 4 + 4, 4 + 4, 4 + 4] == 250  # volume[0, 4, 4, 4] sits at 500
        assert out[0, 4, 4, 5] == 500  # a 1000 voxel at x/2 (pad-first would keep it at 1000)
        assert out[0, 4, 5, 5] == 1000  # a 2000 voxel at x/2 — dynamic range preserved
        assert out[0, 0, 0, 0] == 0  # padding region stays at MR background

    def test_small_volume_is_mapped_then_padded_for_mr(self, run_preprocessing: Callable[..., torch.Tensor]) -> None:
        volume = torch.full((1, 8, 8, 8), 500.0)
        volume[0, 0, 0, 0] = 0.0  # percentile floor
        volume[0, 0, 0, 1] = 2000.0  # above-window tail

        out = run_preprocessing("mr", volume)

        assert out.shape == (1, 16, 16, 16)
        assert out[0, 7, 7, 7] == 1000  # interior 500 maps to 2x = 1000 (p99.5 = 500)
        assert out[0, 4, 4, 5] == 4000  # the 2000 voxel is preserved, NOT clipped to 1000
        assert out.min() == 0
        assert out.max() == 4000

    def test_small_volume_is_clipped_then_padded_for_ct(self, run_preprocessing: Callable[..., torch.Tensor]) -> None:
        volume = torch.full((1, 8, 8, 8), 500.0)
        volume[0, 0, 0, 0] = -500.0  # legal HU: must survive untouched
        volume[0, 0, 0, 1] = 2000.0  # over-range HU: clipped to 1000

        out = run_preprocessing("ct", volume)

        assert out.shape == (1, 16, 16, 16)
        assert out[0, 7, 7, 7] == 500  # interior voxel preserved
        assert out[0, 4, 4, 4] == -500  # negative HU preserved
        assert out[0, 4, 4, 5] == 1000  # over-range clipped to HU max
        assert out[0, 0, 0, 0] == -1000  # padding region uses CT air
        assert out[0, 15, 15, 15] == -1000


# One of the 24 baseline records published in the t8-baseline-freeze-20260911
# release: the exact key set (12) this dataclass has to keep loading.
PUBLISHED_RECORD = {
    "comparison_tag": "label9_seed42",
    "fid_xy": 4.099713194645936,
    "fid_yz": 6.09807093899615,
    "fid_zx": 6.0591007263717085,
    "fid_avg": 5.418961620004598,
    "modality": "mr",
    "model_name": "radimagenet_resnet50",
    "num_images": 50,
    "real_filelist": "runs/t8-baseline-gen-20260911/fid_filelists/ref/mri_t1.txt",
    "synth_filelist": "runs/t8-baseline-gen-20260911/fid_filelists/synth/label9_seed42.txt",
    "target_shape": "256x256x128",
    "center_slices_ratio": 0.4,
}


class TestFidResult:
    """The result record added by T8 (#24): one FID invocation -> one frozen json artifact."""

    def make_result(self) -> "compute_fid.FidResult":
        return compute_fid.FidResult(
            comparison_tag="label9_seed42",
            fid_xy=10.1,
            fid_yz=11.2,
            fid_zx=12.3,
            fid_avg=11.2,
            modality="mr",
            model_name="radimagenet_resnet50",
            num_images=50,
            real_filelist="ref/mri_t1.txt",
            synth_filelist="synth/label9_seed42.txt",
            target_shape="256x256x128",
            center_slices_ratio=0.4,
            enable_padding=True,
            enable_center_cropping=True,
            enable_resampling_spacing="1.0x1.0x1.0",
        )

    def test_round_trips_through_json(self, tmp_path) -> None:
        result = self.make_result()
        path = tmp_path / "fid_label9_seed42.json"

        result.save(path)
        restored = compute_fid.FidResult.load(path)

        assert restored == result

    def test_json_carries_the_provenance_needed_for_the_freeze(self, tmp_path) -> None:
        path = tmp_path / "fid.json"

        self.make_result().save(path)

        payload = json.loads(path.read_text())
        assert payload["comparison_tag"] == "label9_seed42"
        assert payload["fid_avg"] == 11.2
        assert payload["modality"] == "mr"
        assert payload["real_filelist"] == "ref/mri_t1.txt"
        assert payload["synth_filelist"] == "synth/label9_seed42.txt"

    def test_json_carries_the_transform_flags_that_shape_the_features(self, tmp_path) -> None:
        """Padding, cropping and resampling decide what the feature network is handed.

        Unlike ``target_shape``, which only names the box, these decide whether a
        volume is resampled into it and which side gets cropped or padded -- so two
        records differing in them describe different inputs and cannot be compared
        from their numbers alone.
        """
        path = tmp_path / "fid.json"

        self.make_result().save(path)

        payload = json.loads(path.read_text())
        assert payload["enable_padding"] is True
        assert payload["enable_center_cropping"] is True
        assert payload["enable_resampling_spacing"] == "1.0x1.0x1.0"

    def test_a_record_published_before_the_flags_still_loads(self, tmp_path) -> None:
        """The 24 frozen records already published carry this exact key set and must keep loading.

        Their three flags read back as unknown rather than guessed: an assumption of
        ``True`` is what this record exists to stop anyone having to make.
        """
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(PUBLISHED_RECORD))

        restored = compute_fid.FidResult.load(path)

        assert restored.comparison_tag == "label9_seed42"
        assert restored.fid_avg == 5.418961620004598
        assert restored.enable_padding is None
        assert restored.enable_center_cropping is None
        assert restored.enable_resampling_spacing is None
