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

"""Tests for the standalone spine filter heuristic (spec #13 section 3.3, ticket T2 #18)."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.spine_filter import SpineFilter, SpineFilterThresholds, main

VOXEL_MM = 1.0


@pytest.fixture
def volume_factory(tmp_path: Path) -> Callable[..., tuple[Path, Path]]:
    """Factory building image/mask NIfTI pairs on a shared grid; mask_ratio and fov drive the verdict."""

    def build(name: str, shape: tuple[int, ...], zooms: tuple[float, ...], mask_ratio: float) -> tuple[Path, Path]:
        image = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        if mask_ratio > 0:
            filled = int(mask.size * mask_ratio)
            mask.reshape(-1)[:filled] = 1
            image.reshape(-1)[:filled] = 100.0
        image_path = tmp_path / f"{name}_img.nii.gz"
        mask_path = tmp_path / f"{name}_mask.nii.gz"
        affine = np.eye(4) * np.array([*zooms, 1.0])
        nib.save(nib.Nifti1Image(image, affine), image_path)
        nib.save(nib.Nifti1Image(mask, affine), mask_path)
        return image_path, mask_path

    return build


class TestSpineFilter:
    def test_brain_like_volume_passes(self, volume_factory) -> None:
        """Brain-like: mask ratio 10% (within [2%, 35%]), every axis FOV <= 300 mm."""
        image, mask = volume_factory("brain", (240, 240, 200), (VOXEL_MM,) * 3, mask_ratio=0.10)
        result = SpineFilter().check(image, mask)
        assert result.is_brain
        assert result.reasons == ()
        assert result.mask_voxel_ratio == pytest.approx(0.10)
        assert result.fov_mm == (240.0, 240.0, 200.0)

    def test_spine_like_volume_fails_on_mask_ratio(self, volume_factory) -> None:
        """Spine-like: HD-BET finds almost no brain in a spine FOV -> ratio below 2%."""
        image, mask = volume_factory("spine", (512, 512, 32), (0.43, 0.43, 5.2), mask_ratio=0.005)
        result = SpineFilter().check(image, mask)
        assert not result.is_brain
        assert len(result.reasons) == 1
        assert "mask voxel ratio" in result.reasons[0]

    def test_ratio_above_maximum_fails(self, volume_factory) -> None:
        image, mask = volume_factory("huge", (64, 64, 64), (VOXEL_MM,) * 3, mask_ratio=0.50)
        result = SpineFilter().check(image, mask)
        assert not result.is_brain
        assert "mask voxel ratio" in result.reasons[0]

    def test_oversized_fov_fails(self, volume_factory) -> None:
        """Full-spine sagittal acquisition: SI extent beyond the FOV ceiling."""
        image, mask = volume_factory("long", (96, 96, 352), (VOXEL_MM,) * 3, mask_ratio=0.10)
        result = SpineFilter().check(image, mask)
        assert not result.is_brain
        assert len(result.reasons) == 1
        assert "FOV" in result.reasons[0]

    def test_both_conditions_reported_when_both_fail(self, volume_factory) -> None:
        image, mask = volume_factory("both", (96, 96, 352), (VOXEL_MM,) * 3, mask_ratio=0.001)
        result = SpineFilter().check(image, mask)
        assert not result.is_brain
        assert len(result.reasons) == 2

    def test_fov_uses_header_zooms_times_shape(self, volume_factory) -> None:
        """FOV comes from the image header (zooms x shape per axis), not from metadata."""
        image, mask = volume_factory("zoomed", (100, 200, 300), (2.5, 1.0, 0.5), mask_ratio=0.10)
        result = SpineFilter().check(image, mask)
        assert result.fov_mm == (250.0, 200.0, 150.0)
        assert result.is_brain

    def test_empty_mask_fails(self, volume_factory) -> None:
        image, mask = volume_factory("empty", (100, 100, 100), (VOXEL_MM,) * 3, mask_ratio=0.0)
        result = SpineFilter().check(image, mask)
        assert not result.is_brain
        assert result.mask_voxel_ratio == 0.0

    def test_thresholds_are_configurable(self, volume_factory) -> None:
        image, mask = volume_factory("mid", (100, 100, 100), (VOXEL_MM,) * 3, mask_ratio=0.015)
        assert not SpineFilter().check(image, mask).is_brain
        lenient = SpineFilter(thresholds=SpineFilterThresholds(mask_ratio_min=0.01))
        assert lenient.check(image, mask).is_brain

    def test_grid_mismatch_between_image_and_mask_raises(self, volume_factory) -> None:
        image, _ = volume_factory("a", (100, 100, 100), (VOXEL_MM,) * 3, mask_ratio=0.10)
        _, mask = volume_factory("b", (80, 80, 80), (VOXEL_MM,) * 3, mask_ratio=0.10)
        with pytest.raises(ValueError, match="voxel grid"):
            SpineFilter().check(image, mask)

    def test_default_thresholds_match_the_spec_initial_values(self) -> None:
        thresholds = SpineFilterThresholds()
        assert thresholds.mask_ratio_min == pytest.approx(0.02)
        assert thresholds.mask_ratio_max == pytest.approx(0.35)
        assert thresholds.fov_max_mm == pytest.approx(300.0)


class TestCommandLine:
    def test_cli_reports_verdict(self, volume_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        image, mask = volume_factory("brain", (240, 240, 200), (VOXEL_MM,) * 3, mask_ratio=0.10)
        monkeypatch.setattr(
            sys,
            "argv",
            ["spine_filter", "--image", str(image), "--mask", str(mask), "--output", str(tmp_path / "verdict.json")],
        )
        main()
        verdict = json.loads((tmp_path / "verdict.json").read_text())
        assert verdict["is_brain"] is True
        assert "is_brain" in capsys.readouterr().out

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run([sys.executable, "-m", "scripts.spine_filter", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--fov-max-mm" in result.stdout
