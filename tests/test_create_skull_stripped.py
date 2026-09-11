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

"""Tests for scripts.create_skull_stripped — the ``image x brain-mask`` derivation (ticket T6 #22)."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_skull_stripped import SkullStrippedCreator

REPO_ROOT = Path(__file__).resolve().parents[1]
AFFINE = np.diag((1.5, 1.5, 2.0, 1.0))


@pytest.fixture
def pair_factory(tmp_path: Path) -> Callable[..., tuple[Path, Path]]:
    """Factory writing an image/mask pair; the mask fills the half-open voxel box ``mask_box``."""

    def build(
        name: str = "series",
        shape: tuple[int, int, int] = (6, 6, 6),
        mask_box: tuple[int, int, int] = (3, 3, 3),
        mask_value: int = 1,
    ) -> tuple[Path, Path]:
        rng = np.random.default_rng(0)
        image_data = rng.uniform(1.0, 100.0, size=shape).astype(np.float32)
        mask_data = np.zeros(shape, dtype=np.uint8)
        mask_data[: mask_box[0], : mask_box[1], : mask_box[2]] = mask_value
        image_path = tmp_path / f"{name}.nii.gz"
        mask_path = tmp_path / f"{name}_brain-mask.nii.gz"
        nib.save(nib.Nifti1Image(image_data, AFFINE), image_path)
        nib.save(nib.Nifti1Image(mask_data, AFFINE), mask_path)
        return image_path, mask_path

    return build


class TestSkullStrippedCreator:
    def test_voxels_outside_the_mask_are_zeroed(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        image_path, mask_path = pair_factory()
        output = tmp_path / "out.nii.gz"
        SkullStrippedCreator().create(image_path, mask_path, output)

        original = np.asanyarray(nib.load(str(image_path)).dataobj)
        stripped = np.asanyarray(nib.load(str(output)).dataobj)
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj) != 0
        assert np.array_equal(stripped[mask], original[mask])
        assert np.all(stripped[~mask] == 0)

    def test_every_masked_voxel_survives_exactly(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        """No scaling, no clipping: the derivation is a multiply, so intensities are untouched under the mask."""
        image_path, mask_path = pair_factory(shape=(4, 4, 4), mask_box=(2, 2, 2))
        output = tmp_path / "out.nii.gz"
        SkullStrippedCreator().create(image_path, mask_path, output)

        original = np.asanyarray(nib.load(str(image_path)).dataobj)
        stripped = np.asanyarray(nib.load(str(output)).dataobj)
        assert np.array_equal(stripped[:2, :2, :2], original[:2, :2, :2])
        assert np.all(stripped[2:, 2:, 2:] == 0)
        assert np.all(stripped[2:, :2, :2] == 0)  # masked on one axis only
        assert np.all(stripped[:2, 2:, :2] == 0)

    def test_output_keeps_the_image_geometry(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        image_path, mask_path = pair_factory()
        output = tmp_path / "out.nii.gz"
        SkullStrippedCreator().create(image_path, mask_path, output)

        image = nib.load(str(image_path))
        stripped = nib.load(str(output))
        assert stripped.shape == image.shape
        assert np.allclose(stripped.affine, image.affine)
        assert [float(z) for z in stripped.header.get_zooms()[:3]] == pytest.approx([float(z) for z in image.header.get_zooms()[:3]])
        assert stripped.get_data_dtype() == image.get_data_dtype()

    def test_non_binary_mask_is_treated_as_a_support(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        """An HD-BET mask that survived a lossy round-trip (0/2 rather than 0/1) still selects the same voxels."""
        image_path, mask_path = pair_factory(mask_value=2)
        output = tmp_path / "out.nii.gz"
        SkullStrippedCreator().create(image_path, mask_path, output)

        support = np.asanyarray(nib.load(str(mask_path)).dataobj) != 0
        stripped = np.asanyarray(nib.load(str(output)).dataobj)
        original = np.asanyarray(nib.load(str(image_path)).dataobj)
        assert np.count_nonzero(support) == 27
        assert np.array_equal(stripped[support], original[support])
        assert np.all(stripped[~support] == 0)

    def test_off_grid_pair_raises_before_writing(self, tmp_path: Path) -> None:
        image_path = tmp_path / "image.nii.gz"
        mask_path = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), AFFINE), image_path)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 5), dtype=np.uint8), AFFINE), mask_path)
        output = tmp_path / "out.nii.gz"

        with pytest.raises(ValueError, match="voxel grid"):
            SkullStrippedCreator().create(image_path, mask_path, output)
        assert not output.exists()

    def test_creates_missing_output_directories(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        image_path, mask_path = pair_factory()
        output = tmp_path / "mri/batch00/STUDY/img/STUDY_t1w-raw-axi_skull_stripped.nii.gz"
        SkullStrippedCreator().create(image_path, mask_path, output)
        assert output.is_file()


class TestCommandLine:
    def test_end_to_end_derivation(self, tmp_path: Path, pair_factory: Callable[..., tuple[Path, Path]]) -> None:
        image_path, mask_path = pair_factory()
        output = tmp_path / "stripped.nii.gz"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.create_skull_stripped",
                "--image",
                str(image_path),
                "--mask",
                str(mask_path),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert f"wrote {output}" in result.stdout
        assert output.is_file()

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run([sys.executable, "-m", "scripts.create_skull_stripped", "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0
        assert "--mask" in result.stdout
