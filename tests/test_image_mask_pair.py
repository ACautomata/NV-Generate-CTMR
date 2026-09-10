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

"""Tests for scripts.image_mask_pair — the loaded image/mask pair (ticket T6 #22).

Both stages that touch an MR-RATE series' two published volumes ask the same question before
they can work: is this pair usable at all?  What "unusable" is spelled as decides whether the
series is recorded as a reject or its whole study is retried, so it is the contract under test.
"""

from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.image_mask_pair import ImageMaskPair

AFFINE = np.diag((1.0, 1.0, 1.0, 1.0))
CORRUPT_BYTES = b"\x1f\x8b" + bytes(range(40)) * 3


@pytest.fixture
def volume_writer(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing one NIfTI of the given shape and returning its path."""

    def build(name: str, shape: tuple[int, int, int] = (64, 64, 64)) -> Path:
        path = tmp_path / name
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), AFFINE), path)
        return path

    return build


@pytest.fixture
def corrupt_volume(tmp_path: Path) -> Path:
    """A file with the gzip magic bytes and nothing parseable behind them."""
    path = tmp_path / "corrupt.nii.gz"
    path.write_bytes(CORRUPT_BYTES)
    return path


class TestLoadedPair:
    def test_a_pair_on_one_grid_loads(self, volume_writer: Callable[..., Path]) -> None:
        pair = ImageMaskPair(volume_writer("image.nii.gz"), volume_writer("mask.nii.gz"))
        assert pair.image.shape == pair.mask.shape == (64, 64, 64)

    def test_the_image_path_survives_for_the_reports_that_name_it(self, volume_writer: Callable[..., Path]) -> None:
        image = volume_writer("image.nii.gz")
        assert ImageMaskPair(image, volume_writer("mask.nii.gz")).image_path == image


class TestUnusablePair:
    """Every way a pair can be unusable is one ValueError, because callers act on it identically."""

    def test_a_grid_mismatch_is_refused(self, volume_writer: Callable[..., Path]) -> None:
        with pytest.raises(ValueError, match="differ in voxel grid"):
            ImageMaskPair(volume_writer("image.nii.gz"), volume_writer("mask.nii.gz", (32, 32, 32)))

    def test_a_corrupt_volume_is_refused_like_a_mismatch(self, corrupt_volume: Path, volume_writer: Callable[..., Path]) -> None:
        """nibabel's ImageFileError inherits neither ValueError nor OSError.

        Left unnamed it would escape every handler between here and the download loop, and end
        an hours-long run on one bad member instead of recording that series and moving on.
        """
        with pytest.raises(ValueError, match="could not be read"):
            ImageMaskPair(corrupt_volume, volume_writer("mask.nii.gz"))

    def test_a_corrupt_mask_is_refused_too(self, volume_writer: Callable[..., Path], corrupt_volume: Path) -> None:
        with pytest.raises(ValueError, match="could not be read"):
            ImageMaskPair(volume_writer("image.nii.gz"), corrupt_volume)

    def test_a_missing_volume_stays_an_oserror(self, tmp_path: Path, volume_writer: Callable[..., Path]) -> None:
        """A failed read is the study's problem, not the series': the download stage retries the study."""
        with pytest.raises(FileNotFoundError):
            ImageMaskPair(tmp_path / "absent.nii.gz", volume_writer("mask.nii.gz"))
