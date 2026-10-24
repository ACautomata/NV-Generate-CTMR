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

"""Tests for scripts.make_montage (ticket T10, issue #28).

The visual kit is what the HITL verdict reads, so the tests pin the kit's
shape rather than its aesthetics: one triplanar montage per volume (40 for the
BRATS side from 4 labels x 10 volumes), a 4-volume sheet per quickscan label,
central-slice semantics, and a manifest that records every file for the
GitHub Release upload.
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from scripts.make_montage import QuickScanSheetBuilder, TriplanarMontageBuilder, VisualKitAssembler


@pytest.fixture
def volume_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "outputs_brats"
    directory.mkdir()
    generator = np.random.default_rng(0)
    prefix = "bratsftN300e050"
    for label in (40, 41, 42, 43):
        for seed in (42, 1337):
            for index in range(10):
                volume = generator.random((24, 24, 24)).astype(np.float32) * 800 + 100
                path = directory / f"{prefix}_label{label}_seed{seed}_{index:03d}.nii.gz"
                nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), path)
    for label in (9, 16):
        for seed in (42, 1337):
            for index in range(4):
                volume = generator.random((24, 24, 24)).astype(np.float32) * 800 + 100
                path = directory / f"ftN300e050_label{label}_seed{seed}_{index:03d}.nii.gz"
                nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), path)
    return directory


class TestTriplanarMontageBuilder:
    def test_writes_one_png_per_volume_with_three_planes(self, tmp_path: Path, volume_dir: Path) -> None:
        builder = TriplanarMontageBuilder(output_dir=tmp_path / "montage")

        written = builder.build_label_set(
            volume_dir=volume_dir,
            prefix="bratsftN300e050",
            label=40,
            seed=42,
            n_volumes=10,
        )

        assert len(written) == 10
        for png in written:
            assert png.exists()
            width, height = Image.open(png).size
            assert width > height  # 1x3 grid of axial-size panels

    def test_picks_seed_deterministically_and_sorted(self, tmp_path: Path, volume_dir: Path) -> None:
        builder = TriplanarMontageBuilder(output_dir=tmp_path / "montage")

        first = builder.build_label_set(volume_dir=volume_dir, prefix="bratsftN300e050", label=40, seed=42, n_volumes=10)
        second = builder.build_label_set(volume_dir=volume_dir, prefix="bratsftN300e050", label=40, seed=42, n_volumes=10)

        assert [p.name for p in first] == [p.name for p in second]

    def test_rejects_when_volumes_are_missing(self, tmp_path: Path, volume_dir: Path) -> None:
        builder = TriplanarMontageBuilder(output_dir=tmp_path / "montage")

        with pytest.raises(FileNotFoundError, match="found only 10"):
            builder.build_label_set(volume_dir=volume_dir, prefix="bratsftN300e050", label=40, seed=42, n_volumes=99)


class TestQuickScanSheetBuilder:
    def test_one_sheet_per_label_with_four_volumes(self, tmp_path: Path, volume_dir: Path) -> None:
        builder = QuickScanSheetBuilder(output_dir=tmp_path / "quickscan")

        sheets = builder.build_label_set(volume_dir=volume_dir, prefix="bratsftN300e050", label=40, seed=42, n_volumes=4)

        assert sheets == [tmp_path / "quickscan" / "quickscan_bratsftN300e050_label40_seed42.png"]
        width, height = Image.open(sheets[0]).size
        assert height > width  # 4 rows (volumes) x 3 columns (planes)


class TestVisualKitAssembler:
    def test_assembles_brats_kit_quickscan_sheets_and_manifest(self, tmp_path: Path, volume_dir: Path) -> None:
        adjudication = tmp_path / "adjudication.json"
        adjudication.write_text(json.dumps({"quickscan_labels": ["9", "16"]}))
        assembler = VisualKitAssembler(
            outputs_forgetting=volume_dir,
            outputs_brats=volume_dir,
            brats_prefix="bratsftN300e050",
            forgetting_prefix="ftN300e050",
            output_root=tmp_path / "visual_kit",
            adjudication_json=adjudication,
        )

        manifest = assembler.assemble()

        brats_pngs = list((tmp_path / "visual_kit" / "brats").glob("*.png"))
        assert len(brats_pngs) == 40  # 4 labels x 10 volumes
        assert manifest["brats_pngs"] == 40
        assert manifest["quickscan_labels"] == ["9", "16"]
        assert "montage_manifest.json" in {p.name for p in (tmp_path / "visual_kit").glob("*.json")}
