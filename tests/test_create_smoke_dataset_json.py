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

"""Tests for the smoke-run subset builder (ticket T8, issue #24; spec #13 stage D3).

The training smoke test consumes a small slice of a merged dataset.json that must
still exercise the two properties the real training data has: every label the
modality mapping carries (new BRATS labels 40-43 plus old replay labels) and
**mixed latent sizes** (BRATS 256x256x128 alongside MR-RATE native round-to-128
volumes). The training loop silently skips entries whose latent is missing, so the
builder must refuse to emit a subset whose latents or sidecars are not on disk -- a
smoke run that quietly shrank would prove nothing.
"""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.create_smoke_dataset_json import SmokeQuota, SmokeSubsetBuilder, main

BRATS_ENTRY = "brats2023-gli/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz"
REPLAY_ENTRY = "mri/batch00/AAAAA00000/img/AAAAA00000_t1w-raw-axi.nii.gz"


@pytest.fixture
def latent_dir(tmp_path: Path) -> Path:
    """Embedding base dir; latents are created on demand by the helpers."""
    return tmp_path / "embeddings"


def write_latent(embedding_base_dir: Path, image_rel: str, shape: tuple[int, ...]) -> None:
    latent_rel = image_rel.replace(".nii.gz", "_emb.nii.gz")
    latent_path = embedding_base_dir / latent_rel
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), latent_path)
    sidecar_path = latent_path.parent / (latent_path.name + ".json")
    sidecar_path.write_text(json.dumps({"spacing": [1.0, 1.0, 1.0], "modality": "x"}))


def make_entries(brats: int, replay: int) -> list[dict]:
    entries = [{"image": f"brats2023-gli/case{i}-t1n.nii.gz", "modality": "mri_t1n"} for i in range(brats)]
    entries += [{"image": f"mri/batch00/p{i}/img/p{i}_t1w-raw-axi.nii.gz", "modality": "mri_t1"} for i in range(replay)]
    return entries


class TestSmokeSubsetBuilder:
    def test_subset_is_deterministic_for_a_seed(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32) if "brats" in e["image"] else (48, 48, 32))

        quota = SmokeQuota(per_label=2, labels=("mri_t1n", "mri_t1"))
        first = SmokeSubsetBuilder(entries, quota, latent_dir, seed=42).build()
        second = SmokeSubsetBuilder(entries, quota, latent_dir, seed=42).build()

        assert first == second

    def test_subset_respects_the_per_label_quota(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32) if "brats" in e["image"] else (48, 48, 32))

        subset = SmokeSubsetBuilder(entries, SmokeQuota(per_label=2, labels=("mri_t1n", "mri_t1")), latent_dir, seed=42).build()

        assert len(subset) == 4  # 2 mri_t1n + 2 mri_t1
        assert sum(1 for e in subset if e["modality"] == "mri_t1n") == 2
        assert sum(1 for e in subset if e["modality"] == "mri_t1") == 2

    def test_subset_requires_the_mixed_sizes_to_be_real(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries:
            # Everything the same shape -> the "mixed sizes" claim would be hollow.
            write_latent(latent_dir, e["image"], shape=(64, 64, 32))

        with pytest.raises(ValueError, match="single latent shape"):
            SmokeSubsetBuilder(entries, SmokeQuota(per_label=2, labels=("mri_t1n", "mri_t1")), latent_dir, seed=42).build()

    def test_a_missing_latent_aborts_the_smoke(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries[:-1]:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32) if "brats" in e["image"] else (48, 48, 32))

        # Quota covers every entry, so the one without a latent is guaranteed to be sampled.
        with pytest.raises(FileNotFoundError, match="_emb.nii.gz"):
            SmokeSubsetBuilder(entries, SmokeQuota(per_label=6, labels=("mri_t1n", "mri_t1")), latent_dir, seed=42).build()

    def test_a_requested_label_missing_from_the_dataset_is_refused(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=0)
        for e in entries:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32))

        with pytest.raises(ValueError, match="mri_t1"):
            SmokeSubsetBuilder(entries, SmokeQuota(per_label=2, labels=("mri_t1n", "mri_t1")), latent_dir, seed=42).build()

    def test_entries_keep_the_dataset_json_schema(self, latent_dir: Path) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32) if "brats" in e["image"] else (48, 48, 32))

        subset = SmokeSubsetBuilder(entries, SmokeQuota(per_label=2, labels=("mri_t1n", "mri_t1")), latent_dir, seed=42).build()

        assert all(set(e) == {"image", "modality"} for e in subset)
        assert all("brats2023-gli" in e["image"] or e["image"].startswith("mri/") for e in subset)


class TestMain:
    def test_end_to_end_write(self, tmp_path: Path, latent_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        entries = make_entries(brats=6, replay=6)
        for e in entries:
            write_latent(latent_dir, e["image"], shape=(64, 64, 32) if "brats" in e["image"] else (48, 48, 32))
        dataset_json = tmp_path / "dataset_N300.json"
        dataset_json.write_text(json.dumps({"training": entries}))
        output = tmp_path / "smoke" / "dataset_smoke.json"

        sys.argv = [
            "create_smoke_dataset_json.py",
            "--dataset-json",
            str(dataset_json),
            "--embedding-base-dir",
            str(latent_dir),
            "--per-label",
            "2",
            "--labels",
            "mri_t1n",
            "mri_t1",
            "--seed",
            "42",
            "--output",
            str(output),
        ]
        main()

        payload = json.loads(output.read_text())
        assert len(payload["training"]) == 4
        assert all(set(e) == {"image", "modality"} for e in payload["training"])

    def test_cli_is_invocable_as_a_module(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.create_smoke_dataset_json", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--dataset-json" in result.stdout
