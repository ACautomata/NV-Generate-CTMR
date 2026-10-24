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

"""Tests for the ``FeatureCollector`` cache protocol (ticket T10, issue #28).

The T10 acceptance run reads its frozen-reference side from the T8/T17 feature
cache alone: the volume trees behind ``ref_label*`` / ``synth_label*`` lived on
an expired cluster scratch disk, so ``cache_only`` mode must serve features
without touching any volume file -- and must treat a missing cache entry as a
hard error (a silently shrunken feature set would corrupt the FID).

The extraction path is exercised with an injected fake extractor so no
RadImageNet download happens in tests; the cache protocol (write on extract,
skip on hit, ignore_existing override) is what matters here.
"""

import importlib
from pathlib import Path

import monai.transforms
import pytest
import torch

compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")

FeatureCollector = compute_fid.FeatureCollector

RELPATHS = ["caseB/img/vol_b.nii.gz", "caseA/img/vol_a.nii.gz", "caseC/img/vol_c.nii.gz"]


class CountingExtractor:
    """Stands in for get_features_2p5d: records calls, returns distinct per-plane marks."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, image: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(image)
        marker = float(image.sum())
        return (
            torch.tensor([[marker]]),
            torch.tensor([[marker + 1]]),
            torch.tensor([[marker + 2]]),
        )


def make_loader_compose() -> monai.transforms.Compose:
    return monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
        ]
    )


def make_collector(tmp_path: Path, extractor, **overrides) -> FeatureCollector:
    args = dict(
        dataset_root=str(tmp_path / "volumes"),
        filelist_lines=list(RELPATHS),
        cache_dir=str(tmp_path / "cache" / "mr" / "side"),
        compose_transforms=make_loader_compose(),
        feature_extractor=extractor,
        device=torch.device("cpu"),
        num_images=50,
        # macOS DataLoader workers crash under spawn in tests; the cache
        # protocol under test is worker-count independent.
        num_workers=0,
    )
    args.update(overrides)
    return FeatureCollector(**args)


def cache_file(tmp_path: Path, relpath: str) -> Path:
    return tmp_path / "cache" / "mr" / "side" / relpath.replace(".nii.gz", ".pt")


class TestCacheOnlyMode:
    def test_serves_stacked_features_without_touching_volumes(self, tmp_path: Path) -> None:
        extractor = CountingExtractor()
        for i, rel in enumerate(sorted(RELPATHS)):
            cache_file(tmp_path, rel).parent.mkdir(parents=True, exist_ok=True)
            torch.save((torch.tensor([[float(i)]]), torch.tensor([[float(i)]]), torch.tensor([[float(i)]])), cache_file(tmp_path, rel))

        collector = make_collector(tmp_path, extractor, cache_only=True)
        xy, yz, zx = collector.collect()

        assert xy.shape == (3, 1) and yz.shape == (3, 1) and zx.shape == (3, 1)
        assert extractor.calls == []  # no extraction, no volume loading
        assert torch.equal(xy, torch.tensor([[0.0], [1.0], [2.0]]))  # filelist-sorted order

    def test_missing_cache_entry_is_a_hard_error(self, tmp_path: Path) -> None:
        present = sorted(RELPATHS)[0]
        missing = sorted(RELPATHS)[1]
        cache_file(tmp_path, present).parent.mkdir(parents=True, exist_ok=True)
        torch.save((torch.zeros(1, 1),) * 3, cache_file(tmp_path, present))

        collector = make_collector(tmp_path, CountingExtractor(), cache_only=True)

        with pytest.raises(FileNotFoundError, match=missing.split("/")[-1].replace(".nii.gz", ".pt").replace(".", "\\.")):
            collector.collect()

    def test_ignore_existing_is_refused_in_cache_only_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ignore_existing"):
            make_collector(tmp_path, CountingExtractor(), cache_only=True, ignore_existing=True)


class TestExtractionMode:
    def test_extracts_missing_and_writes_cache(self, tmp_path: Path) -> None:
        extractor = CountingExtractor()
        _write_volume(tmp_path, "caseA/img/vol_a.nii.gz")

        xy, _, _ = make_collector(tmp_path, extractor, filelist_lines=["caseA/img/vol_a.nii.gz"]).collect()

        assert len(extractor.calls) == 1
        assert cache_file(tmp_path, "caseA/img/vol_a.nii.gz").exists()
        assert xy.shape == (1, 1)

    def test_cache_hit_skips_extraction(self, tmp_path: Path) -> None:
        extractor = CountingExtractor()
        _write_volume(tmp_path, "caseA/img/vol_a.nii.gz")
        cache_file(tmp_path, "caseA/img/vol_a.nii.gz").parent.mkdir(parents=True)
        torch.save((torch.tensor([[7.0]]), torch.tensor([[8.0]]), torch.tensor([[9.0]])), cache_file(tmp_path, "caseA/img/vol_a.nii.gz"))

        xy, yz, zx = make_collector(tmp_path, extractor, filelist_lines=["caseA/img/vol_a.nii.gz"]).collect()

        assert extractor.calls == []
        assert float(xy[0, 0]) == 7.0 and float(zx[0, 0]) == 9.0

    def test_rank_partition_preserves_every_volume_once(self, tmp_path: Path) -> None:
        extractor = CountingExtractor()
        for rel in sorted(RELPATHS):
            cache_file(tmp_path, rel).parent.mkdir(parents=True, exist_ok=True)
            torch.save((torch.ones(1, 1),) * 3, cache_file(tmp_path, rel))

        world_parts = []
        for rank in range(2):
            collector = make_collector(tmp_path, extractor, cache_only=True, local_rank=rank, world_size=2)
            world_parts.append(collector.collect()[0])

        total = sum(part.shape[0] for part in world_parts)
        assert total == 3  # every volume served exactly once across ranks


def _write_volume(tmp_path: Path, relpath: str) -> None:
    import nibabel as nib
    import numpy as np

    path = tmp_path / "volumes" / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), affine=np.eye(4)), path)


def _dummy_volume() -> torch.Tensor:
    return torch.rand(1, 8, 8, 8)
