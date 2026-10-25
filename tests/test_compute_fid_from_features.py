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

"""Tests for scripts.compute_fid_from_features (ticket T10, issue #28).

The number it prints must be the same statistic the GPU tool computes on the
same features -- the anchor test runs MONAI FIDMetric directly on identical
inputs and demands equality -- and the result json must be byte-compatible
with what summarize_fid_results consumes downstream.
"""

import importlib
import math
from pathlib import Path

import pytest
import torch
from monai.metrics.fid import FIDMetric

from scripts.compute_fid_from_features import FeatureSetFidCalculator, main

fid_from_features = importlib.import_module("scripts.compute_fid_from_features")


def write_feature_set(root: Path, n_volumes: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    for i in range(n_volumes):
        # rglob("*.pt") must find nested files: real caches mirror volume trees.
        directory = root / f"case{i:03d}" / "img"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(tuple(torch.randn(5, 8, generator=generator) for _ in range(3)), directory / f"vol_{i:03d}.pt")


@pytest.fixture
def feature_pair(tmp_path: Path) -> tuple[Path, Path]:
    real, synth = tmp_path / "real", tmp_path / "synth"
    write_feature_set(real, 4, seed=1)
    write_feature_set(synth, 4, seed=2)
    return real, synth


def make_calculator(real: Path, synth: Path, tag: str = "gengen_label9_seed42") -> FeatureSetFidCalculator:
    return FeatureSetFidCalculator(
        real_features_dir=real,
        synth_features_dir=synth,
        comparison_tag=tag,
        num_images=4,
        real_filelist="ref/label9.txt",
        synth_filelist="ft/label9_seed42.txt",
    )


class TestFeatureSetFidCalculator:
    def test_matches_monai_fidmetric_on_identical_inputs(self, feature_pair: tuple[Path, Path]) -> None:
        real, synth = feature_pair

        result = make_calculator(real, synth).compute()

        expected = FIDMetric()
        synth_planes = fid_from_features.FeatureDirectoryStack(synth, "s").stack()
        real_planes = fid_from_features.FeatureDirectoryStack(real, "r").stack()
        assert result.fid_xy == pytest.approx(float(expected(synth_planes[0], real_planes[0])))
        assert result.fid_avg == pytest.approx((result.fid_xy + result.fid_yz + result.fid_zx) / 3.0)

    def test_identical_distributions_give_near_zero_fid(self, tmp_path: Path) -> None:
        same_a, same_b = tmp_path / "a", tmp_path / "b"
        write_feature_set(same_a, 3, seed=7)
        write_feature_set(same_b, 3, seed=7)  # same seed, identical statistics

        result = make_calculator(same_a, same_b).compute()

        assert result.fid_avg == pytest.approx(0.0, abs=1e-4)

    def test_distinct_distributions_give_larger_fid_than_shifted_copy(self, tmp_path: Path) -> None:
        base, shifted, untouched = tmp_path / "base", tmp_path / "shifted", tmp_path / "untouched"
        write_feature_set(base, 3, seed=3)
        generator = torch.Generator().manual_seed(3)
        shifted.mkdir(parents=True)
        for i in range(3):
            directory = shifted / f"case{i:03d}" / "img"
            directory.mkdir(parents=True)
            torch.save(tuple(torch.randn(5, 8, generator=generator) + 50.0 for _ in range(3)), directory / f"vol_{i:03d}.pt")
        write_feature_set(untouched, 3, seed=9)

        small = make_calculator(base, untouched).compute()
        large = make_calculator(base, shifted).compute()

        assert large.fid_avg > small.fid_avg

    def test_result_carries_provenance_for_the_summarizer(self, feature_pair: tuple[Path, Path]) -> None:
        real, synth = feature_pair

        result = make_calculator(real, synth).compute()

        assert result.comparison_tag == "gengen_label9_seed42"
        assert result.num_images == 4
        assert result.real_filelist == "ref/label9.txt"
        assert result.modality == "mr"
        assert all(math.isfinite(v) for v in (result.fid_xy, result.fid_yz, result.fid_zx, result.fid_avg))

    def test_empty_directory_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match="real"):
            make_calculator(empty, tmp_path / "synth").compute()


class TestMainCli:
    def test_writes_result_json(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        real, synth = tmp_path / "real", tmp_path / "synth"
        write_feature_set(real, 2, seed=11)
        write_feature_set(synth, 2, seed=12)
        result_json = tmp_path / "fid" / "gengen_label10_seed42.json"

        import sys

        argv = sys.argv
        sys.argv = [
            "compute_fid_from_features",
            "--real_features_dir", str(real),
            "--synth_features_dir", str(synth),
            "--comparison_tag", "gengen_label10_seed42",
            "--num_images", "2",
            "--result_json", str(result_json),
        ]
        try:
            main()
        finally:
            sys.argv = argv

        loaded = fid_from_features.FidResult.load(result_json)
        assert loaded.comparison_tag == "gengen_label10_seed42"
        assert "avg=" in capsys.readouterr().out
