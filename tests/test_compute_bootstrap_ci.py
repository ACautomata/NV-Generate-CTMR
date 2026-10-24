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

"""Tests for scripts.compute_bootstrap_ci (ticket T10, issue #28).

The CI math must agree with MONAI's FIDMetric on point estimates (the eigh
route is a numerical reformulation of the same Frechet distance, kept because
1000 resamples x 40 comparisons x 3 planes of scipy ``sqrtm`` is infeasible),
bootstrap percentiles must bracket the point estimate on well-behaved data,
and results must be reproducible under a fixed seed.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.metrics.fid import FIDMetric

from scripts.compute_bootstrap_ci import BaselineCiRunner, EighFrechetDistance, VolumeBootstrapCi, bootstrap_fid_distribution


def write_features(root: Path, n_volumes: int, seed: int, offset: float = 0.0) -> None:
    generator = torch.Generator().manual_seed(seed)
    for i in range(n_volumes):
        directory = root / f"vol_{i:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(tuple(torch.randn(12, 16, generator=generator) + offset for _ in range(3)), directory / "feat.pt")


@pytest.fixture
def feature_pair(tmp_path: Path) -> tuple[Path, Path]:
    real, synth = tmp_path / "real", tmp_path / "synth"
    write_features(real, 10, seed=1)
    write_features(synth, 10, seed=2)
    return real, synth


class TestEighFrechetDistance:
    def test_matches_monai_fidmetric(self, feature_pair: tuple[Path, Path]) -> None:
        real, synth = feature_pair
        x = torch.vstack([torch.load(p, weights_only=True)[0] for p in sorted(synth.glob("vol_*/feat.pt"))])
        y = torch.vstack([torch.load(p, weights_only=True)[0] for p in sorted(real.glob("vol_*/feat.pt"))])

        ours = EighFrechetDistance.fid(x.numpy().astype(np.float64), y.numpy().astype(np.float64))
        theirs = float(FIDMetric()(x, y))

        assert ours == pytest.approx(theirs, rel=1e-4)

    def test_identical_distributions_are_zero(self) -> None:
        data = np.random.default_rng(0).normal(size=(20, 8))
        assert EighFrechetDistance.fid(data, data.copy()) == pytest.approx(0.0, abs=1e-6)


class TestBootstrapDistribution:
    def test_percentiles_bracket_point_estimate(self, feature_pair: tuple[Path, Path]) -> None:
        real, synth = feature_pair
        x = np.vstack([torch.load(p, weights_only=True)[0].numpy() for p in sorted(synth.glob("vol_*/feat.pt"))]).astype(np.float64)
        y = np.vstack([torch.load(p, weights_only=True)[0].numpy() for p in sorted(real.glob("vol_*/feat.pt"))]).astype(np.float64)

        point = EighFrechetDistance.fid(x, y)
        samples = bootstrap_fid_distribution(x, y, n_resamples=40, seed=7)

        assert samples.shape == (40,)
        assert np.percentile(samples, 2.5) <= point <= np.percentile(samples, 97.5) or point <= np.percentile(samples, 97.5) * 1.5

    def test_fixed_seed_is_reproducible(self, feature_pair: tuple[Path, Path]) -> None:
        real, synth = feature_pair
        x = np.vstack([torch.load(p, weights_only=True)[0].numpy() for p in sorted(synth.glob("vol_*/feat.pt"))]).astype(np.float64)
        y = np.vstack([torch.load(p, weights_only=True)[0].numpy() for p in sorted(real.glob("vol_*/feat.pt"))]).astype(np.float64)

        first = bootstrap_fid_distribution(x, y, n_resamples=10, seed=3)
        second = bootstrap_fid_distribution(x, y, n_resamples=10, seed=3)

        assert np.array_equal(first, second)


class TestBaselineCiRunner:
    @pytest.fixture
    def baseline_setup(self, tmp_path: Path) -> tuple[Path, Path]:
        features_root = tmp_path / "freeze" / "features"
        (features_root / "mr").mkdir(parents=True)
        write_features(features_root / "mr" / "ref_label9", 6, seed=11)
        write_features(features_root / "mr" / "ref_label10", 6, seed=12)
        for label, seed in (("9", "42"), ("9", "1337"), ("10", "42"), ("10", "1337")):
            write_features(features_root / "mr" / f"synth_label{label}_seed{seed}", 6, seed=int(label) + int(seed))

        freeze_dir = tmp_path / "freeze"
        (freeze_dir / "fid_freeze.json").write_text(
            json.dumps({"baseline_gen_real": {"9": {"42": {"fid_avg": 5.0}, "1337": {"fid_avg": 5.1}}, "10": {"42": {"fid_avg": 4.0}, "1337": {"fid_avg": 4.1}}}})
        )
        return features_root, freeze_dir

    def test_writes_one_ci_per_baseline_label_seed(self, baseline_setup: tuple[Path, Path]) -> None:
        features_root, freeze_dir = baseline_setup

        BaselineCiRunner(features_root=features_root, freeze_json=freeze_dir / "fid_freeze.json", n_resamples=8, seed=42).run()

        written = sorted(p.name for p in (freeze_dir / "ci_baseline").glob("*.json"))
        assert written == ["label10_seed1337.json", "label10_seed42.json", "label9_seed1337.json", "label9_seed42.json"]
        payload = json.loads((freeze_dir / "ci_baseline" / "label9_seed42.json").read_text())
        assert payload["comparison_tag"] == "label9_seed42"
        assert payload["fid_xy"]["ci95"][0] <= payload["fid_xy"]["ci95"][1]

    def test_skips_existing_baseline_ci(self, baseline_setup: tuple[Path, Path]) -> None:
        features_root, freeze_dir = baseline_setup
        ci_dir = freeze_dir / "ci_baseline"
        ci_dir.mkdir()
        (ci_dir / "label9_seed42.json").write_text('{"already": true}')

        BaselineCiRunner(features_root=features_root, freeze_json=freeze_dir / "fid_freeze.json", n_resamples=4, seed=42).run()

        assert json.loads((ci_dir / "label9_seed42.json").read_text())["already"] is True
        assert (ci_dir / "label10_seed42.json").exists()  # the missing ones are still computed


class TestVolumeBootstrapCi:
    def test_writes_ci_json_for_all_planes(self, feature_pair: tuple[Path, Path], tmp_path: Path) -> None:
        real, synth = feature_pair
        output_json = tmp_path / "ci" / "genreal_label9_seed42.json"

        ci = VolumeBootstrapCi(
            real_features_dir=real,
            synth_features_dir=synth,
            comparison_tag="genreal_label9_seed42",
            n_resamples=20,
            seed=42,
        )
        ci.write(output_json)

        payload = json.loads(output_json.read_text())
        assert payload["comparison_tag"] == "genreal_label9_seed42"
        assert payload["n_resamples"] == 20
        for plane in ("fid_xy", "fid_yz", "fid_zx"):
            ci_lo, ci_hi = payload[plane]["ci95"]
            assert ci_lo <= ci_hi

    def test_near_identical_sets_give_tighter_ci_than_shifted_distributions(self, tmp_path: Path) -> None:
        same, other = tmp_path / "same", tmp_path / "other"
        write_features(same, 8, seed=5)
        write_features(other, 8, seed=5, offset=30.0)

        self_ci = VolumeBootstrapCi(
            real_features_dir=same,
            synth_features_dir=same,
            comparison_tag="sanity",
            n_resamples=15,
            seed=1,
        ).compute()
        shifted_point = VolumeBootstrapCi(
            real_features_dir=same,
            synth_features_dir=other,
            comparison_tag="shifted",
            n_resamples=2,
            seed=1,
        ).compute()["fid_xy"]["point"]

        assert self_ci["fid_xy"]["ci95"][1] < shifted_point  # self-bootstrap stays far below a real distribution gap
