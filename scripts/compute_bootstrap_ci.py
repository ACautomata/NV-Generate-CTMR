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

"""Volume-level bootstrap 95% CI for the T10 FID comparisons (ticket T10, issue #28).

Spec #13 §⑤.0 requires every reported FID to carry a volume-level bootstrap
95% CI, and §⑤.2's adjudication table reads the CI overlap relation between
the frozen pretrained baseline and a finetuned snapshot. T8 deferred the CI
computation here because 24 comparisons x 1000 resamples of scipy ``sqrtm``
would have taken ~20 h inline; the T10 suite is 40 comparisons x 18 snapshots,
so the same math needs the symmetric-matrix reformulation: the Frechet
distance's matrix square root becomes two ``eigh`` calls, which on the
cluster's 128-core CPU parallelizes over comparisons to minutes per snapshot.

The statistic is resampling volumes with replacement on both sides
independently — the standard nonparametric bootstrap for a two-sample
distance — and taking the 2.5/97.5 percentiles. Per comparison the three
orthogonal planes are resampled with the same volume indices: one bootstrap
replicate is one resampled set of volumes, not one resampled set of slices.

Runs after (or alongside) the GPU suite; reads only the frozen feature cache
and writes ``ci/<tag>.json`` next to each snapshot's ``fid/`` results. It is
resumable the same way: an existing ci json is not recomputed.
"""

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch


class EighFrechetDistance:
    """Fréchet distance between feature samples via symmetric eigendecomposition.

    Numerically the same statistic MONAI's FIDMetric computes with scipy's
    general ``sqrtm``; ``eigh`` on the symmetrized formulation is both faster
    and stable on the rank-deficient (n < d) sample covariances that 50-volume
    comparisons produce.
    """

    @staticmethod
    def fid(x: np.ndarray, y: np.ndarray) -> float:
        mean_x, cov_x = EighFrechetDistance._moments(x)
        mean_y, cov_y = EighFrechetDistance._moments(y)

        eval_x, evec_x = np.linalg.eigh(cov_x)
        sqrt_x = (evec_x * np.sqrt(np.clip(eval_x, 0.0, None))) @ evec_x.T

        cross = sqrt_x @ cov_y @ sqrt_x
        cross = (cross + cross.T) / 2.0
        trace_sqrt_cross = float(np.sqrt(np.clip(np.linalg.eigvalsh(cross), 0.0, None)).sum())

        diff = mean_x - mean_y
        return float(diff @ diff + np.trace(cov_x) + np.trace(cov_y) - 2.0 * trace_sqrt_cross)

    @staticmethod
    def _moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        return mean, cov


def bootstrap_fid_distribution(x: np.ndarray, y: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """FID values over ``n_resamples`` volume-level resamples of both sides."""
    rng = np.random.default_rng(seed)
    n_x, n_y = x.shape[0], y.shape[0]
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        resampled_x = x[rng.integers(0, n_x, size=n_x)]
        resampled_y = y[rng.integers(0, n_y, size=n_y)]
        samples[i] = EighFrechetDistance.fid(resampled_x, resampled_y)
    return samples


def _stack_planes(features_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(features_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt features under {features_dir}")
    per_plane: list[list[np.ndarray]] = [[], [], []]
    for path in files:
        feats = torch.load(path, weights_only=True)
        for plane_index in range(3):
            per_plane[plane_index].append(feats[plane_index].numpy().astype(np.float64))
    return tuple(np.vstack(plane) for plane in per_plane)  # type: ignore[return-value]


class VolumeBootstrapCi:
    """One comparison's per-plane point FID plus its volume-level bootstrap 95% CI."""

    def __init__(self, real_features_dir: Path, synth_features_dir: Path, comparison_tag: str, n_resamples: int = 1000, seed: int = 42) -> None:
        self._real_dir = Path(real_features_dir)
        self._synth_dir = Path(synth_features_dir)
        self._tag = comparison_tag
        self._n_resamples = n_resamples
        self._seed = seed

    def compute(self) -> dict:
        real_planes = _stack_planes(self._real_dir)
        synth_planes = _stack_planes(self._synth_dir)

        payload: dict = {
            "comparison_tag": self._tag,
            "n_resamples": self._n_resamples,
            "seed": self._seed,
            "real_features_dir": str(self._real_dir),
            "synth_features_dir": str(self._synth_dir),
        }
        for plane_name, x, y in zip(("fid_xy", "fid_yz", "fid_zx"), synth_planes, real_planes):
            point = EighFrechetDistance.fid(x, y)
            samples = bootstrap_fid_distribution(x, y, self._n_resamples, self._seed)
            payload[plane_name] = {
                "point": point,
                "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            }
        payload["fid_avg"] = {
            "point": sum(payload[name]["point"] for name in ("fid_xy", "fid_yz", "fid_zx")) / 3.0,
            "ci95": [
                sum(payload[name]["ci95"][0] for name in ("fid_xy", "fid_yz", "fid_zx")) / 3.0,
                sum(payload[name]["ci95"][1] for name in ("fid_xy", "fid_yz", "fid_zx")) / 3.0,
            ],
        }
        return payload

    def write(self, output_json: Path) -> None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(self.compute(), indent=2) + "\n")


class BootstrapSuiteRunner:
    """Runs the CI stage over a task table's snapshots, one worker per comparison."""

    def __init__(self, task_table: dict, snapshot_run_ids: list[str] | None, n_resamples: int, seed: int, workers: int) -> None:
        self._modality = task_table["protocol"]["modality"]
        self._features_root = Path(task_table["assets"]["features_root"])
        self._n_resamples = n_resamples
        self._seed = seed
        self._workers = workers
        self._comparisons: list[dict] = [
            comparison
            for snapshot in task_table["snapshots"]
            if snapshot_run_ids is None or snapshot["run_id"] in snapshot_run_ids
            for comparison in snapshot["comparisons"]
            if comparison["kind"] in ("gen_real", "gen_gen")
        ]

    def run(self) -> None:
        pending = [c for c in self._comparisons if not self._already_done(c)]
        print(f"bootstrap CI: {len(pending)} of {len(self._comparisons)} comparisons pending", flush=True)
        if not pending:
            return
        with ProcessPoolExecutor(max_workers=self._workers) as pool:
            for tag in pool.map(self._run_one, pending):
                print(f"  ci done: {tag}", flush=True)

    def _already_done(self, comparison: dict) -> bool:
        return self._output_path(comparison).exists()

    def _output_path(self, comparison: dict) -> Path:
        result_json = Path(comparison["result_json"])
        return result_json.parent.parent / "ci" / result_json.name

    def _run_one(self, comparison: dict) -> str:
        real_dir = self._features_root / self._modality / comparison["real_features_dir"]
        synth_dir = self._features_root / self._modality / comparison["synth_features_dir"]
        # Per-worker seed: tag-derived so workers are deterministic and distinct
        # (str hash() is process-randomized; crc32 is not).
        seed = self._seed + (zlib.crc32(comparison["tag"].encode()) % 10000)
        ci = VolumeBootstrapCi(
            real_features_dir=real_dir,
            synth_features_dir=synth_dir,
            comparison_tag=comparison["tag"],
            n_resamples=self._n_resamples,
            seed=seed,
        )
        ci.write(self._output_path(comparison))
        return comparison["tag"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_table", type=Path, required=True)
    parser.add_argument("--snapshots", nargs="*", default=None, help="run_ids (default: every snapshot in the table)")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    with open(args.task_table) as file:
        task_table = json.load(file)

    runner = BootstrapSuiteRunner(
        task_table=task_table,
        snapshot_run_ids=args.snapshots,
        n_resamples=args.n_resamples,
        seed=args.seed,
        workers=args.workers,
    )
    runner.run()


if __name__ == "__main__":
    main()
