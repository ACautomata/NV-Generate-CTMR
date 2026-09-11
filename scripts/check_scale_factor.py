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

"""Pre-finetune scale_factor sanity gate: replay latent std vs the v1 scale_factor
(spec #13 sections 4.5 / 8-C4, ticket T7 #23).

Training recomputes ``scale_factor`` from the mixed data's first batch at startup and never
reads the stored value (``diff_model_train.calculate_scale_factor``), and inference later
reads whatever the checkpoint saved -- so a wildly off latent distribution would not crash
anything, it would silently move every condition's operating point.  This check runs
*before* training starts and turns that failure mode into a hard stop: sample the replay
latents, estimate their std, compare the implied scale_factor against v1's, and refuse to
let training start when the deviation passes the threshold ("超阈值 ⇒ 阻断并报告预处理 OOD
排查，不放行训练").

Reference value: ``--v1-ckpt`` reads ``checkpoint["scale_factor"]`` from the published v1
checkpoint -- exactly the number v1 inference uses (``diff_model_infer.py`` reads the same
key), i.e. v1's declared operating point.  ``--reference-scale-factor`` overrides when the
number comes from elsewhere.

Estimator: the training code computes ``1 / torch.std(first batch)`` with ``batch_size=1``,
i.e. one volume's whole-tensor std; this check draws a stratified sample (seeded shuffle per
label, one of each label per round -- every old label is represented whatever N), takes the
mean per-volume std, and reports ``1 / mean_std``.  ``float32`` and the Bessel-corrected
``ddof=1`` match the training estimator, whose ``torch.std`` is unbiased by default.  The
gate decides in the scale_factor domain (what v1 declares): the report also records the
latent-std-domain deviation, which reads slightly larger for the same shift.

Threshold (the spec's "实现期定" decision, default ``0.2``): replay latents share v1's
training distribution *and* its unmodified preprocessing pipeline, so gross preprocessing OOD
(skipped intensity normalization, a wrong orientation/resample, a mask applied to the wrong
half of the dual derivation) moves the latent std by multiples, not percent, while the
stratified mean over tens of volumes concentrates within a few percent.  A 20% relative
deviation therefore sits far above sampling noise and far below the OOD shifts it must catch;
it is a tripwire, not a calibration, and training's own mixed-data recompute absorbs moderate
drift anyway.

Passing writes the report and exits 0; failing writes the report, prints the OOD instruction
and exits 1 -- the training launch must not proceed until the preprocessing is re-verified.

Usage (on gauss, after the replay latents and sidecars are in place)::

    uv run python -m scripts.check_scale_factor \\
        --dataset-json $RUN_ROOT/runs/replay-latent-20260910/dataset_replay_rflow-mr-brain_N1000.json \\
        --embedding-base-dir $RUN_ROOT/runs/replay-latent-20260910/data \\
        --v1-ckpt $RUN_ROOT/models/diff_unet_3d_rflow-mr-brain_v1.pt \\
        --report $RUN_ROOT/runs/replay-latent-20260910/scale_factor_sanity.json
"""

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scripts.latent_sidecars import LatentEntry

DEFAULT_SEED = 42
DEFAULT_N_SAMPLES = 64
DEFAULT_MAX_RELATIVE_DEVIATION = 0.2
LATENT_DTYPE = np.float32


class ScaleFactorReference:
    """The v1 scale_factor the finetune run is sanity-checked against, and where it came from."""

    def __init__(self, value: float, source: str) -> None:
        if value <= 0:
            raise ValueError(f"scale_factor must be positive, got {value}")
        self.value = value
        self.source = source

    @classmethod
    def from_checkpoint(cls, ckpt_path: Path) -> "ScaleFactorReference":
        """Read ``checkpoint["scale_factor"]`` -- the key ``diff_model_train`` saves and ``diff_model_infer`` consumes."""
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "scale_factor" not in checkpoint:
            raise ValueError(f"{ckpt_path}: checkpoint holds no scale_factor (keys: {sorted(checkpoint.keys())})")
        return cls(value=float(checkpoint["scale_factor"]), source=str(ckpt_path))

    @classmethod
    def from_value(cls, value: float) -> "ScaleFactorReference":
        """An explicitly passed reference (``--reference-scale-factor``)."""
        return cls(value=value, source="--reference-scale-factor")

    @property
    def implied_training_std(self) -> float:
        """The latent std v1's scale_factor implies (``scale_factor = 1/std`` at training time)."""
        return 1.0 / self.value


class StratifiedLatentSample:
    """A seeded, label-balanced draw over the dataset.json latents, with one std per drawn latent.

    Balance comes from drawing round-robin across the labels (one of each per round), each
    label's pool first shuffled with the same ``sha256(seed:label)`` derivation
    ``scripts.create_replay_manifest`` uses -- a smaller N shrinks every label's share instead
    of dropping whole labels, and the same seed redraws the same latents.
    """

    def __init__(
        self,
        entries: list[LatentEntry],
        embedding_base_dir: Path,
        n_samples: int,
        seed: int = DEFAULT_SEED,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        self._embedding_base_dir = embedding_base_dir
        self._n_samples = n_samples
        self._seed = seed
        self._pools = self._label_pools(entries, labels)

    def draw(self) -> list[tuple[LatentEntry, float]]:
        """The drawn latents with their whole-tensor std, round-robin across labels."""
        queues = {label: iter(pool) for label, pool in self._shuffled_pools().items()}
        drawn: list[LatentEntry] = []
        exhausted = set()
        while len(drawn) < self._n_samples and len(exhausted) < len(queues):
            for label, queue in queues.items():
                if label in exhausted:
                    continue
                entry = next(queue, None)
                if entry is None:
                    exhausted.add(label)
                    continue
                drawn.append(entry)
                if len(drawn) == self._n_samples:
                    break
        return [(entry, self._latent_std(entry)) for entry in drawn]

    def _label_pools(self, entries: list[LatentEntry], labels: tuple[str, ...] | None) -> dict[str, list[LatentEntry]]:
        pools: dict[str, list[LatentEntry]] = {}
        for entry in entries:
            if labels is None or entry.modality in labels:
                pools.setdefault(entry.modality, []).append(entry)
        if labels is not None:
            missing = sorted(set(labels) - set(pools))
            if missing:
                raise ValueError(f"no dataset.json entries for label(s): {', '.join(missing)}")
        if not pools:
            raise ValueError("the dataset.json holds no training entries")
        return pools

    def _shuffled_pools(self) -> dict[str, list[LatentEntry]]:
        shuffled = {}
        for label, pool in self._pools.items():
            ordered = sorted(pool, key=lambda entry: entry.image)
            random.Random(self._label_seed(label)).shuffle(ordered)
            shuffled[label] = ordered
        return shuffled

    def _label_seed(self, label: str) -> int:
        digest = hashlib.sha256(f"{self._seed}:{label}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def _latent_std(self, entry: LatentEntry) -> float:
        latent_path = self._embedding_base_dir / entry.embedding_relative_path
        values = np.asarray(nib.load(str(latent_path)).dataobj, dtype=LATENT_DTYPE)
        # Bessel-corrected: the training estimator's torch.std is unbiased by default.
        return float(np.std(values, ddof=1))


class ScaleFactorSanityCheck:
    """Compares the sampled replay latent std against the v1 reference; a deviation beyond the
    threshold blocks the training launch and records the verdict in a JSON report."""

    def __init__(
        self,
        reference: ScaleFactorReference,
        dataset_json: Path,
        embedding_base_dir: Path,
        n_samples: int,
        threshold: float,
        report_path: Path,
        seed: int = DEFAULT_SEED,
        labels: tuple[str, ...] | None = None,
    ) -> None:
        self._reference = reference
        self._sample = StratifiedLatentSample(
            entries=LatentEntry.from_dataset_json(dataset_json),
            embedding_base_dir=embedding_base_dir,
            n_samples=n_samples,
            seed=seed,
            labels=labels,
        )
        self._threshold = threshold
        self._report_path = report_path
        self._inputs = {
            "dataset_json": str(dataset_json),
            "embedding_base_dir": str(embedding_base_dir),
            "seed": seed,
            "labels": list(labels) if labels else None,
        }

    def run(self) -> dict:
        """Estimate, compare, write the report, print the verdict; return the report.

        The verdict alone decides nothing here -- blocking is the caller's contract, and
        ``main`` turns a BLOCK into exit code 1.
        """
        drawn = self._sample.draw()
        mean_std = sum(std for _entry, std in drawn) / len(drawn)
        estimate = 1.0 / mean_std
        deviation = abs(estimate - self._reference.value) / self._reference.value
        std_deviation = abs(mean_std - self._reference.implied_training_std) / self._reference.implied_training_std
        verdict = "BLOCK" if deviation > self._threshold else "PASS"

        report = {
            "verdict": verdict,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "reference": {
                "scale_factor": self._reference.value,
                "source": self._reference.source,
                "implied_training_std": self._reference.implied_training_std,
            },
            "estimate": {
                "latent_std_mean": mean_std,
                "scale_factor_estimate": estimate,
                "n_samples": len(drawn),
                "per_label": self._per_label(drawn),
                "samples": [
                    {"image": entry.image, "modality": entry.modality, "latent": entry.embedding_relative_path, "std": std} for entry, std in drawn
                ],
            },
            "comparison": {
                # The gate decides in the scale_factor domain (what v1 declares); the same shift
                # reads larger in the latent-std domain, so both are recorded.
                "domain": "scale_factor",
                "deviation_relative": deviation,
                "std_domain_deviation_relative": std_deviation,
                "threshold_relative": self._threshold,
            },
            "inputs": self._inputs,
        }
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        with self._report_path.open("w") as file:
            json.dump(report, file, indent=2)
            file.write("\n")

        print(
            f"reference scale_factor {self._reference.value:.6f} (from {self._reference.source}, implies training std {self._reference.implied_training_std:.6f})"
        )
        print(f"replay estimate: latent std {mean_std:.6f} over {len(drawn)} latents -> scale_factor {estimate:.6f}")
        print(
            f"relative deviation {deviation:.4f} in the scale_factor domain ({std_deviation:.4f} in the latent-std domain) vs threshold {self._threshold}"
        )
        per_label_text = ", ".join(f"{label}={stats['std_mean']:.6f}" for label, stats in report["estimate"]["per_label"].items())
        print(f"per-label std: {per_label_text}")
        print(f"verdict {verdict}; report -> {self._report_path}")
        if verdict == "BLOCK":
            print("BLOCKED: deviation exceeds the threshold -- preprocessing OOD suspected. Investigate the replay")
            print("preprocessing (intensity normalization, orientation/resize, dual derivation) before training;")
            print("training must not start on this data.")
        return report

    @staticmethod
    def _per_label(drawn: list[tuple[LatentEntry, float]]) -> dict[str, dict]:
        by_label: dict[str, list[float]] = {}
        for entry, std in drawn:
            by_label.setdefault(entry.modality, []).append(std)
        return {
            label: {"n": len(stdevs), "std_mean": sum(stdevs) / len(stdevs), "std_min": min(stdevs), "std_max": max(stdevs)}
            for label, stdevs in sorted(by_label.items())
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="a dataset.json naming the replay latents (the per-tier replay file, or the merged one with --labels)",
    )
    parser.add_argument("--embedding-base-dir", type=Path, required=True, help="root directory the latents live under")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--v1-ckpt", type=Path, help="the v1 checkpoint; its scale_factor key is the reference")
    source.add_argument("--reference-scale-factor", type=float, help="an explicit reference scale_factor instead of a checkpoint")
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES, help=f"latents to sample (default {DEFAULT_N_SAMPLES}, round-robin across labels)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"sampling seed (default {DEFAULT_SEED}; same seed redraws the same latents)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MAX_RELATIVE_DEVIATION,
        help=f"max relative deviation before blocking (default {DEFAULT_MAX_RELATIVE_DEVIATION})",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL", help="restrict the sample to these modality strings (default: all in the dataset.json)"
    )
    parser.add_argument("--report", type=Path, required=True, help="path of the JSON report the deviation is recorded in")
    args = parser.parse_args()

    reference = ScaleFactorReference.from_checkpoint(args.v1_ckpt) if args.v1_ckpt else ScaleFactorReference.from_value(args.reference_scale_factor)
    report = ScaleFactorSanityCheck(
        reference=reference,
        dataset_json=args.dataset_json,
        embedding_base_dir=args.embedding_base_dir,
        n_samples=args.n_samples,
        threshold=args.threshold,
        report_path=args.report,
        seed=args.seed,
        labels=tuple(args.labels) if args.labels else None,
    ).run()
    if report["verdict"] == "BLOCK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
