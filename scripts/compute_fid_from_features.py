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

"""FID between two already-extracted feature directories, no GPU and no volumes needed.

Ticket T10 (issue #28) runs 44 comparisons per snapshot. In every one of them the
frozen-reference side (T8/T17 baseline gen, MR-RATE val references, BRATS floor
halves) already sits in the feature cache, and the gen-gen comparisons are frozen
features on both sides. Running those through ``compute_fid_2-5d_ct`` would pay a
full torch/monai import and network load per comparison only to re-serve cached
features, so this script computes the identical statistic directly: stack every
``.pt`` feature tuple in both directories, run MONAI ``FIDMetric`` per orthogonal
plane, and write the same ``FidResult`` json the GPU tool emits (the summarizer
and the adjudicator cannot tell the difference).

Feature provenance is inherited from however the directories were produced --
extraction-time preprocessing lives with the cache, not here. The two sides
need not hold the same number of feature files: each side is stacked in full
(the BRATS reference is 64 real cases against 50 synthetic volumes), and a
mismatch is expected rather than refused.
"""

import argparse
import importlib
from pathlib import Path

import torch
from monai.metrics.fid import FIDMetric

compute_fid = importlib.import_module("scripts.compute_fid_2-5d_ct")

FidResult = compute_fid.FidResult


class FeatureDirectoryStack:
    """Reads one feature-cache directory into stacked per-plane feature tensors."""

    def __init__(self, features_dir: Path, side: str) -> None:
        self._dir = Path(features_dir)
        self._side = side

    def stack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        files = sorted(self._dir.rglob("*.pt"))
        if not files:
            raise FileNotFoundError(f"{self._side} features directory has no .pt files: {self._dir}")
        per_plane: list[list[torch.Tensor]] = [[], [], []]
        for path in files:
            # map_location: the T8/T17 caches were written from GPU extraction and
            # hold CUDA tensors; this tool runs CPU-only.
            feats = torch.load(path, weights_only=True, map_location="cpu")
            for plane_index in range(3):
                per_plane[plane_index].append(feats[plane_index].cpu())
        return tuple(torch.vstack(plane) for plane in per_plane)  # type: ignore[return-value]


class FeatureSetFidCalculator:
    """Stacks two feature directories into per-plane FID and a provenance-carrying result."""

    def __init__(
        self,
        real_features_dir: Path,
        synth_features_dir: Path,
        comparison_tag: str,
        num_images: int,
        real_filelist: str = "",
        synth_filelist: str = "",
        modality: str = "mr",
        model_name: str = "radimagenet_resnet50",
        target_shape: str = "256x256x128",
        center_slices_ratio: float | None = 0.4,
        enable_padding: bool | None = True,
        enable_center_cropping: bool | None = True,
        enable_resampling_spacing: str | None = "1.0x1.0x1.0",
    ) -> None:
        self._real_dir = Path(real_features_dir)
        self._synth_dir = Path(synth_features_dir)
        self._tag = comparison_tag
        self._provenance = {
            "num_images": num_images,
            "real_filelist": real_filelist,
            "synth_filelist": synth_filelist,
            "modality": modality,
            "model_name": model_name,
            "target_shape": target_shape,
            "center_slices_ratio": center_slices_ratio,
            "enable_padding": enable_padding,
            "enable_center_cropping": enable_center_cropping,
            "enable_resampling_spacing": enable_resampling_spacing,
        }

    def compute(self) -> FidResult:
        real = FeatureDirectoryStack(self._real_dir, "real").stack()
        synth = FeatureDirectoryStack(self._synth_dir, "synth").stack()

        fid = FIDMetric()
        fid_xy = float(fid(synth[0], real[0]))
        fid_yz = float(fid(synth[1], real[1]))
        fid_zx = float(fid(synth[2], real[2]))
        return FidResult(
            comparison_tag=self._tag,
            fid_xy=fid_xy,
            fid_yz=fid_yz,
            fid_zx=fid_zx,
            fid_avg=(fid_xy + fid_yz + fid_zx) / 3.0,
            **self._provenance,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real_features_dir", type=Path, required=True)
    parser.add_argument("--synth_features_dir", type=Path, required=True)
    parser.add_argument("--comparison_tag", required=True)
    parser.add_argument("--num_images", type=int, required=True)
    parser.add_argument("--real_filelist", default="")
    parser.add_argument("--synth_filelist", default="")
    parser.add_argument("--modality", default="mr")
    parser.add_argument("--model_name", default="radimagenet_resnet50")
    parser.add_argument("--target_shape", default="256x256x128")
    parser.add_argument("--center_slices_ratio", type=float, default=0.4)
    parser.add_argument("--enable_padding", default=True)
    parser.add_argument("--enable_center_cropping", default=True)
    parser.add_argument("--enable_resampling_spacing", default="1.0x1.0x1.0")
    parser.add_argument("--result_json", type=Path, required=True)
    args = parser.parse_args()

    calculator = FeatureSetFidCalculator(
        real_features_dir=args.real_features_dir,
        synth_features_dir=args.synth_features_dir,
        comparison_tag=args.comparison_tag,
        num_images=args.num_images,
        real_filelist=args.real_filelist,
        synth_filelist=args.synth_filelist,
        modality=args.modality,
        model_name=args.model_name,
        target_shape=args.target_shape,
        center_slices_ratio=args.center_slices_ratio,
        enable_padding=args.enable_padding,
        enable_center_cropping=args.enable_center_cropping,
        enable_resampling_spacing=args.enable_resampling_spacing,
    )
    result = calculator.compute()
    result.save(args.result_json)
    print(f"FID for {args.comparison_tag}: xy={result.fid_xy:.4f} yz={result.fid_yz:.4f} zx={result.fid_zx:.4f} avg={result.fid_avg:.4f} -> {args.result_json}")


if __name__ == "__main__":
    main()
