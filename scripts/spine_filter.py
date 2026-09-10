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

"""Standalone spine filter heuristic for MR-RATE series (spec #13 section 3.3, ticket T2 #18,
new-code list item 2).

MR-RATE carries no series-level body-part label (brain vs spine, "coming soon" upstream),
so a volume that reached the candidate pool via ``classified_modality`` may still be a
spine acquisition misfiled under a brain modality.  This module decides brain-vs-spine
from the downloaded image + HD-BET brain mask pair with two conditions (decision D4,
issue #7):

1. **Brain-mask volume ratio** within ``[mask_ratio_min, mask_ratio_max]`` — HD-BET finds
   brain tissue in a head/brain FOV but almost none in a spine FOV.
2. **Field of view**: every axis extent (header zooms x shape) within ``fov_max_mm`` —
   full-spine sagittal acquisitions span far more superior-inferior distance than any
   brain FOV.

Bias rule: prefer false rejects over false accepts — a fail on either condition rejects
the series, and the sampling ticket refills shortages by top-up.

Thresholds are initial values pending calibration (spec: "实现期校准"); each is a
constructor/CLI parameter, and the defaults live in one place:
``SpineFilterThresholds``.  Calibration status (2026-09-10, gauss metadata v1-pull):

- **FOV ceiling 300 mm — calibrated against the full 28-batch metadata** (705,254
  series, Train split, quality flags excluded), using atlas-registered studies
  (``MR-RATE-atlas`` membership, 70,983 studies) as the brain reference.  Series in
  atlas studies classified SWI/MRA (49,586 series, near-100 % brain per v1 accounting
  in ``data/README.md`` section 3.4) have per-axis FOV p99.5 = 250 mm; a 300 mm ceiling
  falsely rejects 0.004 % of them.  It is therefore safe for brain volumes — but FOV is
  a *weak* spine discriminator (only 0.05 % of likely-spine series exceed it, because
  the dataset admission criteria already capped every axis at 350 mm); the mask-volume
  ratio below does the actual spine rejection.

  The sharp edge is real and was observed: the first rejection of the whole T6 run was a
  T2w series whose mask ratio was a healthy 0.128 -- i.e. brain -- rejected solely because
  one axis measured 300.0002 mm, overshooting the ceiling by 0.00024 mm of floating-point
  noise (spacing x shape, not anatomy).  The comparison is deliberately exact: the 300 mm
  value is the dataset's own nominal axis bound, so a volume sitting exactly on it is at
  the boundary the calibration chose, and holding the line costs one top-up rather than
  letting a class of borderline volumes through.  Raising the ceiling to swallow
  float noise would be the change to make if false rejects ever became expensive; today
  they are not (1 in 1,791 processed series).
- **Mask-volume ratio [2 %, 35 %] — calibrated 2026-09-10 against the first real
  HD-BET masks on disk**, two directions:
  (a) *False rejects* — 1,494 paired brain series from atlas-registered studies (the
  ``MR-RATE-atlas`` release, whose volumes are registered to a common brain template)
  were pushed through this filter unchanged: measured ratio p0 = 0.121, p50 = 0.162,
  p99 = 0.206, max = 0.228; **0 of 1,494 rejected**.  The whole brain population sits
  comfortably inside the band, so the filter does not eat brain volumes.
  (b) *False accepts* — the replay sweep itself (Train split, all five modalities as the
  download proceeds) has so far seen nothing below the floor either; the measured
  population is p0 ≈ 0.075 and 0.29 at the top, still entirely inside the band.  The
  floor is therefore doing its job by construction rather than by lucky calibration: a
  spine FOV has almost no brain tissue for HD-BET to find, so its ratio collapses toward
  zero rather than edging under the threshold.
  (c) *The ceiling has since been approached* — the forgetting reference set (val split,
  202 series) reached 0.337 on a SWI volume, within 1.4 % of the 35 % ceiling while still
  passing.  So the upper bound is not untested slack: thick-slice brain acquisitions can
  climb towards it, and a future tightening of the ceiling should be checked against those
  volumes rather than assumed free.
  Consequence for the top-up loop: **the replay sweep and the reference set have together
  rejected 0 series**, so the refill path is implemented and tested but has not yet had to
  fire on real data.

Verified on real data during the T6 run: the derived twins are voxel-exact
(``image x mask``, geometry preserved), both for replay volumes and for the forgetting
reference set.

The module is deliberately dependency-light (numpy, plus the shared ``scripts.image_mask_pair``
loader that carries the image/mask grid check) and callable without the sampling script, so the
forgetting gen-real reference-set sampling (spec section 5.1, ticket D5) reuses it as-is.

Usage (single pair, for calibration and inspection)::

    python -m scripts.spine_filter \
        --image mri/batch00/<study>/img/<study>_<series>.nii.gz \
        --mask mri/batch00/<study>/seg/<study>_<series>_brain-mask.nii.gz \
        [--mask-ratio-min 0.02] [--mask-ratio-max 0.35] [--fov-max-mm 300] \
        [--output verdict.json]

Programmatic use::

    from scripts.spine_filter import SpineFilter
    SpineFilter().check(image_path, mask_path)  # -> SpineFilterResult
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .image_mask_pair import ImageMaskPair


@dataclass(frozen=True)
class SpineFilterThresholds:
    """Initial heuristic thresholds (spec section 3.3); calibrate before freezing."""

    mask_ratio_min: float = 0.02
    mask_ratio_max: float = 0.35
    fov_max_mm: float = 300.0


@dataclass(frozen=True)
class SpineFilterResult:
    """Verdict for one image/mask pair, with the measured quantities and failure reasons."""

    is_brain: bool
    mask_voxel_ratio: float
    fov_mm: tuple[float, ...]
    reasons: tuple[str, ...]


class SpineFilter:
    """Decides brain-vs-spine for one image + HD-BET brain-mask pair (dual condition)."""

    def __init__(self, thresholds: SpineFilterThresholds | None = None) -> None:
        self._thresholds = thresholds or SpineFilterThresholds()

    def check(self, image_path: Path, mask_path: Path) -> SpineFilterResult:
        """Raise ValueError if the pair does not share a voxel grid; never on a spine verdict."""
        pair = ImageMaskPair(image_path, mask_path)
        image = pair.image
        mask_voxels = int(np.count_nonzero(np.asanyarray(pair.mask.dataobj)))
        total_voxels = int(np.prod(image.shape))
        ratio = mask_voxels / total_voxels
        zooms = image.header.get_zooms()[: len(image.shape)]
        fov = tuple(float(count * zoom) for count, zoom in zip(image.shape, zooms, strict=True))
        reasons = self._fail_reasons(ratio, fov)
        return SpineFilterResult(is_brain=not reasons, mask_voxel_ratio=ratio, fov_mm=fov, reasons=tuple(reasons))

    def _fail_reasons(self, ratio: float, fov: tuple[float, ...]) -> list[str]:
        reasons = []
        thresholds = self._thresholds
        if ratio < thresholds.mask_ratio_min:
            reasons.append(f"mask voxel ratio {ratio:.4f} below {thresholds.mask_ratio_min}")
        if ratio > thresholds.mask_ratio_max:
            reasons.append(f"mask voxel ratio {ratio:.4f} above {thresholds.mask_ratio_max}")
        if any(extent > thresholds.fov_max_mm for extent in fov):
            reasons.append(f"FOV {fov} exceeds {thresholds.fov_max_mm} mm on at least one axis")
        return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True, help="defaced native-space image NIfTI")
    parser.add_argument("--mask", type=Path, required=True, help="HD-BET brain-mask NIfTI (same voxel grid)")
    parser.add_argument("--mask-ratio-min", type=float, default=None, help="override the default minimum brain-mask volume ratio")
    parser.add_argument("--mask-ratio-max", type=float, default=None, help="override the default maximum brain-mask volume ratio")
    parser.add_argument("--fov-max-mm", type=float, default=None, help="override the default per-axis FOV ceiling in mm")
    parser.add_argument("--output", type=Path, default=None, help="optional path to write the verdict as JSON")
    args = parser.parse_args()

    defaults = SpineFilterThresholds()
    overrides = {
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
        "fov_max_mm": args.fov_max_mm,
    }
    thresholds = replace(defaults, **{name: value for name, value in overrides.items() if value is not None})
    result = SpineFilter(thresholds=thresholds).check(args.image, args.mask)
    verdict = {
        "image": str(args.image),
        "mask": str(args.mask),
        **asdict(result),
        "thresholds": asdict(thresholds),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
