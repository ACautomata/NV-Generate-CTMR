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

"""The unique measurement entry point (ADR-0010, issue #109).

``InstrumentMeasurer.measure(pred, *, gt=None, condition=None, brain=None)``
composes the measurement collaborators (``RegionMasks`` / ``HierarchyChecker``
/ ``CalibrationCaseUsability`` / ``DiceScore``) into one ``CaseMeasurement``:
a pure transform, numpy in / value object out, no file IO, no cluster or path
coupling (READERS stay with the callers -- alignment, the DM-RAS->LPS flip and
the P2 combined-mask remap are input-adaptation concerns, not measurement).

Column families are gated on the optional references (ADR-0010 decision 2):
generation columns (volumes / centroids / ``pred_empty`` /
``hierarchy_violation``) and ``et_wt`` are always produced; the calibration
family needs ``gt``; ``condition_dice`` needs ``condition`` (already in
instrument label space, 0..3); ``brain_ml`` / ``wt_brain`` need the union of
the four input channels' non-zero voxels (``brain``), from which the count ->
ml and ``wt_brain`` division happen here.

All physical units assume the frozen 1 mm isotropic instrument grid
(ADR-0008): 1 voxel = 0.001 mL, spacing (1,1,1). Connected-component stats use
26-connectivity via ``scipy.ndimage.label`` -- the component partition is a
labelling-invariant graph property, so its counts are identical to the
calibration mother's ``cc3d(connectivity=26)`` output (the frozen ADR-0002
aggregates do not consume component counts).
"""

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from ctmr.domain.measurement.hierarchy import CalibrationCaseUsability, HierarchyChecker
from ctmr.domain.measurement.measurement import CaseMeasurement, GtRegionMetrics
from ctmr.domain.measurement.metrics import DiceScore
from ctmr.domain.measurement.regions import REGION_NAMES, SPACING_MM, VOXEL_ML, RegionMasks

_26_CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


class InstrumentMeasurer:
    """Pure masks -> one canonical measurement, the unique entry (ADR-0010 decision 2)."""

    def measure(
        self,
        pred: np.ndarray,
        *,
        gt: np.ndarray | None = None,
        condition: np.ndarray | None = None,
        brain: Sequence[np.ndarray] | None = None,
    ) -> CaseMeasurement:
        """Measures one prediction; optional references open their column family.

        ``brain`` is the observation's modality channels (protocol §3: the
        brain volume is the four-channel union of non-zero voxels); this
        module owns that union -- callers pass the channel arrays they already
        read, never a pre-summed volume.
        """
        pred_masks = RegionMasks(pred)
        volumes = {region: float(pred_masks.of(region).sum()) * VOXEL_ML for region in REGION_NAMES}
        centroids = {region: self._centroid(pred_masks.of(region)) for region in REGION_NAMES}

        brain_ml = wt_brain = None
        if brain is not None:
            brain_ml = self._brain_volume_ml(brain)
            wt_brain = volumes["WT"] / brain_ml if brain_ml > 0 else None

        condition_dice: dict[str, float | None] | None = None
        if condition is not None:
            condition_masks = RegionMasks(condition)
            condition_dice = {region: DiceScore.of(condition_masks.of(region), pred_masks.of(region)) for region in REGION_NAMES}

        gt_metrics = calibration_usable = et_wt_ratio_gt = None
        if gt is not None:
            gt_masks = RegionMasks(gt)
            gt_metrics = {region: self._gt_region(gt_masks.of(region), pred_masks.of(region), region) for region in REGION_NAMES}
            calibration_usable = CalibrationCaseUsability.usable(gt, pred)
            wt_gt = float(gt_masks.of("WT").sum())
            et_wt_ratio_gt = float(gt_masks.of("ET").sum()) / wt_gt if wt_gt > 0 else None

        return CaseMeasurement(
            hierarchy_violation=HierarchyChecker.violates(pred),
            pred_empty=not bool(pred_masks.of("WT").any()),
            volumes_ml=volumes,
            centroids_mm=centroids,
            et_wt=volumes["ET"] / volumes["WT"] if volumes["WT"] > 0 else None,
            brain_ml=brain_ml,
            wt_brain=wt_brain,
            condition_dice=condition_dice,
            gt_metrics=gt_metrics,
            calibration_usable=calibration_usable,
            et_wt_ratio_gt=et_wt_ratio_gt,
        )

    @staticmethod
    def _brain_volume_ml(channels: Sequence[np.ndarray]) -> float:
        """Protocol §3 brain rule: four-channel union of non-zero voxels, in mL."""
        union: np.ndarray | None = None
        for array in channels:
            nonzero = array > 0
            union = nonzero if union is None else (union | nonzero)
        return float(union.sum()) * VOXEL_ML

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float, float] | None:
        """Physical (x, y, z) mm centroid of a region mask; None when empty."""
        if not mask.any():
            return None
        cz, cy, cx = ndimage.center_of_mass(mask)  # zyx voxel indices on the 1 mm grid
        return float(cx), float(cy), float(cz)

    def _gt_region(self, gt_mask: np.ndarray, pred_mask: np.ndarray, region: str) -> GtRegionMetrics:
        """The calibration per-region readout, verbatim mother arithmetic (ADR-0002)."""
        vol_gt = float(gt_mask.sum()) * VOXEL_ML
        vol_pred = float(pred_mask.sum()) * VOXEL_ML
        intersection = int(np.logical_and(gt_mask, pred_mask).sum())
        gt_voxels = int(gt_mask.sum())
        pred_voxels = int(pred_mask.sum())
        sensitivity: float | None
        if pred_voxels == 0:
            sensitivity = 0.0
        elif gt_voxels > 0:
            sensitivity = intersection / float(gt_voxels)
        else:
            sensitivity = None  # mother: math.nan -- undefined both-sides empty, registered unification
        precision: float | None
        if pred_voxels == 0:
            precision = 0.0
        else:
            precision = intersection / float(pred_voxels)
        n_gt, n_pred, n_fp = self._component_counts(gt_mask, pred_mask)
        return GtRegionMetrics(
            vol_gt_ml=vol_gt,
            vol_pred_ml=vol_pred,
            signed_bias_ml=vol_pred - vol_gt,
            abs_err_ml=abs(vol_pred - vol_gt),
            rel_vol_err=abs(vol_pred - vol_gt) / vol_gt if vol_gt > 0 else None,
            dice=DiceScore.of(gt_mask, pred_mask),
            sensitivity=sensitivity,
            precision=precision,
            hd95_mm=self._hd95_mm(gt_mask, pred_mask),
            centroid_distance_mm=self._centroid_distance_mm(gt_mask, pred_mask),
            n_components_gt=n_gt,
            n_components_pred=n_pred,
            n_false_positive_components=n_fp if region == "WT" else None,  # false-positive lesions are a WT notion
        )

    @staticmethod
    def _hd95_mm(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float | None:
        """HD95 (mm) = max(p95(d_gt_to_pred), p95(d_pred_to_gt)); None when a surface is empty."""
        gt_surface = gt_mask ^ ndimage.binary_erosion(gt_mask)
        pred_surface = pred_mask ^ ndimage.binary_erosion(pred_mask)
        if gt_surface.sum() == 0 or pred_surface.sum() == 0:
            return None  # mother: math.nan
        distance_to_gt = ndimage.distance_transform_edt(~gt_mask, sampling=SPACING_MM)
        distance_to_pred = ndimage.distance_transform_edt(~pred_mask, sampling=SPACING_MM)
        gt_to_pred = distance_to_pred[gt_surface]
        pred_to_gt = distance_to_gt[pred_surface]
        return float(max(np.quantile(gt_to_pred, 0.95), np.quantile(pred_to_gt, 0.95)))

    @staticmethod
    def _centroid_distance_mm(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float | None:
        """Euclidean GT/pred centroid distance (mm); None when either mask is empty."""
        if gt_mask.sum() == 0 or pred_mask.sum() == 0:
            return None  # mother: math.nan
        gt_center = np.array(ndimage.center_of_mass(gt_mask))
        pred_center = np.array(ndimage.center_of_mass(pred_mask))
        return float(np.linalg.norm((gt_center - pred_center) * np.array(SPACING_MM)))

    @staticmethod
    def _component_counts(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[int, int, int]:
        """(n_gt, n_pred, pred components with zero GT overlap) under 26-connectivity."""
        gt_labels, n_gt = ndimage.label(gt_mask, structure=_26_CONNECTIVITY)
        pred_labels, n_pred = ndimage.label(pred_mask, structure=_26_CONNECTIVITY)
        n_fp = 0
        if n_gt > 0 and n_pred > 0:
            overlap = np.unique(pred_labels[gt_labels > 0])
            n_fp = n_pred - (len(overlap) - (1 if 0 in overlap else 0))
        return int(n_gt), int(n_pred), int(n_fp)
