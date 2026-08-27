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

"""Convergence-gate tests for the unique entry (ADR-0010, #109).

Proves ``InstrumentMeasurer.measure`` against embedded verbatim snapshots of
the two frozen implementations it unifies -- the terminal-acceptance measurer
(``MaskMeasurer`` / ``InstrumentFailureChecker`` / ``MeasurementRunner`` row
assembly, pre-#109, no file IO) and the calibration mother's per-region
readout (``measure_case`` loop body, pre-#109). Same synthetic inputs,
bit-identical outputs on every measured quantity: that is the ADR-0010
convergence gate at unit level (the sugon byte-identical rerun of frozen paths
stays with the frozen gate, #T12).

Also pins the three-column-family gating: generation + ``et_wt`` constant,
calibration only with ``gt``, condition dice only with ``condition``,
brain/wt_brain only with ``brain``.
"""

import math
from dataclasses import asdict

import numpy as np
import pytest
from scipy import ndimage

from ctmr.domain.measurement.measurer import InstrumentMeasurer
from ctmr.domain.measurement.regions import RegionMasks

# ── frozen terminal-acceptance reference (verbatim, pre-#109) ───────────────


class FrozenTerminalReference:
    """Verbatim snapshot of the pre-#109 terminal-acceptance measurer.

    Copies of MaskMeasurer.volumes_ml / centroid_mm / condition_dice,
    InstrumentFailureChecker.hierarchy_violation and the MeasurementRunner row
    assembly (measurement part only -- the caller's file IO is replaced by
    direct numpy inputs). Do not edit -- this is the frozen ADR-0004 readout.
    """

    REGION_LABELS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}

    @classmethod
    def volumes_ml(cls, pred):
        return {region: float(np.isin(pred, labels).sum()) * 0.001 for region, labels in cls.REGION_LABELS.items()}

    @classmethod
    def centroid_mm(cls, pred, region):
        mask = np.isin(pred, cls.REGION_LABELS[region])
        if not mask.any():
            return None
        cz, cy, cx = ndimage.center_of_mass(mask)
        return (float(cx), float(cy), float(cz))

    @staticmethod
    def brain_ml_from_channels(channels):
        union = None
        for array in channels:
            nonzero = array > 0
            union = nonzero if union is None else (union | nonzero)
        return float(union.sum()) * 0.001

    @classmethod
    def condition_dice(cls, pred, condition, region):
        gt_mask = np.isin(condition, cls.REGION_LABELS[region])
        pred_mask = np.isin(pred, cls.REGION_LABELS[region])
        denom = int(gt_mask.sum()) + int(pred_mask.sum())
        if denom == 0:
            return None
        return float(2 * np.logical_and(gt_mask, pred_mask).sum() / denom)

    @staticmethod
    def hierarchy_violation(pred):
        wt = np.isin(pred, (1, 2, 3))
        tc = np.isin(pred, (1, 3))
        et = pred == 3
        outside_domain = not np.isin(pred, (0, 1, 2, 3)).all()
        return bool(outside_domain or (et & ~tc).any() or (tc & ~wt).any())

    @classmethod
    def wide_quantities(cls, pred, condition=None, brain_channels=None):
        """The frozen measured quantities, as the runner assembled them."""
        volumes = cls.volumes_ml(pred)
        centroid_values = {f"c{axis}_{region.lower()}_mm": None for axis in "xyz" for region in cls.REGION_LABELS}
        for region in cls.REGION_LABELS:
            centroid = cls.centroid_mm(pred, region)
            if centroid is not None:
                for axis, value in zip("xyz", centroid):
                    centroid_values[f"c{axis}_{region.lower()}_mm"] = value
        quantities = {
            "hier_viol": int(cls.hierarchy_violation(pred)),
            "pred_empty": int(not bool(np.isin(pred, (1, 2, 3)).any())),
            **{f"vol_{region.lower()}_ml": volumes[region] for region in cls.REGION_LABELS},
            "et_wt": volumes["ET"] / volumes["WT"] if volumes["WT"] > 0 else None,
            **centroid_values,
        }
        if brain_channels is not None:
            brain = cls.brain_ml_from_channels(brain_channels)
            quantities["brain_ml"] = brain
            quantities["wt_brain"] = volumes["WT"] / brain if brain > 0 else None
        if condition is not None:
            for region in cls.REGION_LABELS:
                quantities[f"cond_dice_{region.lower()}"] = cls.condition_dice(pred, condition, region)
        return quantities


# ── frozen calibration-mother reference (verbatim, pre-#109) ────────────────


class FrozenCalibrationReference:
    """Verbatim snapshot of the pre-#109 calibration mother per-region readout.

    ``measure_case`` loop body of nnunet_l2_calibration_metrics.py (dice_of /
    hd95_mm / centroid_distance_mm / component_stats / the region-row fields).
    The only difference is the sentinel policy the module must reproduce:
    mother ``math.nan`` == module ``None`` (registered ADR-0010 decision 4).
    Do not edit.
    """

    Z95 = 1.959963984540054

    @classmethod
    def dice_of(cls, gt, pred):
        denom = int(gt.sum()) + int(pred.sum())
        if denom == 0:
            return math.nan
        return float(2 * np.logical_and(gt, pred).sum() / denom)

    @classmethod
    def hd95_mm(cls, gt, pred, spacing_zyx):
        gt_surf = gt ^ ndimage.binary_erosion(gt)
        pred_surf = pred ^ ndimage.binary_erosion(pred)
        if gt_surf.sum() == 0 or pred_surf.sum() == 0:
            return math.nan
        dist_to_gt = ndimage.distance_transform_edt(~gt, sampling=spacing_zyx)
        dist_to_pred = ndimage.distance_transform_edt(~pred, sampling=spacing_zyx)
        d_gt_to_pred = dist_to_pred[gt_surf]
        d_pred_to_gt = dist_to_gt[pred_surf]
        return float(max(np.quantile(d_gt_to_pred, 0.95), np.quantile(d_pred_to_gt, 0.95)))

    @classmethod
    def centroid_distance_mm(cls, gt, pred, spacing_zyx):
        if gt.sum() == 0 or pred.sum() == 0:
            return math.nan
        c_gt = np.array(ndimage.center_of_mass(gt))
        c_pred = np.array(ndimage.center_of_mass(pred))
        return float(np.linalg.norm((c_gt - c_pred) * np.array(spacing_zyx)))

    @classmethod
    def region_stats(cls, gt_arr, pred_arr, labels, spacing_zyx, region):
        """The mother per-region row fields (region loop body, verbatim)."""
        gt_mask = np.isin(gt_arr, labels)
        pred_mask = np.isin(pred_arr, labels)
        vol_gt = float(gt_mask.sum()) * 0.001
        vol_pred = float(pred_mask.sum()) * 0.001
        intersection = int(np.logical_and(gt_mask, pred_mask).sum())
        det = bool(np.isin(pred_arr, (1, 2, 3)).sum() > 0)
        dice = 0.0 if pred_mask.sum() == 0 and gt_mask.sum() > 0 else cls.dice_of(gt_mask, pred_mask)
        sensitivity = 0.0 if pred_mask.sum() == 0 else (intersection / float(gt_mask.sum()) if gt_mask.sum() > 0 else math.nan)
        precision = 0.0 if pred_mask.sum() == 0 else (intersection / float(pred_mask.sum()) if intersection > 0 else 0.0)
        n_gt, n_pred, n_fp = cls.component_stats(gt_mask, pred_mask)
        # case-level counts for the ET/WT ratio (mother computes it from counts)
        et_gt = float(np.isin(gt_arr, (3,)).sum())
        wt_gt = float(np.isin(gt_arr, (1, 2, 3)).sum())
        et_pred = float(np.isin(pred_arr, (3,)).sum())
        wt_pred = float(np.isin(pred_arr, (1, 2, 3)).sum())
        return {
            "vol_gt_ml": vol_gt,
            "vol_pred_ml": vol_pred,
            "signed_bias_ml": vol_pred - vol_gt,
            "abs_err_ml": abs(vol_pred - vol_gt),
            "rel_vol_err": abs(vol_pred - vol_gt) / vol_gt if vol_gt > 0 else math.nan,
            "dice": dice,
            "sensitivity": sensitivity,
            "precision": precision,
            "hd95_mm": cls.hd95_mm(gt_mask, pred_mask, spacing_zyx),
            "centroid_distance_mm": cls.centroid_distance_mm(gt_mask, pred_mask, spacing_zyx),
            "n_components_gt": n_gt,
            "n_components_pred": n_pred,
            "n_false_positive_components": n_fp if region == "WT" else None,  # mother: WT-only column
            "detected": det,
            "et_gt_count": et_gt,
            "wt_gt_count": wt_gt,
            "et_pred_count": et_pred,
            "wt_pred_count": wt_pred,
        }

    @classmethod
    def component_stats(cls, gt_wt, pred_wt):
        gt_labels, n_gt = ndimage.label(gt_wt, structure=np.ones((3, 3, 3), dtype=np.uint8))
        pred_labels, n_pred = ndimage.label(pred_wt, structure=np.ones((3, 3, 3), dtype=np.uint8))
        n_fp = 0
        if n_gt > 0 and n_pred > 0:
            overlap = np.unique(pred_labels[gt_labels > 0])
            n_fp = n_pred - (len(overlap) - (1 if 0 in overlap else 0))
        return int(n_gt), int(n_pred), int(n_fp)


def _synthetic_case(seed=109, dtype=np.uint8, shape=(12, 14, 16)):
    """A well-formed nested case: WT⊃TC⊃ET, plus scattered oedema/background."""
    rng = np.random.default_rng(seed)
    pred = np.zeros(shape, dtype=dtype)
    pred[5:9, 5:10, 6:11] = 1  # core (WT + TC)
    pred[6:8, 6:9, 7:10] = 2  # oedema (WT only) -- inside core region but TC mask excludes 2
    pred[7:9, 7:9, 8:9] = 3  # ET
    pred[rng.random(shape) > 0.999] = 2  # tiny noise, still nested-safe
    return pred


def test_measure_generation_columns_always_produced():
    pred = _synthetic_case()
    measurement = InstrumentMeasurer().measure(pred)
    reference = FrozenTerminalReference.wide_quantities(pred)
    assert measurement.hierarchy_violation == reference["hier_viol"]
    assert measurement.pred_empty == reference["pred_empty"]
    assert measurement.et_wt == reference["et_wt"]
    assert measurement.volumes_ml == {
        "WT": reference["vol_wt_ml"],
        "TC": reference["vol_tc_ml"],
        "ET": reference["vol_et_ml"],
    }
    for region, centroid in measurement.centroids_mm.items():
        assert centroid == tuple(reference[f"c{axis}_{region.lower()}_mm"] for axis in "xyz")
    # closed families stay closed
    assert measurement.gt_metrics is None
    assert measurement.condition_dice is None
    assert measurement.brain_ml is None
    assert measurement.et_wt_ratio_gt is None


def test_measure_brain_family_matches_the_frozen_reference():
    pred = _synthetic_case()
    channels = [np.where(np.fromfunction(lambda z, y, x: (z + y + x) % 3, (12, 14, 16)) == 0, 1.0, 0.0).astype(np.float32) for _ in range(4)]
    channels[1][:4] = 1.0  # channel 2 adds brain voxels
    measurement = InstrumentMeasurer().measure(pred, brain=channels)
    reference = FrozenTerminalReference.wide_quantities(pred, brain_channels=channels)
    assert measurement.brain_ml == reference["brain_ml"]
    assert measurement.wt_brain == reference["wt_brain"]


def test_measure_condition_family_matches_the_frozen_reference():
    pred = _synthetic_case()
    condition = _synthetic_case(seed=211)
    measurement = InstrumentMeasurer().measure(pred, condition=condition)
    reference = FrozenTerminalReference.wide_quantities(pred, condition=condition)
    for region in ("WT", "TC", "ET"):
        assert measurement.condition_dice[region] == reference[f"cond_dice_{region.lower()}"]
    assert measurement.gt_metrics is None and measurement.brain_ml is None


def test_measure_empty_condition_denominator_is_none_per_region():
    pred = np.zeros((8, 8, 8), dtype=np.uint8)
    condition = np.zeros((8, 8, 8), dtype=np.uint8)
    measurement = InstrumentMeasurer().measure(pred, condition=condition)
    assert measurement.condition_dice == {"WT": None, "TC": None, "ET": None}


def test_measure_gt_family_matches_the_frozen_calibration_mother():
    pred = _synthetic_case()
    gt = _synthetic_case(seed=310)
    measurement = InstrumentMeasurer().measure(pred, gt=gt)
    for region in ("WT", "TC", "ET"):
        expected = FrozenCalibrationReference.region_stats(gt, pred, {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}[region], (1.0, 1.0, 1.0), region)
        actual = asdict(measurement.gt_metrics[region])
        for field, value in expected.items():
            if field in ("detected", "et_gt_count", "wt_gt_count", "et_pred_count", "wt_pred_count"):
                continue  # case-level fields, asserted separately
            if isinstance(value, float) and math.isnan(value):
                assert actual[field] is None, f"{region}.{field}: mother nan vs module {actual[field]}"
            else:
                assert actual[field] == value, f"{region}.{field}: {actual[field]} != {value}"
    # case-level fields
    assert measurement.pred_empty is False  # detected == not pred_empty
    assert measurement.calibration_usable is True
    et_gt, wt_gt = float(RegionMasks(gt).of("ET").sum()), float(RegionMasks(gt).of("WT").sum())
    assert measurement.et_wt_ratio_gt == et_gt / wt_gt


def test_measure_all_families_together_and_wide_row_equivalence():
    pred = _synthetic_case()
    gt = _synthetic_case(seed=521)
    condition = _synthetic_case(seed=997)
    channels = [np.ones((12, 14, 16), dtype=np.float32) * 0.5] * 4
    measurement = InstrumentMeasurer().measure(pred, gt=gt, condition=condition, brain=channels)
    row = measurement.to_wide_row(obs_id="O", challenge="GLI", case="C", side="gen", anchor="L")

    reference = FrozenTerminalReference.wide_quantities(pred, condition=condition, brain_channels=channels)
    for key, value in reference.items():
        assert row[key] == value

    rows = measurement.to_long_rows(challenge="GLI", case="C", source="dev", rep=1)
    for region_row in rows:
        expected = FrozenCalibrationReference.region_stats(
            gt, pred, {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}[region_row["region"]], (1.0, 1.0, 1.0), region_row["region"]
        )
        assert region_row["detected"] == (not measurement.pred_empty)
        assert region_row["case_usable"] == measurement.calibration_usable
        for field in (
            "vol_gt_ml",
            "vol_pred_ml",
            "signed_bias_ml",
            "abs_err_ml",
            "rel_vol_err",
            "dice",
            "sensitivity",
            "precision",
            "hd95_mm",
        ):
            value = expected[field]
            assert region_row[field] == (None if isinstance(value, float) and math.isnan(value) else value), f"{region_row['region']}.{field}"
        # the frozen long schema names components / false-positive columns
        # shorter than the value object's fields
        assert region_row["n_comp_gt"] == expected["n_components_gt"]
        assert region_row["n_comp_pred"] == expected["n_components_pred"]
        # the frozen long schema names the centroid distance column ``centroid_mm``
        centroid_value = expected["centroid_distance_mm"]
        assert region_row["centroid_mm"] == (None if isinstance(centroid_value, float) and math.isnan(centroid_value) else centroid_value)
        if region_row["region"] == "WT":
            assert region_row["n_fp_comp"] == expected["n_false_positive_components"]
        else:
            assert region_row["n_fp_comp"] is None
        # et_wt_ratio_gt: count-based in both (module computes it from counts).
        et_gt, wt_gt = float(RegionMasks(gt).of("ET").sum()), float(RegionMasks(gt).of("WT").sum())
        assert region_row["et_wt_ratio_gt"] == (et_gt / wt_gt if wt_gt > 0 else None)
        # et_wt_ratio_pred: module emits the frozen terminal et_wt (ml-ratio);
        # the mother's count-ratio differs by <= a few ulp -- registered in
        # measurement.CaseMeasurement docs (ADR-0010 consequences), not a
        # frozen aggregate input.
        assert region_row["et_wt_ratio_pred"] == pytest.approx(
            expected["et_pred_count"] / expected["wt_pred_count"] if expected["wt_pred_count"] > 0 else None, rel=1e-12
        )


def test_measure_pred_empty_and_gt_present_zero_like_mother():
    pred = np.zeros((9, 9, 9), dtype=np.uint8)
    gt = _synthetic_case(seed=7)[:9, :9, :9]
    measurement = InstrumentMeasurer().measure(pred, gt=gt)
    assert measurement.pred_empty is True
    for region in ("WT", "TC", "ET"):
        metrics = measurement.gt_metrics[region]
        gt_present = RegionMasks(gt).of(region).any()
        # mother: 0.0 when pred empty & GT non-empty; denom-0 (both empty) -> None
        assert metrics.dice == (0.0 if gt_present else None)
        assert metrics.sensitivity == 0.0  # pred empty -> 0.0
        assert metrics.precision == 0.0
        assert metrics.hd95_mm is None
        assert metrics.centroid_distance_mm is None
        assert metrics.n_components_pred == 0
