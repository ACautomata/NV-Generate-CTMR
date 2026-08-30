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

"""Canonical per-case instrument measurement and its two serializations (ADR-0010, issue #109).

``CaseMeasurement`` is the one value object behind the six drifted
implementations -- a frozen row of measured facts plus the two column-family
presentations:

- ``to_long_rows``: the calibration shape (1 row per region; the ADR-0002
  mother CSV schema, 23 columns).
- ``to_wide_row``: the terminal-acceptance shape (1 row per observation; the
  ADR-0004 judge ``MEASUREMENT_FIELDS`` schema, 27 columns).

Execution flags (``input_fail`` / ``run_fail``) and observation identity
(obs_id / challenge / case / side / anchor / source / rep) are caller
bookkeeping -- passed into the serializers, never part of the measurement.
The column families follow the gating of decision 2: generation columns and
``et_wt`` always present; calibration (``gt_metrics`` / ``calibration_usable``
/ ``et_wt_ratio_gt``) only when a GT reference was supplied; ``condition_dice``
only with a condition reference; ``brain_ml`` / ``wt_brain`` only with a brain
mask.

Sentinel policy (registered in ADR-0010 decision 4 / consequences): every
undefined *value-object field* is ``None`` (Dice empty-denominator sentinel
unified from ``nan``; the same ``None`` applies to hd95 / centroid distance /
ratio / sensitivity-undefined fields where the mother wrote ``nan``). The one
exception is ``WilsonUpper`` (metrics): its ``n == 0`` guard returns
``math.nan`` verbatim from the frozen calibration copies -- terminal
acceptance's call site guards itself and keeps producing ``None``. Per-case
CSV column renames (``hier_viol`` -> ``case_usable``) and sentinel differences
are registered and do not belong to the frozen ADR-0002 aggregates.
"""

from dataclasses import dataclass

from ctmr.domain.measurement.regions import REGION_NAMES

# The canonical wide-row schema: mirror of the frozen judge MEASUREMENT_FIELDS
# (ADR-0010 decision 5, #2). tests/measure pins this list equal to
# the distribution measurement_table MEASUREMENT_FIELDS (the shared-vocabulary
# host since ADR-0017/#229) -- drift is the 收编 #110 trigger, not a silent edit.
FINAL_ACCEPTANCE_FIELDS = [
    "obs_id",
    "challenge",
    "case",
    "side",
    "anchor",
    "input_fail",
    "run_fail",
    "hier_viol",
    "pred_empty",
    "vol_wt_ml",
    "vol_tc_ml",
    "vol_et_ml",
    "brain_ml",
    "wt_brain",
    "et_wt",
    "cx_wt_mm",
    "cy_wt_mm",
    "cz_wt_mm",
    "cx_tc_mm",
    "cy_tc_mm",
    "cz_tc_mm",
    "cx_et_mm",
    "cy_et_mm",
    "cz_et_mm",
    "cond_dice_wt",
    "cond_dice_tc",
    "cond_dice_et",
]

# The canonical long-row schema: the calibration mother CSV_FIELDS with one
# registered rename -- calibration case usability is NOT hierarchy_violation,
# so the column is ``case_usable`` (ADR-0010 decision 3; per-case CSV renames
# registered in decision 6).
CALIBRATION_FIELDS = [
    "challenge",
    "case",
    "source",
    "rep",
    "region",
    "input_fail",
    "run_fail",
    "case_usable",
    "detected",
    "dice",
    "sensitivity",
    "precision",
    "vol_gt_ml",
    "vol_pred_ml",
    "signed_bias_ml",
    "abs_err_ml",
    "rel_vol_err",
    "et_wt_ratio_gt",
    "et_wt_ratio_pred",
    "hd95_mm",
    "centroid_mm",
    "n_comp_gt",
    "n_comp_pred",
    "n_fp_comp",
]


@dataclass(frozen=True)
class GtRegionMetrics:
    """Per-region GT-comparison readout (the calibration column family of one region).

    ``n_fp_comp`` is defined for WT only (false-positive lesion components are
    a whole-tumour notion); ``dice`` is None when both masks are empty (the
    single Dice sentinel); undefined hd95 / centroid distance / sensitivity /
    rel_vol_err are None (the mother's ``nan`` -- registered unification).
    """

    vol_gt_ml: float
    vol_pred_ml: float
    signed_bias_ml: float
    abs_err_ml: float
    rel_vol_err: float | None
    dice: float | None
    sensitivity: float | None
    precision: float | None
    hd95_mm: float | None
    centroid_distance_mm: float | None
    n_components_gt: int
    n_components_pred: int
    n_false_positive_components: int | None


@dataclass(frozen=True)
class CaseMeasurement:
    """The canonical per-case measurement row (the six-site union of fields).

    ``gt_metrics`` / ``calibration_usable`` / ``et_wt_ratio_gt`` are None when
    no GT reference was given (calibration family closed); ``condition_dice``
    is None without a condition reference; ``brain_ml`` / ``wt_brain`` are None
    without a brain mask. ``et_wt`` follows the frozen terminal definition
    (ml-volume ratio), so the calibration serializer's ``et_wt_ratio_pred``
    cell may differ from the mother's count-based ratio within one ulp -- a
    registered, aggregate-neutral divergence (ADR-0010 consequences).
    """

    hierarchy_violation: bool
    pred_empty: bool
    volumes_ml: dict[str, float]
    centroids_mm: dict[str, tuple[float, float, float] | None]
    et_wt: float | None
    brain_ml: float | None
    wt_brain: float | None
    condition_dice: dict[str, float | None] | None
    gt_metrics: dict[str, GtRegionMetrics] | None
    calibration_usable: bool | None
    et_wt_ratio_gt: float | None

    def to_wide_row(
        self,
        *,
        obs_id: str,
        challenge: str,
        case: str,
        side: str,
        anchor: str | None = None,
        input_fail: bool | int = 0,
        run_fail: bool | int = 0,
    ) -> dict[str, object]:
        """One observation row in the frozen terminal-acceptance schema (27 columns)."""
        row: dict[str, object] = {
            "obs_id": obs_id,
            "challenge": challenge,
            "case": case,
            "side": side,
            "anchor": anchor or "",
            "input_fail": int(input_fail),
            "run_fail": int(run_fail),
            "hier_viol": int(self.hierarchy_violation),
            "pred_empty": int(self.pred_empty),
            "vol_wt_ml": self.volumes_ml["WT"],
            "vol_tc_ml": self.volumes_ml["TC"],
            "vol_et_ml": self.volumes_ml["ET"],
            "brain_ml": self.brain_ml,
            "wt_brain": self.wt_brain,
            "et_wt": self.et_wt,
        }
        for region in REGION_NAMES:
            centroid = self.centroids_mm[region]
            for axis, value in zip("xyz", centroid or (None, None, None)):
                row[f"c{axis}_{region.lower()}_mm"] = value
        for region in REGION_NAMES:
            row[f"cond_dice_{region.lower()}"] = None if self.condition_dice is None else self.condition_dice[region]
        return row

    def to_long_rows(
        self,
        *,
        challenge: str,
        case: str,
        source: str,
        rep: int,
        input_fail: bool = False,
        run_fail: bool = False,
    ) -> list[dict[str, object]]:
        """Calibration rows (1 per region, WT / TC / ET order) or a misuse error.

        The long schema is the GT-comparison readout: calling it without the
        calibration column family is a wiring bug, not a measurement.
        """
        if self.gt_metrics is None:
            raise ValueError("the long (calibration) schema needs the gt-gated column family")
        rows: list[dict[str, object]] = []
        for region in REGION_NAMES:
            metrics = self.gt_metrics[region]
            rows.append(
                {
                    "challenge": challenge,
                    "case": case,
                    "source": source,
                    "rep": rep,
                    "region": region,
                    "input_fail": input_fail,
                    "run_fail": run_fail,
                    "case_usable": self.calibration_usable,
                    "detected": not self.pred_empty,
                    "dice": metrics.dice,
                    "sensitivity": metrics.sensitivity,
                    "precision": metrics.precision,
                    "vol_gt_ml": metrics.vol_gt_ml,
                    "vol_pred_ml": metrics.vol_pred_ml,
                    "signed_bias_ml": metrics.signed_bias_ml,
                    "abs_err_ml": metrics.abs_err_ml,
                    "rel_vol_err": metrics.rel_vol_err,
                    "et_wt_ratio_gt": self.et_wt_ratio_gt,
                    "et_wt_ratio_pred": self.et_wt,
                    "hd95_mm": metrics.hd95_mm,
                    "centroid_mm": metrics.centroid_distance_mm,
                    "n_comp_gt": metrics.n_components_gt,
                    "n_comp_pred": metrics.n_components_pred,
                    "n_fp_comp": metrics.n_false_positive_components,
                }
            )
        return rows
