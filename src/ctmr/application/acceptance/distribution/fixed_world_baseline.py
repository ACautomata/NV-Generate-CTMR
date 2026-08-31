# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagnostic job T5 (issue #252, parent #247): the fixed-world baseline rerun.

Series-② write-path fix (#249) switched the generated NIfTI sidecar from the
retired unit 1 mm affine to the v1 DM's real sampling spacing (0.94, 0.94,
1.36) mm. On the holdout 530 artifacts that fix changes the DECLARATION, not
the voxel data: the artifacts now declare a z physical domain of
[0, GEN_RESAMPLED_Z) mm -- real tumour mass at z >= 128 mm (146 cases / ~431
ml, quantified by the geometry audit #217) is no longer outside the declared
domain -- and the instrument chain's z alignment world switches from the pad
reading (comp -13) back to the registered crop reading (comp +9). This job
re-expresses the frozen per-observation instrument readings of that SAME
artifact set in the post-fix world: all 14 registered L2 quantity families
are re-paired per case under the crop overlap window (job A's registered
geometry, now the world rather than one of two interpretations), zero
inference -- the segmentation face of the frozen instrument is not re-run
(the retrained T7 candidate will be the first genuinely new prediction set
inside this world; this baseline is what makes those readings comparable
against the recorded pad-world FAIL).

Two certificates bind the report to the recorded diagnostic history:

- Job A anchor (#206): the crop-window re-measurement IS job A's registered
  computation on the same masks with the same registered seed bit-streams, so
  the vol_wt_rel / centroid_wt_z compensated medians must reproduce the
  recorded values bit-for-bit (the fixed-world window arithmetic is certified
  by construction and by this check).
- Job B anchor (#207): the ET discrimination protocol re-runs on the same
  MEASUREMENT_FIELDS CSV through the recorded ``EtDiscrimination`` class and
  the same registered slot -- the readings are deterministic in the CSV, and
  the detection/pairing tallies must equal the recorded ones (the write-path
  fix does not move the ET readings; this baseline states that carry-over
  explicitly, exercising the P3 reuse hook).

This module is ``variant=diagnostic``: it never produces an acceptance
verdict, touches no frozen artifact write-path, and stays outside the
``ctmr accept`` verb surface. The sugon host recipe lives at
``deploy/jobs/run_fixed_world_baseline_t5.sh``; reports land in the sugon
artifact area (controlled storage), never in git.

Usage:
    python -m ctmr.application.acceptance.distribution.fixed_world_baseline \\
        --measurements <l2 run tree>/measurements.csv \\
        --pred-root <l2 run tree>/predictions \\
        --inputs-root <l2 run tree>/inputs \\
        --output-dir <artifact area>/fixed_world_baseline [--run-id <run>]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from ctmr.application.acceptance.distribution.challenge_registry import (
    BOOTSTRAP_B,
    CHALLENGES,
    DIAGNOSTIC_SEED_SLOTS,
    FROZEN_ENVELOPES,
)
from ctmr.application.acceptance.distribution.diagnostic_support import (
    DiagnosticError,
    DiagnosticReportWriter,
    DiagnosticSeedAllocator,
)
from ctmr.application.acceptance.distribution.et_discrimination import EtDiscrimination
from ctmr.application.acceptance.distribution.measurement_table import CHANNEL_SUFFIXES, MeasurementTable
from ctmr.application.acceptance.distribution.statistics import RelativeDifference
from ctmr.application.acceptance.distribution.zcrop_compensation import (
    GEN_RESAMPLED_Z,
    INSTRUMENT_Z,
    AttributionJudge,
    NiftiMaskRepository,
    OverlapWindow,
    PairedCompensation,
    ZCropCompensation,
)
from ctmr.application.acceptance.distribution.zcrop_geometry_audit import (
    GEN_WORKPIECE_Z,
    FieldOfViewAudit,
)
from ctmr.domain.vocabulary import REGIONS as REGION_LABELS

FIXED_WORLD_WINDOW = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
"""The post-fix overlap window: the declared (0.94, 0.94, 1.36) resample is real,
so the instrument grid covers generated physical [9, 164) mm and the shared
window with the real side is [9, 155) -- job A's registered crop window, now
the world, not one of two interpretations."""

COMPENSATION_MM = FIXED_WORLD_WINDOW.crop_start
"""The fixed-world coordinate convention: compensated gen-side z centroids sit at
grid index + 9 mm (physical), dual to the pad world's -13 mm."""

VOL_FIELD = {"WT": "vol_wt_ml", "TC": "vol_tc_ml", "ET": "vol_et_ml"}
"""The judge's per-region CSV volume fields (QuantityRegistry)."""


# ── quantity registry ───────────────────────────────────────────────────


ANCHOR_SEED_SLOTS = {
    "vol_wt_rel": ("zcrop_vol_uncomp", "zcrop_vol_comp"),
    "centroid_wt_z": ("zcrop_centroid_uncomp", "zcrop_centroid_comp"),
}
"""The two job A anchor quantities re-draw job A's registered bit-streams EXACTLY
(the WT vol and WT centroid-z blocks), so those CIs are bit-identical to the
recorded job A report."""


@dataclass(frozen=True)
class QuantitySpec:
    """One registered L2 quantity family's re-measurement spec (judge order).

    ``kind`` selects the reading face: ``vol`` (relative volume), ``cx``/``cy``/
    ``cz`` (signed mm centroid axes), ``wt_brain`` / ``et_wt`` (the ratio
    families). ``region`` names the mask projection for the mask-derived kinds.
    """

    name: str
    kind: str
    region: str | None

    def margin(self, challenge: str) -> float:
        """The frozen ADR-0002 margin the judge bounds this quantity with
        (QuantityRegistry's margin builders over the shared envelope literals)."""
        if self.kind == "vol":
            return FROZEN_ENVELOPES[challenge][self.region][1]
        if self.kind == "wt_brain":
            return FROZEN_ENVELOPES[challenge]["WT"][1]
        if self.kind == "et_wt":
            return FROZEN_ENVELOPES[challenge]["ET"][1] + FROZEN_ENVELOPES[challenge]["WT"][1]
        return FROZEN_ENVELOPES[challenge][self.region][2]  # centroid axes: E_r,centroid

    def seed(self, challenge: str, arm: str) -> int:
        """The diagnostic bootstrap seed of one (challenge, arm) block of this
        quantity, drawn through the allocator -- registered slots only, never
        re-spelled. Arms: ``uncomp`` draws this quantity's uncomp slot, ``comp``
        its comp slot."""
        uncomp_name, comp_name = SEED_SLOTS[self.name]
        return DiagnosticSeedAllocator.seed(challenge, DIAGNOSTIC_SEED_SLOTS[comp_name if arm == "comp" else uncomp_name])


QUANTITIES = (
    QuantitySpec("vol_wt_rel", "vol", "WT"),
    QuantitySpec("centroid_wt_x", "cx", "WT"),
    QuantitySpec("centroid_wt_y", "cy", "WT"),
    QuantitySpec("centroid_wt_z", "cz", "WT"),
    QuantitySpec("vol_tc_rel", "vol", "TC"),
    QuantitySpec("centroid_tc_x", "cx", "TC"),
    QuantitySpec("centroid_tc_y", "cy", "TC"),
    QuantitySpec("centroid_tc_z", "cz", "TC"),
    QuantitySpec("vol_et_rel", "vol", "ET"),
    QuantitySpec("centroid_et_x", "cx", "ET"),
    QuantitySpec("centroid_et_y", "cy", "ET"),
    QuantitySpec("centroid_et_z", "cz", "ET"),
    QuantitySpec("wt_brain_rel", "wt_brain", None),
    QuantitySpec("et_wt_rel", "et_wt", None),
)
"""The 14 registered TOST quantity families in the judge's order (QuantityRegistry)."""

SEED_SLOTS = {spec.name: ANCHOR_SEED_SLOTS.get(spec.name, (f"t5_uncomp_{spec.name}", f"t5_comp_{spec.name}")) for spec in QUANTITIES}
"""Per-quantity (uncomp_slot, comp_slot) registered names: the anchor quantities
draw job A's slots, every other quantity the T5 blocks (challenge_registry:
comp 400+index, uncomp 500+index, anchor indices 0/3 exempt as gaps) -- the
next free banded slots after A (0/1/100/101), B (200) and the geometry audit (300)."""


# ── window re-measurement ───────────────────────────────────────────────


class InstrumentInputsRepository:
    """The instrument input tree's brain unions (protocol §3), the measure
    step's caller-owned channel IO mirrored read-only: ``<inputs_root>/<CH>/
    <obs_id>_<suffix>.nii.gz``, four channels per observation, the brain mask
    the union of their non-zero voxels. A missing or unreadable tree reads as
    ``None`` -- the ratio families drop the case, tallied, never imputed."""

    def __init__(self, inputs_root):
        self._inputs_root = Path(inputs_root)

    def brain_mask(self, challenge: str, obs_id: str) -> np.ndarray | None:
        try:
            union = None
            for suffix in sorted(CHANNEL_SUFFIXES.values()):
                array = sitk.GetArrayFromImage(sitk.ReadImage(str(self._inputs_root / challenge / f"{obs_id}_{suffix}.nii.gz")))
                nonzero = array > 0
                union = nonzero if union is None else (union | nonzero)
        except (RuntimeError, OSError):
            return None
        return union.astype(np.uint8, copy=False)


class MaskRepository(Protocol):
    """Per-observation prediction-mask source (zcrop_compensation's protocol)."""

    def wt_mask(self, challenge: str, obs_id: str) -> np.ndarray | None: ...


class BrainRepository(Protocol):
    """Per-observation brain-union source, injected beside the mask repository."""

    def brain_mask(self, challenge: str, obs_id: str) -> np.ndarray | None: ...


# ── per-case pairing ────────────────────────────────────────────────────


class FixedWorldPairedReadings:
    """Per-case uncompensated (recorded world) vs fixed-world re-paired values
    for all 14 L2 quantity families.

    The uncompensated arm reads the recorded CSV verbatim (the judge's pairing
    rules: relative diffs against the real denominator, a generated-side empty
    prediction kept at rel diff -1.0, centroid axes requiring a non-empty mask
    on both sides); the fixed-world arm re-measures both sides inside the crop
    window and applies the same rules to the window readings. The two arms'
    availability is independent -- a case can carry a recorded relative volume
    and no fixed-world ratio (missing inputs), tallied per case.
    """

    def __init__(self, window: OverlapWindow = FIXED_WORLD_WINDOW):
        self._window = window

    @staticmethod
    def measure_region(mask: np.ndarray, window: OverlapWindow, side: str, region: str) -> dict:
        """Window-restricted volume (ml) and physical-mm centroid (x, y, z) of one
        region and side. Job A's ``restrict_and_measure`` generalized over regions
        and axes -- the WT volume and centroid z arithmetic is verbatim its (the
        job A anchor certifies the bit-for-bit equivalence on the real artifacts):
        volumes 0.001 ml/voxel on the 1 mm grid, gen-side z shifted by the window's
        crop offset into physical mm, x/y plain grid indices (the crop window adds
        no x/y offset), an empty region a zero volume with no centroid.
        """
        z_slice, crop_offset = {"gen": (window.gen_slice, window.crop_start), "real": (window.real_slice, 0)}[side]
        region_mask = np.isin(mask[z_slice], REGION_LABELS[region])
        volume_ml = float(region_mask.sum()) * 0.001
        if not region_mask.any():
            return {"vol_ml": 0.0, "cx_mm": None, "cy_mm": None, "cz_mm": None}
        centroid_z, centroid_y, centroid_x = ndimage.center_of_mass(region_mask)
        return {
            "vol_ml": volume_ml,
            "cx_mm": float(centroid_x),
            "cy_mm": float(centroid_y),
            "cz_mm": float(centroid_z + z_slice.start + crop_offset),
        }

    def read_cases(self, rows, mask_repository: MaskRepository, brain_repository: BrainRepository) -> list[dict]:
        pairs: dict[tuple[str, str], dict[str, dict]] = {}
        for row in rows:
            pairs.setdefault((row["challenge"], row["case"]), {})[row["side"]] = row
        return [self._read_case(challenge, case, sides, mask_repository, brain_repository) for (challenge, case), sides in sorted(pairs.items())]

    def _read_case(self, challenge: str, case: str, sides: dict, masks, inputs) -> dict:
        real_row, gen_row = sides.get("real"), sides.get("gen")
        reading = {
            "challenge": challenge,
            "case": case,
            "excluded": None if real_row is not None and gen_row is not None else "missing_side_row",
            "brain_missing": False,
        }
        real_mask = masks.wt_mask(challenge, PairedCompensation.obs_id(case, "real"))
        gen_mask = masks.wt_mask(challenge, PairedCompensation.obs_id(case, "gen"))
        if real_mask is None or gen_mask is None:
            reading["excluded"] = "missing_prediction"
        window_data = None
        if reading["excluded"] != "missing_prediction":
            real_brain = inputs.brain_mask(challenge, PairedCompensation.obs_id(case, "real"))
            gen_brain = inputs.brain_mask(challenge, PairedCompensation.obs_id(case, "gen"))
            reading["brain_missing"] = real_brain is None or gen_brain is None
            window_data = self._window_data(real_mask, gen_mask, real_brain, gen_brain)
            for side, row in (("real", real_row), ("gen", gen_row)):
                reading[f"win_brain_{side}_ml"] = window_data["brain_ml"][side]
                self._cross_check_brain(challenge, case, side, row, window_data["brain_full_ml"][side])
                for region in REGION_LABELS:
                    reading[f"win_vol_{region.lower()}_{side}_ml"] = window_data["region"][(side, region)]["vol_ml"]
        for spec in QUANTITIES:
            reading[f"{spec.name}_uncomp"] = self._uncomp_value(spec, real_row, gen_row)
            reading[f"{spec.name}_comp"] = None
            if window_data is not None:
                reading[f"{spec.name}_comp"] = self._fixed_world_value(spec, window_data)
        return reading

    def _window_data(self, real_mask, gen_mask, real_brain, gen_brain) -> dict:
        """One case's window readings: per-side region volumes + centroids and
        brain volumes (the brain union restricted to the same window slices --
        equal window depth on both sides, job A's no-single-side-bias rule).
        The full-grid unions ride along for the CSV cross-check."""
        data: dict = {"region": {}, "brain_ml": {}, "brain_full_ml": {}}
        for side, mask, brain in (("real", real_mask, real_brain), ("gen", gen_mask, gen_brain)):
            for region in REGION_LABELS:
                data["region"][(side, region)] = self.measure_region(mask, self._window, side, region)
            z_slice = self._window.real_slice if side == "real" else self._window.gen_slice
            data["brain_ml"][side] = None if brain is None else float(brain[z_slice].sum()) * 0.001
            data["brain_full_ml"][side] = None if brain is None else float(brain.sum()) * 0.001
        return data

    @staticmethod
    def _cross_check_brain(challenge: str, case: str, side: str, row, full_brain_ml):
        """The inputs tree must be the one the recorded CSV was measured on: the
        full-grid brain union recomputed here must equal the CSV's ``brain_ml``
        bit-for-bit (same arrays, same rule -- a mismatch means the tree moved
        and every ratio reading would be silently wrong)."""
        recorded = MeasurementTable.number(row, "brain_ml") if row else None
        if recorded is None or full_brain_ml is None:
            return
        if full_brain_ml != recorded:
            raise DiagnosticError(
                f"brain union mismatch for {challenge}/{case}__{side}: inputs tree gives {full_brain_ml!r} ml, "
                f"the recorded CSV says {recorded!r} ml -- the inputs tree does not back this measurement CSV"
            )

    def _uncomp_value(self, spec: QuantitySpec, real_row, gen_row):
        """The recorded-world value, the judge's own pairing rules on the CSV."""
        if spec.kind == "vol":
            return RelativeDifference.of(self._field(gen_row, VOL_FIELD[spec.region]), self._field(real_row, VOL_FIELD[spec.region]))
        if spec.kind in ("cx", "cy", "cz"):
            vol_field = VOL_FIELD[spec.region]
            if not (self._positive(gen_row, vol_field) and self._positive(real_row, vol_field)):
                return None
            return self._signed_diff(
                self._field(gen_row, f"{spec.kind}_{spec.region.lower()}_mm"), self._field(real_row, f"{spec.kind}_{spec.region.lower()}_mm")
            )
        if spec.kind == "wt_brain":
            return RelativeDifference.of(self._field(gen_row, "wt_brain"), self._field(real_row, "wt_brain"))
        return RelativeDifference.of(self._field(gen_row, "et_wt"), self._field(real_row, "et_wt"))  # et_wt

    def _fixed_world_value(self, spec: QuantitySpec, data: dict):
        """The fixed-world value: the judge's pairing rules on the window
        readings. The gen-side centroid z is already physical (crop offset
        inside ``measure_region``); the ratio families mirror the
        measurer's own conventions (``wt_brain`` defined whenever the brain
        exists, ``et_wt`` requiring a non-empty WT)."""
        gen_region = data["region"][("gen", spec.region)] if spec.region else None
        real_region = data["region"][("real", spec.region)] if spec.region else None
        if spec.kind == "vol":
            return RelativeDifference.of(gen_region["vol_ml"], real_region["vol_ml"])
        if spec.kind in ("cx", "cy", "cz"):
            if gen_region["vol_ml"] <= 0 or real_region["vol_ml"] <= 0:
                return None
            return gen_region[f"{spec.kind}_mm"] - real_region[f"{spec.kind}_mm"]
        gen_brain, real_brain = data["brain_ml"]["gen"], data["brain_ml"]["real"]
        gen_wt = data["region"][("gen", "WT")]["vol_ml"]
        real_wt = data["region"][("real", "WT")]["vol_ml"]
        if spec.kind == "wt_brain":
            gen_ratio = gen_wt / gen_brain if gen_brain else None
            real_ratio = real_wt / real_brain if real_brain else None
            return RelativeDifference.of(gen_ratio, real_ratio)
        # et_wt: the measurer defines the ratio only over a non-empty WT
        gen_et = data["region"][("gen", "ET")]["vol_ml"]
        real_et = data["region"][("real", "ET")]["vol_ml"]
        gen_ratio = gen_et / gen_wt if gen_wt > 0 else None
        real_ratio = real_et / real_wt if real_wt > 0 else None
        return RelativeDifference.of(gen_ratio, real_ratio)

    @staticmethod
    def _field(row, field):
        return MeasurementTable.number(row, field) if row else None

    @classmethod
    def _positive(cls, row, field) -> bool:
        value = cls._field(row, field)
        return value is not None and value > 0

    @staticmethod
    def _signed_diff(gen_value, real_value):
        if gen_value is None or real_value is None:
            return None
        return gen_value - real_value

    @staticmethod
    def within_margin(stats: dict, margin: float) -> bool | None:
        """CI90-subset-of-envelope flag, job A's report rule (its module stays
        untouched per the recorded-jobs convention; the 3-line rule mirrors)."""
        if stats["ci90_low"] is None:
            return None
        return stats["ci90_low"] >= -margin - 1e-12 and stats["ci90_high"] <= margin + 1e-12


# ── field-of-view scan ──────────────────────────────────────────────────


class FixedWorldFieldScan:
    """Real WT mass against the generated domains, both declarations.

    The post-fix declaration extends the generated z domain to
    [0, GEN_RESAMPLED_Z): real mass above GEN_RESAMPLED_Z is the out-of-domain
    readout the write-path fix exists to zero (146 cases / ~431 ml at the
    retired 1 mm edge z >= 128, the legacy tally kept for the head-to-head).
    The window floor (z < 9) is the crop world's own lower edge; generated
    mass below it would be measurement-loss in the fixed world (hygiene
    tally, expected zero on the pad-world masks whose content starts at 13).
    """

    def __init__(self, window: OverlapWindow = FIXED_WORLD_WINDOW, legacy_declared_hi: int = GEN_WORKPIECE_Z):
        self._window = window
        self._legacy_declared_hi = legacy_declared_hi

    def scan(self, rows, repository: MaskRepository) -> dict:
        cases = {}
        for row in rows:
            cases.setdefault((row["challenge"], row["case"]), {})[row["side"]] = row
        per_challenge: dict[str, dict] = {}
        for (challenge, case), sides in sorted(cases.items()):
            tally = per_challenge.setdefault(
                challenge,
                {
                    "real_below_window_ml": 0.0,
                    "real_above_declared_fixed_ml": 0.0,
                    "real_over_declared_fixed_cases": 0,
                    "real_above_declared_legacy_ml": 0.0,
                    "real_over_declared_legacy_cases": 0,
                    "worst_over_declared_legacy": None,
                    "gen_below_window_ml": 0.0,
                    "gen_below_window_cases": 0,
                },
            )
            real_mask = repository.wt_mask(challenge, PairedCompensation.obs_id(case, "real"))
            gen_mask = repository.wt_mask(challenge, PairedCompensation.obs_id(case, "gen"))
            if real_mask is not None:
                outside = FieldOfViewAudit.real_wt_outside(real_mask, window=self._window, declared_hi=GEN_RESAMPLED_Z)
                tally["real_below_window_ml"] += outside["below_content_ml"]
                tally["real_above_declared_fixed_ml"] += outside["above_declared_ml"]
                if outside["above_declared_ml"] > 0:
                    tally["real_over_declared_fixed_cases"] += 1
                legacy = self._mass_above(real_mask, self._legacy_declared_hi)
                tally["real_above_declared_legacy_ml"] += legacy
                if legacy > 0:
                    tally["real_over_declared_legacy_cases"] += 1
                    if tally["worst_over_declared_legacy"] is None or legacy > tally["worst_over_declared_legacy"]["ml"]:
                        tally["worst_over_declared_legacy"] = {"case": case, "ml": legacy}
            if gen_mask is not None:
                below = FieldOfViewAudit.gen_mass_in_padding(gen_mask, window=self._window)
                tally["gen_below_window_ml"] += below
                if below > 0:
                    tally["gen_below_window_cases"] += 1
        return per_challenge

    @staticmethod
    def _mass_above(mask, z_floor: int) -> float:
        """Real WT mass (ml) at z >= z_floor, the audit's own region rule."""
        return float(FieldOfViewAudit.wt_region(mask)[z_floor:].sum()) * 0.001


# ── recorded-history anchors ────────────────────────────────────────────


class JobAAnchor:
    """The bit-reproduction certificate against the recorded job A report.

    The fixed-world window IS job A's registered geometry on the same masks
    with the same registered seed bit-streams, so the compensated medians of
    the two anchor quantities must equal the recorded 6-dp literals (the
    recorded table of `deploy/experiments/20260829-P1根因甄别-作业A-zcrop补偿归因.md`,
    cross-checked bit-for-bit by the geometry audit #217). Drift beyond the
    recorded grid's half-ulp means the window arithmetic, masks or seeds moved
    -- the baseline would not be the recorded history's dual.
    """

    TOLERANCE = 5e-7  # half-ulp of the recorded 6-dp grid

    RECORDED_COMP_MEDIANS = {
        "vol_wt_rel": {"GLI": 5.207930, "MEN": 0.513802, "METS": -0.992933, "PED": 16.009951, "SSA": 5.907041},
        "centroid_wt_z": {"GLI": 2.927747, "MEN": -9.854146, "METS": 7.199478, "PED": 22.425059, "SSA": -2.480662},
    }

    ANCHOR_QUANTITIES = ("vol_wt_rel", "centroid_wt_z")

    @classmethod
    def verify(cls, per_challenge: dict) -> dict:
        """Recorded vs reproduced medians per challenge; ``matched`` False on any
        drift or on a challenge the run did not cover (the caller owns the loud
        exit -- the report is written either way)."""
        block = {}
        for name in cls.ANCHOR_QUANTITIES:
            block[name] = {}
            for challenge, recorded in cls.RECORDED_COMP_MEDIANS[name].items():
                challenge_block = per_challenge.get(challenge, {})
                reproduced = challenge_block.get(name, {}).get("comp", {}).get("median")
                block[name][challenge] = {
                    "recorded": recorded,
                    "reproduced": reproduced,
                    "matched": reproduced is not None and abs(reproduced - recorded) <= cls.TOLERANCE,
                }
        block["matched"] = all(entry["matched"] for quantity in cls.ANCHOR_QUANTITIES for entry in block[quantity].values())
        return block


class JobBAnchor:
    """The protocol-reuse certificate against the recorded job B table.

    The ET discrimination re-run reads the same CSV through the recorded
    class and slot; its detection and pairing tallies are deterministic in
    the CSV, so they must equal the recorded ones (recorded:
    `deploy/experiments/20260829-诊断作业B-ET甄别.md` §3) -- certifying both the
    reuse hook and that the write-path fix leaves the ET readings untouched.
    """

    RECORDED = {
        "GLI": {"gen_k": 250, "gen_n": 250, "real_only": 0, "gen_empty_pred": 0},
        "MEN": {"gen_k": 200, "gen_n": 200, "real_only": 0, "gen_empty_pred": 0},
        "METS": {"gen_k": 38, "gen_n": 48, "real_only": 10, "gen_empty_pred": 5},
        "PED": {"gen_k": 20, "gen_n": 20, "real_only": 0, "gen_empty_pred": 0},
        "SSA": {"gen_k": 12, "gen_n": 12, "real_only": 0, "gen_empty_pred": 0},
    }

    @classmethod
    def verify(cls, et_readings: list[dict]) -> dict:
        """Recorded vs reproduced ET tallies per challenge, ``matched`` False on
        drift or on a challenge the run did not cover."""
        by_challenge = {reading["challenge"]: reading for reading in et_readings}
        block = {}
        for challenge, recorded in cls.RECORDED.items():
            reading = by_challenge.get(challenge)
            if reading is None:
                block[challenge] = {"recorded": recorded, "reproduced": None, "matched": False}
                continue
            reproduced = {
                "gen_k": reading["gen"]["k_detected"],
                "gen_n": reading["gen"]["n"],
                "real_only": reading["pairing"]["real_only"],
                "gen_empty_pred": reading["empty_pred"]["gen"]["k"],
            }
            block[challenge] = {
                "recorded": recorded,
                "reproduced": reproduced,
                "matched": recorded == reproduced,
            }
        block["matched"] = all(entry["matched"] for entry in block.values())
        return block


# ── report ──────────────────────────────────────────────────────────────


class FixedWorldBaselineReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git).

    Margins quote the frozen ADR-0002 literals for orientation only -- this
    run registers no verdict. Seeds draw the registered job A anchor slots and
    the T5 blocks through the diagnostic allocator (challenge_registry).
    """

    SCHEMA = "fixed-world-baseline-diagnostic/1"
    TITLE = "序列②T5:修复后世界基线重跑(L2 诊断口径重测 + 作业 B 协议复用)"

    def __init__(self, measurements_path, pred_root, inputs_root, bootstrap_b: int = BOOTSTRAP_B, run_id: str | None = None):
        self._measurements_path = Path(measurements_path)
        self._pred_root = Path(pred_root)
        self._inputs_root = Path(inputs_root)
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id
        self._writer = DiagnosticReportWriter(
            schema=self.SCHEMA,
            title=self.TITLE,
            issue=252,
            job_label="序列② T5",
            stem="fixed_world_baseline_diagnostic",
            inputs={"measurements": str(self._measurements_path), "pred_root": str(self._pred_root), "inputs_root": str(self._inputs_root)},
            run_id=self._run_id,
            parent_issue=247,
        )

    @staticmethod
    def quantity_block(readings, challenge: str, spec: QuantitySpec, bootstrap_b: int = BOOTSTRAP_B) -> dict:
        """One (challenge, quantity) read-out block: both arms' distribution stats
        under their registered seeds, the within-margin flags and the attribution
        three-way (the recorded job A block shape, margins quote the frozen
        literals for orientation only)."""
        margin = spec.margin(challenge)
        cases = [reading for reading in readings if reading["challenge"] == challenge]
        blocks = {}
        for arm in ("uncomp", "comp"):
            values = [case[f"{spec.name}_{arm}"] for case in cases if case[f"{spec.name}_{arm}"] is not None]
            blocks[arm] = PairedCompensation.summary_stats(values, bootstrap_b, spec.seed(challenge, arm))
        return {
            "margin": margin,
            "uncomp": blocks["uncomp"],
            "comp": blocks["comp"],
            "uncomp_ci_within_margin": FixedWorldPairedReadings.within_margin(blocks["uncomp"], margin),
            "comp_ci_within_margin": FixedWorldPairedReadings.within_margin(blocks["comp"], margin),
            "attribution": AttributionJudge().classify(blocks["uncomp"]["median"], blocks["comp"]["median"]),
        }

    def write(self, readings, per_challenge: dict, field_of_view: dict, et_readings: list[dict], job_a_anchor: dict, job_b_anchor: dict, output_dir):
        """Writes the json + markdown pair from the caller-computed quantity
        blocks (the bootstrap is the run's expensive face -- computed once)."""
        payload = self._writer.payload(
            {
                "worlds": self._world_blocks(),
                "quantity_order": [spec.name for spec in QUANTITIES],
                "per_case": readings,
                "per_challenge": per_challenge,
                "attribution_overall": self._attribution_overall(per_challenge),
                "job_a_anchor": job_a_anchor,
                "job_b_anchor": job_b_anchor,
                "field_of_view": field_of_view,
                "et_discrimination": {
                    reading["challenge"]: {key: value for key, value in reading.items() if key != "per_case_rows"} for reading in et_readings
                },
            }
        )
        return self._writer.write(payload, self._markdown(payload), output_dir)

    @staticmethod
    def _world_blocks() -> dict:
        return {
            "fixed_world": {
                "declaration": "#249 写出协议修复:sidecar affine 盖生成器真实采样 spacing (0.94, 0.94, 1.36) mm",
                "declared_domain_mm": [0, GEN_RESAMPLED_Z],
                "instrument_window_mm": [FIXED_WORLD_WINDOW.phys_lo, FIXED_WORLD_WINDOW.phys_hi],
                "compensation_mm": COMPENSATION_MM,
                "identity": "comp = uncomp + 9(质心轴世界由 pad(comp -13)切回 crop(comp +9);不跨窗 case 解析恒等)",
            },
            "legacy_world": {
                "declaration": "单位 1 mm affine(sidecar 写出约定 np.diag([1,1,1]),#249 之前的工件现状)",
                "declared_domain_mm": [0, GEN_WORKPIECE_Z],
                "content_domain_mm": [13, 141],
                "compensation_mm": -13.0,
                "identity": "comp_pad = uncomp - 13(复核作业 A #217 的工件几何读数)",
            },
            "note": (
                "同一批 holdout 530 生成工件与同一批冻结仪器逐观测读数,#249 只改声明不改体素:"
                "本作业零推理,不重跑冻结仪器的分割面(T7 重训候选才是修复后世界第一批真预测),"
                "把 14 个注册量族按修复后窗口逐 case 重配对;仪器未补偿读数(量即窗内栅格读数)与"
                "comp_pad(现状世界)读数共享于已落盘记录,本报告的 uncomp 臂即记录 CSV 的判官配对规则读数。"
            ),
        }

    @staticmethod
    def _attribution_overall(per_challenge: dict) -> dict:
        """Cross-challenge tally of the per-quantity attribution classes."""
        overall = {}
        for spec in QUANTITIES:
            counts: dict[str, int] = {}
            for challenge in sorted(per_challenge):
                classification = per_challenge[challenge][spec.name]["attribution"]["classification"]
                counts[classification] = counts.get(classification, 0) + 1
            overall[spec.name] = {"counts": counts, "majority": max(counts, key=counts.get) if counts else None}
        return overall

    @staticmethod
    def _fmt(value) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def _markdown(self, payload: dict) -> str:
        worlds = payload["worlds"]
        lines = self._writer.markdown_preamble(payload) + [
            "## 世界口径",
            "",
            f"- 修复后世界:{worlds['fixed_world']['declaration']};声明域 {worlds['fixed_world']['declared_domain_mm']} mm;"
            f"仪器重叠窗 {worlds['fixed_world']['instrument_window_mm']} mm;{worlds['fixed_world']['identity']}",
            f"- 现状世界(对照):{worlds['legacy_world']['declaration']};声明域 {worlds['legacy_world']['declared_domain_mm']} mm;"
            f"{worlds['legacy_world']['identity']}",
            f"- {worlds['note']}",
            f"- 输入:measurements `{payload['inputs']['measurements']}`;predictions `{payload['inputs']['pred_root']}`;"
            f"inputs `{payload['inputs']['inputs_root']}`",
            "",
            "## 锚点对账(逐位证书)",
            "",
            "### 作业 A(#206):comp median 复现",
            "",
            "| 量 | 挑战 | 记录 median | 本作业复现 | 一致 |",
            "|---|---|---:|---:|---|",
        ]
        for name in JobAAnchor.ANCHOR_QUANTITIES:
            for challenge, entry in payload["job_a_anchor"][name].items():
                lines.append(
                    f"| {name} | {challenge} | {self._fmt(entry['recorded'])} | {self._fmt(entry['reproduced'])} "
                    f"| {'是' if entry['matched'] else '**否**'} |"
                )
        lines += [
            "",
            f"作业 A 锚点整体:**{'逐位复现' if payload['job_a_anchor']['matched'] else '**漂移**'}**;"
            "种子 = 作业 A 注册槽位(zcrop_vol_comp=100 / zcrop_centroid_comp=101,逐臂逐槽复用)。",
            "",
            "### 作业 B(#207):协议复用读数对账",
            "",
            "| 挑战 | 记录 k/n | 复现 k/n | 记录 real_only | 复现 real_only | 记录空 pred | 复现空 pred | 一致 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for challenge, entry in payload["job_b_anchor"].items():
            if not isinstance(entry, dict) or "matched" not in entry:
                continue
            recorded, reproduced = entry["recorded"], entry["reproduced"]
            if reproduced is None:  # a challenge the run did not cover
                lines.append(
                    f"| {challenge} | {recorded['gen_k']}/{recorded['gen_n']} | 缺 | {recorded['real_only']} | 缺 | {recorded['gen_empty_pred']} | 缺 | **否** |"
                )
                continue
            lines.append(
                f"| {challenge} | {recorded['gen_k']}/{recorded['gen_n']} | {reproduced['gen_k']}/{reproduced['gen_n']} "
                f"| {recorded['real_only']} | {reproduced['real_only']} | {recorded['gen_empty_pred']} | {reproduced['gen_empty_pred']} "
                f"| {'是' if entry['matched'] else '**否**'} |"
            )
        lines += [
            "",
            f"作业 B 锚点整体:**{'一致' if payload['job_b_anchor']['matched'] else '**漂移**'}**;"
            "同一 CSV、同一注册槽位(et_rel_diff=200),写出修复不动 ET 读数(P3 复用钩子按 MEASUREMENT_FIELDS 契约生效)。",
            "",
            "## 逐量读数(14 量族 × 5 挑战)",
            "",
        ]
        for spec in QUANTITIES:
            lines += [
                f"### {spec.name}(margin 见行内;种子槽 {SEED_SLOTS[spec.name][0]}/{SEED_SLOTS[spec.name][1]})",
                "",
                "| 挑战 | 记录世界 uncomp median (CI90) | 修复后世界 comp median (CI90) | margin | uncomp CI ⊆ 包络 | comp CI ⊆ 包络 | 测量轴占比 | 归因 |",
                "|---|---|---|---:|---|---|---:|---|",
            ]
            for challenge in sorted(payload["per_challenge"]):
                block = payload["per_challenge"][challenge][spec.name]
                attribution = block["attribution"]
                lines.append(
                    f"| {challenge} "
                    f"| {self._fmt(block['uncomp']['median'])} ({self._fmt(block['uncomp']['ci90_low'])}, {self._fmt(block['uncomp']['ci90_high'])}) "
                    f"| {self._fmt(block['comp']['median'])} ({self._fmt(block['comp']['ci90_low'])}, {self._fmt(block['comp']['ci90_high'])}) "
                    f"| ±{self._fmt(block['margin'])} "
                    f"| {self._fmt_flag(block['uncomp_ci_within_margin'])} | {self._fmt_flag(block['comp_ci_within_margin'])} "
                    f"| {self._fmt(attribution['measurement_fraction'])} | {attribution['classification']} |"
                )
            lines.append("")
        lines += ["## 跨挑战归因汇总", ""]
        for spec in QUANTITIES:
            tally = payload["attribution_overall"][spec.name]
            counts = "、".join(f"{name} ×{count}" for name, count in sorted(tally["counts"].items())) or "无可用挑战"
            lines.append(f"- {spec.name}:多数归因 **{tally['majority']}**({counts})")
        lines += [
            "",
            "## 视场缺口(声明域外 real WT 质量)",
            "",
            "| 挑战 | z<9 ml(修复后窗下缘) | z≥174 ml(修复后声明域外,**预期归零**) | 声明域外例数(修复后) "
            "| z≥128 ml(旧 1 mm 声明域外,复核 A 口径) | 旧口径例数 | 最重旧口径 case | gen z<9 ml |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
        for challenge, tally in payload["field_of_view"].items():
            worst = tally["worst_over_declared_legacy"]
            lines.append(
                f"| {challenge} | {self._fmt(tally['real_below_window_ml'])} | {self._fmt(tally['real_above_declared_fixed_ml'])} "
                f"| {tally['real_over_declared_fixed_cases']} | {self._fmt(tally['real_above_declared_legacy_ml'])} "
                f"| {tally['real_over_declared_legacy_cases']} "
                f"| {worst['case'] if worst else '无'} ({self._fmt(worst['ml']) if worst else 'n/a'}) "
                f"| {self._fmt(tally['gen_below_window_ml'])} |"
            )
        lines += [
            "",
            "## ET 甄别(作业 B 协议复用读数)",
            "",
            "| 挑战 | gen n | 检出率 | Wilson 95% 上界 | 空 pred | real n | real 检出 | real_only | rel diff median (CI90) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for challenge, reading in payload["et_discrimination"].items():
            gen, real = reading["gen"], reading["real"]
            rel = reading["rel_diff"]
            lines.append(
                f"| {challenge} | {gen['n']} | {self._fmt(gen['rate'])} | {self._fmt(gen['wilson_95_upper'])} "
                f"| {reading['empty_pred']['gen']['k']}/{gen['n']} | {real['n']} | {real['k_detected']}/{real['n']} "
                f"| {reading['pairing']['real_only']} "
                f"| {self._fmt(rel['median'])} ({self._fmt(rel['ci90_low'])}, {self._fmt(rel['ci90_high'])}) |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _fmt_flag(value: bool | None) -> str:
        return "n/a" if value is None else ("是" if value else "否")


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", required=True, help="the L2 run tree's per-observation measurement CSV (controlled storage)")
    parser.add_argument("--pred-root", required=True, help="the L2 run tree's predictions directory (<challenge>/<obs_id>.nii.gz)")
    parser.add_argument("--inputs-root", required=True, help="the L2 run tree's instrument inputs directory (<challenge>/<obs_id>_<suffix>.nii.gz)")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the baseline report (never git)")
    parser.add_argument(
        "--challenges",
        nargs="+",
        default=list(CHALLENGES),
        choices=CHALLENGES,
        help="challenges to re-measure (default: all five; a subset cannot satisfy the "
        "recorded-history anchors, which cover every challenge -- such a run reports "
        "matched=False and exits 1, fine for smoke, not for the baseline)",
    )
    parser.add_argument("--bootstrap-b", type=int, default=BOOTSTRAP_B, help=f"bootstrap resamples (default {BOOTSTRAP_B})")
    parser.add_argument("--run-id", default=None, help="the candidate's L2 terminal-acceptance run id, recorded into the report")
    args = parser.parse_args(argv)

    rows = [row for row in MeasurementTable.read(args.measurements) if row["challenge"] in set(args.challenges)]
    if not rows:
        raise DiagnosticError(f"no observations for challenges {sorted(set(args.challenges))} in {args.measurements}")
    mask_repository = NiftiMaskRepository(args.pred_root)
    inputs_repository = InstrumentInputsRepository(args.inputs_root)
    readings = FixedWorldPairedReadings(FIXED_WORLD_WINDOW).read_cases(rows, mask_repository, inputs_repository)

    per_challenge = {}
    for challenge in sorted({reading["challenge"] for reading in readings}):
        per_challenge[challenge] = {
            spec.name: FixedWorldBaselineReport.quantity_block(readings, challenge, spec, args.bootstrap_b) for spec in QUANTITIES
        }
    et_readings = EtDiscrimination(args.bootstrap_b).discriminate(rows)
    job_a_anchor = JobAAnchor.verify(per_challenge)
    job_b_anchor = JobBAnchor.verify(et_readings)
    field_of_view = FixedWorldFieldScan(FIXED_WORLD_WINDOW).scan(rows, mask_repository)

    report = FixedWorldBaselineReport(Path(args.measurements), Path(args.pred_root), Path(args.inputs_root), args.bootstrap_b, run_id=args.run_id)
    json_path, md_path = report.write(readings, per_challenge, field_of_view, et_readings, job_a_anchor, job_b_anchor, Path(args.output_dir))
    skipped = sum(1 for reading in readings if reading["excluded"])
    brain_missing = sum(1 for reading in readings if reading["brain_missing"])
    print(f"[OK] {len(readings)} cases ({skipped} skipped, {brain_missing} without inputs, variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    print(f"[ANCHOR] job A: {'matched' if job_a_anchor['matched'] else 'DRIFT'}; job B: {'matched' if job_b_anchor['matched'] else 'DRIFT'}")
    if not (job_a_anchor["matched"] and job_b_anchor["matched"]):
        print("[ANCHOR-FAIL] 记录字面量对账漂移——窗口算术、工件或种子已变动,读数不可作为记录历史的对偶;报告已写出供取证")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
