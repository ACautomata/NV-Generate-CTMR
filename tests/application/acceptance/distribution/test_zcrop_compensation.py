"""Diagnostic job A (issue #206): z-crop compensated re-measurement, observed as pytest.

The z-crop axis (RC-1, parent #205): in the frozen L2 instrument chain the
modality-label-conditioned candidate's generated volumes are centre-cropped from
the resampled 241x241x174 to the 240x240x155 instrument grid (19 z slices
dropped, crop start 9), while the real side passes through natively at 155
slices -- the measurement domains are asymmetric. Job A restricts BOTH sides to
the overlapping z window [9, 155) mm and re-measures vol_wt_rel / centroid_wt_z
per case, so the compensated readings can be compared head-to-head with the
uncompensated instrument readings.

Every test here runs on synthetic volumes with hand-computed expectations; the
mask source is an in-memory fake, so the suite never touches sitk file IO. The
physical anchor case: a tumour at the SAME physical location shows a -9 mm
centroid-z difference uncompensated (the generated array origin sits at physical
z=9 mm) and exactly 0 mm compensated.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ctmr.application.acceptance.distribution.final_acceptance import FROZEN_ENVELOPES
from ctmr.application.acceptance.distribution.zcrop_compensation import (
    AttributionJudge,
    DiagnosticReport,
    OverlapWindow,
    PairedCompensation,
    ZCropCompensation,
)


class InMemoryMaskRepository:
    """A dict-backed stand-in for the NIfTI mask repository (keys are pseudo-quad obs ids)."""

    def __init__(self, masks):
        self.masks = masks

    def wt_mask(self, challenge, obs_id):
        return self.masks.get(obs_id)


# The registered modality-label-stage sampling geometry (stage code P1 in the
# experiment ledger): 256x256x128 @ (0.94, 0.94, 1.36) resampled to 1 mm
# -> (241, 241, 174), centre-cropped onto the (240, 240, 155) instrument grid.
GEN_RESAMPLED_Z = 174
INSTRUMENT_Z = 155

ARRAY_SHAPE = (INSTRUMENT_Z, 240, 240)  # zyx, the frozen prediction shape


# ------------------------------------------------------------------- crop geometry


def test_crop_start_of_the_registered_modality_label_geometry_is_nine():
    assert ZCropCompensation.crop_start(GEN_RESAMPLED_Z, INSTRUMENT_Z) == 9
    assert ZCropCompensation.crop_start(INSTRUMENT_Z, INSTRUMENT_Z) == 0  # no crop, no offset
    assert ZCropCompensation.crop_start(175, INSTRUMENT_Z) == 10  # odd remainder rounds down


def test_overlap_window_matches_the_registered_geometry():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    assert window == OverlapWindow(gen_slice=slice(0, 146), real_slice=slice(9, 155), crop_start=9, phys_lo=9, phys_hi=155)
    assert len(range(*window.gen_slice.indices(INSTRUMENT_Z))) == len(range(*window.real_slice.indices(INSTRUMENT_Z)))  # equal domains
    assert window.phys_hi - window.phys_lo == 146  # the overlapping window is 146 mm on both sides


def test_overlap_window_without_crop_is_the_full_real_domain():
    window = ZCropCompensation.overlap_window(INSTRUMENT_Z, INSTRUMENT_Z)
    assert window.gen_slice == slice(0, 155)
    assert window.real_slice == slice(0, 155)


# --------------------------------------------------- re-measurement on synthetic masks


def _tumour_mask(slices):
    """A WT-only label volume (labels 1) filling the given z ranges, 20x20 in xy."""
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    for z_lo, z_hi in slices:
        mask[z_lo:z_hi, 100:120, 100:120] = 1
    return mask


def test_remeasure_maps_both_sides_of_one_physical_tumour_to_the_same_physical_centroid():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    # the same physical slab z in [40, 60) mm: generated array index = physical - 9
    gen_volume = 20 * 20 * 20 * 0.001  # 400 voxels = 8.0 ml
    gen = ZCropCompensation.restrict_and_measure(_tumour_mask([(31, 51)]), window, side="gen")
    real = ZCropCompensation.restrict_and_measure(_tumour_mask([(40, 60)]), window, side="real")
    assert gen["vol_ml"] == pytest.approx(gen_volume)
    assert real["vol_ml"] == pytest.approx(gen_volume)
    assert gen["centroid_z_mm"] == pytest.approx(49.5)  # array 40.5 + crop start 9
    assert real["centroid_z_mm"] == pytest.approx(49.5)  # array index IS physical on the real side


def test_remeasure_anchor_uncompensated_minus_nine_compensated_zero():
    """The physical anchor: one tumour, one physical place, two measurement domains."""
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    gen = ZCropCompensation.restrict_and_measure(_tumour_mask([(31, 51)]), window, side="gen")
    real = ZCropCompensation.restrict_and_measure(_tumour_mask([(40, 60)]), window, side="real")
    uncompensated_diff = 41 - 50  # raw array indices as the instrument subtracts them
    compensated_diff = gen["centroid_z_mm"] - real["centroid_z_mm"]
    assert uncompensated_diff == -9
    assert compensated_diff == pytest.approx(0.0)


def test_remeasure_drops_real_slices_below_the_window():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    # a slab spanning z in [0, 20) mm: slices [0, 9) fall outside the overlap window
    full = ZCropCompensation.restrict_and_measure(_tumour_mask([(0, 20)]), window, side="real")
    restricted = ZCropCompensation.restrict_and_measure(_tumour_mask([(9, 20)]), window, side="real")
    assert full["vol_ml"] == pytest.approx(restricted["vol_ml"])  # exactly the in-window voxels survive
    assert full["centroid_z_mm"] == pytest.approx((9 + 19) / 2)  # centroid pushed inside the window


def test_remeasure_drops_generated_slices_at_or_above_the_window():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    # generated array z in [150, 155) = physical [159, 164) mm: wholly outside [9, 155)
    outside = ZCropCompensation.restrict_and_measure(_tumour_mask([(150, 155)]), window, side="gen")
    assert outside["vol_ml"] == 0.0
    assert outside["centroid_z_mm"] is None


def test_remeasure_of_an_empty_wt_region_reports_zero_volume_and_no_centroid():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    reading = ZCropCompensation.restrict_and_measure(np.zeros(ARRAY_SHAPE, dtype=np.uint8), window, side="real")
    assert reading == {"vol_ml": 0.0, "centroid_z_mm": None}


def test_remeasure_keeps_all_three_wt_labels():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    mask[30, 100, 100] = 1  # label 1
    mask[40, 100, 100] = 2  # label 2
    mask[50, 100, 100] = 3  # label 3
    reading = ZCropCompensation.restrict_and_measure(mask, window, side="real")
    assert reading["vol_ml"] == pytest.approx(0.003)
    assert reading["centroid_z_mm"] == pytest.approx(40.0)  # mean of physical 30+40+50


# ------------------------------------------------------------- paired case readings


class _FixedWindows:
    """A two-case synthetic measurement table (csv rows as dicts)."""

    @staticmethod
    def rows():
        return [
            {"case": "CASE-A", "challenge": "GLI", "side": "real", "vol_wt_ml": "10.0", "cz_wt_mm": "77.5"},
            {"case": "CASE-A", "challenge": "GLI", "side": "gen", "vol_wt_ml": "15.0", "cz_wt_mm": "70.0"},
            {"case": "CASE-B", "challenge": "GLI", "side": "real", "vol_wt_ml": "8.0", "cz_wt_mm": "60.0"},
            {"case": "CASE-B", "challenge": "GLI", "side": "gen", "vol_wt_ml": "0.0", "cz_wt_mm": ""},
        ]

    @staticmethod
    def masks():
        real_a = _tumour_mask([(40, 60)])  # physical centroid 50, 8.0 ml
        gen_a = _tumour_mask([(31, 51)])  # same physical slab -> compensated diff 0
        real_b = _tumour_mask([(60, 70)])  # 4.0 ml, physical centroid 64.5
        gen_b = np.zeros(ARRAY_SHAPE, dtype=np.uint8)  # empty prediction
        return InMemoryMaskRepository({"CASE-A__real": real_a, "CASE-A__gen": gen_a, "CASE-B__real": real_b, "CASE-B__gen": gen_b})


def test_paired_compensation_reproduces_the_uncompensated_csv_readings():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    paired = PairedCompensation(window, bootstrap_b=200)
    readings = paired.read_cases(_FixedWindows.rows(), _FixedWindows.masks())
    case_a = next(item for item in readings if item["case"] == "CASE-A")
    assert case_a["vol_wt_rel_uncomp"] == pytest.approx(0.5)  # (15 - 10) / 10
    assert case_a["centroid_wt_z_uncomp"] == pytest.approx(-7.5)  # 70.0 - 77.5


def test_paired_compensation_compensates_against_the_in_memory_masks():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    paired = PairedCompensation(window, bootstrap_b=200)
    readings = paired.read_cases(_FixedWindows.rows(), _FixedWindows.masks())
    case_a = next(item for item in readings if item["case"] == "CASE-A")
    assert case_a["vol_wt_ml_real_comp"] == pytest.approx(8.0)
    assert case_a["vol_wt_ml_gen_comp"] == pytest.approx(8.0)
    assert case_a["vol_wt_rel_comp"] == pytest.approx(0.0)  # identical slabs -> zero relative diff
    assert case_a["centroid_wt_z_comp"] == pytest.approx(0.0)  # the -9 mm coordinate artefact is gone


def test_paired_compensation_keeps_empty_generation_in_volumes_but_excludes_centroid():
    """The judge's rule: a generated-side empty prediction stays in the volume distributions
    at rel diff -1.0; centroid axes need a non-empty mask on both sides."""
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    paired = PairedCompensation(window, bootstrap_b=200)
    readings = paired.read_cases(_FixedWindows.rows(), _FixedWindows.masks())
    case_b = next(item for item in readings if item["case"] == "CASE-B")
    assert case_b["excluded"] is None
    assert case_b["vol_wt_rel_uncomp"] == pytest.approx(-1.0)  # (0 - 8) / 8, kept
    assert case_b["vol_wt_rel_comp"] == pytest.approx(-1.0)  # compensated: (0 - 4) / 4, kept
    assert case_b["centroid_wt_z_uncomp"] is None
    assert case_b["centroid_wt_z_comp"] is None


def test_paired_compensation_reports_a_missing_prediction_as_skipped():
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    paired = PairedCompensation(window, bootstrap_b=200)
    rows = [row for row in _FixedWindows.rows() if row["case"] == "CASE-A"]
    repo = _FixedWindows.masks()
    del repo.masks["CASE-A__gen"]  # simulate a cleaned-up prediction file
    readings = paired.read_cases(rows, repo)
    assert readings[0]["excluded"] == "missing_prediction"


def test_paired_compensation_treats_a_run_fail_placeholder_row_as_undefined():
    """A run-fail placeholder row carries empty quantity cells; the uncompensated
    quantities measure as undefined -- the exclusion the formal judge applies."""
    paired = PairedCompensation(ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z), bootstrap_b=200)
    rows = [
        {"case": "CASE-A", "challenge": "GLI", "side": "real", "vol_wt_ml": "10.0", "cz_wt_mm": "77.5"},
        {"case": "CASE-A", "challenge": "GLI", "side": "gen", "vol_wt_ml": "", "cz_wt_mm": ""},  # placeholder
    ]
    reading = paired.read_cases(rows, _FixedWindows.masks())[0]
    assert reading["vol_wt_rel_uncomp"] is None
    assert reading["centroid_wt_z_uncomp"] is None
    assert reading["excluded"] is None  # the case itself stays in the report, quantities undefined


def test_paired_compensation_reports_obs_ids_per_the_assembly_plan_spelling():
    """The modality-label pseudo-quad assembly spells obs ids <case>__real / <case>__gen."""
    assert PairedCompensation.obs_id("BraTS-GLI-00001-000", "real") == "BraTS-GLI-00001-000__real"
    assert PairedCompensation.obs_id("BraTS-GLI-00001-000", "gen") == "BraTS-GLI-00001-000__gen"


# ------------------------------------------------------------------ attribution judge


def test_attribution_judge_calls_measurement_axis_dominant_when_compensation_resolves_most():
    verdict = AttributionJudge().classify(median_uncomp=9.0, median_comp=0.5)
    assert verdict["measurement_fraction"] == pytest.approx((9.0 - 0.5) / 9.0)
    assert verdict["classification"] == "measurement_axis_dominant"


def test_attribution_judge_calls_candidate_dominant_when_compensation_barely_moves():
    verdict = AttributionJudge().classify(median_uncomp=1.0, median_comp=0.95)
    assert verdict["measurement_fraction"] == pytest.approx(0.05)
    assert verdict["classification"] == "candidate_dominant"


def test_attribution_judge_calls_mixed_in_the_middle_band():
    verdict = AttributionJudge().classify(median_uncomp=2.0, median_comp=1.0)
    assert verdict["classification"] == "mixed"


def test_attribution_judge_handles_no_central_shift_and_worsening():
    assert AttributionJudge().classify(median_uncomp=0.0, median_comp=0.0)["classification"] == "no_central_shift"
    worsened = AttributionJudge().classify(median_uncomp=1.0, median_comp=2.0)  # compensation moved it away from zero
    assert worsened["measurement_fraction"] == 0.0
    assert worsened["classification"] == "candidate_dominant"


# --------------------------------------------------- summary statistics and report


def test_summary_stats_quantiles_follow_the_linear_interpolation_rule():
    stats = PairedCompensation.summary_stats([1.0, 2.0, 3.0, 4.0], bootstrap_b=100, seed=1)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["q05"] == pytest.approx(1.15)  # q*(n-1) linear rule
    assert stats["q95"] == pytest.approx(3.85)
    assert stats["ci90_low"] <= stats["median"] <= stats["ci90_high"]
    assert stats["n_cases"] == 4


def test_summary_stats_of_an_empty_case_set_is_all_none():
    stats = PairedCompensation.summary_stats([], bootstrap_b=100, seed=1)
    assert stats == {"median": None, "mean": None, "q05": None, "q95": None, "ci90_low": None, "ci90_high": None, "n_cases": 0}


def test_diagnostic_report_writes_json_and_markdown_with_the_disclaimer(tmp_path):
    window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    paired = PairedCompensation(window, bootstrap_b=200)
    readings = paired.read_cases(_FixedWindows.rows(), _FixedWindows.masks())
    report = DiagnosticReport(
        measurements_path=Path("/controlled/measurements.csv"),
        pred_root=Path("/controlled/predictions"),
        bootstrap_b=200,
    )
    json_path, md_path = report.write(readings, window, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["variant"] == "diagnostic"
    assert payload["issue"] == 206
    assert "不产生任何验收判定" in payload["disclaimer"]
    assert payload["per_case"][0]["case"] == "CASE-A"
    gli = payload["per_challenge"]["GLI"]["vol_wt_rel"]
    assert gli["uncomp"]["n_cases"] == 2  # CASE-A (0.5) and CASE-B (-1.0) both stay in the volume distribution
    assert gli["attribution"]["classification"] in {
        "measurement_axis_dominant",
        "candidate_dominant",
        "mixed",
        "no_central_shift",
    }
    md = md_path.read_text()
    assert "诊断作业 A" in md
    assert "GLI" in md and "vol_wt_rel" in md and "centroid_wt_z" in md


def test_diagnostic_report_wt_margins_read_the_frozen_envelopes():
    for challenge in ("GLI", "MEN", "METS", "PED", "SSA"):
        assert DiagnosticReport.wt_vol_margin(challenge) == FROZEN_ENVELOPES[challenge]["WT"][1]
        assert DiagnosticReport.wt_centroid_margin(challenge) == FROZEN_ENVELOPES[challenge]["WT"][2]
