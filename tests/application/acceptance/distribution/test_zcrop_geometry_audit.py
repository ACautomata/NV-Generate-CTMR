"""Geometry audit of diagnostic job A (issue #217, parent #205), observed as pytest.

Job A (zcrop_compensation.py, #206/#213) registered the modality-label-stage
sampling geometry: 256x256x128 @ (0.94, 0.94, 1.36) mm resampled to 1 mm ->
174 z slices -> centre-cropped onto the instrument grid (crop start 9, 19
slices dropped). The holdout NIfTI artifacts, however, carry the sidecar
writer's unit affine ``np.diag([1, 1, 1])`` (discovered while executing job C,
#208/PR #216), so on the artifacts the 1 mm resample is a no-op and the
instrument chain PADS the 128-slice volume onto the 155-slice grid (13 below,
14 above) instead of cropping 19 slices. Under the workpiece affine the
generated grid index i sits at physical z = i - 13 mm, the coordinate artefact
is +13 mm (uncomp = true + 13), the overlapping window is [13, 141) mm, and
for any case whose tumour crosses no window edge

    comp_pad = uncomp - 13 = (uncomp + 9) - 22 = comp_crop - 22.

The prediction masks themselves are whatever the frozen chain produced (pad
geometry); a geometry assumption only re-interprets them. These tests pin, on
synthetic volumes with hand-computed expectations: the pad-window
construction, the +13 mm physical anchor, the -22 mm identity spread against
the registered crop window, the enlarged pad-window edge-case set (real
tumours crossing z=13, a superset of job A's crossing-z=9 list), the
field-of-view audit (real WT mass outside the generated content/declared z
domains, and generated mass inside the padding), and the diagnostic report
surface (variant=diagnostic, issue 217, job A-reproducing and slot-300 seeds).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.final_acceptance import MEASUREMENT_FIELDS, ClusterBootstrap, MeasurementTable
from ctmr.application.acceptance.distribution.zcrop_compensation import (
    DIAGNOSTIC_SEED_BASE,
    GEN_RESAMPLED_Z,
    INSTRUMENT_Z,
    OverlapWindow,
    PairedCompensation,
    ZCropCompensation,
)
from ctmr.application.acceptance.distribution.zcrop_geometry_audit import (
    GEN_WORKPIECE_Z,
    JOB_AUDIT_SEED_SLOT,
    FieldOfViewAudit,
    GeometryAuditReport,
    PairedGeometryAudit,
    WorkpieceGeometry,
    main,
)


class InMemoryMaskRepository:
    """A dict-backed stand-in for the NIfTI mask repository (keys are pseudo-quad obs ids)."""

    def __init__(self, masks):
        self.masks = masks

    def wt_mask(self, challenge, obs_id):
        return self.masks.get(obs_id)


ARRAY_SHAPE = (INSTRUMENT_Z, 240, 240)  # zyx, the frozen prediction shape


# ------------------------------------------------------------- pad geometry


def test_pad_start_of_the_workpiece_affine_geometry_is_thirteen():
    """(target - source) // 2, the CenterCropOrPad pad rule (grid.py)."""
    assert WorkpieceGeometry.pad_start(GEN_WORKPIECE_Z, INSTRUMENT_Z) == 13
    assert WorkpieceGeometry.pad_start(INSTRUMENT_Z, INSTRUMENT_Z) == 0
    assert WorkpieceGeometry.pad_start(129, INSTRUMENT_Z) == 13  # odd remainder rounds down


def test_overlap_window_matches_the_workpiece_affine_geometry():
    """The generated content domain [13, 141) IS the overlap window: the pad
    margins carry no generated mass, and the crop_start slot carries the NEGATIVE
    grid-to-physical offset (generated array index 0 sits at declared z = -13)."""
    window = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
    assert window == OverlapWindow(gen_slice=slice(13, 141), real_slice=slice(13, 141), crop_start=-13, phys_lo=13, phys_hi=141)
    assert window.phys_hi - window.phys_lo == GEN_WORKPIECE_Z  # 128 mm, narrower than job A's 146 mm


def test_pad_anchor_one_physical_tumour_uncompensated_plus_thirteen_compensated_zero():
    """The pad-geometry physical anchor, dual to job A's -9 mm one: a tumour at
    the same physical slab shows a +13 mm uncompensated centroid diff (the pad
    layout shifts the generated content up 13 slices on the instrument grid)
    and exactly 0 mm compensated."""
    window = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
    real = ZCropCompensation.restrict_and_measure(_slab([(40, 60)]), window, side="real")
    gen = ZCropCompensation.restrict_and_measure(_slab([(53, 73)]), window, side="gen")
    assert real["centroid_z_mm"] == pytest.approx(49.5)
    assert gen["centroid_z_mm"] == pytest.approx(49.5)  # array 62.5 - pad start 13
    uncompensated_diff = 62.5 - 49.5  # raw grid indices as the instrument subtracts them
    assert uncompensated_diff == pytest.approx(13.0)
    assert gen["centroid_z_mm"] - real["centroid_z_mm"] == pytest.approx(0.0)


def _slab(slices):
    """A WT-only label volume (labels 1) filling the given z ranges, 20x20 in xy."""
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    for z_lo, z_hi in slices:
        mask[z_lo:z_hi, 100:120, 100:120] = 1
    return mask


# --------------------------------------------- identity spread against job A


class _IdentityCase:
    """One in-window case, the same physical slab under both geometry readings.

    real grid [40, 60), generated grid [53, 73) (pad layout of physical [40, 60)).
    """

    @staticmethod
    def rows():
        return [
            {"case": "CASE-A", "challenge": "GLI", "side": "real", "vol_wt_ml": "8.0", "cz_wt_mm": "49.5"},
            {"case": "CASE-A", "challenge": "GLI", "side": "gen", "vol_wt_ml": "8.0", "cz_wt_mm": "62.5"},
        ]

    @staticmethod
    def masks():
        return InMemoryMaskRepository({"CASE-A__real": _slab([(40, 60)]), "CASE-A__gen": _slab([(53, 73)])})


def test_identity_spread_comp_pad_is_comp_crop_minus_twenty_two():
    """For an in-window case the two geometry compensations differ by exactly
    the spread +9 - (-13) = 22 mm; the readings are re-interpretations of one
    and the same prediction mask, so the uncompensated CSV value is shared."""
    crop_window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    pad_window = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
    paired = PairedCompensation(crop_window, bootstrap_b=200)
    reading_crop = paired.read_cases(_IdentityCase.rows(), _IdentityCase.masks())[0]
    reading_pad = PairedCompensation(pad_window, bootstrap_b=200).read_cases(_IdentityCase.rows(), _IdentityCase.masks())[0]
    assert reading_crop["centroid_wt_z_uncomp"] == pytest.approx(13.0)
    assert reading_crop["centroid_wt_z_comp"] == pytest.approx(22.0)  # uncomp + 9: job A's reading
    assert reading_pad["centroid_wt_z_comp"] == pytest.approx(0.0)  # uncomp - 13
    assert reading_pad["centroid_wt_z_comp"] == pytest.approx(reading_crop["centroid_wt_z_comp"] - 22.0)


def test_pad_window_edge_case_invisible_to_job_a():
    """A real tumour at [10, 20) crosses the pad floor z=13 but not job A's
    floor z=9: job A's window leaves it intact (identity holds there), while
    the pad window truncates it and the in-window real centroid rises 1.5 mm.
    The pad-window edge-case set is therefore a SUPERSET of job A's 10 cases."""
    crop_window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    pad_window = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
    rows = [
        {"case": "CASE-E", "challenge": "MEN", "side": "real", "vol_wt_ml": "4.0", "cz_wt_mm": "14.5"},
        {"case": "CASE-E", "challenge": "MEN", "side": "gen", "vol_wt_ml": "4.0", "cz_wt_mm": "54.5"},
    ]
    masks = InMemoryMaskRepository({"CASE-E__real": _slab([(10, 20)]), "CASE-E__gen": _slab([(50, 60)])})
    reading_crop = PairedCompensation(crop_window, bootstrap_b=200).read_cases(rows, masks)[0]
    reading_pad = PairedCompensation(pad_window, bootstrap_b=200).read_cases(rows, masks)[0]
    assert reading_crop["centroid_wt_z_comp"] == pytest.approx(reading_crop["centroid_wt_z_uncomp"] + 9.0)
    # pad: gen physical 54.5 - 13 = 41.5; real in-window [13, 20) centroid 16 (rose 1.5)
    assert reading_pad["centroid_wt_z_comp"] == pytest.approx(25.5)
    assert reading_pad["centroid_wt_z_comp"] == pytest.approx(reading_pad["centroid_wt_z_uncomp"] - 13.0 - 1.5)


def test_real_tumour_wholly_below_the_pad_window_has_no_compensated_centroid():
    """A real tumour entirely below z=13 keeps its uncompensated reading but has
    no in-window mass under the pad geometry -- a measurement result (None),
    never an error; job A's [9, 155) window still saw three of its slices."""
    crop_window = ZCropCompensation.overlap_window(GEN_RESAMPLED_Z, INSTRUMENT_Z)
    pad_window = WorkpieceGeometry.overlap_window(GEN_WORKPIECE_Z, INSTRUMENT_Z)
    rows = [
        {"case": "CASE-L", "challenge": "MEN", "side": "real", "vol_wt_ml": "1.6", "cz_wt_mm": "9.5"},
        {"case": "CASE-L", "challenge": "MEN", "side": "gen", "vol_wt_ml": "4.0", "cz_wt_mm": "54.5"},
    ]
    masks = InMemoryMaskRepository({"CASE-L__real": _slab([(8, 12)]), "CASE-L__gen": _slab([(50, 60)])})
    reading_crop = PairedCompensation(crop_window, bootstrap_b=200).read_cases(rows, masks)[0]
    reading_pad = PairedCompensation(pad_window, bootstrap_b=200).read_cases(rows, masks)[0]
    assert reading_crop["centroid_wt_z_comp"] is not None  # [9, 12) survives job A's window
    assert reading_pad["centroid_wt_z_comp"] is None  # nothing survives [13, 141)


# --------------------------------------------------------- paired audit core


def test_paired_audit_runs_both_windows_and_reports_identity_residuals():
    """The audit pairs the crop-window and pad-window re-measurements per case
    and flags the identity residual comp_pad - (uncomp - 13): zero in-window,
    negative when the real tumour crosses the pad floor."""
    rows = _IdentityCase.rows() + [
        {"case": "CASE-E", "challenge": "MEN", "side": "real", "vol_wt_ml": "4.0", "cz_wt_mm": "14.5"},
        {"case": "CASE-E", "challenge": "MEN", "side": "gen", "vol_wt_ml": "4.0", "cz_wt_mm": "54.5"},
    ]
    repo = InMemoryMaskRepository({**_IdentityCase.masks().masks, "CASE-E__real": _slab([(10, 20)]), "CASE-E__gen": _slab([(50, 60)])})
    audit = PairedGeometryAudit(bootstrap_b=200).audit(rows, repo)
    by_case = {(item["challenge"], item["case"]): item for item in audit["per_case"]}
    in_window = by_case[("GLI", "CASE-A")]
    assert in_window["comp_crop"] == pytest.approx(22.0)
    assert in_window["comp_pad"] == pytest.approx(0.0)
    assert in_window["identity_residual"] == pytest.approx(0.0)
    edge = by_case[("MEN", "CASE-E")]
    assert edge["identity_residual"] == pytest.approx(-1.5)
    assert audit["window_edge_cases"]["crop_window"] == []  # job A's list, empty on this synthetic pair
    assert [(item["challenge"], item["case"]) for item in audit["window_edge_cases"]["pad_window"]] == [("MEN", "CASE-E")]


def test_paired_audit_skips_a_missing_prediction():
    repo = InMemoryMaskRepository({})
    audit = PairedGeometryAudit(bootstrap_b=200).audit(_IdentityCase.rows(), repo)
    assert audit["per_case"][0]["excluded"] == "missing_prediction"
    assert audit["per_case"][0]["comp_pad"] is None


# ---------------------------------------------------------- field-of-view audit


def test_field_of_view_audit_quantifies_real_mass_outside_the_generated_domains():
    """Real WT mass below the content floor (a placement gap: the pad margins
    sit INSIDE the declared domain) and above the content ceiling / declared
    ceiling (the true field-of-view mismatch, both measured from the mask)."""
    mask = np.zeros(ARRAY_SHAPE, dtype=np.uint8)
    mask[5:7, 100:120, 100:120] = 1  # 2 layers below the content floor z=13
    mask[130:132, 100:120, 100:120] = 1  # 2 layers in [128, 141): declared-out, content-in
    mask[145:147, 100:120, 100:120] = 1  # 2 layers above both ceilings (z >= 141 > 128)
    reading = FieldOfViewAudit.real_wt_outside(mask)
    assert reading["below_content_ml"] == pytest.approx(0.8)  # 2 layers x 400 voxels x 0.001
    assert reading["above_content_ml"] == pytest.approx(0.8)
    assert reading["above_declared_ml"] == pytest.approx(1.6)  # 4 layers at z >= 128


def test_field_of_view_audit_of_a_fully_in_domain_mask_is_all_zero():
    reading = FieldOfViewAudit.real_wt_outside(_slab([(40, 60)]))
    assert reading == {"below_content_ml": 0.0, "above_content_ml": 0.0, "above_declared_ml": 0.0}


def test_field_of_view_audit_counts_generated_mass_inside_the_padding():
    """The pad margins of a generated prediction should be empty (pure zeros
    went in); any mass there is an instrument-chain anomaly, counted in ml."""
    gen = _slab([(53, 73)])
    assert FieldOfViewAudit.gen_mass_in_padding(gen) == pytest.approx(0.0)
    gen[2, 100:120, 100:120] = 1  # z=2 < pad start 13
    gen[150, 100:120, 100:120] = 1  # z=150 >= 141
    assert FieldOfViewAudit.gen_mass_in_padding(gen) == pytest.approx(0.8)


# ------------------------------------------------------------------- report


def _audit_for_report():
    rows = _IdentityCase.rows()
    audit = PairedGeometryAudit(bootstrap_b=200).audit(rows, _IdentityCase.masks())
    return audit


def test_report_writes_json_and_markdown_with_two_geometry_reading_blocks(tmp_path):
    report = GeometryAuditReport(measurements_path=Path("/controlled/measurements.csv"), pred_root=Path("/controlled/predictions"), bootstrap_b=200)
    json_path, md_path = report.write(_audit_for_report(), None, tmp_path)
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "zcrop-geometry-audit-diagnostic/1"
    assert payload["issue"] == 217
    assert payload["variant"] == "diagnostic"
    assert "不产生任何验收判定" in payload["disclaimer"]
    assert payload["geometries"]["registered_crop"]["crop_start"] == 9
    assert payload["geometries"]["workpiece_pad"]["pad_start"] == 13
    assert payload["geometries"]["workpiece_pad"]["content_domain_mm"] == [13, 141]
    assert payload["geometries"]["workpiece_pad"]["declared_domain_mm"] == [0, 128]
    gli = payload["per_challenge"]["GLI"]["centroid_wt_z"]
    assert gli["comp_crop"]["median"] == pytest.approx(22.0)
    assert gli["comp_pad"]["median"] == pytest.approx(0.0)
    assert gli["attribution_crop"]["classification"] in {"candidate_dominant", "measurement_axis_dominant", "mixed", "no_central_shift"}
    assert gli["attribution_pad"]["classification"] == "measurement_axis_dominant"  # the pad compensation removes the full offset
    md = md_path.read_text()
    assert "复核" in md and "GLI" in md and "centroid_wt_z" in md
    assert "comp_pad" in md and "comp_crop" in md


def test_report_seeds_reproduce_job_a_and_take_the_next_free_slot():
    """comp_crop re-draws job A's compensated seed bit-stream exactly (slot 1 +
    stride 100); comp_pad takes slot 300, the next free slot after job B's 200."""
    assert JOB_AUDIT_SEED_SLOT == 300
    report = GeometryAuditReport(measurements_path=Path("/controlled/measurements.csv"), pred_root=Path("/controlled/predictions"), bootstrap_b=200)
    payload_seeds = report.centroid_seeds("GLI")
    assert payload_seeds["uncomp"] == DIAGNOSTIC_SEED_BASE + 1 * 1000 + 1
    assert payload_seeds["comp_crop"] == DIAGNOSTIC_SEED_BASE + 1 * 1000 + 1 + 100
    assert payload_seeds["comp_pad"] == DIAGNOSTIC_SEED_BASE + 1 * 1000 + JOB_AUDIT_SEED_SLOT


def test_report_pad_ci_matches_its_slot_seed_bit_stream():
    values = [1.0, 2.0, 3.0]
    stats = PairedGeometryAudit(bootstrap_b=200).summary_stats(values, "GLI", "comp_pad")
    expected = ClusterBootstrap(200).ci90([[v] for v in values], DIAGNOSTIC_SEED_BASE + 1 * 1000 + JOB_AUDIT_SEED_SLOT)
    assert stats["ci90_low"] == expected["low"]
    assert stats["ci90_high"] == expected["high"]


# ---------------------------------------------------------------- CLI e2e


def test_cli_end_to_end_writes_json_and_markdown(tmp_path):
    pred_root = tmp_path / "predictions" / "GLI"
    pred_root.mkdir(parents=True)
    for obs_id, slab in (("CASE-A__real", [(40, 60)]), ("CASE-A__gen", [(53, 73)])):
        sitk.WriteImage(sitk.GetImageFromArray(_slab(slab)), str(pred_root / f"{obs_id}.nii.gz"))

    def _row(obs_id, side, cz):
        row = dict.fromkeys(MEASUREMENT_FIELDS, "")
        row.update(obs_id=obs_id, challenge="GLI", case="CASE-A", side=side, cz_wt_mm=cz, vol_wt_ml="8.0")
        return row

    csv_path = tmp_path / "measurements.csv"
    MeasurementTable.write([_row("CASE-A__real", "real", "49.5"), _row("CASE-A__gen", "gen", "62.5")], csv_path)
    exit_code = main(
        [
            "--measurements",
            str(csv_path),
            "--pred-root",
            str(tmp_path / "predictions"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 0
    payload = json.loads((tmp_path / "out" / "zcrop_geometry_audit_diagnostic.json").read_text())
    assert payload["per_case"][0]["identity_residual"] == pytest.approx(0.0)
    assert (tmp_path / "out" / "zcrop_geometry_audit_diagnostic.md").is_file()
