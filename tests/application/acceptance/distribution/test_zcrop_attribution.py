"""The z-crop compensation attribution job, observed as pytest (issue #206, parent #205 job A).

The job re-measures WT relative volume and centroid z after restricting both
sides to the overlapping z range of the 19-slice crop, then attributes the L2
final-acceptance FAIL to the measurement axis vs the candidate defect. Every
number here is a synthetic fixture: the measurement CSV is produced from an
independent source of truth (scipy ``center_of_mass``, the frozen measurer's
own implementation) so the reconciliation guards are exercised for real, and
the final-acceptance JSON baseline is replayed with the exact judge seed
stream (registry order: vol_wt_rel=0, centroid_wt_z=3).

Geometry under test (handoff-pinned, independently verified): the instrument
array is 155 slices on both sides; the generated side's slice i sits at
physical z i+9 (its resampled 241x241x174 volume was centre-cropped with
start=(174-155)//2=9), the real side's slice i sits at physical z i. The
overlap windows are therefore gen [0,146) and real [9,155) -- both mapping to
the same physical range [9,155) mm via local+9.

Torch-marked tier (ADR-0015 §6): runs for real in the CI full-dependency tier.
"""

import csv
import json

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage

from ctmr.application.acceptance.distribution.final_acceptance import (
    CHALLENGE_SEED_OFFSET,
    FROZEN_ENVELOPES,
    GLOBAL_SEED,
    MEASUREMENT_FIELDS,
    ClusterBootstrap,
)
from ctmr.application.acceptance.distribution.measurement_run import MaskMeasurer
from ctmr.application.acceptance.distribution.zcrop_attribution import (
    CROP_START,
    OVERLAP_SLICES,
    QUANTITY_TOST_INDEX,
    RESAMPLED_Z,
    TARGET_Z,
    AttributionError,
    AttributionReport,
    CompensationAttributor,
    CompensationJudge,
    OverlapGeometry,
    OverlapRemeasurer,
    ZcropAttributionJob,
    main,
)

pytestmark = pytest.mark.torch

CHALLENGE = "GLI"
BOOTSTRAP_B = 200
VOL_MARGIN = FROZEN_ENVELOPES[CHALLENGE]["WT"][1]  # e_r_vol(GLI, WT)
CZ_MARGIN = FROZEN_ENVELOPES[CHALLENGE]["WT"][2]  # e_r_centroid(GLI, WT)

A_B_CASES = {
    "GLI-A-000": (((77, 120, 120), 30), ((87, 120, 120), 33)),  # crop-free pair, pure coordinate offset
    "GLI-B-000": (((5, 120, 120), 25), ((140, 120, 120), 30)),  # both tumours touch their crop band
}


# ── synthetic fixtures ──────────────────────────────────────────────────


def _sphere(shape, center, radius):
    """Three-label tumour sphere (1=WT shell, 2, 3 core), array layout zyx."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d2 = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    out = np.zeros(shape, dtype=np.uint8)
    out[d2 <= (radius * 0.5) ** 2] = 3
    out[d2 <= (radius * 0.8) ** 2] = 2
    out[d2 <= radius**2] = 1
    return out


def _measure_with_scipy(mask):
    """The frozen-measurer source of truth, independent of the job's numpy math."""
    wt = np.isin(mask, (1, 2, 3))
    if not wt.any():
        return 0.0, None
    return float(wt.sum()) * 0.001, float(ndimage.center_of_mass(wt)[0])


def _plan_observations(cases):
    observations = []
    for case in cases:
        for side in ("real", "gen"):
            observations.append(
                {
                    "obs_id": f"{case}__{side}",
                    "challenge": CHALLENGE,
                    "case": case,
                    "side": side,
                    "anchor": None,
                    "channels": {},
                    "condition_mask": None,
                }
            )
    return observations


def _acceptance_entry(quantity, diffs_per_case, exclusions, margin):
    """Replays the final-acceptance TOST for one quantity, judge-identically."""
    ci = ClusterBootstrap(BOOTSTRAP_B).ci90(diffs_per_case, seed=GLOBAL_SEED + CHALLENGE_SEED_OFFSET[CHALLENGE] + QUANTITY_TOST_INDEX[quantity])
    if ci is None:
        return {
            "quantity": quantity,
            "margin": margin,
            "ci90_low": None,
            "ci90_high": None,
            "n_cases": 0,
            "n_excluded": sum(exclusions.values()),
            "exclusion_reasons": exclusions,
            "passed": False,
        }
    passed = ci["low"] >= -margin - 1e-12 and ci["high"] <= margin + 1e-12
    return {
        "quantity": quantity,
        "margin": margin,
        "ci90_low": ci["low"],
        "ci90_high": ci["high"],
        "n_cases": ci["n_cases"],
        "n_excluded": sum(exclusions.values()),
        "exclusion_reasons": exclusions,
        "passed": passed,
    }


def _write_case_tree(tmp_path, cases, tamper=None):
    """Masks + measurement CSV (scipy truth) + plan + final-acceptance JSON.

    ``cases`` maps case id to ((real_center, real_radius), (gen_center,
    gen_radius)); radius 0 writes an empty (all-zero) mask. ``tamper``
    mutates the baseline JSON for guard-failure tests.
    """
    pred_dir = tmp_path / "predictions" / CHALLENGE
    pred_dir.mkdir(parents=True)
    rows, tost_seed = [], {}
    for case, (real_spec, gen_spec) in cases.items():
        side_values = {}
        for side, (center, radius) in (("real", real_spec), ("gen", gen_spec)):
            mask = np.zeros((TARGET_Z, 240, 240), dtype=np.uint8) if radius == 0 else _sphere((TARGET_Z, 240, 240), center, radius)
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(pred_dir / f"{case}__{side}.nii.gz"))
            side_values[side] = _measure_with_scipy(mask)
            row = {field: "" for field in MEASUREMENT_FIELDS}
            row.update(
                obs_id=f"{case}__{side}",
                challenge=CHALLENGE,
                case=case,
                side=side,
                anchor="",
                vol_wt_ml=side_values[side][0],
                cz_wt_mm=side_values[side][1],
            )
            rows.append(row)
        tost_seed[case] = side_values
    table = tmp_path / "measurements.csv"
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    vol_diffs, vol_exclusions, cz_diffs, cz_exclusions = [], {}, [], {}
    for case in sorted(tost_seed):
        (rv, rc), (gv, gc) = tost_seed[case]["real"], tost_seed[case]["gen"]
        if rv:
            vol_diffs.append([(gv - rv) / rv])
        else:
            vol_exclusions["real_volume_zero"] = vol_exclusions.get("real_volume_zero", 0) + 1
            vol_diffs.append([])
        if rv and gv:
            cz_diffs.append([gc - rc])
        else:
            cz_exclusions["empty_mask_side"] = cz_exclusions.get("empty_mask_side", 0) + 1
            cz_diffs.append([])
    tost = [
        _acceptance_entry("vol_wt_rel", vol_diffs, vol_exclusions, VOL_MARGIN),
        _acceptance_entry("centroid_wt_z", cz_diffs, cz_exclusions, CZ_MARGIN),
    ]
    if tamper:
        tamper(tost)
    acceptance = {
        "run_id": "p1-20260822T131947Z",
        "bootstrap": {"B": BOOTSTRAP_B},
        "per_challenge": {CHALLENGE: {"tost": tost}},
    }
    report_path = tmp_path / "l2_final_acceptance_p1.json"
    report_path.write_text(json.dumps(acceptance, indent=2))
    plan = {
        "schema": "l2-final-acceptance-plan/1",
        "phase": "P1",
        "challenges": {CHALLENGE: {"n_cases": len(cases), "quota": 250, "provisional": True}},
        "observations": _plan_observations(cases),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return {"preds": tmp_path / "predictions", "table": table, "plan": plan_path, "report": report_path}


def _read_table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _pair_sides(rows):
    real_by_case = {row["case"]: row for row in rows if row["side"] == "real"}
    gen_by_case = {}
    for row in rows:
        if row["side"] == "gen":
            gen_by_case.setdefault(row["case"], []).append(row)
    return real_by_case, gen_by_case


# ── geometry ────────────────────────────────────────────────────────────


def test_geometry_matches_center_crop_or_pad_and_overlap_windows():
    """crop start = (174-155)//2 = 9, the domain/grid CenterCropOrPad formula;
    the overlap windows are gen [0,146) and real [9,155), both 146 slices, and
    both map local index -> the same physical range [9,155) mm via +9."""
    assert (RESAMPLED_Z, TARGET_Z) == (174, 155)
    assert CROP_START == (RESAMPLED_Z - TARGET_Z) // 2 == 9
    assert OVERLAP_SLICES == TARGET_Z - CROP_START == 146
    assert OverlapGeometry.gen_window() == slice(0, 146)
    assert OverlapGeometry.real_window() == slice(9, 155)
    # both sides' local slice 0 is physical z=9; the mapping is the same +9
    assert OverlapGeometry.overlap_z_mm(0) == 9
    assert OverlapGeometry.overlap_z_mm(145) == 154
    described = OverlapGeometry.describe()
    assert described["overlap_slices"] == 146
    assert described["crop_start"] == 9


# ── remeasurement ───────────────────────────────────────────────────────


def test_full_array_remeasurement_matches_frozen_measurer():
    """Global (before) re-measurement must reproduce the frozen measurer's
    scipy numbers within the 1e-9 tolerance (accumulation order differs);
    an empty mask measures as vol 0 with an undefined centroid."""
    rng = np.random.default_rng(7)
    pred = _sphere((TARGET_Z, 240, 240), (77, 120, 120), 30)
    pred[rng.random(pred.shape) < 0.01] = 1  # ragged boundary: not a perfect sphere
    truth_vol = MaskMeasurer.volumes_ml(pred)["WT"]
    _, _, truth_cz = MaskMeasurer.centroid_mm(pred, "WT")

    measured = OverlapRemeasurer.measure(pred, "gen")

    assert measured["vol_ml"] == pytest.approx(truth_vol, abs=1e-9)
    assert measured["cz_index"] == pytest.approx(truth_cz, abs=1e-9)
    empty = OverlapRemeasurer.measure(np.zeros((TARGET_Z, 240, 240), dtype=np.uint8), "real")
    assert empty["vol_ml"] == 0.0
    assert empty["cz_index"] is None
    assert empty["vol_ml_overlap"] == 0.0
    assert empty["cz_overlap_mm"] is None


def test_overlap_window_semantics_per_side():
    """Gen local i is physical i+9 (crop-free sphere: same volume, centroid
    shifted by exactly +9 mm); real local i maps the same way (window crops
    the bottom 9 slices, so an in-window real sphere keeps volume and
    physical centroid); out-of-window parts are cut on each side."""
    in_gen_window = OverlapRemeasurer.measure(_sphere((TARGET_Z, 240, 240), (87, 120, 120), 20), "gen")  # fits [0,146)
    assert in_gen_window["vol_ml_overlap"] == pytest.approx(in_gen_window["vol_ml"], abs=1e-9)
    assert in_gen_window["cz_overlap_mm"] == pytest.approx(in_gen_window["cz_index"] + 9, abs=1e-9)

    crossing_gen = OverlapRemeasurer.measure(_sphere((TARGET_Z, 240, 240), (140, 120, 120), 30), "gen")  # spills past 146
    assert crossing_gen["vol_ml_overlap"] < crossing_gen["vol_ml"]

    in_real_window = OverlapRemeasurer.measure(_sphere((TARGET_Z, 240, 240), (90, 120, 120), 20), "real")
    assert in_real_window["vol_ml_overlap"] == pytest.approx(in_real_window["vol_ml"], abs=1e-9)
    assert in_real_window["cz_overlap_mm"] == pytest.approx(in_real_window["cz_index"], abs=1e-9)

    crossing_real = OverlapRemeasurer.measure(_sphere((TARGET_Z, 240, 240), (5, 120, 120), 25), "real")  # spills below 9
    assert crossing_real["vol_ml_overlap"] < crossing_real["vol_ml"]


# ── pairing / exclusion semantics (registry reuse) ──────────────────────


def test_registry_pair_semantics_in_the_replay(tmp_path):
    """Pairing goes through the frozen QuantityFamily on both scales: real
    denominator zero excludes the relative quantity, an empty side excludes
    the centroid quantity, a generated empty prediction stays in at rel -1.0,
    and undefined (failed-run) rows exclude as undefined_measurement."""
    cases = {
        "GLI-A-000": (((90, 120, 120), 20), ((90, 120, 120), 0)),  # gen empty: rel -1 kept, centroid excluded
        "GLI-B-000": (((90, 120, 120), 0), ((90, 120, 120), 20)),  # real empty: both quantities exclude it
    }
    paths = _write_case_tree(tmp_path, cases)
    real_by_case, gen_by_case = _pair_sides(_read_table(paths["table"]))
    after_real, after_gen = _pair_sides(_read_table(paths["table"]))  # fresh rows: the after scale reads the real masks
    # a failed-run row has no measurement at all (empty cells), like real CSVs
    gen_by_case["GLI-A-000"][0]["vol_wt_ml"] = ""
    gen_by_case["GLI-A-000"][0]["cz_wt_mm"] = ""
    acceptance = json.loads(paths["report"].read_text())

    judge = CompensationJudge(acceptance)
    vol = judge.judge(CHALLENGE, "vol_wt_rel", real_by_case, gen_by_case, after_real, after_gen, verify_baseline=False)
    assert vol["exclusion_reasons"] == {"undefined_measurement": 1, "real_volume_zero": 1}
    assert vol["per_obs_diff"]["GLI-A-000__gen"]["before"] is None  # undefined before
    assert vol["per_obs_diff"]["GLI-A-000__gen"]["after"] == pytest.approx(-1.0)  # gen-empty stays in at -1

    judge = CompensationJudge(acceptance)
    cz = judge.judge(CHALLENGE, "centroid_wt_z", real_by_case, gen_by_case, after_real, after_gen, verify_baseline=False)
    assert cz["exclusion_reasons"] == {"empty_mask_side": 2}
    assert cz["ci90_after"]["n_cases"] == 0


# ── before replay + guards ──────────────────────────────────────────────


def test_before_replay_reproduces_final_acceptance_ci_and_exclusions(tmp_path):
    """The replayed before-CI (registry seed order: vol_wt_rel=0,
    centroid_wt_z=3) matches an independently computed judge-identical
    baseline, and the reconciliation guards accept honest fixtures."""
    paths = _write_case_tree(tmp_path, A_B_CASES)
    acceptance = json.loads(paths["report"].read_text())
    real_by_case, gen_by_case = _pair_sides(_read_table(paths["table"]))

    judge = CompensationJudge(acceptance)
    for quantity, margin in (("vol_wt_rel", VOL_MARGIN), ("centroid_wt_z", CZ_MARGIN)):
        entry = judge.judge(CHALLENGE, quantity, real_by_case, gen_by_case, real_by_case, gen_by_case)
        baseline = next(item for item in acceptance["per_challenge"][CHALLENGE]["tost"] if item["quantity"] == quantity)
        assert entry["ci90_before"]["low"] == pytest.approx(baseline["ci90_low"], abs=1e-9)
        assert entry["ci90_before"]["high"] == pytest.approx(baseline["ci90_high"], abs=1e-9)
        assert entry["before_passed"] == baseline["passed"]
        assert entry["margin"] == margin
        assert entry["n_excluded"] == baseline["n_excluded"] == 0


def test_replay_guard_rejects_tampered_baseline(tmp_path):
    """A baseline CI or exclusion count that the replay cannot reproduce is a
    FATAL, never a silent pass."""

    def bump_ci(tost):
        tost[0]["ci90_low"] += 1e-3

    def bump_exclusions(tost):
        tost[1]["n_excluded"] = 1

    for tamper in (bump_ci, bump_exclusions):
        paths = _write_case_tree(tmp_path / tamper.__name__, A_B_CASES, tamper=tamper)
        with pytest.raises(AttributionError):
            ZcropAttributionJob(paths["plan"], paths["table"], paths["preds"], paths["report"], tmp_path / f"out-{tamper.__name__}").run()


# ── compensation & attribution ──────────────────────────────────────────


def test_compensation_removes_pure_crop_offset(tmp_path):
    """A generated side shifted by exactly -9 slices against the real side
    (the RC-1 hypothesis): the before centroid diff sits at the array-index
    offset and FAILs the ±5.38 mm line, the compensated after diff is zero
    and passes -- the measurement axis alone explains the failure."""
    cases = {f"GLI-{tag}-000": (((90, 110, 130), 20), ((81, 110 + 7 * index, 130), 20)) for index, tag in enumerate(("A", "B"))}
    paths = _write_case_tree(tmp_path, cases)
    report = ZcropAttributionJob(paths["plan"], paths["table"], paths["preds"], paths["report"], tmp_path / "out").run()
    entry = next(item for item in report["per_quantity"] if item["quantity"] == "centroid_wt_z")
    assert entry["before_passed"] is False
    assert entry["after_passed"] is True
    assert report["attribution"]["verdict"] == "measurement_axis_dominant"


def test_attribution_verdicts_shares_and_fourth_state():
    """Three-way attribution with shares over the before-fail base; the
    fourth state (nothing failed before) is a boundary, not an error."""
    attributor = CompensationAttributor()

    def entry(challenge, quantity, before, after):
        return {"challenge": challenge, "quantity": quantity, "before_passed": before, "after_passed": after}

    all_repaired = [entry("GLI", quantity, False, True) for quantity in ("vol_wt_rel", "centroid_wt_z")]
    assert attributor.attribute(all_repaired)["verdict"] == "measurement_axis_dominant"
    assert attributor.attribute(all_repaired)["shares"]["measurement_axis"] == pytest.approx(1.0)

    all_persistent = [entry("GLI", quantity, False, False) for quantity in ("vol_wt_rel", "centroid_wt_z")]
    verdict = attributor.attribute(all_persistent)
    assert verdict["verdict"] == "candidate_defect_dominant"
    assert verdict["shares"]["candidate_defect"] == pytest.approx(1.0)

    verdict = attributor.attribute(
        [
            entry("GLI", "vol_wt_rel", False, True),
            entry("MEN", "vol_wt_rel", False, False),
            entry("GLI", "centroid_wt_z", False, True),
            entry("MEN", "centroid_wt_z", False, True),
        ]
    )
    assert verdict["verdict"] == "mixed"
    assert verdict["shares"]["measurement_axis"] == pytest.approx(0.75)
    assert verdict["per_axis"]["vol_wt_rel"]["verdict"] == "mixed"
    assert verdict["per_axis"]["centroid_wt_z"]["verdict"] == "measurement_axis_dominant"

    fourth = attributor.attribute([entry("GLI", "vol_wt_rel", True, True)])
    assert fourth["verdict"] == "no_failure_to_attribute"
    assert fourth["base"] == 0


# ── report shape ────────────────────────────────────────────────────────


def test_report_shape_variant_geometry_and_coordinate_note(tmp_path):
    """The diagnostic report declares variant=diagnostic, carries the overlap
    geometry, the cross-scale coordinate note and no acceptance verdict."""
    paths = _write_case_tree(tmp_path, A_B_CASES)
    report = ZcropAttributionJob(paths["plan"], paths["table"], paths["preds"], paths["report"], tmp_path / "out").run()

    assert report["variant"] == "diagnostic"
    assert "不产生" in report["disclaimer"]
    assert report["geometry"]["overlap_slices"] == 146
    assert "仪器数组" in report["coordinate_note"]
    assert set(report["attribution"]) >= {"verdict", "shares", "base", "per_axis", "items"}
    for quantity_entry in report["per_quantity"]:
        assert set(quantity_entry) >= {"challenge", "quantity", "ci90_before", "ci90_after", "before_passed", "after_passed", "margin"}


def test_per_case_rows_carry_both_scales(tmp_path):
    """Per-case rows keep the before (instrument array index) and after
    (overlap physical mm) scales side by side; the diff columns follow the
    smoke contract: a crop-free pair keeps the volume diff and gains exactly
    the +9 coordinate offset, crop-band pairs shrink the z diff and grow the
    (conservative) relative-volume diff."""
    paths = _write_case_tree(tmp_path, A_B_CASES)
    report = ZcropAttributionJob(paths["plan"], paths["table"], paths["preds"], paths["report"], tmp_path / "out").run()
    rows = {row["obs_id"]: row for row in report["per_case"]}
    a, b = rows["GLI-A-000__gen"], rows["GLI-B-000__gen"]

    assert a["vol_wt_rel_after"] == pytest.approx(a["vol_wt_rel_before"], abs=1e-9)
    assert abs(a["centroid_wt_z_after"] - a["centroid_wt_z_before"] - 9) < 1
    assert b["centroid_wt_z_after"] < b["centroid_wt_z_before"]
    assert b["vol_wt_rel_after"] > b["vol_wt_rel_before"]
    # before values are instrument-array indices; the real side keeps its
    # physical position under compensation (in-window sphere: index == mm)
    assert rows["GLI-A-000__real"]["cz_wt_mm_before"] == pytest.approx(77, abs=1e-6)
    assert rows["GLI-A-000__real"]["cz_wt_mm_after"] == pytest.approx(77, abs=1e-6)


# ── CLI / markdown ──────────────────────────────────────────────────────


def test_job_end_to_end_writes_report(tmp_path):
    """The full CLI on a synthetic tree: rc 0, report json + markdown on
    disk, reconciliation guards reported as passed."""
    paths = _write_case_tree(tmp_path, A_B_CASES)
    out = tmp_path / "out"
    rc = main(
        [
            "--plan",
            str(paths["plan"]),
            "--table",
            str(paths["table"]),
            "--preds",
            str(paths["preds"]),
            "--report",
            str(paths["report"]),
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    json_path = out / "zcrop_attribution_report.json"
    md_path = out / "zcrop_attribution_report.md"
    assert json_path.is_file() and md_path.is_file()
    report = json.loads(json_path.read_text())
    assert report["attribution"]["base"] <= 2
    assert report["reconciliation"]["guards_passed"] is True


def test_markdown_renders_attribution_and_disclaimer(tmp_path):
    """The markdown twin carries the attribution verdict, the diagnostic
    disclaimer and the reconciliation statement (tolerance wording, not
    bit-exact claims)."""
    paths = _write_case_tree(tmp_path, A_B_CASES)
    report = ZcropAttributionJob(paths["plan"], paths["table"], paths["preds"], paths["report"], tmp_path / "out").run()
    md = AttributionReport("p1-20260822T131947Z", BOOTSTRAP_B).markdown(report)
    assert "diagnostic" in md
    assert report["attribution"]["verdict"] in md
    assert "1e-9" in md
