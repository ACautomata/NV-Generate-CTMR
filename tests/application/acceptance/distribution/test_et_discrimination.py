"""Diagnostic job B (issue #207, parent #205): frozen-instrument ET discrimination,
observed as pytest.

The ET-missing axis (RC chain, parent #205): expert review (#58) reads missing
enhancing tumour in the modality-label-conditioned candidate's generated t1c;
job B turns that impression into per-challenge numbers. Every generated holdout
pseudo-quad (530 cases = 250 GLI + 200 MEN + 48 METS + 20 PED + 12 SSA) already
passed the frozen instrument during the P1 L2 terminal acceptance, so the job
re-reads the retained per-observation measurement CSV -- the instrument readings
themselves -- and never touches any frozen artifact.

Every test here runs on synthetic CSV rows with hand-computed expectations. The
reading vocabulary follows the #38 synthetic-domain precedents: empty pred =
instrument argmax all-zero (a measurement result, never a failure), Wilson 95%
upper bounds, and the terminal-acceptance convention that a generated-side
empty prediction stays in the volume distributions at rel diff -1.0.
"""

import json
from pathlib import Path

import pytest

from ctmr.application.acceptance.distribution.challenge_registry import (
    CHALLENGE_SEED_OFFSET,
    DIAGNOSTIC_SEED_BASE,
    DIAGNOSTIC_SEED_SLOTS,
    GLOBAL_SEED,
    HOLDOUT_QUOTAS,
)
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError, DiagnosticSeedAllocator
from ctmr.application.acceptance.distribution.et_discrimination import EtDiscrimination, EtDiscriminationReport, main
from ctmr.application.acceptance.distribution.measurement_table import MEASUREMENT_FIELDS, MeasurementTable


def _row(challenge, case, side, vol_et=None, *, pred_empty=0, input_fail=0, run_fail=0, hier_viol=0):
    """One synthetic measurement-CSV row, typed like the instrument writes it."""
    return {
        "obs_id": f"{case}__{side}",
        "challenge": challenge,
        "case": case,
        "side": side,
        "anchor": "",
        "input_fail": str(input_fail),
        "run_fail": str(run_fail),
        "hier_viol": str(hier_viol),
        "pred_empty": str(pred_empty),
        "vol_wt_ml": "10.0",
        "vol_tc_ml": "5.0",
        "vol_et_ml": "" if vol_et is None else str(vol_et),
        "brain_ml": "1000.0",
        "wt_brain": "0.01",
        "et_wt": "0.5" if vol_et else "",
        "cx_wt_mm": "100",
        "cy_wt_mm": "100",
        "cz_wt_mm": "70",
        "cx_tc_mm": "",
        "cy_tc_mm": "",
        "cz_tc_mm": "",
        "cx_et_mm": "",
        "cy_et_mm": "",
        "cz_et_mm": "",
        "cond_dice_wt": "",
        "cond_dice_tc": "",
        "cond_dice_et": "",
    }


# ------------------------------------------------------------------ detection statistics


def test_wilson_95_upper_matches_the_38_synthetic_domain_anchor():
    """The #38 report table publishes 0.1611 for 0/20 -- the same vocabulary here."""
    assert EtDiscrimination.wilson_95_upper(0, 20) == pytest.approx(0.1611, abs=5e-5)
    # METS empty-pred precedent 2/20 from the P1 direct-out front evidence
    assert EtDiscrimination.wilson_95_upper(2, 20) == pytest.approx(0.3010, abs=5e-4)
    assert EtDiscrimination.wilson_95_upper(5, 5) == 1.0  # capped
    assert EtDiscrimination.wilson_95_upper(0, 0) is None


def test_detection_counts_each_side_by_challenge():
    rows = [
        _row("GLI", "c1", "gen", 2.0),
        _row("GLI", "c2", "gen", 0.0),  # measured, ET absent
        _row("GLI", "c3", "gen", 1.0),
        _row("GLI", "c4", "gen", 0.0),
        _row("GLI", "c5", "gen", 3.0),
        _row("GLI", "c1", "real", 4.0),
        _row("GLI", "c2", "real", 5.0),
        _row("GLI", "c3", "real", 0.0),  # a real side can miss ET too
        _row("MEN", "m1", "gen", 1.0),
    ]
    readings = {r["challenge"]: r for r in EtDiscrimination().discriminate(rows)}
    gli = readings["GLI"]
    assert gli["gen"]["n"] == 5
    assert gli["gen"]["k_detected"] == 3
    assert gli["gen"]["rate"] == pytest.approx(0.6)
    assert gli["real"]["n"] == 3
    assert gli["real"]["k_detected"] == 2
    men = readings["MEN"]
    assert men["gen"]["k_detected"] == 1


def test_volume_distribution_quantiles_follow_the_linear_rule():
    rows = [_row("GLI", f"c{i}", "gen", vol) for i, vol in enumerate([0.0, 2.0, 4.0, 8.0, 16.0])]
    dist = EtDiscrimination().discriminate(rows)[0]["gen"]["vol_ml"]
    assert dist["median"] == pytest.approx(4.0)  # q*(n-1)=2 -> exact element
    assert dist["q05"] == pytest.approx(0.4)  # index 0.2 -> 0 + 0.2*2
    assert dist["q95"] == pytest.approx(14.4)  # index 3.8 -> 8 + 0.8*8
    assert dist["mean"] == pytest.approx(6.0)


# ------------------------------------------------------------------- pairing & empty pred


def test_pairing_classifies_real_only_as_the_et_missing_readout():
    rows = [
        _row("GLI", "both", "gen", 2.0),
        _row("GLI", "both", "real", 4.0),
        _row("GLI", "miss", "gen", 0.0),
        _row("GLI", "miss", "real", 4.0),  # ET missing on gen
        _row("GLI", "genly", "gen", 2.0),
        _row("GLI", "genly", "real", 0.0),
        _row("GLI", "none", "gen", 0.0),
        _row("GLI", "none", "real", 0.0),
    ]
    gli = {r["challenge"]: r for r in EtDiscrimination().discriminate(rows)}["GLI"]
    assert gli["pairing"] == {"both_detected": 1, "real_only": 1, "gen_only": 1, "neither": 1, "unpaired": 0}


def test_empty_prediction_tally_counts_gen_side_argmax_all_zero():
    rows = [
        _row("METS", "a", "gen", 0.0, pred_empty=1),  # whole volume empty -> ET absent
        _row("METS", "b", "gen", 0.0),  # WT present, ET absent: measured miss, NOT empty pred
        _row("METS", "c", "gen", 1.0),
        _row("METS", "a", "real", 3.0),
        _row("METS", "b", "real", 2.0),
        _row("METS", "c", "real", 4.0),
    ]
    mets = {r["challenge"]: r for r in EtDiscrimination().discriminate(rows)}["METS"]
    assert mets["empty_pred"]["gen"] == {"k": 1, "n": 3}
    assert mets["gen"]["k_detected"] == 1  # the empty-pred case sits inside the undetected count


def test_failed_rows_leave_the_denominator_but_hier_viol_stays():
    rows = [
        _row("GLI", "ok1", "gen", 2.0),
        _row("GLI", "ok2", "gen", 0.0),
        _row("GLI", "inp", "gen", None, input_fail=1),
        _row("GLI", "run", "gen", None, run_fail=1),
        _row("GLI", "hier", "gen", 1.0, hier_viol=1),  # measured; violation recorded, not excluded
        _row("GLI", "ok1", "real", 4.0),
        _row("GLI", "ok2", "real", 4.0),
        _row("GLI", "hier", "real", 4.0),
    ]
    gli = {r["challenge"]: r for r in EtDiscrimination().discriminate(rows)}["GLI"]
    assert gli["gen"]["n"] == 3  # the two failed rows are out of the denominator
    assert gli["excluded"] == {"input_fail": 1, "run_fail": 1}
    assert gli["hier_viol"] == 1


def test_rel_diff_keeps_generated_empty_at_minus_one_and_needs_a_real_denominator():
    assert EtDiscrimination.rel_diff(0.0, 4.0) == pytest.approx(-1.0)  # protocol §4 keep
    assert EtDiscrimination.rel_diff(2.0, 4.0) == pytest.approx(-0.5)
    assert EtDiscrimination.rel_diff(2.0, 0.0) is None  # unbounded, captured by pairing
    assert EtDiscrimination.rel_diff(0.0, 0.0) is None


def test_rel_diff_distribution_and_unpaired_tally():
    rows = [
        _row("SSA", "s1", "gen", 2.0),
        _row("SSA", "s1", "real", 4.0),  # -0.5
        _row("SSA", "s2", "gen", 0.0),
        _row("SSA", "s2", "real", 4.0),  # -1.0 (kept)
        _row("SSA", "s3", "gen", 4.0),
        _row("SSA", "s3", "real", 4.0),  # 0.0
        _row("SSA", "solo", "gen", 3.0),  # no real side: unpaired, no rel diff
    ]
    ssa = {r["challenge"]: r for r in EtDiscrimination().discriminate(rows)}["SSA"]
    assert ssa["pairing"]["unpaired"] == 1
    assert ssa["rel_diff"]["n_cases"] == 3
    assert ssa["rel_diff"]["median"] == pytest.approx(-0.5)
    assert ssa["rel_diff"]["ci90_low"] is not None and ssa["rel_diff"]["ci90_high"] is not None


def test_empty_challenge_yields_null_readings_without_raising():
    readings = EtDiscrimination().discriminate([])
    assert readings == []
    rows = [_row("PED", "x", "real", 1.0)]  # a challenge with only one side still reports
    ped = EtDiscrimination().discriminate(rows)
    assert ped[0]["gen"]["n"] == 0
    assert ped[0]["gen"]["rate"] is None
    assert ped[0]["gen"]["wilson_95_upper"] is None


# ------------------------------------------------------------------------ report & CLI


def _write_csv(rows, path):
    MeasurementTable.write([{field: row.get(field, "") for field in MEASUREMENT_FIELDS} for row in rows], path)
    return path


def test_cli_end_to_end_writes_json_and_markdown(tmp_path):
    rows = [_row("GLI", f"g{i}", "gen", 2.0) for i in range(3)] + [_row("GLI", f"g{i}", "real", 4.0) for i in range(3)]
    rows += [_row("SSA", "s", "gen", 0.0, pred_empty=1), _row("SSA", "s", "real", 1.0)]
    csv_path = _write_csv(rows, tmp_path / "measurements.csv")
    out = tmp_path / "diag"
    rc = main(["--measurements", str(csv_path), "--output-dir", str(out), "--run-id", "p1-20260822T131947Z"])
    assert rc == 0
    payload = json.loads((out / "et_discrimination_diagnostic.json").read_text())
    assert payload["schema"] == "et-discrimination-diagnostic/1"
    assert payload["variant"] == "diagnostic"
    assert payload["issue"] == 207
    assert payload["run_id"] == "p1-20260822T131947Z"
    assert "不产生任何验收判定" in payload["disclaimer"]
    assert set(payload["per_challenge"]) == {"GLI", "SSA"}
    assert len(payload["per_case"]) == len(rows)
    md = (out / "et_discrimination_diagnostic.md").read_text()
    assert "variant: diagnostic" in md
    assert "空 pred" in md
    assert "real_only" in md


def test_report_declares_no_verdict_anywhere(tmp_path):
    rows = [_row("GLI", "c", "gen", 0.0), _row("GLI", "c", "real", 1.0)]
    out = tmp_path / "diag"
    report = EtDiscriminationReport(Path("measurements.csv"), run_id=None)
    json_path, _md_path = report.write(EtDiscrimination().discriminate(rows), out)
    payload = json.loads(Path(json_path).read_text())
    assert "verdict" not in json.dumps(payload)


def test_holdout_quotas_sum_to_the_530_case_holdout():
    """The issue's '530 例 × 4 模态' denominator: quotas frozen in final_acceptance."""
    assert sum(HOLDOUT_QUOTAS[ch] for ch in ("GLI", "MEN", "METS", "PED", "SSA")) == 530


def test_diagnostic_seed_draws_the_registered_slot_through_the_allocator():
    """Job B's rel-diff CI draws slot 200 of each challenge band -- the
    pre-#232 module constants, now registry data (issue #232), byte-exact."""
    seed = DiagnosticSeedAllocator.seed("GLI", DIAGNOSTIC_SEED_SLOTS["et_rel_diff"])
    assert seed == DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET["GLI"] * 1000 + 200  # the legacy formula
    assert GLOBAL_SEED < DIAGNOSTIC_SEED_BASE
    assert DIAGNOSTIC_SEED_SLOTS["et_rel_diff"] == 200  # job A occupies slots 0/1 and 100/101 of each band


def test_missing_required_column_raises_a_diagnostic_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("challenge,case,side\nGLI,c,gen\n")
    with pytest.raises(DiagnosticError):
        main(["--measurements", str(path), "--output-dir", str(tmp_path / "out")])
