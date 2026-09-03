"""Dev selection-point ET/WT monitor (issue #253, parent #247): the diagnostic
reading face, observed as pytest.

The monitor reuses the job-B discrimination verbatim (the MEASUREMENT_FIELDS
contract over the frozen instrument's per-observation readings), adds the WT
volume addendum (the overestimation axis job B measured on the holdout), and
evaluates the pre-recorded observation line -- a selection surface that never
produces an acceptance verdict. Every test runs on synthetic CSV rows with
hand-computed expectations.
"""

import json
from pathlib import Path

import pytest

from ctmr.application.acceptance.distribution.dev_monitor import DevMonitorReport, WtMonitor, main
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError
from ctmr.application.acceptance.distribution.measurement_table import MEASUREMENT_FIELDS, MeasurementTable


def _row(challenge, case, side, vol_et, vol_wt="10.0", *, pred_empty=0, run_fail=0, input_fail=0):
    return {
        "obs_id": f"{case}__{side}",
        "challenge": challenge,
        "case": case,
        "side": side,
        "anchor": "",
        "input_fail": str(input_fail),
        "run_fail": str(run_fail),
        "hier_viol": "0",
        "pred_empty": str(pred_empty),
        "vol_wt_ml": vol_wt,
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


def _sample_plan(challenges, quota=4):
    return {
        "schema": "l2-final-acceptance-plan/1",
        "phase": "P1",
        "population": "dev",
        "challenges": {ch: {"n_cases": n, "quota": quota, "provisional": n < quota} for ch, n in challenges.items()},
    }


def _write_csv(rows, path):
    MeasurementTable.write([{field: row.get(field, "") for field in MEASUREMENT_FIELDS} for row in rows], path)
    return path


def _monitor_rows():
    """METS misses on 2/4 (yellow-flag shape) + a clean GLI; WT double on gen."""
    rows = []
    for index in range(4):
        rows.append(_row("METS", f"m{index}", "gen", 0.0 if index < 2 else 1.0, vol_wt="20.0", pred_empty=1 if index == 0 else 0))
        rows.append(_row("METS", f"m{index}", "real", 3.0, vol_wt="10.0"))
    for index in range(4):
        rows.append(_row("GLI", f"g{index}", "gen", 2.0, vol_wt="20.0"))
        rows.append(_row("GLI", f"g{index}", "real", 2.0, vol_wt="10.0"))
    return rows


def _report(csv_path, plan_path=None, run_id="p1-20260822T131947Z", bootstrap_b=100):
    return DevMonitorReport(Path(csv_path), sample_plan=plan_path, run_id=run_id, bootstrap_b=bootstrap_b)


def test_report_end_to_end_writes_json_and_markdown_with_the_flag(tmp_path):
    csv_path = _write_csv(_monitor_rows(), tmp_path / "measurements_dev.csv")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_sample_plan({"METS": 4, "GLI": 4})))
    out = tmp_path / "report"
    json_path, md_path = _report(csv_path, plan_path).write(out)
    assert json_path.name == "dev_monitor_diagnostic.json" and md_path.name == "dev_monitor_diagnostic.md"
    payload = json.loads(Path(json_path).read_text())
    assert payload["schema"] == "dev-etwt-monitor-diagnostic/1"
    assert payload["variant"] == "diagnostic"
    assert payload["issue"] == 253
    assert payload["run_id"] == "p1-20260822T131947Z"
    assert "不产生任何验收判定" in payload["disclaimer"]
    # the observation line: METS 2/4 = 0.5 < 0.9 -> yellow flag
    assert payload["observation_line"]["flag"] is True
    assert any("METS" in fired for fired in payload["observation_line"]["fired"])
    assert set(payload["per_challenge"]) == {"GLI", "METS"}
    # the job-B ET reading vocabulary rides verbatim
    assert payload["per_challenge"]["METS"]["gen"]["k_detected"] == 2
    assert payload["per_challenge"]["METS"]["empty_pred"]["gen"] == {"k": 1, "n": 4}
    # the WT addendum: gen median 20 vs real median 10 -> rel median +1.0
    assert payload["per_challenge"]["METS"]["wt"]["gen"]["vol_ml"]["median"] == pytest.approx(20.0)
    assert payload["per_challenge"]["METS"]["wt"]["real"]["vol_ml"]["median"] == pytest.approx(10.0)
    assert payload["per_challenge"]["METS"]["wt"]["rel_diff"]["median"] == pytest.approx(1.0)
    assert payload["per_challenge"]["METS"]["wt"]["rel_diff"]["n_cases"] == 4
    # the sample protocol provenance
    assert payload["sample"]["population"] == "dev"
    assert payload["sample"]["challenges"]["METS"]["n_cases"] == 4
    # per-case detail survives for the T8 comparison
    assert len(payload["per_case"]) == 16
    md = Path(md_path).read_text()
    assert "variant: diagnostic" in md
    assert "黄旗" in md
    assert "ET/WT" in md
    assert "选择面" in md


def test_report_declares_no_verdict_anywhere(tmp_path):
    csv_path = _write_csv(_monitor_rows(), tmp_path / "m.csv")
    json_path, _md = _report(csv_path, run_id=None).write(tmp_path / "report")
    payload = json.loads(Path(json_path).read_text())
    assert "verdict" not in json.dumps(payload)
    assert payload["observation_line"]["flag"] is True  # the flag is the only judgement-shaped output


def test_nonpositive_real_wt_excludes_the_pair_and_counts_it():
    """RelativeDifference.of is None for a non-positive real denominator (the
    callers own what undefined means) -- the WT addendum must keep such pairs
    out of the rel statistics and count them visibly, not crash sorting Nones
    (the 2026-09-03 baseline run's live TypeError on real dev volumes)."""
    rows = []
    for index in range(4):
        rows.append(_row("GLI", f"g{index}", "gen", 2.0, vol_wt="20.0"))
        rows.append(_row("GLI", f"g{index}", "real", 2.0, vol_wt="0.0" if index == 0 else "10.0"))
    reading = {item["challenge"]: item for item in WtMonitor(bootstrap_b=100).readings(rows)}["GLI"]
    wt_rel = reading["rel_diff"]
    assert wt_rel["n_cases"] == 3  # the (20-10)/10 pairs only
    assert wt_rel["n_undefined"] == 1  # the zero-WT real denominator, visible not silent
    assert wt_rel["median"] == pytest.approx(1.0)


def test_missing_wt_value_on_either_side_counts_the_pair_undefined():
    """A measured row whose vol_wt_ml is absent leaves the pair without a
    difference -- counted in n_undefined, never dropped silently (the
    plan-less CLI path has no completeness gate behind it)."""
    rows = []
    for index in range(4):
        # g0: real value missing; g1: gen value missing; g2/g3: defined pairs
        rows.append(_row("GLI", f"g{index}", "gen", 2.0, vol_wt="" if index == 1 else "20.0"))
        rows.append(_row("GLI", f"g{index}", "real", 2.0, vol_wt="" if index == 0 else "10.0"))
    reading = {item["challenge"]: item for item in WtMonitor(bootstrap_b=100).readings(rows)}["GLI"]
    wt_rel = reading["rel_diff"]
    assert wt_rel["n_cases"] == 2
    assert wt_rel["n_undefined"] == 2
    assert wt_rel["median"] == pytest.approx(1.0)


def test_cli_end_to_end(tmp_path):
    csv_path = _write_csv(_monitor_rows(), tmp_path / "m.csv")
    rc = main(["--measurements", str(csv_path), "--output-dir", str(tmp_path / "out"), "--bootstrap-b", "100"])
    assert rc == 0
    payload = json.loads((tmp_path / "out" / "dev_monitor_diagnostic.json").read_text())
    assert payload["sample"] is None  # no --sample-plan: provenance block stays None, run proceeds
    assert payload["observation_line"]["flag"] is True


def test_missing_mets_challenge_raises_a_protocol_error(tmp_path):
    rows = [_row("GLI", "g", "gen", 2.0), _row("GLI", "g", "real", 2.0)]
    csv_path = _write_csv(rows, tmp_path / "m.csv")
    with pytest.raises(DiagnosticError):
        main(["--measurements", str(csv_path), "--output-dir", str(tmp_path / "out")])


def test_unusable_table_raises_a_diagnostic_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("challenge,case,side\nMETS,c,gen\n")
    with pytest.raises(DiagnosticError):
        main(["--measurements", str(path), "--output-dir", str(tmp_path / "out")])


# ----------------------------------------------------- completeness gate (#253)


def test_partial_measurement_denominator_is_refused_before_the_flag(tmp_path):
    """The Codex P2 shape: 1 valid detected METS + 3 run_fail rows -> a 1-case
    denominator reading rate 1.0 (unflagged) even though the pinned 24-case
    observation line was not measured. The report must refuse to evaluate the
    flag on a partial measurement -- no report, no invented clean bill."""
    rows = [_row("METS", "m0", "gen", 1.0)]  # the single valid, detected case
    rows += [_row("METS", f"m{index}", "gen", 1.0, run_fail=1) for index in (1, 2, 3)]
    rows += [_row("METS", f"m{index}", "real", 3.0) for index in range(4)]
    csv_path = _write_csv(rows, tmp_path / "m.csv")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_sample_plan({"METS": 4})))
    with pytest.raises(DiagnosticError):
        _report(csv_path, plan_path).write(tmp_path / "report")
    assert not (tmp_path / "report" / "dev_monitor_diagnostic.json").exists()


def test_real_side_shortfall_is_also_refused(tmp_path):
    """The gate holds both denominators to the plan: a real-side input_fail is
    not a measured case either."""
    rows = [_row("METS", f"m{index}", "gen", 1.0) for index in range(4)]
    rows += [_row("METS", f"m{index}", "real", 3.0) for index in range(3)]
    rows.append(_row("METS", "m3", "real", 3.0, input_fail=1))
    csv_path = _write_csv(rows, tmp_path / "m.csv")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_sample_plan({"METS": 4})))
    with pytest.raises(DiagnosticError):
        _report(csv_path, plan_path).write(tmp_path / "report")


def test_planned_challenge_with_no_rows_is_refused(tmp_path):
    rows = [_row("GLI", "g", "gen", 2.0), _row("GLI", "g", "real", 2.0)]
    csv_path = _write_csv(rows, tmp_path / "m.csv")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_sample_plan({"GLI": 1, "METS": 4})))
    with pytest.raises(DiagnosticError):
        _report(csv_path, plan_path).write(tmp_path / "report")


def test_complete_measurement_against_the_plan_still_evaluates(tmp_path):
    """The gate is not a regression on the healthy path: a full measurement
    matching the pinned quotas evaluates the flag as before."""
    csv_path = _write_csv(_monitor_rows(), tmp_path / "measurements_dev.csv")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_sample_plan({"METS": 4, "GLI": 4})))
    json_path, _md = _report(csv_path, plan_path).write(tmp_path / "report")
    payload = json.loads(Path(json_path).read_text())
    assert payload["observation_line"]["flag"] is True  # METS 2/4 < 0.9 still fires
