"""Observation-line yellow flag (issue #253, parent #247): the dev selection
point's ET/WT monitoring line, observed as pytest.

The adoption ruling #5 (``20260830-P1根因甄别-读数收编与整改方向决议.md`` §4)
de-blinds candidate selection: outside dev FID, the selection point runs the
job-B measurement vocabulary and evaluates a pre-recorded observation line --
METS ET detection rate < 0.9, or a per-challenge vol_et_rel median > 2, is a
YELLOW FLAG. The line is a selection surface, never an acceptance verdict:
every test here pins the pure function on synthetic readings, and the module's
import face stays clear of the judge chain (final_acceptance).
"""

import inspect
import json

import pytest

from ctmr.application.acceptance.distribution.challenge_registry import (
    CHALLENGE_SEED_OFFSET,
    DIAGNOSTIC_SEED_BASE,
    DIAGNOSTIC_SEED_SLOTS,
    GLOBAL_SEED,
)
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError, DiagnosticSeedAllocator
from ctmr.application.acceptance.distribution.et_discrimination import EtDiscrimination
from ctmr.application.acceptance.distribution.observation_line import ObservationLine


def _reading(challenge, gen_n, gen_k, rel_median):
    """One job-B-shaped per-challenge reading (the EtDiscriminate output contract,
    narrowed to the fields the line consumes)."""
    rate = gen_k / gen_n if gen_n else None
    return {
        "challenge": challenge,
        "gen": {"n": gen_n, "k_detected": gen_k, "rate": rate},
        "rel_diff": {"median": rel_median},
    }


def _holdout_replica_readings():
    """The recorded job-B holdout shape (20260829-诊断作业B-ET甄别.md §3):
    METS detection 38/48, GLI/PED/SSA volume overestimation."""
    return [
        _reading("GLI", 250, 250, 1.962),
        _reading("SSA", 12, 12, 15.018),
        _reading("MEN", 200, 200, 1.196),
        _reading("METS", 48, 38, -0.996),
        _reading("PED", 20, 20, 12.540),
    ]


# --------------------------------------------------------------------- the line


def test_mets_detection_below_floor_fires_the_yellow_flag():
    """The recorded holdout shape trips BOTH axes of the line: the METS rate
    rule plus the SSA/PED volume-overestimation medians (GLI +1.962 and
    MEN +1.196 stay under the ceiling)."""
    verdict = ObservationLine().evaluate(_holdout_replica_readings())
    mets = verdict["per_challenge"]["METS"]
    assert mets["mets_et_rate"] == {"rate": pytest.approx(38 / 48), "floor": 0.9, "fires": True}
    assert mets["flag"] is True
    assert verdict["flag"] is True
    assert len(verdict["fired"]) == 3
    assert "METS" in verdict["fired"][1] and "0.7917" in verdict["fired"][1]
    assert any(fired.startswith("SSA") for fired in verdict["fired"])
    assert any(fired.startswith("PED") for fired in verdict["fired"])


def test_volume_overestimation_fires_per_challenge_even_with_full_mets_detection():
    """The recorded overestimation axis (GLI/PED/SSA): the median rule is
    per-challenge, not METS-bound; MEN +1.196 and GLI +1.962 stay under 2."""
    readings = [
        _reading("GLI", 250, 250, 1.962),
        _reading("SSA", 12, 12, 15.018),
        _reading("MEN", 200, 200, 1.196),
        _reading("METS", 48, 48, -0.996),
        _reading("PED", 20, 20, 12.540),
    ]
    verdict = ObservationLine().evaluate(readings)
    assert verdict["per_challenge"]["METS"]["mets_et_rate"]["fires"] is False
    assert verdict["per_challenge"]["PED"]["vol_et_rel_median"]["fires"] is True
    assert verdict["per_challenge"]["SSA"]["flag"] is True
    assert verdict["per_challenge"]["GLI"]["flag"] is False  # +1.962 is below the ceiling
    assert verdict["per_challenge"]["MEN"]["flag"] is False
    assert verdict["flag"] is True
    assert len(verdict["fired"]) == 2  # SSA and PED


def test_clean_readings_leave_the_line_unflagged():
    readings = [
        _reading("GLI", 250, 250, 0.2),
        _reading("SSA", 12, 12, 0.5),
        _reading("MEN", 200, 200, 0.3),
        _reading("METS", 48, 48, 0.4),
        _reading("PED", 20, 20, 1.9),
    ]
    verdict = ObservationLine().evaluate(readings)
    assert verdict["flag"] is False
    assert verdict["fired"] == []


def test_thresholds_are_strict_inequalities():
    """rate exactly 0.9 and median exactly 2.0 sit ON the line: no fire."""
    readings = [
        _reading("METS", 10, 9, 2.0),
        _reading("PED", 10, 10, 2.0),
    ]
    verdict = ObservationLine().evaluate(readings)
    assert verdict["flag"] is False


def test_undefined_rel_median_is_recorded_not_evaluable_without_firing():
    """A challenge whose pairing produced no relative differences reports the
    None median honestly; the rule records it and does not fire."""
    readings = [
        _reading("METS", 48, 48, None),
        _reading("PED", 20, 20, None),
    ]
    verdict = ObservationLine().evaluate(readings)
    assert verdict["per_challenge"]["PED"]["vol_et_rel_median"] == {"median": None, "ceiling": 2.0, "fires": False}
    assert verdict["flag"] is False


def test_missing_mets_challenge_is_a_protocol_error():
    """The line's primary subject is the METS challenge: a monitoring run that
    produced no METS readings is a broken protocol, not a clean bill."""
    with pytest.raises(DiagnosticError):
        ObservationLine().evaluate([_reading("GLI", 250, 250, 0.2)])


def test_empty_mets_denominator_is_a_protocol_error():
    with pytest.raises(DiagnosticError):
        ObservationLine().evaluate([_reading("METS", 0, 0, None)])


# ------------------------------------------------- integration with job B's shape


def _row(challenge, case, side, vol_et, *, pred_empty=0):
    """A synthetic measurement row (typed like the instrument writes it)."""
    return {
        "obs_id": f"{case}__{side}",
        "challenge": challenge,
        "case": case,
        "side": side,
        "anchor": "",
        "input_fail": "0",
        "run_fail": "0",
        "hier_viol": "0",
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


def test_the_line_consumes_the_et_discrimination_output_verbatim():
    """End to end over the real producer: EtDiscrimination readings -> the flag,
    replicating the recorded holdout METS miss pattern (10/48 real_only)."""
    rows = []
    for index in range(48):
        gen_vol = 0.0 if index < 10 else 1.0  # 38/48 detected, the recorded pattern
        rows.append(_row("METS", f"m{index}", "gen", gen_vol, pred_empty=1 if index < 5 else 0))
        rows.append(_row("METS", f"m{index}", "real", 3.0))
    rows += [_row("GLI", f"g{index}", "gen", 2.0) for index in range(4)]
    rows += [_row("GLI", f"g{index}", "real", 1.0) for index in range(4)]  # rel median +1.0
    readings = EtDiscrimination(bootstrap_b=100).discriminate(rows)
    verdict = ObservationLine().evaluate(readings)
    assert verdict["per_challenge"]["METS"]["mets_et_rate"]["rate"] == pytest.approx(38 / 48)
    assert verdict["per_challenge"]["METS"]["flag"] is True
    assert verdict["per_challenge"]["GLI"]["flag"] is False
    assert verdict["flag"] is True


# ----------------------------------------------------- registration & discipline


def test_monitoring_bootstrap_seed_draws_the_new_registered_slot():
    """The monitoring job's own bootstrap draw (WT rel-diff CI90) takes the next
    free slot after jobs A/B and the C/D blocks (#247 seed discipline: the C/D
    bandless blocks occupy 300..320, so the monitoring slot starts at 400)."""
    assert DIAGNOSTIC_SEED_SLOTS["dev_monitor_wt_rel_diff"] == 400
    prior_occupancy = (0, 1, 100, 101, 200, *range(300, 321))  # jobs A/B/C/D + the geometry audit
    assert DIAGNOSTIC_SEED_SLOTS["dev_monitor_wt_rel_diff"] not in prior_occupancy
    seed = DiagnosticSeedAllocator.seed("GLI", DIAGNOSTIC_SEED_SLOTS["dev_monitor_wt_rel_diff"])
    assert seed == DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET["GLI"] * 1000 + 400
    assert GLOBAL_SEED < DIAGNOSTIC_SEED_BASE


def test_observation_modules_never_touch_the_acceptance_judgement_chain():
    """零验收判定链接触: the line and the monitor report import neither the
    final-acceptance judge nor any verdict surface -- source-level ratchet."""
    for module_name in (
        "ctmr.application.acceptance.distribution.observation_line",
        "ctmr.application.acceptance.distribution.dev_monitor",
    ):
        module = __import__(module_name, fromlist=["__doc__"])
        source = inspect.getsource(module)
        assert "final_acceptance" not in source, module_name
        assert "ctmr accept" not in source, module_name


def test_line_payload_is_json_serializable(tmp_path):
    verdict = ObservationLine().evaluate(_holdout_replica_readings())
    path = tmp_path / "flag.json"
    path.write_text(json.dumps(verdict))
    assert json.loads(path.read_text())["flag"] is True
