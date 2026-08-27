"""The L2 final-acceptance judge chain, observed as pytest (ADR-0002/0004 gate suite; issue #140).

The resident ``SelfTest`` of the production script became this file when the
judge chain moved into the distribution package (ADR-0015 §6: ``selftest``
subcommands die with the script move; assertion logic turns into real test
functions). Every assertion below is the pre-registered ADR-0004 protocol on a
synthetic fixture: frozen envelope literals (drift AND narrowing both reject),
the freeze-audit verdict pin, the five-key run binding, the per-phase assembly
rules and the full TOST / round-trip / undecided verdict chain. Subject ids are
synthetic non-patient strings; stdlib-only logic aside from the package import.

Torch-marked tier (ADR-0015 §6): runs for real in the CI torch tier.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from ctmr.application.acceptance.contract.artifacts import ArtifactFingerprinter, ManifestSides
from ctmr.application.acceptance.contract.binding import FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.distribution.final_acceptance import (
    CHALLENGE_SEED_OFFSET,
    CHALLENGES,
    FROZEN_ENVELOPES,
    GLOBAL_SEED,
    HOLDOUT_QUOTAS,
    MEASUREMENT_FIELDS,
    MODALITIES,
    REGIONS,
    AcceptanceError,
    AcceptanceReport,
    AssemblyPlanner,
    ChallengeJudge,
    ClusterBootstrap,
    FailureGate,
    FreezeGuard,
    FrozenEnvelopes,
    MeasurementTable,
    P1PseudoQuadPlan,
    P2SharedMaskPlan,
    P3FourAnchorPlan,
    QuantityRegistry,
    RealReferenceResolver,
)

pytestmark = pytest.mark.torch

BOOTSTRAP_B = 400


def _frozen_envelopes_match_the_published_literals():
    """Every pass line reads its numbers from the published ADR-0002 table."""
    for challenge in CHALLENGES:
        for region in REGIONS:
            expected = FROZEN_ENVELOPES[challenge][region]
            assert len(expected) == 3
            d_r_low, e_r_vol, e_r_centroid = expected
            if math.isnan(d_r_low) or math.isnan(e_r_vol) or math.isnan(e_r_centroid):
                raise AssertionError(f"{challenge}/{region}: frozen literal is NaN")


# ── frozen-envelope verification gate ───────────────────────────────────


def test_envelope_verification_accepts_exact_literals_and_rejects_drift_or_loss(tmp_path):
    _frozen_envelopes_match_the_published_literals()
    envelopes = FrozenEnvelopes()
    summary_dir = tmp_path / "calibration_summaries"
    summary_dir.mkdir(parents=True)
    for challenge in CHALLENGES:
        summary = {
            "per_region": {
                region: {
                    "D_r_low": envelopes.d_r_low(challenge, region),
                    "E_r_vol": envelopes.e_r_vol(challenge, region),
                    "E_r_centroid": envelopes.e_r_centroid(challenge, region),
                }
                for region in REGIONS
            }
        }
        (summary_dir / f"summary_{challenge}.json").write_text(json.dumps(summary))
    assert envelopes.verify_against_summary(summary_dir) is True  # exact literals pass

    drifted = json.loads((summary_dir / "summary_GLI.json").read_text())
    drifted["per_region"]["WT"]["E_r_vol"] = 0.2000  # narrowed / drifted margin
    (summary_dir / "summary_GLI.json").write_text(json.dumps(drifted))
    with pytest.raises(AcceptanceError, match="envelope drift"):
        envelopes.verify_against_summary(summary_dir)

    (summary_dir / "summary_GLI.json").unlink()
    with pytest.raises(AcceptanceError, match="missing"):
        envelopes.verify_against_summary(summary_dir)


def test_mets_floors_stay_the_wide_zero_literal():
    """METS keeps its wide envelope: floors stay zero and are carried, never narrowed."""
    envelopes = FrozenEnvelopes()
    for region in REGIONS:
        assert envelopes.d_r_low("METS", region) == 0.0


# ── freeze-audit verdict pin ────────────────────────────────────────────


def _write_verdict(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def test_freeze_guard_accepts_only_true_verdicts_with_the_pinned_hash(tmp_path):
    guard = FreezeGuard(ArtifactFingerprinter())
    good = tmp_path / "freeze_audit_good.json"
    pinned = _write_verdict(good, {"all_passed": True, "challenges": []})
    record = guard.verify(good, expect_sha256=pinned)  # pinned hash passes
    assert record["pinned"] is True

    failed_payload = json.dumps({"all_passed": False}) + "\n"
    failed = tmp_path / "freeze_audit_failed.json"
    failed.write_text(failed_payload)
    with pytest.raises(AcceptanceError, match="all_passed is not true"):
        guard.verify(failed, expect_sha256=hashlib.sha256(failed_payload.encode()).hexdigest())

    wrong_hash = tmp_path / "freeze_audit_bad_hash.json"
    _write_verdict(wrong_hash, {"all_passed": True, "challenges": []})
    with pytest.raises(AcceptanceError, match="sha256 .* != pinned"):
        guard.verify(wrong_hash, expect_sha256="0" * 64)


def test_freeze_guard_unpinned_mode_reports_its_own_hash(tmp_path):
    guard = FreezeGuard(ArtifactFingerprinter())
    good = tmp_path / "freeze_audit_good.json"
    _write_verdict(good, {"all_passed": True, "challenges": []})
    record = guard.verify(good, expect_sha256=None)  # fresh re-run: hash recorded, not pinned
    assert record["pinned"] is False


def test_freeze_guard_refuses_a_missing_verdict_file(tmp_path):
    guard = FreezeGuard(ArtifactFingerprinter())
    with pytest.raises(AcceptanceError, match="not found"):
        guard.verify(tmp_path / "absent.json", expect_sha256=None)


# ── five-key run binding (issue #58 attachment gate) ────────────────────


_RUN_RECORD = {
    "schema": "brats-phase-run/1",
    "run_id": "p1-bindtest",
    "phase": "P1",
    "status": "frozen",
    "manifest": {"path": "/private/m.json", "sha256": "a" * 64},
    "selection": {"checkpoint": {"path": "/private/ckpt.pt", "sha256": "b" * 64, "epoch": 7}},
    "samples": {"path": "/private/samples.json", "sha256": "c" * 64},
}


def _bind(path):
    """Shared five-key binding with the frozen gate; translated to the L2 error surface."""
    try:
        return FrozenRunBinding.from_path(path)
    except FrozenRunBindingError as error:
        raise AcceptanceError(str(error)) from error


def test_run_binding_extract_exactly_the_five_keys(tmp_path):
    path = tmp_path / "run_bindtest.json"
    path.write_text(json.dumps(_RUN_RECORD))
    binding = _bind(path).as_dict()
    assert binding == {
        "run_id": "p1-bindtest",
        "phase": "P1",
        "manifest_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "samples_sha256": "c" * 64,
    }


def test_run_binding_rejects_open_runs_and_missing_records(tmp_path):
    open_path = tmp_path / "run_open.json"
    open_path.write_text(json.dumps(dict(_RUN_RECORD, status="open")))
    with pytest.raises(AcceptanceError):
        _bind(open_path)
    with pytest.raises(AcceptanceError):
        _bind(tmp_path / "run_absent.json")


# ── assembly planning fixtures ──────────────────────────────────────────


def _fixture_entry(challenge, case, phase):
    real_paths = {m: f"/private/real/{challenge}/{case}/{case}-{m}.nii.gz" for m in MODALITIES}
    if phase == "P1":
        return {
            "case_id": case,
            "challenge": challenge,
            "phase": "P1",
            "samples": {m: {"path": f"/private/gen/{case}-{m}.nii.gz", "seed": 100 + i} for i, m in enumerate(MODALITIES)},
            "real_paths": real_paths,
        }
    if phase == "P2":
        return {
            "case_id": case,
            "challenge": challenge,
            "phase": "P2",
            "condition_mask": f"/private/cond/{case}-cond.nii.gz",
            "samples": {m: {"path": f"/private/gen/{case}-{m}.nii.gz"} for m in MODALITIES},
            "real_paths": real_paths,
        }
    return {
        "case_id": case,
        "challenge": challenge,
        "phase": "P3",
        "anchors": {
            m: {
                "real": f"/private/real/{challenge}/{case}/{case}-{m}.nii.gz",
                "generated": {t: {"path": f"/private/gen/{case}-{t}-from-{m}.nii.gz"} for t in MODALITIES if t != m},
            }
            for m in MODALITIES
        },
    }


def _holdout_manifest(workdir):
    manifest = {"split_id": "spec-test", "challenges": {}}
    for challenge in CHALLENGES:
        manifest["challenges"][challenge] = {
            "cases": {
                "train": [f"FIX{challenge}-0000-{i:03d}" for i in range(2)],
                "dev": [f"FIX{challenge}-0100-{i:03d}" for i in range(2)],
                "holdout": [f"FIX{challenge}-0200-{i:03d}" for i in range(HOLDOUT_QUOTAS[challenge])],
            }
        }
    path = workdir / "holdout_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path, manifest


def _holdout_case(manifest, challenge):
    return manifest["challenges"][challenge]["cases"]["holdout"][0]


def _planner(phase, manifest_path, workdir):
    strategies = {"P1": P1PseudoQuadPlan(), "P2": P2SharedMaskPlan(), "P3": P3FourAnchorPlan()}
    return AssemblyPlanner(
        phase,
        strategies[phase],
        RealReferenceResolver(workdir / "real"),
        ManifestSides(json.loads(Path(manifest_path).read_text())),
        ArtifactFingerprinter(),
    )


def _write_samples(workdir, entries, name):
    path = workdir / name
    path.write_text(json.dumps(entries, indent=2) + "\n")
    return path


def test_p1_plan_builds_two_sided_observations_and_flags_provisional_quota(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    case = _holdout_case(manifest, "GLI")
    entry = _fixture_entry("GLI", case, "P1")
    plan = _planner("P1", manifest_path, tmp_path).build(_write_samples(tmp_path, [entry], "p1.json"), manifest_path, "p1-run", tmp_path / "real")
    obs_ids = sorted(obs["obs_id"] for obs in plan["observations"])
    assert obs_ids == sorted([f"{case}__real", f"{case}__gen"])
    assert plan["challenges"]["GLI"]["provisional"] is True  # one case vs quota 250
    assert set(plan["observations"][0]["channels"]) == {"0000", "0001", "0002", "0003"}


def test_p1_identical_noise_seeds_reject_independence(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    entry = dict(_fixture_entry("GLI", _holdout_case(manifest, "GLI"), "P1"))
    entry["samples"] = {m: {"path": entry["samples"][m]["path"], "seed": 7} for m in MODALITIES}
    planner = _planner("P1", manifest_path, tmp_path)
    samples = _write_samples(tmp_path, [entry], "bad_seed.json")
    with pytest.raises(AcceptanceError, match="four distinct noise seeds"):
        planner.build(samples, manifest_path, "p1-bad", tmp_path / "real")


def test_dev_side_case_never_enters_final_acceptance(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    dev_entry = _fixture_entry("GLI", manifest["challenges"]["GLI"]["cases"]["dev"][0], "P1")
    planner = _planner("P1", manifest_path, tmp_path)
    samples = _write_samples(tmp_path, [dev_entry], "dev.json")
    with pytest.raises(AcceptanceError, match="holdout side only"):
        planner.build(samples, manifest_path, "p1-dev", tmp_path / "real")


def test_phase_mismatched_sample_entry_rejects(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    wrong_phase = _fixture_entry("GLI", _holdout_case(manifest, "GLI"), "P2")  # assembled as P1
    planner = _planner("P1", manifest_path, tmp_path)
    samples = _write_samples(tmp_path, [wrong_phase], "wrong.json")
    with pytest.raises(AcceptanceError, match="phase"):
        planner.build(samples, manifest_path, "p1-wrong", tmp_path / "real")


def test_p2_without_condition_mask_rejects(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    entry = _fixture_entry("MEN", _holdout_case(manifest, "MEN"), "P2")
    planner = _planner("P2", manifest_path, tmp_path)
    maskless = {k: v for k, v in entry.items() if k != "condition_mask"}
    samples = _write_samples(tmp_path, [maskless], "maskless.json")
    with pytest.raises(AcceptanceError, match="condition_mask"):
        planner.build(samples, manifest_path, "p2-bad", tmp_path / "real")


def test_p3_requires_all_four_anchor_rounds_with_unique_obs_ids(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    case = _holdout_case(manifest, "SSA")
    entry = _fixture_entry("SSA", case, "P3")
    plan = _planner("P3", manifest_path, tmp_path).build(_write_samples(tmp_path, [entry], "p3.json"), manifest_path, "p3-run", tmp_path / "real")
    gen_obs = [obs for obs in plan["observations"] if obs["side"] == "gen"]
    assert len(gen_obs) == 4 and len({obs["anchor"] for obs in gen_obs}) == 4
    assert len({obs["obs_id"] for obs in plan["observations"]}) == 5  # four rounds + the shared real row
    broken = {m: entry["anchors"][m] for m in ("t1n", "t1c", "t2w")}
    broken_entry = {**entry, "anchors": broken}
    planner = _planner("P3", manifest_path, tmp_path)
    samples = _write_samples(tmp_path, [broken_entry], "broken.json")
    with pytest.raises(AcceptanceError, match="anchors must carry exactly"):
        planner.build(samples, manifest_path, "p3-bad", tmp_path / "real")


def test_duplicate_obs_ids_are_rejected_before_any_overwrite(tmp_path):
    manifest_path, manifest = _holdout_manifest(tmp_path)
    case = _holdout_case(manifest, "METS")
    entry = _fixture_entry("METS", case, "P1")
    entries = [entry, entry]  # same case twice -> duplicate __gen/__real ids
    planner = _planner("P1", manifest_path, tmp_path)
    samples = _write_samples(tmp_path, entries, "dupe.json")
    with pytest.raises(AcceptanceError, match="duplicate obs_id"):
        planner.build(samples, manifest_path, "mets-dupe", tmp_path / "real")


# ── verdict chain (TOST + round trip + undecided gate) ──────────────────


def _measurement_row(obs_id, challenge, case, side, anchor=None, **overrides):
    row = {field: "" for field in MEASUREMENT_FIELDS}
    row.update(
        obs_id=obs_id,
        challenge=challenge,
        case=case,
        side=side,
        anchor=anchor or "",
        input_fail="0",
        run_fail="0",
        hier_viol="0",
        pred_empty="0",
        vol_wt_ml="50.0",
        vol_tc_ml="30.0",
        vol_et_ml="10.0",
        brain_ml="1200.0",
        wt_brain="0.0417",
        et_wt="0.20",
        cx_wt_mm="120.0",
        cy_wt_mm="120.0",
        cz_wt_mm="77.0",
        cx_tc_mm="121.0",
        cy_tc_mm="121.0",
        cz_tc_mm="78.0",
        cx_et_mm="122.0",
        cy_et_mm="122.0",
        cz_et_mm="79.0",
        cond_dice_wt="0.95",
        cond_dice_tc="0.93",
        cond_dice_et="0.90",
    )
    row.update(overrides)
    return row


def _challenge_rows(challenge, cases, phase, mutate=None):
    rows = []
    for index, case in enumerate(cases):
        real = _measurement_row(f"{case}__real", challenge, case, "real")
        gen_anchors = [None] if phase != "P3" else list(MODALITIES)
        for anchor in gen_anchors:
            suffix = "" if anchor is None else f"__a{anchor}"
            gen = _measurement_row(f"{case}__gen{suffix}", challenge, case, "gen", anchor)
            if mutate is not None:
                mutate(index, case, anchor, real, gen)
            rows.append(gen)
        rows.append(real)
    return rows


_ENVELOPES = FrozenEnvelopes()


def _judge(phase, challenge, rows, b=BOOTSTRAP_B):
    return ChallengeJudge(_ENVELOPES, ClusterBootstrap(b), phase).judge(rows, challenge, GLOBAL_SEED + CHALLENGE_SEED_OFFSET[challenge])


def test_quantity_registry_is_the_pre_registered_list():
    registry = QuantityRegistry().all()
    names = [q.name for q in registry]
    # 3 volumes rel-diff + 9 centroid axes + wt_brain_rel + et_wt_rel (ADR-0004 decision 1)
    assert names.count("vol_wt_rel") == 1 and names.count("vol_tc_rel") == 1 and names.count("vol_et_rel") == 1
    assert sum(name.startswith("centroid_") for name in names) == 9
    assert "wt_brain_rel" in names and "et_wt_rel" in names


def test_equivalent_volumes_pass_the_gli_tost():
    rows = _challenge_rows(
        "GLI",
        [f"FIXGLI-0200-{i:03d}" for i in range(6)],
        "P1",
        mutate=lambda i, c, a, r, g: g.update(vol_wt_ml="51.0", vol_tc_ml="30.5", vol_et_ml="10.2"),
    )
    verdict = _judge("P1", "GLI", rows)
    assert verdict["verdict"] == "pass", [q for q in verdict["tost"] if not q["passed"]]


def test_single_hierarchy_violation_forces_undecided_and_blocks():
    def break_one(index, case, anchor, real, gen):
        if index == 2 and anchor is None:
            gen.update(hier_viol="1")

    rows = _challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1", mutate=break_one)
    verdict = _judge("P1", "GLI", rows)
    assert verdict["verdict"] == "undecided"
    assert verdict["failure_audit"]["n_failed"] == 1


def test_real_side_input_failure_is_equally_undecided():
    rows = _challenge_rows(
        "GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1", mutate=lambda i, c, a, r, g: r.update(input_fail="1") if i == 0 else None
    )
    assert _judge("P1", "GLI", rows)["verdict"] == "undecided"


def test_volume_bias_outside_the_margin_fails_inside_carries():
    bias = lambda i, c, a, r, g: g.update(vol_wt_ml="80.0")  # noqa: E731  (60% WT volume shift)
    gli = _judge("P1", "GLI", _challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P1", mutate=bias))
    assert gli["verdict"] == "fail"
    assert not all(q["passed"] for q in gli["tost"])
    # METS keeps its wide envelope: the same 60% bias sits inside +-1.651 (resolving-power limit).
    mets = _judge("P1", "METS", _challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P1", mutate=bias))
    vol_wt = next(q for q in mets["tost"] if q["quantity"] == "vol_wt_rel")
    assert vol_wt["passed"]
    assert mets["verdict"] == "pass"


def test_centroid_shift_beyond_e_r_centroid_fails_gli():
    rows = _challenge_rows(
        "GLI",
        [f"FIXGLI-0200-{i:03d}" for i in range(6)],
        "P1",
        mutate=lambda i, c, a, r, g: g.update(cx_wt_mm="130.0", cx_tc_mm="131.0", cx_et_mm="132.0"),
    )
    assert _judge("P1", "GLI", rows)["verdict"] == "fail"


def test_zero_real_et_excludes_per_quantity_without_leakage_into_siblings():
    rows = _challenge_rows(
        "GLI",
        [f"FIXGLI-0200-{i:03d}" for i in range(6)],
        "P1",
        mutate=lambda i, c, a, r, g: r.update(vol_et_ml="0.0", et_wt="") if i < 3 else None,
    )
    verdict = _judge("P1", "GLI", rows)
    et_vol = next(q for q in verdict["tost"] if q["quantity"] == "vol_et_rel")
    et_wt = next(q for q in verdict["tost"] if q["quantity"] == "et_wt_rel")
    assert et_vol["n_excluded"] == 3 and et_wt["n_excluded"] == 3
    # Late-binding guard: zero real-side ET must NOT leak into WT/TC quantities.
    wt_vol = next(q for q in verdict["tost"] if q["quantity"] == "vol_wt_rel")
    tc_vol = next(q for q in verdict["tost"] if q["quantity"] == "vol_tc_rel")
    cent_wt = next(q for q in verdict["tost"] if q["quantity"] == "centroid_wt_x")
    assert (wt_vol["n_excluded"], tc_vol["n_excluded"], cent_wt["n_excluded"]) == (0, 0, 0)
    assert et_wt["margin"] == _ENVELOPES.e_r_vol("GLI", "ET") + _ENVELOPES.e_r_vol("GLI", "WT")


def test_generated_side_empty_prediction_stays_in_distribution_at_minus_one():
    rows = _challenge_rows(
        "METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P1", mutate=lambda i, c, a, r, g: g.update(vol_wt_ml="0.0") if i < 3 else None
    )
    vol_wt = next(q for q in _judge("P1", "METS", rows)["tost"] if q["quantity"] == "vol_wt_rel")
    assert vol_wt["n_excluded"] == 0


def test_p3_cluster_bootstrap_resamples_cases_not_observations():
    rows = _challenge_rows("SSA", [f"FIXSSA-0200-{i:03d}" for i in range(5)], "P3", mutate=lambda i, c, a, r, g: g.update(vol_wt_ml="52.0"))
    verdict = _judge("P3", "SSA", rows)
    assert verdict["verdict"] == "pass"
    vol_wt = next(q for q in verdict["tost"] if q["quantity"] == "vol_wt_rel")
    assert vol_wt["n_cases"] == 5


def test_p2_round_trip_dice_floor_pass_fail_and_vacuous_zero_floor():
    dice_ok = _judge("P2", "GLI", _challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P2"))
    rt = {item["region"]: item for item in dice_ok["round_trip"]}
    assert dice_ok["verdict"] == "pass" and all(item["passed"] for item in rt.values())

    collapsed = _judge(
        "P2",
        "GLI",
        _challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P2", mutate=lambda i, c, a, r, g: g.update(cond_dice_et="0.10")),
    )
    assert collapsed["verdict"] == "fail"

    metrows = _judge(
        "P2",
        "METS",
        _challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(6)], "P2", mutate=lambda i, c, a, r, g: g.update(cond_dice_wt="0.0")),
    )
    met_rt = {item["region"]: item for item in metrows["round_trip"]}
    assert all(item["vacuous_pass"] and item["passed"] for item in met_rt.values())


def test_report_is_aggregate_and_leaks_no_case_ids_even_when_undecided():
    def fail_one(index, case, anchor, real, gen):
        if index == 1:
            gen.update(run_fail="1")

    passing = _challenge_rows("GLI", [f"FIXGLI-0200-{i:03d}" for i in range(6)], "P2")
    failing = _challenge_rows("METS", [f"FIXMETS-0200-{i:03d}" for i in range(4)], "P2", mutate=fail_one)
    failing_verdict = ChallengeJudge(_ENVELOPES, ClusterBootstrap(BOOTSTRAP_B), "P2").judge(
        failing, "METS", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["METS"]
    )
    assert failing_verdict["verdict"] == "undecided"  # carries failures for the leak check below
    verdict_gliv2 = ChallengeJudge(_ENVELOPES, ClusterBootstrap(BOOTSTRAP_B), "P2").judge(passing, "GLI", GLOBAL_SEED + CHALLENGE_SEED_OFFSET["GLI"])

    report = AcceptanceReport(
        "P2",
        "report-leak-check",
        BOOTSTRAP_B,
        {"path": "/private/freeze-audit.json", "sha256": "0" * 64, "pinned": True},
        provisional_challenges=["METS"],
    ).build([verdict_gliv2, failing_verdict], [])
    blob = json.dumps(report) + "\n".join(
        AcceptanceReport("P2", "report-leak-check", BOOTSTRAP_B, report["frozen_audit"], ["METS"])._markdown(report)
    )
    for challenge in CHALLENGES:
        assert f"FIX{challenge}" not in blob, f"report leaks case ids for {challenge}"
    assert report["overall_verdict"] in ("pass", "fail", "undecided")
    assert "METS" in report["provisional_challenges"]


def test_failure_gate_counts_by_side_and_carries_a_diagnostic_wilson_bound():
    audit = FailureGate.audit(
        [
            {"side": "gen", "input_fail": "1", "run_fail": "", "hier_viol": ""},
            {"side": "real", "input_fail": "", "run_fail": "1", "hier_viol": ""},
            {"side": "gen", "input_fail": "", "run_fail": "", "hier_viol": ""},
        ]
    )
    assert audit["n_failed"] == 2
    assert audit["breakdown"] == {"input_fail": 1, "run_fail": 1, "hier_viol": 0}
    assert audit["n_failed_by_side"] == {"gen": 1, "real": 1}
    assert 0 < audit["wilson_95_upper"] <= 1.0


def test_measurement_table_roundtrip_preserves_the_column_contract(tmp_path):
    rows = [_measurement_row("FIXTC-0000-000__gen", "GLI", "FIXTC-0000-000", "gen")]
    path = MeasurementTable.write(rows, tmp_path / "table.csv")
    parsed = MeasurementTable.read(path)
    assert list(parsed[0].keys()) == MEASUREMENT_FIELDS
    missing = {field: "" for field in MEASUREMENT_FIELDS}
    bad = tmp_path / "bad.csv"
    bad.write_text(",".join(k for k in missing if k != "vol_et_ml") + "\n")
    with pytest.raises(AcceptanceError, match="missing columns"):
        MeasurementTable.read(bad)
