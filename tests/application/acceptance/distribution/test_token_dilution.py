"""Diagnostic job D (issue #209, parent #205): same-seed token-swap bright-core
discrimination, observed as pytest.

The L3 conditional-dilution axis (RC-2, parent #205): the modality-label
perturber (PINNED_PROB=0.1, per-element Bernoulli) spends ~19% of the t1c
training steps teaching the model that token 34 carries no specific semantics
(→8 pan-MR, →0 unknown). Job D isolates that axis from everything else: the
frozen sampling rule keeps its per-case seed and every recipe knob, ONLY the
condition token is swapped (t1n 29 / t1c 34 / t2w 30 / t2f 31 plus the pan-MR
control 8 the augmentation itself perturbs into), so within one case the
initial noise is bit-identical across arms and every output difference
attributes to the token condition alone.

Every test here runs on synthetic int16 volumes with hand-computed
expectations: bright-core top statistics on the nonzero-voxel basis, the
linear q*(n-1) quantile rule shared with the calibration side, the gain-share
bands (2/3–1/3) following job A's attribution precedent, and the repository's
seed-anchor invariants (cross-arm consistency + the frozen-rule anchor).
"""

import hashlib
import json

import numpy as np
import pytest
import SimpleITK as sitk

from ctmr.application.acceptance.distribution.challenge_registry import (
    DIAGNOSTIC_SEED_BASE as JOB_A_B_SEED_BASE,
)
from ctmr.application.acceptance.distribution.challenge_registry import (
    DIAGNOSTIC_SEED_SLOTS,
    GLOBAL_SEED,
)
from ctmr.application.acceptance.distribution.token_dilution import (
    ANCHOR_MODALITY,
    ARM_CI_SLOT_BASE,
    ARM_ORDER,
    CANDIDATE_ARM,
    CONTRAST_CI_SLOT_BASE,
    CONTRAST_PAIRS,
    CONTROL_ARM,
    DIAGNOSTIC_SEED_BASE,
    SHARE_CI_SLOT,
    TOKEN_ARMS,
    BrightCoreStats,
    DiagnosticError,
    NiftiSampleRepository,
    SeedAnchor,
    TokenDilution,
    main,
)


def _volume(peak: int) -> np.ndarray:
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr[1, 1, 1] = peak
    return arr


def _write_arm(directory, case, arm, seed, peak):
    sitk.WriteImage(sitk.GetImageFromArray(_volume(peak)), str(directory / f"{case}_{arm}_seed{seed}.nii.gz"))


# ------------------------------------------------------------------ bright-core statistics


def test_bright_core_stats_matches_hand_computed_anchors():
    values = np.arange(1, 201, dtype=np.int16).reshape(10, 10, 2)
    stats = BrightCoreStats.of(values)
    assert stats["n_nonzero"] == 200
    assert stats["max"] == 200
    assert stats["p99"] == pytest.approx(198.01)  # index 0.99*199 = 197.01
    assert stats["p99_9"] == pytest.approx(199.801)  # index 0.999*199 = 198.801
    assert stats["top05pct_mean"] == pytest.approx(200.0)  # top 0.5% of 200 = exactly the max voxel


def test_top_fraction_keeps_at_least_one_voxel():
    arr = _volume(10)
    arr[2, 2, 2] = 5
    stats = BrightCoreStats.of(arr)
    assert stats["n_nonzero"] == 2
    assert stats["top05pct_mean"] == pytest.approx(10.0)


def test_all_zero_volume_reads_as_nulls_not_an_error():
    stats = BrightCoreStats.of(np.zeros((2, 2, 2), dtype=np.int16))
    assert stats == {"p99": None, "p99_9": None, "top05pct_mean": None, "max": None, "n_nonzero": 0}


def test_over_1000_output_domain_is_preserved_not_clipped():
    arr = np.zeros((2, 2, 2), dtype=np.int16)
    arr[0, 0, 0] = 1500  # >1.0 in the pre-int16 output domain (RC-4/E axis): measured as-is
    arr[1, 1, 1] = 500
    stats = BrightCoreStats.of(arr)
    assert stats["max"] == 1500
    assert stats["top05pct_mean"] == pytest.approx(1500.0)
    assert stats["n_nonzero"] == 2


# ------------------------------------------------------------------- arms & seed anchor


def test_token_arms_pin_the_frozen_modality_tokens_plus_the_panmr_control():
    # 29/30/31/34 verbatim from shell.MODALITY_TOKENS (frozen sampling rule; changes
    # gate through the frozen-artifact surface) + 8, the pan-MR token the
    # augmentation itself perturbs 34 into -- the diagnostic control arm.
    assert TOKEN_ARMS == {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31, "panmr": 8}
    assert CANDIDATE_ARM == "t1c"
    assert CONTROL_ARM == "panmr"
    assert ARM_ORDER == ("t1n", "t1c", "t2w", "t2f", "panmr")
    assert CONTRAST_PAIRS == (("t1c", "t1n"), ("t1c", "t2w"), ("t1c", "t2f"), ("t1c", "panmr"))


def test_seed_anchor_reproduces_the_frozen_sampling_rule():
    case = "BraTS-GLI-00016-000"
    expected = int(hashlib.sha256(f"{case}|{ANCHOR_MODALITY}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
    assert SeedAnchor.of(case) == expected
    assert ANCHOR_MODALITY == "t1c"  # the discriminated channel owns the seed anchor
    assert SeedAnchor.of("BraTS-MEN-00005-000") != SeedAnchor.of(case)


def test_diagnostic_seed_slots_stay_clear_of_the_formal_chain_and_jobs_ab():
    assert GLOBAL_SEED < DIAGNOSTIC_SEED_BASE
    assert DIAGNOSTIC_SEED_BASE == JOB_A_B_SEED_BASE
    assert DIAGNOSTIC_SEED_SLOTS["et_rel_diff"] == 200  # job A occupies slots 0/1 and 100/101, job B 200
    assert ARM_CI_SLOT_BASE == 300 and CONTRAST_CI_SLOT_BASE == 310 and SHARE_CI_SLOT == 320
    # job D has no challenge band: slots hang directly off the diagnostic base,
    # 300+ never collides with any A/B slot inside the base-aligned block
    for slot in (ARM_CI_SLOT_BASE + i for i in range(len(ARM_ORDER))):
        assert DIAGNOSTIC_SEED_BASE + slot >= DIAGNOSTIC_SEED_BASE


# ------------------------------------------------------------------- gain share & bands


def test_gain_share_bands_follow_the_two_thirds_one_third_attribution_precedent():
    share, classification = TokenDilution.gain_share(200.0, 100.0)
    assert share == pytest.approx(0.5)
    assert classification == "mixed"
    assert TokenDilution.gain_share(200.0, 190.0)[1] == "dilution_dominant"
    assert TokenDilution.gain_share(200.0, 50.0)[1] == "semantics_intact"
    # a control arm BRIGHTER than the candidate is the strongest dilution reading
    share, classification = TokenDilution.gain_share(200.0, 300.0)
    assert share == pytest.approx(0.0)
    assert classification == "dilution_dominant"


def test_control_arm_all_zero_reads_full_semantics():
    # an all-zero control has no bright core at all: every top voxel of the
    # candidate is token-34 semantics gain
    assert TokenDilution.gain_share(200.0, None) == (pytest.approx(1.0), "semantics_intact")


def test_candidate_arm_without_bright_core_has_no_share():
    assert TokenDilution.gain_share(None, 100.0) == (None, "no_bright_core")


# ------------------------------------------------------------------- repository invariants


def test_repository_parses_cohort_and_enforces_the_seed_anchor(tmp_path):
    case = "BraTS-GLI-00016-000"
    seed = SeedAnchor.of(case)
    for arm, peak in zip(ARM_ORDER, (500, 900, 600, 550, 850)):
        _write_arm(tmp_path, case, arm, seed, peak)
    cohort = NiftiSampleRepository(tmp_path).load_cohort()
    assert len(cohort) == 1
    assert cohort[0]["case"] == case
    assert cohort[0]["sub"] == "GLI"
    assert cohort[0]["seed"] == seed
    assert int(cohort[0]["arms"]["t1c"][1, 1, 1]) == 900
    assert cohort[0]["missing_arms"] == []


def test_repository_rejects_cross_arm_seed_mismatch(tmp_path):
    case = "BraTS-GLI-00016-000"
    seed = SeedAnchor.of(case)
    for arm in ARM_ORDER:
        _write_arm(tmp_path, case, arm, seed, 100)
    _write_arm(tmp_path, case, "t2f", seed + 1, 100)  # same arm rewritten at another seed
    with pytest.raises(DiagnosticError, match="跨臂"):
        NiftiSampleRepository(tmp_path).load_cohort()


def test_repository_rejects_a_non_frozen_seed_rule(tmp_path):
    case = "BraTS-GLI-00016-000"
    for arm in ARM_ORDER:
        _write_arm(tmp_path, case, arm, 12345, 100)
    with pytest.raises(DiagnosticError, match="冻结采样规则"):
        NiftiSampleRepository(tmp_path).load_cohort()


def test_repository_records_missing_arms_and_rejects_an_empty_directory(tmp_path):
    case = "BraTS-GLI-00016-000"
    seed = SeedAnchor.of(case)
    for arm in ("t1n", "t1c", "t2w"):
        _write_arm(tmp_path, case, arm, seed, 100)
    cohort = NiftiSampleRepository(tmp_path).load_cohort()
    assert cohort[0]["missing_arms"] == ["t2f", "panmr"]
    with pytest.raises(DiagnosticError, match="未发现"):
        NiftiSampleRepository(tmp_path / "nowhere").load_cohort()


# ------------------------------------------------------------------- readings & aggregation


def _two_case_cohort():
    peaks = {
        "BraTS-GLI-00016-000": ("GLI", {"t1n": 500, "t1c": 900, "t2w": 600, "t2f": 550, "panmr": 850}),
        "BraTS-MEN-00005-000": ("MEN", {"t1n": 800, "t1c": 1000, "t2w": 850, "t2f": 700, "panmr": 500}),
    }
    entries = []
    for case, (sub, arm_peaks) in peaks.items():
        entries.append(
            {
                "case": case,
                "sub": sub,
                "seed": SeedAnchor.of(case),
                "arms": {arm: _volume(peak) for arm, peak in arm_peaks.items()},
                "missing_arms": [],
            }
        )
    return entries


def test_read_cases_match_hand_computed_share_and_classification():
    cases = TokenDilution().read_cases(_two_case_cohort())
    assert cases[0]["gain_share"] == pytest.approx((900 - 850) / 900, abs=1e-6)
    assert cases[0]["classification"] == "dilution_dominant"
    assert cases[1]["gain_share"] == pytest.approx(0.5)
    assert cases[1]["classification"] == "mixed"
    assert cases[0]["excluded"] is None


def test_aggregate_matches_hand_computed_distributions_and_contrasts():
    cases = TokenDilution().read_cases(_two_case_cohort())
    agg = TokenDilution().aggregate(cases)
    assert agg["per_arm"]["t1c"]["token"] == 34
    assert agg["per_arm"]["t1c"]["top05pct_mean"]["median"] == pytest.approx(950.0)  # median(900,1000), linear
    assert agg["per_arm"]["t1c"]["top05pct_mean"]["ci90_low"] is not None
    assert agg["per_arm"]["t1c"]["p99"]["n_cases"] == 2
    contrast = agg["contrasts"]["t1c_vs_panmr"]
    assert contrast["median"] == pytest.approx(275.0)  # median(50, 500)
    assert contrast["candidate_token"] == 34 and contrast["reference_token"] == 8
    # shares median(0.0556, 0.5) = 0.2778 -> at or below 1/3: dilution dominant
    assert agg["attribution"]["median_share"] == pytest.approx(0.2778, abs=1e-3)
    assert agg["attribution"]["classification"] == "dilution_dominant"
    assert agg["n_excluded"] == 0


def test_incomplete_case_is_excluded_from_aggregates_but_listed():
    entries = _two_case_cohort()
    entries[1]["arms"] = {arm: volume for arm, volume in entries[1]["arms"].items() if arm != "panmr"}
    entries[1]["missing_arms"] = ["panmr"]
    cases = TokenDilution().read_cases(entries)
    assert cases[1]["excluded"] == "missing_arms:panmr"
    agg = TokenDilution().aggregate(cases)
    assert agg["n_excluded"] == 1
    assert agg["per_arm"]["t1c"]["top05pct_mean"]["n_cases"] == 1
    assert agg["contrasts"]["t1c_vs_panmr"]["n_cases"] == 1  # the complete case still contrasts
    assert agg["contrasts"]["t1c_vs_panmr"]["median"] == pytest.approx(50.0)


def test_single_case_ci_collapses_onto_the_value():
    cases = TokenDilution().read_cases(_two_case_cohort()[:1])
    agg = TokenDilution().aggregate(cases)
    top = agg["per_arm"]["t1c"]["top05pct_mean"]
    assert top["ci90_low"] == pytest.approx(900.0)
    assert top["ci90_high"] == pytest.approx(900.0)


# ------------------------------------------------------------------------ report & CLI


def test_cli_end_to_end_writes_json_and_markdown(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    plans = {
        "BraTS-GLI-00016-000": {"t1n": 500, "t1c": 900, "t2w": 600, "t2f": 550, "panmr": 950},
        "BraTS-MEN-00005-000": {"t1n": 800, "t1c": 1000, "t2w": 850, "t2f": 700, "panmr": 900},
    }
    for case, arm_peaks in plans.items():
        for arm, peak in arm_peaks.items():
            _write_arm(samples, case, arm, SeedAnchor.of(case), peak)
    out = tmp_path / "diag"
    rc = main(
        [
            "--samples-dir",
            str(samples),
            "--output-dir",
            str(out),
            "--run-id",
            "p1-20260822T131947Z",
            "--checkpoint",
            "/root/private_data/ctmr/runs/p1/ckpt/epoch_20.pt",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "token_dilution_diagnostic.json").read_text())
    assert payload["schema"] == "token-dilution-diagnostic/1"
    assert payload["variant"] == "diagnostic"
    assert payload["issue"] == 209
    assert payload["run_id"] == "p1-20260822T131947Z"
    assert payload["inputs"]["checkpoint"].endswith("epoch_20.pt")
    assert "不产生任何验收判定" in payload["disclaimer"]
    assert set(payload["per_arm"]) == {"t1n", "t1c", "t2w", "t2f", "panmr"}
    assert payload["per_arm"]["panmr"]["token"] == 8
    assert len(payload["per_case"]) == 2
    # shares: (900-950)->0.0 clamped and (1000-900)->0.1; median 0.05 -> dilution dominant
    assert payload["attribution"]["classification"] == "dilution_dominant"
    assert "verdict" not in json.dumps(payload)
    md = (out / "token_dilution_diagnostic.md").read_text()
    assert "variant: diagnostic" in md
    assert "同 seed" in md
    assert "panmr" in md
