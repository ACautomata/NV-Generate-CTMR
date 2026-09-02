"""Dev selection-point monitor sampling arm (issue #253, parent #247), observed
as pytest.

The arm turns the 1060-case dev list into the monitor's stratified sample
(never the holdout 530), samples the four pseudo-quad modalities per case with
the candidate checkpoint under the frozen sidecar recipe, and assembles the
two-sided instrument plan. The plan is schema-compatible with the terminal
acceptance's ``l2-final-acceptance-plan/1`` so the frozen-instrument execution
side (predict scripts / assemble-execute / measure) runs verbatim, read-only.
"""

import hashlib
import json
from pathlib import Path

import pytest

from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError
from ctmr.application.acceptance.distribution.final_acceptance import PLAN_SCHEMA, PredictScriptWriter
from ctmr.application.generation.modality_label.dev_monitor_sampling import (
    MONITOR_QUOTAS,
    DevMonitorCohort,
    DevMonitorPlanBuilder,
    SamplingProvenance,
    main,
)
from ctmr.application.generation.modality_label.monitor import CandidateSampler
from ctmr.application.generation.trend import DevCohortBuilder
from ctmr.application.shell import TARGET_MODALITIES

pytestmark = pytest.mark.torch


def _dev_list(path, cases_by_challenge):
    word_of = {
        "mri_t1_skull_stripped": "t1n",
        "mri_t1c_skull_stripped": "t1c",
        "mri_t2_skull_stripped": "t2w",
        "mri_flair_skull_stripped": "t2f",
    }
    entries = []
    for challenge, cases in cases_by_challenge.items():
        for case in cases:
            for modality, word in word_of.items():
                entries.append({"sub": challenge, "case": case, "modality": modality, "image": f"{challenge}/{case}/{case}_{word}.nii.gz"})
    path.write_text(json.dumps({"training": entries}))
    return path


def _cases(challenge, count):
    return [f"DEV-{challenge}-{index:04d}-000" for index in range(count)]


# ------------------------------------------------------------------- the sample


def test_monitor_quotas_are_pinned_stratified_and_holdout_free():
    assert MONITOR_QUOTAS == {"GLI": 50, "MEN": 40, "METS": 24, "PED": 10, "SSA": 6}
    assert sum(MONITOR_QUOTAS.values()) == 130


def test_cohort_is_stratified_deterministic_and_reproducible(tmp_path):
    population = {challenge: _cases(challenge, quota + 7) for challenge, quota in MONITOR_QUOTAS.items()}
    dev_list = _dev_list(tmp_path / "dev.json", population)

    cohort = DevMonitorCohort(dev_list).build()

    assert DevMonitorCohort(dev_list).build() == cohort  # deterministic
    by_challenge = {}
    for item in cohort:
        by_challenge.setdefault(item["sub"], []).append(item["case"])
    assert {challenge: len(cases) for challenge, cases in by_challenge.items()} == MONITOR_QUOTAS
    # the sha256-order rule is the DevCohortBuilder's, generalized to the monitor quotas
    expected_first = sorted(population["METS"], key=lambda case: hashlib.sha256(f"METS/{case}".encode()).hexdigest())[0]
    assert by_challenge["METS"][0] == expected_first


def test_cohort_shortfall_fails_loudly_instead_of_shrinking_the_sample(tmp_path):
    """A silent shortfall would change the flag rule's resolution: fail."""
    population = {
        "GLI": _cases("GLI", 3),
        "MEN": _cases("MEN", 40),
        "METS": _cases("METS", 24),
        "PED": _cases("PED", 10),
        "SSA": _cases("SSA", 6),
    }
    dev_list = _dev_list(tmp_path / "dev.json", population)
    with pytest.raises(DiagnosticError):
        DevMonitorCohort(dev_list).build()


def test_cohort_superset_of_the_16_case_dev_cohort(tmp_path):
    """The monitor sample and the 16-case sidecar cohort draw from the same dev
    population with the same sha256 ordering rule (the sidecar quota is a prefix
    of that order) — both holdout-free, no contradiction."""
    population = {challenge: _cases(challenge, quota + 5) for challenge, quota in MONITOR_QUOTAS.items()}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    monitor = {item["case"] for item in DevMonitorCohort(dev_list).build()}
    sidecar = {item["case"] for item in DevCohortBuilder(dev_list).build()}
    assert sidecar <= monitor


# ------------------------------------------------------------------- the plan


def _make_samples(samples_dir, cohort):
    for item in cohort:
        for modality in TARGET_MODALITIES:
            seed = CandidateSampler.seed_of(item["case"], modality)
            path = samples_dir / f"{item['case']}_{modality}_seed{seed}.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"nii-placeholder")


def _make_raw(raw_root, cases_by_challenge):
    for challenge, cases in cases_by_challenge.items():
        for case in cases:
            for modality in ("t1n", "t1c", "t2w", "t2f"):
                path = raw_root / challenge / case / f"{case}_{modality}.nii.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"nii-placeholder")


def test_plan_builder_emits_the_execution_side_schema_with_both_sides(tmp_path):
    population = {"GLI": _cases("GLI", 6), "METS": _cases("METS", 5)}
    quotas = {"GLI": 4, "METS": 4}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    _make_raw(tmp_path / "raw", population)
    cohort = DevMonitorCohort(dev_list, quotas=quotas).build()
    _make_samples(tmp_path / "samples", cohort)
    builder = DevMonitorPlanBuilder(dev_list, tmp_path / "raw", tmp_path / "samples", quotas=quotas, run_id="p1-20260822T131947Z")

    plan = builder.build(cohort)

    assert plan["schema"] == PLAN_SCHEMA
    assert plan["phase"] == "P1"
    assert plan["population"] == "dev"
    assert plan["run_id"] == "p1-20260822T131947Z"
    assert plan["challenges"] == {
        "GLI": {"n_cases": 4, "quota": 4, "provisional": False},
        "METS": {"n_cases": 4, "quota": 4, "provisional": False},
    }
    observations = plan["observations"]
    assert len(observations) == 16  # (4 gen + 4 real) x 2 challenges
    obs_ids = [obs["obs_id"] for obs in observations]
    assert len(set(obs_ids)) == len(obs_ids)
    gen = next(obs for obs in observations if obs["obs_id"].endswith("__gen"))
    real = next(obs for obs in observations if obs["obs_id"].endswith("__real"))
    assert set(gen["channels"]) == {"0000", "0001", "0002", "0003"}
    assert all(Path(path).is_file() for path in gen["channels"].values())  # the sampled volumes
    assert all(str(tmp_path / "raw") in path for path in real["channels"].values())  # the dev-list real paths
    assert real["side"] == "real" and gen["side"] == "gen"


def test_plan_is_accepted_by_the_frozen_instrument_script_writer(tmp_path):
    """The execution-side contract: PredictScriptWriter consumes the monitoring
    plan verbatim (schema gate) and writes the per-challenge frozen scripts."""
    population = {"METS": _cases("METS", 5)}
    quotas = {"METS": 4}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    _make_raw(tmp_path / "raw", population)
    cohort = DevMonitorCohort(dev_list, quotas=quotas).build()
    _make_samples(tmp_path / "samples", cohort)
    plan = DevMonitorPlanBuilder(dev_list, tmp_path / "raw", tmp_path / "samples", quotas=quotas).build(cohort)

    runner = PredictScriptWriter(plan, tmp_path / "exec").write()

    assert runner.is_file()
    scripts = list((tmp_path / "exec").glob("predict_*.sh"))
    assert len(scripts) == 2  # the METS predict script + the all-runner


def test_plan_builder_fails_loudly_on_a_missing_sample(tmp_path):
    population = {"METS": _cases("METS", 5)}
    quotas = {"METS": 4}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    _make_raw(tmp_path / "raw", population)
    cohort = DevMonitorCohort(dev_list, quotas=quotas).build()  # no samples written
    with pytest.raises(DiagnosticError):
        DevMonitorPlanBuilder(dev_list, tmp_path / "raw", tmp_path / "samples", quotas=quotas).build(cohort)


def test_plan_builder_fails_loudly_on_a_missing_real_modality(tmp_path):
    population = {"METS": _cases("METS", 5)}
    quotas = {"METS": 4}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    cohort = DevMonitorCohort(dev_list, quotas=quotas).build()
    _make_samples(tmp_path / "samples", cohort)
    with pytest.raises(DiagnosticError):
        DevMonitorPlanBuilder(dev_list, tmp_path / "raw", tmp_path / "samples", quotas=quotas).build(cohort)


# ------------------------------------------------------------- seeds & CLI stage


def _ckpt(path, content):
    path.write_bytes(content)
    return path


def test_sampling_provenance_records_then_accepts_the_same_checkpoint(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    ckpt = _ckpt(tmp_path / "epoch_20.pt", b"candidate-A")
    provenance = SamplingProvenance(samples)

    provenance.verify_or_record(ckpt)  # the first sampling run records the fingerprint
    provenance.verify_or_record(ckpt)  # re-entry under the same candidate is a no-op

    manifest = json.loads((samples / SamplingProvenance.MANIFEST_NAME).read_text())
    assert manifest["ckpt_sha256"] == hashlib.sha256(b"candidate-A").hexdigest()


def test_sampling_provenance_rejects_a_different_checkpoint_into_the_same_dir(tmp_path):
    """The T8 silent-reuse bug: the re-entrant skip keys on the sample filename
    (case+modality+seed, no checkpoint), so swapping only CKPT/RUN_ID into an
    existing samples dir would reuse the baseline's volumes under the new run_id.
    The provenance manifest must instead fail loudly."""
    samples = tmp_path / "samples"
    samples.mkdir()
    SamplingProvenance(samples).verify_or_record(_ckpt(tmp_path / "epoch_20.pt", b"candidate-A"))

    with pytest.raises(DiagnosticError):
        SamplingProvenance(samples).verify_or_record(_ckpt(tmp_path / "epoch_40.pt", b"candidate-B"))


def test_sampling_provenance_fails_loudly_on_a_missing_checkpoint(tmp_path):
    with pytest.raises(DiagnosticError):
        SamplingProvenance(tmp_path / "samples").verify_or_record(tmp_path / "nope.pt")


def test_pseudo_quad_seed_rule_gives_four_distinct_seeds_per_case():
    seeds = {CandidateSampler.seed_of("DEV-METS-0000-000", modality) for modality in TARGET_MODALITIES}
    assert len(seeds) == 4  # the P1 independence obligation (manifest seeds pairwise distinct)


def test_plan_only_cli_writes_cohort_and_plan_without_touching_the_gpu(tmp_path):
    """--plan-only: (re)assemble the chain artifacts from an existing samples
    dir — the shakedown and re-entry path, zero GPU."""
    population = {challenge: _cases(challenge, quota) for challenge, quota in MONITOR_QUOTAS.items()}
    dev_list = _dev_list(tmp_path / "dev.json", population)
    _make_raw(tmp_path / "raw", population)
    cohort = DevMonitorCohort(dev_list).build()
    _make_samples(tmp_path / "samples", cohort)

    rc = main(
        [
            "--dev-list",
            str(dev_list),
            "--raw-root",
            str(tmp_path / "raw"),
            "--samples-dir",
            str(tmp_path / "samples"),
            "--output-dir",
            str(tmp_path / "monitor"),
            "--plan-only",
        ]
    )
    assert rc == 0
    cohort_doc = json.loads((tmp_path / "monitor" / "cohort.json").read_text())
    assert cohort_doc["quotas"] == MONITOR_QUOTAS
    assert cohort_doc["population"] == "dev"
    assert len(cohort_doc["cohort"]) == 130
    plan = json.loads((tmp_path / "monitor" / "plan.json").read_text())
    assert plan["schema"] == PLAN_SCHEMA and plan["population"] == "dev"


def test_sampling_only_flag_discipline(tmp_path):
    """--sampling-only defers the plan (raw-root not needed); the flag pair is
    mutually exclusive; the sampling args stay required (the sampling itself is
    a server-side GPU path, not a CI e2e)."""
    population = {challenge: _cases(challenge, quota) for challenge, quota in MONITOR_QUOTAS.items()}
    dev_list = _dev_list(tmp_path / "dev.json", population)

    with pytest.raises(SystemExit):
        main(
            [
                "--dev-list",
                str(dev_list),
                "--samples-dir",
                str(tmp_path / "samples"),
                "--output-dir",
                str(tmp_path / "monitor"),
                "--sampling-only",
                "--plan-only",
            ]
        )
    assert not (tmp_path / "monitor" / "cohort.json").exists()  # the pair check precedes any work

    with pytest.raises(SystemExit):
        main(
            [
                "--dev-list",
                str(dev_list),
                "--samples-dir",
                str(tmp_path / "samples"),
                "--output-dir",
                str(tmp_path / "monitor"),
                "--sampling-only",
            ]
        )  # sampling args still required (no --ckpt given)
