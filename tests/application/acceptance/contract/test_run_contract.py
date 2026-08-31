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

"""The run-contract orchestration face, observed as pytest (#141).

The resident ``ContractSelfTest`` of ``brats_phase_run_contract.py`` (retired scripts layer, git history)
became this file when the contract moved into the acceptance.contract package
(ADR-0015 §6): the full P1 positive path with negative attachment probes, the
holdout/replay guards, the L2 binding/coverage/verdict chain, the
non-compensatory final acceptance with traceable blockers and DM-source
registration, the phase chain (P2/P3 pin the registered P1-DM; P3 never pins
a P2), the stage-0 baseline floor, DM-retrain supersession, tamper detection
and the controlled-storage guard. ``attach``/``freeze``/``conclude``/``verify``
behavior is asserted through the same classes the CLI drives, so the moved
verbs stay behavior-equivalent to the retired script. Synthetic non-subject
ids only; stdlib only.
"""

import json
from pathlib import Path

import pytest

from ctmr.application.acceptance.contract import (
    DISTRIBUTION_CHALLENGES,
    EXPERT_REVIEW_DIMENSIONS,
    EXPERT_REVIEW_MODALITIES,
    QUANTITATIVE_MODALITIES,
    QUANTITATIVE_PLANES,
    QUANTITATIVE_SCHEMA,
    QUANTITATIVE_T1N_TO_T1C,
    SCHEMA,
    STATUS_FROZEN,
    STATUS_OPEN,
    ArtifactFingerprinter,
    CandidateFreezer,
    ContractViolationError,
    FinalAcceptanceJudge,
    ManifestSides,
    ReportAttacher,
    RunInitializer,
    RunRecordStore,
    RunVerifier,
    SelectionRecorder,
)
from ctmr.domain.dmsource import DmSourceViolationError
from ctmr.domain.identity import WeightsRef
from ctmr.infrastructure.dmsource import DmSourceLedger

QUOTAS = {
    "GLI": {"train": ["FIXGLI-0000-000", "FIXGLI-0001-000"], "dev": ["FIXGLI-0100-000"], "holdout": ["FIXGLI-0200-000"]},
    "SSA": {"train": ["FIXSSA-0000-000"], "dev": ["FIXSSA-0100-000"], "holdout": ["FIXSSA-0200-000"]},
}


@pytest.fixture()
def fixture_root(tmp_path):
    """The shared synthetic evidence tree (manifest, lists, configs, checkpoints, metrics)."""
    root = tmp_path / "fixture"
    manifest = {"split_id": "selftest", "challenges": {}}
    for ch, sides in QUOTAS.items():
        manifest["challenges"][ch] = {"cases": dict(sides)}
    manifest_path = root / "phase_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lists_dir = root / "lists"
    lists_dir.mkdir(parents=True)
    (lists_dir / "train.json").write_text(
        json.dumps({"training": [{"sub": "GLI", "case": "FIXGLI-0000-000"}, {"sub": "SSA", "case": "FIXSSA-0000-000"}]})
    )
    (lists_dir / "dev.json").write_text(json.dumps({"training": [{"sub": "GLI", "case": "FIXGLI-0100-000"}]}))
    (lists_dir / "holdout.json").write_text(json.dumps({"training": [{"sub": "GLI", "case": "FIXGLI-0200-000"}]}))
    (lists_dir / "mislabelled.json").write_text(json.dumps({"training": [{"sub": "GLI", "case": "FIXGLI-0100-000"}]}))
    (lists_dir / "combined_sided.json").write_text(
        json.dumps({"training": [{"sub": "GLI", "case": "FIXGLI-0000-000"}, {"sub": "GLI", "case": "FIXGLI-0100-000"}]})
    )
    (lists_dir / "replay.json").write_text(
        json.dumps({"training": [{"sub": "MRRATE", "case": "AB12CD34EF"}, {"sub": "MRRATE", "case": "FG56HI78JK"}]})
    )
    (lists_dir / "replay_collision.json").write_text(json.dumps({"training": [{"sub": "MRRATE", "case": "FIXGLI-0000-000"}]}))

    (root / "env_config.json").write_text('{"lr": 2e-06, "n_epochs": 100}\n')
    (root / "model_config.json").write_text('{"batch_size": 1}\n')
    (root / "infer_config.json").write_text(
        '{"schema": "brats-p3-stage0-infer/1", "scheduler": "RFlowScheduler", "num_inference_steps": 30, '
        '"cfg_guidance_scale": 10.0, "strength": 0.9, "modality_tokens": {"t1n": 29, "t2w": 30, "t2f": 31, "t1c": 34}, '
        '"seed_rule": "sha256"}\n'
    )
    (root / "base_ckpt.pt").write_bytes(b"rflow-mr-brain-v1-fixture")
    (root / "candidate.pt").write_bytes(b"candidate-fixture")
    (root / "controlnet_candidate.pt").write_bytes(b"controlnet-candidate-fixture")
    (root / "dev_metrics.json").write_text(json.dumps({"metrics": [{"sub": "GLI", "case": "FIXGLI-0100-000", "fid": 0.42}]}))
    (root / "holdout_metrics.json").write_text(json.dumps({"metrics": [{"sub": "GLI", "case": "FIXGLI-0200-000", "fid": 0.1}]}))
    (root / "train_metrics.json").write_text(json.dumps({"metrics": [{"sub": "SSA", "case": "FIXSSA-0000-000", "loss": 0.1}]}))
    (root / "empty_metrics.json").write_text('{"summary": {"aggregate_fid": 0.5}}\n')
    (root / "samples.json").write_text('{"samples": ["sample-t1n-000.nii.gz"]}\n')
    (root / "l1_report.json").write_text('{"fid": {"t1n": 0.05}}\n')
    (root / "invalid_l1_report.json").write_text('{"schema": "brats-l1-report/1", "binding": {"run_id": "wrong-run"}}\n')
    (root / "platform.json").write_text('{"world_size": 1, "amp_dtype": "bf16"}\n')
    return root


@pytest.fixture()
def fingerprinter():
    return ArtifactFingerprinter()


def store_at(tmp_path, name):
    return RunRecordStore(tmp_path / name)


def expect_reject(action, label):
    try:
        action()
    except ContractViolationError:
        return
    pytest.fail(f"expected rejection but succeeded: {label}")


def initializer(store, fingerprinter, fixture_root):
    """The real-adapter wiring: the json-backed ledger factory rides every use case (the composition root's injection, observed)."""
    return RunInitializer(store, fingerprinter, ManifestSides.from_path(fixture_root / "phase_manifest.json"), DmSourceLedger)


def open_passing_candidate(tmp_path, store, fingerprinter, fixture_root, run_id, checkpoint=None):
    """A frozen P1 candidate shell (init -> select -> freeze) ready for report attachments."""
    run_path = initializer(store, fingerprinter, fixture_root).init(
        "P1",
        run_id,
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        fixture_root / "base_ckpt.pt",
        None,
        None,
    )
    SelectionRecorder(store, fingerprinter).select(
        run_path, checkpoint or fixture_root / "candidate.pt", "dev FID trend, early stop patience 10", [fixture_root / "dev_metrics.json"], epoch=7
    )
    CandidateFreezer(store, fingerprinter).freeze(run_path, fixture_root / "samples.json")
    return run_path


def write_l1_report(path, record, passing=False):
    interval = {"point": 0.4, "ci95": [0.3, 0.5]}
    baseline = {"planes": {plane: interval for plane in QUANTITATIVE_PLANES}, "mean": interval, "mean_bootstrap_median": 0.4}
    generated_interval = {"point": 0.8, "ci95": [0.7, 0.9 if passing else 1.1]}
    generated = {"planes": {plane: generated_interval for plane in QUANTITATIVE_PLANES}, "mean": generated_interval}
    fid_results = []
    for challenge in QUOTAS:
        for modality in QUANTITATIVE_MODALITIES:
            fid_results.append(
                {
                    "challenge": challenge,
                    "target_modality": modality,
                    "generated_source_modalities": [source for source in QUANTITATIVE_MODALITIES if source != modality]
                    if record["phase"] == "P3"
                    else [],
                    "generated_vs_holdout": generated,
                    "train_vs_holdout_baseline": baseline,
                    "threshold": 1.0,
                    "verdict": "pass" if passing else "fail",
                }
            )
    p3_results = []
    if record["phase"] == "P3":
        paired_interval = {"point": 0.11, "ci95": [0.01, 0.20]}
        ssim_interval = {"point": 0.03, "ci95": [0.01, 0.04]}
        for challenge in QUOTAS:
            for source in QUANTITATIVE_MODALITIES:
                for target in QUANTITATIVE_MODALITIES:
                    if source == target:
                        continue
                    exceptional = (source, target) == QUANTITATIVE_T1N_TO_T1C
                    p3_results.append(
                        {
                            "challenge": challenge,
                            "src_modality": source,
                            "target_modality": target,
                            "case_count": 2,
                            "mae_relative_reduction": paired_interval,
                            "ssim_increase": ssim_interval,
                            "gate_applicable": not exceptional,
                            "verdict": "not_applicable_known_unobservable" if exceptional else "pass",
                        }
                    )
    report = {
        "schema": QUANTITATIVE_SCHEMA,
        "binding": {
            "run_id": record["run_id"],
            "phase": record["phase"],
            "manifest_sha256": record["manifest"]["sha256"],
            "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
            "samples_sha256": record["samples"]["sha256"],
        },
        "protocol": {
            "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
            "mr_preprocessing": "percentile_0_99.5_to_0_1_ras_1mm_zero_pad",
            "planes": list(QUANTITATIVE_PLANES),
            "bootstrap": {"method": "case_level_percentile_pcg64", "confidence_level": 0.95, "resamples": 32},
            "fid_multiplier": 2.5,
        },
        "fid_results": fid_results,
        "p3_paired_results": p3_results,
        "summary": {"verdict": "pass" if passing else "fail"},
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def write_l3_report(path, record):
    challenges = tuple(sorted(QUOTAS))
    per_cell = 5
    coverage = []
    for challenge in challenges:
        for modality in EXPERT_REVIEW_MODALITIES:
            coverage.append({"challenge": challenge, "target_modality": modality, "real": per_cell, "synth": per_cell})
    real_total = per_cell * len(challenges) * len(EXPERT_REVIEW_MODALITIES)  # 5 * 2 * 4 = 40
    per_reviewer = []
    for reviewer in ("R1", "R2"):
        per_reviewer.append(
            {
                "reviewer": reviewer,
                "n": real_total + real_total,
                "balanced_accuracy": 0.5,
                "confusion": {
                    "real_said_real": real_total // 2,
                    "real_said_synth": real_total // 2,
                    "synth_said_real": real_total // 2,
                    "synth_said_synth": real_total // 2,
                },
                "ci95": [0.42, 0.58],
                "verdict": "pass",
            }
        )
    pooled = {"reviewers": 2, "n": per_reviewer[0]["n"] * 2, "balanced_accuracy": 0.5, "ci95": [0.44, 0.56], "verdict": "pass"}
    likert = []
    for dimension in EXPERT_REVIEW_DIMENSIONS:
        phase = {"point": 4.2, "ci95_lower": 4.1, "n": per_reviewer[0]["n"] * 2, "na": 0, "verdict": "pass"}
        per_modality = {
            modality: {"point": 4.2, "ci95_lower": 4.1, "n": per_cell * 2 * len(challenges), "na": 0, "verdict": "pass"}
            for modality in EXPERT_REVIEW_MODALITIES
        }
        likert.append({"dimension": dimension, "phase": phase, "per_modality": per_modality, "fleiss_kappa": 0.4})
    report = {
        "schema": "brats-l3-report/1",
        "binding": {
            "run_id": record["run_id"],
            "phase": record["phase"],
            "manifest_sha256": record["manifest"]["sha256"],
            "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
            "samples_sha256": record["samples"]["sha256"],
        },
        "protocol": {
            "reviewers": 2,
            "dimensions": list(EXPERT_REVIEW_DIMENSIONS),
            "target_modalities": list(EXPERT_REVIEW_MODALITIES),
            "visual_turing_ci_window": [0.40, 0.60],
            "likert_minimum": 4.0,
            "likert_scale": {"min": 1, "max": 5},
            "confidence_level": 0.95,
            "bootstrap": {"method": "entry_level_stratified_percentile_mt19937", "resamples": 100, "seed": 20260821},
            "per_cell": per_cell,
            "total_entries": per_cell * 2 * len(challenges) * len(EXPERT_REVIEW_MODALITIES),
        },
        "coverage": coverage,
        "provenance": {"catalog_sha256": "c" * 64, "blind_map_sha256": "b" * 64},
        "visual_turing": {"per_reviewer": per_reviewer, "pooled": pooled, "verdict": "pass", "fleiss_kappa": 0.35},
        "likert": likert,
        "verdict": {"visual_turing": "pass", "likert": "pass", "overall": "pass"},
    }
    Path(path).write_text(json.dumps(report, indent=2) + "\n")


def write_l2_report(path, record, failing_challenges=(), undecided_challenges=()):
    """A five-challenge L2 fixture; failing/undecided challenge names override their verdicts."""
    per_challenge = {}
    for challenge in DISTRIBUTION_CHALLENGES:
        n_failed = 1 if challenge in undecided_challenges else 0
        passed = challenge not in failing_challenges and challenge not in undecided_challenges
        if challenge in undecided_challenges:
            verdict = "undecided"
        elif challenge in failing_challenges:
            verdict = "fail"
        else:
            verdict = "pass"
        tost = [
            {
                "quantity": "vol_wt_rel",
                "margin": 0.2802,
                "ci90_low": -0.02,
                "ci90_high": 0.02,
                "n_cases": 6,
                "n_excluded": 0,
                "exclusion_reasons": {},
                "passed": passed if challenge not in undecided_challenges else True,
            }
        ]
        round_trip = None
        if record["phase"] == "P2":
            round_trip = [
                {
                    "region": region,
                    "floor": 0.0,
                    "bound": 0.9,
                    "n_cases": 6,
                    "n_excluded": 0,
                    "vacuous_pass": False,
                    "passed": passed,
                }
                for region in ("WT", "TC", "ET")
            ]
        per_challenge[challenge] = {
            "challenge": challenge,
            "n_observations": 12,
            "failure_audit": {
                "n_failed": n_failed,
                "breakdown": {"input_fail": 0, "run_fail": n_failed, "hier_viol": 0},
                "n_failed_by_side": {"gen": n_failed, "real": 0},
                "wilson_95_upper": 0.08 if n_failed else 0.0,
            },
            "r_fail_point": 0.0,
            "tost": tost,
            "round_trip": round_trip,
            "verdict": verdict,
        }
        if verdict == "undecided":
            per_challenge[challenge]["reason"] = (
                "instrument failure on tested samples (input/run/hierarchy); blocks final acceptance -- "
                "fix direction is the instrument or a re-run, not the candidate"
            )
    verdicts = [info["verdict"] for info in per_challenge.values()]
    overall = "undecided" if "undecided" in verdicts else "pass" if all(v == "pass" for v in verdicts) else "fail"
    report = {
        "schema": "l2-final-acceptance-report/1",
        "title": "L2 冻结仪器最终验收报告",
        "phase": record["phase"],
        "run_id": record["run_id"],
        "binding": {
            "run_id": record["run_id"],
            "phase": record["phase"],
            "manifest_sha256": record["manifest"]["sha256"],
            "candidate_checkpoint_sha256": record["selection"]["checkpoint"]["sha256"],
            "samples_sha256": record["samples"]["sha256"],
        },
        "provisional_challenges": [],
        "challenges_missing": [],
        "complete_coverage": True,
        "overall_verdict": overall,
        "per_challenge": per_challenge,
    }
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def test_p1_positive_path_with_negative_attachment_probes(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = initializer(records, fingerprinter, fixture_root).init(
        "P1",
        "p1-fixture",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json"), ("model", fixture_root / "model_config.json")],
        [("train", fixture_root / "lists/train.json")],
        fixture_root / "base_ckpt.pt",
        None,
        fixture_root / "platform.json",
    )
    SelectionRecorder(records, fingerprinter).select(
        p1_path, fixture_root / "candidate.pt", "dev FID trend, early stop patience 10", [fixture_root / "dev_metrics.json"], epoch=5
    )
    CandidateFreezer(records, fingerprinter).freeze(p1_path, fixture_root / "samples.json")
    write_l1_report(fixture_root / "l1_report.json", records.load_by_path(p1_path))

    # unbound L1 report
    with pytest.raises(ContractViolationError):
        ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", fixture_root / "invalid_l1_report.json")
    # L1 summary inconsistent with results
    invalid_summary = json.loads((fixture_root / "l1_report.json").read_text())
    invalid_summary["summary"] = {"verdict": "pass"}
    (fixture_root / "invalid_l1_summary.json").write_text(json.dumps(invalid_summary))
    expect_reject(lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", fixture_root / "invalid_l1_summary.json"), "summary")
    # incomplete L1 metrics (undecided)
    incomplete = json.loads((fixture_root / "l1_report.json").read_text())
    incomplete["fid_results"][0]["verdict"] = "undecided"
    incomplete["summary"] = {"verdict": "undecided"}
    (fixture_root / "incomplete_l1_report.json").write_text(json.dumps(incomplete))
    expect_reject(
        lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", fixture_root / "incomplete_l1_report.json"), "incomplete"
    )
    # unverified L1 feature provenance
    invalid_provenance = json.loads((fixture_root / "l1_report.json").read_text())
    invalid_provenance["protocol"]["mr_preprocessing"] = "unverified"
    (fixture_root / "invalid_l1_provenance.json").write_text(json.dumps(invalid_provenance))
    expect_reject(
        lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", fixture_root / "invalid_l1_provenance.json"), "provenance"
    )
    # L1 report in a public work tree
    public_report = tmp_path / "l1-public-repo" / "l1_report.json"
    (public_report.parent / ".git").mkdir(parents=True)
    public_report.write_text((fixture_root / "l1_report.json").read_text())
    expect_reject(lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", public_report), "public work tree")

    ReportAttacher(records, fingerprinter).attach(p1_path, "l1_report", fixture_root / "l1_report.json")
    write_l3_report(fixture_root / "l3_report.json", records.load_by_path(p1_path))
    # L3 non-compensatory AND mismatch
    invalid_l3 = json.loads((fixture_root / "l3_report.json").read_text())
    invalid_l3["visual_turing"]["verdict"] = "fail"
    invalid_l3["verdict"] = {"visual_turing": "fail", "likert": "pass", "overall": "pass"}
    (fixture_root / "invalid_l3_verdict.json").write_text(json.dumps(invalid_l3))
    expect_reject(lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l3_report", fixture_root / "invalid_l3_verdict.json"), "L3 AND")
    # malformed L3 report rejected without a crash
    malformed_l3 = json.loads((fixture_root / "l3_report.json").read_text())
    del malformed_l3["protocol"]["bootstrap"]
    malformed_l3["visual_turing"]["pooled"] = None
    (fixture_root / "malformed_l3_report.json").write_text(json.dumps(malformed_l3))
    expect_reject(
        lambda: ReportAttacher(records, fingerprinter).attach(p1_path, "l3_report", fixture_root / "malformed_l3_report.json"), "malformed L3"
    )
    ReportAttacher(records, fingerprinter).attach(p1_path, "l3_report", fixture_root / "l3_report.json")

    verifier = RunVerifier(fingerprinter, DmSourceLedger)
    failures = verifier.verify(records.load_by_path(p1_path), record_path=p1_path)
    assert failures == []

    frozen_run = records.load_by_path(p1_path)
    assert frozen_run["status"] == "frozen"
    assert frozen_run.get("samples") is not None
    # selection is locked once frozen; unknown attachment kinds never exist
    with pytest.raises(ContractViolationError, match="locked"):
        SelectionRecorder(records, fingerprinter).select(
            p1_path, fixture_root / "candidate.pt", "re-select", [fixture_root / "dev_metrics.json"], None
        )
    with pytest.raises(ContractViolationError, match="attachment kind"):
        ReportAttacher(records, fingerprinter).attach(p1_path, "l4_report", fixture_root / "l1_report.json")


@pytest.mark.parametrize(
    "label,data_lists",
    (
        ("holdout data list", [("train", "lists/holdout.json")]),
        ("mislabelled side list", [("train", "lists/mislabelled.json")]),
        ("replay collision list", [("train", "lists/train.json"), ("replay", "lists/replay_collision.json")]),
    ),
)
def test_init_guards_reject_bad_data_lists(tmp_path, fixture_root, fingerprinter, label, data_lists):
    fresh = store_at(tmp_path, "records_reject")
    resolved = [(side, fixture_root / path) for side, path in data_lists]

    with pytest.raises(ContractViolationError):
        initializer(fresh, fingerprinter, fixture_root).init(
            "P1",
            "p1-bad",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            resolved,
            fixture_root / "base_ckpt.pt",
            None,
            None,
        )


def test_p1_replay_positive_path(tmp_path, fixture_root, fingerprinter):
    replay_store = store_at(tmp_path, "records_replay")
    replay_path = initializer(replay_store, fingerprinter, fixture_root).init(
        "P1",
        "p1-replay-fixture",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json"), ("replay", fixture_root / "lists/replay.json")],
        fixture_root / "base_ckpt.pt",
        None,
        None,
    )

    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(replay_store.load_by_path(replay_path), record_path=replay_path)

    assert failures == []


def test_open_run_evidence_guards(tmp_path, fixture_root, fingerprinter):
    open_store = store_at(tmp_path, "records_open")
    open_path = initializer(open_store, fingerprinter, fixture_root).init(
        "P1",
        "p1-open",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        fixture_root / "base_ckpt.pt",
        None,
        None,
    )
    recorder = SelectionRecorder(open_store, fingerprinter)
    for evidence in ("holdout_metrics.json", "train_metrics.json", "empty_metrics.json"):
        with pytest.raises(ContractViolationError):
            recorder.select(open_path, fixture_root / "candidate.pt", "rule", [fixture_root / evidence], None)
    with pytest.raises(ContractViolationError, match="selection basis"):
        CandidateFreezer(open_store, fingerprinter).freeze(open_path, fixture_root / "samples.json")


def test_l2_attachment_binding_coverage_and_verdict_chain(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-fixture")
    p1_record = records.load_by_path(p1_path)
    write_l2_report(fixture_root / "l2_report.json", p1_record)
    attacher = ReportAttacher(records, fingerprinter)
    l2_mutations = (
        ("unbound L2 report", lambda r: r["binding"].update(run_id="wrong-run")),
        (
            "provisional L2 coverage",
            lambda r: (r.update(provisional_challenges=["GLI"], complete_coverage=False), r.update(challenges_missing=["PED"]))[0],
        ),
        (
            "L2 overall-verdict mismatch",
            lambda r: (
                r["per_challenge"]["SSA"].update(verdict="fail", tost=[dict(r["per_challenge"]["SSA"]["tost"][0], passed=False)]),
                r,
            )[1],
        ),
        ("L2 P1 carrying round-trip evidence", lambda r: r["per_challenge"]["GLI"].update(round_trip=[{"region": "WT", "passed": True}])),
        (
            "L2 challenge verdict disagreeing with its evidence",
            lambda r: r["per_challenge"]["MEN"].update(verdict="pass", tost=[dict(r["per_challenge"]["MEN"]["tost"][0], passed=False)]),
        ),
    )
    for label, mutate in l2_mutations:
        report = json.loads((fixture_root / "l2_report.json").read_text())
        mutate(report)
        bad_path = fixture_root / f"bad_l2_{label.replace(' ', '_')}.json"
        bad_path.write_text(json.dumps(report))
        with pytest.raises(ContractViolationError):
            attacher.attach(p1_path, "l2_report", bad_path)
    malformed_l2 = json.loads((fixture_root / "l2_report.json").read_text())
    malformed_l2["per_challenge"]["GLI"] = "not-an-object"
    (fixture_root / "malformed_l2_report.json").write_text(json.dumps(malformed_l2))
    with pytest.raises(ContractViolationError):
        attacher.attach(p1_path, "l2_report", fixture_root / "malformed_l2_report.json")

    attacher.attach(p1_path, "l2_report", fixture_root / "l2_report.json")  # the valid report attaches


def test_final_acceptance_blocked_with_traceable_reasons_and_no_dm_registration(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-fixture")
    write_l1_report(fixture_root / "l1_report.json", records.load_by_path(p1_path))
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_path, "l1_report", fixture_root / "l1_report.json")
    write_l3_report(fixture_root / "l3_report.json", records.load_by_path(p1_path))
    attacher.attach(p1_path, "l3_report", fixture_root / "l3_report.json")
    write_l2_report(fixture_root / "l2_report.json", records.load_by_path(p1_path))
    attacher.attach(p1_path, "l2_report", fixture_root / "l2_report.json")
    judge = FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger)

    blocked_entry, _blocked_path = judge.conclude(p1_path)  # L1 fail + L2/L3 pass -> blocked

    assert blocked_entry["verdict"] == "blocked"
    assert blocked_entry["blocked_reasons"]
    assert any(reason.startswith("L1 FID") for reason in blocked_entry["blocked_reasons"])  # no offset by passing layers
    assert DmSourceLedger(records.root()).current() is None  # a blocked conclusion registers no DM source
    with pytest.raises(ContractViolationError, match="immutable"):
        judge.conclude(p1_path)

    # a hand-edited flip must not survive verification
    flipped_path = FinalAcceptanceJudge.verdict_path_for(p1_path)
    flipped = json.loads(flipped_path.read_text())
    flipped["verdict"] = "pass"
    flipped["blocked_reasons"] = []
    flipped_path.write_text(json.dumps(flipped))
    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p1_path), record_path=p1_path)

    assert any("non-compensatory AND" in failure for failure in failures)


def test_l2_undecided_blocks_final_acceptance(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_undecided_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-undecided")
    write_l1_report(fixture_root / "l1_pass_undecided_run.json", records.load_by_path(p1_undecided_path), passing=True)
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_undecided_path, "l1_report", fixture_root / "l1_pass_undecided_run.json")
    write_l2_report(fixture_root / "l2_undecided_report.json", records.load_by_path(p1_undecided_path), undecided_challenges=("SSA",))
    attacher.attach(p1_undecided_path, "l2_report", fixture_root / "l2_undecided_report.json")
    write_l3_report(fixture_root / "l3_undecided_run_report.json", records.load_by_path(p1_undecided_path))
    attacher.attach(p1_undecided_path, "l3_report", fixture_root / "l3_undecided_run_report.json")

    undecided_entry, _ = FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger).conclude(p1_undecided_path)

    assert undecided_entry["verdict"] == "blocked"
    assert any("L2 SSA: undecided" in r for r in undecided_entry["blocked_reasons"])
    assert DmSourceLedger(records.root()).current() is None


def test_final_acceptance_pass_registers_dm_source(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_final_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-final")
    p1_final_record = records.load_by_path(p1_final_path)
    write_l1_report(fixture_root / "l1_pass_report.json", p1_final_record, passing=True)
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_final_path, "l1_report", fixture_root / "l1_pass_report.json")
    write_l2_report(fixture_root / "l2_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l2_report", fixture_root / "l2_pass_report.json")
    write_l3_report(fixture_root / "l3_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l3_report", fixture_root / "l3_pass_report.json")
    judge = FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger)

    pass_entry, _ = judge.conclude(p1_final_path)

    assert pass_entry["verdict"] == "pass"
    assert pass_entry["blocked_reasons"] == []
    assert pass_entry["dm_source_registered"] is True
    source = DmSourceLedger(records.root()).current()
    assert source is not None
    assert source["run_id"] == "p1-final"
    assert source["checkpoint"]["sha256"] == p1_final_record["selection"]["checkpoint"]["sha256"]
    with pytest.raises(ContractViolationError, match="immutable"):
        judge.conclude(p1_final_path)
    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p1_final_path), record_path=p1_final_path)
    assert failures == []


@pytest.fixture()
def registered_source(tmp_path, fixture_root, fingerprinter):
    """A record store whose DM source is a concluded passing P1 candidate."""
    records = store_at(tmp_path, "records")
    p1_final_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-final")
    p1_final_record = records.load_by_path(p1_final_path)
    write_l1_report(fixture_root / "l1_pass_report.json", p1_final_record, passing=True)
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_final_path, "l1_report", fixture_root / "l1_pass_report.json")
    write_l2_report(fixture_root / "l2_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l2_report", fixture_root / "l2_pass_report.json")
    write_l3_report(fixture_root / "l3_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l3_report", fixture_root / "l3_pass_report.json")
    entry, _ = FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger).conclude(p1_final_path)
    assert entry["verdict"] == "pass"
    open_store = store_at(tmp_path, "records_open")
    open_path = initializer(open_store, fingerprinter, fixture_root).init(
        "P1",
        "p1-open",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        fixture_root / "base_ckpt.pt",
        None,
        None,
    )
    return records, p1_final_path, open_store, open_path


def test_phase_chain_gates(tmp_path, fixture_root, fingerprinter, registered_source):
    records, p1_final_path, open_store, open_path = registered_source

    # P2 with a replay list
    with pytest.raises(ContractViolationError):
        initializer(records, fingerprinter, fixture_root).init(
            "P2",
            "p2-replay",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json"), ("replay", fixture_root / "lists/replay.json")],
            None,
            p1_final_path,
            None,
        )
    # P2 pinned to an open P1
    with pytest.raises(ContractViolationError):
        initializer(open_store, fingerprinter, fixture_root).init(
            "P2",
            "p2-early",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            None,
            open_path,
            None,
        )
    # a passing-but-unregistered P1 is not the DM source
    other_p1 = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-other")
    with pytest.raises(ContractViolationError, match="registered DM source"):
        initializer(records, fingerprinter, fixture_root).init(
            "P2",
            "p2-offsource",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            None,
            other_p1,
            None,
        )

    p2_path = initializer(records, fingerprinter, fixture_root).init(
        "P2",
        "p2-fixture",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        None,
        p1_final_path,
        None,
    )
    # P2 fold-split combined list (train+dev under one train label) opens and
    # verifies; P1 must reject the same list (spec #51 decision 8).
    p2_combined_path = initializer(records, fingerprinter, fixture_root).init(
        "P2",
        "p2-combined-split",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/combined_sided.json")],
        None,
        p1_final_path,
        None,
    )
    assert RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p2_combined_path), record_path=p2_combined_path) == []
    with pytest.raises(ContractViolationError):
        initializer(store_at(tmp_path, "records_comb_reject"), fingerprinter, fixture_root).init(
            "P1",
            "p1-combined-split",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/combined_sided.json")],
            fixture_root / "base_ckpt.pt",
            None,
            None,
        )
    SelectionRecorder(records, fingerprinter).select(
        p2_path, fixture_root / "candidate.pt", "dev light acceptance", [fixture_root / "dev_metrics.json"], None
    )
    CandidateFreezer(records, fingerprinter).freeze(p2_path, fixture_root / "samples.json")
    # P3 warm-started from a P2 run
    with pytest.raises(ContractViolationError, match="P2 ControlNet"):
        initializer(records, fingerprinter, fixture_root).init(
            "P3",
            "p3-warm",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json"), ("inference", fixture_root / "infer_config.json")],
            [("train", fixture_root / "lists/train.json")],
            None,
            p2_path,
            None,
        )
    # P1 with an upstream run
    with pytest.raises(ContractViolationError):
        initializer(open_store, fingerprinter, fixture_root).init(
            "P1",
            "p1-with-upstream",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            fixture_root / "base_ckpt.pt",
            other_p1,
            None,
        )
    # P3 positive path: pins the same registered P1-DM (independent init), full record verifies.
    p3_path = initializer(records, fingerprinter, fixture_root).init(
        "P3",
        "p3-fixture",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json"), ("inference", fixture_root / "infer_config.json")],
        [("train", fixture_root / "lists/train.json")],
        None,
        p1_final_path,
        None,
    )
    SelectionRecorder(records, fingerprinter).select(
        p3_path, fixture_root / "controlnet_candidate.pt", "dev light acceptance", [fixture_root / "dev_metrics.json"], None
    )
    CandidateFreezer(records, fingerprinter).freeze(p3_path, fixture_root / "samples.json")
    write_l1_report(fixture_root / "p3_l1_report.json", records.load_by_path(p3_path))
    ReportAttacher(records, fingerprinter).attach(p3_path, "l1_report", fixture_root / "p3_l1_report.json")

    assert RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p2_path), record_path=p2_path) == []
    assert RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p3_path), record_path=p3_path) == []


def test_stage0_baseline_is_the_comparison_floor(tmp_path, fixture_root, fingerprinter, registered_source):
    records, p1_final_path, _open_store, _open_path = registered_source
    stage0_initializer = initializer(records, fingerprinter, fixture_root)

    with pytest.raises(ContractViolationError):
        stage0_initializer.init(
            "P3",
            "p3-stage0-bogus",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            None,
            p1_final_path,
            None,
            variant="bogus-variant",
        )
    with pytest.raises(ContractViolationError, match="inference"):
        stage0_initializer.init(
            "P3",
            "p3-stage0-noinfer",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            None,
            p1_final_path,
            None,
            variant="stage0-baseline",
        )
    with pytest.raises(ContractViolationError, match="P3-only"):
        stage0_initializer.init(
            "P1",
            "p1-stage0",
            fixture_root / "phase_manifest.json",
            [("env", fixture_root / "env_config.json")],
            [("train", fixture_root / "lists/train.json")],
            fixture_root / "base_ckpt.pt",
            None,
            None,
            variant="stage0-baseline",
        )

    stage0_path = stage0_initializer.init(
        "P3",
        "p3-stage0",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json"), ("inference", fixture_root / "infer_config.json")],
        [("train", fixture_root / "lists/train.json")],
        None,
        p1_final_path,
        None,
        variant="stage0-baseline",
    )
    stage0_record = records.load_by_path(stage0_path)
    assert stage0_record.get("variant") == "stage0-baseline"
    with pytest.raises(ContractViolationError, match="upstream P1-DM checkpoint"):
        SelectionRecorder(records, fingerprinter).select(
            stage0_path, fixture_root / "base_ckpt.pt", "zero-training baseline", [fixture_root / "dev_metrics.json"], None
        )
    upstream_ckpt = Path(stage0_record["upstream"]["checkpoint"]["path"])
    SelectionRecorder(records, fingerprinter).select(
        stage0_path, upstream_ckpt, "zero-training stage-0 baseline: DM is the upstream P1-DM selection", [fixture_root / "dev_metrics.json"], None
    )
    CandidateFreezer(records, fingerprinter).freeze(stage0_path, fixture_root / "samples.json")
    assert RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(stage0_path), record_path=stage0_path) == []
    with pytest.raises(ContractViolationError, match="comparison floor"):
        ReportAttacher(records, fingerprinter).attach(stage0_path, "l1_report", fixture_root / "l1_report.json")
    with pytest.raises(ContractViolationError, match="never takes final acceptance"):
        FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger).conclude(stage0_path)

    # A P1 record carrying the P3-only variant marker must fail verification.
    p1_frozen = records.load_by_path(next(path for path in records.all_record_paths() if "p1-final" in str(path)))
    tainted = dict(p1_frozen)
    tainted["variant"] = "stage0-baseline"
    tainted_path = Path(stage0_path).parent.parent / "runs" / "p1-tainted-variant" / "run.json"
    tainted_path.parent.mkdir(parents=True, exist_ok=True)
    tainted_path.write_text(json.dumps(tainted))
    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(tainted_path), record_path=tainted_path)
    assert any("variant" in failure for failure in failures)

    # A hand-edited stage-0 record with a formal report attached must fail verification.
    tainted_stage0 = json.loads(Path(stage0_path).read_text())  # re-read: the frozen record carries the selection
    tainted_stage0["attachments"] = [{"kind": "l1_report", "path": str(fixture_root / "l1_report.json"), "sha256": "0" * 64}]
    tainted_stage0_path = Path(stage0_path).parent.parent / "runs" / "p3-stage0-tainted" / "run.json"
    tainted_stage0_path.parent.mkdir(parents=True, exist_ok=True)
    tainted_stage0_path.write_text(json.dumps(tainted_stage0))
    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(tainted_stage0_path), record_path=tainted_stage0_path)
    assert any("stage-0" in failure for failure in failures)


def test_dm_retrain_supersedes_source_and_mismatches_old_bypasses(tmp_path, fixture_root, fingerprinter, registered_source):
    records, p1_final_path, _open_store, _open_path = registered_source
    # the old bypass is pinned while p1-final is still the registered DM source
    p2_stale = initializer(records, fingerprinter, fixture_root).init(
        "P2",
        "p2-stale",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        None,
        p1_final_path,
        None,
    )
    # a later passing P1 on a different checkpoint supersedes the source
    retrained_ckpt = fixture_root / "candidate_retrained.pt"
    retrained_ckpt.write_bytes(b"candidate-retrained-fixture")
    p1_retrained_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-retrained", checkpoint=retrained_ckpt)
    retrained_record = records.load_by_path(p1_retrained_path)
    write_l1_report(fixture_root / "l1_retrained_report.json", retrained_record, passing=True)
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_retrained_path, "l1_report", fixture_root / "l1_retrained_report.json")
    write_l2_report(fixture_root / "l2_retrained_report.json", retrained_record)
    attacher.attach(p1_retrained_path, "l2_report", fixture_root / "l2_retrained_report.json")
    write_l3_report(fixture_root / "l3_retrained_report.json", retrained_record)
    attacher.attach(p1_retrained_path, "l3_report", fixture_root / "l3_retrained_report.json")

    retrained_entry, _ = FinalAcceptanceJudge(records, fingerprinter, DmSourceLedger).conclude(p1_retrained_path)

    assert retrained_entry["verdict"] == "pass"
    superseded = DmSourceLedger(records.root()).current()
    assert superseded["run_id"] == "p1-retrained"
    assert superseded["superseded_run_id"] == "p1-final"

    # the old bypass now fails verification explicitly (a retrained DM invalidates it)
    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p2_stale), record_path=p2_stale)
    assert any("DM was retrained" in failure for failure in failures)
    assert RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p1_retrained_path), record_path=p1_retrained_path) == []


def test_tamper_detection_flags_a_changed_checkpoint(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-fixture")
    tampered = fixture_root / "candidate.pt"
    original = tampered.read_bytes()
    tampered.write_bytes(b"tampered")
    try:
        failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p1_path), record_path=p1_path)
    finally:
        tampered.write_bytes(original)
    assert any("sha256 changed" in f or "missing on disk" in f for f in failures)


def test_storage_guard_flags_a_record_inside_a_git_work_tree(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-fixture")
    fake_repo = tmp_path / "fakerepo"
    (fake_repo / ".git").mkdir(parents=True, exist_ok=True)
    repo_records = fake_repo / "records" / "runs" / "p1-fixture" / "run.json"
    repo_records.parent.mkdir(parents=True, exist_ok=True)
    repo_records.write_text(p1_path.read_text())

    failures = RunVerifier(fingerprinter, DmSourceLedger).verify(records.load_by_path(p1_path), record_path=repo_records)

    assert any("git work tree" in f for f in failures)


# ------------------------------------------------------- the injected DM-source port (#271)
#
# The three ledger-gated use cases (derive_upstream / conclude / verify) ride the
# injected ``DmSourceLedger`` port; the json adapter is the composition root's
# choice, not theirs. These gates drive the port with an in-memory fake -- no
# dm_source.json is touched -- and pin the boundary translation: a domain ledger
# violation surfaces as the contract's own violation type.


class FakeDmSourceLedger:
    """In-memory DmSourceLedger port: calls recorded, rejection/failure behavior programmable."""

    def __init__(self, check_upstream_error=None, register_error=None, record_failures=()):
        self._check_upstream_error = check_upstream_error
        self._register_error = register_error
        self._record_failures = list(record_failures)
        self.check_upstream_calls = []
        self.register_calls = []
        self.check_record_calls = []

    def check_upstream(self, upstream_run_id, checkpoint):
        self.check_upstream_calls.append((upstream_run_id, checkpoint))
        if self._check_upstream_error is not None:
            raise self._check_upstream_error

    def register(self, record, run_record_path):
        self.register_calls.append((record["run_id"], run_record_path))
        if self._register_error is not None:
            raise self._register_error
        return {"schema": "brats-dm-source/1", "run_id": record["run_id"]}

    def check_record(self, record):
        self.check_record_calls.append(record)
        return list(self._record_failures)


class FakeLedgerFactory:
    """The ``(record_root) -> ledger`` injection: every root it is asked for is recorded."""

    def __init__(self, **ledger_behavior):
        self._ledger_behavior = ledger_behavior
        self.roots = []
        self.ledgers = []

    def __call__(self, record_root):
        self.roots.append(Path(record_root))
        ledger = FakeDmSourceLedger(**self._ledger_behavior)
        self.ledgers.append(ledger)
        return ledger


def _write_frozen_p1_record(root, run_id, checkpoint_path, checkpoint_sha):
    """A minimal frozen P1 run.json: enough shape for load_by_path, derive_upstream, and a chain recursion."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text('{"challenges": {}}')
    record = {
        "schema": SCHEMA,
        "run_id": run_id,
        "phase": "P1",
        "variant": None,
        "status": STATUS_FROZEN,
        "manifest": {"path": str(root / "manifest.json"), "sha256": "0" * 64},
        "configs": [],
        "data_lists": [],
        "base_ckpt": None,
        "upstream": None,
        "selection": {"checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha, "epoch": 5}, "evidence": []},
    }
    run_path = root / "runs" / run_id / "run.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(record))
    return run_path


def test_derive_upstream_pins_the_registered_source_through_the_injected_port(tmp_path, fingerprinter):
    store = RunRecordStore(tmp_path / "records")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-fixture")
    checkpoint_sha = fingerprinter.file_sha256(checkpoint)
    upstream_path = _write_frozen_p1_record(store.root(), "p1-src", checkpoint, checkpoint_sha)
    fake_factory = FakeLedgerFactory()

    upstream = RunInitializer(store, fingerprinter, ManifestSides({"challenges": {}}), fake_factory).derive_upstream(upstream_path)

    assert upstream["run_id"] == "p1-src"
    assert fake_factory.roots == [store.root()]  # the ledger is drawn from the store's record root
    assert fake_factory.ledgers[0].check_upstream_calls == [("p1-src", WeightsRef(sha256=checkpoint_sha))]


def test_derive_upstream_translates_a_ledger_violation_into_the_contract_violation(tmp_path, fingerprinter):
    store = RunRecordStore(tmp_path / "records")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-fixture")
    upstream_path = _write_frozen_p1_record(store.root(), "p1-src", checkpoint, fingerprinter.file_sha256(checkpoint))
    fake_factory = FakeLedgerFactory(check_upstream_error=DmSourceViolationError("no P1 candidate has passed final acceptance yet"))

    with pytest.raises(ContractViolationError, match="no P1 candidate has passed final acceptance yet"):
        RunInitializer(store, fingerprinter, ManifestSides({"challenges": {}}), fake_factory).derive_upstream(upstream_path)


def test_conclude_registers_the_passing_p1_through_the_injected_port(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_final_path = open_passing_candidate(tmp_path, records, fingerprinter, fixture_root, "p1-final")
    p1_final_record = records.load_by_path(p1_final_path)
    write_l1_report(fixture_root / "l1_pass_report.json", p1_final_record, passing=True)
    attacher = ReportAttacher(records, fingerprinter)
    attacher.attach(p1_final_path, "l1_report", fixture_root / "l1_pass_report.json")
    write_l2_report(fixture_root / "l2_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l2_report", fixture_root / "l2_pass_report.json")
    write_l3_report(fixture_root / "l3_pass_report.json", p1_final_record)
    attacher.attach(p1_final_path, "l3_report", fixture_root / "l3_pass_report.json")
    fake_factory = FakeLedgerFactory()

    entry, _ = FinalAcceptanceJudge(records, fingerprinter, fake_factory).conclude(p1_final_path)

    assert entry["verdict"] == "pass" and entry["dm_source_registered"] is True
    assert fake_factory.ledgers[0].register_calls == [("p1-final", p1_final_path)]


def test_verify_consults_the_injected_ledger_and_flags_its_mismatches(tmp_path, fixture_root, fingerprinter):
    records = store_at(tmp_path, "records")
    p1_path = initializer(records, fingerprinter, fixture_root).init(
        "P1",
        "p1-fixture",
        fixture_root / "phase_manifest.json",
        [("env", fixture_root / "env_config.json")],
        [("train", fixture_root / "lists/train.json")],
        fixture_root / "base_ckpt.pt",
        None,
        None,
    )
    record = records.load_by_path(p1_path)
    fake_factory = FakeLedgerFactory(record_failures=["DM was retrained: this bypass is pinned to superseded DM p1-final"])

    failures = RunVerifier(fingerprinter, fake_factory).verify(record, record_path=p1_path)

    assert fake_factory.ledgers[0].check_record_calls == [record]
    assert "dm source: DM was retrained: this bypass is pinned to superseded DM p1-final" in failures


def test_verify_draws_a_ledger_per_record_root_along_the_chain(tmp_path, fixture_root, fingerprinter):
    """The chain recursion verifies the upstream record under its own record root -- the
    reason the injection is a factory over roots, not a single instance."""
    upstream_store = store_at(tmp_path, "records_a")
    bypass_store = store_at(tmp_path, "records_b")
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate-fixture")
    upstream_path = _write_frozen_p1_record(upstream_store.root(), "p1-src", checkpoint, fingerprinter.file_sha256(checkpoint))
    upstream_entry = RunInitializer(bypass_store, fingerprinter, ManifestSides({"challenges": {}}), FakeLedgerFactory()).derive_upstream(
        upstream_path
    )
    bypass_record = {
        "schema": SCHEMA,
        "run_id": "p2-bypass",
        "phase": "P2",
        "variant": None,
        "status": STATUS_OPEN,
        "created_utc": "2026-08-31T00:00:00Z",
        "frozen_utc": None,
        "manifest": {"path": str(fixture_root / "phase_manifest.json"), "sha256": "0" * 64},
        "configs": [],
        "data_lists": [],
        "base_ckpt": None,
        "upstream": upstream_entry,
        "platform": None,
        "selection": None,
        "samples": None,
        "attachments": [],
    }
    bypass_path = bypass_store.write(bypass_record)
    fake_factory = FakeLedgerFactory()

    RunVerifier(fingerprinter, fake_factory).verify(bypass_store.load_by_path(bypass_path), record_path=bypass_path)

    assert fake_factory.roots == [bypass_store.root(), upstream_store.root()]
    assert fake_factory.ledgers[0].check_record_calls == [bypass_store.load_by_path(bypass_path)]
    assert fake_factory.ledgers[1].check_record_calls[0]["run_id"] == "p1-src"
