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

"""The quantitative acceptance chain, observed as pytest (#141).

The resident ``L1SelfTest`` of ``scripts/brats_l1_quantitative.py`` became
this file when the FID/paired-error chain moved into the quantitative package
(ADR-0015 §6: ``selftest`` subcommands die with the script move; assertion
logic turns into real test functions). Every assertion is the deterministic
statistical behavior on a synthetic fixture: the FID location shift, the
PCG64-seeded case bootstrap, the pinned MR [0, 1] intensity protocol, the P3
paired gate with the t1n->t1c exception, the frozen-candidate binding, the
relative-FID threshold rule, and the controlled-storage writer.

Torch-marked tier (ADR-0015 §6): runs for real in the CI torch tier
(light sci stack: numpy / nibabel / scikit-image).
"""

import json

import nibabel as nib
import numpy as np
import pytest

from ctmr.application.acceptance.quantitative.evaluate import L1EvaluationCommand
from ctmr.application.acceptance.quantitative.evaluate import main as evaluate_main
from ctmr.application.acceptance.quantitative.evidence import (
    MR_PREPROCESSING,
    ControlledJsonReader,
    FeatureManifestReader,
    FrozenRunRecordReader,
    L1ReportWriter,
)
from ctmr.application.acceptance.quantitative.fid import (
    BootstrapProtocol,
    CaseFeatureBootstrap,
    FeatureRecord,
    FidScoreCalculator,
    L1QuantitativeError,
)
from ctmr.application.acceptance.quantitative.paired import P3DirectionAssessor, P3PairObservation
from ctmr.application.acceptance.quantitative.report import L1ReportProducer
from ctmr.domain.intensity_protocol import MRIntensityNormalizer

pytestmark = pytest.mark.torch

FIXTURE_PROTOCOL = BootstrapProtocol(resamples=32, seed=20260821)


def test_fid_location_shift():
    real_features = np.array([[0.0], [2.0]])
    generated_features = np.array([[1.0], [3.0]])

    fid = FidScoreCalculator().score(real_features, generated_features)

    assert np.isclose(fid, 1.0)


def test_case_bootstrap_is_deterministic():
    real_cases = {"case-a": np.array([[0.0]]), "case-b": np.array([[2.0]])}
    generated_cases = {"case-a": np.array([[1.0]]), "case-b": np.array([[3.0]])}

    first = CaseFeatureBootstrap(FidScoreCalculator(), FIXTURE_PROTOCOL).evaluate(real_cases, generated_cases)
    second = CaseFeatureBootstrap(FidScoreCalculator(), FIXTURE_PROTOCOL).evaluate(real_cases, generated_cases)

    assert np.isclose(first.point, 1.0)
    assert first == second  # deterministic for a fixed PCG64 seed


def test_mr_normalizer_clips_metric_range():
    volume = np.linspace(0.0, 100.0, 7**3).reshape((7, 7, 7))

    normalized = MRIntensityNormalizer().normalize(volume, "test")

    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_p3_pair_gate_and_t1n_t1c_exception():
    reference = np.linspace(0.0, 1.0, 7**3).reshape((7, 7, 7))
    assessor = P3DirectionAssessor(FIXTURE_PROTOCOL)
    informative = assessor.assess(
        [P3PairObservation("GLI", f"case-{index}", "t2w", "t1n", reference, reference + 0.2, reference + 0.01) for index in range(2)]
    )
    exceptional = assessor.assess(
        [P3PairObservation("GLI", f"case-{index}", "t1n", "t1c", reference, reference + 0.2, reference + 0.01) for index in range(2)]
    )

    assert informative.verdict == "pass"
    assert exceptional.verdict == "not_applicable_known_unobservable"


def test_report_binding_and_relative_fid_gate():
    run = {
        "run_id": "p1-fixture",
        "phase": "P1",
        "status": "frozen",
        "manifest": {"sha256": "manifest-sha"},
        "selection": {"checkpoint": {"sha256": "candidate-sha"}},
        "samples": {"sha256": "samples-sha"},
    }
    records = []
    values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        for cohort, cohort_values in values.items():
            for plane in ("xy", "yz", "zx"):
                for index, value in enumerate(cohort_values):
                    records.append(FeatureRecord(cohort, "GLI", f"case-{index}", modality, plane, np.array([[value]])))

    report = L1ReportProducer(FIXTURE_PROTOCOL).produce(
        run,
        ("GLI",),
        records,
        [],
        {
            "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
            "mr_preprocessing": MR_PREPROCESSING,
        },
    )

    assert report["binding"]["candidate_checkpoint_sha256"] == "candidate-sha"
    assert report["summary"]["verdict"] == "pass"
    first_fid = report["fid_results"][0]
    bootstrap_median = first_fid["train_vs_holdout_baseline"].get("mean_bootstrap_median")
    assert bootstrap_median is not None
    assert np.isclose(first_fid["threshold"], 2.5 * bootstrap_median)  # the relative gate uses the real bootstrap median


def _write_evaluate_fixture(root):
    """A frozen P1 run plus a full controlled brats-l1-features/1 manifest."""
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    phase_manifest_path = root / "phase_manifest.json"
    phase_manifest_path.write_text(json.dumps({"challenges": {"GLI": {"cases": {}}}}))
    run_path = root / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "run_id": "p1-cli-fixture",
                "phase": "P1",
                "status": "frozen",
                "manifest": {"path": str(phase_manifest_path), "sha256": "manifest-sha"},
                "selection": {"checkpoint": {"sha256": "candidate-sha"}},
                "samples": {"sha256": "samples-sha"},
            }
        )
    )
    rows = []
    values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        for cohort, cohort_values in values.items():
            for plane in ("xy", "yz", "zx"):
                for index, value in enumerate(cohort_values):
                    feature_path = features_dir / f"{cohort}-{modality}-{plane}-{index}.npy"
                    np.save(feature_path, np.array([[value]]))
                    rows.append(
                        {
                            "cohort": cohort,
                            "challenge": "GLI",
                            "case": f"case-{index}",
                            "target_modality": modality,
                            "plane": plane,
                            "path": str(feature_path),
                        }
                    )
    features_path = root / "features.json"
    features_path.write_text(
        json.dumps(
            {
                "schema": "brats-l1-features/1",
                "protocol": {
                    "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
                    "mr_preprocessing": "percentile_0_99.5_to_0_1_ras_1mm_zero_pad",
                },
                "records": rows,
            }
        )
    )
    return run_path, features_path


def test_evaluate_command_writes_bound_report(tmp_path, capsys):
    run_path, features_path = _write_evaluate_fixture(tmp_path)
    report_path = tmp_path / "l1_report.json"

    status = evaluate_main(
        [
            "--run",
            str(run_path),
            "--features",
            str(features_path),
            "--output",
            str(report_path),
            "--bootstrap-resamples",
            "32",
            "--seed",
            "20260821",
        ]
    )

    assert status == 0
    assert json.loads(report_path.read_text())["binding"]["run_id"] == "p1-cli-fixture"
    assert "L1 report written" in capsys.readouterr().out


def test_feature_manifest_without_fixed_preprocessing_provenance_is_rejected(tmp_path):
    _run_path, features_path = _write_evaluate_fixture(tmp_path)
    invalid_features = json.loads(features_path.read_text())
    invalid_features["protocol"].pop("mr_preprocessing")
    invalid_features_path = tmp_path / "invalid_features.json"
    invalid_features_path.write_text(json.dumps(invalid_features))

    with pytest.raises(L1QuantitativeError, match="mr_preprocessing"):
        FeatureManifestReader(ControlledJsonReader()).read(invalid_features_path)


def test_p3_evaluate_command_writes_paired_results(tmp_path):
    root = tmp_path / "p3-evaluate-command"
    features_dir = root / "features"
    volumes_dir = root / "volumes"
    features_dir.mkdir(parents=True, exist_ok=True)
    volumes_dir.mkdir(parents=True, exist_ok=True)
    phase_manifest_path = root / "phase_manifest.json"
    phase_manifest_path.write_text(json.dumps({"challenges": {"GLI": {"cases": {}}}}))
    run_path = root / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "run_id": "p3-cli-fixture",
                "phase": "P3",
                "status": "frozen",
                "manifest": {"path": str(phase_manifest_path), "sha256": "manifest-sha"},
                "selection": {"checkpoint": {"sha256": "candidate-sha"}},
                "samples": {"sha256": "samples-sha"},
            }
        )
    )
    feature_rows = []
    values = {"train": (-5.0, 5.0), "holdout": (1.0, 3.0), "generated": (1.0, 3.0)}
    for modality in ("t1n", "t1c", "t2w", "t2f"):
        for cohort, cohort_values in values.items():
            sources = (None,) if cohort != "generated" else tuple(source for source in ("t1n", "t1c", "t2w", "t2f") if source != modality)
            for source in sources:
                for plane in ("xy", "yz", "zx"):
                    for index, value in enumerate(cohort_values):
                        suffix = "" if source is None else f"-{source}"
                        feature_path = features_dir / f"{cohort}-{modality}-{plane}-{index}{suffix}.npy"
                        np.save(feature_path, np.array([[value]]))
                        feature_rows.append(
                            {
                                "cohort": cohort,
                                "challenge": "GLI",
                                "case": f"case-{index}",
                                "target_modality": modality,
                                "plane": plane,
                                "path": str(feature_path),
                                "src_modality": source,
                            }
                        )
    features_path = root / "features.json"
    features_path.write_text(
        json.dumps(
            {
                "schema": "brats-l1-features/1",
                "protocol": {
                    "feature_extractor": {"name": "radimagenet_resnet50", "weights_sha256": "f" * 64},
                    "mr_preprocessing": "percentile_0_99.5_to_0_1_ras_1mm_zero_pad",
                },
                "records": feature_rows,
            }
        )
    )
    reference = np.linspace(0.0, 1.0, 7**3).reshape((7, 7, 7))
    pair_rows = []
    for source in ("t1n", "t1c", "t2w", "t2f"):
        for target in ("t1n", "t1c", "t2w", "t2f"):
            if source == target:
                continue
            for index in range(2):
                paths = {}
                for label, volume in {
                    "reference": reference,
                    "baseline": reference**2,
                    "candidate": reference + (0.01 + index * 0.005) * np.sin(np.pi * reference),
                }.items():
                    path = volumes_dir / f"{source}-{target}-{index}-{label}.nii.gz"
                    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)
                    paths[label] = str(path)
                pair_rows.append(
                    {
                        "challenge": "GLI",
                        "case": f"case-{index}",
                        "src_modality": source,
                        "target_modality": target,
                        **paths,
                    }
                )
    pairs_path = root / "pairs.json"
    pairs_path.write_text(json.dumps({"schema": "brats-l1-pairs/1", "records": pair_rows}))
    report_path = root / "l1_report.json"

    status = evaluate_main(
        [
            "--run",
            str(run_path),
            "--features",
            str(features_path),
            "--pairs",
            str(pairs_path),
            "--output",
            str(report_path),
            "--bootstrap-resamples",
            "32",
            "--seed",
            "20260821",
        ]
    )

    assert status == 0
    report = json.loads(report_path.read_text())
    p3_results = report["p3_paired_results"]
    t1n_t1c = next(result for result in p3_results if (result["src_modality"], result["target_modality"]) == ("t1n", "t1c"))
    t1c_fid = next(result for result in report["fid_results"] if result["target_modality"] == "t1c")
    assert len(p3_results) == 12
    assert t1n_t1c["verdict"] == "not_applicable_known_unobservable"
    assert "t1n" in t1c_fid.get("generated_source_modalities", [])  # t1n->t1c stays covered by the target-t1c FID
    assert report["summary"]["verdict"] == "pass"  # all same-case source directions aggregate without becoming undecided


def test_public_report_output_is_rejected(tmp_path):
    output = tmp_path / "public-report-repo" / "l1_report.json"
    (output.parent / ".git").mkdir(parents=True, exist_ok=True)

    with pytest.raises(L1QuantitativeError, match="git work tree"):
        L1ReportWriter().write({"fixture": True}, output)


def test_evaluation_command_rejects_pairs_outside_p3(tmp_path):
    run_path, features_path = _write_evaluate_fixture(tmp_path)
    documents = ControlledJsonReader()
    command = L1EvaluationCommand(FrozenRunRecordReader(documents), FeatureManifestReader(documents), None, None)  # pair reader never reached

    with pytest.raises(L1QuantitativeError, match="--pairs applies only to P3"):
        command.evaluate(run_path, features_path, tmp_path / "pairs.json", tmp_path / "out.json", FIXTURE_PROTOCOL)
