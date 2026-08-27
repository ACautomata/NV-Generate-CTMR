"""Convergence-gate tests for the instrument-provisioning package (ADR-0015 §2, ticket #140).

Covers the CPU-runnable behaviour of the migrated modules on synthetic
fixtures. The DCU/GPU-side contracts (DdpPreflight runtime probe, trainer
installation into a real nnunetv2 tree) stay with their integration gates;
here we pin the pure contract logic and the installation record protocol.
"""

import json

import pytest

from ctmr.infrastructure.provisioning.dataset_prep import SPLIT_ID, BraTSSplitPlanner
from ctmr.infrastructure.provisioning.ddp_preflight import GLOBAL_BATCH_SIZE, WORLD_SIZE
from ctmr.infrastructure.provisioning.plan_variant import PLANS_IDENTIFIER, PlanVariantBuilder


def test_split_planner_sort_key_is_deterministic_and_salted():
    planner = BraTSSplitPlanner(SPLIT_ID, {}, {})
    assert planner.sort_key("GLI", "BraTS-GLI-00001-000") == planner.sort_key("GLI", "BraTS-GLI-00001-000")
    assert planner.sort_key("GLI", "case-a") != planner.sort_key("SSA", "case-a")  # challenge separates keys
    assert planner.sort_key("GLI", "case-a", salt="fold0") != planner.sort_key("GLI", "case-a")  # salt separates fold_0


def test_split_planner_reports_missing_sources(tmp_path):
    planner = BraTSSplitPlanner(SPLIT_ID, {"GLI": {"train": 1, "dev": 1, "holdout": 1}}, {})
    manifest, failures = planner.build_manifest(tmp_path)
    assert manifest["challenges"] == {}
    assert failures == ["GLI: no source directory under " + str(tmp_path)]


def test_split_planner_allocates_by_sha256_stable_order(tmp_path):
    quota = {"train": 2, "dev": 1, "holdout": 1}
    source = tmp_path / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    for case in ("c4", "c3", "c2", "c1"):
        case_dir = source / case
        case_dir.mkdir(parents=True)
        (case_dir / f"{case}-seg.nii.gz").write_bytes(b"")
    manifest, failures = BraTSSplitPlanner(SPLIT_ID, {"GLI": quota}, {}).build_manifest(tmp_path)
    assert failures == []
    cases = manifest["challenges"]["GLI"]["cases"]
    all_cases = cases["train"] + cases["dev"] + cases["holdout"]
    assert all_cases == sorted(all_cases, key=lambda c: BraTSSplitPlanner(SPLIT_ID, {}, {}).sort_key("GLI", c))
    assert len(cases["train"]) == 2 and len(cases["dev"]) == 1 and len(cases["holdout"]) == 1


def test_plan_variant_builds_the_approved_batch_only_delta(monkeypatch, tmp_path):
    from ctmr.infrastructure.provisioning import plan_variant as pv

    root = tmp_path / "private"
    monkeypatch.setattr(pv, "PERSISTENT_ROOT", root)
    preprocessed = root / "nnUNet_preprocessed" / "Dataset502_BraTS2023SSA"
    preprocessed.mkdir(parents=True)
    source_plans = {
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {"batch_size": 2, "data_identifier": "nnUNetPlans_3d_fullres"},
        },
    }
    (preprocessed / "nnUNetPlans.json").write_text(json.dumps(source_plans))

    audit = PlanVariantBuilder(preprocessed / "nnUNetPlans.json", preprocessed / f"{PLANS_IDENTIFIER}.json", root / "audit.json").build()

    derived = json.loads((preprocessed / f"{PLANS_IDENTIFIER}.json").read_text())
    assert derived["configurations"]["3d_fullres_bs16"]["batch_size"] == GLOBAL_BATCH_SIZE
    assert audit["approved_delta"] == {"batch_size": GLOBAL_BATCH_SIZE}
    assert audit["local_batch_size"] == GLOBAL_BATCH_SIZE // WORLD_SIZE == 2
    # the source plans file is untouched -- byte identity of the original config set
    assert json.loads((preprocessed / "nnUNetPlans.json").read_text()) == source_plans


def test_plan_variant_rejects_a_second_configuration_field_change(monkeypatch, tmp_path):
    from ctmr.infrastructure.provisioning import plan_variant as pv

    root = tmp_path / "private"
    monkeypatch.setattr(pv, "PERSISTENT_ROOT", root)
    preprocessed = root / "pre"
    preprocessed.mkdir(parents=True)
    plans = {
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {"batch_size": 8, "patch_size": [96, 96, 96], "data_identifier": "x"},
        },
    }
    (preprocessed / "src.json").write_text(json.dumps(plans))
    builder = PlanVariantBuilder(preprocessed / "src.json", preprocessed / "derived.json", root / "audit.json")
    with pytest.raises(ValueError, match="expected SSA parent global batch_size=2"):
        builder.build()


def test_provisioning_modules_keep_the_frozen_constants():
    from ctmr.infrastructure.provisioning.ddp_preflight import DATASET_NAME, VARIANT_CONFIGURATION
    from ctmr.infrastructure.provisioning.install_trainer import TARGET_RELATIVE_PATH

    assert DATASET_NAME == "Dataset502_BraTS2023SSA"
    assert VARIANT_CONFIGURATION == "3d_fullres_bs16"
    assert TARGET_RELATIVE_PATH.as_posix() == "training/nnUNetTrainer/variants/training_length/nnUNetTrainer250Epochs.py"
