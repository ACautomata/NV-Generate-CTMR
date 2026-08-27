"""The closing verification chain over a synthetic controlled tree (issue #140).

The issue #35 closing gate was script-resident; with the judge chain's move it
becomes a pytest fixture suite (ADR-0015 §6). The fixture builds a miniature
but structurally faithful ``Dataset50x`` tree -- dataset.json / splits_final /
preprocessed artifacts / installed-trainer shadow module / final checkpoint --
and asserts that the verifier recomputes every hash against the recorded
manifest, restores nothing, refuses to overwrite its verdict, and fails loudly
on any drift. Torch-marked tier (ADR-0015 §6).
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch  # noqa: E402

from ctmr.application.acceptance.distribution import closing as closing_module  # noqa: E402
from ctmr.application.acceptance.distribution.instrument_training import TRAINER_CLASS, TRAINER_MODULE  # noqa: E402
from ctmr.infrastructure.provisioning.trainer_250_epochs import nnUNetTrainer250Epochs as TrainerClass  # noqa: E402

pytestmark = pytest.mark.torch

CHALLENGE = "MEN"  # smallest non-SSA registry entry (700/560/140; no plans exception)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A miniature MEN instrument tree plus its closing-verifier inputs."""
    monkeypatch.setattr(closing_module, "PERSISTENT_ROOT", tmp_path)  # the audit dir must sit under the pinned root

    raw_root = tmp_path / "brats2023_nnunet"
    dataset_dir = raw_root / "Dataset503_BraTS2023MEN"
    dataset_dir.mkdir(parents=True)
    fold_train = [f"MEN-{i:03d}" for i in range(560)]
    fold_val = [f"MEN-v{i:03d}" for i in range(140)]
    (dataset_dir / "dataset.json").write_text(
        json.dumps({"numTraining": 700, "channel_names": {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}})
    )
    (dataset_dir / "splits_final.json").write_text(json.dumps([{"train": fold_train, "val": fold_val}]))

    preprocessed_root = tmp_path / "nnUNet_preprocessed" / "Dataset503_BraTS2023MEN"
    preprocessed_root.mkdir(parents=True)
    fingerprint = preprocessed_root / "dataset_fingerprint.json"
    fingerprint.write_text("{}\n")
    source_plans = {
        "plans_name": "nnUNetPlans",
        "configurations": {"3d_fullres": {"batch_size": 2, "data_identifier": "nnUNetPlans_3d_fullres"}},
    }
    plans_path = preprocessed_root / "nnUNetPlans.json"
    plans_path.write_text(json.dumps(source_plans))

    results_root = tmp_path / "results"
    fold_dir = results_root / "Dataset503_BraTS2023MEN" / f"{TRAINER_CLASS}__nnUNetPlans__3d_fullres" / "fold_0"
    fold_dir.mkdir(parents=True)
    checkpoint = fold_dir / "checkpoint_final.pth"
    with closing_module.nnunet_safe_globals():
        torch.save({"current_epoch": 250, "trainer_name": TRAINER_CLASS}, checkpoint)
    log = fold_dir / "training_log_0.txt"
    log.write_text("epoch 248\nEpoch 249 done\n")

    # the audited installed trainer is discovered through its module path; inject a
    # shadow module whose source lives on disk so inspect.getsourcefile() resolves
    trainer_source_path = tmp_path / "shadow" / "nnUNetTrainer250Epochs.py"
    trainer_source_path.parent.mkdir(parents=True)
    trainer_source_path.write_text(Path(sys.modules[TrainerClass.__module__].__file__).read_text())
    shadow = types.ModuleType(TRAINER_MODULE)
    code = compile(trainer_source_path.read_text(), str(trainer_source_path), "exec")
    shadow.__file__ = str(trainer_source_path)
    exec(code, shadow.__dict__)
    monkeypatch.setitem(sys.modules, TRAINER_MODULE, shadow)

    import importlib.metadata

    audit_dir = tmp_path / "audit-runs" / CHALLENGE / "fold_0" / "attempt-001"
    audit_dir.mkdir(parents=True)
    manifest = {
        "protocol": {"plans_identifier": "nnUNetPlans", "configuration": "3d_fullres"},
        "raw_contract": {
            "dataset_json": {"sha256": _sha(dataset_dir / "dataset.json")},
            "splits_final_json": {"sha256": _sha(dataset_dir / "splits_final.json")},
        },
        "preprocessed_artifacts": {
            "dataset_fingerprint": {"sha256": _sha(fingerprint)},
            "source_plans": {"sha256": _sha(plans_path)},
        },
        "trainer": {
            "trainer_source": {"sha256": _sha(trainer_source_path)},
            "upstream_trainer_source": {"sha256": _sha(_base_module_file())},
        },
        "environment": {
            "monai": importlib.metadata.version("monai"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "torch": torch.__version__,
        },
    }
    completion = {
        "checkpoint_final": {"sha256": _sha(checkpoint)},
        "training_logs": {log.name: {"sha256": _sha(log)}},
    }
    (audit_dir / "run-manifest.json").write_text(json.dumps(manifest, indent=2))
    (audit_dir / "completion-audit.json").write_text(json.dumps(completion, indent=2))
    return {"audit_dir": audit_dir, "results_root": results_root, "checkpoint": checkpoint, "root": tmp_path}


def _base_module_file():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    return Path(sys.modules[nnUNetTrainer.__module__].__file__)


def _verifier(tree):
    return closing_module.ClosingVerifier(
        CHALLENGE,
        tree["root"] / "brats2023_nnunet",
        tree["root"] / "nnUNet_preprocessed",
        tree["results_root"],
        tree["audit_dir"],
        False,
    )


def test_closing_verifier_recomputes_every_hash_and_passes(tree):
    verdict = _verifier(tree).verify()
    assert verdict["all_passed"] is True
    names = [check["check"] for check in verdict["checks"]]
    for expected in (
        "protocol",
        "raw.dataset_json",
        "raw.splits_final_json",
        "trainer.recipe",
        "environment.torch",
        "checkpoint_final.hash",
        "training_logs.epoch_coverage",
    ):
        assert expected in names
    destination = tree["audit_dir"] / "closing-verification.json"
    assert json.loads(destination.read_text())["all_passed"] is True


def test_closing_verdict_is_write_once(tree):
    _verifier(tree).verify()
    with pytest.raises(FileExistsError):
        _verifier(tree).verify()


def test_drifted_checkpoint_hash_fails_the_gate(tree):
    with closing_module.nnunet_safe_globals():
        torch.save({"current_epoch": 249, "trainer_name": TRAINER_CLASS}, tree["checkpoint"])  # tampered bytes
    verdict = _verifier(tree).verify()
    checkpoint_checks = [c for c in verdict["checks"] if c["check"].startswith("checkpoint_final")]
    assert any(not c["passed"] for c in checkpoint_checks)
    assert verdict["all_passed"] is False


def test_protocol_mismatch_fails_the_gate(tree):
    manifest_path = tree["audit_dir"] / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"]["configuration"] = "3d_fullres_bs16"  # the unapproved variant
    manifest_path.write_text(json.dumps(manifest, indent=2))
    verdict = _verifier(tree).verify()
    protocol_check = next(c for c in verdict["checks"] if c["check"] == "protocol")
    assert protocol_check["passed"] is False
