#!/usr/bin/env python3
"""Issue #37 冻结工件审计：对 L2 仪器全部受控工件做终态全量 hash 核验。

对五个正式 run（GLI/MEN/METS/PED @ 52667a3、SSA @ be683ee）重算
raw / preprocessed / trainer / checkpoint / 训练日志的全部冻结 hash 并与
run-manifest.json、completion-audit.json 逐条比对；对校准冻结件
（protocol/SHA256SUMS，3,181 条）全量重算；再对审计记录文件自身记录
hash 锚点。只读、拒绝覆盖，verdict 写入受控 freeze-audit 目录。
checkpoint 元数据（current_epoch=250）不重开 torch.load——checkpoint
hash 与完成审计一致即证明文件未变，其结论随 hash 成立。
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nnunet_l2_instrument import (  # noqa: E402
    PERSISTENT_ROOT,
    TRAINER_CLASS,
    ArtifactHasher,
    ChallengeRegistry,
    DatasetContract,
    TrainerContract,
    TrainingProtocol,
)


@dataclass(frozen=True)
class ChallengeFreezeSpec:
    """一个正式 run 的冻结审计输入位置与协议例外标记。"""

    challenge: str
    base_dir: Path
    ssa_exception: bool

    def results_root(self) -> Path:
        return self.base_dir / "results" / self.challenge / "attempt-001"

    def audit_dir(self) -> Path:
        return self.base_dir / "audit-runs" / self.challenge / "fold_0" / "attempt-001"


class TrainingArtifactAudit:
    """重算一个挑战的训练侧冻结工件并与受控记录逐条比对。"""

    def __init__(self, spec: ChallengeFreezeSpec, raw_root: Path, preprocessed_root: Path):
        self.spec = spec
        self.challenge = ChallengeRegistry().get(spec.challenge)
        self.protocol = TrainingProtocol(self.challenge, spec.ssa_exception)
        self.dataset = DatasetContract(self.challenge, raw_root, preprocessed_root)
        self.results_root = spec.results_root()
        self.audit_dir = spec.audit_dir()
        self.hasher = ArtifactHasher()
        self.checks: list[dict] = []

    def verify(self) -> dict:
        manifest = json.loads((self.audit_dir / "run-manifest.json").read_text())
        completion = json.loads((self.audit_dir / "completion-audit.json").read_text())
        self._check_protocol(manifest)
        self._check_raw(manifest)
        self._check_preprocessed(manifest)
        self._check_trainer(manifest)
        self._check_checkpoint(manifest, completion)
        self._check_training_logs(completion)
        return {
            "challenge": self.challenge.code,
            "dataset": self.challenge.dataset_name,
            "base_dir": str(self.spec.base_dir),
            "all_passed": all(item["passed"] for item in self.checks),
            "checks": self.checks,
        }

    def _record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})

    def _check_protocol(self, manifest: dict) -> None:
        protocol = manifest["protocol"]
        expected = {
            "plans_identifier": self.protocol.plans_identifier(),
            "configuration": self.protocol.configuration(),
        }
        ok = all(protocol.get(k) == v for k, v in expected.items())
        self._record("protocol", ok, f"{expected} vs recorded {protocol}")

    def _check_raw(self, manifest: dict) -> None:
        live = self.dataset.verify_raw()
        for key in ("dataset_json", "splits_final_json"):
            ok = live[key]["sha256"] == manifest["raw_contract"][key]["sha256"]
            self._record(f"raw.{key}", ok, f"live {live[key]['sha256'][:16]}… vs manifest {manifest['raw_contract'][key]['sha256'][:16]}…")
        channels_ok = manifest["raw_contract"]["channels"] == {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}
        self._record("raw.channels", channels_ok, "t1n/t1c/t2w/t2f → 0000..0003")
        counts_ok = (
            live["training_side_cases"] == self.challenge.training_cases
            and live["fold_0_train_cases"] == self.challenge.fold_train_cases
            and live["fold_0_validation_cases"] == self.challenge.fold_val_cases
        )
        self._record("raw.counts", counts_ok, f"{self.challenge.training_cases}/{self.challenge.fold_train_cases}/{self.challenge.fold_val_cases}")

    def _check_preprocessed(self, manifest: dict) -> None:
        live = self.dataset.verify_preprocessed(self.protocol)
        recorded = manifest["preprocessed_artifacts"]
        for key in ("dataset_fingerprint", "source_plans", "derived_plans"):
            if key in recorded:
                ok = live[key]["sha256"] == recorded[key]["sha256"]
                self._record(f"preprocessed.{key}", ok, f"live {live[key]['sha256'][:16]}… vs manifest {recorded[key]['sha256'][:16]}…")

    def _check_trainer(self, manifest: dict) -> None:
        live = TrainerContract().verify()
        for key in ("trainer_source", "upstream_trainer_source"):
            ok = live[key]["sha256"] == manifest["trainer"][key]["sha256"]
            self._record(f"trainer.{key}", ok, f"live {live[key]['sha256'][:16]}… vs manifest {manifest['trainer'][key]['sha256'][:16]}…")
        ok = live["num_epochs"] == 250 and live["num_iterations_per_epoch"] == 250
        self._record("trainer.recipe", ok, "250 epochs x 250 iterations")

    def _check_checkpoint(self, manifest: dict, completion: dict) -> None:
        fold_dir = (
            self.results_root
            / self.challenge.dataset_name
            / f"{TRAINER_CLASS}__{self.protocol.plans_identifier()}__{self.protocol.configuration()}"
            / "fold_0"
        )
        checkpoint = fold_dir / "checkpoint_final.pth"
        live_hash = self.hasher.sha256(checkpoint)
        ok = live_hash == completion["checkpoint_final"]["sha256"]
        self._record("checkpoint_final.hash", ok, f"live {live_hash[:16]}… vs completion {completion['checkpoint_final']['sha256'][:16]}…")
        ok = manifest["dataset"] == self.challenge.dataset_name and manifest["fold"] == 0
        self._record("checkpoint_final.run_identity", ok, f"{self.challenge.dataset_name} fold_0")

    def _check_training_logs(self, completion: dict) -> None:
        fold_dir = (
            self.results_root
            / self.challenge.dataset_name
            / f"{TRAINER_CLASS}__{self.protocol.plans_identifier()}__{self.protocol.configuration()}"
            / "fold_0"
        )
        texts = [log.read_text(errors="replace") for log in sorted(fold_dir.glob("training_log_*.txt"))]
        combined = "\n".join(texts)
        self._record("training_logs.epoch_coverage", "Epoch 249" in combined, "logs contain Epoch 249")
        recorded_logs = completion["training_logs"]
        for log in sorted(fold_dir.glob("training_log_*.txt")):
            if log.name in recorded_logs:
                ok = self.hasher.sha256(log) == recorded_logs[log.name]["sha256"]
                self._record(f"training_logs.{log.name}", ok, "hash matches completion audit")


class CalibrationFreezeAudit:
    """全量重算校准协议冻结清单（protocol/SHA256SUMS）中的每一条 hash。"""

    def __init__(self, calibration_root: Path):
        self.calibration_root = calibration_root
        self.hasher = ArtifactHasher()
        self.checks: list[dict] = []

    def verify(self) -> dict:
        sums_file = self.calibration_root / "protocol" / "SHA256SUMS"
        entries = 0
        mismatches = []
        missing = []
        for line in sums_file.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            entries += 1
            target = self.calibration_root / relative
            if not target.exists():
                missing.append(relative)
                continue
            digest = self.hasher.sha256(target)
            if digest != expected:
                mismatches.append(relative)
        passed = not mismatches and not missing
        return {
            "root": str(self.calibration_root),
            "manifest": "protocol/SHA256SUMS",
            "entries": entries,
            "all_passed": passed,
            "mismatches": mismatches,
            "missing": missing,
        }


class AuditLedgerAnchor:
    """对审计记录与版本锁文件自身记录 SHA-256 锚点（防篡改基线）。"""

    def __init__(self):
        self.hasher = ArtifactHasher()

    def verify(self, specs: list[ChallengeFreezeSpec], version_lock_dir: Path) -> dict:
        anchors: dict[str, dict] = {}
        for path in (
            *(spec.audit_dir() / name for spec in specs for name in ("run-manifest.json", "completion-audit.json", "closing-verification.json")),
            version_lock_dir / "version-lock.json",
            version_lock_dir / "trainer-install.json",
            PERSISTENT_ROOT / "l2-instrument-audit" / "ssa-bs16-v1" / "plans-variant-audit.json",
        ):
            anchors[str(path.relative_to(PERSISTENT_ROOT))] = self.hasher.record(path)
        return {"anchors": anchors}


class FreezeAudit:
    """汇总三类审计并写出终态冻结核验 verdict。"""

    def __init__(self, output_dir: Path, calibration_root: Path, version_lock_dir: Path):
        self.output_dir = output_dir
        self.calibration_root = calibration_root
        self.version_lock_dir = version_lock_dir
        self.hasher = ArtifactHasher()

    def verify(self) -> dict:
        if not self.output_dir.resolve().is_relative_to(PERSISTENT_ROOT):
            raise ValueError(f"output dir must remain under {PERSISTENT_ROOT}")
        specs = [
            ChallengeFreezeSpec("GLI", PERSISTENT_ROOT / "l2-instrument" / "52667a345ec9e1885a983bb2b8f063aa0827e997", False),
            ChallengeFreezeSpec("MEN", PERSISTENT_ROOT / "l2-instrument" / "52667a345ec9e1885a983bb2b8f063aa0827e997", False),
            ChallengeFreezeSpec("METS", PERSISTENT_ROOT / "l2-instrument" / "52667a345ec9e1885a983bb2b8f063aa0827e997", False),
            ChallengeFreezeSpec("PED", PERSISTENT_ROOT / "l2-instrument" / "52667a345ec9e1885a983bb2b8f063aa0827e997", False),
            ChallengeFreezeSpec("SSA", PERSISTENT_ROOT / "l2-instrument" / "be683eefb071022b2b62646234e4f7e469ae8dbc", True),
        ]
        raw_root = PERSISTENT_ROOT / "brats2023_nnunet"
        preprocessed_root = PERSISTENT_ROOT / "nnUNet_preprocessed"
        challenges = [TrainingArtifactAudit(spec, raw_root, preprocessed_root).verify() for spec in specs]
        calibration = CalibrationFreezeAudit(self.calibration_root).verify()
        anchors = AuditLedgerAnchor().verify(specs, self.version_lock_dir)
        verdict = {
            "schema_version": 1,
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "issue": 37,
            "all_passed": all(item["all_passed"] for item in challenges) and calibration["all_passed"],
            "challenges": challenges,
            "calibration": calibration,
            "ledger_anchors": anchors,
        }
        destination = self.output_dir / "freeze-audit.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite verdict: {destination}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="受控 freeze-audit 输出目录（verdict 拒绝覆盖）")
    parser.add_argument("--calibration-root", type=Path, required=True, help="校准受控目录（含 protocol/SHA256SUMS）")
    parser.add_argument("--version-lock-dir", type=Path, required=True, help="版本锁 version-lock.json / trainer-install.json 所在目录")
    args = parser.parse_args()
    verdict = FreezeAudit(args.output_dir, args.calibration_root, args.version_lock_dir).verify()
    summary = {
        "all_passed": verdict["all_passed"],
        "challenges": {item["challenge"]: item["all_passed"] for item in verdict["challenges"]},
        "calibration_entries": verdict["calibration"]["entries"],
        "calibration_passed": verdict["calibration"]["all_passed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
