"""Issue #35 收尾验证：逐条重算受控 run 的全部工件 hash 并核验完成证据。

对每个子挑战读取既有 run-manifest.json 与 completion-audit.json，重新执行
raw / preprocessed / trainer 契约验证并重算所有记录过的 hash，与 manifest
逐条比对；再核验 checkpoint 元数据与日志 epoch 覆盖。只读、拒绝覆盖，
verdict 写入同一 audit 目录的 closing-verification.json。
"""

import argparse
import importlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from ctmr.application.acceptance.distribution.instrument_training import (
    PERSISTENT_ROOT,
    TRAINER_CLASS,
    ArtifactHasher,
    ChallengeRegistry,
    DatasetContract,
    TrainerContract,
    TrainingProtocol,
)
from ctmr.domain.checkpoints import InstrumentCheckpointReader
from ctmr.wiring.distribution import instrument_checkpoint_reader


class ClosingVerifier:
    """逐条重算一个受控 run 的 manifest 证据。

    checkpoint 元数据的读取经注入的 ``InstrumentCheckpointReader`` 端口
    (ADR-0019 §3, #275):weights_only 白名单作用域是适配器的保证,本类不触
    torch 序列化状态。
    """

    def __init__(
        self,
        challenge: str,
        raw_root: Path,
        preprocessed_root: Path,
        results_root: Path,
        audit_dir: Path,
        ssa_exception: bool,
        *,
        checkpoint_reader: InstrumentCheckpointReader,
    ):
        self.spec = ChallengeRegistry().get(challenge)
        self.protocol = TrainingProtocol(self.spec, ssa_exception)
        self.dataset = DatasetContract(self.spec, raw_root, preprocessed_root)
        self.results_root = results_root
        self.audit_dir = audit_dir
        self.hasher = ArtifactHasher()
        self.checkpoint_reader = checkpoint_reader
        self.checks: list[dict] = []

    def verify(self) -> dict:
        if not self.audit_dir.resolve().is_relative_to(PERSISTENT_ROOT):
            raise ValueError(f"audit dir must remain under {PERSISTENT_ROOT}")
        manifest = json.loads((self.audit_dir / "run-manifest.json").read_text())
        completion = json.loads((self.audit_dir / "completion-audit.json").read_text())

        self._check_protocol(manifest)
        self._check_raw(manifest)
        self._check_preprocessed(manifest)
        self._check_trainer(manifest)
        self._check_environment(manifest)
        self._check_completion(manifest, completion)

        verdict = {
            "schema_version": 1,
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "issue": 35,
            "challenge": self.spec.code,
            "dataset": self.spec.dataset_name,
            "fold": 0,
            "all_passed": all(item["passed"] for item in self.checks),
            "checks": self.checks,
        }
        destination = self.audit_dir / "closing-verification.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite verdict: {destination}")
        destination.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        return verdict

    def _record(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})

    def _check_protocol(self, manifest: dict) -> None:
        protocol = manifest["protocol"]
        expected = {
            "plans_identifier": self.protocol.plans_identifier(),
            "configuration": self.protocol.configuration(),
        }
        ok = all(protocol.get(k) == v for k, v in expected.items())
        self._record("protocol", ok, f"{expected} vs recorded {dict((k, protocol.get(k)) for k in expected)}")

    def _check_raw(self, manifest: dict) -> None:
        live = self.dataset.verify_raw()
        for key in ("dataset_json", "splits_final_json"):
            ok = live[key]["sha256"] == manifest["raw_contract"][key]["sha256"]
            self._record(f"raw.{key}", ok, f"live {live[key]['sha256'][:16]}… vs manifest {manifest['raw_contract'][key]['sha256'][:16]}…")
        counts_ok = (
            live["training_side_cases"] == self.spec.training_cases
            and live["fold_0_train_cases"] == self.spec.fold_train_cases
            and live["fold_0_validation_cases"] == self.spec.fold_val_cases
        )
        self._record("raw.counts", counts_ok, f"{self.spec.training_cases}/{self.spec.fold_train_cases}/{self.spec.fold_val_cases}")

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

    def _check_environment(self, manifest: dict) -> None:
        recorded = manifest["environment"]
        comparisons = {
            "monai": importlib.metadata.version("monai"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "torch": torch.__version__,
        }
        for name, value in comparisons.items():
            ok = recorded.get(name) == value
            self._record(f"environment.{name}", ok, f"live {value} vs manifest {recorded.get(name)}")

    def _check_completion(self, manifest: dict, completion: dict) -> None:
        fold_dir = (
            self.results_root
            / self.spec.dataset_name
            / f"{TRAINER_CLASS}__{self.protocol.plans_identifier()}__{self.protocol.configuration()}"
            / "fold_0"
        )
        checkpoint = fold_dir / "checkpoint_final.pth"
        live_hash = self.hasher.sha256(checkpoint)
        ok = live_hash == completion["checkpoint_final"]["sha256"]
        self._record("checkpoint_final.hash", ok, f"live {live_hash[:16]}… vs completion {completion['checkpoint_final']['sha256'][:16]}…")

        data = self.checkpoint_reader.read(checkpoint)
        ok = data.get("current_epoch") == 250
        self._record("checkpoint_final.current_epoch", ok, str(data.get("current_epoch")))
        ok = data.get("trainer_name") == TRAINER_CLASS
        self._record("checkpoint_final.trainer_name", ok, str(data.get("trainer_name")))

        texts = [log.read_text(errors="replace") for log in sorted(fold_dir.glob("training_log_*.txt"))]
        combined = "\n".join(texts)
        ok = "Epoch 249" in combined
        self._record("training_logs.epoch_coverage", ok, "logs contain Epoch 249")

        recorded_logs = completion["training_logs"]
        for log in sorted(fold_dir.glob("training_log_*.txt")):
            if log.name in recorded_logs:
                ok = self.hasher.sha256(log) == recorded_logs[log.name]["sha256"]
                self._record(f"training_logs.{log.name}", ok, "hash matches completion audit")


def platform_version() -> str:
    import platform

    return platform.python_version()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", choices=("GLI", "SSA", "MEN", "METS", "PED"), required=True)
    parser.add_argument("--raw-root", type=Path, default=PERSISTENT_ROOT / "brats2023_nnunet")
    parser.add_argument("--preprocessed-root", type=Path, default=PERSISTENT_ROOT / "nnUNet_preprocessed")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--ssa-batch16-exception", action="store_true")
    args = parser.parse_args()
    # 进程级跨层收口(#275):进程入口经组合根装配函数取得真实 reader 适配器;
    # 库路径只认注入的端口,适配器知识唯一定居于 ctmr.wiring(ADR-0019 §2)。
    verdict = ClosingVerifier(
        args.challenge,
        args.raw_root,
        args.preprocessed_root,
        args.results_root,
        args.audit_dir,
        args.ssa_batch16_exception,
        checkpoint_reader=instrument_checkpoint_reader(),
    ).verify()
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
