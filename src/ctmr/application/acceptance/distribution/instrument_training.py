"""运行并审计一个 Issue #35 MONAI nnU-Net fold_0 仪器。

所有输出均约束于 ``/root/private_data``。本程序绝不向 Git 写入患者 ID、NIfTI
数据、checkpoint 或完整日志；其 JSON 记录也只保留在运行时指定的受控 audit 目录。
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# weights_only 白名单已收编到 ADR-0009 的单一 scoped 定义(ctmr.instrument.
# safeglobals.nnunet_safe_globals)：load 处 `with` 包裹，import 本模块不再
# 改全局 torch 状态。sugon 部署须连同 src/ 树一起同步(同族 shim,见
# 本包 measurement_run)。
import torch  # noqa: E402
from monai.apps.nnunet import nnUNetV2Runner  # noqa: E402

from ctmr.infrastructure.nnunet_runner import nnunet_safe_globals

PERSISTENT_ROOT = Path("/root/private_data")
TRAINER_CLASS = "nnUNetTrainer250Epochs"
TRAINER_MODULE = "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer250Epochs"
CHANNELS = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}


@dataclass(frozen=True)
class ChallengeSpec:
    """一个子挑战受控训练输入的固定、非识别元数据。"""

    code: str
    dataset_id: int
    dataset_name: str
    training_cases: int
    fold_train_cases: int
    fold_val_cases: int


@dataclass(frozen=True)
class RunConfiguration:
    """一个 run 的 operator 输入位置和不可变 provenance。"""

    challenge: str
    raw_root: Path
    preprocessed_root: Path
    results_root: Path
    work_dir: Path
    audit_dir: Path
    gpu_ids: tuple[int, ...]
    container_digest: str
    repo_commit: str
    monai_commit: str
    nnunetv2_commit: str
    nnunetv2_distribution_sha256: str
    ssa_exception: bool


class TrainingProtocol:
    """为一个子挑战选择唯一获批的训练 configuration。"""

    def __init__(self, spec: ChallengeSpec, ssa_exception: bool):
        self.spec = spec
        self.ssa_exception = ssa_exception

    def plans_identifier(self) -> str:
        return "nnUNetPlans_SSA_bs16_v1" if self.ssa_exception else "nnUNetPlans"

    def configuration(self) -> str:
        return "3d_fullres_bs16" if self.ssa_exception else "3d_fullres"

    def batch_summary(self) -> dict:
        if not self.ssa_exception:
            return {"mode": "default-plans"}
        return {"mode": "ssa-batch16-exception", "global_batch_size": 16, "local_batch_size": 2, "world_size": 8}

    def validate(self, gpu_ids: tuple[int, ...]) -> None:
        if self.ssa_exception:
            if self.spec.code != "SSA":
                raise ValueError("the SSA batch-16 exception is limited to Dataset502")
            if gpu_ids != tuple(range(8)):
                raise ValueError("the SSA batch-16 exception requires exactly GPUs 0..7")
        elif self.spec.code == "SSA":
            raise ValueError("SSA formal training requires the approved batch-16 exception")


class ArtifactHasher:
    """为受控 run artifacts 计算可重复的 SHA-256 证据。"""

    def sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def record(self, path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"required artifact not found: {path}")
        return {"sha256": self.sha256(path), "size_bytes": path.stat().st_size}


class DatasetContract:
    """验证固定的训练侧数据和 fold_0 输入契约。"""

    def __init__(self, spec: ChallengeSpec, raw_root: Path, preprocessed_root: Path):
        self.spec = spec
        self.raw_root = raw_root
        self.preprocessed_root = preprocessed_root

    def dataset_dir(self) -> Path:
        return self.raw_root / self.spec.dataset_name

    def preprocessed_dir(self) -> Path:
        return self.preprocessed_root / self.spec.dataset_name

    def verify_raw(self) -> dict:
        dataset_json = self.dataset_dir() / "dataset.json"
        splits_json = self.dataset_dir() / "splits_final.json"
        dataset = json.loads(dataset_json.read_text())
        splits = json.loads(splits_json.read_text())
        if dataset.get("numTraining") != self.spec.training_cases:
            raise ValueError(f"{self.spec.code}: unexpected training-side case count")
        if dataset.get("channel_names") != CHANNELS:
            raise ValueError(f"{self.spec.code}: channel order differs from t1n/t1c/t2w/t2f")
        if len(splits) != 1:
            raise ValueError(f"{self.spec.code}: expected one frozen fold_0 split")
        fold = splits[0]
        if len(fold.get("train", [])) != self.spec.fold_train_cases:
            raise ValueError(f"{self.spec.code}: unexpected fold_0 train count")
        if len(fold.get("val", [])) != self.spec.fold_val_cases:
            raise ValueError(f"{self.spec.code}: unexpected fold_0 validation count")
        if set(fold["train"]) & set(fold["val"]):
            raise ValueError(f"{self.spec.code}: fold_0 train/validation overlap")
        hasher = ArtifactHasher()
        return {
            "training_side_cases": self.spec.training_cases,
            "fold_0_train_cases": self.spec.fold_train_cases,
            "fold_0_validation_cases": self.spec.fold_val_cases,
            "dataset_json": hasher.record(dataset_json),
            "splits_final_json": hasher.record(splits_json),
            "channels": CHANNELS,
        }

    def verify_preprocessed(self, protocol: TrainingProtocol) -> dict:
        fingerprint = self.preprocessed_dir() / "dataset_fingerprint.json"
        source_plans = self.preprocessed_dir() / "nnUNetPlans.json"
        hasher = ArtifactHasher()
        evidence = {"dataset_fingerprint": hasher.record(fingerprint), "source_plans": hasher.record(source_plans)}
        source = json.loads(source_plans.read_text())
        if source.get("plans_name") != "nnUNetPlans":
            raise ValueError(f"{self.spec.code}: source plans identity is not nnUNetPlans")
        if protocol.ssa_exception:
            derived = self.preprocessed_dir() / f"{protocol.plans_identifier()}.json"
            evidence["derived_plans"] = hasher.record(derived)
            self._verify_ssa_batch_only_delta(source, json.loads(derived.read_text()), protocol)
        else:
            resolved = self._resolve(source["configurations"], protocol.configuration())
            if resolved.get("batch_size") is None:
                raise ValueError(f"{self.spec.code}: default 3d_fullres lacks batch_size")
        return evidence

    def _resolve(self, configurations: dict, name: str, seen: tuple[str, ...] = ()) -> dict:
        if name in seen:
            raise ValueError(f"cyclic plan inheritance: {' -> '.join((*seen, name))}")
        config = configurations.get(name)
        if not isinstance(config, dict):
            raise ValueError(f"configuration not found: {name}")
        resolved = {}
        parent = config.get("inherits_from")
        if parent is not None:
            resolved.update(self._resolve(configurations, parent, (*seen, name)))
        resolved.update({key: value for key, value in config.items() if key != "inherits_from"})
        return resolved

    def _verify_ssa_batch_only_delta(self, source: dict, derived: dict, protocol: TrainingProtocol) -> None:
        if derived.get("plans_name") != protocol.plans_identifier():
            raise ValueError("SSA derived plans identity does not match the approved exception")
        parent = self._resolve(source["configurations"], "3d_fullres")
        variant = self._resolve(derived["configurations"], protocol.configuration())
        if parent.get("batch_size") != 2 or variant.get("batch_size") != 16:
            raise ValueError("SSA batch exception must be source global batch 2 -> derived global batch 16")
        parent_without_batch = {key: value for key, value in parent.items() if key != "batch_size"}
        variant_without_batch = {key: value for key, value in variant.items() if key != "batch_size"}
        if parent_without_batch != variant_without_batch:
            raise ValueError("SSA derived plan changes fields beyond global batch size")


class TrainerContract:
    """验证已安装且可追溯 source 的 250-epoch trainer 变体。"""

    def verify(self) -> dict:
        module = importlib.import_module(TRAINER_MODULE)
        trainer = getattr(module, TRAINER_CLASS, None)
        if trainer is None:
            raise ImportError(f"trainer class unavailable: {TRAINER_CLASS}")
        source_path = Path(inspect.getsourcefile(trainer) or "")
        if not source_path.is_file():
            raise FileNotFoundError("unable to locate installed trainer source")
        source_text = source_path.read_text()
        if "self.num_epochs = 250" not in source_text:
            raise ValueError("installed trainer source does not set num_epochs = 250")
        if not issubclass(trainer, self._base_class()):
            raise TypeError("250-epoch trainer must inherit the upstream nnUNetTrainer")
        hasher = ArtifactHasher()
        base_path = Path(inspect.getsourcefile(self._base_class()) or "")
        return {
            "class_name": TRAINER_CLASS,
            "num_epochs": 250,
            "num_iterations_per_epoch": 250,
            "total_optimizer_steps": 62500,
            "trainer_source": hasher.record(source_path),
            "upstream_trainer_source": hasher.record(base_path),
        }

    def _base_class(self):
        module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        return getattr(module, "nnUNetTrainer")


class AuditLedger:
    """写入只追加的受控 run 记录，记录中不包含患者 ID。"""

    def __init__(self, audit_dir: Path):
        self.audit_dir = audit_dir

    def create(self, record: dict) -> None:
        if self.audit_dir.exists():
            raise FileExistsError(f"refusing to reuse audit directory: {self.audit_dir}")
        self.audit_dir.mkdir(parents=True)
        self._write_new("run-manifest.json", record)

    def append_completion(self, record: dict) -> None:
        self._write_new("completion-audit.json", record)

    def _write_new(self, name: str, record: dict) -> None:
        destination = self.audit_dir / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite audit record: {destination}")
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


class InstrumentRun:
    """执行一个隔离的 nnU-Net preprocessing、training 或完成校验。"""

    def __init__(self, configuration: RunConfiguration, spec: ChallengeSpec):
        self.configuration = configuration
        self.spec = spec
        self.protocol = TrainingProtocol(spec, configuration.ssa_exception)
        self.dataset = DatasetContract(spec, configuration.raw_root, configuration.preprocessed_root)
        self.ledger = AuditLedger(configuration.audit_dir)
        self.trainer = TrainerContract()

    def preprocess(self) -> dict:
        self._verify_paths()
        raw = self.dataset.verify_raw()
        runner = self._runner()
        runner.plan_and_process(
            verify_dataset_integrity=True,
            c=("3d_fullres",),
            n_proc=(8,),
        )
        plans = self.dataset.verify_preprocessed(self.protocol)
        return {"challenge": self.spec.code, "raw_contract": raw, "preprocessed_artifacts": plans}

    def train(self) -> None:
        self._verify_paths()
        self.protocol.validate(self.configuration.gpu_ids)
        raw = self.dataset.verify_raw()
        preprocessed = self.dataset.verify_preprocessed(self.protocol)
        trainer = self.trainer.verify()
        manifest = self._manifest(raw, preprocessed, trainer)
        self.ledger.create(manifest)
        self._runner().train_single_model(
            self.protocol.configuration(),
            0,
            gpu_id=self.configuration.gpu_ids,
            p=self.protocol.plans_identifier(),
        )
        # MONAI 的 run_cmd 默认 check=False：训练命令失败不会在此抛错，
        # 必须显式确认 final checkpoint 已产生。
        checkpoint = (
            self.configuration.results_root
            / self.spec.dataset_name
            / f"{TRAINER_CLASS}__{self.protocol.plans_identifier()}__{self.protocol.configuration()}"
            / "fold_0"
            / "checkpoint_final.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "training command returned without producing a final checkpoint; "
                f"the training itself failed — see its output/log. Expected: {checkpoint}"
            )
        self.ledger.append_completion(self._completion())

    def verify_completion(self) -> dict:
        self._verify_paths()
        completion = self._completion()
        print(json.dumps(completion, indent=2, sort_keys=True))
        return completion

    def _runner(self) -> nnUNetV2Runner:
        return nnUNetV2Runner(
            {
                "nnunet_raw": str(self.configuration.raw_root),
                "nnunet_preprocessed": str(self.configuration.preprocessed_root),
                "nnunet_results": str(self.configuration.results_root),
                "dataset_name_or_id": str(self.spec.dataset_id),
            },
            trainer_class_name=TRAINER_CLASS,
            export_validation_probabilities=True,
            work_dir=str(self.configuration.work_dir),
        )

    def _verify_paths(self) -> None:
        for label, path in {
            "raw_root": self.configuration.raw_root,
            "preprocessed_root": self.configuration.preprocessed_root,
            "results_root": self.configuration.results_root,
            "work_dir": self.configuration.work_dir,
            "audit_dir": self.configuration.audit_dir,
        }.items():
            if not path.resolve().is_relative_to(PERSISTENT_ROOT):
                raise ValueError(f"{label} must remain under {PERSISTENT_ROOT}")
        value = os.environ.get("nnUNet_compile", "").lower()
        if value not in {"f", "false", "0"}:
            raise ValueError("formal Issue #35 training requires nnUNet_compile=f on this DCU stack")

    def _manifest(self, raw: dict, preprocessed: dict, trainer: dict) -> dict:
        return {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "issue": 35,
            "challenge": self.spec.code,
            "dataset": self.spec.dataset_name,
            "fold": 0,
            "repo_commit": self.configuration.repo_commit,
            "container_digest": self.configuration.container_digest,
            "monai_commit": self.configuration.monai_commit,
            "nnunetv2_commit": self.configuration.nnunetv2_commit,
            "nnunetv2_distribution_sha256": self.configuration.nnunetv2_distribution_sha256,
            "protocol": {
                "plans_identifier": self.protocol.plans_identifier(),
                "configuration": self.protocol.configuration(),
                "batch": self.protocol.batch_summary(),
                "nnUNet_compile": os.environ.get("nnUNet_compile"),
            },
            "raw_contract": raw,
            "preprocessed_artifacts": preprocessed,
            "trainer": trainer,
            "environment": self._environment(),
        }

    def _environment(self) -> dict:
        return {
            "python": platform.python_version(),
            "monai": importlib.metadata.version("monai"),
            "nnunetv2": importlib.metadata.version("nnunetv2"),
            "torch_distribution": importlib.metadata.version("torch"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "visible_dcus": torch.cuda.device_count(),
        }

    def _completion(self) -> dict:
        fold_dir = (
            self.configuration.results_root
            / self.spec.dataset_name
            / f"{TRAINER_CLASS}__{self.protocol.plans_identifier()}__{self.protocol.configuration()}"
            / "fold_0"
        )
        checkpoint = fold_dir / "checkpoint_final.pth"
        hasher = ArtifactHasher()
        with nnunet_safe_globals():
            checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint_data.get("current_epoch") != 250:
            raise ValueError("checkpoint_final does not record current_epoch=250")
        if checkpoint_data.get("trainer_name") != TRAINER_CLASS:
            raise ValueError("checkpoint_final trainer does not match the formal training contract")
        logs = sorted(fold_dir.glob("training_log_*.txt"))
        if not logs:
            raise FileNotFoundError("rank-0 training log not found")
        log_evidence = {log.name: hasher.record(log) for log in logs}
        rank_zero_text = "\n".join(log.read_text(errors="replace") for log in logs)
        if "Epoch 249" not in rank_zero_text:
            raise ValueError("training logs do not prove coverage through epoch 249")
        return {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "challenge": self.spec.code,
            "dataset": self.spec.dataset_name,
            "fold": 0,
            "checkpoint_final": hasher.record(checkpoint),
            "checkpoint_current_epoch": checkpoint_data["current_epoch"],
            "checkpoint_trainer_name": checkpoint_data["trainer_name"],
            "training_logs": log_evidence,
        }


class ChallengeRegistry:
    """持有 issue 钉死的数据集 ID 和只含计数的 fold 契约。"""

    def __init__(self):
        self.specifications = {
            "GLI": ChallengeSpec("GLI", 501, "Dataset501_BraTS2023GLI", 876, 701, 175),
            "SSA": ChallengeSpec("SSA", 502, "Dataset502_BraTS2023SSA", 42, 34, 8),
            "MEN": ChallengeSpec("MEN", 503, "Dataset503_BraTS2023MEN", 700, 560, 140),
            "METS": ChallengeSpec("METS", 504, "Dataset504_BraTS2023METS", 166, 133, 33),
            "PED": ChallengeSpec("PED", 505, "Dataset505_BraTS2023PED", 68, 54, 14),
        }

    def get(self, challenge: str) -> ChallengeSpec:
        return self.specifications[challenge]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preprocess", "train", "verify-completion"))
    parser.add_argument("--challenge", choices=("GLI", "SSA", "MEN", "METS", "PED"), required=True)
    parser.add_argument("--raw-root", type=Path, default=PERSISTENT_ROOT / "brats2023_nnunet")
    parser.add_argument("--preprocessed-root", type=Path, default=PERSISTENT_ROOT / "nnUNet_preprocessed")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", required=True, help="comma-separated visible DCU ids")
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--monai-commit", required=True)
    parser.add_argument("--nnunetv2-commit", required=True)
    parser.add_argument("--nnunetv2-distribution-sha256", required=True)
    parser.add_argument("--ssa-batch16-exception", action="store_true")
    args = parser.parse_args()
    configuration = RunConfiguration(
        challenge=args.challenge,
        raw_root=args.raw_root,
        preprocessed_root=args.preprocessed_root,
        results_root=args.results_root,
        work_dir=args.work_dir,
        audit_dir=args.audit_dir,
        gpu_ids=tuple(int(item) for item in args.gpu_ids.split(",")),
        container_digest=args.container_digest,
        repo_commit=args.repo_commit,
        monai_commit=args.monai_commit,
        nnunetv2_commit=args.nnunetv2_commit,
        nnunetv2_distribution_sha256=args.nnunetv2_distribution_sha256,
        ssa_exception=args.ssa_batch16_exception,
    )
    run = InstrumentRun(configuration, ChallengeRegistry().get(args.challenge))
    if args.mode == "preprocess":
        print(json.dumps(run.preprocess(), indent=2, sort_keys=True))
    elif args.mode == "train":
        run.train()
    else:
        run.verify_completion()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
