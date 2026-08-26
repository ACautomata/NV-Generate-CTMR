#!/usr/bin/env python3
"""将经审计的 250-epoch trainer 安装进 external nnunetv2。

源模块仍独立受版本控制。本安装器只把 byte-identical 文件复制到
nnunetv2 的 trainer-discovery package，并在受控 audit root 下写入简短 hash 记录。
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import nnunetv2

PERSISTENT_ROOT = Path("/root/private_data")
TARGET_RELATIVE_PATH = Path("training/nnUNetTrainer/variants/training_length/nnUNetTrainer250Epochs.py")


class TrainerInstaller:
    """将获批的训练长度变体放入 external nnunetv2。"""

    def __init__(self, source: Path, audit_path: Path):
        self.source = source
        self.audit_path = audit_path

    def install(self) -> dict:
        self._verify_controlled_inputs()
        source_bytes = self.source.read_bytes()
        target = Path(nnunetv2.__file__).resolve().parent / TARGET_RELATIVE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source_bytes:
            raise FileExistsError(f"refusing to overwrite a different installed trainer: {target}")
        if not target.exists():
            target.write_bytes(source_bytes)
        record = {
            "schema_version": 1,
            "installed_at_utc": datetime.now(UTC).isoformat(),
            "trainer_class": "nnUNetTrainer250Epochs",
            "source_sha256": self._sha256(source_bytes),
            "installed_sha256": self._sha256(target.read_bytes()),
            "installed_module_relative_path": str(TARGET_RELATIVE_PATH),
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record

    def _verify_controlled_inputs(self) -> None:
        if not self.source.is_file():
            raise FileNotFoundError(f"trainer source not found: {self.source}")
        if not self.source.resolve().is_relative_to(PERSISTENT_ROOT):
            raise ValueError(f"trainer source must be under {PERSISTENT_ROOT}")
        if not self.audit_path.resolve().is_relative_to(PERSISTENT_ROOT):
            raise ValueError(f"audit path must be under {PERSISTENT_ROOT}")
        if self.audit_path.exists():
            raise FileExistsError(f"refusing to overwrite trainer installation audit: {self.audit_path}")

    def _sha256(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = TrainerInstaller(args.source, args.audit).install()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
