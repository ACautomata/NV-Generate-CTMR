# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the audited 250-epoch trainer into external nnunetv2 (migrated from scripts/install_nnunet_trainer_250, ticket #140).

The trainer module itself stays version-controlled in this package. This installer
only copies the byte-identical file into nnunetv2's trainer-discovery package and
writes a short hash record under the controlled audit root. The script ``__main__``
glue and argparse block do not travel: a later CLI slice takes over.
"""

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
