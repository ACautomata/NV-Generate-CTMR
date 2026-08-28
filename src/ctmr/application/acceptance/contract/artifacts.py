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

"""Contract evidence micro-tools (issue #140; the single definition point since #141).

``ArtifactFingerprinter`` and ``ManifestSides`` are the evidence-chain helpers
the run-contract orchestration face shares with the distribution judge chain:
SHA-256 fingerprints linking record entries to bytes on disk, and the pinned
phase manifest viewed as a ``(challenge, case) -> side`` map. A missing
fingerprint target is a contract violation (the legacy script's semantics,
normalized here at its retirement). The raised type is the dm_source ledger
violation per issue #135 -- the same class object the record module aliases
as ``ContractViolationError`` -- raised directly so the record module can
import ``ArtifactFingerprinter`` (CodeVersion) without an import cycle.
"""

import hashlib
import json
from pathlib import Path

from ctmr.infrastructure.dmsource import DmSourceViolationError


class ArtifactFingerprinter:
    """SHA-256 fingerprints linking record entries to bytes on disk."""

    def file_sha256(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def must_fingerprint(self, path, label):
        resolved = Path(path)
        if not resolved.is_file():
            raise DmSourceViolationError(f"{label} not found: {resolved}")
        return {"path": str(resolved.resolve()), "sha256": self.file_sha256(resolved)}

    def content_sha256(self, text):
        return hashlib.sha256(text.encode()).hexdigest()


class ManifestSides:
    """The pinned phase manifest viewed as a (challenge, case) -> side map."""

    def __init__(self, manifest):
        self._manifest = manifest
        self._side_of = {}
        for ch, info in manifest["challenges"].items():
            for side in ("train", "dev", "holdout"):
                for case in info["cases"][side]:
                    self._side_of[(ch, case)] = side

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def side_of(self, challenge, case):
        return self._side_of.get((challenge, case))

    def all_case_keys(self):
        return {case for (_challenge, case) in self._side_of}

    def holdout_keys(self):
        return {key for key, side in self._side_of.items() if side == "holdout"}
