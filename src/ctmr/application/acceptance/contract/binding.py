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

"""Frozen-candidate identity: the five-key binding (ADR-0012 决定 4 / CONTEXT.md 冻结候选绑定).

``FrozenRunBinding`` is the single construction point that extracts the frozen
candidate identity (run_id / phase / manifest_sha256 /
candidate_checkpoint_sha256 / samples_sha256) from a run record, with the
``require_frozen`` gate built in -- extracting is validating the run state.
Identity is sharable (drift risk is the two sides losing sync, which sharing
removes); gate-constant mirrors and verdict recomputation stay judgement and
remain different-sourced on both sides (ADR-0006 referee independence).
"""

import json
from dataclasses import dataclass
from pathlib import Path

STATUS_FROZEN = "frozen"
BINDING_KEYS = ("run_id", "phase", "manifest_sha256", "candidate_checkpoint_sha256", "samples_sha256")

_EXTRACTORS = {
    "run_id": lambda record: record["run_id"],
    "phase": lambda record: record["phase"],
    "manifest_sha256": lambda record: record["manifest"]["sha256"],
    "candidate_checkpoint_sha256": lambda record: record["selection"]["checkpoint"]["sha256"],
    "samples_sha256": lambda record: record["samples"]["sha256"],
}


class FrozenRunBindingError(ValueError):
    """Raised when a run record cannot yield the frozen candidate identity (extraction or freeze gate)."""


@dataclass(frozen=True)
class FrozenRunBinding:
    """The immutable frozen-candidate fields copied into acceptance reports."""

    run_id: str
    phase: str
    manifest_sha256: str
    candidate_checkpoint_sha256: str
    samples_sha256: str

    @classmethod
    def from_record(cls, record):
        """Extract the five keys from a run record, requiring the frozen state.

        The freeze gate is inside the extraction, so every consumer gets the
        same ``is not frozen`` failure instead of a per-script gate.
        """
        if not isinstance(record, dict):
            raise FrozenRunBindingError("run record must be a JSON object")
        if record.get("status") != STATUS_FROZEN:
            raise FrozenRunBindingError(f"run {record.get('run_id')!r} is {record.get('status')!r}; frozen-candidate binding requires a frozen run")
        values = {}
        missing = []
        for key, extract in _EXTRACTORS.items():
            try:
                values[key] = extract(record)
            except (KeyError, TypeError):
                missing.append(key)
        if missing:
            raise FrozenRunBindingError(f"run record lacks frozen candidate binding key(s): {', '.join(missing)}")
        return cls(**values)

    @classmethod
    def from_path(cls, record_path):
        """Load a run record file and extract its five-key binding (freeze gate included)."""
        path = Path(record_path)
        if not path.is_file():
            raise FrozenRunBindingError(f"run record not found: {path}")
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise FrozenRunBindingError(f"run record is not valid JSON: {path} ({error})") from error
        return cls.from_record(record)

    def as_dict(self):
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "manifest_sha256": self.manifest_sha256,
            "candidate_checkpoint_sha256": self.candidate_checkpoint_sha256,
            "samples_sha256": self.samples_sha256,
        }

    @staticmethod
    def mismatches_for(record, report_binding):
        """Tolerant five-key identity comparison against a report's binding object.

        Used by the contract-side validators (identity check, not judgement):
        keys missing from the record surface as mismatches, never as exceptions.
        Returns ``None`` when the report binding is not an object (the caller
        reports its own "must be an object" message).
        """
        if not isinstance(report_binding, dict):
            return None
        expected = {
            "run_id": record.get("run_id"),
            "phase": record.get("phase"),
            "manifest_sha256": (record.get("manifest") or {}).get("sha256"),
            "candidate_checkpoint_sha256": ((record.get("selection") or {}).get("checkpoint") or {}).get("sha256"),
            "samples_sha256": (record.get("samples") or {}).get("sha256"),
        }
        return [key for key, expected_value in expected.items() if report_binding.get(key) != expected_value]
