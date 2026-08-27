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

"""The frozen-run five-key binding (ADR-0012 decision 4, moved with #136).

The five keys -- ``run_id`` / ``phase`` / ``manifest_sha256`` /
``candidate_checkpoint_sha256`` / ``samples_sha256`` -- are the pure
field-path identity of a run record: shareable by construction (the drift risk
of hand copies is dual-side desynchronisation), so every producer extracts
them from here instead of re-writing the loop. Judgement never rides along:
gate constants and verdict recomputation stay deliberately dual-sourced
(ADR-0006).

Two faces:

- :func:`expected_binding` -- pure structural extraction (missing freeze
  stages read as ``None``); used by report validators that compare a reported
  binding against the record's own identity.
- :class:`FrozenRunBinding` -- the same five keys with the require-frozen gate
  built in; the face producers and generation-family slices adopt when they
  bind a report to the frozen candidate.
"""

from dataclasses import dataclass

BINDING_KEYS = ("run_id", "phase", "manifest_sha256", "candidate_checkpoint_sha256", "samples_sha256")


def expected_binding(record):
    """Extracts the five-key identity dict from a run record via the single field-path set.

    Pure structural extraction -- no status check here (pre-freeze records read
    as ``None``); callers that must enforce frozenness use
    :class:`FrozenRunBinding`. Key order follows :data:`BINDING_KEYS`, which is
    load-bearing for failure-list ordering downstream.
    """
    return {
        "run_id": record.get("run_id"),
        "phase": record.get("phase"),
        "manifest_sha256": record.get("manifest", {}).get("sha256"),
        "candidate_checkpoint_sha256": record.get("selection", {}).get("checkpoint", {}).get("sha256"),
        "samples_sha256": record.get("samples", {}).get("sha256"),
    }


class FrozenRunBindingError(ValueError):
    """Raised when five-key extraction hits a run that is not a frozen candidate."""


@dataclass(frozen=True)
class FrozenRunBinding:
    """A frozen run's five-key identity with the require-frozen gate built in.

    One extraction point for producers binding a report to the frozen candidate
    exactly the way the run contract validates attachments; a ``from_path``
    classmethod arrives when the producer-side bindings adopt this home.
    """

    run_id: str
    phase: str
    manifest_sha256: str
    candidate_checkpoint_sha256: str
    samples_sha256: str

    @classmethod
    def from_record(cls, record):
        """Extracts from a run record dict; refuses anything not yet frozen."""
        if record.get("status") != "frozen":
            raise FrozenRunBindingError(f"run {record.get('run_id')} is {record.get('status')!r}; binding requires a frozen candidate")
        return cls(**expected_binding(record))

    def as_dict(self):
        """The five-key dict, in :data:`BINDING_KEYS` order."""
        return {key: getattr(self, key) for key in BINDING_KEYS}
