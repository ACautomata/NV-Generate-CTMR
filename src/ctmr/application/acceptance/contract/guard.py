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

"""The holdout guard: final-holdout cases never enter run inputs or selection.

Migrated verbatim from ``brats_phase_run_contract.py`` (retired scripts layer, git history) (#141, spec
decision 3): while a run is open, no data list and no selection evidence may
reference a final-holdout case of the pinned manifest, and selection evidence
may only reference dev-side cases.
"""

import json
from pathlib import Path

from ctmr.application.acceptance.contract.record import ContractViolationError


class HoldoutGuard:
    """Rejects final-holdout cases in run inputs and dev-selection evidence (spec decision 3)."""

    def __init__(self, sides):
        self._sides = sides

    def scan_case_pairs(self, payload):
        """Recursively collects (sub, case) pairs from any parsed-JSON structure."""
        pairs = []
        if isinstance(payload, dict):
            if "sub" in payload and "case" in payload:
                pairs.append((payload["sub"], payload["case"]))
            for value in payload.values():
                pairs += self.scan_case_pairs(value)
        elif isinstance(payload, list):
            for item in payload:
                pairs += self.scan_case_pairs(item)
        return pairs

    def guard_data_list(self, list_entry, phase=None):
        """A labelled list must exist, carry cases, and match its side label with no holdout.

        A ``replay`` list (P1 only, spec #51 decision 6) inverts the membership
        check: every entry must be an external replay-cohort study that is NOT
        in the pinned BraTS manifest — by pair identity or by bare case id, so
        a BraTS case cannot re-enter training under the replay label.

        A P2/P3 ControlNet run (spec #51 decision 8) uses a single fold-split
        list labelled ``train`` that carries both train (fold=1) and dev
        (fold=0) entries — dev is a legitimate light-acceptance run input, not
        holdout — so a ``train`` list in those phases may hold train or dev
        cases. P1 (full-param continuation) keeps the strict side match."""
        path = Path(list_entry["path"])
        if not path.is_file():
            raise ContractViolationError(f"data list not found: {path}")
        pairs = self.scan_case_pairs(json.loads(path.read_text()))
        if not pairs:
            raise ContractViolationError(f"data list carries no (sub, case) entries: {path}")
        label = list_entry["side"]
        manifest_cases = self._sides.all_case_keys() if label == "replay" else None
        allowed_sides = {label}
        if phase in {"P2", "P3"} and label == "train":
            allowed_sides = {"train", "dev"}
        for challenge, case in pairs:
            side = self._sides.side_of(challenge, case)
            if label == "replay":
                if side is not None or case in manifest_cases:
                    raise ContractViolationError(
                        f"{path}: replay list carries manifest case ({challenge}, {case}); "
                        "the replay cohort is external to the BraTS split and must not shadow a split case"
                    )
                continue
            if side is None:
                raise ContractViolationError(f"{path}: ({challenge}, {case}) is not in the pinned manifest")
            if side == "holdout":
                raise ContractViolationError(
                    f"{path}: final-holdout case ({challenge}, {case}) must not enter a run input (holdout runs only after candidate freeze)"
                )
            if side not in allowed_sides:
                raise ContractViolationError(f"{path}: ({challenge}, {case}) is {side}-side but the list is labelled {label}")

    def guard_evidence(self, path):
        """Selection evidence may cite dev-side cases only (dev light acceptance, spec decision 3).

        An evidence file must carry at least one (sub, case) reference: a vacuous
        file cannot substantiate a dev-side selection basis."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ContractViolationError(f"selection evidence not found: {file_path}")
        pairs = self.scan_case_pairs(json.loads(file_path.read_text()))
        if not pairs:
            raise ContractViolationError(f"{file_path}: selection evidence cites no (sub, case) at all")
        for challenge, case in pairs:
            side = self._sides.side_of(challenge, case)
            if side is None:
                raise ContractViolationError(f"{file_path}: ({challenge}, {case}) is not in the pinned manifest")
            if side != "dev":
                raise ContractViolationError(
                    f"{file_path}: selection evidence cites {side}-side case ({challenge}, {case}); "
                    "checkpoint selection may reference dev light acceptance only"
                )

    def guard_config(self, path):
        """A frozen config may reference train/dev cases but never final-holdout ones (tuning input)."""
        file_path = Path(path)
        pairs = self.scan_case_pairs(json.loads(file_path.read_text()))
        for challenge, case in pairs:
            if self._sides.side_of(challenge, case) == "holdout":
                raise ContractViolationError(
                    f"{file_path}: config references final-holdout case ({challenge}, {case}); holdout must not enter tuning inputs"
                )
