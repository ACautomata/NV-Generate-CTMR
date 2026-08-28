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

"""Expert-review blinding protocol constants and the controlled image catalog.

Migrated verbatim from ``brats_l3_blind_eval.py`` (retired scripts layer, git history) (#141). The gate
constants here are the production-side mirror of the contract-side
``ExpertReviewReportValidator`` constants -- different sources on purpose
(ADR-0006 referee independence): a production bug must not be let through by
a same-source checker. Stdlib only.
"""

import json
from dataclasses import dataclass
from pathlib import Path

CATALOG_SCHEMA = "brats-l3-catalog/1"
PACKAGE_SCHEMA = "brats-l3-package/1"
BLIND_MAP_SCHEMA = "brats-l3-blind-map/1"
RESPONSES_SCHEMA = "brats-l3-responses/1"
REPORT_SCHEMA = "brats-l3-report/1"

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
DIMENSIONS = (
    "overall_realism",
    "anatomical_plausibility",
    "tumor_authenticity",
    "artifact_slice_consistency",
)
TURING_LABELS = ("real", "synth")
LIKERT_MIN, LIKERT_MAX = 1, 5
NA = "NA"
TURING_WINDOW = (0.40, 0.60)
LIKERT_BOUND = 4.0
TOTAL_ENTRIES = 200


class L3Error(Exception):
    """Raised when a blinding package or judgment does not satisfy the L3 protocol."""


@dataclass
class CatalogEntry:
    """One available candidate image of a given source on a given target modality."""

    challenge: str
    case: str
    target_modality: str
    source: str
    path: str
    sha256: str
    src_modality: str | None = None


class Catalog:
    """The controlled catalog of candidate images available for L3 sampling."""

    def __init__(self, payload):
        self._payload = payload
        self._entries = [CatalogEntry(**record) for record in payload["records"]]

    @classmethod
    def from_path(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != CATALOG_SCHEMA:
            raise L3Error(f"catalog schema must be {CATALOG_SCHEMA!r}, got {payload.get('schema')!r}")
        return cls(payload)

    @property
    def payload(self):
        return self._payload

    def entries(self):
        return self._entries

    def cell(self, challenge, target_modality):
        real, synth = [], []
        for entry in self._entries:
            if (entry.challenge, entry.target_modality) == (challenge, target_modality):
                (real if entry.source == "real" else synth).append(entry)
        return real, synth

    def challenges(self):
        return tuple(sorted({entry.challenge for entry in self._entries}))
