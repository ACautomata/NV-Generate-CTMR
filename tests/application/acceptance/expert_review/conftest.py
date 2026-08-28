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

"""Shared synthetic fixtures for the expert-review chain tests.

Synthetic non-patient identities only; the catalog/blind-map/response
builders mirror the legacy ``L3SelfTest`` fixtures verbatim (#141).
"""

import hashlib

import pytest

from ctmr.application.acceptance.expert_review.catalog import (
    CATALOG_SCHEMA,
    DIMENSIONS,
    MODALITIES,
    RESPONSES_SCHEMA,
    Catalog,
)
from ctmr.application.acceptance.expert_review.package import BlindPackageBuilder

FIXTURE_CHALLENGES = ("GLI", "SSA", "MEN", "METS", "PED")


def deterministic_bit(entry_id):
    return int(hashlib.sha256(entry_id.encode()).hexdigest(), 16) % 2


@pytest.fixture()
def run_record():
    return {
        "run_id": "p1-fixture",
        "phase": "P1",
        "status": "frozen",
        "manifest": {"sha256": "m" * 64},
        "selection": {"checkpoint": {"sha256": "c" * 64}},
        "samples": {"sha256": "s" * 64},
    }


@pytest.fixture()
def catalog_payload():
    records = []
    for challenge in FIXTURE_CHALLENGES:
        for modality in MODALITIES:
            for index in range(6):
                real_id = f"REAL-{challenge}-{modality}-{index}"
                synth_id = f"SYNTH-{challenge}-{modality}-{index}"
                records.append(
                    {
                        "challenge": challenge,
                        "case": real_id,
                        "target_modality": modality,
                        "source": "real",
                        "path": f"/ctrl/real/{real_id}.nii.gz",
                        "sha256": "d" * 64,
                    }
                )
                records.append(
                    {
                        "challenge": challenge,
                        "case": synth_id,
                        "target_modality": modality,
                        "source": "synth",
                        "path": f"/ctrl/synth/{synth_id}.nii.gz",
                        "sha256": "f" * 64,
                    }
                )
    return {"schema": CATALOG_SCHEMA, "records": records}


@pytest.fixture()
def catalog(catalog_payload):
    return Catalog(catalog_payload)


@pytest.fixture()
def blind_map_doc(run_record, catalog):
    _package_doc, blind_map_doc = BlindPackageBuilder(seed=20260821, per_cell=5).build(run_record, catalog)
    return blind_map_doc


@pytest.fixture()
def make_responses():
    """Build two-reviewer response sets: near-chance or a strong distinguisher, with per-dimension Likert scores."""

    def _make(blind_map_entries, likert_scores=None, strong_distinguisher=False):
        scores = likert_scores or {dimension: 4 for dimension in DIMENSIONS}
        responses = []
        for reviewer in ("R1", "R2"):
            entries = []
            for entry in blind_map_entries:
                real = entry["source"] == "real"
                if strong_distinguisher and reviewer == "R2":
                    prediction = "real" if real else "synth"  # balanced accuracy near 1.0 -> window fail
                else:
                    prediction = "real" if deterministic_bit(entry["entry_id"]) == 0 else "synth"  # near chance
                entries.append({"entry_id": entry["entry_id"], "turing": prediction, **scores, "notes": ""})
            responses.append({"schema": RESPONSES_SCHEMA, "reviewer": reviewer, "entries": entries})
        return responses

    return _make
