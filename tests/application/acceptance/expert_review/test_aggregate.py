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

"""The blinded-judgment aggregation chain, observed as pytest (#141).

The resident ``L3SelfTest`` aggregate assertions of
``brats_l3_blind_eval.py`` (retired scripts layer, git history) became this file when the chain moved into
the expert_review package (ADR-0015 §6): the candidate-bound report binding,
the over-confident reviewer failing the visual-Turing window, the sub-bound
Likert failing the non-compensatory AND, determinism for a fixed seed, and
the fewer-than-two-reviewers rejection.

Light stack (stdlib only).
"""

import pytest

from ctmr.application.acceptance.expert_review.aggregate import L3ReportProducer
from ctmr.application.acceptance.expert_review.catalog import DIMENSIONS, L3Error

REPORT_SCHEMA = "brats-l3-report/1"


def test_aggregate_verdict_fails_on_overconfident_reviewer(run_record, catalog, blind_map_doc, make_responses):
    responses = make_responses(blind_map_doc["entries"], strong_distinguisher=True)

    report = L3ReportProducer(resamples=200, seed=20260821)._produce(run_record, responses, blind_map_doc, catalog.payload)

    assert report["schema"] == REPORT_SCHEMA
    assert report["binding"]["candidate_checkpoint_sha256"] == "c" * 64
    assert all(result["verdict"] in ("pass", "fail") for result in report["visual_turing"]["per_reviewer"])
    assert report["visual_turing"]["verdict"] == "fail"  # an over-confident reviewer fails the window gate
    assert report["verdict"]["overall"] == "fail"  # the AND is non-compensatory


def test_likert_below_bound_fails_the_and(run_record, catalog, blind_map_doc, make_responses):
    responses = make_responses(
        blind_map_doc["entries"],
        likert_scores={**{dimension: 4 for dimension in DIMENSIONS}, "anatomical_plausibility": 3},  # under the 4.0 bound
    )

    report = L3ReportProducer(resamples=200, seed=20260821)._produce(run_record, responses, blind_map_doc, catalog.payload)

    assert report["visual_turing"]["verdict"] == "pass"  # near-chance reviewers pass the window
    assert report["verdict"]["overall"] == "fail"  # a sub-4.0 Likert lower bound fails the AND


def test_aggregate_is_deterministic(run_record, catalog, blind_map_doc, make_responses):
    responses = make_responses(blind_map_doc["entries"])
    first = L3ReportProducer(resamples=100, seed=7)._produce(run_record, responses, blind_map_doc, catalog.payload)
    second = L3ReportProducer(resamples=100, seed=7)._produce(run_record, responses, blind_map_doc, catalog.payload)

    assert first == second


def test_fewer_than_two_reviewers_is_rejected(run_record, catalog, blind_map_doc, make_responses):
    responses = make_responses(blind_map_doc["entries"])[:1]

    with pytest.raises(L3Error, match="at least two independent reviewers"):
        L3ReportProducer(resamples=100, seed=20260821)._produce(run_record, responses, blind_map_doc, catalog.payload)
