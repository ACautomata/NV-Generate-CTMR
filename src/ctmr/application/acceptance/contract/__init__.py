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

"""Run-contract orchestration face: frozen five-key binding, acceptance-layer registry, per-layer validators.

ADR-0015 §2 甲案住址 ``ctmr.application.acceptance.contract``;the judgement
chain (conclude/verify) follows in the acceptance migration tickets.
"""

from ctmr.application.acceptance.contract.binding import BINDING_KEYS, STATUS_FROZEN, FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.contract.registry import (
    ACCEPTANCE_LAYERS,
    ATTACH_KINDS,
    DISTRIBUTION_SCHEMA,
    EXPERT_REVIEW_SCHEMA,
    FORMAL_LAYER_KINDS,
    LAYER_BY_KIND,
    LAYER_KINDS,
    QUANTITATIVE_SCHEMA,
    AcceptanceLayer,
)
from ctmr.application.acceptance.contract.validators import (
    DISTRIBUTION_CHALLENGES,
    DISTRIBUTION_VERDICTS,
    EXPERT_REVIEW_DIMENSIONS,
    EXPERT_REVIEW_LIKERT_BOUND,
    EXPERT_REVIEW_MODALITIES,
    EXPERT_REVIEW_TURING_WINDOW,
    EXPERT_REVIEW_VERDICTS,
    QUANTITATIVE_FEATURE_EXTRACTOR,
    QUANTITATIVE_MODALITIES,
    QUANTITATIVE_MR_PREPROCESSING,
    QUANTITATIVE_PLANES,
    QUANTITATIVE_T1N_TO_T1C,
    QUANTITATIVE_VERDICTS,
    DistributionReportValidator,
    ExpertReviewReportValidator,
    QuantitativeReportValidator,
)

__all__ = [
    "ACCEPTANCE_LAYERS",
    "ATTACH_KINDS",
    "BINDING_KEYS",
    "DISTRIBUTION_CHALLENGES",
    "DISTRIBUTION_SCHEMA",
    "DISTRIBUTION_VERDICTS",
    "EXPERT_REVIEW_DIMENSIONS",
    "EXPERT_REVIEW_LIKERT_BOUND",
    "EXPERT_REVIEW_MODALITIES",
    "EXPERT_REVIEW_SCHEMA",
    "EXPERT_REVIEW_TURING_WINDOW",
    "EXPERT_REVIEW_VERDICTS",
    "FORMAL_LAYER_KINDS",
    "LAYER_BY_KIND",
    "LAYER_KINDS",
    "QUANTITATIVE_FEATURE_EXTRACTOR",
    "QUANTITATIVE_MODALITIES",
    "QUANTITATIVE_MR_PREPROCESSING",
    "QUANTITATIVE_PLANES",
    "QUANTITATIVE_SCHEMA",
    "QUANTITATIVE_T1N_TO_T1C",
    "QUANTITATIVE_VERDICTS",
    "STATUS_FROZEN",
    "AcceptanceLayer",
    "DistributionReportValidator",
    "ExpertReviewReportValidator",
    "FrozenRunBinding",
    "FrozenRunBindingError",
    "QuantitativeReportValidator",
]
