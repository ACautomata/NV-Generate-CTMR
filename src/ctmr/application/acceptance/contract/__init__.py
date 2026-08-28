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

"""Run-contract orchestration face (ADR-0015 §2 / issue #141).

The whole contract now lives here: the frozen five-key binding
(``binding``), the acceptance-layer registry and per-layer validators
(``registry`` / ``validators``), the evidence micro-tools (``artifacts``),
and -- migrated from ``brats_phase_run_contract.py`` (retired scripts layer, git history) at its
retirement (#141) -- the record vocabulary + store (``record``), the holdout
guard (``guard``), the lifecycle mutations open/select/freeze/attach
(``lifecycle``), the final-acceptance orchestration over the domain kernel
(``conclude`` + ``ctmr.domain.acceptance``), the run verifier (``verify``)
and the ``ctmr accept contract <verb>`` entry (``cli``).
"""

from ctmr.application.acceptance.contract.artifacts import ArtifactFingerprinter, ManifestSides
from ctmr.application.acceptance.contract.binding import BINDING_KEYS, STATUS_FROZEN, FrozenRunBinding, FrozenRunBindingError
from ctmr.application.acceptance.contract.conclude import FinalAcceptanceJudge
from ctmr.application.acceptance.contract.guard import HoldoutGuard
from ctmr.application.acceptance.contract.lifecycle import CandidateFreezer, ReportAttacher, RunInitializer, SelectionRecorder
from ctmr.application.acceptance.contract.record import (
    CONTROLNET_CANDIDATE,
    FINAL_ACCEPTANCE_SCHEMA,
    LIST_SIDES,
    P3_VARIANTS,
    PHASES,
    SCHEMA,
    STAGE0_BASELINE,
    STATUS_OPEN,
    UPSTREAM_PHASE,
    CodeVersion,
    ContractViolationError,
    RunRecordStore,
)
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
from ctmr.application.acceptance.contract.verify import RunVerifier

__all__ = [
    "ACCEPTANCE_LAYERS",
    "ATTACH_KINDS",
    "BINDING_KEYS",
    "CONTROLNET_CANDIDATE",
    "DISTRIBUTION_CHALLENGES",
    "DISTRIBUTION_SCHEMA",
    "DISTRIBUTION_VERDICTS",
    "EXPERT_REVIEW_DIMENSIONS",
    "EXPERT_REVIEW_LIKERT_BOUND",
    "EXPERT_REVIEW_MODALITIES",
    "EXPERT_REVIEW_SCHEMA",
    "EXPERT_REVIEW_TURING_WINDOW",
    "EXPERT_REVIEW_VERDICTS",
    "FINAL_ACCEPTANCE_SCHEMA",
    "FORMAL_LAYER_KINDS",
    "LAYER_BY_KIND",
    "LAYER_KINDS",
    "LIST_SIDES",
    "PHASES",
    "P3_VARIANTS",
    "QUANTITATIVE_FEATURE_EXTRACTOR",
    "QUANTITATIVE_MODALITIES",
    "QUANTITATIVE_MR_PREPROCESSING",
    "QUANTITATIVE_PLANES",
    "QUANTITATIVE_SCHEMA",
    "QUANTITATIVE_T1N_TO_T1C",
    "QUANTITATIVE_VERDICTS",
    "SCHEMA",
    "STAGE0_BASELINE",
    "STATUS_FROZEN",
    "STATUS_OPEN",
    "UPSTREAM_PHASE",
    "AcceptanceLayer",
    "ArtifactFingerprinter",
    "CandidateFreezer",
    "CodeVersion",
    "ContractViolationError",
    "DistributionReportValidator",
    "ExpertReviewReportValidator",
    "FinalAcceptanceJudge",
    "FrozenRunBinding",
    "FrozenRunBindingError",
    "HoldoutGuard",
    "ManifestSides",
    "QuantitativeReportValidator",
    "ReportAttacher",
    "RunInitializer",
    "RunRecordStore",
    "RunVerifier",
    "SelectionRecorder",
]
