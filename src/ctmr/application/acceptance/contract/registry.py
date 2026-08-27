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

"""Acceptance-layer registry (ADR-0012 决定 2 / CONTEXT.md 验收层注册表).

``ACCEPTANCE_LAYERS`` is the single declaration point for every formal acceptance
layer (attachment kind, layer name, report schema string, validator factory,
layer verdict reader, blocked-reason builder). ``ATTACH_KINDS`` (layer kinds plus
``env``, the non-layer attachment that takes no validator and never joins
``conclude``), ``FORMAL_LAYER_KINDS`` and ``LAYER_KINDS`` all derive from the
registry -- no parallel wiring on the contract side. Adding a layer is one entry
plus the layer's own validator class group.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ctmr.application.acceptance.contract.validators import DistributionReportValidator, ExpertReviewReportValidator, QuantitativeReportValidator


class LayerReportValidator(Protocol):
    """The validator contract every layer's factory yields: (run record, report path) -> failures."""

    def validate(self, record, path): ...


@dataclass(frozen=True)
class AcceptanceLayer:
    """One formal acceptance layer's full wiring in a frozen declaration.

    ``kind`` and ``name`` keep their frozen artifact values (``l1_report`` /
    ``L1`` ...): they are burned into run-record attachments, final-acceptance
    verdict artifacts and blocked-reason prefixes, and the judgement chain and
    existing artifacts stay compatible. The schema string is registry-sourced on
    the contract side; production-side report scripts keep their own ``SCHEMA`` /
    ``REPORT_SCHEMA`` declarations -- schema drift then surfaces in the
    validator instead of being hidden by a shared source (identifier, not
    judgement; ADR-0012 决定 5).
    """

    kind: str
    name: str
    schema: str
    validator_factory: Callable[[], LayerReportValidator]
    verdict_reader: Callable[[dict], object]
    reasons_builder: Callable[[dict], list[str]]


# The single declaration point (ADR-0012 决定 2). Judgement stays inside the
# layer validator classes: gate-constant mirrors and verdict recomputation are
# never parameterized into registry data (referee independence). Schema strings
# come from the validator classes (contract-side single source); the schema *
# values themselves stay registry-visible through the entries.
ACCEPTANCE_LAYERS = (
    AcceptanceLayer(
        kind="l1_report",
        name="L1",
        schema=QuantitativeReportValidator.REPORT_SCHEMA,
        validator_factory=QuantitativeReportValidator,
        verdict_reader=QuantitativeReportValidator.verdict_of,
        reasons_builder=QuantitativeReportValidator.blocked_reasons,
    ),
    AcceptanceLayer(
        kind="l2_report",
        name="L2",
        schema=DistributionReportValidator.REPORT_SCHEMA,
        validator_factory=DistributionReportValidator,
        verdict_reader=DistributionReportValidator.verdict_of,
        reasons_builder=DistributionReportValidator.blocked_reasons,
    ),
    AcceptanceLayer(
        kind="l3_report",
        name="L3",
        schema=ExpertReviewReportValidator.REPORT_SCHEMA,
        validator_factory=ExpertReviewReportValidator,
        verdict_reader=ExpertReviewReportValidator.verdict_of,
        reasons_builder=ExpertReviewReportValidator.blocked_reasons,
    ),
)

# Derived relations -- the registry is the only place the layer kinds are declared.
FORMAL_LAYER_KINDS = tuple(layer.kind for layer in ACCEPTANCE_LAYERS)
ATTACH_KINDS = FORMAL_LAYER_KINDS + ("env",)  # env is a non-layer attachment: no validator, never concludes
LAYER_BY_KIND = {layer.kind: layer for layer in ACCEPTANCE_LAYERS}
LAYER_KINDS = {layer.name: layer.kind for layer in ACCEPTANCE_LAYERS}

# Contract-side schema aliases, all registry-sourced (the former free constants
# are deleted; these keep the read sites single-sourced).
QUANTITATIVE_SCHEMA = LAYER_BY_KIND["l1_report"].schema
DISTRIBUTION_SCHEMA = LAYER_BY_KIND["l2_report"].schema
EXPERT_REVIEW_SCHEMA = LAYER_BY_KIND["l3_report"].schema
