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

"""The acceptance-layer registry (ADR-0012 decision 2, moved with #136).

``ACCEPTANCE_LAYERS`` is the single declaration point of the L1/L2/L3 identity
wiring -- layer name, attachment kind, report schema string -- and every kind
relation (``LAYER_KINDS``, ``FORMAL_LAYER_KINDS``, ``ATTACH_KINDS``) is derived
from it here, never re-declared beside it.

Identity only, by design: gate constants (FID multiplier, Turing window,
Likert bound) and verdict recomputation keep their dual-side mirror sources
(ADR-0006 referee independence -- shared identity, independent judgement), and
the per-layer validators stay with their layers until the acceptance batches
land. Declaration order is load-bearing: it drives reason collection in the
final acceptance judge and the verdict record's layer entry order.
"""

from dataclasses import dataclass

L1_SCHEMA = "brats-l1-report/1"
L2_SCHEMA = "l2-final-acceptance-report/1"  # mirrors scripts/nnunet_l2_final_acceptance.REPORT_SCHEMA (independent declarations, ADR-0012 decision 5)
L3_SCHEMA = "brats-l3-report/1"

ENV_ATTACHMENT_KIND = "env"  # non-layer attachment: no validator, never part of conclude


@dataclass(frozen=True)
class AcceptanceLayer:
    """One acceptance layer's identity wiring: name, attachment kind, report schema."""

    layer_name: str  # "L1" | "L2" | "L3"
    kind: str  # "l1_report" | "l2_report" | "l3_report"
    report_schema: str


ACCEPTANCE_LAYERS = (
    AcceptanceLayer(layer_name="L1", kind="l1_report", report_schema=L1_SCHEMA),
    AcceptanceLayer(layer_name="L2", kind="l2_report", report_schema=L2_SCHEMA),
    AcceptanceLayer(layer_name="L3", kind="l3_report", report_schema=L3_SCHEMA),
)

LAYER_KINDS = {layer.layer_name: layer.kind for layer in ACCEPTANCE_LAYERS}
FORMAL_LAYER_KINDS = tuple(layer.kind for layer in ACCEPTANCE_LAYERS)
ATTACH_KINDS = (*FORMAL_LAYER_KINDS, ENV_ATTACHMENT_KIND)
