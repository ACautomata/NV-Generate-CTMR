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

"""The acceptance protocol's fixed vocabulary (ticket T10, issue #28).

One owner for the label/seed/band constants that the orchestrator, the
bootstrap-CI stage, the adjudicator, the montage kit and the report renderer
all read. Spec #13 §⑤ fixes these values; a change here is a protocol change
and must be reflected in the task table and the frozen baseline, so the
constants live alone in this module rather than being re-declared per script.
"""

FORGETTING_LABELS: tuple[int, ...] = (9, 10, 11, 16, 20, 29, 30, 31, 32, 33)
"""The 10 pre-existing rflow-mr-brain labels the forgetting check must cover in full (§⑤.1)."""

BRATS_LABELS: tuple[int, ...] = (40, 41, 42, 43)
"""The finetuned-in new labels (§②.2), generated at BRATS geometry and checked against val64 (§⑤.3)."""

GENERATION_SEEDS: tuple[int, ...] = (42, 1337)
"""The two seeds every forgetting comparison runs under (§⑤.1 pairing); BRATS generation uses seed 42 (§⑤.3)."""

QUICKSCAN_BAND: tuple[float, float] = (0.10, 0.20)
"""Δ band whose verdict is delegated to the quickscan sheet, bounds inclusive (§⑤.2)."""

VAL_FULL_LABELS: tuple[str, ...] = ("mri_t1n", "mri_t1ce", "mri_t2w", "mri_t2f")
"""The four BRATS modality strings the val64 full-reference feature trees are assembled for."""

VOLUMES_PER_TASK: int = 50
"""Volumes per (label, seed) generation task (§⑤.1 sample size)."""

BRATS_REFERENCE_COUNT: int = 64
"""BRATS val64 full-reference size; the real side of BRATS comparisons is never truncated below it."""
