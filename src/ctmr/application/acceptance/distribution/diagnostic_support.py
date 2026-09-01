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

"""Diagnostic reading support pieces (ADR-0017 decision 6, issue #232).

The single home of the machinery the diagnostic jobs used to copy from one
another: ``DiagnosticError`` (one error type for the whole diagnostic fleet),
``DiagnosticReportWriter`` (the ``variant=diagnostic`` json + markdown
artifact scaffolding) and ``DiagnosticSeedAllocator`` (bootstrap seeds drawn
from the unified registry's diagnostic namespace -- ``challenge_registry``
holds the registration data, this module the allocation mechanism; registered
slots are drawn only through the allocator, never re-spelled). Jobs A/B are
fully on these pieces; jobs C/D and the geometry audit still carry their
pre-registry slot blocks as a recorded follow-up. The P3 candidate reuse
(#205 series-③) lands here: a future P3 diagnostic job implements its own
reading logic and takes error type, report writer and seeds from this
module. The diagnostic face stays out of the ``ctmr accept`` verb surface --
diagnostic readings are never acceptance evidence (parent decision,
CONTEXT.md「诊断读数」).

The dependency closure is third-party-free -- stdlib only (the registry it
draws from is stdlib-only too) -- registered by the import-face probe in
``tests/application/acceptance/distribution/test_shared_vocab.py``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from ctmr.application.acceptance.distribution.challenge_registry import (
    CHALLENGE_SEED_OFFSET,
    DIAGNOSTIC_SEED_BAND,
    DIAGNOSTIC_SEED_BASE,
)


class DiagnosticError(Exception):
    """Raised when a diagnostic job's inputs cannot support its run."""


class DiagnosticSeedAllocator:
    """Bootstrap seeds of the diagnostic namespace, drawn from the registry.

    Seed = ``DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET[challenge] *
    DIAGNOSTIC_SEED_BAND + slot``: one 1000-wide band per challenge, slot
    numbers allocated in ``challenge_registry.DIAGNOSTIC_SEED_SLOTS``. The
    pre-#232 job-module formulas reproduce byte-exactly through this
    allocator; the「L2 全域种子无碰撞」invariant is a unit test on the
    registry table.
    """

    @classmethod
    def seed(cls, challenge: str, slot: int) -> int:
        return DIAGNOSTIC_SEED_BASE + CHALLENGE_SEED_OFFSET[challenge] * DIAGNOSTIC_SEED_BAND + slot


class DiagnosticReportWriter:
    """``variant=diagnostic`` json + markdown artifact writer.

    The shared report scaffolding of the diagnostic jobs (sugon artifact area,
    never git): the payload prologue (schema/title/issue/variant/disclaimer/
    run_id/generated_utc/inputs) and the markdown preamble render here once;
    each job contributes its own body and markdown sections. P3 diagnostic
    jobs (#205 series-③) reuse this writer as-is.
    """

    def __init__(
        self, schema: str, title: str, issue: int, job_label: str, stem: str, inputs: dict, run_id: str | None = None, parent_issue: int = 205
    ):
        self._schema = schema
        self._title = title
        self._issue = issue
        self._parent_issue = parent_issue
        self._job_label = job_label
        self._stem = stem
        self._inputs = inputs
        self._run_id = run_id

    @property
    def disclaimer(self) -> str:
        return (
            f"诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#{self._parent_issue} {self._job_label})。"
            f"bootstrap 种子独立于正式判定链(诊断基 {DIAGNOSTIC_SEED_BASE})。"
        )

    def payload(self, body: dict) -> dict:
        """The report payload: the prologue fields first, then the job's body."""
        return {
            "schema": self._schema,
            "title": self._title,
            "issue": self._issue,
            "variant": "diagnostic",
            "disclaimer": self.disclaimer,
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": self._inputs,
            **body,
        }

    def markdown_preamble(self, payload: dict) -> list:
        return [
            f"# {payload['title']}",
            "",
            f"**Issue**: [#{self._issue}](https://github.com/ACautomata/NV-Generate-CTMR/issues/{self._issue})(父 #{self._parent_issue} {self._job_label})"
            f" · **run**: `{payload['run_id'] or '未绑定'}`",
            f"**variant: diagnostic —— {payload['disclaimer']}**",
            "",
        ]

    def write(self, payload: dict, markdown_text: str, output_dir) -> tuple:
        """Write the json + markdown pair into the artifact area; returns both paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self._stem}.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        md_path = output_dir / f"{self._stem}.md"
        md_path.write_text(markdown_text)
        return json_path, md_path
