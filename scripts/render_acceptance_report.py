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

"""Render one snapshot's acceptance record into the experiment-point issue format (ticket T10, issue #28).

Spec #13 §⑤.6 fixes what each proportion point's record carries: ckpt and
Release links, the forgetting FID table (10 labels x pre/ft/Δ/CI/verdict x 2
seeds), the BRATS FID table (4 labels + the real-real floor), montage links,
and the verdict. This renderer turns the adjudication json (the §⑤.2 gate's
output) plus the T8 freeze into exactly that markdown, ending with the HITL
section: the five visual checks the user must rule on, with the quickscan
labels and their sheets called out.
"""

import argparse
import json
from pathlib import Path

from scripts.acceptance_protocol import BRATS_LABELS, FORGETTING_LABELS, GENERATION_SEEDS
VERDICT_ZH = {"pass": "通过", "quickscan": "快扫", "hard_fail": "硬 fail"}
SNAPSHOT_ZH = {"pass": "遗忘全过", "quickscan_pending": "待快扫裁决", "fail": "遗忘 fail（淘汰）"}


class AcceptanceReportRenderer:
    """Adjudication json + freeze -> the §⑤.6 markdown record."""

    def __init__(self, adjudication_json: Path, freeze_json: Path, release_url: str) -> None:
        self._payload = json.loads(Path(adjudication_json).read_text())
        self._freeze = json.loads(Path(freeze_json).read_text())
        self._release_url = release_url

    def render(self) -> str:
        sections = [
            self._header(),
            self._forgetting_table(),
            self._brats_table(),
            self._montage_section(),
            self._hitl_section(),
        ]
        return "\n\n".join(sections) + "\n"

    def write(self, output: Path) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.render())

    def _header(self) -> str:
        verdict = self._payload["forgetting_verdict"]
        return "\n".join(
            [
                f"## 验收记录 — {self._payload['run_id']}",
                "",
                f"- 遗忘裁决：**{SNAPSHOT_ZH[verdict]}**（`{verdict}`）",
                f"- BRATS FID 均值（4 label 跨 seed）：{self._payload['brats_fid_mean']}",
                f"- 产物（生成卷、特征、FID json、CI、montage）：{self._release_url}",
            ]
        )

    def _forgetting_table(self) -> str:
        lines = [
            "### 遗忘 FID 表（gen-real vs 冻结 val 参照，spec §⑤.1/§⑤.2）",
            "",
            "| label | seed | FID_pre | FID_ft | Δ% | ft CI95 | pre CI95 | 判定 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label in FORGETTING_LABELS:
            payload = self._payload["labels"][str(label)]
            for seed in GENERATION_SEEDS:
                record = payload[f"seed{seed}"]
                lines.append(
                    f"| {label} | seed{seed} | {record['fid_pre']:.3f} | {record['fid_ft']:.3f} | {record['delta_pct']:+.1f}% "
                    f"| [{record['ci_ft'][0]:.3f}, {record['ci_ft'][1]:.3f}] | [{record['ci_pre'][0]:.3f}, {record['ci_pre'][1]:.3f}] "
                    f"| {VERDICT_ZH[record['verdict']]} |"
                )
        return "\n".join(lines)

    def _brats_table(self) -> str:
        lines = [
            "### BRATS FID 表（4 新 label vs val64 全量；地板 = real-real，spec §⑤.3）",
            "",
            "| label | seed42 | seed1337 | mean |",
            "|---|---|---|---|",
        ]
        for label in BRATS_LABELS:
            entry = self._payload["brats"][str(label)]
            per_seed = entry.get("per_seed", {})
            lines.append(
                f"| {label} | {per_seed.get('42', '—')} | {per_seed.get('1337', '—')} | {entry.get('mean', '—')} |"
            )
        lines.append("")
        floor = self._freeze.get("real_real_floor", {})
        for floor_label, value in sorted(floor.items()):
            lines.append(f"地板 `{floor_label}`：{value['fid_avg']}")
        return "\n".join(lines)

    def _montage_section(self) -> str:
        quickscan = self._payload.get("quickscan_labels", [])
        return "\n".join(
            [
                "### 视觉备料（montage）",
                "",
                f"- BRATS montage：40 张（4 label × 10 volumes × 三平面中央切片）见 Release",
                f"- 快扫 sheet：label {', '.join(quickscan) if quickscan else '（无 — 全部 label 离开快扫带）'}",
            ]
        )

    def _hitl_section(self) -> str:
        quickscan = self._payload.get("quickscan_labels", [])
        lines = [
            "### HITL 视觉裁决（待用户）",
            "",
            "五项检查（①解剖合理性 ②对比度正确性 ③颅骨不存在 ④label 判别力 ⑤无解码伪影）：",
            "",
            "- [ ] BRATS 40 张 montage 逐项检查（①③⑤硬伤或②④系统性错误 → 该点 fail）",
        ]
        if quickscan:
            lines.append(f"- [ ] 快扫裁决：label {', '.join(quickscan)}（无崩坏 → 通过；有 → 该 label fail）")
        else:
            lines.append("- [ ] 快扫裁决：无边缘带 label，跳过")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adjudication_json", type=Path, required=True)
    parser.add_argument("--freeze_json", type=Path, required=True)
    parser.add_argument("--release_url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    renderer = AcceptanceReportRenderer(adjudication_json=args.adjudication_json, freeze_json=args.freeze_json, release_url=args.release_url)
    renderer.write(args.output)
    print(f"report written: {args.output}")


if __name__ == "__main__":
    main()
