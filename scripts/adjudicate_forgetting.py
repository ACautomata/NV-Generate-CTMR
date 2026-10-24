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

"""Spec §⑤.2 forgetting adjudication from the T10 FID + CI artifacts (ticket T10, issue #28).

The table is the ticket's hard gate, per finetuned snapshot:

| Δ (gen-real vs FID_pre) | CI relation | verdict |
|---|---|---|
| ≤ 10% | any | pass |
| 10–20% | overlapping | quickscan (visual, HITL) |
| 10–20% | separated | hard fail |
| > 20% | any | hard fail |

The last row closes the "large delta but coincidentally overlapping CI" hole
in the conservative direction (spec §⑤.2). Both seeds must pass for a label
to pass; any hard fail fails the snapshot outright (zero tolerance across the
10 old labels, §⑤.1). The gen-gen paired FIDs are reported but never gated.

The BRATS comparisons carry no gate of their own (§⑤.3: visual-first, FID for
cross-tier comparison only) — they are summarized per snapshot into the mean
the finalization path ranks candidates by (§⑤.4).
"""

import argparse
import json
from pathlib import Path

from scripts.acceptance_protocol import BRATS_LABELS, FORGETTING_LABELS, GENERATION_SEEDS, QUICKSCAN_BAND


class SeedComparison:
    """One (label, seed) gen-real verdict under the §⑤.2 table."""

    def __init__(self, fid_pre: float, ci_pre: list[float], fid_ft: float, ci_ft: list[float]) -> None:
        self.fid_pre = fid_pre
        self.fid_ft = fid_ft
        self.ci_pre = ci_pre
        self.ci_ft = ci_ft
        self.delta = (fid_ft - fid_pre) / fid_pre
        self.overlap = not (ci_pre[1] < ci_ft[0] or ci_ft[1] < ci_pre[0])

    def verdict(self) -> str:
        if self.delta <= QUICKSCAN_BAND[0]:
            return "pass"
        if self.delta > QUICKSCAN_BAND[1]:
            return "hard_fail"
        if self.overlap:
            return "quickscan"
        return "hard_fail"

    def record(self) -> dict:
        return {
            "fid_pre": self.fid_pre,
            "fid_ft": self.fid_ft,
            "delta": self.delta,
            "delta_pct": self.delta * 100.0,
            "ci_pre": self.ci_pre,
            "ci_ft": self.ci_ft,
            "ci_overlap": self.overlap,
            "verdict": self.verdict(),
        }


class ForgettingAdjudicator:
    """Reads one snapshot's fid/ + ci/ artifacts and the frozen baseline into a §⑤.2 verdict."""

    def __init__(self, snapshot_dir: Path, freeze_json: Path, baseline_ci_dir: Path) -> None:
        self._snapshot_dir = Path(snapshot_dir)
        self._freeze = json.loads(Path(freeze_json).read_text())
        self._baseline_ci_dir = Path(baseline_ci_dir)

    def adjudicate(self) -> dict:
        labels: dict[str, dict] = {}
        for label in FORGETTING_LABELS:
            seed_records = {f"seed{seed}": self._adjudicate_seed(label, seed) for seed in GENERATION_SEEDS}
            labels[str(label)] = {**seed_records, "verdict": self._aggregate_verdicts(seed_records)}

        brats = self._brats_summary()
        return {
            "labels": labels,
            "forgetting_verdict": self._snapshot_verdict(labels),
            "quickscan_labels": sorted(label for label, payload in labels.items() if payload["verdict"] == "quickscan"),
            "brats": brats,
            "brats_fid_mean": self._brats_mean(brats),
        }

    def _adjudicate_seed(self, label: int, seed: int) -> dict:
        tag = f"label{label}_seed{seed}"
        baseline_ci_path = self._baseline_ci_dir / f"{tag}.json"
        if not baseline_ci_path.exists():
            raise FileNotFoundError(f"baseline CI missing (run the baseline CI stage first): {baseline_ci_path}")

        fid_ft = self._read_fid_avg(self._snapshot_dir / "fid" / f"genreal_{tag}.json")
        ci_ft = self._read_ci(self._snapshot_dir / "ci" / f"genreal_{tag}.json")
        fid_pre = self._freeze["baseline_gen_real"][str(label)][str(seed)]["fid_avg"]
        ci_pre = self._read_ci(baseline_ci_path)

        return SeedComparison(fid_pre=fid_pre, ci_pre=ci_pre, fid_ft=fid_ft, ci_ft=ci_ft).record()

    @staticmethod
    def _read_fid_avg(path: Path) -> float:
        return float(json.loads(path.read_text())["fid_avg"])

    @staticmethod
    def _read_ci(path: Path) -> list[float]:
        return [float(v) for v in json.loads(path.read_text())["fid_avg"]["ci95"]]

    @staticmethod
    def _aggregate_verdicts(seed_records: dict[str, dict]) -> str:
        verdicts = [record["verdict"] for record in seed_records.values()]
        if "hard_fail" in verdicts:
            return "hard_fail"
        if "quickscan" in verdicts:
            return "quickscan"
        return "pass"

    @staticmethod
    def _snapshot_verdict(labels: dict[str, dict]) -> str:
        label_verdicts = [payload["verdict"] for payload in labels.values()]
        if "hard_fail" in label_verdicts:
            return "fail"
        if "quickscan" in label_verdicts:
            return "quickscan_pending"
        return "pass"

    def _brats_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for label in BRATS_LABELS:
            per_seed = {}
            for seed in GENERATION_SEEDS:
                path = self._snapshot_dir / "fid" / f"brats_label{label}_seed{seed}.json"
                if path.exists():
                    per_seed[str(seed)] = self._read_fid_avg(path)
            entry: dict = {"per_seed": per_seed}
            if per_seed:
                entry["mean"] = sum(per_seed.values()) / len(per_seed)
            summary[str(label)] = entry
        return summary

    @staticmethod
    def _brats_mean(brats: dict[str, dict]) -> float | None:
        means = [entry["mean"] for entry in brats.values() if "mean" in entry]
        if not means:
            return None
        return sum(means) / len(means)


class MatrixCandidateRanker:
    """§⑤.4 finalization path over the whole matrix: rank forgetting-clean snapshots by BRATS FID.

    Also classifies whether the §⑤.4 retest rule fires, across its three branches:
    every snapshot through the forgetting gate failing means replay-scale
    up-probe; the BRATS visual check failing everywhere means down-probe; a
    cross (forgetting-clean snapshots and visually-clean snapshots exist but
    none is both) means interpolating the midpoint replay size between the
    forgetting-clean anchor with the best BRATS FID and the visually-clean
    anchor with the lowest replay tier. Branches needing the HITL visual
    outcome stay open until every snapshot's ``brats_visual`` is recorded.
    """

    def __init__(self, adjudication_dir: Path) -> None:
        self._dir = Path(adjudication_dir)

    def rank(self) -> dict:
        records = [self._read(path) for path in sorted(self._dir.glob("*.json"))]

        candidates = sorted(
            (r for r in records if r["forgetting_verdict"] == "pass" and r["brats_fid_mean"] is not None),
            key=lambda r: r["brats_fid_mean"],
        )
        ranking: dict = {"candidates": candidates, "retest": None}

        decided = [r for r in records if r["forgetting_verdict"] in ("pass", "fail")]
        if len(decided) < len(records):
            return ranking  # a quickscan-pending snapshot keeps the whole-matrix decision open

        if records and all(r["forgetting_verdict"] == "fail" for r in records):
            ranking["retest"] = {"branch": "scale_up_replay", "reason": "no snapshot passes forgetting"}
            return ranking

        visual_decided = [r for r in records if r["brats_visual"] is not None]
        if len(visual_decided) < len(records):
            return ranking  # waiting on the HITL visual verdicts before the down-probe/cross call

        if all(r["brats_visual"] == "fail" for r in records):
            ranking["retest"] = {"branch": "scale_down_replay", "reason": "every snapshot failed the BRATS visual check"}
            return ranking

        forgetting_clean = [r for r in records if r["forgetting_verdict"] == "pass"]
        brats_clean = [r for r in records if r["brats_visual"] == "pass"]
        if forgetting_clean and brats_clean and not any(r in forgetting_clean and r in brats_clean for r in records):
            anchor_forgetting = min(forgetting_clean, key=lambda r: r["brats_fid_mean"])
            anchor_brats = min(brats_clean, key=lambda r: r["tier"])
            midpoint = (anchor_forgetting["tier"] + anchor_brats["tier"]) // 2
            ranking["retest"] = {
                "branch": "interpolate_midpoint",
                "forgetting_clean_anchor": anchor_forgetting,
                "brats_clean_anchor": anchor_brats,
                "midpoint_replay_n": midpoint,
                "reason": "forgetting-clean and visually-clean snapshots are disjoint (§⑤.4 cross)",
            }
        return ranking

    @staticmethod
    def _read(path: Path) -> dict:
        payload = json.loads(path.read_text())
        run_id = payload.get("run_id", path.stem)
        # run_id shape N{tier}e{epoch} — the tier is what sits between N and e.
        tier = int(run_id[1:].split("e", 1)[0]) if run_id.startswith("N") and "e" in run_id else 0
        return {
            "run_id": run_id,
            "tier": tier,
            "forgetting_verdict": payload["forgetting_verdict"],
            "brats_fid_mean": payload.get("brats_fid_mean"),
            "brats_visual": payload.get("brats_visual"),
        }


class VisualVerdictRecorder:
    """Writes the user's HITL visual outcomes back into an adjudication json.

    The §⑤.3 five-check ruling on the 40 BRATS montages and the §⑤.2 quickscan
    conclusions are the human's words; this records them verbatim so the
    adjudication file stays the single source the ranker and the report read.
    """

    def __init__(self, adjudication_json: Path) -> None:
        self._path = Path(adjudication_json)

    def record(self, brats_visual: str | None, quickscan: dict[str, str] | None) -> dict:
        payload = json.loads(self._path.read_text())
        if brats_visual is not None:
            if brats_visual not in ("pass", "fail"):
                raise ValueError(f"brats_visual must be pass/fail, got {brats_visual!r}")
            payload["brats_visual"] = brats_visual
        if quickscan:
            for label, verdict in quickscan.items():
                if verdict not in ("pass", "fail"):
                    raise ValueError(f"quickscan verdict for label {label} must be pass/fail, got {verdict!r}")
            payload.setdefault("quickscan_conclusions", {}).update(quickscan)
            # A quickscan verdict resolves the label and can flip the snapshot verdict.
            label_verdicts = []
            for label, entry in payload["labels"].items():
                resolved = payload["quickscan_conclusions"].get(label)
                label_verdicts.append("fail" if resolved == "fail" else ("pass" if resolved or entry["verdict"] == "pass" else entry["verdict"]))
            payload["forgetting_verdict"] = ForgettingAdjudicator._snapshot_verdict({k: {"verdict": v} for k, v in zip(payload["labels"], label_verdicts)})
        self._path.write_text(json.dumps(payload, indent=2) + "\n")
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot_dir", type=Path, help="one snapshot's directory (holds fid/ and ci/) — single-snapshot adjudication mode")
    parser.add_argument("--freeze_json", type=Path)
    parser.add_argument("--baseline_ci_dir", type=Path)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--rank", action="store_true", help="matrix mode: rank candidates and classify the §⑤.4 retest branch")
    parser.add_argument("--adjudication_dir", type=Path, help="directory of per-snapshot adjudication jsons (with --rank)")
    parser.add_argument("--record_visual", type=Path, help="adjudication json to record HITL visual outcomes into")
    parser.add_argument("--brats_visual", choices=("pass", "fail"), help="the §⑤.3 five-check ruling")
    parser.add_argument("--quickscan", nargs="*", help="quickscan conclusions as label=pass|fail pairs, e.g. 16=pass 20=fail")
    args = parser.parse_args()

    if args.record_visual:
        quickscan = dict(pair.split("=", 1) for pair in args.quickscan) if args.quickscan else None
        payload = VisualVerdictRecorder(args.record_visual).record(brats_visual=args.brats_visual, quickscan=quickscan)
        print(f"recorded: brats_visual={payload.get('brats_visual')} forgetting_verdict={payload['forgetting_verdict']}")
        return

    if args.rank:
        if not args.adjudication_dir or not args.output_json:
            parser.error("--rank requires --adjudication_dir and --output_json")
        ranking = MatrixCandidateRanker(adjudication_dir=args.adjudication_dir).rank()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(ranking, indent=2) + "\n")
        print(f"ranking: {len(ranking['candidates'])} candidates, retest={ranking['retest']} -> {args.output_json}")
        return

    if not args.snapshot_dir or not args.freeze_json or not args.baseline_ci_dir or not args.output_json:
        parser.error("single-snapshot mode requires --snapshot_dir/--freeze_json/--baseline_ci_dir/--output_json")

    payload = ForgettingAdjudicator(snapshot_dir=args.snapshot_dir, freeze_json=args.freeze_json, baseline_ci_dir=args.baseline_ci_dir).adjudicate()
    payload["run_id"] = args.snapshot_dir.name

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"adjudication for {args.snapshot_dir.name}: forgetting={payload['forgetting_verdict']} -> {args.output_json}")


if __name__ == "__main__":
    main()
