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

"""RAS re-encode job (issue #312, series-③ T2): the direction-unified corpus
pass.

The #312 dim-decision fix (``new_dim_from_ras_shape``: the resize target now
reads off the RAS-reoriented spatial shape instead of the NIfTI header's
storage-axis dim) invalidates the clip=True-era training embeddings the same
way the T4 recipe change did: every volume whose direction permutes axes under
reorientation was encoded against a mis-sized, axis-scrambled resize target --
the direction audit's ~65% corrupted embedding-shape share in the replay/
MR-RATE arm (2026-09-03, #310). The BraTS arm escaped only because its
directions are flip-only, so this pass covers ALL arms, into a NEW embedding
root -- the legacy and clip=True trees stay untouched as audit anchors -- in
three steps:

1. Re-encode: the vendored ``diff_model_create_training_data`` chain (fixed
   dim decision + the clip=True factory) runs against an env json whose
   ``embedding_base_dir`` points at the NEW root. The chain skips files that
   already exist, so the job re-enters after an instance loss (the T4
   precedent, twice exercised).
2. Manifest + sidecars + SHAPE GUARD: the new tree is walked and every
   ``*_emb.nii.gz`` registered with md5/bytes/shape into ``manifest.jsonl``;
   the T7 sidecars (<emb>.json, spacing/modality) are fetched per entry from
   the first ``--sidecar-source`` root that carries a valid one (the arms'
   sidecars live in different old roots). The guard then derives each entry's
   EXPECTED latent shape from its raw volume -- the RAS-reoriented shape
   (nibabel's canonical form, the same reorientation MONAI's Orientationd
   applies), rounded to the encode grid and divided by the latent downsample,
   channels last -- and any artifact that disagrees FAILS the job (after the
   report lands): a shape the training loader must reject must never exit
   green here. Missing entries (encode failures, unusable sidecars) reconcile
   nonzero exactly like T4.
3. Spot-check, two pools: the primary list's t1c subset (job C's clip=True
   convention, T4's environment check reused as-is) AND the replay subset --
   the corrupted-arm majority gets a content-level control the shape guard
   cannot provide: a scrambled encode whose shape is coincidentally legal
   reads a far worse artifact MAE than its direct-chain arm, same pool, same
   chain.

Encoding scope: the raw-cohort lists (``--train-list`` + ``--extra-list``, P1
primary 7404 + MR-RATE replay 7404 + the dev cohort 1060 the P2/P3 arms
consume). The P2/P3 arm lists (``--emb-list``) reference embeddings by their
legacy-tree paths instead of raws; they join the reconciliation and the guard
-- proving those lists satisfiable against the new tree -- but drive no
encoding, their artifacts being a subset of the raw cohorts'.

``variant=diagnostic`` throughout: the frozen VAE is read-only, the evaluation
chain and the judgment lines are untouched, and the report lands in the sugon
artifact area, never git. Bootstrap seeds ride T4's diagnostic block (base
+400..407, KNOWN DEBT pending the challenge_registry follow-up). The sugon
host recipe lives at ``deploy/jobs/run_embedding_reencode_t2.sh``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError, DiagnosticReportWriter
from ctmr.application.acceptance.distribution.intensity_domain import VaeReconstructor
from ctmr.application.generation.modality_label.reencode import (
    ARTIFACT_METRICS,
    DIRECT_METRICS,
    JOB_C_REFERENCE,
    EmbeddingManifest,
    ReencodeSelfCheck,
)

if TYPE_CHECKING:
    from ctmr.domain.engine import GenerationEngine

# The frozen autoencoder_v1's latent geometry: 4x spatial downsample (job C /
# T4's read_case uses the same factor in reverse: grid = latent * 4), 4 latent
# channels written channels-last.
LATENT_DOWNSAMPLE = 4
LATENT_CHANNELS = 4
# The vendored encode chain's grid base (create_training_data.round_number) —
# re-stated here because the modality_label family imports zero infrastructure
# (ADR-0019 §1, #272); the local restatement is pinned pointwise against the
# chain's own rounding in tests/application/generation/modality_label/
# test_reencode_ras.py.
ENCODE_GRID_BASE = 128


class EmbeddingShapeGuard:
    """The canonical-shape contract of the new tree: an embedding's shape must
    equal its raw volume's RAS-reoriented spatial shape, rounded to the encode
    grid (multiples of 128) and divided by the latent downsample, channels
    last. Deriving the expectation from the raw -- not from a whitelist of
    accepted shapes -- pins each artifact to its own volume, so an
    axis-scrambled encode fails its guard even when its shape happens to be a
    legal one elsewhere in the corpus."""

    @staticmethod
    def round_to_encode_grid(size: int) -> int:
        """Round one axis to the encode grid: nearest 128-multiple, floored at
        one base -- the vendored chain's ``round_number`` semantics, restated
        locally (layering) and pinned pointwise in the tests."""
        return int(max(round(float(size) / float(ENCODE_GRID_BASE)), 1.0) * float(ENCODE_GRID_BASE))

    def derive_expected_latent_shape(self, raw_path) -> tuple:
        """The expected latent shape for one raw volume: RAS-reorient (nibabel's
        canonical form -- the same reorientation the encode chain's
        ``Orientationd(axcodes="RAS")`` applies; pinned equal in
        tests/infrastructure/maisi_engine/test_engine_smoke.py), squeeze
        trailing singleton dims (MONAI's LoadImage does the same), round to the
        encode grid, divide by the latent downsample, append the channel axis."""
        import nibabel as nib

        image = nib.load(str(raw_path))
        shape = [int(s) for s in nib.as_closest_canonical(image).shape]
        while len(shape) > 3 and shape[-1] == 1:
            shape.pop()
        return tuple(self.round_to_encode_grid(size) // LATENT_DOWNSAMPLE for size in shape) + (LATENT_CHANNELS,)

    def check(self, pairs, shapes_by_path) -> list:
        """The violations among ``[(emb_rel, raw_path)]`` given the manifest's
        per-path shapes: a shape that disagrees with the derived expectation
        (or an unreadable artifact -- manifest shape null, or a raw volume the
        expectation cannot even be derived from) is a violation; absence is
        NOT -- the missing reconciliation owns that verdict."""
        rows = []
        for emb_rel, raw_path in pairs:
            if emb_rel not in shapes_by_path:
                continue
            actual = shapes_by_path[emb_rel]
            try:
                expected = list(self.derive_expected_latent_shape(raw_path))
            except (OSError, ValueError):
                rows.append({"path": emb_rel, "expected": None, "actual": actual, "reason": "unreadable_raw"})
                continue
            if actual != expected:
                rows.append(
                    {
                        "path": emb_rel,
                        "expected": expected,
                        "actual": actual,
                        "reason": "unreadable_shape" if actual is None else "shape_mismatch",
                    }
                )
        return rows


class EmbCohortList:
    """One P2/P3 arm list: entries reference already-encoded embeddings by
    their legacy-tree path (``embeddings/<case>/..._emb.nii.gz`` relative to
    the phase root), not raw volumes. The cohort resolves every referenced
    path -- ``image`` and, for cross-modal pairs, ``src_image`` -- to a
    (new-root-relative emb path, raw path) pair for the reconciliation and the
    guard, deduped in list order."""

    LEGACY_TREE_PREFIX = "embeddings"

    def __init__(self, path):
        self._path = Path(path)

    def pairs(self, data_root) -> list:
        listing = json.loads(self._path.read_text())["training"]
        pairs = []
        seen = set()
        for entry in listing:
            for field in ("image", "src_image"):
                rel = entry.get(field)
                if rel is None:
                    continue
                parts = PurePosixPath(rel).parts
                if not parts or parts[0] != self.LEGACY_TREE_PREFIX:
                    raise DiagnosticError(
                        f"{self._path.name}: embedding path `{rel}` does not start with the legacy tree prefix "
                        f"`{self.LEGACY_TREE_PREFIX}/` -- the P2/P3 lists resolve against the old tree; refusing to guess"
                    )
                emb_rel = "/".join(parts[1:])
                if emb_rel in seen:
                    continue
                seen.add(emb_rel)
                raw_rel = emb_rel[: -len("_emb.nii.gz")] + ".nii.gz"
                pairs.append((emb_rel, Path(data_root) / raw_rel))
        return pairs


class SidecarMultiSource:
    """The new tree's T7 sidecars, fetched per entry from the FIRST source root
    that carries a valid one -- the arms' sidecars live in different old roots
    (primary + replay cohorts' in the clip=True tree, the dev cohort's in the
    legacy tree). Statuses follow T4's vocabulary: ``present`` (already on the
    new root, valid), ``copied`` (fetched and valid), ``invalid`` (reachable
    copies exist but none valid), ``absent`` (no source carries the file at
    all)."""

    def __init__(self, emb_root, source_roots):
        self._emb_root = Path(emb_root)
        self._source_roots = [Path(root) for root in source_roots]

    def ensure(self, emb_rels) -> dict:
        statuses = {}
        for emb_rel in emb_rels:
            name = emb_rel + ".json"
            dst = self._emb_root / name
            if dst.is_file():
                statuses[emb_rel] = "present" if EmbeddingManifest.valid_sidecar(dst) else "invalid"
                continue
            copied = False
            saw_file = False
            for root in self._source_roots:
                src = root / name
                if not src.is_file():
                    continue
                saw_file = True
                dst.parent.mkdir(parents=True, exist_ok=True)  # the encode chain only creates dirs it wrote embeddings under
                shutil.copyfile(src, dst)
                if EmbeddingManifest.valid_sidecar(dst):
                    copied = True
                    break
            statuses[emb_rel] = "copied" if copied else ("invalid" if saw_file else "absent")
        return statuses


class ReencodeRasReport:
    """The job's diagnostic report artifact: the payload body (multi-arm
    reconciliation + shape-guard verdicts + spot-check) around the shared
    ``variant=diagnostic`` writer, with the markdown rendering (sugon artifact
    area, never git)."""

    SCHEMA = "embedding-reencode-ras/1"
    TITLE = "序列③T2:RAS 编码修复下的全量 embedding 重编码——多臂对账、形状守卫与域内自评抽检"
    # The spot-check metric labels, shared verbatim with the T4 report's rendering.
    METRIC_LABELS = {
        "clip_within": "工件臂 [0,1] 域内 → clip 后输入(滑窗链,落盘工件)",
        "clip_over": "工件臂 >1.0 外推 → clip 后输入(滑窗链,落盘工件)",
        "extrapolation_max": "noclip 世界外推高度(分层轴)",
        "raw_percentile_upper": "raw 99.5 百分位锚点",
        "direct_clip_within": "直通臂 [0,1] 域内 → clip 后输入(现场 encode,环境对照)",
        "direct_clip_over": "直通臂 >1.0 外推 → clip 后输入(现场 encode,环境对照)",
    }

    def __init__(self, inputs: dict, run_id: str | None):
        self._writer = DiagnosticReportWriter(
            schema=self.SCHEMA,
            title=self.TITLE,
            issue=312,
            parent_issue=310,
            job_label="序列③ T2",
            stem="embedding_reencode_ras_report",
            inputs=inputs,
            run_id=run_id,
        )

    def write(self, *, reencode: dict, check: dict | None, output_dir):
        payload = self._writer.payload(
            {
                "generated_utc_note": "variant=diagnostic;冻结 VAE 只读;评估链与判定线零改动;新树落盘,legacy/cliptrue 旧树均不动",
                "reencode": {**reencode, "manifest_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                "self_check": check,
            }
        )
        markdown = "\n".join(self._writer.markdown_preamble(payload)) + "\n" + self._markdown(payload) + "\n"
        return self._writer.write(payload, markdown, output_dir)

    def _markdown(self, payload):
        reencode = payload["reencode"]
        lines = [
            "## 重编码对账(多臂)",
            "",
            f"- 新 embedding 根:`{reencode['embedding_root']}`(legacy/cliptrue 旧树均不动)",
            f"- raw 臂:主 list 条目 {reencode['n_train_list']};附加 list(replay/dev)条目 {reencode['n_extra_list']}",
            f"- P2/P3 臂(emb 引用 list):{reencode.get('emb_cohort_lists') or '无'}——"
            f"唯一引用路径 {reencode['n_emb_cohort_paths']}(只对账+守卫,不驱动编码)",
            f"- 检查集(去重后){reencode['n_checked']};新树落盘 embedding {reencode['n_entries']};"
            f"缺失 {len(reencode['missing'])};形状守卫违例 {len(reencode['shape_violations'])}",
        ]
        sidecars = reencode.get("sidecars")
        if sidecars is not None:
            lines.append("- T7 sidecar(多源拷贝校验):" + ", ".join(f"{status} {count}" for status, count in sorted(sidecars.items())))
        missing = reencode["missing"]
        if missing:
            shown = ", ".join(missing[:10]) + ("…" if len(missing) > 10 else "")
            lines.append(f"- 缺失清单(前 10):{shown}")
        violations = reencode["shape_violations"]
        if violations:
            lines += [
                "",
                "### 形状守卫违例(前 20)",
                "",
                "期望形状 = raw 的 RAS 重排后 spatial_shape → round_number(128 基)→ ÷4 latent 下采样,channels-last。",
                "",
                "| path | 期望 | 实际 | 原因 |",
                "|---|---|---|---|",
            ]
            for row in violations[:20]:
                lines.append(f"| {row['path']} | {row['expected']} | {row['actual']} | {row['reason']} |")
            if len(violations) > 20:
                lines.append(f"| …(共 {len(violations)} 条,全文见 json)| | | |")
        lines += ["", "## 域内自评 MAE 抽检(clip=True 世界,decode 落盘工件)", ""]
        check = payload["self_check"]
        if check is None:
            lines.append("本次未运行(--self-check-limit 0)。")
            return "\n".join(lines)
        for pool_key, pool_label in (
            ("brats_t1c", "BraTS 主 list t1c 池(作业 C 锚同池口径)"),
            ("replay", "replay(MR-RATE)池(本票新增:错乱主力臂的内容级对照)"),
        ):
            pool = check[pool_key]
            lines += [
                f"### {pool_label}",
                "",
            ]
            if pool is None:
                lines.append("该池无样本(0 例或全部排除)。")
                lines.append("")
                continue
            lines += [
                f"n_cases={pool['n_cases']}(排除 {pool['n_excluded']})。",
                "",
                "| 读数 | n_cases | median (q05, q95) | 分布包络 CI90 [low, high] | mean |",
                "|---|---:|---|---|---:|",
            ]
            for metric in (*ARTIFACT_METRICS, *DIRECT_METRICS):
                block = pool["aggregate"][metric]
                ci = "n/a" if block["median"] is None else f"{block['median']:.4f} ({block['q05']:.4f}, {block['q95']:.4f})"
                envelope = "n/a" if block.get("ci90_low") is None else f"[{block['ci90_low']:.4f}, {block['ci90_high']:.4f}]"
                mean = "n/a" if block["mean"] is None else f"{block['mean']:.4f}"
                lines.append(f"| {self.METRIC_LABELS[metric]} | {block['n_cases']} | {ci} | {envelope} | {mean} |")
        lines += [
            "",
            f"作业 C 双链锚(参照值非判定线):直通臂域内 median {JOB_C_REFERENCE['direct_clip_within_median']}、"
            f"外推层 {JOB_C_REFERENCE['direct_clip_over_median']};工件臂自评 {JOB_C_REFERENCE['artifact_within_median']}、"
            f"外推层 {JOB_C_REFERENCE['artifact_over_median']}(t1c 口径,braTS 池对锚)。",
            "replay 池无作业 C 锚:判读 = 工件臂(落盘工件 decode)与直通臂(现场 encode)的差——两臂同链同池,"
            "轴序错乱工件的内容级 MAE 会显著高于直通臂;同档即内容级自洽(形状守卫之外的独立防线)。",
        ]
        return "\n".join(lines)


def main(
    argv=None,
    *,
    encode_runner: Callable | None = None,
    decode: Callable | None = None,
    engine: GenerationEngine | None = None,
):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-list", required=True, help="the P1 image-only primary list json (raw cohort, drives encoding)")
    parser.add_argument(
        "--extra-list",
        action="append",
        default=[],
        help="additional raw-cohort list(s) encoded into and reconciled against the SAME embedding root "
        "(the MR-RATE replay list, the dev cohort list; repeatable)",
    )
    parser.add_argument(
        "--emb-list",
        action="append",
        default=[],
        help="P2/P3-style list(s) whose entries reference embeddings by their legacy-tree path "
        "(reconciled + guarded against the new tree, but driving no encoding; repeatable)",
    )
    parser.add_argument("-e", "--env-config", required=True, help="env json with embedding_base_dir = the NEW embedding root")
    parser.add_argument("-c", "--model-config", required=True, help="model config json")
    parser.add_argument("-t", "--model-def", required=True, help="network def json")
    parser.add_argument("--device", default="cpu", help="torch device for the spot-check decode arm (cpu / cuda:0)")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPU count handed to the encode chain")
    parser.add_argument(
        "--embedding-root",
        default=None,
        help="override for the new embedding root; must equal the env json's embedding_base_dir (re-encode target, never an old tree)",
    )
    parser.add_argument(
        "--sidecar-source",
        action="append",
        default=[],
        help="retained old embedding root(s) to fetch the T7 sidecars (<emb>.json, spacing/modality) from, in fallback order; "
        "when given, an entry without a usable sidecar reconciles missing",
    )
    parser.add_argument("--self-check-limit", type=int, default=200, help="uniform-stride spot-check subsample (0 switches it off)")
    parser.add_argument("--bootstrap-b", type=int, default=10000, help="bootstrap resamples for the spot-check CI90")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the report (never git)")
    parser.add_argument("--run-id", default=None, help="the retrain binding's run id, recorded into the report")
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="shard-worker pass: run only the encode chain (skips existing files), no manifest/guard/spot-check/report; "
        "the finishing pass runs with --skip-encode",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="finishing pass: skip the encode chain (already run by shard workers) and run manifest + guard + spot-check + report",
    )
    args = parser.parse_args(argv)

    if args.encode_only and args.skip_encode:
        raise DiagnosticError("--encode-only and --skip-encode are mutually exclusive")

    env = json.loads(Path(args.env_config).read_text())
    embedding_root = Path(args.embedding_root or env["embedding_base_dir"])
    if args.embedding_root is not None and embedding_root != Path(env["embedding_base_dir"]):
        raise DiagnosticError(
            f"--embedding-root {args.embedding_root} disagrees with the env json's "
            f"embedding_base_dir {env['embedding_base_dir']} -- the manifest must describe the root the encode chain wrote"
        )

    primary = json.loads(Path(args.train_list).read_text())["training"]
    extras = [entry for path in args.extra_list for entry in json.loads(Path(path).read_text())["training"]]
    raw_entries = [*primary, *extras]
    data_root = Path(env["data_base_dir"])

    # Raw cohorts: the list's image paths ARE the raw volumes (relative to
    # data_base_dir); the new-tree embedding path mirrors the T4 layout.
    pairs_raw = [(entry["image"].replace(".nii.gz", "_emb.nii.gz"), data_root / entry["image"]) for entry in raw_entries]
    # P2/P3 cohorts: legacy-tree embedding references, resolved against the
    # new root for reconciliation and against the raw tree for the guard.
    pairs_emb = [pair for path in args.emb_list for pair in EmbCohortList(path).pairs(data_root)]
    n_emb_cohort_paths = len({emb_rel for emb_rel, _ in pairs_emb})

    # One check set, deduped in first-seen order: the P2/P3 references are a
    # subset of the raw cohorts in the planned corpus, but the job must not
    # ASSUME that -- any reference outside the raw cohorts is checked (and, if
    # unmet, reported) exactly the same way.
    all_pairs = list({emb_rel: raw for emb_rel, raw in [*pairs_raw, *pairs_emb]}.items())

    if encode_runner is None:
        from ctmr.wiring.generate import modality_label_reencode_runtime

        encode_runner = modality_label_reencode_runtime()[0]

    # Step 1: the vendored encode chain (fixed dim decision; skips existing
    # files -> re-entrant). Only the RAW cohorts drive encoding -- the chain
    # consumes the env json's json_data_list, so the raw corpus rides a derived
    # env (ephemeral -- reproducible from the operator env plus the lists).
    if not args.skip_encode:
        with tempfile.TemporaryDirectory(prefix="reencode_ras_chain_") as chain_dir:
            combined = Path(chain_dir) / "train_list_chain.json"
            combined.write_text(json.dumps({"training": raw_entries}))
            chain_env = dict(env)
            chain_env["json_data_list"] = str(combined)
            chain_env_path = Path(chain_dir) / "env_chain.json"
            chain_env_path.write_text(json.dumps(chain_env))
            encode_runner(str(chain_env_path), args.model_config, args.model_def, args.num_gpus)
        if not embedding_root.is_dir():
            raise DiagnosticError(f"embedding root still missing after the encode chain ran: {embedding_root}")
        if args.encode_only:
            print(f"[reencode-ras] encode-only pass complete (embedding root: {embedding_root})")
            return
    elif not embedding_root.is_dir():
        raise DiagnosticError(f"embedding root missing for the finishing pass: {embedding_root}")

    # Step 2: sidecars, the md5 manifest, the missing reconciliation, and the
    # shape guard -- the guard derives each entry's expectation from its own
    # raw volume, so an axis-scrambled encode cannot pass even with a legal
    # shape from elsewhere in the corpus.
    manifest = EmbeddingManifest(embedding_root)
    sidecar_statuses = None
    if args.sidecar_source:
        sidecar_statuses = SidecarMultiSource(embedding_root, args.sidecar_source).ensure([emb_rel for emb_rel, _ in all_pairs])
    manifest_rows = manifest.walk()
    manifest_path = manifest.write(manifest_rows)
    shapes_by_path = {row["path"]: row["shape"] for row in manifest_rows}
    missing = [emb_rel for emb_rel, _ in all_pairs if emb_rel not in shapes_by_path]
    if sidecar_statuses is not None:
        missing += [rel for rel, status in sidecar_statuses.items() if status in ("absent", "invalid") and rel not in missing]
    violations = EmbeddingShapeGuard().check(all_pairs, shapes_by_path)

    # Step 3: the in-domain spot-check, two pools. The BraTS t1c pool rides
    # T4's convention (job C's anchors are t1c-only readings); the replay pool
    # is this job's addition -- the corrupted-arm majority gets a CONTENT-level
    # control the shape guard cannot provide: a scrambled encode with a
    # coincidentally legal shape reads a far worse artifact MAE than its
    # direct-chain arm, same pool, same chain.
    check_body = None
    if args.self_check_limit != 0:
        if decode is None:
            from ctmr.wiring.generate import modality_label_reencode_runtime

            runtime_engine = engine if engine is not None else modality_label_reencode_runtime()[1]
            reconstructor = VaeReconstructor(args.env_config, args.model_config, args.model_def, args.device, engine=runtime_engine)
            decode = reconstructor.decode
            encode = reconstructor.encode
        else:
            encode = None
        pool = ReencodeSelfCheck(env["data_base_dir"], embedding_root, decode, encode=encode)
        t1c_entries = [entry for entry in primary if entry["image"].endswith("-t1c.nii.gz")]
        replay_entries = [entry for entry in extras if entry["image"].startswith("MR-RATE/")]
        check_body = {
            "brats_t1c": pool.summarize(pool.read_cases(pool.stride(t1c_entries, args.self_check_limit)), bootstrap_b=args.bootstrap_b)
            if t1c_entries
            else None,
            "replay": pool.summarize(pool.read_cases(pool.stride(replay_entries, args.self_check_limit)), bootstrap_b=args.bootstrap_b)
            if replay_entries
            else None,
        }

    report = ReencodeRasReport(
        inputs={
            "train_list": str(args.train_list),
            "extra_lists": ", ".join(args.extra_list) if args.extra_list else None,
            "emb_lists": ", ".join(args.emb_list) if args.emb_list else None,
            "sidecar_sources": ", ".join(args.sidecar_source) if args.sidecar_source else None,
            "env_config": str(args.env_config),
            "embedding_root": str(embedding_root),
            "manifest": str(manifest_path),
        },
        run_id=args.run_id,
    )
    report.write(
        reencode={
            "embedding_root": str(embedding_root),
            "n_train_list": len(primary),
            "n_extra_list": len(extras),
            "emb_cohort_lists": args.emb_list,
            "n_emb_cohort_paths": n_emb_cohort_paths,
            "n_checked": len(all_pairs),
            "sidecars": dict(Counter(sidecar_statuses.values())) if sidecar_statuses is not None else None,
            "n_entries": len(manifest_rows),
            "missing": missing,
            "shape_violations": violations,
        },
        check=check_body,
        output_dir=args.output_dir,
    )
    print(f"[reencode-ras] manifest: {manifest_path} ({len(manifest_rows)} entries, missing={len(missing)}, shape_violations={len(violations)})")
    json_path, md_path = Path(args.output_dir) / "embedding_reencode_ras_report.json", Path(args.output_dir) / "embedding_reencode_ras_report.md"
    print(f"[reencode-ras] report: {json_path}")
    print(f"[reencode-ras] report: {md_path}")
    if missing:
        raise DiagnosticError(
            f"{len(missing)} entries are unusable after the re-encode (embedding or T7 sidecar missing/invalid), "
            f"first: {missing[:3]} -- the tree cannot serve its lists; see the report for the full list"
        )
    if violations:
        first = violations[0]
        raise DiagnosticError(
            f"{len(violations)} embeddings violate the canonical-shape guard (expected {first['expected']}, "
            f"got {first['actual']} at {first['path']}, …) -- axis-scrambled or mis-sized encodes must not serve "
            "training; see the report for the full list"
        )


if __name__ == "__main__":
    main()
