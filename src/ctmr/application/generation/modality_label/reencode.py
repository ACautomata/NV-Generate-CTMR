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

"""Embedding re-encode job (issue #251, series-② T4): the clip=True recipe's
training-corpus pass.

The T4 recipe change (the mri normalization flag ``clip=False -> clip=True``,
instance_definition's one recorded deviation) invalidated the clip=False-encoded
training embeddings: in the clip=True world they are not reusable, so the whole
P1 image-only training list must be re-encoded before the T7 retrain. This job
is that pass, in three steps:

1. Re-encode: the vendored ``diff_model_create_training_data`` chain (now under
   the clip=True factory) runs against an env json whose ``embedding_base_dir``
   points at the NEW embedding root -- the old tree stays untouched (it is the
   pre-T4 world's artifact and the retrain's fallback anchor). The chain skips
   files that already exist, so the job re-enters after an interruption.
2. Manifest: the new tree is walked and every ``*_emb.nii.gz`` registered with
   its md5/bytes/shape into ``manifest.jsonl`` beside it; entries the encode
   pass failed to produce are reconciled against the training list as
   ``missing`` -- never silently dropped -- and a missing-nonempty job exits
   nonzero (after the report lands): a tree the downstream list cannot
   satisfy must never exit green. T7's loader also reads ``<emb>.json``
   sidecars (spacing/modality) the encode chain never writes, so with
   ``--sidecar-source`` the job copies each sidecar from the retained old
   root and validates it; an entry without a usable sidecar reconciles
   missing exactly like a missing embedding.
3. Spot-check: the t1c subset of the primary list (job C's anchors are
   t1c-only readings; a mixed-modality pool would compare t1n/t2w/t2f
   reconstructions against t1c anchors) is stride-sampled, decoded from the
   fresh embedding, and read as the conditioned MAE against the clip-encoded
   input, tiered by the noclip world's extrapolation bands (job C's clip=True
   reading convention) -- plus a direct-chain control arm that re-encodes
   in-process, which must reproduce job C's 0.006-magnitude in-domain MAE to
   prove the environment (VAE load / normalization / resize) still matches
   job C's. The artifact arm's own anchor is job C's 0.0823 in-domain reading
   on the retained sliding-window embeddings: same magnitude means the new
   artifacts are chain-identical to the old ones, clip aside.

The primary list covers the BraTS corpus; T7's DataCatalog concatenates a
replay list with it and resolves both cohorts under one embedding root, so
``--extra-list`` joins a replay cohort into the encode/manifest/reconciliation
scope while the spot-check pool stays the primary list's t1c subset.

``variant=diagnostic`` throughout: the frozen VAE is read-only, the evaluation
chain and the judgment lines are untouched, and the report lands in the sugon
artifact area, never git. Bootstrap seeds draw the diagnostic namespace at the
T4 block (base+400..407, slot offsets reusing job C's MAE slots) -- KNOWN DEBT
alongside jobs C/D's bandless blocks, pending their challenge_registry
follow-up. The sugon host recipe lives at ``deploy/jobs/run_embedding_reencode_t4.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ctmr.application.acceptance.distribution.challenge_registry import DIAGNOSTIC_SEED_BASE
from ctmr.application.acceptance.distribution.diagnostic_support import DiagnosticError, DiagnosticReportWriter
from ctmr.application.acceptance.distribution.intensity_domain import (
    MAE_SEED_SLOTS,
    TieredIntensityStats,
    TrainingPreprocessing,
    VaeReconstructor,
)

if TYPE_CHECKING:
    from ctmr.domain.engine import GenerationEngine

# The T4 block of the diagnostic seed namespace (see module docstring): slot
# offsets ride job C's MAE slots so the two jobs' per-metric streams stay
# number-identical in meaning.
REENCODE_SEED_SLOT = 400
ARTIFACT_METRICS = ("clip_within", "clip_over", "extrapolation_max", "raw_percentile_upper")
DIRECT_METRICS = ("direct_clip_within", "direct_clip_over")

# Job C's fifth-round medians (deploy/experiments/
# 20260829-P1根因甄别-作业C-t1c强度域甄别.md): the magnitude anchors this job's
# spot-check reads against. Two chains must not be conflated -- job C's clip
# arm encoded in-process (direct chain, no sliding window) while the training
# artifacts ride the sliding-window encode chain, whose in-domain fidelity job
# C's noclip arm measured at 0.0823 on the retained embeddings themselves.
# Reference constants, never a pass line.
JOB_C_REFERENCE = {
    # the direct chain's clip=True arm (job C rows "clip=True 对照臂")
    "direct_clip_within_median": 0.0062,
    "direct_clip_over_median": 0.0559,
    # the artifact chain's own self-eval (job C rows "clip=False 现网臂", the
    # retained sliding-window embeddings decoded back)
    "artifact_within_median": 0.0823,
    "artifact_over_median": 0.8673,
}


class EmbeddingManifest:
    """The re-encoded embedding tree's md5 manifest: walk, register, reconcile.

    One row per ``*_emb.nii.gz`` with md5, byte size and the channels-last
    latent shape. An unreadable artifact is recorded with ``shape: null`` -- a
    manifest that drops entries would hide the very failures the reconciliation
    exists to catch. Rows sort by relative path so the manifest is byte-stable
    across reruns; it lands as ``manifest.jsonl`` beside the embeddings, where
    the T7 retrain consumes it as the artifact audit trail."""

    FILENAME = "manifest.jsonl"

    def __init__(self, emb_root):
        self._emb_root = Path(emb_root)

    def walk(self, threads: int = 1):
        """Walk the tree into manifest rows, path-sorted (byte-stable). ``threads``
        parallelizes the per-file reads (md5 + header) across a thread pool --
        pure IO, order untouched; the default 1 keeps the T4 serial walk."""
        paths = sorted(self._emb_root.rglob("*_emb.nii.gz"))
        if threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=threads) as executor:
                return list(executor.map(self._row_of, paths))
        return [self._row_of(path) for path in paths]

    def _row_of(self, path):
        import nibabel as nib
        from nibabel.filebasedimages import ImageFileError

        row = {"path": path.relative_to(self._emb_root).as_posix(), "bytes": path.stat().st_size, "shape": None}
        row["md5"] = self._md5_of(path)
        try:
            row["shape"] = [int(s) for s in nib.load(str(path)).shape]
        except (OSError, ValueError, ImageFileError):
            pass
        return row

    def write(self, rows):
        """Register the walked rows beside the embeddings; returns the manifest path."""
        manifest = self._emb_root / self.FILENAME
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        manifest.write_text("\n".join(lines) + ("\n" if lines else ""))
        return manifest

    def missing_entries(self, entries, sidecar_statuses=None):
        """Entries of the training list with no usable artifact on disk, in list
        order -- the encode pass's failures, reconciled instead of dropped. A
        sidecar-status map (from :meth:`ensure_sidecars`) tightens the check:
        the T7 loader needs embedding AND sidecar, so an entry without a valid
        sidecar is as unusable as one without an embedding."""
        missing = [entry["image"] for entry in entries if not (self._emb_root / entry["image"].replace(".nii.gz", "_emb.nii.gz")).is_file()]
        if sidecar_statuses is not None:
            missing += [image for image, status in sidecar_statuses.items() if status in ("absent", "invalid") and image not in missing]
        return missing

    @staticmethod
    def _sidecar_name(image):
        return image.replace(".nii.gz", "_emb.nii.gz") + ".json"

    @staticmethod
    def valid_sidecar(path):
        """The T7 loader reads spacing/modality out of the sidecar unconditionally;
        a sidecar that does not parse with both keys is loader-unusable. Public:
        the series-③ T2 job's multi-source copier (reencode_ras) reuses it."""
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return False
        return "spacing" in payload and "modality" in payload

    def ensure_sidecars(self, entries, source_root):
        """Copy each entry's ``<emb>.json`` sidecar from the retained old root
        into the new one and validate it (the encode chain writes embeddings
        only). Returns the per-entry status map: ``present`` (already on the
        new root, valid), ``copied`` (fetched and valid), ``invalid`` (on root
        or fetched but unusable), ``absent`` (no source sidecar)."""
        statuses = {}
        for entry in entries:
            name = self._sidecar_name(entry["image"])
            dst = self._emb_root / name
            if dst.is_file():
                statuses[entry["image"]] = "present" if self.valid_sidecar(dst) else "invalid"
                continue
            src = Path(source_root) / name
            if not src.is_file():
                statuses[entry["image"]] = "absent"
                continue
            shutil.copyfile(src, dst)
            statuses[entry["image"]] = "copied" if self.valid_sidecar(dst) else "invalid"
        return statuses

    @staticmethod
    def _md5_of(path):
        digest = hashlib.md5()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


class ReencodeSelfCheck:
    """The spot-check pool: reads the raw t1c + the fresh embedding per sampled
    case and reads the conditioned MAE against the injected decode arm, plus a
    direct-chain control arm.

    Each case rides job C's clip=True arm: the clip-encoded input is the decode
    target, the tier masks come from the noclip world (the >1.0 extrapolated
    band vs [0, 1] -- the tiering axis that made job C's two arms comparable),
    and the artifact arm's latent is the freshly written embedding file itself
    so the reading certifies the on-disk artifact, not a transient tensor. The
    optional ``encode`` arm re-encodes the clip input in-process (job C's
    direct chain, no sliding window) -- the environment-consistency control:
    it must reproduce job C's 0.006-magnitude in-domain MAE, or the VAE load /
    normalization / resize environment drifted and the artifact reading cannot
    be compared against the job C anchors."""

    def __init__(self, data_root, emb_root, decode: Callable, encode: Callable | None = None):
        self._data_root = Path(data_root)
        self._emb_root = Path(emb_root)
        self._decode = decode
        self._encode = encode

    @staticmethod
    def stride(entries, limit):
        """Uniform stride subsample (job C's convention): keep the list's position
        spread at diagnostic scale instead of taking a head slice. ``None``, 0 or
        a cap past the end reads as no cap."""
        if limit is None or limit <= 0 or limit >= len(entries):
            return list(entries)
        step = max(1, len(entries) // limit)
        return list(entries)[::step][:limit]

    def read_cases(self, entries):
        return [self.read_case(entry["image"]) for entry in entries]

    def read_case(self, rel):
        """One case's spot-check row off disk, or an excluded row with a stated
        reason -- a missing reading is a measurement result, never a zero."""
        import nibabel as nib

        case = Path(rel).name.replace("-t1c.nii.gz", "")
        emb_path = self._emb_root / rel.replace(".nii.gz", "_emb.nii.gz")
        try:
            t1c = nib.load(str(self._data_root / rel)).get_fdata(dtype=np.float32)
            latent = nib.load(str(emb_path)).get_fdata(dtype=np.float32)
        except (OSError, FileNotFoundError, ValueError):
            return {"case": case, "excluded": "unreadable_inputs", "mae": None}
        if latent.ndim != 4:
            return {"case": case, "excluded": "embedding_not_channels_last", "mae": None}
        grid = tuple(int(s) * 4 for s in latent.shape[:3])
        return self.measure(case, t1c, latent, grid)

    def measure(self, case, t1c, latent, grid):
        """The pure reading behind one spot-check row (job C's clip=True arm
        against the caller's decode arm; the direct-chain control arm when an
        encode callable is injected)."""
        row = {"case": case, "excluded": None, "mae": None}

        def exclude(reason):
            row["excluded"] = reason
            return row

        try:
            norm_clip, _lower, upper = TrainingPreprocessing.normalize_percentile(t1c, clip=True)
            norm_noclip, _lower_n, _upper_n = TrainingPreprocessing.normalize_percentile(t1c, clip=False)
        except ValueError:
            return exclude("degenerate_percentile_range")
        input_clip = TrainingPreprocessing.resize_image(norm_clip, grid, "trilinear")
        input_noclip = TrainingPreprocessing.resize_image(norm_noclip, grid, "trilinear")
        recon = self._decode(np.asarray(latent, dtype=np.float32))
        if recon.shape != input_clip.shape:
            return exclude("recon_shape_mismatch")
        hi = input_noclip > 1.0
        lo = (input_noclip >= 0.0) & (input_noclip <= 1.0)
        conditioned = TieredIntensityStats.conditioned_mae(input_clip, recon, hi_mask=hi, lo_mask=lo)
        row["mae"] = {
            "clip_within": conditioned["mae_within"],
            "clip_over": conditioned["mae_over"],
            "n_within": conditioned["n_within"],
            "n_over": conditioned["n_over"],
            "extrapolation_max": float(input_noclip.max()),
            "raw_percentile_upper": upper,
        }
        if self._encode is not None:
            direct = TieredIntensityStats.conditioned_mae(input_clip, self._decode(self._encode(input_clip)), hi_mask=hi, lo_mask=lo)
            row["mae"]["direct_clip_within"] = direct["mae_within"]
            row["mae"]["direct_clip_over"] = direct["mae_over"]
        return row

    def summarize(self, rows, bootstrap_b):
        """The pool's aggregate body for the report: per-metric cross-case
        distributions with diagnostic-seed CI90, plus job C's magnitude anchors."""
        kept = [row for row in rows if row["excluded"] is None]
        aggregate = {}
        for metric in (*ARTIFACT_METRICS, *DIRECT_METRICS):
            values = [row["mae"][metric] for row in kept if row["mae"] is not None and row["mae"].get(metric) is not None]
            seed = DIAGNOSTIC_SEED_BASE + REENCODE_SEED_SLOT + MAE_SEED_SLOTS[metric.replace("direct_", "")]
            aggregate[metric] = TieredIntensityStats.distribution_stats(values, bootstrap_b=bootstrap_b, seed=seed)
        return {
            "n_cases": len(kept),
            "n_excluded": len(rows) - len(kept),
            "excluded_reasons": dict(Counter(row["excluded"] for row in rows if row["excluded"] is not None)),
            "aggregate": aggregate,
            "per_case": rows,
            "job_c_reference": {
                **JOB_C_REFERENCE,
                "note": "作业 C 第五轮 median(deploy/experiments/20260829-P1根因甄别-作业C-t1c强度域甄别.md);"
                "量级参照,非判定线。两条链不可混读:direct_* 臂=作业 C 直通链(现场 encode,复现 0.0062 即环境一致);"
                "clip_* 臂=落盘工件(滑窗链,与旧工件同链——0.0823 是作业 C 对既有工件的同链自评,"
                "clip_over 应显著低于旧工件外推层 0.8673 且外推层被截到 1.0)",
            },
        }


class ReencodeReport:
    """The job's diagnostic report artifact: the payload body (re-encode
    reconciliation + spot-check) around the shared ``variant=diagnostic``
    writer, with the markdown rendering (sugon artifact area, never git)."""

    SCHEMA = "embedding-reencode/1"
    TITLE = "序列②T4:训练 embedding 重编码(clip=True)——md5 清单与域内自评 MAE 抽检"
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
            issue=251,
            job_label="序列② T4",
            stem="embedding_reencode_report",
            inputs=inputs,
            run_id=run_id,
        )

    def write(self, *, reencode: dict, check: dict | None, output_dir):
        payload = self._writer.payload(
            {
                "generated_utc_note": "variant=diagnostic;冻结 VAE 只读;评估链与判定线零改动",
                "reencode": {**reencode, "manifest_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                "self_check": check,
            }
        )
        markdown = "\n".join(self._writer.markdown_preamble(payload)) + "\n" + self._markdown(payload) + "\n"
        return self._writer.write(payload, markdown, output_dir)

    def _markdown(self, payload):
        reencode = payload["reencode"]
        lines = [
            "## 重编码对账",
            "",
            f"- 新 embedding 根:`{reencode['embedding_root']}`",
            f"- 训练 list 条目 {reencode['n_train_list']};附加 list(如 replay)条目 {reencode.get('n_extra_list', 0)}"
            f";落盘 embedding {reencode['n_entries']};缺失 {len(reencode['missing'])}",
        ]
        sidecars = reencode.get("sidecars")
        if sidecars is not None:
            lines.append("- T7 sidecar(拷贝自旧根并校验):" + ", ".join(f"{status} {count}" for status, count in sorted(sidecars.items())))
        missing = reencode["missing"]
        if missing:
            shown = ", ".join(missing[:10]) + ("…" if len(missing) > 10 else "")
            lines.append(f"- 缺失清单(前 10):{shown}")
        lines += ["", "## 域内自评 MAE 抽检(clip=True 世界,decode 落盘工件)", ""]
        check = payload["self_check"]
        if check is None:
            lines.append("本次未运行(--self-check-limit 0)。")
            return "\n".join(lines)
        lines += [
            f"n_cases={check['n_cases']}(排除 {check['n_excluded']});作业 C 双链锚(deploy/experiments/"
            "20260829-P1根因甄别-作业C-t1c强度域甄别.md,参照值非判定线):直通链 clip 臂"
            f"域内 median {JOB_C_REFERENCE['direct_clip_within_median']}/外推层 median {JOB_C_REFERENCE['direct_clip_over_median']};"
            f"滑窗链工件自评(旧 embedding 解码)域内 median {JOB_C_REFERENCE['artifact_within_median']}/"
            f"外推层 median {JOB_C_REFERENCE['artifact_over_median']}。",
            "",
            "| 读数 | n_cases | median (q05, q95) | 分布包络 CI90 [low, high] | mean |",
            "|---|---:|---|---|---:|",
        ]
        for metric in (*ARTIFACT_METRICS, *DIRECT_METRICS):
            block = check["aggregate"][metric]
            ci = "n/a" if block["median"] is None else f"{block['median']:.4f} ({block['q05']:.4f}, {block['q95']:.4f})"
            envelope = "n/a" if block.get("ci90_low") is None else f"[{block['ci90_low']:.4f}, {block['ci90_high']:.4f}]"
            mean = "n/a" if block["mean"] is None else f"{block['mean']:.4f}"
            lines.append(f"| {self.METRIC_LABELS[metric]} | {block['n_cases']} | {ci} | {envelope} | {mean} |")
        lines += [
            "",
            "## 逐 case 明细",
            "",
            "| case | 工件臂 clip_within | 工件臂 clip_over | 直通臂 within | 直通臂 over | 外推高 max | raw 99.5p | n_over/n_within | 排除 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in check["per_case"]:
            mae = row["mae"] or {}

            def fmt(key):
                value = mae.get(key)
                return "n/a" if value is None else f"{value:.4f}"

            lines.append(
                f"| {row['case']} | {fmt('clip_within')} | {fmt('clip_over')} | {fmt('direct_clip_within')} "
                f"| {fmt('direct_clip_over')} | {fmt('extrapolation_max')} | {fmt('raw_percentile_upper')} "
                f"| {mae.get('n_over', '')}/{mae.get('n_within', '')} | {row['excluded'] or ''} |"
            )
        return "\n".join(lines)


def main(
    argv=None,
    *,
    encode_runner: Callable | None = None,
    decode: Callable | None = None,
    engine: GenerationEngine | None = None,
):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-list", required=True, help="the P1 image-only training list json (the spot-check pool's source)")
    parser.add_argument(
        "--extra-list",
        action="append",
        default=[],
        help="additional cohort list(s) encoded into and reconciled against the SAME embedding root "
        "(T7's DataCatalog concatenates the replay list with the primary list; repeatable)",
    )
    parser.add_argument("-e", "--env-config", required=True, help="env json with embedding_base_dir = the NEW embedding root")
    parser.add_argument("-c", "--model-config", required=True, help="model config json")
    parser.add_argument("-t", "--model-def", required=True, help="network def json")
    parser.add_argument("--device", default="cpu", help="torch device for the spot-check decode arm (cpu / cuda:0)")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPU count handed to the encode chain")
    parser.add_argument(
        "--embedding-root",
        default=None,
        help="override for the new embedding root; must equal the env json's embedding_base_dir (re-encode target, never the old tree)",
    )
    parser.add_argument(
        "--sidecar-source",
        default=None,
        help="retained old embedding root to copy the T7 sidecars (<emb>.json, spacing/modality) from; "
        "when given, an entry without a usable sidecar reconciles missing",
    )
    parser.add_argument("--self-check-limit", type=int, default=200, help="uniform-stride spot-check subsample (0 switches it off)")
    parser.add_argument("--bootstrap-b", type=int, default=10000, help="bootstrap resamples for the spot-check CI90")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the report (never git)")
    parser.add_argument("--run-id", default=None, help="the retrain binding's run id, recorded into the report")
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="shard-worker pass: run only the encode chain (skips existing files), no manifest/spot-check/report; "
        "the finishing pass runs with --skip-encode",
    )
    parser.add_argument(
        "--skip-encode",
        action="store_true",
        help="finishing pass: skip the encode chain (already run by shard workers) and run manifest + spot-check + report",
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
    entries = [*primary, *extras]

    if encode_runner is None:
        from ctmr.wiring.generate import modality_label_reencode_runtime

        encode_runner = modality_label_reencode_runtime()[0]

    # Step 1: the vendored encode chain (skips existing files -> re-entrant);
    # it owns creating the embedding root. The chain consumes the env json's
    # json_data_list, so the combined corpus rides a derived env (ephemeral --
    # reproducible from the operator env plus the lists). The shard-worker
    # pass stops here; its --train-list is the shard list, so the derived env
    # covers exactly that shard.
    if not args.skip_encode:
        with tempfile.TemporaryDirectory(prefix="reencode_chain_") as chain_dir:
            combined = Path(chain_dir) / "train_list_chain.json"
            combined.write_text(json.dumps({"training": entries}))
            chain_env = dict(env)
            chain_env["json_data_list"] = str(combined)
            chain_env_path = Path(chain_dir) / "env_chain.json"
            chain_env_path.write_text(json.dumps(chain_env))
            encode_runner(str(chain_env_path), args.model_config, args.model_def, args.num_gpus)
        if not embedding_root.is_dir():
            raise DiagnosticError(f"embedding root still missing after the encode chain ran: {embedding_root}")
        if args.encode_only:
            print(f"[reencode] encode-only pass complete (embedding root: {embedding_root})")
            return
    elif not embedding_root.is_dir():
        raise DiagnosticError(f"embedding root missing for the finishing pass: {embedding_root}")

    # Step 2: the T7 sidecars (the encode chain never writes them) and the md5
    # manifest, registered beside the new embeddings. Sidecar enforcement is
    # opt-in via --sidecar-source; the report records the statuses either way.
    manifest = EmbeddingManifest(embedding_root)
    sidecar_statuses = manifest.ensure_sidecars(entries, args.sidecar_source) if args.sidecar_source else None
    manifest_rows = manifest.walk()
    manifest_path = manifest.write(manifest_rows)
    missing = manifest.missing_entries(entries, sidecar_statuses=sidecar_statuses)

    # Step 3: the in-domain spot-check (decode arm over the fresh artifacts,
    # plus the direct-chain control arm for the environment-consistency check).
    # The pool is the PRIMARY list's t1c subset -- job C's anchors are t1c-only
    # readings; the reconciliation above stays over the combined corpus.
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
        rows = pool.read_cases(pool.stride(t1c_entries, args.self_check_limit))
        check_body = pool.summarize(rows, bootstrap_b=args.bootstrap_b)

    report = ReencodeReport(
        inputs={
            "train_list": str(args.train_list),
            "extra_lists": ", ".join(args.extra_list) if args.extra_list else None,
            "sidecar_source": args.sidecar_source,
            "env_config": str(args.env_config),
            "embedding_root": str(embedding_root),
            "manifest": str(manifest_path),
        },
        run_id=args.run_id,
    )
    json_path, md_path = report.write(
        reencode={
            "embedding_root": str(embedding_root),
            "n_train_list": len(primary),
            "n_extra_list": len(extras),
            "sidecars": dict(Counter(sidecar_statuses.values())) if sidecar_statuses is not None else None,
            "n_entries": len(manifest_rows),
            "missing": missing,
        },
        check=check_body,
        output_dir=args.output_dir,
    )
    print(f"[reencode] manifest: {manifest_path} ({len(manifest_rows)} entries, missing={len(missing)})")
    print(f"[reencode] report: {json_path}")
    print(f"[reencode] report: {md_path}")
    if missing:
        raise DiagnosticError(
            f"{len(missing)} entries are unusable after the re-encode (embedding or T7 sidecar missing/invalid), "
            f"first: {missing[:3]} -- the tree cannot serve the training list; see the report for the full list"
        )


if __name__ == "__main__":
    main()
