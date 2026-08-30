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

"""Baseline: zero-training img2img sample-plan assembly and generation (issue #60).

The comparison floor of the cross-modal family. Runs the four-anchor-round
protocol over a manifest side with the frozen P1-DM (the run record's upstream
selection — nothing is trained): each real modality anchors one round, the
other three modalities are generated from ``x_t = (1-t)*src_latent + t*noise``
with the target modality label driving denoising (spec #51 decision 8). 12
ordered src->tgt pairs per case, exactly the #38 img2img recipe
re-pointed at the P1-DM checkpoint.

Outputs (controlled storage, shard-suffixed when ``--num-shards > 1``):

- ``generated/<CH>/<case>/a<src>/<tgt>_seed<seed>.nii.gz`` — baseline volumes;
- ``reference_grid/<CH>/<case>/<tgt>.nii.gz`` — real target volumes resampled
  onto the generation grid (RAS + trilinear 256x256x128, raw intensity domain),
  so the quantitative pair triplets share shape and affine by construction;
- ``samples<...>.json`` — distribution ``four-anchor-plan``-compatible entries (raw real
  paths; the distribution assembly resamples to the instrument grid itself);
- ``pairs<...>.json`` — the quantitative-side flat baseline/reference records.

Run order: ``--side dev`` first as a small smoke that only validates the
inference pipeline (its samples are a dev cot; they are not bound to the
contract). The run's freeze, and the quantitative/distribution deliverables it must be auditable
against, use the ``--side holdout`` samples manifest.

The plan-layer pieces (inference-config validation, run guard, sample plan
builder) are pure logic — stdlib only, no torch/nibabel; the manifest layer came
from the retired stage0-baseline plan entry and is testable off the DCU box.
The generation driver came from the retired stage0-baseline generate entry
(ticket 08, ADR-0015 §2); their ``selftest`` subcommands retired with them.
Per ADR-0016 the img2img chain runs on the domain ``DiffusionModel`` (issue
#173): the strength-truncated trajectory, the RF interpolation start
``x_t = (1-t)*src*scale_factor + t*noise`` and the CFG denoising loop are the
entity's ``begin_img2img`` + ``denoise`` behaviour; the VAE encode/decode and
the int16 post-processing stay application adapters.

Usage::

    ctmr generate cross-modal generate baseline \
        --run runs/p3-stage0-.../run.json \
        --manifest /ctrl/phase/phase_manifest.json \
        --out-root /ctrl/p3/stage0_holdout \
        --raw-root /ctrl/phase/raw \
        -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
        -c configs/config_maisi_diff_model_rflow-mr-brain.json \
        -t configs/config_network_rflow.json \
        --infer-config configs/config_p3_stage0_infer.json \
        [--side holdout] [--shard 0 --num-shards 8] [--limit N] [--challenge GLI]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import monai.transforms as monai_t
import nibabel as nib
import numpy as np
import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

from ctmr.application.generation.cross_modal.anchor import AnchorLatentEncoder
from ctmr.application.generation.cross_modal.plan import MODALITIES, seed_of
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.identity import WeightsRef
from ctmr.infrastructure.maisi_engine.diff_model_infer import load_models
from ctmr.infrastructure.maisi_engine.diff_model_setting import load_config, setup_logging
from ctmr.infrastructure.maisi_engine.inference_primitives import dynamic_infer
from ctmr.infrastructure.maisi_engine.utils_infer import ReconModel
from ctmr.infrastructure.weightsref import weights_ref_of_file

INFER_SCHEMA = "brats-p3-stage0-infer/1"
PAIRS_SCHEMA = "brats-p3-stage0-pairs/1"
BASELINE_VARIANT = "stage0-baseline"

GRID = (256, 256, 128)


class BaselinePlanError(Exception):
    """Raised when the inference config or the assembled plan breaks the baseline contract."""


class BaselineGenerateError(Exception):
    """Raised when the generation run breaks the baseline contract (cohort, layout, failed jobs)."""


class BaselineInferenceConfig:
    """The official baseline inference configuration (configs/config_p3_stage0_infer.json)."""

    def __init__(self, payload):
        if payload.get("schema") != INFER_SCHEMA:
            raise BaselinePlanError(f"baseline inference config schema must be {INFER_SCHEMA!r}")
        self.scheduler = self._text(payload, "scheduler")
        if self.scheduler != "RFlowScheduler":
            raise BaselinePlanError("baseline img2img only runs on the RFlow scheduler (rectified flow interpolation start)")
        self.num_inference_steps = self._positive_int(payload, "num_inference_steps")
        self.cfg_guidance_scale = self._number(payload, "cfg_guidance_scale")
        if self.cfg_guidance_scale < 0:
            raise BaselinePlanError("cfg_guidance_scale must be >= 0")
        self.strength = self._number(payload, "strength")
        if not 0.0 < self.strength < 1.0:
            raise BaselinePlanError("strength must lie strictly in (0, 1); 1 would erase the src latent, 0 would copy it")
        self.grid = self._grid(payload)
        self.modality_tokens = payload.get("modality_tokens")
        if not isinstance(self.modality_tokens, dict) or set(self.modality_tokens) != set(MODALITIES):
            raise BaselinePlanError(f"modality_tokens must cover exactly {list(MODALITIES)} (P1-DM tokens)")
        if not all(isinstance(token, int) and token > 0 for token in self.modality_tokens.values()):
            raise BaselinePlanError("modality tokens must be positive integers (P1-DM class labels)")
        self.seed_rule = self._text(payload, "seed_rule")

    @classmethod
    def from_path(cls, path):
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _text(payload, key):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise BaselinePlanError(f"baseline inference config needs a non-empty {key}")
        return value

    @staticmethod
    def _positive_int(payload, key):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise BaselinePlanError(f"baseline inference config {key} must be a positive integer")
        return value

    @staticmethod
    def _number(payload, key):
        value = payload.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BaselinePlanError(f"baseline inference config {key} must be numeric")
        return float(value)

    @staticmethod
    def _grid(payload):
        value = payload.get("grid")
        if not isinstance(value, list | tuple) or len(value) != 3 or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in value):
            raise BaselinePlanError("baseline inference config grid must be three positive integers (SRI24 256x256x128)")
        return tuple(value)


class BaselineRunGuard:
    """Checks the frozen baseline run record and pins the DM checkpoint + inference config.

    The generation-time recheck of the phase-run contract markers (issue #60
    acceptance criterion 1): the run must be the zero-training cross-modal
    baseline variant, its selection must pin the upstream P1-DM checkpoint, and
    the inference config used on the command line must byte-match the
    ``role=inference`` config the run pinned at init — so every generated volume
    traces back to the recorded official inference provenance.
    """

    def __init__(self, run_record, infer_config_path):
        self._record = run_record
        self._infer_config_path = Path(infer_config_path)

    def check(self):
        record = self._record
        if record.get("phase") != "P3" or record.get("variant") != BASELINE_VARIANT:
            raise BaselinePlanError(
                f"run {record.get('run_id')} is not a cross-modal {BASELINE_VARIANT} run; "
                "baseline generation must hang off its own zero-training record"
            )
        if record.get("status") != "frozen":
            raise BaselinePlanError(f"run {record['run_id']} is {record.get('status')}; generate samples only after the freeze")
        selection = record.get("selection") or {}
        upstream = record.get("upstream") or {}
        if selection.get("checkpoint", {}).get("sha256") != upstream.get("checkpoint", {}).get("sha256"):
            raise BaselinePlanError("baseline selection does not pin the upstream P1-DM checkpoint; the record is inconsistent")
        checkpoint = Path(upstream["checkpoint"]["path"])
        if not checkpoint.is_file():
            raise BaselinePlanError(f"pinned P1-DM checkpoint missing: {checkpoint}")
        if weights_ref_of_file(checkpoint) != WeightsRef(sha256=upstream["checkpoint"]["sha256"]):
            raise BaselinePlanError(f"pinned P1-DM checkpoint changed on disk: {checkpoint}")
        inference_entries = [entry for entry in record.get("configs", []) if entry.get("role") == "inference"]
        if len(inference_entries) != 1:
            raise BaselinePlanError(
                "the baseline run must pin exactly one --config inference=<official baseline inference config> "
                "(the recorded inference provenance, issue #60 acceptance criterion 1)"
            )
        if weights_ref_of_file(self._infer_config_path).sha256 != inference_entries[0]["sha256"]:
            raise BaselinePlanError(
                f"--infer-config sha256 does not match the config pinned by run {record['run_id']}; "
                "regenerate with the recorded official inference config"
            )
        return checkpoint

    @staticmethod
    def file_sha256(path):
        return weights_ref_of_file(path).sha256


class BaselineSamplePlanBuilder:
    """Builds the distribution-compatible four-anchor entries and the quantitative-side pair records.

    One real anchor per modality per case; each anchor round conditions the
    other three modalities (12 ordered src->tgt pairs per case, spec #51
    decision 8). Every carried path marker keeps the ``stage0-baseline``
    variant explicit so the manifests cannot masquerade as a trained
    candidate's evidence.
    """

    def __init__(self, run_id, dm_checkpoint_sha256, side, config):
        self._run_id = run_id
        self._dm_sha = dm_checkpoint_sha256
        self._side = side
        self._config = config

    def generated_path(self, generated_root, challenge, case, src, tgt):
        seed = seed_of(case, src, tgt)
        return Path(generated_root) / challenge / case / f"a{src}" / f"{tgt}_seed{seed}.nii.gz"

    def entries(self, cohort, real_of, generated_root):
        """``real_of(challenge, case, modality) -> path`` maps each case's real volumes."""
        entries = []
        for item in cohort:
            challenge, case = item["sub"], item["case"]
            anchors = {}
            for anchor in MODALITIES:
                generated = {}
                for tgt in MODALITIES:
                    if tgt == anchor:
                        continue
                    path = self.generated_path(generated_root, challenge, case, anchor, tgt)
                    generated[tgt] = {"path": str(path), "seed": seed_of(case, anchor, tgt)}
                anchors[anchor] = {"real": str(real_of(challenge, case, anchor)), "generated": generated}
            entries.append(
                {
                    "case_id": case,
                    "challenge": challenge,
                    "phase": "P3",
                    "variant": BASELINE_VARIANT,
                    "run_id": self._run_id,
                    "side": self._side,
                    "dm_checkpoint_sha256": self._dm_sha,
                    "anchors": anchors,
                }
            )
        return entries

    def pairs(self, cohort, real_of, generated_root):
        """The flat quantitative-side records: zero-training baseline + real reference per ordered pair."""
        records = []
        for item in cohort:
            challenge, case = item["sub"], item["case"]
            for src in MODALITIES:
                for tgt in MODALITIES:
                    if src == tgt:
                        continue
                    records.append(
                        {
                            "challenge": challenge,
                            "case": case,
                            "src_modality": src,
                            "target_modality": tgt,
                            "seed": seed_of(case, src, tgt),
                            "baseline": str(self.generated_path(generated_root, challenge, case, src, tgt)),
                            "reference": str(real_of(challenge, case, tgt)),
                        }
                    )
        return {
            "schema": PAIRS_SCHEMA,
            "run_id": self._run_id,
            "variant": BASELINE_VARIANT,
            "side": self._side,
            "dm_checkpoint_sha256": self._dm_sha,
            "records": records,
        }

    def ordered_pairs(self, case):
        """The 12 ordered src!=tgt modality pairs this plan must cover per case."""
        return [(src, tgt) for src in MODALITIES for tgt in MODALITIES if src != tgt]


class SideCohortBuilder:
    """The dev or holdout cohort entries (sub, case) from the pinned manifest, sharded deterministically."""

    def __init__(self, manifest, side, shard=0, num_shards=1, limit=None, only_challenge=None, only_cases=None):
        self._manifest = manifest
        self._side = side
        self._shard = shard
        self._num_shards = num_shards
        self._limit = limit
        self._only_challenge = only_challenge
        self._only_cases = set(only_cases or [])

    def build(self):
        if self._side not in ("dev", "holdout"):
            raise BaselineGenerateError(f"side must be dev or holdout: {self._side!r}")
        cohort = []
        for challenge, info in self._manifest["challenges"].items():
            if self._only_challenge is not None and challenge != self._only_challenge:
                continue
            cases = [case for case in info["cases"][self._side] if not self._only_cases or case in self._only_cases]
            for case in cases[: self._limit] if self._limit is not None else cases:
                cohort.append({"sub": challenge, "case": case})
        if self._num_shards > 1:
            cohort = cohort[self._shard :: self._num_shards]
        return cohort


class RawCaseLayout:
    """Real volume paths and per-case post-resize spacing from the raw t1n header (#52 formula)."""

    def __init__(self, raw_root, manifest):
        self._raw_root = Path(raw_root)
        self._dirs = {}
        for challenge, info in manifest["challenges"].items():
            source_dir = info.get("source_dir", "")
            rel = Path(source_dir).relative_to(self._raw_root) if source_dir.startswith(str(self._raw_root)) else Path(challenge)
            for side in ("dev", "holdout"):
                for case in info["cases"][side]:
                    self._dirs[case] = rel / case

    def directory_of(self, case):
        return self._dirs[case]

    def real_of(self, challenge, case, modality):
        path = self._raw_root / self._dirs[case] / f"{case}-{modality}.nii.gz"
        if not path.is_file():
            raise BaselineGenerateError(f"real {modality} volume missing for {challenge}/{case}: {path}")
        return path

    def spacing_of(self, case):
        path = self._raw_root / self._dirs[case] / f"{case}-t1n.nii.gz"
        if not path.is_file():
            raise BaselineGenerateError(f"raw t1n missing for spacing derivation: {path}")
        image = nib.load(path)
        zoom = [float(z) for z in image.header.get_zooms()[:3]]
        shape = [float(s) for s in image.shape[:3]]
        return [zoom[i] * shape[i] / GRID[i] for i in range(3)]


class ReferenceGridWriter:
    """Resamples real target volumes onto the generation grid so quantitative pair triplets share geometry."""

    TRANSFORMS = monai_t.Compose(
        [
            monai_t.LoadImage(image_only=True),
            monai_t.EnsureChannelFirst(),
            monai_t.Orientation(axcodes="RAS"),
            monai_t.EnsureType(dtype=torch.float32),
            monai_t.Resize(spatial_size=GRID, mode="trilinear"),
        ]
    )

    def __init__(self, out_root):
        self._out_root = Path(out_root)

    def path_of(self, challenge, case, modality):
        return self._out_root / "reference_grid" / challenge / case / f"{modality}.nii.gz"

    def write(self, challenge, case, modality, real_path, spacing):
        out = self.path_of(challenge, case, modality)
        if out.is_file():
            return out
        data = self.TRANSFORMS(str(real_path)).squeeze().cpu().numpy()
        image = nib.Nifti1Image(np.clip(np.rint(data), -32768, 32767).astype(np.int16), affine=np.diag([*spacing, 1.0]))
        out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, out)
        return out


class BaselineSampleWriter:
    """Generates the 12 ordered pairs per case and writes the distribution/quantitative manifests."""

    def __init__(self, merged, run_record, side, config, device, out_root, logger):
        self._merged = merged
        self._run_record = run_record
        self._side = side
        self._config = config
        self._device = device
        self._out_root = Path(out_root)
        self._logger = logger

    @torch.inference_mode()
    def write(self, cohort, layout):
        # the migrated chain's fast-fail: baseline img2img is RF-only (the interpolation start is the RFlow path)
        if not str(self._merged.noise_scheduler.get("_target_", "")).endswith("RFlowScheduler"):
            raise BaselinePlanError("baseline img2img only runs on the RFlow scheduler (rectified flow interpolation start)")
        autoencoder, unet, scale_factor = load_models(self._merged, self._device, self._logger)
        # The domain entity carries the img2img rules (strength truncation, RF
        # interpolation start, CFG denoising, ADR-0016); the VAE decode stays
        # the writer's render adapter below the latent the entity produces.
        autoencoder.eval()
        unet.eval()
        model = DiffusionModel(
            unet=unet,
            scale_factor=torch.tensor(float(scale_factor), device=self._device),
            noise_scheduler=RFlowScheduler(**{k: v for k, v in self._merged.noise_scheduler.items() if k != "_target_"}),
        )
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(self._device)
        anchor_encoder = AnchorLatentEncoder(autoencoder, self._device, GRID, self._logger)
        builder = BaselineSamplePlanBuilder(
            self._run_record["run_id"],
            self._run_record["upstream"]["checkpoint"]["sha256"],
            self._side,
            self._config,
        )
        generated_root = self._out_root / "generated"
        references = ReferenceGridWriter(self._out_root)
        failures = []
        try:
            for item in cohort:
                challenge, case = item["sub"], item["case"]
                spacing = layout.spacing_of(case)
                spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device).half()
                for anchor in MODALITIES:
                    anchor_path = layout.real_of(challenge, case, anchor)
                    # encode once per anchor; the three targets reuse the latent (#38 convention)
                    latent = anchor_encoder.encode(str(anchor_path))
                    for tgt in MODALITIES:
                        if tgt == anchor:
                            continue
                        out = builder.generated_path(generated_root, challenge, case, anchor, tgt)
                        if out.is_file():
                            continue
                        seed = seed_of(case, anchor, tgt)
                        try:
                            set_determinism(seed)
                            token = self._config.modality_tokens[tgt]
                            modality_tensor = (token * torch.ones((1,), dtype=torch.long)).to(self._device)
                            with torch.amp.autocast("cuda", enabled=True):
                                scheduler, image = model.begin_img2img(latent, self._config.strength, self._config.num_inference_steps)
                                while not scheduler.complete:
                                    image = model.denoise(scheduler, image, spacing_tensor, modality_tensor, self._config.cfg_guidance_scale)
                                data = self.render(recon_model, image, token)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            nib.save(nib.Nifti1Image(data, affine=np.diag([*spacing, 1.0])), out)
                            self._logger.info(f"[gen] {challenge}/{case}/{anchor}->{tgt} seed={seed}")
                        except Exception as error:  # one failed job must not kill the shard
                            failures.append(f"{challenge}/{case}/{anchor}->{tgt}: {error}")
                            self._logger.info(f"[fail] {challenge}/{case}/{anchor}->{tgt}: {error}")
                # quantitative-side reference: the real target on the generation grid (shared geometry)
                for tgt in MODALITIES:
                    references.write(challenge, case, tgt, layout.real_of(challenge, case, tgt), spacing)
        finally:
            del autoencoder, unet, model, recon_model
            torch.cuda.empty_cache()
        if failures:
            raise BaselineGenerateError(f"{len(failures)} baseline jobs failed; manifests not written (first: {failures[0]})")

        entries = builder.entries(cohort, layout.real_of, generated_root)
        pairs = builder.pairs(cohort, references.path_of, generated_root)
        return entries, pairs

    def render(self, recon_model, latent, modality_token) -> np.ndarray:
        """The denoised latent → int16 volume: production sliding-window decode + MR/CT intensity rescale.

        Matches the migrated run_img2img tail verbatim (sliding-window decode
        on the writer's device with the aggregation on CPU, MR → [0,1000],
        CT → [-1000,1000], direct int16 cast); the autocast context flows in
        from the caller.
        """
        inferer = SlidingWindowInferer(
            roi_size=[96, 96, 96],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.25,
            sw_device=self._device,
            device=torch.device("cpu"),
        )
        synthetic = dynamic_infer(inferer, recon_model, latent).squeeze().cpu().detach().numpy()
        token = int(modality_token)
        if token >= 8:  # MR: model output [0,1] -> [0,1000]
            synthetic = synthetic * 1000.0
            synthetic = np.clip(synthetic, 0, None)
        else:  # CT
            synthetic = synthetic * 2000.0 - 1000.0
            synthetic = np.clip(synthetic, -1000, 1000)
        return np.int16(synthetic)


def parse_args(argv=None):
    """The baseline generation entry argparse surface (verbatim from the retired stage0-baseline generate entry).

    Exposed for the argv↔namespace equivalence gate (ADR-0015 Testing).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="frozen cross-modal stage0-baseline run record (run.json)")
    parser.add_argument("--manifest", required=True, help="pinned phase phase_manifest.json")
    parser.add_argument("--out-root", required=True, help="controlled output root")
    parser.add_argument("--raw-root", required=True, help="phase raw root (real BraTS volumes)")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--infer-config", required=True, help="the official baseline inference config pinned by the run")
    parser.add_argument(
        "--side", default="holdout", choices=("dev", "holdout"), help="dev smoke validates the pipeline only; the contract freeze uses holdout"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="max cases per challenge (dev smoke)")
    parser.add_argument("--challenge", default=None)
    parser.add_argument("--only-cases", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    run_record = json.loads(Path(args.run).read_text())
    try:
        checkpoint = BaselineRunGuard(run_record, args.infer_config).check()
        config = BaselineInferenceConfig.from_path(args.infer_config)
        manifest = json.loads(Path(args.manifest).read_text())
    except (BaselineGenerateError, BaselinePlanError) as error:
        print(error, file=sys.stderr)
        return 1

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    # Point the #38 img2img chain at the frozen P1-DM checkpoint (same unet_state_dict + scale_factor layout).
    merged.model_dir = str(checkpoint.parent)
    merged.model_filename = checkpoint.name
    if tuple(config.grid) != GRID:
        print(f"baseline infer config grid {list(config.grid)} != fixed SRI24 grid {GRID}", file=sys.stderr)
        return 1
    if tuple(merged.diffusion_unet_inference["dim"]) != config.grid:
        print(f"model config dim {merged.diffusion_unet_inference['dim']} != baseline grid {list(config.grid)}", file=sys.stderr)
        return 1

    cohort = SideCohortBuilder(manifest, args.side, args.shard, args.num_shards, args.limit, args.challenge, args.only_cases).build()
    if not cohort:
        print("empty cohort after sharding/filters; nothing to generate", file=sys.stderr)
        return 1
    layout = RawCaseLayout(args.raw_root, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("stage0")
    writer = BaselineSampleWriter(merged, run_record, args.side, config, device, args.out_root, logger)
    entries, pairs = writer.write(cohort, layout)

    suffix = f"_shard_{args.shard}" if args.num_shards > 1 else ""
    samples_path = Path(args.out_root) / f"samples{suffix}.json"
    pairs_path = Path(args.out_root) / f"pairs{suffix}.json"
    samples_path.write_text(json.dumps(entries, indent=1) + "\n")
    pairs_path.write_text(json.dumps(pairs, indent=1) + "\n")
    print(f"wrote {len(entries)} case entries ({len(pairs['records'])} ordered pairs) -> {samples_path} / {pairs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
