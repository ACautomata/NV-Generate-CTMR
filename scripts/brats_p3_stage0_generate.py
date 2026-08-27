# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P3 stage-0 zero-training img2img baseline generation (issue #60).

Runs the four-anchor-round protocol over a manifest side with the frozen
P1-DM (the run record's upstream selection — nothing is trained): each real
modality anchors one round, the other three modalities are generated from
``x_t = (1-t)*src_latent + t*noise`` with the target modality label driving
denoising (spec #51 decision 8). 12 ordered src->tgt pairs per case, exactly
the #38 img2img chain (``img2img_infer.load_anchor_latent`` + ``run_img2img``)
re-pointed at the P1-DM checkpoint.

Outputs (controlled storage, shard-suffixed when ``--num-shards > 1``):

- ``generated/<CH>/<case>/a<src>/<tgt>_seed<seed>.nii.gz`` — baseline volumes;
- ``reference_grid/<CH>/<case>/<tgt>.nii.gz`` — real target volumes resampled
  onto the generation grid (RAS + trilinear 256x256x128, raw intensity domain),
  so L1 ``brats-l1-pairs/1`` triplets share shape and affine by construction;
- ``samples<...>.json`` — L2 ``P3FourAnchorPlan``-compatible entries (raw real
  paths; the L2 assembly resamples to the instrument grid itself);
- ``pairs<...>.json`` — the L1-side flat baseline/reference records.

Run order: ``--side dev`` first as a small smoke that only validates the
inference pipeline (its samples are a dev cot; they are not bound to the
contract). The run's freeze, and the L1/L2 deliverables it must be auditable
against, use the ``--side holdout`` samples manifest — the final-acceptance
holdout side the ``ctmr.application.acceptance.distribution.final_acceptance assemble --phase P3`` plan and the
``brats-l1-pairs/1`` stage-0 records are built from.

Usage::

    python -m scripts.brats_p3_stage0_generate \
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

import argparse
import json
import sys
from pathlib import Path

import monai.transforms as monai_t
import nibabel as nib
import numpy as np
import torch
from monai.utils import set_determinism

from .brats_p3_stage0_manifest import (
    MODALITIES,
    Stage0InferenceConfig,
    Stage0PlanError,
    Stage0RunGuard,
    Stage0SamplePlanBuilder,
)
from .diff_model_infer import load_models, prepare_tensors
from .diff_model_setting import load_config, setup_logging
from .img2img_infer import load_anchor_latent, run_img2img

GRID = (256, 256, 128)


class Stage0GenerateError(Exception):
    """Raised when the generation run breaks the stage-0 contract (cohort, layout, failed jobs)."""


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
            raise Stage0GenerateError(f"side must be dev or holdout: {self._side!r}")
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
            raise Stage0GenerateError(f"real {modality} volume missing for {challenge}/{case}: {path}")
        return path

    def spacing_of(self, case):
        path = self._raw_root / self._dirs[case] / f"{case}-t1n.nii.gz"
        if not path.is_file():
            raise Stage0GenerateError(f"raw t1n missing for spacing derivation: {path}")
        image = nib.load(path)
        zoom = [float(z) for z in image.header.get_zooms()[:3]]
        shape = [float(s) for s in image.shape[:3]]
        return [zoom[i] * shape[i] / GRID[i] for i in range(3)]


class ReferenceGridWriter:
    """Resamples real target volumes onto the generation grid so L1 pair triplets share geometry."""

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


class Stage0SampleWriter:
    """Generates the 12 ordered pairs per case and writes the L2/L1 manifests."""

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
        autoencoder, unet, scale_factor = load_models(self._merged, self._device, self._logger)
        top_ri, bottom_ri, _spacing, _modality = prepare_tensors(self._merged, self._device)
        self._merged.cfg_guidance_scale = self._config.cfg_guidance_scale
        builder = Stage0SamplePlanBuilder(
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
                    latent = load_anchor_latent(str(anchor_path), autoencoder, self._device, GRID, self._logger)
                    for tgt in MODALITIES:
                        if tgt == anchor:
                            continue
                        out = builder.generated_path(generated_root, challenge, case, anchor, tgt)
                        if out.is_file():
                            continue
                        seed = self._config.seed_of(case, anchor, tgt)
                        try:
                            set_determinism(seed)
                            token = self._config.modality_tokens[tgt]
                            modality_tensor = (token * torch.ones((1,), dtype=torch.long)).to(self._device)
                            data = run_img2img(
                                self._merged,
                                self._device,
                                autoencoder,
                                unet,
                                scale_factor,
                                latent,
                                top_ri,
                                bottom_ri,
                                spacing_tensor,
                                modality_tensor,
                                self._config.strength,
                                self._logger,
                            )
                            out.parent.mkdir(parents=True, exist_ok=True)
                            nib.save(nib.Nifti1Image(data, affine=np.diag([*spacing, 1.0])), out)
                            self._logger.info(f"[gen] {challenge}/{case}/{anchor}->{tgt} seed={seed}")
                        except Exception as error:  # one failed job must not kill the shard
                            failures.append(f"{challenge}/{case}/{anchor}->{tgt}: {error}")
                            self._logger.info(f"[fail] {challenge}/{case}/{anchor}->{tgt}: {error}")
                # L1-side reference: the real target on the generation grid (shared geometry)
                for tgt in MODALITIES:
                    references.write(challenge, case, tgt, layout.real_of(challenge, case, tgt), spacing)
        finally:
            del autoencoder, unet
            torch.cuda.empty_cache()
        if failures:
            raise Stage0GenerateError(f"{len(failures)} stage-0 jobs failed; manifests not written (first: {failures[0]})")

        entries = builder.entries(cohort, layout.real_of, generated_root)
        pairs = builder.pairs(cohort, references.path_of, generated_root)
        return entries, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="frozen P3 stage0-baseline run record (run.json)")
    parser.add_argument("--manifest", required=True, help="pinned phase phase_manifest.json")
    parser.add_argument("--out-root", required=True, help="controlled output root")
    parser.add_argument("--raw-root", required=True, help="phase raw root (real BraTS volumes)")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--infer-config", required=True, help="the official stage-0 inference config pinned by the run")
    parser.add_argument(
        "--side", default="holdout", choices=("dev", "holdout"), help="dev smoke validates the pipeline only; the contract freeze uses holdout"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="max cases per challenge (dev smoke)")
    parser.add_argument("--challenge", default=None)
    parser.add_argument("--only-cases", nargs="*", default=None)
    args = parser.parse_args(argv)

    run_record = json.loads(Path(args.run).read_text())
    try:
        checkpoint = Stage0RunGuard(run_record, args.infer_config).check()
        config = Stage0InferenceConfig.from_path(args.infer_config)
        manifest = json.loads(Path(args.manifest).read_text())
    except (Stage0GenerateError, Stage0PlanError) as error:
        print(error, file=sys.stderr)
        return 1

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    # Point the #38 img2img chain at the frozen P1-DM checkpoint (same unet_state_dict + scale_factor layout).
    merged.model_dir = str(checkpoint.parent)
    merged.model_filename = checkpoint.name
    merged.diffusion_unet_inference = dict(merged.diffusion_unet_inference)
    merged.diffusion_unet_inference["num_inference_steps"] = config.num_inference_steps
    if tuple(config.grid) != GRID:
        print(f"stage-0 infer config grid {list(config.grid)} != fixed SRI24 grid {GRID}", file=sys.stderr)
        return 1
    if tuple(merged.diffusion_unet_inference["dim"]) != config.grid:
        print(f"model config dim {merged.diffusion_unet_inference['dim']} != stage-0 grid {list(config.grid)}", file=sys.stderr)
        return 1

    cohort = SideCohortBuilder(manifest, args.side, args.shard, args.num_shards, args.limit, args.challenge, args.only_cases).build()
    if not cohort:
        print("empty cohort after sharding/filters; nothing to generate", file=sys.stderr)
        return 1
    layout = RawCaseLayout(args.raw_root, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("stage0")
    writer = Stage0SampleWriter(merged, run_record, args.side, config, device, args.out_root, logger)
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
