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

"""P3 image-conditioned ControlNet candidate generation (issue #61).

Runs the four-anchor-round protocol over a manifest side with the frozen P1-DM
+ the run's trained P3 ControlNet checkpoint (``controlnet-candidate``): each
real modality anchors one round, the other three target modalities are generated
by conditioning the frozen DM on the anchor's 4ch src latent (with the target
modality label riding the ``class_labels`` path into both DM and ControlNet) and
denoising from pure noise with CFG OFF (``cfg_guidance_scale == 0``, zero latent
unconditional branch — issue #61 acceptance criterion 1-2). This is the trained
counterpart of the stage-0 img2img baseline: it is *not* an interpolated image
start, it is a ControlNet condition.

Outputs (controlled storage, shard-suffixed when ``--num-shards > 1``):

- ``generated/<CH>/<case>/a<src>/<tgt>_seed<seed>.nii.gz`` — candidate volumes;
- ``reference_grid/<CH>/<case>/<tgt>.nii.gz`` — real target volumes resampled
  onto the generation grid (RAS + trilinear 256x256x128, raw intensity domain),
  shared with the stage-0 run so reference/baseline/candidate triplets align;
- ``samples<...>.json`` — L2 ``P3FourAnchorPlan``-compatible candidate entries
  (variant=controlnet-candidate); ``nnunet_l2_final_acceptance assemble --phase P3``
  consumes it directly;
- ``pairs<...>.json`` — L1-side flat candidate records. The full
  ``brats-l1-pairs/1`` triplets are merged by the manifest ``pairs`` builder from
  the stage-0 baseline/reference records plus these candidate volumes.

Run order: select/freeze the candidate first (the run must be frozen before
generation), then ``--side dev`` as a small smoke that only validates the inference
pipeline (its samples are a dev cot; they are not bound to the contract). The
L1/L2/L3 deliverables it must be auditable against use the ``--side holdout``
samples manifest.

Usage::

    python -m scripts.brats_p3_controlnet_generate \
        --run runs/p3-candidate-.../run.json \
        --manifest /ctrl/phase/phase_manifest.json \
        --out-root /ctrl/p3/candidate_holdout \
        --raw-root /ctrl/phase/raw \
        --stage0-pairs /ctrl/p3/stage0_holdout/pairs.json \
        -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
        -c configs/config_maisi_diff_model_rflow-mr-brain.json \
        -t configs/config_network_p3.json \
        --infer-config configs/config_p3_controlnet_infer.json \
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

from .brats_p3_controlnet_manifest import (
    MODALITIES,
    P3CandidateInferenceConfig,
    P3CandidatePlanError,
    P3CandidateRunGuard,
    P3CandidateSamplePlanBuilder,
)
from .brats_p3_stage0_generate import (
    GRID,
    RawCaseLayout,
    SideCohortBuilder,
    Stage0GenerateError,
)
from .diff_model_infer import prepare_tensors
from .diff_model_setting import load_config, setup_logging
from .img2img_infer import load_anchor_latent
from .utils_infer import load_image_models, run_controlnet_conditioned_image_dm

LATENT = (4, 64, 64, 32)


class P3CandidateGenerateError(Stage0GenerateError):
    """Raised when the candidate generation run breaks the P3 candidate contract."""


def load_candidate_models(merged, checkpoint, controlnet_ckpt, device, logger):
    """Load AE + frozen P1-DM + trained P3 ControlNet via the image-side loader.

    Sets ``trained_diffusion_path`` (the frozen P1-DM) and ``trained_controlnet_path``
    (the run's candidate checkpoint) on the merged config; ``load_image_models`` also
    re-initializes the ControlNet from the DM encoder/mid and then overlays the
    trained weights (``strict=False``) — the same "copy_model_state from dm" source
    the training run used, so the candidate inherits no P2 ControlNet.
    """
    merged.trained_diffusion_path = str(checkpoint)
    merged.trained_controlnet_path = str(controlnet_ckpt)
    autoencoder, unet, controlnet, scale_factor, noise_scheduler = load_image_models(merged, device)
    top_ri, bottom_ri, _spacing, _modality = prepare_tensors(merged, device)
    for model in (autoencoder, unet, controlnet):
        model.to(device).eval()
    logger.info(f"candidate models loaded: DM={merged.trained_diffusion_path} ControlNet={merged.trained_controlnet_path}")
    return autoencoder, unet, controlnet, scale_factor, noise_scheduler, top_ri, bottom_ri


class P3CandidateSampleWriter:
    """Generates the 12 ordered pairs per case and writes the L2/L1 candidate manifests."""

    def __init__(self, merged, run_record, side, config, device, out_root, logger):
        self._merged = merged
        self._run_record = run_record
        self._side = side
        self._config = config
        self._device = device
        self._out_root = Path(out_root)
        self._logger = logger

    @torch.inference_mode()
    def write(self, cohort, layout, stage0_records):
        autoencoder, unet, controlnet, scale_factor, noise_scheduler, top_ri, bottom_ri = load_candidate_models(
            self._merged, self._run_record["upstream"]["checkpoint"]["path"], self._run_record["selection"]["checkpoint"]["path"], self._device, self._logger
        )
        builder = P3CandidateSamplePlanBuilder(
            self._run_record["run_id"],
            self._run_record["upstream"]["checkpoint"]["sha256"],
            self._run_record["selection"]["checkpoint"]["sha256"],
            self._side,
            self._config,
        )
        generated_root = self._out_root / "generated"
        failures = []
        try:
            for item in cohort:
                challenge, case = item["sub"], item["case"]
                spacing = layout.spacing_of(case)
                spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device).half()
                for anchor in MODALITIES:
                    anchor_path = layout.real_of(challenge, case, anchor)
                    # encode once per anchor; its 4ch latent is the ControlNet condition
                    src_latent = load_anchor_latent(str(anchor_path), autoencoder, self._device, GRID, self._logger)
                    cond = (src_latent * scale_factor).half().to(self._device)
                    for tgt in MODALITIES:
                        if tgt == anchor:
                            continue
                        out = builder.generated_path(generated_root, challenge, case, anchor, tgt)
                        if out.is_file():
                            continue
                        seed = self._config.seed_of(case, anchor, tgt)
                        try:
                            set_determinism(seed)
                            modality_tensor = torch.tensor([self._config.modality_tokens[tgt]], device=self._device)
                            synthetic = run_controlnet_conditioned_image_dm(
                                autoencoder,
                                unet,
                                controlnet,
                                noise_scheduler,
                                scale_factor,
                                self._device,
                                controlnet_cond_tensor=cond,
                                spacing_tensor=spacing_tensor,
                                latent_shape=LATENT,
                                output_size=GRID,
                                noise_factor=1.0,
                                top_region_index_tensor=top_ri,
                                bottom_region_index_tensor=bottom_ri,
                                modality_tensor=modality_tensor,
                                num_inference_steps=self._config.num_inference_steps,
                                autoencoder_sliding_window_infer_size=(96, 96, 96),
                                autoencoder_sliding_window_infer_overlap=0.25,
                                cfg_guidance_scale=0.0,
                                controlnet_uncond_tensor=None,
                            )
                            data = synthetic.squeeze().cpu().numpy()
                            data = np.clip(data, 0, None).astype(np.int16)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            nib.save(nib.Nifti1Image(data, affine=np.diag([*spacing, 1.0])), out)
                            self._logger.info(f"[gen] {challenge}/{case}/{anchor}->{tgt} seed={seed} cfg=0")
                        except Exception as error:  # one failed job must not kill the shard
                            failures.append(f"{challenge}/{case}/{anchor}->{tgt}: {error}")
                            self._logger.info(f"[fail] {challenge}/{case}/{anchor}->{tgt}: {error}")
        finally:
            del autoencoder, unet, controlnet
            torch.cuda.empty_cache()
        if failures:
            raise P3CandidateGenerateError(f"{len(failures)} candidate jobs failed; manifests not written (first: {failures[0]})")

        # pairs triples reference volumes this shard actually generated: scope the
        # stage-0 records to the shard's cohort so a sharded/filtered run never
        # emits candidate paths it did not create (or 8x-duplicates every shard).
        cohort_keys = {(item["sub"], item["case"]) for item in cohort}
        shard_records = [r for r in stage0_records if (r["challenge"], r["case"]) in cohort_keys]
        entries = builder.entries(cohort, layout.real_of, generated_root)
        pairs = builder.pairs(shard_records, generated_root)
        return entries, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="frozen P3 controlnet-candidate run record (run.json)")
    parser.add_argument("--manifest", required=True, help="pinned phase phase_manifest.json")
    parser.add_argument("--out-root", required=True, help="controlled output root")
    parser.add_argument("--raw-root", required=True, help="phase raw root (real BraTS volumes)")
    parser.add_argument("--stage0-pairs", required=True, help="the stage-0 pairs.json (baseline + reference records) to merge")
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("--infer-config", required=True, help="the official P3 candidate inference config pinned by the run")
    parser.add_argument("--side", default="holdout", choices=("dev", "holdout"), help="dev smoke validates the pipeline only; the contract freeze uses holdout")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="max cases per challenge (dev smoke)")
    parser.add_argument("--challenge", default=None)
    parser.add_argument("--only-cases", nargs="*", default=None)
    args = parser.parse_args(argv)

    run_record = json.loads(Path(args.run).read_text())
    try:
        controlnet_ckpt = P3CandidateRunGuard(run_record, args.infer_config).check()
        config = P3CandidateInferenceConfig.from_path(args.infer_config)
        manifest = json.loads(Path(args.manifest).read_text())
        stage0_pairs = json.loads(Path(args.stage0_pairs).read_text())
    except (P3CandidateGenerateError, P3CandidatePlanError) as error:
        print(error, file=sys.stderr)
        return 1

    if tuple(config.grid) != GRID:
        print(f"P3 candidate infer config grid {list(config.grid)} != fixed SRI24 grid {GRID}", file=sys.stderr)
        return 1

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.model_dir = str(Path(controlnet_ckpt).parent)

    stage0_records = stage0_pairs.get("records")
    if not isinstance(stage0_records, list):
        print("stage-0 pairs.json missing records; cannot merge candidate triplets", file=sys.stderr)
        return 1

    cohort = SideCohortBuilder(manifest, args.side, args.shard, args.num_shards, args.limit, args.challenge, args.only_cases).build()
    if not cohort:
        print("empty cohort after sharding/filters; nothing to generate", file=sys.stderr)
        return 1
    layout = RawCaseLayout(args.raw_root, manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("p3-candidate")
    writer = P3CandidateSampleWriter(merged, run_record, args.side, config, device, args.out_root, logger)
    entries, pairs = writer.write(cohort, layout, stage0_records)

    suffix = f"_shard_{args.shard}" if args.num_shards > 1 else ""
    samples_path = Path(args.out_root) / f"samples{suffix}.json"
    pairs_path = Path(args.out_root) / f"pairs{suffix}.json"
    samples_path.write_text(json.dumps(entries, indent=1) + "\n")
    pairs_path.write_text(json.dumps(pairs, indent=1) + "\n")
    print(f"wrote {len(entries)} case entries -> {samples_path}")
    print(f"wrote {len(pairs['records'])} L1 triplets (reference+baseline+candidate) -> {pairs_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
