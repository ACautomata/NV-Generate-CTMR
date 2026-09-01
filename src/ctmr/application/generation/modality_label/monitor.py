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

"""Modality-label dev light-acceptance sidecar: fixed samples + FID trend + L2 trend (issue #57, spec #51 §6).

Runs beside the modality-label finetune on a reserved GPU. For every
``epoch_<N>.pt`` the trainer persists (N a multiple of ``--eval-every``), it:

1. generates the FIXED dev cohort — 16 dev cases x 4 target modalities
   (t1n/t1c/t2w/t2f), one sample per (case, modality) with a fixed
   per-(case, modality) seed, cfg=10, 30 steps, per-case spacing from the
   phase companions — the "fixed four-modality samples" the spec requires;
2. computes the 2.5D RadImageNet FID trend per target modality against the
   dev-side REAL volume bank (percentile 0-99.5 -> [0,1], RAS, 1 mm, zero pad
   240x240x160 — the pinned L1 MR preprocessing);
3. runs the frozen L2 instruments (nnUNetv2, ADR-0003 chain) on the generated
   pseudo-four-modality volumes and records WT/TC/ET volume medians plus
   input/run/hierarchy failure counts as the L2 trend;
4. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop`` for the trainer; ``select`` emits the final
   dev-side checkpoint selection (argmin mean FID) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts):
  metric m(N) = mean over the four target modalities of the plane-mean dev FID
  at epoch N; stop when N >= --min-epoch AND the last --patience consecutive
  evals produced no new best m; never past --max-epoch (= the trainer cap).

The shared trend machinery (cohort/FID bank/plane features/instrument runner)
lives in ``ctmr.application.generation.trend``; the watch/select polling
skeleton (``WatchEngine`` / ``SelectionEmitter``) in ``ctmr.application.shell``
-- this module only assembles the stage sampler/scorer/post-score collaborators
and dispatches the reference/watch/select verbs.

Migrated from the retired modality-label dev-eval script entry (ticket 10,
ADR-0015 §2); its ``selftest`` subcommand retired with it — its assertions
live as pytest functions. Per ADR-0016 the denoising loop runs on the domain
``DiffusionModel`` through a fresh ``DiffusionScheduler`` per sample call
(CFG / timestep / RF advance semantics unchanged); the trend machinery stays
application collaborators. Since #272 (ADR-0019 §2-§3) the model loading and
inference primitives ride the injected ``GenerationEngine`` port -- the
concrete adapter is assembled by the composition root (``ctmr.wiring.generate``),
which this entry reuses as its dispatch face.

Usage (sugon, one reserved GPU):
    ctmr generate modality-label dev-eval reference --dev-list ... --raw-root ... --eval-root DIR
    ctmr generate modality-label dev-eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --emb-root ... -e env.json -c config.json -t network.json
    ctmr generate modality-label dev-eval select --eval-root DIR --ckpt-dir DIR --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.networks.schedulers import RFlowScheduler

from ctmr.application.generation.devices import add_device_flag, resolve_device
from ctmr.application.generation.trend import DevCohortBuilder, L2TrendRunner, MrTrendFeatures, RealReferenceBank, TrendFid
from ctmr.application.shell import (
    MODALITY_TOKENS,
    TARGET_MODALITIES,
    EarlyStopRule,
    SelectionEmitter,
    WatchEngine,
)
from ctmr.domain.dm_output_grid import V1_DM_OUTPUT_GRID
from ctmr.domain.engine import GenerationEngine
from ctmr.domain.generation.model import DiffusionModel
from ctmr.wiring.generate import modality_label_engine


class CohortSpacingSource:
    """Per-case post-resize spacing from the phase embedding companions (t1n entry)."""

    def __init__(self, dev_list_path, emb_root):
        self._emb_root = Path(emb_root)
        self._entries = {}
        for entry in json.loads(Path(dev_list_path).read_text())["training"]:
            if entry["modality"] == "mri_t1_skull_stripped":
                self._entries[entry["case"]] = entry["image"]

    def spacing_of(self, case):
        rel = self._entries[case].replace(".nii.gz", "_emb.nii.gz") + ".json"
        return json.loads((self._emb_root / rel).read_text())["spacing"]


class FrozenAutoencoder:
    """Loads the frozen VAE at both samplers' fp16 conventions (shared payload allowlist).

    Upstream inference convention is fp16 on the DCU (float16 latents); a
    half-precision model keeps the conv input/weight/bias set consistent (the
    HIP bf16 SDPA flash path emits fp16 and breaks the mixed chain). The retired
    entry allowlisted numpy reconstruction at import time; the load keeps the
    same exposure at its load point instead (never an import-time global mutation).
    """

    def __init__(self, args, device, engine: GenerationEngine):
        self._args = args
        self._device = device
        self._engine = engine

    def load(self):
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType])
        autoencoder = self._engine.define_instance(self._args, "autoencoder_def").to(self._device)
        ae_ckpt = torch.load(self._args.trained_autoencoder_path, map_location=self._device, weights_only=True)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        autoencoder.eval()
        return autoencoder.half()


class CandidateSampler:
    """Generates the fixed dev cohort samples with a candidate checkpoint (cfg=10, 30 steps).

    The model-loading and inference faces ride the injected ``GenerationEngine``
    port (ADR-0019 §3, #272): define_instance for both networks, the frozen
    sliding-window primitive for the VAE decode, and the recon wrapper factory.
    """

    def __init__(self, args, device, logger, engine: GenerationEngine):
        self._args = args
        self._device = device
        self._logger = logger
        self._engine = engine

    @staticmethod
    def seed_of(case, modality):
        return int(hashlib.sha256(f"{case}|{modality}".encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def load_models(self, checkpoint_path):
        autoencoder = FrozenAutoencoder(self._args, self._device, self._engine).load()
        unet = self._engine.define_instance(self._args, "diffusion_unet_def").to(self._device)
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        unet.eval()
        unet = unet.half()
        scale = float(ckpt["scale_factor"])
        # The domain entity carries the sampling rules: the RF scheduler shape
        # and the denoising loop (CFG composition, fresh DiffusionScheduler per
        # sample call, ADR-0016). The VAE decode wrapper comes from the injected
        # engine port's recon primitive.
        model = DiffusionModel(
            unet=unet,
            scale_factor=torch.tensor(scale, device=self._device),
            noise_scheduler=RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"}),
        )
        return model, self._engine.recon_model(autoencoder, scale).to(self._device).half()

    @torch.inference_mode()
    def sample_one(self, model, recon_model, modality_token, spacing, seed, output_size=(256, 256, 128)):
        from monai.inferers import SlidingWindowInferer

        torch.manual_seed(seed)
        divisor = 4
        image = torch.randn((1, 4, output_size[0] // divisor, output_size[1] // divisor, output_size[2] // divisor), device=self._device)
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
        cfg = self._args.cfg_guidance_scale
        scheduler = model.begin_sampling(image.shape, self._args.diffusion_unet_inference["num_inference_steps"])
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            while not scheduler.complete:
                image = model.denoise(scheduler, image, spacing_tensor, modality_tensor, cfg)
        inferer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25, sw_device=self._device, device=torch.device("cpu"))
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            synthetic = self._engine.dynamic_infer(inferer, recon_model, image).squeeze().float().cpu().numpy()
        data = synthetic * 1000.0  # [0,1] -> MR 0..1000 scale, upstream int16 convention
        return np.clip(data, 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cohort, spacings, out_dir):
        model, recon = self.load_models(checkpoint_path)
        samples = []
        for item in cohort:
            for modality in TARGET_MODALITIES:
                seed = self.seed_of(item["case"], modality)
                out = Path(out_dir) / item["sub"] / f"{item['case']}_{modality}_seed{seed}.nii.gz"
                if not out.is_file():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data = self.sample_one(model, recon, MODALITY_TOKENS[modality], spacings.spacing_of(item["case"]), seed)
                    # Ruling #6: declare the v1 DM's real sampling spacing, not unit 1 mm,
                    # so the instrument chain's 1 mm resample is no longer a no-op.
                    image = nib.Nifti1Image(data, affine=V1_DM_OUTPUT_GRID.affine())
                    nib.save(image, out)
                samples.append({"sub": item["sub"], "case": item["case"], "modality": modality, "path": str(out)})
        del model, recon
        torch.cuda.empty_cache()
        return samples


class FidTrendScorer:
    """The watch scorer seam: plane-mean 2.5D RadImageNet FID per target modality."""

    def __init__(self, features, fid):
        self._features = features
        self._fid = fid

    def __call__(self, samples):
        plane_cache = {sample["path"]: self._features.volume_features(sample["path"]) for sample in samples}
        generated = {modality: {plane: [] for plane in ("xy", "yz", "zx")} for modality in TARGET_MODALITIES}
        for sample in samples:
            for plane in ("xy", "yz", "zx"):
                matrix = plane_cache[sample["path"]][plane]
                if matrix is not None:
                    generated[sample["modality"]][plane].append(matrix.mean(axis=0))
        return self._fid.trend_fields(generated)


class CohortFeatureScorer:
    """The embedded-validation scorer seam (ADR-0019 §5, #278): the all_gathered
    per-item plane-mean features fold into the per-modality FID trend fields.

    Same output contract as ``FidTrendScorer`` (``(fields, log_line)``), but the
    input is the merged shard entries -- the features were extracted on the
    sampling rank, never re-extracted on every rank.
    """

    def __init__(self, fid):
        self._fid = fid

    def __call__(self, entries):
        generated = {modality: {plane: [] for plane in ("xy", "yz", "zx")} for modality in TARGET_MODALITIES}
        for entry in entries:
            for plane, vector in (entry.get("features") or {}).items():
                if vector is not None:
                    generated[entry["modality"]][plane].append(vector)
        return self._fid.trend_fields(generated)


class LiveCohortSampler:
    """The embedded-validation sampler seam (ADR-0019 §5, #278): the live training
    weights render this rank's cohort shard and each entry carries its plane-mean
    features back for the all_gather.

    Composition, never inheritance: the single-sample render is the shared
    ``CandidateSampler.sample_one`` (the verbatim denoising loop), the plane
    features are ``MrTrendFeatures``; the training UNet arrives DDP-stripped
    through the kernel's ``sampling_unet`` face, so the training weights are
    sampled but never mutated (no half(), no state_dict copy -- zero
    training-math drift). The frozen VAE is loaded per validation call and
    released after, keeping the training residency unchanged between stages.
    """

    def __init__(self, args, device, engine: GenerationEngine, kernel, spacings, features):
        self._args = args
        self._device = device
        self._engine = engine
        self._kernel = kernel
        self._spacings = spacings
        self._features = features

    def __call__(self, ctx, shard_items, out_dir):
        model = DiffusionModel(
            unet=self._kernel.sampling_unet(),
            scale_factor=torch.tensor(float(ctx.scale), device=self._device),
            noise_scheduler=RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"}),
        )
        autoencoder = FrozenAutoencoder(self._args, self._device, self._engine).load()
        recon = self._engine.recon_model(autoencoder, float(ctx.scale)).to(self._device).half()
        renderer = CandidateSampler(self._args, self._device, None, self._engine)
        entries = []
        for item in shard_items:
            seed = CandidateSampler.seed_of(item["case"], item["modality"])
            out = Path(out_dir) / item["sub"] / f"{item['case']}_{item['modality']}_seed{seed}.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            data = renderer.sample_one(model, recon, MODALITY_TOKENS[item["modality"]], self._spacings.spacing_of(item["case"]), seed)
            # Ruling #6 (same as the sidecar): declare the v1 DM's real sampling spacing.
            nib.save(nib.Nifti1Image(data, affine=V1_DM_OUTPUT_GRID.affine()), out)
            planes = self._features.volume_features(out)
            entries.append(
                {
                    "sub": item["sub"],
                    "case": item["case"],
                    "modality": item["modality"],
                    "path": str(out),
                    "features": {plane: (None if matrix is None else matrix.mean(axis=0)) for plane, matrix in planes.items()},
                }
            )
        del model, autoencoder, recon
        torch.cuda.empty_cache()
        return entries


class L2PostScore:
    """The optional post-score extension: the frozen L2 instruments trend (``--skip-l2`` degrades to None).

    The extension owns its failure tolerance: a single-epoch instrument hiccup
    records the None field and must not kill the sidecar -- the engine's skip
    path is reserved for the score itself.
    """

    def __init__(self, l2, cohort, skip):
        self._l2 = l2
        self._cohort = cohort
        self._skip = skip

    def __call__(self, epoch, samples, epoch_dir):
        fields = {"l2_trend": None}
        if self._skip:
            return fields
        try:
            fields["l2_trend"] = self._l2.run(samples, self._cohort, epoch_dir)
        except Exception as error:
            print(f"[eval] epoch {epoch} l2 skipped: {error}", file=sys.stderr, flush=True)
        return fields


def parse_args(argv=None):
    """The sidecar entry argparse surface (verbatim from the retired dev-eval script entry).

    Exposed for the argv↔namespace equivalence gate (ADR-0015 Testing: the
    assertion lives in tests/application/generation/modality_label).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="build the dev real-feature bank once")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)
    add_device_flag(p)

    p = sub.add_parser("watch", help="sidecar loop: evaluate epoch checkpoints as they land")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--skip-l2", action="store_true", help="FID-only trend (instruments unavailable)")
    p.add_argument("--instrument-results", action="append", default=[], help="CHALLENGE=nnUNet_results path")
    p.add_argument("--nnunet-raw", default="/root/private_data/ctmr/data/nnunet_raw")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/ctmr/data/nnunet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")
    add_device_flag(p)

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    eval_root = Path(args.eval_root)

    if args.command == "reference":
        features = MrTrendFeatures(resolve_device(args.device))
        RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
        print(f"real reference bank -> {eval_root / 'reference' / 'real_reference_bank.pt'}")
        return 0

    if args.command == "select":
        return SelectionEmitter(eval_root).emit(args.out, rule_text="argmin mean dev FID over eval points (pre-recorded)")

    # watch mode: assemble the stage collaborators, the shell engine drives the loop
    device = resolve_device(args.device)
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(args.dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = CohortSpacingSource(args.dev_list, args.emb_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    engine = modality_label_engine()  # the composition root's engine assembly (ADR-0019 §2)
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
    sampler = CandidateSampler(merged, device, None, engine)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.nnunet_raw, args.nnunet_preprocessed)
    return WatchEngine(
        ckpt_dir=args.ckpt_dir,
        eval_root=eval_root,
        eval_every=args.eval_every,
        max_epoch=args.max_epoch,
        rule=rule,
        sampler_factory=partial(sampler.generate_cohort, cohort=cohort, spacings=spacings),
        scorer=FidTrendScorer(features, TrendFid(bank)),
        poll_seconds=args.poll_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        post_score=L2PostScore(l2, cohort, args.skip_l2),
    ).run(cohort_file=str(cohort_path))


if __name__ == "__main__":
    sys.exit(main())
