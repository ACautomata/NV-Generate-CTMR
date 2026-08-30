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

"""Diagnostic job C (issue #208, parent #205): t1c intensity-domain discrimination.

Local forensics (parent #205) proved the direct "preprocessing clipped the bright
core" story false -- ``clip=False`` extrapolates the top ~0.5% of training t1c
voxels above 1.0 -- and left the L1 encoding domain as the one unmeasured link:
whether the frozen autoencoder_v1 reconstructs that extrapolated band, and
whether the generated t1c ever emits a >1.0 bright core at all. Job C turns the
three verdicts into numbers, all read-only, no acceptance verdict:

1. Three-way tiered histograms over real / VAE reconstruction / generated t1c,
   intra-tumour (WT) vs whole brain, reading P99 / P99.9 / top-0.5% mean.
   Real volumes live in the training normalisation domain (per-volume 0-99.5
   percentile -> [0, 1], ``clip=False`` -- the training protocol arm); the VAE
   reconstruction is the existing fp32 training embedding decoded straight back
   (the embeddings are raw encoder outputs, fed to ``decode_stage_2_outputs``
   as-is -- the diffusion scaling ``raw * scale_factor`` never touches them);
   the generated volume is the retained int16 artifact divided by 1000 (the
   writer's own "[0,1] -> MR 0..1000" convention). Each side is measured on its
   own native grid -- top-tail statistics are domain scalars, so no resampling
   blurs them.
2. Conditioned MAE of the VAE reconstruction with voxels split by the training
   input domain: >1.0 (the extrapolated band) vs within [0, 1]. The clip=True
   normalisation arm (same frozen VAE, same resize, only the tail truncated)
   reuses the noclip arm's tier masks. Three readings: each arm against its own
   input (reconstruction fidelity per input domain) plus the clip-encoded
   reconstruction against the SHARED noclip target -- the clip policy's signal
   loss measured, not assumed away (#216 review).
3. The share of generated int16 voxels above 1000 -- i.e. above 1.0 in the
   generator's own output domain (the writer only down-clips) -- split by whole
   brain / WT / ET, the material-basis probe for the "bright core lives above
   1.0 but the evaluation clip hides it" hypothesis.

Geometry conventions: real volumes and their segmentation share the native BraTS
grid; the reconstruction sits on the training grid (the embedding's own shape);
the generated volume sits on the generated grid. The generated-side tumour mask
is the retained frozen-instrument prediction mapped back by the exact inverse of
the instrument's centred crop/pad (crop 8 on x/y, pad 13 in front on z) -- a pure
array index shift, no interpolation. Artifact fact recorded at execution time:
the holdout generated NIfTIs carry a unit 1 mm affine (the sidecar writer's
convention), so the instrument's 1 mm resample was a no-op and the z direction
was padded, not cropped -- this job reads the artifacts as they are and does not
re-litigate job A's registered geometry.

This module is ``variant=diagnostic``: no acceptance verdict, bootstrap seeds
kept far from the formal judge chain (diagnostic base shared with jobs A/B, job
C occupying slots 300..307). The sugon host recipe lives at
``deploy/jobs/run_intensity_domain_c.sh``; reports land in the sugon artifact
area (controlled storage), never in git.

Heavy dependencies (torch / monai / nibabel) are imported lazily inside the
adapters on purpose: unlike the sibling diagnostic jobs (which top-import
SimpleITK or stay stdlib), this module's statistics core must stay importable
without a deep-learning stack so the numpy-only unit tests carry no torch weight.

P3 candidate reuse (#205 series-③ merge point): the emb-pool surface is the
MONAI training list plus the shared fp32 embeddings -- phase-agnostic, so a P3
run re-points ``--train-list``/``--emb-root``; the gen-pool surface is the P1
holdout ``samples.json`` layout (case_id/challenge/samples) + instrument
predictions -- P3 carries a different manifest layout (anchors-keyed) and is
NOT supported by this reader. Whether P3 needs a clip-arm re-run is a series-③
decision, recorded here as the reuse hook, not pre-decided.

Usage:
    python -m ctmr.application.acceptance.distribution.intensity_domain \
        --train-list <p1_image_only.json> --data-root <raw> --emb-root <embeddings> \
        --samples <samples.json> --real-root <holdout raw> --pred-root <predictions> \
        -e <env json> -c <model config json> -t <network def json> \
        --output-dir <artifact area>/intensity_domain [--run-id <run>] [--limit N]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ctmr.application.acceptance.distribution.statistics import ClusterBootstrap
from ctmr.domain.measurement import WilsonUpper

# Diagnostic bootstrap seeds share jobs A/B's namespace (zcrop_compensation.py /
# et_discrimination.py DIAGNOSTIC_SEED_BASE): far away from the formal judge
# chain's GLOBAL_SEED (20260821). Job A occupies slots 0/1 and 100/101, job B
# slot 200 of each challenge band; job C's MAE blocks and input-domain scalars
# take 300..307 -- the emb pool is not challenge-stratified, so the slots ride
# the base directly.
DIAGNOSTIC_SEED_BASE = 900_000_000
JOB_C_SEED_SLOT = 300
MAE_SEED_SLOTS = {
    "noclip_over": 0,
    "noclip_within": 1,
    "clip_over": 2,
    "clip_within": 3,
    "clip_over_shared": 4,
    "clip_within_shared": 5,
    "extrapolation_max": 6,
    "raw_percentile_upper": 7,
}

# The official BraTS release tree names challenges by their release suffix,
# which diverges from the pinned challenge code for METS ("...-MET-...").
# Matching the literal code
# finds nothing and silently drops all 48 METS real companions (#216 review).
OFFICIAL_TREE_SUFFIX = {"METS": "MET"}

# The training resize target (create_training_data round_number: multiples of
# 128); the generated grid is the same voxel count. Array axes are zyx.
TRAINING_GRID = (256, 256, 128)  # xyz, MONAI spatial_size convention
GEN_SHAPE_ZYX = (128, 256, 256)
INSTRUMENT_SHAPE_ZYX = (155, 240, 240)
CROP_XY = 8  # (256 - 240) // 2, the instrument's centred x/y crop
PAD_FRONT_Z = 13  # (155 - 128) // 2, the instrument's centred front z pad

TUMOUR_LABELS = (1, 2, 3)  # whole tumour, BraTS/instrument label space
ET_LABEL = 3
EXTRAPOLATION_THRESHOLD = 1.0  # the clip=False affine maps the 99.5th percentile to exactly 1.0
GEN_INT16_SCALE = 1000.0  # the generation writer's "[0,1] -> MR 0..1000" int16 convention
OVER_THRESHOLD = 1000.0  # the >1.0 output-domain probe in int16 units

HIST_RANGE = (0.0, 2.0)
HIST_BINS = 50
HIST_EDGES = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
TIERS = ("brain", "tumour")
GEN_TIERS = ("brain", "tumour", "et")


class DiagnosticError(Exception):
    """Raised when the diagnostic inputs cannot support an intensity-domain run."""


class TrainingPreprocessing:
    """The training pipeline's own intensity/geometry preprocessing, both arms.

    The training arm is ``ScaleIntensityRangePercentilesd(0, 99.5 -> 0, 1,
    clip=False)``: the affine is kept and the tail extrapolates above 1.0 (the
    very band this job probes); the clip arm truncates the same affine at 1.
    The resize is the ``Resized(spatial_size, mode)`` transform of
    ``create_training_data.create_transforms`` (trilinear for images, nearest
    for labels), pinned here so the diagnostic arms reuse the training
    transforms verbatim instead of a reimplementation.
    """

    @staticmethod
    def normalize_percentile(volume, *, clip):
        lower, upper = np.percentile(np.asarray(volume, dtype=np.float64), (0.0, 99.5))
        if upper <= lower:
            raise ValueError("volume has no usable 0-99.5 percentile intensity range")
        norm = (np.asarray(volume, dtype=np.float64) - lower) / (upper - lower)
        if clip:
            norm = np.clip(norm, 0.0, 1.0)
        return norm, float(lower), float(upper)

    @staticmethod
    def resize_image(volume, spatial_size, mode):
        import torch
        from monai.transforms import Resize

        array = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None]  # channel-first
        return Resize(spatial_size=spatial_size, mode=mode)(array).numpy()[0]


class TieredIntensityStats:
    """The job's tiered intensity readings (pure statistics, numpy only).

    Tiers are voxel supports (boolean masks broadcastable to the volume):
    whole brain = intensity > 0, intra-tumour = the WT label mask, ET = label
    3. Empty tiers read as ``None`` (json has no NaN), never 0 -- a missing
    reading is a measurement result, not a zero.
    """

    @staticmethod
    def top_stats(values):
        """Top-tail read-out of one tier: P99, P99.9, and the mean of the brightest
        0.5% of voxels (at least one, so tiny tiers still carry a reading)."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return {"n": 0, "p99": None, "p99_9": None, "top05_mean": None}
        k = max(1, int(np.ceil(0.005 * values.size)))
        top = np.sort(values)[-k:]
        return {
            "n": int(values.size),
            "p99": float(np.quantile(values, 0.99)),
            "p99_9": float(np.quantile(values, 0.999)),
            "top05_mean": float(top.mean()),
        }

    @staticmethod
    def tier_stats(values, mask):
        """Top statistics of one tier, or None when the tier has no voxel support."""
        if mask is None:
            return None
        return TieredIntensityStats.top_stats(values[mask])

    @staticmethod
    def histograms(values, mask_by_tier):
        """Per-tier fixed-edge histograms (pooled across cases by the report); values
        above the top edge land in the overflow count, never silently dropped."""
        hist = {}
        for tier, mask in mask_by_tier.items():
            if mask is None:
                hist[tier] = None
                continue
            hist[tier] = TieredIntensityStats.histogram_counts(values[mask], edges=HIST_EDGES)
        return hist

    @staticmethod
    def histogram_counts(values, *, edges):
        values = np.asarray(values, dtype=np.float64).ravel()
        counts, _ = np.histogram(values, bins=edges)
        overflow = int((values > edges[-1]).sum())
        underflow = int((values < edges[0]).sum())  # decoder outputs can be negative inside a WT mask
        return counts.astype(int), overflow, underflow

    @staticmethod
    def conditioned_mae(input_vol, recon, *, hi_mask, lo_mask):
        """Reconstruction MAE split by the input-domain tiers.

        The masks are caller-supplied so the clip arm can borrow the noclip
        arm's tiers -- shared voxel support is what makes the two arms directly
        comparable. An empty tier reads as ``None``, never 0.
        """
        err = np.abs(np.asarray(recon, dtype=np.float64) - np.asarray(input_vol, dtype=np.float64)).ravel()
        hi_mask = np.asarray(hi_mask, dtype=bool).ravel()
        lo_mask = np.asarray(lo_mask, dtype=bool).ravel()
        n_over, n_within = int(hi_mask.sum()), int(lo_mask.sum())
        return {
            "mae_over": float(err[hi_mask].mean()) if n_over else None,
            "n_over": n_over,
            "mae_within": float(err[lo_mask].mean()) if n_within else None,
            "n_within": n_within,
        }

    @staticmethod
    def over_threshold_counts(volume, *, threshold, tier_masks):
        """Per-tier ``k``/``n`` of voxels above the threshold. An empty tier is
        {k: 0, n: 0} and a missing tier mask reads as ``None`` -- measurement
        results, never errors."""
        above = np.asarray(volume) > threshold
        return {name: ({"k": int((above & mask).sum()), "n": int(mask.sum())} if mask is not None else None) for name, mask in tier_masks.items()}

    @staticmethod
    def distribution_stats(values, bootstrap_b=None, seed=None):
        """Cross-case distribution of one per-case quantity (ClusterBootstrap linear
        quantile rule); with a seed, adds the diagnostic-seed cluster-bootstrap CI90."""
        if not values:
            stats = {"median": None, "mean": None, "q05": None, "q95": None, "n_cases": 0}
            if seed is not None:
                stats.update({"ci90_low": None, "ci90_high": None})
            return stats
        stats = {
            "median": ClusterBootstrap.quantile(values, 0.5),
            "mean": sum(values) / len(values),
            "q05": ClusterBootstrap.quantile(values, 0.05),
            "q95": ClusterBootstrap.quantile(values, 0.95),
            "n_cases": len(values),
        }
        if seed is not None:
            ci = ClusterBootstrap(bootstrap_b).ci90([[value] for value in values], seed)
            stats["ci90_low"], stats["ci90_high"] = ci["low"], ci["high"]
        return stats


# ── pools (IO / VAE / resize injected) ──────────────────────────────────


class EmbPool:
    """Per-case readings over the training-embedding pool (readings 1 and 2).

    Every case contributes: the real t1c in the training normalisation domain
    (native grid), its VAE reconstruction decoded straight from the retained
    fp32 embedding (training grid), and the two-arm conditioned MAE. The clip
    arm re-encodes the capped input with the same frozen VAE -- same resize,
    same decoder, only the normalisation tail differs.
    """

    def read_cases(self, entries, repo, recon, resize, grid=TRAINING_GRID):
        return [self._read_case(entry, repo, recon, resize, grid) for entry in entries]

    def _read_case(self, entry, repo, recon, resize, grid):
        case, sub = entry["case"], entry.get("sub")
        row = {
            "case": case,
            "sub": sub,
            "excluded": None,
            "real_native": {"brain": None, "tumour": None},
            "recon": {"brain": None, "tumour": None},
            "mae": None,
            "histograms": {"real_native": {"brain": None, "tumour": None}, "recon": {"brain": None, "tumour": None}},
        }

        def exclude(reason):
            row["excluded"] = reason
            return row

        data = repo.train_case(entry)
        if data is None:
            return exclude("missing_case")
        if data["t1c"] is None:
            return exclude("unreadable_t1c")
        if data["emb"] is None:
            return exclude("missing_embedding")
        if data["seg"] is None:
            return exclude("missing_segmentation")
        t1c, seg, emb = data["t1c"], data["seg"], data["emb"]

        norm, _lower, upper = TrainingPreprocessing.normalize_percentile(t1c, clip=False)
        norm_clip, _lower_c, _upper_c = TrainingPreprocessing.normalize_percentile(t1c, clip=True)
        input_noclip = resize(norm, grid, "trilinear")
        input_clip = resize(norm_clip, grid, "trilinear")
        wt_native = np.isin(seg, TUMOUR_LABELS)
        wt_grid = resize(wt_native.astype(np.float32), grid, "nearest") > 0.5

        recon_noclip = recon.decode(emb)
        recon_clip = recon.decode(recon.encode(input_clip))

        hi = input_noclip > EXTRAPOLATION_THRESHOLD
        lo = (input_noclip >= 0.0) & (input_noclip <= EXTRAPOLATION_THRESHOLD)
        arm_noclip = TieredIntensityStats.conditioned_mae(input_noclip, recon_noclip, hi_mask=hi, lo_mask=lo)
        arm_clip = TieredIntensityStats.conditioned_mae(input_clip, recon_clip, hi_mask=hi, lo_mask=lo)
        # the third reading: the clip-encoded reconstruction against the SHARED
        # noclip target, so the clip policy's signal loss is measured instead of
        # assumed away by each arm scoring its own input (#216 review)
        arm_clip_shared = TieredIntensityStats.conditioned_mae(input_noclip, recon_clip, hi_mask=hi, lo_mask=lo)
        row["mae"] = {
            "raw_percentile_upper": upper,
            "extrapolation_max": float(input_noclip.max()),
            "noclip_over": arm_noclip["mae_over"],
            "noclip_within": arm_noclip["mae_within"],
            "clip_over": arm_clip["mae_over"],
            "clip_within": arm_clip["mae_within"],
            "clip_over_shared": arm_clip_shared["mae_over"],
            "clip_within_shared": arm_clip_shared["mae_within"],
            "n_over": arm_noclip["n_over"],
            "n_within": arm_noclip["n_within"],
        }

        row["real_native"] = {
            "brain": TieredIntensityStats.tier_stats(norm, norm > 0),
            "tumour": TieredIntensityStats.tier_stats(norm, wt_native),
        }
        row["recon"] = {
            "brain": TieredIntensityStats.tier_stats(recon_noclip, recon_noclip > 0),
            "tumour": TieredIntensityStats.tier_stats(recon_noclip, wt_grid),
        }
        row["histograms"] = {
            "real_native": TieredIntensityStats.histograms(norm, {"brain": norm > 0, "tumour": wt_native}),
            "recon": TieredIntensityStats.histograms(recon_noclip, {"brain": recon_noclip > 0, "tumour": wt_grid}),
        }
        return row


class GenPool:
    """Per-case readings over the retained-generation pool (readings 1 and 3).

    The generated t1c is the retained int16 artifact divided by 1000, tiered by
    whole brain (>0) and by the frozen-instrument prediction mapped back onto
    the generated grid; the real t1c of the same case anchors the pool in the
    training normalisation domain. A missing prediction keeps the whole-brain
    reading but drops the tumour tiers -- measured, never imputed.
    """

    @staticmethod
    def align_pred_to_generated_grid(pred_mask):
        """The instrument prediction mapped back onto the generated grid.

        The instrument centred the 256x256x128 generated volume onto
        240x240x155 (crop 8 on x/y, pad 13 in front on z); this is the exact
        inverse array mapping ``gen[z, y, x] = pred[z + 13, y - 8, x - 8]`` --
        a pure index shift, no interpolation. Instrument voxels outside the
        generated domain (the padded z bands) read as False.
        """
        pred_mask = np.asarray(pred_mask, dtype=bool)
        if pred_mask.shape != INSTRUMENT_SHAPE_ZYX:
            raise ValueError(f"prediction must have shape {INSTRUMENT_SHAPE_ZYX} (zyx), got {pred_mask.shape}")
        mask = np.zeros(GEN_SHAPE_ZYX, dtype=bool)
        mask[0 : GEN_SHAPE_ZYX[0], CROP_XY : CROP_XY + INSTRUMENT_SHAPE_ZYX[1], CROP_XY : CROP_XY + INSTRUMENT_SHAPE_ZYX[2]] = pred_mask[
            PAD_FRONT_Z : PAD_FRONT_Z + GEN_SHAPE_ZYX[0], 0 : INSTRUMENT_SHAPE_ZYX[1], 0 : INSTRUMENT_SHAPE_ZYX[2]
        ]
        return mask

    def read_cases(self, entries, repo, align=None):
        align = align or self.align_pred_to_generated_grid
        return [self._read_case(entry, repo, align) for entry in entries]

    def _read_case(self, entry, repo, align):
        row = {
            "case": None,
            "challenge": None,
            "excluded": None,
            "pred_empty": None,
            "gen": {"brain": None, "tumour": None},
            "real_native": {"brain": None, "tumour": None},
            "over_1000": {"brain": None, "tumour": None, "et": None},
            "histograms": {"gen": {"brain": None, "tumour": None}, "real_native": {"brain": None, "tumour": None}},
        }

        def exclude(reason):
            row["excluded"] = reason
            return row

        data = repo.gen_case(entry)
        if data is None:
            return exclude("missing_case")
        row["case"], row["challenge"] = data["case"], data["challenge"]
        if data["gen"] is None:
            return exclude("unreadable_generated")

        gen = data["gen"].astype(np.float64) / GEN_INT16_SCALE
        gen_domain = {"brain": gen > 0, "tumour": None, "et": None}
        tumour_mask = et_mask = None
        row["pred_empty"] = bool((data["pred"] == 0).all()) if data["pred"] is not None else None
        if data["pred"] is not None:
            tumour_mask = align(np.isin(data["pred"], TUMOUR_LABELS))
            et_mask = align(data["pred"] == ET_LABEL)
            gen_domain["tumour"], gen_domain["et"] = tumour_mask, et_mask
        row["gen"] = {
            "brain": TieredIntensityStats.tier_stats(gen, gen_domain["brain"]),
            "tumour": TieredIntensityStats.tier_stats(gen, tumour_mask),
        }
        row["over_1000"] = TieredIntensityStats.over_threshold_counts(gen, threshold=OVER_THRESHOLD / GEN_INT16_SCALE, tier_masks=gen_domain)
        row["histograms"]["gen"] = TieredIntensityStats.histograms(gen, {"brain": gen_domain["brain"], "tumour": tumour_mask})

        if data["real"] is not None:
            norm, _lower, _upper = TrainingPreprocessing.normalize_percentile(data["real"], clip=False)
            wt_native = np.isin(data["seg"], TUMOUR_LABELS) if data["seg"] is not None else None
            row["real_native"] = {
                "brain": TieredIntensityStats.tier_stats(norm, norm > 0),
                "tumour": TieredIntensityStats.tier_stats(norm, wt_native),
            }
            row["histograms"]["real_native"] = TieredIntensityStats.histograms(norm, {"brain": norm > 0, "tumour": wt_native})
        return row


# ── adapters (lazy heavy deps) ──────────────────────────────────────────


class VaeReconstructor:
    """The frozen autoencoder_v1 encode/decode adapter (all arms share it).

    Encoding mirrors ``create_training_data`` (single sliding window -- the
    320x320x160 roi covers the whole training grid, so the windowed call
    degenerates to one forward -- under the chain's own autocast context, see
    ``_autocast``); decoding feeds the raw latent straight to
    ``decode_stage_2_outputs``: the stored embeddings are raw encoder outputs
    (create_training_data saves ``encode_stage_2_inputs`` unscaled) and the
    diffusion scaling (``raw * scale_factor``, domain/generation/model.py)
    never touches them -- ReconModel's ``z / scale_factor`` convention applies
    to diffusion-domain latents, not to this input (#216 review). Latents keep
    the training artifacts' channels-last fp32 layout.
    """

    def __init__(self, env_config, model_config, model_def, device="cpu"):
        self._paths = (env_config, model_config, model_def)
        self._device = device
        self._autoencoder = None
        self._scale = None

    def _ensure_loaded(self):
        if self._autoencoder is not None:
            return
        import torch

        from ctmr.infrastructure.maisi_engine.diff_model_setting import load_config
        from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

        env_config, model_config, model_def = self._paths
        args = load_config(env_config, model_config, model_def)
        autoencoder = define_instance(args, "autoencoder_def").to(self._device)
        checkpoint = torch.load(args.trained_autoencoder_path, map_location=self._device, weights_only=True)  # monitor.py precedent
        # The base DM checkpoint is the one file that needs weights_only=False:
        # the published release carries monai metadata (TraceKeys) that the
        # PyTorch 2.6+ allowlist rejects (observed on sugon, torch 2.9).
        if "unet_state_dict" in checkpoint:
            checkpoint = checkpoint["unet_state_dict"]
        autoencoder.load_state_dict(checkpoint)
        autoencoder.eval()
        self._autoencoder = autoencoder

    def _autocast(self):
        """The training encode chain's own mixed-precision execution context.

        ``create_training_data`` encodes under ``torch.amp.autocast("cuda")``:
        the pinned ``norm_float16`` config hands the group-norm output to the
        next conv as fp16, which only lands on a matching-precision conv inside
        autocast -- a bare fp32 forward crashes there, DCU and CPU alike. CPU
        fallbacks mirror the same regime with bf16 (the CPU autocast dtype).
        """
        import torch

        if self._device.startswith("cuda"):
            return torch.amp.autocast("cuda")
        return torch.amp.autocast("cpu", dtype=torch.bfloat16)

    def encode(self, image):
        import torch

        self._ensure_loaded()
        x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None].to(self._device)
        with torch.inference_mode(), self._autocast():
            z = self._autoencoder.encode_stage_2_inputs(x)
        # channels-last, like the stored embeddings, and fp32 like their dtype
        return z.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)

    def decode(self, latent):
        import torch

        self._ensure_loaded()
        z = torch.from_numpy(np.ascontiguousarray(latent, dtype=np.float32)).permute(3, 0, 1, 2)[None].to(self._device)
        with torch.inference_mode(), self._autocast():
            recon = self._autoencoder.decode_stage_2_outputs(z)
        return recon.squeeze().cpu().numpy().astype(np.float32)


class NiftiTrainCaseRepository:
    """Reads the emb-pool cases: raw t1c + segmentation from the BraTS tree and
    the retained fp32 training embedding. A missing artifact reads as ``None``
    -- the case is excluded with a stated reason, never silently imputed."""

    def __init__(self, data_root, emb_root):
        self._data_root = Path(data_root)
        self._emb_root = Path(emb_root)

    def train_case(self, entry):
        import nibabel as nib

        raw_path = self._data_root / entry["image"]
        emb_path = self._emb_root / entry["image"].replace(".nii.gz", "_emb.nii.gz")
        seg_path = Path(str(raw_path).replace("-t1c.nii.gz", "-seg.nii.gz"))

        def read(path):
            try:
                return nib.load(str(path)).get_fdata(dtype=np.float32)
            except (OSError, FileNotFoundError, ValueError):
                return None

        emb = read(emb_path)
        if emb is not None and emb.ndim != 4:
            emb = None  # not a channels-last latent; treat as missing
        return {"t1c": read(raw_path), "seg": read(seg_path), "emb": emb}


class NiftiGenCaseRepository:
    """Reads the gen-pool cases: the retained generated int16 artifact, the
    frozen-instrument prediction (``<pred_root>/<CH>/<case>__gen.nii.gz``) and
    the real companion in the L2 holdout layout
    (``<real_root>/<CH>/<case>/<case>-{t1c,seg}.nii.gz``). The prediction is
    delivered in the instrument's zyx array convention: nibabel exposes the
    written xyz layout, so a (240, 240, 155) array is transposed to
    (155, 240, 240) before the grid mapping."""

    def __init__(self, real_root, pred_root):
        self._real_root = Path(real_root)
        self._pred_root = Path(pred_root)
        self._challenge_dirs = {}

    def _challenge_dir(self, challenge):
        """The real-tree challenge directory: the L2 layout (<real-root>/<CH>)
        first, then the official BraTS tree naming via the canonical release
        suffix (OFFICIAL_TREE_SUFFIX: METS ships as "...-MET-..." -- the
        literal challenge code matches nothing there, #216 review)."""
        if challenge not in self._challenge_dirs:
            direct = self._real_root / challenge
            if direct.is_dir():
                self._challenge_dirs[challenge] = direct
            else:
                suffix = OFFICIAL_TREE_SUFFIX.get(challenge, challenge)
                match = None
                for child in sorted(self._real_root.iterdir()):
                    if child.is_dir() and f"-{suffix}-" in child.name:
                        match = child
                        break
                self._challenge_dirs[challenge] = match
        return self._challenge_dirs[challenge]

    def gen_case(self, entry):
        import nibabel as nib

        case, challenge = entry["case_id"], entry["challenge"]

        def read(path):
            try:
                return nib.load(str(path)).get_fdata(dtype=np.float32)
            except (OSError, FileNotFoundError, ValueError):
                return None

        pred = read(self._pred_root / challenge / f"{case}__gen.nii.gz")
        if pred is not None and pred.shape == (240, 240, 155):  # written xyz -> instrument zyx
            pred = pred.transpose(2, 1, 0)
        if pred is not None and pred.shape != INSTRUMENT_SHAPE_ZYX:
            pred = None
        gen = read(entry["samples"]["t1c"]["path"])
        if gen is not None and gen.shape == (256, 256, 128):  # written xyz -> zyx, same as the prediction
            gen = gen.transpose(2, 1, 0)
        challenge_dir = self._challenge_dir(challenge)
        real = seg = None
        if challenge_dir is not None:
            real_dir = challenge_dir / case
            real = read(real_dir / f"{case}-t1c.nii.gz")
            seg = read(real_dir / f"{case}-seg.nii.gz")
        return {
            "case": case,
            "challenge": challenge,
            "gen": gen,
            "pred": pred,
            "real": real,
            "seg": seg,
        }


# ── report ──────────────────────────────────────────────────────────────


class IntensityDomainReport:
    """Diagnostic json + markdown artifacts (sugon artifact area, never git)."""

    SCHEMA = "intensity-domain-diagnostic/1"
    TITLE = "诊断作业 C:t1c 强度域甄别(三方直方图+条件 MAE+>1000 输出域计数)"

    def __init__(self, train_list, samples, bootstrap_b=10000, run_id=None):
        self._train_list = train_list
        self._samples = samples
        self._bootstrap_b = bootstrap_b
        self._run_id = run_id

    @staticmethod
    def _reading_conventions():
        return {
            "domains": "real 侧为训练归一化域(逐例 0-99.5 百分位→[0,1],clip=False,顶部外推 >1.0);"
            "VAE 重建=既有 fp32 训练 embedding(raw encoder 输出)直接 decode,不做扩散域缩放,不再归一化;"
            "生成=保留 int16 工件 ÷1000(写出方「[0,1]→MR 0..1000」约定,只下截不上截),不再归一化",
            "tiers": "全脑=强度 >0 体素;瘤内=WT(标签 1/2/3)。real/重建用 BraTS seg(重建侧经 nearest resize 到训练栅格),"
            "生成用冻结仪器 pred 经仪器居中 crop/pad 的精确逆映射(gen[z,y,x]=pred[z+13,y-8,x-8],纯数组索引,零插值)",
            "top_stats": "P99/P99.9 为线性分位;top-0.5% 均值=最亮 ceil(0.5%·n) 个体素的均值(至少 1 个)",
            "conditioned_mae": "现网臂分层掩码=clip=False 归一化输入的 >1.0 与 [0,1] 体素;clip=True 对照臂(同 VAE/同 resize,"
            "仅归一化截尾)复用同掩码。三组读数:各臂对本臂输入的重建保真,外加 clip 编码对共同 noclip 目标的读数"
            "(clip 造成的信号损失被直接度量,不被各臂自评抵消);CI90=池分布 q05/q95 的 cluster-bootstrap 包络,非 median 的置信区间",
            "over_1000": "生成 int16 >1000 即输出域 >1.0;分母为该层体素数;Wilson 95% 上界为域唯一定义",
            "geometry": "工件事实:holdout 生成 NIfTI 携带单位 1mm affine(sidecar 写出约定),故仪器 1mm 重采样为 no-op、"
            "z 向 128→155 为 pad 而非 crop;本作业按工件现状读数,不回改作业 A 的注册几何",
            "p3_reuse_hook": "emb 池输入面=MONAI 训练 list+共享 fp32 embedding(phase 无关);gen 池=P1 holdout samples.json 布局"
            "(case_id/challenge/samples)+仪器 pred,P3 anchors 布局不支持;P3 是否重跑 clip 对照臂由 #205 序列③拍板",
        }

    def write(self, emb_rows, gen_rows, output_dir):
        def slim(row):
            return {key: value for key, value in row.items() if key != "histograms"}

        payload = {
            "schema": self.SCHEMA,
            "title": self.TITLE,
            "issue": 208,
            "variant": "diagnostic",
            "disclaimer": (
                f"诊断读数,不产生任何验收判定;与正式 L2 验收面严格分离(#205 作业 C)。bootstrap 种子独立于正式判定链"
                f"(诊断基 {DIAGNOSTIC_SEED_BASE},作业 C 占槽 {JOB_C_SEED_SLOT}..{JOB_C_SEED_SLOT + 7})。"
            ),
            "run_id": self._run_id,
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"train_list": str(self._train_list), "samples": str(self._samples)},
            "reading_conventions": self._reading_conventions(),
            "hypothesis_over_1000": None,
            "emb_pool": None,
            "gen_pool": None,
        }
        if emb_rows:
            payload["emb_pool"] = {
                "aggregate": self._aggregate_emb_pool(emb_rows),
                "per_case": [slim(row) for row in emb_rows],
            }
        if gen_rows:
            aggregate = self._aggregate_gen_pool(gen_rows)
            payload["gen_pool"] = {
                "aggregate": aggregate,
                "per_challenge": aggregate["per_challenge"],
                "over_1000_totals": aggregate["over_1000_totals"],
                "per_case": [slim(row) for row in gen_rows],
            }
            payload["hypothesis_over_1000"] = {
                "convention": "「亮核在 >1.0 输出域、被评估 clip 掉」假说的材料基础检验:生成写出只下截不上截,int16 >1000 即浮点输出域 >1.0 的体素",
                "totals": aggregate["over_1000_totals"],
                "reading": "若瘤内(尤其 ET)占比接近 0,则生成输出域内不存在亮核材料,「被评估 clip 掉」假说失去基础,"
                "亮核缺失应归因生成侧(编码域/条件语义)而非评估 clip",
            }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "intensity_domain_diagnostic.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
        md_path = output_dir / "intensity_domain_diagnostic.md"
        md_path.write_text(self._markdown(payload))
        return json_path, md_path

    @staticmethod
    def _fmt(value):
        return "n/a" if value is None else f"{value:.4f}"

    @staticmethod
    def _ci(stats):
        if stats["median"] is None:
            return "n/a"
        if stats.get("ci90_low") is None:
            return f"{stats['median']:.4f}"
        return f"{stats['median']:.4f} ({stats['ci90_low']:.4f}, {stats['ci90_high']:.4f})"

    # ── aggregation ──

    @staticmethod
    def _aggregate_tiers(rows, side):
        """Cross-case distribution of every tier metric of one side; also returns the
        pooled histogram (per-case counts summed over cases, shared edges). Cases
        whose tier came up empty (e.g. no WT voxels: top_stats returned the all-None
        n=0 block) contribute no per-metric value -- an absent reading, never a zero."""
        out, hist = {}, {}
        for tier in TIERS:
            per_metric = {"p99": [], "p99_9": [], "top05_mean": []}
            pooled = np.zeros(HIST_BINS, dtype=np.int64)
            overflow = 0
            underflow = 0
            n_cases = 0
            for row in rows:
                block = row[side][tier]
                if block is None or block["p99"] is None:
                    continue
                n_cases += 1
                for metric in per_metric:
                    per_metric[metric].append(block[metric])
                if row["histograms"][side][tier] is not None:
                    counts, over, under = row["histograms"][side][tier]
                    pooled += counts
                    overflow += over
                    underflow += under
            out[tier] = {metric: TieredIntensityStats.distribution_stats(values) for metric, values in per_metric.items()}
            out[tier]["n_cases"] = n_cases
            hist[tier] = {"counts": pooled.tolist(), "overflow": overflow, "underflow": underflow}
        return out, hist

    def _aggregate_emb_pool(self, rows):
        """Reading-2 aggregates (per-metric distributions + diagnostic-seed CI90) plus
        the reading-1 emb-pool aggregates and pooled histograms."""
        kept = [row for row in rows if row["excluded"] is None]
        out = {
            "n_cases": len(kept),
            "n_excluded": len(rows) - len(kept),
            "excluded_reasons": dict(Counter(row["excluded"] for row in rows if row["excluded"] is not None)),
        }
        for side in ("real_native", "recon"):
            out[side], hist = self._aggregate_tiers(kept, side)
            out[f"{side}_histograms"] = hist
        out["mae"] = {}
        for metric, slot in MAE_SEED_SLOTS.items():
            values = [row["mae"][metric] for row in kept if row["mae"] and row["mae"][metric] is not None]
            seed = DIAGNOSTIC_SEED_BASE + JOB_C_SEED_SLOT + slot
            out["mae"][metric] = TieredIntensityStats.distribution_stats(values, bootstrap_b=self._bootstrap_b, seed=seed)
        return out

    @staticmethod
    def _over_1000_block(ch_rows):
        layers = {}
        for tier in GEN_TIERS:
            k = sum(row["over_1000"][tier]["k"] for row in ch_rows if row["over_1000"][tier] is not None)
            n = sum(row["over_1000"][tier]["n"] for row in ch_rows if row["over_1000"][tier] is not None)
            layers[tier] = {
                "k": k,
                "n": n,
                "rate": k / n if n else None,
                "wilson_95_upper": WilsonUpper.of(k, n) if n else None,
            }
        return layers

    def _aggregate_gen_pool(self, rows):
        """Reading-3 aggregates (per-challenge and total >1000 counts with the domain's
        Wilson upper bounds) plus the reading-1 gen-pool aggregates and histograms."""
        kept = [row for row in rows if row["excluded"] is None]
        out = {
            "n_cases": len(kept),
            "n_excluded": len(rows) - len(kept),
            "excluded_reasons": dict(Counter(row["excluded"] for row in rows if row["excluded"] is not None)),
            "n_pred_empty": sum(1 for row in kept if row["pred_empty"]),
        }
        for side in ("gen", "real_native"):
            out[side], hist = self._aggregate_tiers(kept, side)
            out[f"{side}_histograms"] = hist
        out["per_challenge"] = {}
        for challenge in sorted({row["challenge"] for row in kept}):
            ch_rows = [row for row in kept if row["challenge"] == challenge]
            out["per_challenge"][challenge] = {"n_cases": len(ch_rows), "over_1000": self._over_1000_block(ch_rows)}
        out["over_1000_totals"] = self._over_1000_block(kept)
        return out

    # ── markdown ──

    def _markdown(self, payload):
        lines = [
            f"# {payload['title']}",
            "",
            f"**Issue**: [#208](https://github.com/ACautomata/NV-Generate-CTMR/issues/208)(父 #205 作业 C)"
            f" · **run**: `{payload['run_id'] or '未绑定'}`",
            f"**variant: diagnostic —— {payload['disclaimer']}**",
            "",
            "## 读数口径",
            "",
        ]
        lines += [f"- {value}" for value in payload["reading_conventions"].values()]
        if payload["emb_pool"]:
            lines += [
                "",
                "## ① 三方顶部统计 —— emb 池(real / VAE 重建)",
                "",
                "| 侧 | 层 | n_cases | P99 median (q05, q95) | P99.9 median (q05, q95) | top-0.5% 均值 median (q05, q95) |",
                "|---|---|---:|---|---|---|",
            ]
            emb = payload["emb_pool"]["aggregate"]
            for side, label in (("real_native", "real(归一化域)"), ("recon", "VAE 重建(decode)")):
                for tier in TIERS:
                    block = emb[side][tier]
                    lines.append(
                        f"| {label} | {tier} | {block['n_cases']} "
                        f"| {self._ci(block['p99'])} | {self._ci(block['p99_9'])} | {self._ci(block['top05_mean'])} |"
                    )
            lines += [
                "",
                "## ② VAE 条件 MAE(输入域分层;各臂目标见「层(目标)」列)",
                "",
                "| 臂 | 层(目标) | n_cases | MAE median (q05, q95) | 分布包络 CI90 [low, high] | MAE mean |",
                "|---|---|---:|---|---|---:|",
            ]
            mae = emb["mae"]
            for metric, label, tier in (
                ("noclip_over", "clip=False 现网臂", ">1.0 外推 → 本臂输入"),
                ("noclip_within", "clip=False 现网臂", "[0,1] 域内 → 本臂输入"),
                ("clip_over", "clip=True 对照臂", ">1.0 外推 → clip 后输入"),
                ("clip_within", "clip=True 对照臂", "[0,1] 域内 → clip 后输入"),
                ("clip_over_shared", "clip=True 对照臂", ">1.0 外推 → 共同 noclip 目标"),
                ("clip_within_shared", "clip=True 对照臂", "[0,1] 域内 → 共同 noclip 目标"),
            ):
                block = mae[metric]
                lines.append(
                    f"| {label} | {tier} | {block['n_cases']} | {self._ci(block)} "
                    f"| [{self._fmt(block['ci90_low'])}, {self._fmt(block['ci90_high'])}] | {self._fmt(block['mean'])} |"
                )
            ext = mae["extrapolation_max"]
            raw = mae["raw_percentile_upper"]
            lines += [
                "",
                f"现网臂 resize 后输入在 1.0 之上的最大外推高度跨例 median **{self._fmt(ext['median'])}**"
                f"(q05 {self._fmt(ext['q05'])}, q95 {self._fmt(ext['q95'])});"
                f"raw 99.5 百分位锚点(归一化分母,raw 值域)median {self._fmt(raw['median'])}"
                f"(q05 {self._fmt(raw['q05'])}, q95 {self._fmt(raw['q95'])})——1.0 之上的外推应力每例可见。",
                "",
                "注:各臂对本臂输入的读数衡量「该输入域上的重建保真」;clip 后输入可被高保真重建说明截尾输入落在 VAE 重建域内,"
                "但「对照臂更低」≠「clip 保留信号」——信号损失由「共同 noclip 目标」两行直接度量(CI90 为池分布 q05/q95 的"
                "cluster-bootstrap 包络,不是 median 的置信区间)。",
            ]
        if payload["gen_pool"]:
            gen = payload["gen_pool"]["aggregate"]
            lines += [
                "",
                "## ① 三方顶部统计 —— gen 池(生成 / real 同池锚)",
                "",
                "| 侧 | 层 | n_cases | P99 median (q05, q95) | P99.9 median (q05, q95) | top-0.5% 均值 median (q05, q95) |",
                "|---|---|---:|---|---|---|",
            ]
            for side, label in (("gen", "生成(输出域/1000)"), ("real_native", "real(归一化域)")):
                for tier in TIERS:
                    block = gen[side][tier]
                    lines.append(
                        f"| {label} | {tier} | {block['n_cases']} "
                        f"| {self._ci(block['p99'])} | {self._ci(block['p99_9'])} | {self._ci(block['top05_mean'])} |"
                    )
            lines += [
                "",
                "## ③ 生成输出 >1000 计数(>1.0 输出域假说)",
                "",
                "| 挑战 | n_cases | 层 | k/n | 占比 | Wilson 95% 上界 |",
                "|---|---:|---|---|---:|---:|",
            ]
            for challenge, block in payload["gen_pool"]["per_challenge"].items():
                for tier in GEN_TIERS:
                    over = block["over_1000"][tier]
                    lines.append(
                        f"| {challenge} | {block['n_cases']} | {tier} | {over['k']}/{over['n']} "
                        f"| {self._fmt(over['rate'])} | {self._fmt(over['wilson_95_upper'])} |"
                    )
            hypothesis = payload["hypothesis_over_1000"]
            lines += ["", f"**假说对照**:{hypothesis['convention']}", "", f"**判读**:{hypothesis['reading']}", ""]
            total_line = "、".join(f"{tier} {totals['k']}/{totals['n']}" for tier, totals in hypothesis["totals"].items() if totals["n"])
            lines.append(f"跨挑战合计:>1000 体素 {total_line or '无有效层'}。")
        lines += [
            "",
            "## 逐 case 明细",
            "",
        ]
        if payload["emb_pool"]:
            lines += [
                "### emb 池",
                "",
                "| case | sub | raw 99.5p | 外推高 max | MAE noclip >1.0 | MAE noclip [0,1] | MAE clip >1.0 | MAE clip [0,1] | MAE clip 共同目标 >1.0 | MAE clip 共同目标 [0,1] | n_over/n_within | 排除 |",
                "|---|---|---:|---:|---|---|---|---|---|---|---|---|",
            ]
            for row in payload["emb_pool"]["per_case"]:
                mae = row["mae"] or {}
                lines.append(
                    f"| {row['case']} | {row['sub'] or ''} | {self._fmt(mae.get('raw_percentile_upper'))} "
                    f"| {self._fmt(mae.get('extrapolation_max'))} "
                    f"| {self._fmt(mae.get('noclip_over'))} | {self._fmt(mae.get('noclip_within'))} "
                    f"| {self._fmt(mae.get('clip_over'))} | {self._fmt(mae.get('clip_within'))} "
                    f"| {self._fmt(mae.get('clip_over_shared'))} | {self._fmt(mae.get('clip_within_shared'))} "
                    f"| {mae.get('n_over') if mae else ''}/{mae.get('n_within') if mae else ''} | {row['excluded'] or ''} |"
                )
        if payload["gen_pool"]:
            lines += [
                "",
                "### gen 池",
                "",
                "| case | 挑战 | gen top05 全脑 | gen top05 瘤内 | >1000 全脑 | >1000 瘤内 | >1000 ET | 空 pred | 排除 |",
                "|---|---|---:|---:|---|---|---|---|---|",
            ]
            for row in payload["gen_pool"]["per_case"]:
                over = row["over_1000"] or {}

                def _kn(tier):
                    block = over.get(tier)
                    return f"{block['k']}/{block['n']}" if block else ""

                gen_tumour = (row["gen"] or {}).get("tumour")
                lines.append(
                    f"| {row['case']} | {row['challenge'] or ''} "
                    f"| {self._fmt((row['gen'].get('brain') or {}).get('top05_mean'))} "
                    f"| {self._fmt(gen_tumour and gen_tumour.get('top05_mean'))} "
                    f"| {_kn('brain')} | {_kn('tumour')} | {_kn('et')} "
                    f"| {row['pred_empty'] if row['pred_empty'] is not None else ''} | {row['excluded'] or ''} |"
                )
        lines += [
            "",
            "直方图数据(固定边 [0, 2],50 桶+溢出/下溢桶,池级计数累加;下溢承接 decode 负值)见同名 json 的"
            " `emb_pool.aggregate.*_histograms` / `gen_pool.aggregate.*_histograms`。",
            "",
            ">1000 输出域假说(「亮核在 >1.0 输出域、被评估 clip 掉」)的对照读数见 json `hypothesis_over_1000`;gen 池未运行时无读数。",
            "",
        ]
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv=None, *, reconstructor_factory=None, grid=TRAINING_GRID, align=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-list", default=None, help="MONAI training list JSON; its t1c entries form the emb pool")
    parser.add_argument("--data-root", default=None, help="data_base_dir the list's image paths resolve against")
    parser.add_argument("--emb-root", default=None, help="embedding_base_dir holding the fp32 <case>_emb.nii.gz artifacts")
    parser.add_argument("--samples", default=None, help="the retained generation samples.json (P1 holdout layout: case_id/challenge/samples)")
    parser.add_argument("--real-root", default=None, help="holdout real root (<real-root>/<CH>/<case>/<case>-t1c.nii.gz)")
    parser.add_argument("--pred-root", default=None, help="frozen-instrument prediction root (<pred-root>/<CH>/<case>__gen.nii.gz)")
    parser.add_argument("-e", "--env-config", default=None, help="environment json (trained_autoencoder_path)")
    parser.add_argument("-c", "--model-config", default=None, help="model config json")
    parser.add_argument("-t", "--model-def", default=None, help="network def json (autoencoder_def)")
    parser.add_argument("--device", default="cpu", help="torch device for the VAE arms (cpu / cuda:0)")
    parser.add_argument("--limit", type=int, default=None, help="uniform stride subsample per pool (diagnostic scale)")
    parser.add_argument(
        "--gen-limit",
        type=int,
        default=None,
        help="overrides --limit for the gen pool alone (its readings are the per-challenge k/n denominators, so full scope is cheap: pure CPU)",
    )
    parser.add_argument("--bootstrap-b", type=int, default=10000, help="bootstrap resamples for the MAE CI90")
    parser.add_argument("--output-dir", required=True, help="sugon artifact area for the diagnostic report (never git)")
    parser.add_argument("--run-id", default=None, help="the candidate's run id, recorded into the report")
    args = parser.parse_args(argv)

    if args.train_list is None and args.samples is None:
        raise DiagnosticError("nothing to read: pass --train-list (emb pool) and/or --samples (gen pool)")
    factory = reconstructor_factory or VaeReconstructor
    align = align or GenPool.align_pred_to_generated_grid

    def stride(entries, limit):
        """Uniform stride subsample: keep the pool's position spread at diagnostic
        scale instead of taking a head slice. ``None`` or 0 or less = no cap (full
        scope; the gen pool passes 0 when --gen-limit is set empty)."""
        if limit is None or limit <= 0 or limit >= len(entries):
            return entries
        indices = np.round(np.linspace(0, len(entries) - 1, limit)).astype(int)
        return [entries[i] for i in dict.fromkeys(indices)]

    emb_rows = []
    if args.train_list is not None:
        required = {
            "--data-root": args.data_root,
            "--emb-root": args.emb_root,
            "-e/--env-config": args.env_config,
            "-c/--model-config": args.model_config,
            "-t/--model-def": args.model_def,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise DiagnosticError(f"the emb pool needs {', '.join(missing)}")
        entries = [entry for entry in json.loads(Path(args.train_list).read_text())["training"] if "t1c" in entry["image"]]
        recon = factory(args.env_config, args.model_config, args.model_def, args.device)
        repo = NiftiTrainCaseRepository(args.data_root, args.emb_root)
        emb_rows = EmbPool().read_cases(stride(entries, args.limit), repo, recon, TrainingPreprocessing.resize_image, grid)
        print(f"[OK] emb pool: {len(emb_rows)} cases -> aggregating")

    gen_rows = []
    if args.samples is not None:
        entries = stride(json.loads(Path(args.samples).read_text()), args.gen_limit if args.gen_limit is not None else args.limit)
        repo = NiftiGenCaseRepository(args.real_root, args.pred_root)
        gen_rows = GenPool().read_cases(entries, repo, align=align)
        print(f"[OK] gen pool: {len(gen_rows)} cases -> aggregating")

    report = IntensityDomainReport(args.train_list, args.samples, args.bootstrap_b, run_id=args.run_id)
    json_path, md_path = report.write(emb_rows, gen_rows, Path(args.output_dir))
    print(f"[OK] {len(emb_rows)} emb + {len(gen_rows)} gen cases (variant=diagnostic) -> {json_path}")
    print(f"[OK] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
