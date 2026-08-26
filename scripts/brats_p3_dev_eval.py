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

"""P3 dev light-acceptance sidecar: fixed image-conditioned samples + PSNR/SSIM trend (issue #61).

Runs beside the P3 ControlNet finetune on a reserved GPU. For every ``epoch_<N>.pt`` the
trainer persists (N a multiple of ``--eval-every``), it:

1. generates the FIXED dev cohort — the dev-side cases × the 12 ordered src->tgt pairs,
   conditioned on the case's **src-image latent** (``src_image``, 4ch, no mask) with the
   target modality label and **CFG off** (``cfg_guidance_scale=0``, issue #61 acceptance
   criterion 1-2), fixed per-(case, src, tgt) seed;
2. scores every pair with 3D PSNR/SSIM against the case's REAL target volume (the paired
   metric family the L1 acceptance uses): the real target is resampled onto the 256x256x128
   generation grid (the same ``ReferenceGridWriter`` chain as the phase's reference_grid,
   cached under ``reference_grid/``), both volumes run the pinned L1 [0,1] protocol
   (per-volume 0-99.5 percentile, ``MRIntensityNormalizer``), then skimage PSNR and 3D SSIM
   (``data_range=1.0``, ``win_size=7``) — identical parameters to ``P3PairMetricCalculator``;
3. applies the PRE-RECORDED early-stop rule and, when it fires, writes
   ``<ckpt_dir>/.early_stop``; ``select`` emits the final dev-side checkpoint selection
   (argmax mean SSIM) for the phase-run contract.

The early-stop rule (recorded verbatim in the run dir before training starts): metric
m(N) = mean over the four target modalities of the case-mean dev 3D SSIM at epoch N
(PSNR recorded alongside); stop when N >= --min-epoch AND the last --patience consecutive
evals produced no new best m (higher is better).

The dev cohort / real bank / spacing / src-latent source are all filtered to the dev side:
``p3_pairs.json`` mixes train (fold=1) and dev (fold=0); this script uses only fold=0.

Usage (sugon, one reserved GPU):
    python -m scripts.brats_p3_dev_eval reference --dev-list ... --raw-root ... --eval-root DIR
    python -m scripts.brats_p3_dev_eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --phase-root ... -e env.json -c config.json -t network_p3.json
    python -m scripts.brats_p3_dev_eval select --eval-root DIR --ckpt-dir DIR
    python -m scripts.brats_p3_dev_eval selftest --workdir TMP
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .brats_l1_quantitative import MRIntensityNormalizer
from .brats_p1_dev_eval import (
    COHORT_QUOTAS,
    MODALITY_TOKENS,
    STOP_FILE,
    CheckpointWatcher,
    EarlyStopRule,
    TrendLedger,
)
from .brats_p3_controlnet_manifest import P3CandidateInferenceConfig
from .brats_p3_stage0_generate import ReferenceGridWriter
from .brats_phase_prep import MODALITIES as PAIR_MODALITIES
from .diff_model_setting import load_config
from .utils_infer import load_image_models, run_controlnet_conditioned_image_dm

LATENT = (4, 64, 64, 32)
GRID = (256, 256, 128)
SSIM_WIN = 7
PSNR_CAP_DB = 100.0  # identical volumes (mse=0) are capped instead of inf so JSON stays finite
P3_RULE_TEXT = (
    "metric m(N) = mean over t1n/t1c/t2w/t2f of dev case-mean 3D SSIM (skimage, data_range=1.0, "
    "win_size=7, per-volume 0-99.5 percentile [0,1] protocol) of candidates vs the real target "
    "resampled onto the generation grid; PSNR (same protocol) is recorded alongside; stop when "
    "N >= {min_epoch} and the last {patience} consecutive evals set no new best m; "
    "hard cap = trainer n_epochs"
)


def read_src_latent(src_image_path, device):
    """Loads a 4ch src-image latent NIfTI as (1,4,H,W,D) on the pinned grid (RAS, float).

    Uses the same transform chain as the training dataloader (LoadImage ->
    EnsureChannelFirst -> Orientation RAS) so the dev-eval condition matches the
    training condition exactly: the stored 4D latent is (X,Y,Z,C) and
    EnsureChannelFirst moves the channel axis to the front.
    """
    import monai.transforms as monai_t

    transform = monai_t.Compose(
        [
            monai_t.LoadImage(image_only=True),
            monai_t.EnsureChannelFirst(),
            monai_t.Orientation(axcodes="RAS"),
            monai_t.EnsureType(dtype=torch.float32),
        ]
    )
    x = transform(str(src_image_path))  # (C,H,W,D)
    return x[None].to(device)  # (1,C,H,W,D)


class P3DevList:
    """The dev (fold=0) view of the ``p3_pairs.json`` list, with raw tgt paths for the real bank."""

    def __init__(self, dev_list_path, eval_root):
        self._source = Path(dev_list_path)
        self._eval_root = Path(eval_root)

    def built_path(self):
        return self._eval_root / "dev_list.json"

    def build(self):
        out = self.built_path()
        if out.is_file():
            return out
        entries = json.loads(self._source.read_text())["training"]
        dev = []
        for entry in entries:
            if entry["fold"] != 0:
                continue
            # p3 image is the *embedding* path (embeddings/.../<case>-<mod>_emb.nii.gz);
            # the real bank needs the raw tgt volume relative to --raw-root.
            raw = entry["image"].replace("embeddings/", "").replace("_emb.nii.gz", ".nii.gz")
            dev.append({**copy.deepcopy(entry), "image": raw})
        self._eval_root.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"training": dev}, indent=1) + "\n")
        print(f"p3 dev list: {len(dev)} entries -> {out}")
        return out


class P3DevCohort:
    """Builds the dev-case × 12-ordered-pair generation plan from the dev list."""

    def __init__(self, dev_list_path):
        self._entries = json.loads(Path(dev_list_path).read_text())["training"]

    def cases(self):
        seen, cases = set(), []
        for entry in self._entries:
            if entry["case"] not in seen:
                seen.add(entry["case"])
                cases.append({"sub": entry["sub"], "case": entry["case"]})
        return cases

    def spacing_of(self, case):
        for entry in self._entries:
            if entry["case"] == case:
                return entry["spacing"]

    def src_image_of(self, case, src_suffix):
        # list fields carry the long mapping keys (mri_*), translate the BraTS file suffix
        for entry in self._entries:
            if entry["case"] == case and entry["src_modality"] == PAIR_MODALITIES[src_suffix][0]:
                return entry["src_image"]

    def tgt_of(self, case, tgt_suffix):
        for entry in self._entries:
            if entry["case"] == case and entry["modality"] == PAIR_MODALITIES[tgt_suffix][0]:
                return entry["image"]


class P3PairwiseScorer:
    """3D PSNR/SSIM of candidates vs real targets under the pinned L1 [0,1] protocol.

    The real target is resampled onto the 256x256x128 generation grid (the same
    ``ReferenceGridWriter`` chain the phase's reference_grid uses, cached under
    ``<eval_root>/reference_grid/<CH>/<case>/<tgt>.nii.gz`` so each (case, tgt) is
    resampled once across all eval points); each pair then runs the pinned L1
    intensity protocol (per-volume 0-99.5 percentile -> [0,1]) and is scored with
    skimage PSNR and 3D SSIM (``data_range=1.0``, ``win_size=7``) — the identical
    parameters ``P3PairMetricCalculator`` uses for the final L1 judgment, so the
    dev-selection trend stays predictive of the acceptance verdict.
    """

    def __init__(self, workers=32):
        self._workers = max(1, int(workers))

    @staticmethod
    def reference_job(job):
        ref_root, challenge, case, tgt, real_path, spacing = job
        try:
            out = ReferenceGridWriter(Path(ref_root)).write(challenge, case, tgt, real_path, spacing)
            return {"error": None, "path": str(out)}
        except Exception as error:
            return {"error": f"reference {challenge}/{case}/{tgt}: {error}"}

    @staticmethod
    def score_arrays(reference, sample):
        """PSNR/SSIM of two same-shape raw intensity arrays under the pinned L1 [0,1] protocol."""
        normalizer = MRIntensityNormalizer()
        scaled_reference = normalizer.normalize(reference, "reference")
        scaled_sample = normalizer.normalize(sample, "candidate")
        psnr = float(min(peak_signal_noise_ratio(scaled_reference, scaled_sample, data_range=1.0), PSNR_CAP_DB))
        ssim = float(
            structural_similarity(
                scaled_reference, scaled_sample, data_range=1.0, channel_axis=None, win_size=SSIM_WIN
            )
        )
        return {"error": None, "psnr": psnr, "ssim": ssim}

    @staticmethod
    def score_job(job):
        reference_path, sample_path = job
        try:
            reference = np.asarray(nib.load(str(reference_path)).dataobj, dtype=np.float64)
            sample = np.asarray(nib.load(str(sample_path)).dataobj, dtype=np.float64)
            if reference.shape != sample.shape:
                raise ValueError(f"shape mismatch reference {reference.shape} vs sample {sample.shape}")
            return P3PairwiseScorer.score_arrays(reference, sample)
        except Exception as error:
            return {"error": str(error)}

    def score_cohort(self, samples, cohort_source, raw_root, ref_root):
        """Scores the generated cohort; returns per-target-modality mean PSNR/SSIM + m (mean SSIM).

        ``samples`` is the ``generate_cohort`` output (one dict per pair, with
        ``sub``, ``case``, ``target_modality``, ``path``). Real target paths are
        resolved from the dev list via ``cohort_source.tgt_of`` (relative to
        ``raw_root``); every (case, tgt) reference is resampled to the grid once.
        Raises when any reference or pair fails (a partial trend would be
        misleading for the patience rule).
        """
        raw_root = Path(raw_root)
        ref_root = Path(ref_root)
        references = {}
        for sample in samples:
            key = (sample["sub"], sample["case"], sample["target_modality"])
            if key in references:
                continue
            real_path = raw_root / cohort_source.tgt_of(sample["case"], sample["target_modality"])
            references[key] = (
                ref_root,
                sample["sub"],
                sample["case"],
                sample["target_modality"],
                str(real_path),
                cohort_source.spacing_of(sample["case"]),
            )
        with ProcessPoolExecutor(max_workers=self._workers, mp_context=mp.get_context("fork")) as pool:
            reference_results = list(pool.map(P3PairwiseScorer.reference_job, references.values()))
        reference_paths = {}
        for key, result in zip(references, reference_results):
            if result["error"] is not None:
                raise RuntimeError(f"reference grid failed for {key}: {result['error']}")
            reference_paths[key] = result["path"]
        jobs = [(reference_paths[(s["sub"], s["case"], s["target_modality"])], s["path"]) for s in samples]
        with ProcessPoolExecutor(max_workers=self._workers, mp_context=mp.get_context("fork")) as pool:
            pair_results = list(pool.map(P3PairwiseScorer.score_job, jobs))
        per_modality = {modality: {"psnr": [], "ssim": []} for modality in MODALITY_TOKENS}
        for sample, result in zip(samples, pair_results):
            if result["error"] is not None:
                raise RuntimeError(f"pair score failed for {sample['case']} {sample['target_modality']}: {result['error']}")
            per_modality[sample["target_modality"]]["psnr"].append(result["psnr"])
            per_modality[sample["target_modality"]]["ssim"].append(result["ssim"])
        report = {
            modality: {
                "psnr": float(np.mean(per_modality[modality]["psnr"])),
                "ssim": float(np.mean(per_modality[modality]["ssim"])),
            }
            for modality in MODALITY_TOKENS
        }
        m = float(np.mean([report[modality]["ssim"] for modality in MODALITY_TOKENS]))
        mean_psnr = float(np.mean([report[modality]["psnr"] for modality in MODALITY_TOKENS]))
        return {"report": report, "m": m, "mean_psnr": mean_psnr}


class P3CandidateSampler:
    """Generates the fixed dev cohort with a P3 ControlNet checkpoint (cfg=0, 30 steps)."""

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    def load_models(self, checkpoint_path):
        self._args.trained_controlnet_path = str(checkpoint_path)
        autoencoder, unet, controlnet, scale_factor, _noise_scheduler = load_image_models(self._args, self._device)
        for model in (autoencoder, unet, controlnet):
            model.eval()
        torch.cuda.empty_cache()
        return autoencoder, unet, controlnet, scale_factor

    @torch.inference_mode()
    def sample_one(self, autoencoder, unet, controlnet, scale_factor, spacing, modality_token, seed, src_latent):
        from monai.networks.schedulers import RFlowScheduler

        torch.manual_seed(seed)
        noise_scheduler = RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"})
        # ControlNet condition: the (already scaled to the model's normalized space) src latent.
        cond = (src_latent * scale_factor).half().to(self._device)
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
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
            modality_tensor=modality_tensor,
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            cfg_guidance_scale=0.0,
            controlnet_uncond_tensor=None,
        )
        return np.clip(synthetic.squeeze().float().cpu().numpy(), 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cases, cohort_source, phase_root, out_dir):
        import nibabel as nib

        autoencoder, unet, controlnet, scale_factor = self.load_models(checkpoint_path)
        samples = []
        for case in cases:
            spacing = cohort_source.spacing_of(case["case"])
            for src in MODALITY_TOKENS:
                for tgt in MODALITY_TOKENS:
                    if src == tgt:
                        continue
                    seed = P3CandidateInferenceConfig.seed_of(case["case"], src, tgt)
                    src_latent = read_src_latent(phase_root / cohort_source.src_image_of(case["case"], src), self._device)
                    out = Path(out_dir) / case["sub"] / f"{case['case']}_{src}_to_{tgt}_seed{seed}.nii.gz"
                    if not out.is_file():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        data = self.sample_one(autoencoder, unet, controlnet, scale_factor, spacing, MODALITY_TOKENS[tgt], seed, src_latent)
                        nib.save(nib.Nifti1Image(data, np.diag([spacing[0], spacing[1], spacing[2], 1.0])), out)
                    samples.append({"sub": case["sub"], "case": case["case"], "src_modality": src, "target_modality": tgt, "path": str(out)})
        del autoencoder, unet, controlnet
        torch.cuda.empty_cache()
        return samples


class P3DevEvalSelfTest:
    """Fixture check of p3-specific logic: dev-view, cohort, src-latent read, cfg=0 (numpy/stdlib)."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        src_entries = []
        for challenge, quota in COHORT_QUOTAS.items():
            for index in range(quota):
                case = f"FIX{challenge}-{index:04d}-000"
                for src in ("t1n", "t1c", "t2w", "t2f"):
                    for tgt in ("t1n", "t1c", "t2w", "t2f"):
                        if src == tgt:
                            continue
                        src_entries.append(
                            {
                                "image": f"embeddings/{challenge}/{case}/{case}-{tgt}_emb.nii.gz",
                                "src_image": f"embeddings/{challenge}/{case}/{case}-{src}_emb.nii.gz",
                                "label": f"labels/{challenge}/{case}/{case}-tumor129.nii.gz",
                                "spacing": [1.0, 1.0, 1.0],
                                "modality": PAIR_MODALITIES[tgt][0],
                                "src_modality": PAIR_MODALITIES[src][0],
                                "fold": 0,
                                "sub": challenge,
                                "case": case,
                            }
                        )
        src = self._workdir / "p3_src.json"
        src.write_text(json.dumps({"training": src_entries}))
        out = P3DevList(src, self._workdir).build()
        entries = json.loads(out.read_text())["training"]
        expected = sum(quota for quota in COHORT_QUOTAS.values()) * 12
        if len(entries) != expected:
            self.failures.append(f"dev-view kept {len(entries)} entries, expected {expected} (12 ordered pairs per dev case)")
        if not entries[0]["image"].endswith("-t1c.nii.gz") or "_emb" in entries[0]["image"]:
            self.failures.append(f"raw tgt not derived from embedding path: {entries[0]['image']}")
        if "src_image" not in entries[0]:
            self.failures.append("dev-view dropped the src_image condition")

        cohort_source = P3DevCohort(out)
        cohort = cohort_source.cases()
        n_cases = sum(quota for quota in COHORT_QUOTAS.values())
        if len(cohort) != n_cases:
            self.failures.append(f"cohort has {len(cohort)} dev cases, expected {n_cases}")
        if {item["sub"] for item in cohort} != set(COHORT_QUOTAS):
            self.failures.append("cohort missing a challenge")
        # the real pairs list keys src_modality/modality by the long mapping keys (mri_*);
        # the lookups must resolve the BraTS suffixes through that translation
        for suffix in PAIR_MODALITIES:
            if cohort_source.src_image_of(cohort[0]["case"], suffix) is None:
                self.failures.append(f"src lookup unresolved for {cohort[0]['case']} {suffix}")

        # src-latent channel-axis read: write a (H,W,D,C) NIfTI and confirm (C,H,W,D).
        import nibabel as nib

        latent = np.zeros((32, 32, 16, 4), dtype=np.float32)
        latent[..., 0] = 1.0
        latent_path = self._workdir / "latent.nii.gz"
        nib.save(nib.Nifti1Image(latent, np.eye(4)), str(latent_path))
        tensor = read_src_latent(latent_path, torch.device("cpu"))
        if tuple(tensor.shape) != (1, 4, 32, 32, 16):
            self.failures.append(f"src-latent read shape {tuple(tensor.shape)} != (1,4,32,32,16)")
        elif float(tensor[0, 0].mean()) != 1.0:
            self.failures.append("src-latent channel axis mis-read (channel 0 not the brain-modality slot)")

        # early-stop rule + selection: P3 pre-registered the max direction (PSNR/SSIM higher is better)
        rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100, direction="max")
        improving = [{"epoch": e, "m": 0.1 + 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
        stop, _ = rule.should_stop(improving)
        if stop:
            self.failures.append("rule stopped an improving (max-direction) trend")
        plateau = improving + [{"epoch": e, "m": 0.4} for e in (35, 40, 45)]
        stop, reason = rule.should_stop(plateau)
        if not stop:
            self.failures.append(f"rule failed to stop a 3-eval plateau ({reason})")
        selection = EarlyStopRule.selection(
            [{"epoch": 5, "m": 1.2}, {"epoch": 10, "m": 0.8}, {"epoch": 20, "m": 0.8}], direction="max", metric_name="mean_ssim"
        )
        if selection["epoch"] != 5 or selection["mean_ssim"] != 1.2:
            self.failures.append(f"max selection picked {selection}, expected epoch 5 m=1.2")

        # pairwise scorer: identical volumes score ssim=1.0 and capped psnr; degraded ones score lower
        rng = np.random.default_rng(0)
        reference = (50 + 400 * rng.random((32, 32, 32))).astype(np.int16)
        identical = P3PairwiseScorer.score_arrays(reference, reference)
        if abs(identical["ssim"] - 1.0) > 1e-6 or identical["psnr"] != PSNR_CAP_DB:
            self.failures.append(f"identical volumes scored {identical}, expected ssim=1.0 psnr=capped")
        degraded = P3PairwiseScorer.score_arrays(
            reference,
            np.clip((reference.astype(np.float64) + rng.normal(0, 80, reference.shape)).round(), 0, None).astype(np.int16),
        )
        if degraded["ssim"] >= identical["ssim"] or degraded["psnr"] >= identical["psnr"]:
            self.failures.append(f"degraded volume scored {degraded}, expected lower than identical")
        shuffled = P3PairwiseScorer.score_arrays(reference, np.roll(reference, 8, axis=0))
        if shuffled["ssim"] > 0.9:
            self.failures.append(f"rolled volume scored ssim={shuffled['ssim']:.3f}, expected substantial degradation")
        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="pre-resample all dev real targets onto the generation grid")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--score-workers", type=int, default=32)

    p = sub.add_parser("watch", help="sidecar loop: evaluate epoch checkpoints as they land")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--eval-root", required=True)
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--phase-root", required=True, help="phase root holding embeddings/labels (src-image latents)")
    p.add_argument("-e", "--env_config_path", required=True)
    p.add_argument("-c", "--model_config_path", required=True)
    p.add_argument("-t", "--model_def_path", required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-epoch", type=int, default=30)
    p.add_argument("--max-epoch", type=int, default=100)
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--score-workers", type=int, default=32, help="parallel CPU workers for reference resampling + PSNR/SSIM")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("selftest")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        failures = P3DevEvalSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        return 1 if failures else 0

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        dev_list = P3DevList(args.dev_list, eval_root).build()
        cohort_source = P3DevCohort(dev_list)
        raw_root, ref_root = Path(args.raw_root), eval_root / "reference_grid"
        references = {}
        for case in cohort_source.cases():
            for tgt in MODALITY_TOKENS:
                real = raw_root / cohort_source.tgt_of(case["case"], tgt)
                references[(case["sub"], case["case"], tgt)] = (
                    str(ref_root),
                    case["sub"],
                    case["case"],
                    tgt,
                    str(real),
                    cohort_source.spacing_of(case["case"]),
                )
        with ProcessPoolExecutor(max_workers=args.score_workers, mp_context=mp.get_context("fork")) as pool:
            results = list(pool.map(P3PairwiseScorer.reference_job, references.values()))
        failures = [f"reference {key}: {r['error']}" for key, r in zip(references, results) if r["error"] is not None]
        if failures:
            print("\n".join(failures[:10]), file=sys.stderr)
            return 1
        print(f"real reference grids -> {ref_root} ({len(references)} (case, tgt) targets)")
        return 0

    if args.command == "select":
        trend = ledger.read()
        selection = EarlyStopRule.selection(trend, direction="max", metric_name="mean_ssim")
        if selection is None:
            print("no eval points; nothing to select", file=sys.stderr)
            return 1
        best = next(point for point in trend if point["epoch"] == selection["epoch"] and point["m"] is not None)
        selection["mean_psnr"] = best.get("mean_psnr")
        selection["rule"] = "argmax mean dev 3D SSIM over eval points (pre-registered; PSNR recorded alongside)"
        selection["trend"] = trend
        selection["recorded_utc"] = datetime.now(UTC).isoformat()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(selection, indent=2) + "\n")
        print(
            f"selection -> {out} (epoch {selection['epoch']}, mean_ssim {selection['mean_ssim']:.4f}, "
            f"mean_psnr {selection['mean_psnr']:.2f})"
        )
        return 0

    # watch mode
    dev_list = P3DevList(args.dev_list, eval_root).build()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_source = P3DevCohort(dev_list)
    cohort = cohort_source.cases()
    phase_root = Path(args.phase_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch, direction="max")
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps(
            {
                "rule": P3_RULE_TEXT.format(min_epoch=args.min_epoch, patience=args.patience),
                "patience": args.patience,
                "min_epoch": args.min_epoch,
                "max_epoch": args.max_epoch,
                "direction": "max",
            },
            indent=2,
        )
        + "\n"
    )
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 0.0
    scorer = P3PairwiseScorer(workers=args.score_workers)
    sampler = P3CandidateSampler(merged, device, None)
    watcher = CheckpointWatcher(args.ckpt_dir, args.eval_every, args.max_epoch, {r["epoch"] for r in ledger.read()})
    idle_since = None

    while True:
        pending = watcher.pending()
        if not pending:
            if args.idle_exit_seconds and idle_since is not None and time.time() - idle_since > args.idle_exit_seconds:
                break
            if args.idle_exit_seconds and idle_since is None:
                idle_since = time.time()
            time.sleep(args.poll_seconds)
            continue
        idle_since = None
        for epoch, path in pending:
            if any(r["epoch"] == epoch for r in ledger.read()):
                watcher.mark_done(epoch)
                continue
            epoch_dir = eval_root / f"epoch_{epoch}"
            try:
                samples = sampler.generate_cohort(path, cohort, cohort_source, phase_root, epoch_dir / "samples")
                scored = scorer.score_cohort(samples, cohort_source, args.raw_root, eval_root / "reference_grid")
            except Exception as error:
                print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                continue
            record = {
                "eval_utc": datetime.now(UTC).isoformat(),
                "epoch": epoch,
                "checkpoint": str(path),
                "metric": "paired-psnr-ssim",
                "report": scored["report"],
                "m": scored["m"],
                "mean_psnr": scored["mean_psnr"],
                "cohort_file": str(dev_list),
            }
            ledger.append(record)
            (epoch_dir / "trend.json").write_text(json.dumps(record, indent=2) + "\n")
            watcher.mark_done(epoch)
            stop, reason = rule.should_stop(ledger.read())
            print(
                f"[eval] epoch {epoch}: mean_ssim={scored['m']:.4f} mean_psnr={scored['mean_psnr']:.2f} "
                f"stop={stop} ({reason})",
                flush=True,
            )
            if stop:
                (Path(args.ckpt_dir) / STOP_FILE).write_text(json.dumps({"reason": reason, "epoch": epoch}) + "\n")
                print(f"early-stop fired ({reason}); wrote {Path(args.ckpt_dir) / STOP_FILE}", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
