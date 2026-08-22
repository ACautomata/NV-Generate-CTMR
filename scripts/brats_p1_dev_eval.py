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

"""P1 dev light-acceptance sidecar: fixed samples + FID trend + L2 trend (issue #57, spec #51 §6).

Runs beside the P1 finetune on a reserved GPU. For every ``epoch_<N>.pt`` the
trainer persists (N a multiple of ``--eval-every``), it:

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

Usage (sugon, one reserved GPU):
    python -m scripts.brats_p1_dev_eval reference --dev-list ... --raw-root ... --eval-root DIR
    python -m scripts.brats_p1_dev_eval watch --ckpt-dir ... --eval-root ... \
        --dev-list ... --raw-root ... --emb-root ... -e env.json -c config.json -t network.json
    python -m scripts.brats_p1_dev_eval select --eval-root DIR --ckpt-dir DIR
    python -m scripts.brats_p1_dev_eval selftest --workdir TMP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .brats_l1_quantitative import FidScoreCalculator
from .diff_model_setting import load_config
from .utils import define_instance, dynamic_infer
from .utils_infer import ReconModel

MODALITY_TOKENS = {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}

# The run-local reference bank payload holds numpy arrays; weights_only rejects
# them by default, so allowlist numpy reconstruction (bank is a local artifact).
torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType]
)
TARGET_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
PLANES = ("xy", "yz", "zx")
COHORT_QUOTAS = {"GLI": 4, "SSA": 2, "MEN": 4, "METS": 3, "PED": 3}
TREND_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad_240x240x160"
FEATURE_SHAPE = (240, 240, 160)
EMPTY_SLICE_THRESHOLD = 0.05
SAMPLE_EVERY_K = 2
STOP_FILE = ".early_stop"

# Frozen instrument chain (ADR-0003 / l2_calibration_predict.sh conventions).
INSTRUMENT_DATASETS = {
    "GLI": "Dataset501_BraTS2023GLI",
    "SSA": "Dataset502_BraTS2023SSA",
    "MEN": "Dataset503_BraTS2023MEN",
    "METS": "Dataset504_BraTS2023METS",
    "PED": "Dataset505_BraTS2023PED",
}
INSTRUMENT_DEFAULT = {"plans": "nnUNetPlans", "config": "3d_fullres"}
INSTRUMENT_SSA = {"plans": "nnUNetPlans_SSA_bs16_v1", "config": "3d_fullres_bs16"}


class DevCohortBuilder:
    """The fixed 16-case dev cohort: sha256((sub, case)) order within per-challenge quotas."""

    def __init__(self, dev_list_path):
        self._path = Path(dev_list_path)

    def build(self):
        cases_by_challenge = {}
        for entry in json.loads(self._path.read_text())["training"]:
            cases_by_challenge.setdefault(entry["sub"], set()).add(entry["case"])
        cohort = []
        for challenge in sorted(COHORT_QUOTAS):
            ordered = sorted(
                cases_by_challenge.get(challenge, set()),
                key=lambda case: hashlib.sha256(f"{challenge}/{case}".encode()).hexdigest(),
            )
            for case in ordered[: COHORT_QUOTAS[challenge]]:
                cohort.append({"sub": challenge, "case": case})
        return cohort

    def write(self, out_path):
        cohort = self.build()
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cohort": cohort, "quotas": COHORT_QUOTAS}, indent=1) + "\n")
        return cohort


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


class CheckpointWatcher:
    """Polls the trainer's epoch checkpoints; yields un-evaluated eval points."""

    def __init__(self, ckpt_dir, eval_every, max_epoch, done_epochs=()):
        self._ckpt_dir = Path(ckpt_dir)
        self._eval_every = eval_every
        self._max_epoch = max_epoch
        # Seed from the ledger so a sidecar restart does not re-evaluate history
        # (re-appended trend points would corrupt the early-stop patience count).
        self._done = set(done_epochs)

    def pending(self):
        found = []
        for path in sorted(self._ckpt_dir.glob("epoch_*.pt")):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if epoch % self._eval_every == 0 and epoch <= self._max_epoch and epoch not in self._done:
                found.append((epoch, path))
        return sorted(found)

    def mark_done(self, epoch):
        self._done.add(epoch)


class CandidateSampler:
    """Generates the fixed dev cohort samples with a candidate checkpoint (cfg=10, 30 steps)."""

    def __init__(self, args, device, logger):
        self._args = args
        self._device = device
        self._logger = logger

    @staticmethod
    def seed_of(case, modality):
        return int(hashlib.sha256(f"{case}|{modality}".encode()).hexdigest()[:8], 16) % (2**31 - 1)

    def load_models(self, checkpoint_path):
        autoencoder = define_instance(self._args, "autoencoder_def").to(self._device)
        ae_ckpt = torch.load(self._args.trained_autoencoder_path, map_location=self._device, weights_only=True)
        if "unet_state_dict" in ae_ckpt:
            ae_ckpt = ae_ckpt["unet_state_dict"]
        autoencoder.load_state_dict(ae_ckpt)
        unet = define_instance(self._args, "diffusion_unet_def").to(self._device)
        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
        autoencoder.eval()
        unet.eval()
        # Upstream inference convention is fp16 on the DCU (float16 latents);
        # a half-precision model keeps the conv input/weight/bias set consistent
        # (the HIP bf16 SDPA flash path emits fp16 and breaks the mixed chain).
        autoencoder = autoencoder.half()
        unet = unet.half()
        scale = float(ckpt["scale_factor"])
        return unet, ReconModel(autoencoder=autoencoder, scale_factor=scale).to(self._device).half()

    @torch.inference_mode()
    def sample_one(self, unet, recon_model, modality_token, spacing, seed, output_size=(256, 256, 128)):
        from monai.inferers import SlidingWindowInferer
        from monai.networks.schedulers import RFlowScheduler

        torch.manual_seed(seed)
        noise_scheduler = RFlowScheduler(**{k: v for k, v in self._args.noise_scheduler.items() if k != "_target_"})
        divisor = 4
        image = torch.randn((1, 4, output_size[0] // divisor, output_size[1] // divisor, output_size[2] // divisor), device=self._device)
        noise_scheduler.set_timesteps(
            num_inference_steps=self._args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(image.shape[2:])),
        )
        spacing_tensor = torch.tensor([[s * 1e2 for s in spacing]], device=self._device)
        modality_tensor = torch.tensor([modality_token], device=self._device)
        all_timesteps = noise_scheduler.timesteps
        all_next = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
        cfg = self._args.cfg_guidance_scale
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            for t, next_t in zip(all_timesteps, all_next):
                unet_inputs = {
                    "x": image,
                    "timesteps": torch.Tensor((t,)).to(self._device),
                    "spacing_tensor": spacing_tensor,
                    "class_labels": modality_tensor,
                }
                if cfg > 0:
                    unet_inputs = {
                        key: (torch.cat([value, value]) if key != "class_labels" else torch.cat([value, torch.zeros_like(value)]))
                        for key, value in unet_inputs.items()
                    }
                    model_t, model_uncond = unet(**unet_inputs).chunk(2)
                    model_output = model_uncond + cfg * (model_t - model_uncond)
                else:
                    model_output = unet(**unet_inputs)
                image, _ = noise_scheduler.step(model_output, t, image, next_t)
        inferer = SlidingWindowInferer(
            roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25, sw_device=self._device, device=torch.device("cpu")
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            synthetic = dynamic_infer(inferer, recon_model, image).squeeze().float().cpu().numpy()
        data = synthetic * 1000.0  # [0,1] -> MR 0..1000 scale, upstream int16 convention
        return np.clip(data, 0, None).astype(np.int16)

    def generate_cohort(self, checkpoint_path, cohort, spacings, out_dir):
        unet, recon = self.load_models(checkpoint_path)
        samples = []
        for item in cohort:
            for modality in TARGET_MODALITIES:
                seed = self.seed_of(item["case"], modality)
                out = Path(out_dir) / item["sub"] / f"{item['case']}_{modality}_seed{seed}.nii.gz"
                if not out.is_file():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data = self.sample_one(unet, recon, MODALITY_TOKENS[modality], spacings.spacing_of(item["case"]), seed)
                    image = nib.Nifti1Image(data, affine=np.diag([1.0, 1.0, 1.0, 1.0]))
                    nib.save(image, out)
                samples.append({"sub": item["sub"], "case": item["case"], "modality": modality, "path": str(out)})
        del unet, recon
        torch.cuda.empty_cache()
        return samples


class MrTrendFeatures:
    """Percentile->RAS->1mm->zero-pad preprocessing + per-plane RadImageNet slice features."""

    def __init__(self, device):
        self._device = device
        self._network = None

    def network(self):
        if self._network is None:
            self._network = torch.hub.load(
                # Explicit ":main" ref: torch.hub probes github.com for the default
                # branch when no ref is given; on the sugon the probe dies with
                # RemoteDisconnected (not URLError) before the cache fallback.
                "Warvito/radimagenet-models:main", model="radimagenet_resnet50", trust_repo=True, verbose=False
            ).to(self._device).eval()
        return self._network

    @staticmethod
    def preprocess(path):
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        values = data[data > 0] if (data > 0).any() else data.ravel()
        lo, hi = np.percentile(values, [0.0, 99.5])
        data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        # 1 mm isotropic resample (linear, background 0) then zero pad/crop.
        zoom = image.header.get_zooms()[:3]
        new_shape = tuple(max(1, int(round(dim * spacing))) for dim, spacing in zip(data.shape, zoom))
        rows = (np.arange(new_shape[0]) * (data.shape[0] / new_shape[0])).astype(int).clip(max=data.shape[0] - 1)
        cols = (np.arange(new_shape[1]) * (data.shape[1] / new_shape[1])).astype(int).clip(max=data.shape[1] - 1)
        depths = (np.arange(new_shape[2]) * (data.shape[2] / new_shape[2])).astype(int).clip(max=data.shape[2] - 1)
        resampled = data[np.ix_(rows, cols, depths)]
        padded = np.zeros(FEATURE_SHAPE, dtype=np.float32)
        limits = tuple(min(s, t) for s, t in zip(resampled.shape, FEATURE_SHAPE))
        padded[: limits[0], : limits[1], : limits[2]] = resampled[: limits[0], : limits[1], : limits[2]]
        return padded

    def volume_features(self, path):
        """Per-plane RadImageNet slice features of one preprocessed volume."""
        array = self.preprocess(path)  # [X, Y, Z] float32 in [0, 1]
        planes = {}
        for plane, axis in (("xy", 2), ("yz", 0), ("zx", 1)):
            stack = np.take(array, range(0, array.shape[axis], SAMPLE_EVERY_K), axis=axis)
            stack = np.moveaxis(stack, axis, 0)  # [N, H, W]
            images_2d = torch.from_numpy(np.ascontiguousarray(stack[:, None])).to(self._device)  # [N, 1, H, W]
            keep = images_2d.amax(dim=(1, 2, 3)) > EMPTY_SLICE_THRESHOLD
            images_2d = images_2d[keep]
            if images_2d.shape[0] < 2:
                planes[plane] = None
                continue
            mini, maxi = images_2d.min(), images_2d.max()
            images_2d = (images_2d - mini) / (maxi - mini + 1e-10)
            images_2d = images_2d.repeat(1, 3, 1, 1)[:, [2, 1, 0]]
            images_2d[:, 0] -= 0.406
            images_2d[:, 1] -= 0.456
            images_2d[:, 2] -= 0.485
            with torch.inference_mode():
                features = self.network().forward(images_2d)
                planes[plane] = features.mean(dim=(2, 3)).cpu().numpy().astype(np.float64)  # [N, 2048]
        return planes


class RealReferenceBank:
    """Caches per-modality per-plane dev real volume features (built once per run root)."""

    def __init__(self, dev_list_path, raw_root, features, out_dir):
        self._entries = json.loads(Path(dev_list_path).read_text())["training"]
        self._raw_root = Path(raw_root)
        self._features = features
        self._out_dir = Path(out_dir)

    def build(self):
        bank_path = self._out_dir / "real_reference_bank.pt"
        if bank_path.is_file():
            return torch.load(bank_path, weights_only=True)
        bank = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
        for entry in self._entries:
            modality = {"mri_t1_skull_stripped": "t1n", "mri_t2_skull_stripped": "t2w", "mri_flair_skull_stripped": "t2f", "mri_t1c_skull_stripped": "t1c"}[entry["modality"]]
            path = self._raw_root / entry["image"]
            planes = self._features.volume_features(path)
            for plane, matrix in planes.items():
                if matrix is not None:
                    bank[modality][plane].append(matrix.mean(axis=0))
        payload = {m: {p: np.stack(v[p]) for p in PLANES if len(v[p])} for m, v in bank.items()}
        self._out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, bank_path)
        return payload


class TrendFid:
    """Plane FID points per target modality against the real bank."""

    def __init__(self, bank):
        self._bank = bank
        self._calculator = FidScoreCalculator()

    def score(self, generated_features):
        """generated_features: {modality: {plane: [N,2048]}} -> per-modality plane FIDs + mean."""
        report = {}
        for modality in TARGET_MODALITIES:
            planes = {}
            for plane in PLANES:
                generated = generated_features.get(modality, {}).get(plane)
                real = self._bank[modality].get(plane)
                if not generated or real is None:
                    planes[plane] = None
                    continue
                planes[plane] = self._calculator.score(real, np.stack(generated))
            values = [value for value in planes.values() if value is not None]
            planes["mean"] = float(np.mean(values)) if values else None
            report[modality] = planes
        valid = [report[m]["mean"] for m in TARGET_MODALITIES if report[m]["mean"] is not None]
        return report, (float(np.mean(valid)) if valid else None)


class L2TrendRunner:
    """Frozen-instrument measurements on the generated pseudo-four-modality cohort."""

    NN_CHANNELS = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
    REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
    NN_SIZE = (240, 240, 155)

    def __init__(self, instrument_results, instrument_entry, nnunet_raw, nnunet_preprocessed):
        self._results = instrument_results
        self._entry = instrument_entry
        self._nnunet_raw = nnunet_raw
        self._nnunet_preprocessed = nnunet_preprocessed

    def prep_inputs(self, samples, out_dir):
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        out = Path(out_dir)
        for sample in samples:
            dst = out / sample["sub"] / f"{sample['case']}_{self.NN_CHANNELS[sample['modality']]}.nii.gz"
            if dst.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            image = sitk.ReadImage(sample["path"])
            spacing = image.GetSpacing()
            size = image.GetSize()
            new_size = [int(round(s * sp)) for s, sp in zip(size, spacing)]
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing((1.0, 1.0, 1.0))
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetDefaultPixelValue(0.0)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled = resampler.Execute(image)
            arr = sitk.GetArrayFromImage(resampled)
            cropped = np.zeros(self.NN_SIZE[::-1], dtype=arr.dtype)
            limits = tuple(min(s, t) for s, t in zip(arr.shape, self.NN_SIZE[::-1]))
            cropped[: limits[0], : limits[1], : limits[2]] = arr[: limits[0], : limits[1], : limits[2]]
            out_img = sitk.GetImageFromArray(cropped)
            out_img.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(out_img, str(dst))
        return out

    def predict(self, challenge, input_dir, output_dir, log_path):
        spec = INSTRUMENT_SSA if challenge == "SSA" else INSTRUMENT_DEFAULT
        command = [
            sys.executable,
            str(self._entry),
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", INSTRUMENT_DATASETS[challenge],
            "-c", spec["config"],
            "-p", spec["plans"],
            "-tr", "nnUNetTrainer250Epochs",
            "-f", "0",
        ]
        env = {
            **os.environ,
            "nnUNet_compile": "f",
            "nnUNet_raw": str(self._nnunet_raw),
            "nnUNet_preprocessed": str(self._nnunet_preprocessed),
            "nnUNet_results": str(self._results[challenge]),
        }
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode

    def measure(self, cohort, input_dir, output_dir):
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        rows = []
        for item in cohort:
            case = item["case"]
            pred = Path(output_dir) / item["sub"] / f"{case}.nii.gz"
            row = {"sub": item["sub"], "case": case, "run_fail": not pred.is_file()}
            if row["run_fail"]:
                rows.append(row)
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred)))
            labels = set(int(v) for v in np.unique(arr)) - {0}
            row["hier_viol"] = not labels <= {1, 2, 3}
            volumes = {}
            voxel_ml = 1e-3  # 1 mm iso
            for region, members in self.REGIONS.items():
                volumes[region] = float(np.isin(arr, members).sum() * voxel_ml)
            row["volumes_ml"] = volumes
            row["empty_pred"] = volumes["WT"] == 0.0
            rows.append(row)
        return rows

    def run(self, samples, cohort, work_dir):
        inputs = self.prep_inputs(samples, Path(work_dir) / "nnunet_inputs")
        predictions = Path(work_dir) / "nnunet_predictions"
        rows = []
        for challenge in sorted({item["sub"] for item in cohort}):
            case_ids = {item["case"] for item in cohort if item["sub"] == challenge}
            files = sorted(p for p in (inputs / challenge).glob("*_0000.nii.gz") if p.name.rsplit("_", 1)[0] in case_ids)
            if not files:
                continue
            rc = self.predict(challenge, inputs / challenge, predictions / challenge, Path(work_dir) / f"predict_{challenge}.log")
            if rc != 0:
                rows += [{"sub": challenge, "case": p.name.rsplit("_", 1)[0], "run_fail": True} for p in files]
                continue
            rows += self.measure([item for item in cohort if item["sub"] == challenge], inputs / challenge, predictions / challenge)
        summary = {
            "per_case": rows,
            "n_run_fail": sum(1 for row in rows if row.get("run_fail")),
            "n_hier_viol": sum(1 for row in rows if row.get("hier_viol")),
            "n_empty_pred": sum(1 for row in rows if row.get("empty_pred")),
            "median_volumes_ml": {
                region: float(np.median([row["volumes_ml"][region] for row in rows if "volumes_ml" in row])) for region in self.REGIONS
            },
        }
        return summary


class EarlyStopRule:
    """Pre-recorded rule: patience on the mean dev FID trend (never before min_epoch)."""

    RULE_TEXT = (
        "metric m(N) = mean over t1n/t1c/t2w/t2f of plane-mean dev 2.5D RadImageNet FID on the "
        "fixed 16-case dev cohort (fixed seeds, cfg=10, 30 steps); stop when N >= {min_epoch} and "
        "the last {patience} consecutive evals set no new best m; hard cap = trainer n_epochs"
    )

    def __init__(self, patience, min_epoch, max_epoch):
        self.patience = patience
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch

    def rule_text(self):
        return self.RULE_TEXT.format(min_epoch=self.min_epoch, patience=self.patience)

    def should_stop(self, trend):
        """trend: list of {epoch, m} in epoch order; returns (stop, reason)."""
        points = [point for point in trend if point["m"] is not None]
        if not points:
            return False, "no eval points yet"
        last_epoch = points[-1]["epoch"]
        if last_epoch < self.min_epoch:
            return False, f"before min_epoch {self.min_epoch}"
        best_index = min(range(len(points)), key=lambda i: (points[i]["m"], i))
        best = points[best_index]["m"]
        since_best = len(points) - 1 - best_index
        if since_best >= self.patience:
            return True, f"no new best for {since_best} evals (best {best:.4f})"
        return False, f"best {best:.4f}, {since_best} evals since"

    @staticmethod
    def selection(trend):
        points = [point for point in trend if point["m"] is not None]
        if not points:
            return None
        best = min(points, key=lambda point: (point["m"], point["epoch"]))
        return {"epoch": best["epoch"], "mean_fid": best["m"], "checkpoint": best.get("checkpoint")}


class TrendLedger:
    """Appends eval records to dev_trend.jsonl and keeps the cohort + rule on disk."""

    def __init__(self, eval_root):
        self._root = Path(eval_root)

    def path(self):
        return self._root / "dev_trend.jsonl"

    def read(self):
        path = self.path()
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def append(self, record):
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self.path(), "a") as handle:
            handle.write(json.dumps(record) + "\n")


class DevEvalSelfTest:
    """Fixture check of cohort/rule/selection logic (numpy/stdlib only, no GPU)."""

    def __init__(self, workdir):
        self._workdir = Path(workdir)
        self.failures = []

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        dev_list = self._workdir / "p1_dev.json"
        entries = []
        for challenge, quota in COHORT_QUOTAS.items():
            for index in range(quota + 2):
                entries.append({"sub": challenge, "case": f"FIX{challenge}-{index:04d}-000", "modality": "mri_t1_skull_stripped"})
        dev_list.write_text(json.dumps({"training": entries}))
        cohort = DevCohortBuilder(dev_list).build()
        if len(cohort) != sum(COHORT_QUOTAS.values()):
            self.failures.append(f"cohort size {len(cohort)} != {sum(COHORT_QUOTAS.values())}")
        if DevCohortBuilder(dev_list).build() != cohort:
            self.failures.append("cohort not deterministic")
        if {item["sub"] for item in cohort} != set(COHORT_QUOTAS):
            self.failures.append("cohort missing a challenge")

        rule = EarlyStopRule(patience=3, min_epoch=30, max_epoch=100)
        improving = [{"epoch": e, "m": 1.0 - 0.01 * e} for e in (5, 10, 15, 20, 25, 30)]
        stop, _ = rule.should_stop(improving)
        if stop:
            self.failures.append("rule stopped an improving trend")
        plateau = improving + [{"epoch": e, "m": 0.75} for e in (35, 40, 45)]
        stop, reason = rule.should_stop(plateau)
        if not stop:
            self.failures.append(f"rule failed to stop a {3}-eval plateau ({reason})")
        short = improving + [{"epoch": 35, "m": 0.75}, {"epoch": 40, "m": 0.75}]
        stop, _ = rule.should_stop(short)
        if stop:
            self.failures.append("rule stopped before patience exhausted")
        selection = EarlyStopRule.selection(plateau)
        if selection["epoch"] != 30 or abs(selection["mean_fid"] - 0.70) > 1e-9:
            self.failures.append(f"selection picked {selection}, expected epoch 30 / m 0.70")

        trend = [{"epoch": 5, "m": 1.2, "checkpoint": "epoch_5.pt"}, {"epoch": 10, "m": 0.8, "checkpoint": "epoch_10.pt"}]
        ledger = TrendLedger(self._workdir)
        for record in trend:
            ledger.append(record)
        if ledger.read() != trend:
            self.failures.append("ledger roundtrip mismatch")
        return self.failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reference", help="build the dev real-feature bank once")
    p.add_argument("--dev-list", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--eval-root", required=True)

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
    p.add_argument("--instrument-entry", default="scripts/l2_calibration_predict_entry.py")
    p.add_argument("--nnunet-raw", default="/root/private_data/brats2023_nnunet")
    p.add_argument("--nnunet-preprocessed", default="/root/private_data/nnUNet_preprocessed")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="0 = run until stopped")

    p = sub.add_parser("select", help="emit the final dev-side selection for the contract")
    p.add_argument("--eval-root", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("selftest")
    p.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        failures = DevEvalSelfTest(args.workdir).run()
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0

    eval_root = Path(args.eval_root)
    ledger = TrendLedger(eval_root)

    if args.command == "reference":
        features = MrTrendFeatures(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
        print(f"real reference bank -> {eval_root / 'reference' / 'real_reference_bank.pt'}")
        return 0

    if args.command == "select":
        trend = ledger.read()
        selection = EarlyStopRule.selection(trend)
        if selection is None:
            print("no eval points; nothing to select", file=sys.stderr)
            return 1
        selection["rule"] = "argmin mean dev FID over eval points (pre-recorded)"
        selection["trend"] = trend
        selection["recorded_utc"] = datetime.now(UTC).isoformat()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(selection, indent=2) + "\n")
        print(f"selection -> {out} (epoch {selection['epoch']}, mean_fid {selection['mean_fid']:.4f})")
        return 0

    # watch mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_path = eval_root / "dev_cohort.json"
    cohort = DevCohortBuilder(args.dev_list).write(cohort_path) if not cohort_path.is_file() else json.loads(cohort_path.read_text())["cohort"]
    spacings = CohortSpacingSource(args.dev_list, args.emb_root)
    rule = EarlyStopRule(args.patience, args.min_epoch, args.max_epoch)
    (eval_root / "early_stop_rule.json").write_text(
        json.dumps({"rule": rule.rule_text(), "patience": args.patience, "min_epoch": args.min_epoch, "max_epoch": args.max_epoch}, indent=2) + "\n"
    )
    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.diffusion_unet_inference = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else {"num_inference_steps": 30}
    merged.cfg_guidance_scale = 10.0
    features = MrTrendFeatures(device)
    bank = RealReferenceBank(args.dev_list, args.raw_root, features, eval_root / "reference").build()
    fid = TrendFid(bank)
    sampler = CandidateSampler(merged, device, None)
    instrument_results = dict(item.split("=", 1) for item in args.instrument_results)
    l2 = L2TrendRunner(instrument_results, args.instrument_entry, args.nnunet_raw, args.nnunet_preprocessed)
    watcher = CheckpointWatcher(
        args.ckpt_dir, args.eval_every, args.max_epoch, {r["epoch"] for r in ledger.read()}
    )
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
                samples = sampler.generate_cohort(path, cohort, spacings, epoch_dir / "samples")
                plane_cache = {sample["path"]: features.volume_features(sample["path"]) for sample in samples}
                generated = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
                for sample in samples:
                    for plane in PLANES:
                        matrix = plane_cache[sample["path"]][plane]
                        if matrix is not None:
                            generated[sample["modality"]][plane].append(matrix.mean(axis=0))
                report, mean_fid = fid.score(generated)
            except Exception as error:
                # A broken checkpoint, a transient network/model failure, or any
                # single-epoch hiccup must not kill the sidecar: without it
                # nobody writes .early_stop. Skip and retry on the next poll.
                print(f"[eval] epoch {epoch} skipped: {error}", file=sys.stderr, flush=True)
                continue
            l2_trend = None
            if not args.skip_l2:
                try:
                    l2_trend = l2.run(samples, cohort, epoch_dir)
                except Exception as error:
                    print(f"[eval] epoch {epoch} l2 skipped: {error}", file=sys.stderr, flush=True)
            record = {
                "eval_utc": datetime.now(UTC).isoformat(),
                "epoch": epoch,
                "checkpoint": str(path),
                "fid": report,
                "m": mean_fid,
                "l2_trend": l2_trend,
                "cohort_file": str(cohort_path),
            }
            ledger.append(record)
            (epoch_dir / "trend.json").write_text(json.dumps(record, indent=2) + "\n")
            watcher.mark_done(epoch)
            stop, reason = rule.should_stop(ledger.read())
            print(f"[eval] epoch {epoch}: mean_fid={mean_fid} stop={stop} ({reason})", flush=True)
            if stop:
                (Path(args.ckpt_dir) / STOP_FILE).write_text(json.dumps({"reason": reason, "epoch": epoch}) + "\n")
                print(f"early-stop fired ({reason}); wrote {Path(args.ckpt_dir) / STOP_FILE}", flush=True)
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
