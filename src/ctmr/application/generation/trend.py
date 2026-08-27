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

"""Dev-sidecar trend machinery shared by the label-conditioned families (ticket 10).

The FID-trend and frozen-instrument L2-trend pieces both the modality-label
and mask dev light-acceptance sidecars build on, moved verbatim out of the
retiring P1 dev-eval script entry (ADR-0015 §2): the fixed 16-case dev cohort
builder, the RadImageNet plane-feature extractor with the pinned MR trend
preprocessing, the cached real reference bank, the per-modality plane FID, and
the instrument trend runner. The watch/select polling skeleton itself stays in
``ctmr.application.shell``; family-specific samplers stay in the family
modules. The quantitative chain relocates these to its own package with its
migration ticket.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ctmr.application.acceptance.quantitative.fid import FidScoreCalculator
from ctmr.application.shell import COHORT_QUOTAS, TARGET_MODALITIES
from ctmr.domain.grid import TREND_FEATURE_GRID, CenterCropOrPad, GridResampler, InstrumentGridAdapter
from ctmr.domain.instrument_spec import INSTRUMENT_SPECS, FrozenInstrumentCommand
from ctmr.domain.measurement import LABEL_DOMAIN, REGIONS

PLANES = ("xy", "yz", "zx")
TREND_PREPROCESSING = "percentile_0_99.5_to_0_1_ras_1mm_zero_pad_240x240x160"
EMPTY_SLICE_THRESHOLD = 0.05
SAMPLE_EVERY_K = 2


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


class MrTrendFeatures:
    """Percentile->RAS->1mm->zero-pad preprocessing + per-plane RadImageNet slice features."""

    def __init__(self, device):
        self._device = device
        self._network = None

    def network(self):
        if self._network is None:
            self._network = (
                torch.hub.load(
                    # Explicit ":main" ref: torch.hub probes github.com for the default
                    # branch when no ref is given; on the sugon the probe dies with
                    # RemoteDisconnected (not URLError) before the cache fallback.
                    "Warvito/radimagenet-models:main",
                    model="radimagenet_resnet50",
                    trust_repo=True,
                    verbose=False,
                )
                .to(self._device)
                .eval()
            )
        return self._network

    @staticmethod
    def preprocess(path):
        import SimpleITK as sitk  # deferred: execution-side only (sugon system env)

        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        values = data[data > 0] if (data > 0).any() else data.ravel()
        lo, hi = np.percentile(values, [0.0, 99.5])
        data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        # ADR-0008 decision 4: the 1 mm resample + centring go through the generic
        # engine (linear, grid 240x240x160 in xyz); the percentile normalisation
        # stays in the feature extractor -- it is intensity, not geometry.
        zoom = [float(v) for v in image.header.get_zooms()[:3]]  # nib zooms are np.float32; sitk needs float
        sitk_image = sitk.GetImageFromArray(np.ascontiguousarray(data.transpose(2, 1, 0)).astype(np.float32))  # xyz -> zyx
        sitk_image.SetSpacing(zoom)
        aligned = CenterCropOrPad().crop_or_pad(GridResampler(sitk.sitkLinear).resample(sitk_image, TREND_FEATURE_GRID), TREND_FEATURE_GRID)
        return sitk.GetArrayFromImage(aligned).transpose(2, 1, 0).astype(np.float32)  # zyx -> xyz (nibabel order)

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
            # The run-local reference bank payload holds numpy arrays; weights_only
            # rejects them by default, so allowlist numpy reconstruction at the load
            # point (bank is a local artifact; never an import-time global mutation).
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float64DType])
            return torch.load(bank_path, weights_only=True)
        bank = {modality: {plane: [] for plane in PLANES} for modality in TARGET_MODALITIES}
        for entry in self._entries:
            modality = {
                "mri_t1_skull_stripped": "t1n",
                "mri_t2_skull_stripped": "t2w",
                "mri_flair_skull_stripped": "t2f",
                "mri_t1c_skull_stripped": "t1c",
            }[entry["modality"]]
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
    """Frozen-instrument measurements on the generated pseudo-four-modality cohort.

    The instrument invocation is the ADR-0009 single construction point: argv is
    exactly ``FrozenInstrumentCommand.build`` (canonical entry
    ``python -m ctmr.instrument.predict``, frozen config inside the spec).
    """

    NN_CHANNELS = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}

    def __init__(self, instrument_results, nnunet_raw, nnunet_preprocessed):
        self._results = instrument_results
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
            # ADR-0008 adoption: the #38 InputPreparator geometry via the frozen
            # instrument adapter (B-spline + centred crop/pad) -- registered linear->B-spline
            # + centreing changes vs the pre-adoption top-left linear resize.
            aligned = InstrumentGridAdapter.continuum().align(image)
            sitk.WriteImage(aligned, str(dst))
        return out

    def predict(self, challenge, input_dir, output_dir, log_path):
        command = FrozenInstrumentCommand(INSTRUMENT_SPECS[challenge]).build(input_dir, output_dir)
        env = {
            **os.environ,
            # the canonical entry runs in a fresh child process: the installed
            # package's src tree goes onto the child PYTHONPATH (sys.path entries
            # are process-local; the ADR-0009 decision 6 shim).
            "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[3])] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
            ),
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
            pred = Path(output_dir) / f"{case}.nii.gz"
            row = {"sub": item["sub"], "case": case, "run_fail": not pred.is_file()}
            if row["run_fail"]:
                rows.append(row)
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred)))
            labels = set(int(v) for v in np.unique(arr)) - {0}
            row["hier_viol"] = not labels <= set(LABEL_DOMAIN)
            volumes = {}
            voxel_ml = 1e-3  # 1 mm iso
            for region, members in REGIONS.items():  # the canonical projections (ctmr.domain.measurement, ADR-0010)
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
            "median_volumes_ml": {region: float(np.median([row["volumes_ml"][region] for row in rows if "volumes_ml" in row])) for region in REGIONS},
        }
        return summary
