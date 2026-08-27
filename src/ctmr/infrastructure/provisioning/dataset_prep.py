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

"""Prepare BraTS2023 nnU-Net datasets for the self-trained L2 measurement instrument (issue #34; migrated from scripts/brats2023_nnunet_prep, ticket #140).

Implements the pinned decisions of issues #13/#32:
- case-level 70/10/20 split, ``split_id=brats2023-rflow-v1``, SHA-256 stable sort;
- training-side cases only (70%) become nnU-Net ``Dataset{ID}_{name}`` inputs, raw NIfTI via symlink;
- channel order t1n/t1c/t2w/t2f -> ``_0000.._0003`` (pinned, see #32 frozen-input contract);
- fold_0 80/20 case-level inside the training side; fold val cases are half of the calibration set.

Quotas below cover all five sub-challenges. METS is based on the available 238-case release
(five-institution public training set); its separately hosted NYU portion (164 cases) is
unavailable and is not a completeness gap (issue #42 re-pin, largest-remainder rounding).
The script ``__main__`` glue and argparse block do not travel: a later CLI slice takes over.
"""

import hashlib
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

SPLIT_ID = "brats2023-rflow-v1"

# Pinned 70/10/20 quotas per sub-challenge (issue #13; METS 238-case scope re-pinned in issue #42).
QUOTAS = {
    "GLI": {"train": 876, "dev": 125, "holdout": 250},
    "SSA": {"train": 42, "dev": 6, "holdout": 12},
    "MEN": {"train": 700, "dev": 100, "holdout": 200},
    "METS": {"train": 166, "dev": 24, "holdout": 48},
    "PED": {"train": 68, "dev": 10, "holdout": 20},
}

# nnU-Net dataset ids, one instrument per sub-challenge (issue #32 section 1).
DATASET_IDS = {"GLI": 501, "SSA": 502, "MEN": 503, "METS": 504, "PED": 505}

# Modality suffix -> nnU-Net channel index. Pinned order t1n/t1c/t2w/t2f (issue #32 section 4).
CHANNELS = [("t1n", "0000"), ("t1c", "0001"), ("t2w", "0002"), ("t2f", "0003")]

# Label 1/2/3 semantics per sub-challenge (BraTS2023 census: docs/research/brats2023-data-census.md).
LABELS = {
    "GLI": {"background": 0, "NCR": 1, "ED": 2, "ET": 3},
    "SSA": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3},
    "MEN": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3},
    "METS": {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3},
    "PED": {"background": 0, "NC": 1, "ED": 2, "ET": 3},
}

# Full-release directory naming: METS ships as "...-MET-..." while the pinned challenge code is METS.
DIR_SUFFIX = {"GLI": "GLI", "SSA": "SSA", "MEN": "MEN", "METS": "MET", "PED": "PED"}

# Cases excluded from the split input (challenge evaluation counts 98 PED training cases, not 99;
# result paper Kazerooni et al. 2024, per-case exclusion list unpublished).
EXCLUSIONS = {
    "PED": {
        "BraTS-PED-00024-000": (
            "challenge evaluation cohort is 98 training cases (results paper); this case shows anomalous "
            "intensity across all four modalities (max 6.1k-12.3k vs <1.6k normal) and is flagged by TCIA "
            "as fully skull-stripped; ruled the excluded case by user decision 2026-08-19"
        )
    },
}


class BraTSSplitPlanner:
    """Plans the pinned case-level 70/10/20 split over BraTS2023 training sources (issue #13 algorithm)."""

    def __init__(self, split_id, quotas, exclusions):
        self.split_id = split_id
        self.quotas = quotas
        self.exclusions = exclusions

    def sort_key(self, challenge, case_id, salt=""):
        """Deterministic SHA-256 stable-sort key; ``salt`` distinguishes the 70/10/20 split from fold_0."""
        return hashlib.sha256("|".join([self.split_id, salt, challenge, case_id]).encode()).hexdigest()

    def find_source_dir(self, brats_root, challenge):
        """Locates the per-challenge case directory under either layout: full release or sample set."""
        root = Path(brats_root)
        full = root / f"ASNR-MICCAI-BraTS2023-{DIR_SUFFIX[challenge]}-Challenge-TrainingData"
        sample = root / challenge
        for cand in (full, sample):
            if cand.is_dir():
                return cand
        return None

    def list_cases(self, source_dir):
        """Case ids present in a source directory (dirs named like the case, files {case}-{modality}.nii.gz)."""
        cases = set()
        for p in Path(source_dir).iterdir():
            if not p.is_dir():
                continue
            for f in p.glob("*-seg.nii.gz"):
                cases.add(f.name[: -len("-seg.nii.gz")])
        return sorted(cases)

    def build_manifest(self, brats_root):
        """Builds the split manifest; returns (manifest, failures) where failures is a list of str."""
        manifest = {
            "split_id": self.split_id,
            "algorithm": {
                "sort_key": "sha256('<split_id>|<salt>|<challenge>|<case_id>') hex ascending",
                "allocation": "ascending sort; first n_train -> train, next n_dev -> dev, rest -> holdout",
                "fold0": "salt='fold0'; first round(0.8*n_train_side) cases -> fold train, rest -> fold val",
            },
            "quotas": self.quotas,
            "dataset_ids": DATASET_IDS,
            "exclusions": {k: self.exclusions[k] for k in self.quotas if k in self.exclusions},
            "challenges": {},
        }
        failures = []
        for ch in sorted(self.quotas):
            src = self.find_source_dir(brats_root, ch)
            if src is None:
                failures.append(f"{ch}: no source directory under {brats_root}")
                continue
            cases = self.list_cases(src)
            excluded = sorted(set(cases) & set(self.exclusions.get(ch, {})))
            eligible = [c for c in cases if c not in self.exclusions.get(ch, {})]
            ordered = sorted(eligible, key=lambda c: self.sort_key(ch, c))
            q = self.quotas[ch]
            total = q["train"] + q["dev"] + q["holdout"]
            if len(ordered) != total:
                failures.append(f"{ch}: {len(ordered)} eligible cases ({len(excluded)} excluded) != pinned quota total {total}")
                continue
            manifest["challenges"][ch] = {
                "source_dir": str(src.resolve()),
                "excluded": excluded,
                "cases": {
                    "train": ordered[: q["train"]],
                    "dev": ordered[q["train"] : q["train"] + q["dev"]],
                    "holdout": ordered[q["train"] + q["dev"] :],
                },
            }
        return manifest, failures


class NNUNetDatasetBuilder:
    """Builds nnU-Net Dataset{ID}_* dirs (symlinks to raw NIfTI) plus dataset.json and fold_0 splits."""

    def __init__(self, nnunet_root, manifest, copy=False):
        self.nnunet_root = Path(nnunet_root)
        self.manifest = manifest
        self.copy = copy

    def dataset_dir_name(self, challenge):
        return f"Dataset{DATASET_IDS[challenge]:03d}_BraTS2023{challenge}"

    def link_or_copy(self, src, dst):
        if self.copy:
            shutil.copy2(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

    def build_datasets(self):
        """Materialises imagesTr/labelsTr symlinks + dataset.json for every challenge in the manifest."""
        for ch, info in self.manifest["challenges"].items():
            src = Path(info["source_dir"])
            ds_dir = self.nnunet_root / self.dataset_dir_name(ch)
            images, labels = ds_dir / "imagesTr", ds_dir / "labelsTr"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            for case in info["cases"]["train"]:
                for modality, channel in CHANNELS:
                    self.link_or_copy(src / case / f"{case}-{modality}.nii.gz", images / f"{case}_{channel}.nii.gz")
                self.link_or_copy(src / case / f"{case}-seg.nii.gz", labels / f"{case}.nii.gz")
            dataset_json = {
                "channel_names": {channel: modality for modality, channel in CHANNELS},
                "labels": LABELS[ch],
                "numTraining": len(info["cases"]["train"]),
                "file_ending": ".nii.gz",
                "name": f"BraTS2023{ch}",
                "description": f"BraTS2023 {ch} training-side cases (split_id={self.manifest['split_id']}, "
                "70% portion); raw NIfTI, no pre-normalization.",
                "licence": "CC-BY-NC 4.0 (Synapse DUA). Derivative weights/calibration artifacts are never redistributed.",
                "reference": "https://www.synapse.org/Synapse:syn51156910",
            }
            (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
            print(f"{ch}: {len(info['cases']['train'])} cases -> {ds_dir}")

    def write_fold0_splits(self):
        """Writes splits_final.json (fold_0 80/20) per dataset and the calibration val case lists."""
        planner = BraTSSplitPlanner(self.manifest["split_id"], self.manifest["quotas"], {})
        val_dir = self.nnunet_root / "splits"
        val_dir.mkdir(parents=True, exist_ok=True)
        for ch, info in self.manifest["challenges"].items():
            train_side = sorted(info["cases"]["train"], key=lambda c: planner.sort_key(ch, c, salt="fold0"))
            n_fold_train = round(0.8 * len(train_side))
            fold_train, fold_val = train_side[:n_fold_train], train_side[n_fold_train:]
            ds_dir = self.nnunet_root / self.dataset_dir_name(ch)
            (ds_dir / "splits_final.json").write_text(json.dumps([{"train": fold_train, "val": fold_val}], indent=2) + "\n")
            # Fold val cases are half of the calibration set (issue #32 section 3); export the controlled list.
            (val_dir / f"fold0_val_cases_{ch}.txt").write_text("\n".join(fold_val) + "\n")
            print(f"{ch}: fold_0 train={len(fold_train)} val={len(fold_val)}")


class NNUNetDatasetVerifier:
    """Reconciles manifest counts against built datasets and spot-checks channel order, labels, affines."""

    def __init__(self, nnunet_root, manifest, copy=False, sample_n=3):
        self.nnunet_root = Path(nnunet_root)
        self.manifest = manifest
        self.copy = copy
        self.sample_n = sample_n
        self.failures = []

    def check(self, cond, msg):
        if not cond:
            self.failures.append(msg)

    def dataset_dir_name(self, challenge):
        return f"Dataset{DATASET_IDS[challenge]:03d}_BraTS2023{challenge}"

    def verify_reconciliation(self, ch, info):
        """Count-level reconciliation: manifest vs imagesTr/labelsTr vs dataset.json vs splits_final."""
        ds_dir = self.nnunet_root / self.dataset_dir_name(ch)
        train_cases = info["cases"]["train"]
        meta = json.loads((ds_dir / "dataset.json").read_text())
        self.check(meta["numTraining"] == len(train_cases), f"{ch}: dataset.json numTraining mismatch")
        self.check(
            [meta["channel_names"][c] for c in ("0000", "0001", "0002", "0003")] == ["t1n", "t1c", "t2w", "t2f"],
            f"{ch}: channel order is not t1n/t1c/t2w/t2f",
        )
        self.check(meta["labels"] == LABELS[ch], f"{ch}: label semantics mismatch")
        images = sorted(p.name for p in (ds_dir / "imagesTr").iterdir())
        expected_images = sorted(f"{c}_{k}.nii.gz" for c in train_cases for _, k in CHANNELS)
        self.check(images == expected_images, f"{ch}: imagesTr file set mismatch ({len(images)} files)")
        labels = sorted(p.name for p in (ds_dir / "labelsTr").iterdir())
        self.check(labels == sorted(f"{c}.nii.gz" for c in train_cases), f"{ch}: labelsTr file set mismatch")
        splits = json.loads((ds_dir / "splits_final.json").read_text())
        self.check(sorted(splits[0]["train"] + splits[0]["val"]) == sorted(train_cases), f"{ch}: fold_0 cases != training side")
        self.check(not set(splits[0]["train"]) & set(splits[0]["val"]), f"{ch}: fold_0 train/val overlap")
        return splits

    def verify_case(self, ch, info, case):
        """Spot-check one case: symlink target, seg label domain, affine agreement across channels+seg."""
        src = Path(info["source_dir"])
        ds_dir = self.nnunet_root / self.dataset_dir_name(ch)
        affines = []
        for modality, channel in CHANNELS:
            src_file = src / case / f"{case}-{modality}.nii.gz"
            dst = ds_dir / "imagesTr" / f"{case}_{channel}.nii.gz"
            self.check(dst.exists(), f"{ch} {dst.name} missing")
            if not dst.exists():
                continue
            if not self.copy:
                self.check(dst.resolve() == src_file.resolve(), f"{ch} {dst.name}: link does not point to source {src_file}")
            affines.append(nib.load(str(dst)).affine.tolist())
        seg_path = ds_dir / "labelsTr" / f"{case}.nii.gz"
        self.check(seg_path.exists(), f"{ch} {seg_path.name} missing")
        if seg_path.exists():
            seg_img = nib.load(str(seg_path))
            affines.append(seg_img.affine.tolist())
            seg_values = set(np.unique(np.asanyarray(seg_img.dataobj)).tolist())
            self.check(seg_values <= {0, 1, 2, 3}, f"{ch} {case}: seg labels {sorted(seg_values)} not within 0/1/2/3")
        self.check(all(a == affines[0] for a in affines), f"{ch} {case}: affine mismatch across channels+seg")

    def verify(self):
        """Runs the full reconciliation + spot-check pass; returns the failure list."""
        planner = BraTSSplitPlanner(self.manifest["split_id"], self.manifest["quotas"], {})
        for ch, info in self.manifest["challenges"].items():
            splits = self.verify_reconciliation(ch, info)
            ordered = sorted(info["cases"]["train"], key=lambda c: planner.sort_key(ch, c, salt="fold0"))
            for case in ordered[: self.sample_n]:
                self.verify_case(ch, info, case)
            print(
                f"{ch}: cases={len(info['cases']['train'])} fold0_train={len(splits[0]['train'])} "
                f"fold0_val={len(splits[0]['val'])} spot-checked={min(self.sample_n, len(ordered))}"
            )
        return self.failures
