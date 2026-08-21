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

"""Generative-side BraTS phase manifest & data pipeline (issue #52, spec #51).

Consumes the pinned case-level split (same planner as the L2 instrument data,
issue #34: split_id=brats2023-rflow-v1) and produces the controlled artifacts
P1/P2/P3 directly consume (spec #51 implementation decisions 3-5):

- ``phase_manifest.json``     — reuse of the issue #34 three-side manifest, or a
                                bit-identical rebuild from the raw sources;
- ``encode_source.json``      — VAE-encode input list for the upstream
                                ``diff_model_create_training_data.py`` (train+dev
                                sides only; holdout is never encoded before
                                candidate freeze);
- ``labels/<CH>/<case>/``     — P2 condition ``-combined.nii.gz`` (brain=22 union
                                over the four modalities, overlaid with the
                                1/2/3 -> 129/130/131 tumour remap) and P3
                                loss-only ``-tumor129.nii.gz``; nearest-neighbour
                                resize onto the pinned 256x256x128 grid;
- ``<emb>.nii.gz.json``       — per-embedding companion {spacing, modality} the
                                DM training loader reads (upstream gap #15-1);
- ``lists/``                  — p1_image_only(.json/_dev), p2_mask_cond,
                                p3_pairs data lists (P1/P2 four modality entries
                                per case; P3 twelve ordered src!=tgt pairs per
                                case, all pairs of a case on the same side).

Modality tokens are pinned in ``configs/modality_mapping.json``:
t1n=29 (mri_t1_skull_stripped), t2w=30 (mri_t2_skull_stripped),
t2f=31 (mri_flair_skull_stripped), t1c=34 (mri_t1c_skull_stripped).

Usage (each subcommand standalone, manifest first):
    python scripts/brats_phase_prep.py manifest --reuse splits.json --out phase_manifest.json
    python scripts/brats_phase_prep.py encode-list --manifest phase_manifest.json --out encode_source.json
    python scripts/brats_phase_prep.py labels --manifest phase_manifest.json --out-root DIR
    python scripts/brats_phase_prep.py companions --manifest phase_manifest.json --emb-root DIR
    python scripts/brats_phase_prep.py lists --manifest phase_manifest.json --phase-root DIR
    python scripts/brats_phase_prep.py verify --manifest phase_manifest.json --phase-root DIR \
        [--nnunet-manifest splits.json] [--manifest-v2 phase_manifest_v2.json]
    python scripts/brats_phase_prep.py selftest --workdir TMP
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Orientation, Resize

from .brats2023_nnunet_prep import (
    BraTSSplitPlanner,
    CHANNELS,
    DATASET_IDS,
    DIR_SUFFIX,
    EXCLUSIONS,
    QUOTAS,
    SPLIT_ID,
)

# BraTS file suffix -> (modality_mapping key, class-label token). Pinned by
# spec #51 decision 5; t1c=34 is the P1-planned addition (untrained row in the
# frozen v1 DM until P1 finishes — it is present in the lists regardless).
MODALITIES = {
    "t1n": ("mri_t1_skull_stripped", 29),
    "t2w": ("mri_t2_skull_stripped", 30),
    "t2f": ("mri_flair_skull_stripped", 31),
    "t1c": ("mri_t1c_skull_stripped", 34),
}

# P2 condition vocabulary (spec #51 decision 5): brain envelope 22 (union of
# the four modalities' >0 masks) overlaid with BraTS 1/2/3 -> 129/130/131
# (NCR/NETC, ED/SNFH, ET). 401/402/403 would alias in the 8-bit condition and
# are forbidden; no CT-body or brain-parcellation pseudo-labels.
BRAIN_LABEL = 22
TUMOUR_REMAP = {1: 129, 2: 130, 3: 131}

# Pinned generative grid (spec #51 decision 4): image 256x256x128, latent
# 64x64x32 — the nearest-multiple-of-128 target for 240x240x155 BraTS volumes.
GRID = (256, 256, 128)

SIDES = ("train", "dev")
MODALITY_KEYS = {key for _suffix, (key, _token) in MODALITIES.items()}

# Source-file corruption found during the first real run (issue #52, 2026-08-21):
# BraTS-MET-00232-000's t2w is corrupt in the authoritative gauss copy — the gzip
# data segment breaks at 59% (~10.5 of 17.9 MB recoverable, header intact), md5
# 09ade1982d19a9e7d52795527743d71d on every copy, no pristine copy on gauss/sugon.
# The pinned split sides are NOT changed (the manifest stays frozen); the case is
# excluded from generative artifacts until a pristine copy lands, then this
# pipeline is re-run for that case (new manifest versions per the split rules).
PIPELINE_EXCLUSIONS = {
    ("METS", "BraTS-MET-00232-000"): (
        "source t2w corrupt in the authoritative copy (gzip breaks at 59%); "
        "excluded from generative artifacts pending a pristine re-download"
    )
}


class RawLayout:
    """Paths between the manifest's source dirs, per-case files and embeddings."""

    def __init__(self, manifest):
        self._manifest = manifest

    def source_dir(self, challenge):
        return Path(self._manifest["challenges"][challenge]["source_dir"])

    def raw_rel(self, challenge, case, suffix):
        """Raw image path relative to the raw-data root (what the mirror logic mirrors)."""
        source_dir = self.source_dir(challenge)
        return (Path(source_dir.parent.name) / source_dir.name / case / f"{case}-{suffix}.nii.gz").as_posix()

    def emb_rel(self, challenge, case, suffix):
        """Embedding path relative to the encoding run's embedding_base_dir."""
        return self.raw_rel(challenge, case, suffix).replace(".nii.gz", "_emb.nii.gz")

    def emb_phase_rel(self, challenge, case, suffix, prefix="embeddings"):
        """Embedding path relative to the phase root (P2/P3 list reference form)."""
        return f"{prefix}/{self.emb_rel(challenge, case, suffix)}"


class PhaseManifestProvider:
    """Loads, reuses or rebuilds the three-side phase manifest with the issue #34 planner."""

    def __init__(self, quotas):
        self._quotas = quotas

    def load(self, path, raw_root=None):
        """Loads a manifest, optionally remapping source_dir onto a local view of the raw data."""
        manifest = json.loads(Path(path).read_text())
        if manifest.get("split_id") != SPLIT_ID:
            raise ValueError(f"manifest split_id {manifest.get('split_id')!r} != pinned {SPLIT_ID!r}")
        for ch, quota in self._quotas.items():
            info = manifest["challenges"].get(ch)
            if info is None:
                raise ValueError(f"manifest is missing challenge {ch}")
            for side, count in quota.items():
                if len(info["cases"][side]) != count:
                    raise ValueError(f"{ch} {side}: {len(info['cases'][side])} cases != pinned quota {count}")
        if raw_root is not None:
            # A linked view (see LinkViewBuilder) reproduces the official
            # per-challenge layout under a single root; remap without losing
            # the original provenance.
            for ch, info in manifest["challenges"].items():
                source_dir = Path(info["source_dir"])
                info["source_dir_original"] = info["source_dir"]
                info["source_dir"] = str(Path(raw_root) / source_dir.name)
        return manifest

    def reuse(self, source_path, out_path, raw_root=None):
        """Copies an existing (already verified) manifest, recording its SHA-256."""
        source = Path(source_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = self.load(source, raw_root=raw_root)
        manifest["phase_reuse"] = {"source": str(source.resolve()), "sha256": digest}
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"reused {source} (sha256 {digest[:12]}) -> {out}")
        return manifest

    def rebuild(self, brats_root, out_path):
        """Recomputes the split from raw sources with the identical planner (bit-identical expectation)."""
        planner = BraTSSplitPlanner(SPLIT_ID, self._quotas, EXCLUSIONS)
        manifest, failures = planner.build_manifest(brats_root)
        if failures:
            raise ValueError("manifest rebuild failed:\n" + "\n".join(failures))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n")
        for ch, info in manifest["challenges"].items():
            c = info["cases"]
            print(f"{ch}: train={len(c['train'])} dev={len(c['dev'])} holdout={len(c['holdout'])}")
        print(f"rebuilt -> {out}")
        return manifest


class LinkViewBuilder:
    """Links the sugon-controlled copies into the official per-challenge layout the manifest expects.

    Train-side volumes live in the issue #34 nnU-Net datasets (``{case}_{channel}.nii.gz``
    under imagesTr/labelsTr); dev-side volumes live in the calibration ``dev_raw`` copy
    (official ``{case}/{case}-{modality}.nii.gz`` layout). The view mirrors both into
    ``<out-root>/ASNR-MICCAI-BraTS2023/{CH}-Challenge-TrainingData/{case}/`` symlinks so
    ``--raw-root <out-root>/ASNR-MICCAI-BraTS2023`` gives every consumer one layout.
    """

    def __init__(self, manifest, nnunet_root, dev_raw_root, out_root):
        self._manifest = manifest
        self._nnunet_root = Path(nnunet_root)
        self._dev_raw_root = Path(dev_raw_root)
        self._out_root = Path(out_root)

    def _link(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source.resolve())

    def build_all(self):
        n_train, n_dev = 0, 0
        for ch, info in sorted(self._manifest["challenges"].items()):
            challenge_dir_name = f"ASNR-MICCAI-BraTS2023-{DIR_SUFFIX[ch]}-Challenge-TrainingData"
            dataset = self._nnunet_root / f"Dataset{DATASET_IDS[ch]:03d}_BraTS2023{ch}"
            dev_source = self._dev_raw_root / challenge_dir_name
            for case in info["cases"]["train"]:
                case_view = self._out_root / "ASNR-MICCAI-BraTS2023" / challenge_dir_name / case
                for modality, channel in CHANNELS:
                    self._link(
                        dataset / "imagesTr" / f"{case}_{channel}.nii.gz",
                        case_view / f"{case}-{modality}.nii.gz",
                    )
                self._link(dataset / "labelsTr" / f"{case}.nii.gz", case_view / f"{case}-seg.nii.gz")
                n_train += 1
            for case in info["cases"]["dev"]:
                case_view = self._out_root / "ASNR-MICCAI-BraTS2023" / challenge_dir_name / case
                for modality, _channel in CHANNELS:
                    self._link(
                        dev_source / case / f"{case}-{modality}.nii.gz",
                        case_view / f"{case}-{modality}.nii.gz",
                    )
                self._link(dev_source / case / f"{case}-seg.nii.gz", case_view / f"{case}-seg.nii.gz")
                n_dev += 1
        print(f"link view: train={n_train} dev={n_dev} cases under {self._out_root}")
        return self._out_root


class EncodeListBuilder:
    """Emits the VAE-encode input list for the upstream create-training-data script (train+dev only)."""

    def __init__(self, manifest):
        self._manifest = manifest
        self._layout = RawLayout(manifest)

    def build(self):
        entries = []
        for ch, info in sorted(self._manifest["challenges"].items()):
            for side in SIDES:
                for case in info["cases"][side]:
                    if (ch, case) in PIPELINE_EXCLUSIONS:
                        continue
                    for suffix, (modality_key, _token) in MODALITIES.items():
                        entries.append(
                            {
                                "image": self._layout.raw_rel(ch, case, suffix),
                                "modality": modality_key,
                                "sub": ch,
                                "case": case,
                                "side": side,
                            }
                        )
        return entries

    def write(self, out_path):
        entries = self.build()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"training": entries}, indent=1) + "\n")
        print(f"encode source list: {len(entries)} entries -> {out}")
        return out


class LabelVolumeBuilder:
    """Builds the P2 combined condition and the P3 loss-only tumour label on the pinned grid."""

    def __init__(self, manifest, out_root):
        self._manifest = manifest
        self._layout = RawLayout(manifest)
        self._out_root = Path(out_root)
        self._loader = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Orientation(axcodes="RAS")])
        self._resize = Resize(spatial_size=GRID, mode="nearest")

    def build_case(self, challenge, case):
        """Writes <case>-combined.nii.gz and <case>-tumor129.nii.gz under labels/<CH>/<case>/."""
        case_dir = self._layout.source_dir(challenge) / case
        seg = self._loader(str(case_dir / f"{case}-seg.nii.gz")).short()
        brain_union = None
        for suffix in MODALITIES:
            image = self._loader(str(case_dir / f"{case}-{suffix}.nii.gz"))
            nonzero = (image > 0).short()
            brain_union = nonzero if brain_union is None else (brain_union | nonzero)
        combined = brain_union * BRAIN_LABEL
        tumour = seg.clone()
        for orig, mapped in TUMOUR_REMAP.items():
            combined[seg == orig] = mapped
            tumour[seg == orig] = mapped
        out_dir = self._out_root / "labels" / challenge / case
        out_dir.mkdir(parents=True, exist_ok=True)
        affine = seg.meta["affine"].numpy()
        for name, volume in (("-combined", combined), ("-tumor129", tumour)):
            arr = self._resize(volume).squeeze(0).short().cpu().numpy()
            nib.save(nib.Nifti1Image(arr, affine), str(out_dir / f"{case}{name}.nii.gz"))
        return out_dir

    def build_all(self, sides=SIDES):
        n_cases = 0
        for ch, info in sorted(self._manifest["challenges"].items()):
            for side in sides:
                for case in info["cases"][side]:
                    if (ch, case) in PIPELINE_EXCLUSIONS:
                        continue
                    self.build_case(ch, case)
                    n_cases += 1
        print(f"labels: {n_cases} cases -> {self._out_root / 'labels'}")
        return n_cases


class EmbCompanionWriter:
    """Writes the per-embedding {spacing, modality} companion the DM training loader reads."""

    def __init__(self, manifest, emb_root):
        self._manifest = manifest
        self._layout = RawLayout(manifest)
        self._emb_root = Path(emb_root)

    def write_all(self, sides=SIDES):
        written = 0
        for ch, info in sorted(self._manifest["challenges"].items()):
            for side in sides:
                for case in info["cases"][side]:
                    if (ch, case) in PIPELINE_EXCLUSIONS:
                        continue
                    for suffix, (modality_key, _token) in MODALITIES.items():
                        emb_path = self._emb_root / self._layout.emb_rel(ch, case, suffix)
                        if not emb_path.is_file():
                            raise FileNotFoundError(f"embedding missing (encode first): {emb_path}")
                        spacing = [float(v) for v in nib.load(str(emb_path)).header["pixdim"][1:4]]
                        companion = {"spacing": spacing, "modality": modality_key}
                        out_path = emb_path.with_name(emb_path.name + ".json")
                        out_path.write_text(json.dumps(companion))
                        written += 1
        print(f"companions: {written} written under {self._emb_root}")
        return written


class PhaseListBuilder:
    """Emits the P1/P2/P3 training data lists (P1/P2 four entries per case, P3 twelve ordered pairs)."""

    def __init__(self, manifest, phase_root):
        self._manifest = manifest
        self._layout = RawLayout(manifest)
        self._phase_root = Path(phase_root)

    def spacing_of(self, challenge, case):
        """Reads the actual post-resample spacing from the case's first companion."""
        rel = self._layout.emb_phase_rel(challenge, case, "t1n")
        payload = json.loads((self._phase_root / (rel + ".json")).read_text())
        return payload["spacing"]

    def p1_entries(self, challenge, case):
        # diff_model_train replaces .nii.gz with _emb.nii.gz itself and joins
        # embedding_base_dir, so P1 entries carry the raw relative path.
        return [
            {
                "image": self._layout.raw_rel(challenge, case, suffix),
                "modality": modality_key,
                "sub": challenge,
                "case": case,
            }
            for suffix, (modality_key, _token) in MODALITIES.items()
        ]

    def p2_entries(self, challenge, case, fold):
        spacing = self.spacing_of(challenge, case)
        return [
            {
                "image": self._layout.emb_phase_rel(challenge, case, suffix),
                "label": f"labels/{challenge}/{case}/{case}-combined.nii.gz",
                "spacing": spacing,
                "modality": modality_key,
                "fold": fold,
                "sub": challenge,
                "case": case,
            }
            for suffix, (modality_key, _token) in MODALITIES.items()
        ]

    def p3_entries(self, challenge, case, fold):
        spacing = self.spacing_of(challenge, case)
        entries = []
        for src_suffix, (src_key, _s) in MODALITIES.items():
            for tgt_suffix, (tgt_key, _t) in MODALITIES.items():
                if src_suffix == tgt_suffix:
                    continue
                entries.append(
                    {
                        "image": self._layout.emb_phase_rel(challenge, case, tgt_suffix),
                        "src_image": self._layout.emb_phase_rel(challenge, case, src_suffix),
                        "label": f"labels/{challenge}/{case}/{case}-tumor129.nii.gz",
                        "spacing": spacing,
                        "modality": tgt_key,
                        "src_modality": src_key,
                        "fold": fold,
                        "sub": challenge,
                        "case": case,
                    }
                )
        return entries

    def write(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p1_train, p1_dev, p2, p3 = [], [], [], []
        for ch, info in sorted(self._manifest["challenges"].items()):
            for case in info["cases"]["train"]:
                if (ch, case) in PIPELINE_EXCLUSIONS:
                    continue
                p1_train += self.p1_entries(ch, case)
                p2 += self.p2_entries(ch, case, fold=1)
                p3 += self.p3_entries(ch, case, fold=1)
            for case in info["cases"]["dev"]:
                if (ch, case) in PIPELINE_EXCLUSIONS:
                    continue
                p1_dev += self.p1_entries(ch, case)
                p2 += self.p2_entries(ch, case, fold=0)
                p3 += self.p3_entries(ch, case, fold=0)
        lists = {
            "p1_image_only.json": {"training": p1_train},
            "p1_image_only_dev.json": {"training": p1_dev},
            "p2_mask_cond.json": {"training": p2},
            "p3_pairs.json": {"training": p3},
        }
        for name, payload in lists.items():
            (out_dir / name).write_text(json.dumps(payload, indent=1) + "\n")
        print(
            f"lists: p1_train={len(p1_train)} p1_dev={len(p1_dev)} p2={len(p2)} p3={len(p3)} -> {out_dir}"
        )
        return out_dir


class PhaseVerifier:
    """Reconciles the phase artifacts against the pinned split, data and vocabulary contracts."""

    def __init__(self, manifest, phase_root, quotas=None):
        self._manifest = manifest
        self._phase_root = Path(phase_root)
        self._layout = RawLayout(manifest)
        self._quotas = QUOTAS if quotas is None else quotas
        self.failures = []

    def check(self, cond, msg):
        if not cond:
            self.failures.append(msg)

    def side_of_cases(self):
        mapping = {}
        for ch, info in self._manifest["challenges"].items():
            for side in ("train", "dev", "holdout"):
                for case in info["cases"][side]:
                    mapping[(ch, case)] = side
        return mapping

    def verify_manifest(self):
        for ch, quota in self._quotas.items():
            info = self._manifest["challenges"][ch]
            sides = {side: set(info["cases"][side]) for side in ("train", "dev", "holdout")}
            for side, count in quota.items():
                self.check(len(sides[side]) == count, f"{ch} {side}: {len(sides[side])} cases != pinned quota {count}")
            for a, b in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
                self.check(not sides[a] & sides[b], f"{ch}: case leakage between {a} and {b}")

    def verify_companions_and_tokens(self):
        token_by_key = {key: token for _suffix, (key, token) in MODALITIES.items()}
        for ch, info in sorted(self._manifest["challenges"].items()):
            for side in SIDES:
                for case in info["cases"][side]:
                    if (ch, case) in PIPELINE_EXCLUSIONS:
                        continue
                    for suffix, (modality_key, token) in MODALITIES.items():
                        rel = self._layout.emb_rel(ch, case, suffix)
                        emb_path = self._phase_root / "embeddings" / rel
                        self.check(emb_path.is_file(), f"missing embedding {rel}")
                        companion_path = emb_path.with_name(emb_path.name + ".json")
                        if not companion_path.is_file():
                            self.failures.append(f"missing companion {rel}.json")
                            continue
                        payload = json.loads(companion_path.read_text())
                        if "spacing" not in payload or "modality" not in payload:
                            self.failures.append(f"{rel}: companion lacks spacing/modality")
                            continue
                        self.check(payload["modality"] == modality_key, f"{rel}: companion modality != {modality_key}")
                        self.check(token_by_key.get(payload["modality"]) == token, f"{rel}: token mismatch")
                        header_spacing = [float(v) for v in nib.load(str(emb_path)).header["pixdim"][1:4]]
                        self.check(
                            all(abs(a - b) < 1e-6 for a, b in zip(payload["spacing"], header_spacing)),
                            f"{rel}: companion spacing != embedding pixdim",
                        )
            # Holdout isolation: no embedding may exist for a holdout case.
            for case in info["cases"]["holdout"]:
                for suffix in MODALITIES:
                    rel = self._layout.emb_rel(ch, case, suffix)
                    self.check(
                        not (self._phase_root / "embeddings" / rel).exists(),
                        f"holdout case encoded: {rel}",
                    )

    def verify_labels(self, spot_n=3):
        for ch, info in sorted(self._manifest["challenges"].items()):
            for side in SIDES:
                for case in info["cases"][side]:
                    if (ch, case) in PIPELINE_EXCLUSIONS:
                        continue
                    if spot_n <= 0:
                        continue
                    spot_n -= 1
                    combined = self._phase_root / "labels" / ch / case / f"{case}-combined.nii.gz"
                    tumour = self._phase_root / "labels" / ch / case / f"{case}-tumor129.nii.gz"
                    for path, allowed in ((combined, {0, BRAIN_LABEL, 129, 130, 131}), (tumour, {0, 129, 130, 131})):
                        if not path.is_file():
                            self.failures.append(f"missing label {path}")
                            continue
                        img = nib.load(str(path))
                        values = set(np.unique(np.asanyarray(img.dataobj)).tolist())
                        self.check(values <= allowed, f"{path.name} {case}: values {sorted(values)} not within {sorted(allowed)}")
                        self.check(img.shape == GRID, f"{path.name} {case}: shape {img.shape} != {GRID}")

    def verify_lists(self):
        side_of = self.side_of_cases()
        lists_dir = self._phase_root / "lists"
        for name in ("p1_image_only.json", "p1_image_only_dev.json", "p2_mask_cond.json", "p3_pairs.json"):
            self.check((lists_dir / name).is_file(), f"missing list {name}")
        if self.failures:
            return

        for name, expected_side in (("p1_image_only.json", "train"), ("p1_image_only_dev.json", "dev")):
            entries = json.loads((lists_dir / name).read_text())["training"]
            seen = {}
            for entry in entries:
                key = (entry["sub"], entry["case"])
                self.check(key not in PIPELINE_EXCLUSIONS, f"{name}: pipeline-excluded case {key} must not appear")
                seen.setdefault(key, set()).add(entry["modality"])
                self.check(side_of.get(key) == expected_side, f"{name}: {key} not on {expected_side} side")
            self.check(
                all(v == MODALITY_KEYS for v in seen.values()),
                f"{name}: a case does not carry exactly the four pinned modality entries",
            )

        for name, per_case_count in (("p2_mask_cond.json", len(MODALITIES)), ("p3_pairs.json", 12)):
            entries = json.loads((lists_dir / name).read_text())["training"]
            seen, folds = {}, {}
            for entry in entries:
                key = (entry["sub"], entry["case"])
                self.check(key not in PIPELINE_EXCLUSIONS, f"{name}: pipeline-excluded case {key} must not appear")
                seen.setdefault(key, []).append(entry)
                folds.setdefault(key, set()).add(entry["fold"])
                expected_side = "train" if entry["fold"] == 1 else "dev"
                self.check(side_of.get(key) == expected_side, f"{name}: {key} fold={entry['fold']} side mismatch")
                self.check((self._phase_root / entry["image"]).is_file(), f"{name}: embedding not on disk {entry['image']}")
                self.check((self._phase_root / entry["label"]).is_file(), f"{name}: label not on disk {entry['label']}")
            for key, items in seen.items():
                self.check(len(items) == per_case_count, f"{name}: {key} has {len(items)} entries != {per_case_count}")
                self.check(len(folds[key]) == 1, f"{name}: {key} ordered entries span multiple sides")
            if name == "p3_pairs.json":
                for key, items in seen.items():
                    pairs = {(e["src_modality"], e["modality"]) for e in items}
                    expected_pairs = {(s, t) for s in MODALITY_KEYS for t in MODALITY_KEYS if s != t}
                    self.check(pairs == expected_pairs, f"{name}: {key} pairs != 12 ordered src!=tgt pairs")

    def verify_nnunet_agreement(self, nnunet_manifest_path):
        """The generative sides must be bit-identical to the instrument manifest (issue #34)."""
        other = json.loads(Path(nnunet_manifest_path).read_text())
        for ch, info in self._manifest["challenges"].items():
            other_cases = other["challenges"][ch]["cases"]
            for side in ("train", "dev", "holdout"):
                self.check(
                    info["cases"][side] == other_cases[side],
                    f"{ch} {side}: generative manifest is not bit-identical to the instrument manifest",
                )

    def verify_increment_stability(self, manifest_v2_path):
        """Every case of the current manifest must keep its side in a future (incremental) manifest."""
        newer = json.loads(Path(manifest_v2_path).read_text())
        for ch, info in self._manifest["challenges"].items():
            new_sides = {
                side: set(cases) for side, cases in newer["challenges"].get(ch, {}).get("cases", {}).items()
            }
            for side in ("train", "dev", "holdout"):
                for case in info["cases"][side]:
                    for other_side, case_set in new_sides.items():
                        if case in case_set and other_side != side:
                            self.failures.append(f"{ch} {case}: moved {side} -> {other_side} in v2 manifest")
                    if not any(case in s for s in new_sides.values()):
                        self.failures.append(f"{ch} {case}: dropped from v2 manifest")

    def verify_modality_mapping(self, modality_mapping_path):
        """The repo vocabulary must pin the four generative tokens (t1c=34 is the P1-planned addition)."""
        if not modality_mapping_path:
            return
        mapping = json.loads(Path(modality_mapping_path).read_text())
        for suffix, (key, token) in MODALITIES.items():
            self.check(mapping.get(key) == token, f"modality_mapping: {key} != {token} ({suffix})")

    def verify(self, spot_n=3, modality_mapping_path=None):
        self.verify_manifest()
        self.verify_modality_mapping(modality_mapping_path)
        self.verify_companions_and_tokens()
        self.verify_labels(spot_n=spot_n)
        self.verify_lists()
        return self.failures


class PhaseSelfTest:
    """Fixture-driven end-to-end check on synthetic cases with non-subject ids (no GPU encode)."""

    FIXTURE_QUOTAS = {
        "GLI": {"train": 2, "dev": 1, "holdout": 1},
        "SSA": {"train": 2, "dev": 1, "holdout": 1},
    }

    def __init__(self, workdir):
        self._workdir = Path(workdir)

    def write_fixture_sources(self):
        root = self._workdir / "sources" / "ASNR-MICCAI-BraTS2023"
        for ch, quota in self.FIXTURE_QUOTAS.items():
            suffix = "MET" if ch == "METS" else ch
            challenge_dir = root / f"ASNR-MICCAI-BraTS2023-{suffix}-Challenge-TrainingData"
            for i in range(sum(quota.values())):
                case = f"FIX{ch}-{i:04d}-000"
                case_dir = challenge_dir / case
                case_dir.mkdir(parents=True, exist_ok=True)
                affine = np.eye(4)
                volume = np.zeros((240, 240, 155), dtype=np.float32)
                volume[60:180, 60:180, 60:100] = 0.3  # non-zero brain region for the 22-union
                for modality in MODALITIES:
                    nib.save(nib.Nifti1Image(volume, affine), str(case_dir / f"{case}-{modality}.nii.gz"))
                seg = np.zeros((240, 240, 155), dtype=np.int16)
                seg[80:100, 80:100, 70:80] = 1
                seg[100:120, 100:120, 70:80] = 2
                seg[80:100, 100:120, 75:90] = 3
                nib.save(nib.Nifti1Image(seg, affine), str(case_dir / f"{case}-seg.nii.gz"))
        return root

    def write_fake_embeddings(self, manifest, phase_root):
        """Writes synthetic latents with a resize-consistent affine so companions/lists run for real."""
        scale = np.diag([240 / GRID[0], 240 / GRID[1], 155 / GRID[2], 1.0])
        layout = RawLayout(manifest)
        latent = np.zeros((GRID[0] // 4, GRID[1] // 4, GRID[2] // 4, 4), dtype=np.float32)
        for ch, info in manifest["challenges"].items():
            for side in SIDES:
                for case in info["cases"][side]:
                    for suffix in MODALITIES:
                        emb_path = phase_root / "embeddings" / layout.emb_rel(ch, case, suffix)
                        emb_path.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(nib.Nifti1Image(latent, scale), str(emb_path))

    def build_fixture_nnunet_view(self, manifest, sources):
        """Links fixture train-side files into a nnU-Net-style layout for the link-view check."""
        nnunet_root = self._workdir / "nnunet"
        for ch, info in manifest["challenges"].items():
            dataset = nnunet_root / f"Dataset{DATASET_IDS[ch]:03d}_BraTS2023{ch}"
            (dataset / "imagesTr").mkdir(parents=True, exist_ok=True)
            (dataset / "labelsTr").mkdir(parents=True, exist_ok=True)
            source_dir = Path(info["source_dir"])
            for case in info["cases"]["train"]:
                for modality, channel in CHANNELS:
                    target = dataset / "imagesTr" / f"{case}_{channel}.nii.gz"
                    if not target.exists():
                        target.symlink_to((source_dir / case / f"{case}-{modality}.nii.gz").resolve())
                label = dataset / "labelsTr" / f"{case}.nii.gz"
                if not label.exists():
                    label.symlink_to((source_dir / case / f"{case}-seg.nii.gz").resolve())
        return nnunet_root

    def run(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        # The pipeline exclusions are a pinned, reason-carrying record (the
        # real-run METS-00232 corruption); assert they stay documented.
        if not all(reason.strip() for reason in PIPELINE_EXCLUSIONS.values()):
            return ["pipeline exclusions must all carry a reason"]
        sources = self.write_fixture_sources()
        phase_root = self._workdir / "phase"
        phase_root.mkdir(parents=True, exist_ok=True)

        provider = PhaseManifestProvider(self.FIXTURE_QUOTAS)
        manifest = provider.rebuild(sources, phase_root / "phase_manifest.json")

        # Link-view round trip: nnU-Net-style train side + official dev side
        # must reproduce a fully readable official layout under the view root.
        nnunet_root = self.build_fixture_nnunet_view(manifest, sources)
        view_root = self._workdir / "view"
        LinkViewBuilder(manifest, nnunet_root, sources, view_root).build_all()
        remapped = PhaseManifestProvider(self.FIXTURE_QUOTAS).load(
            phase_root / "phase_manifest.json", raw_root=view_root / "ASNR-MICCAI-BraTS2023"
        )
        link_failures = []
        for ch, info in remapped["challenges"].items():
            for side in SIDES:
                for case in info["cases"][side]:
                    for suffix in list(MODALITIES) + ["seg"]:
                        path = Path(info["source_dir"]) / case / f"{case}-{suffix}.nii.gz"
                        if not path.is_file():
                            link_failures.append(f"link view missing {ch} {side} {case} {suffix}")

        EncodeListBuilder(manifest).write(phase_root / "encode_source.json")
        LabelVolumeBuilder(manifest, phase_root).build_all()
        self.write_fake_embeddings(manifest, phase_root)
        EmbCompanionWriter(manifest, phase_root / "embeddings").write_all()
        PhaseListBuilder(manifest, phase_root).write(phase_root / "lists")

        failures = list(PhaseVerifier(manifest, phase_root, quotas=self.FIXTURE_QUOTAS).verify())
        failures += link_failures

        # Incremental-manifest discipline: an identical re-split keeps every
        # case on its side; a re-split under a different salt must be flagged.
        stable_path = self._workdir / "manifest_stable.json"
        provider.rebuild(sources, stable_path)
        stable_verifier = PhaseVerifier(manifest, phase_root)
        stable_verifier.verify_increment_stability(stable_path)
        failures += stable_verifier.failures

        shuffled = json.loads(stable_path.read_text())
        first_case = shuffled["challenges"]["GLI"]["cases"]["train"][0]
        shuffled["challenges"]["GLI"]["cases"]["train"].remove(first_case)
        shuffled["challenges"]["GLI"]["cases"]["dev"].append(first_case)
        moved_path = self._workdir / "manifest_moved.json"
        moved_path.write_text(json.dumps(shuffled, indent=2) + "\n")
        moved_verifier = PhaseVerifier(manifest, phase_root)
        moved_verifier.verify_increment_stability(moved_path)
        if not moved_verifier.failures:
            failures.append("increment stability: a moved case was not flagged")

        return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", help="reuse an existing verified manifest or rebuild bit-identically")
    p.add_argument("--reuse", help="path to the issue #34 splits manifest (preferred)")
    p.add_argument("--brats-root", help="raw data root (rebuild mode)")
    p.add_argument("--raw-root", help="official-layout view root to remap source_dir onto (reuse mode)")
    p.add_argument("--out", required=True)
    p.set_defaults(command_name="manifest")

    p = sub.add_parser("link-view", help="link sugon nnU-Net + dev_raw copies into the official layout")
    p.add_argument("--manifest", required=True)
    p.add_argument("--nnunet-root", required=True)
    p.add_argument("--dev-raw-root", required=True)
    p.add_argument("--out-root", required=True)
    p.set_defaults(command_name="link-view")

    p = sub.add_parser("encode-list", help="write the VAE-encode input list (train+dev sides)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(command_name="encode-list")

    p = sub.add_parser("labels", help="write P2 combined + P3 tumour labels on the pinned grid")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--sides", default=",".join(SIDES))
    p.set_defaults(command_name="labels")

    p = sub.add_parser("companions", help="write <emb>.nii.gz.json {spacing, modality} companions")
    p.add_argument("--manifest", required=True)
    p.add_argument("--emb-root", required=True)
    p.add_argument("--sides", default=",".join(SIDES))
    p.set_defaults(command_name="companions")

    p = sub.add_parser("lists", help="write the P1/P2/P3 data lists")
    p.add_argument("--manifest", required=True)
    p.add_argument("--phase-root", required=True)
    p.set_defaults(command_name="lists")

    p = sub.add_parser("verify", help="verify quotas, isolation, companions, labels, lists")
    p.add_argument("--manifest", required=True)
    p.add_argument("--phase-root", required=True)
    p.add_argument("--nnunet-manifest", help="issue #34 manifest to check bit-identical sides")
    p.add_argument("--manifest-v2", help="future manifest to check incremental stability")
    p.add_argument("--report", help="write the failure list as JSON (controlled dir)")
    p.add_argument("--modality-mapping", help="configs/modality_mapping.json to check the pinned tokens")
    p.add_argument("--spot-n", type=int, default=3)
    p.set_defaults(command_name="verify")

    p = sub.add_parser("selftest", help="fixture-driven end-to-end check (synthetic ids, no GPU)")
    p.add_argument("--workdir", required=True)
    p.set_defaults(command_name="selftest")

    args = parser.parse_args(argv)
    if args.command_name == "manifest":
        provider = PhaseManifestProvider(QUOTAS)
        if args.reuse:
            provider.reuse(args.reuse, args.out, raw_root=args.raw_root)
        elif args.brats_root:
            provider.rebuild(args.brats_root, args.out)
        else:
            parser.error("manifest needs --reuse or --brats-root")
        return 0

    if args.command_name == "link-view":
        manifest = json.loads(Path(args.manifest).read_text())
        LinkViewBuilder(manifest, args.nnunet_root, args.dev_raw_root, args.out_root).build_all()
        return 0

    if args.command_name == "selftest":
        failures = PhaseSelfTest(args.workdir).run()
        for f in failures:
            print("FAIL " + f, file=sys.stderr)
        if failures:
            return 1
        print("SELFTEST PASS")
        return 0

    manifest = json.loads(Path(args.manifest).read_text())
    if args.command_name == "encode-list":
        EncodeListBuilder(manifest).write(args.out)
    elif args.command_name == "labels":
        sides = tuple(s for s in args.sides.split(",") if s)
        LabelVolumeBuilder(manifest, args.out_root).build_all(sides=sides)
    elif args.command_name == "companions":
        sides = tuple(s for s in args.sides.split(",") if s)
        EmbCompanionWriter(manifest, args.emb_root).write_all(sides=sides)
    elif args.command_name == "lists":
        PhaseListBuilder(manifest, args.phase_root).write(Path(args.phase_root) / "lists")
    else:
        verifier = PhaseVerifier(manifest, args.phase_root)
        verifier.verify(spot_n=args.spot_n, modality_mapping_path=args.modality_mapping)
        if args.nnunet_manifest:
            verifier.verify_nnunet_agreement(args.nnunet_manifest)
        if args.manifest_v2:
            verifier.verify_increment_stability(args.manifest_v2)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps({"failures": verifier.failures}, indent=2) + "\n")
        for f in verifier.failures:
            print("FAIL " + f, file=sys.stderr)
        if verifier.failures:
            return 1
        print(f"VERIFY PASS ({len(manifest['challenges'])} challenges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
