# PROTOTYPE (throwaway, wayfinder #18) — P3 data prep for the smoke test.
#
# Implements the data side of the issue #12 §7 checklist:
#   - VAE-encode 12 cases x 4 modalities -> *_emb.nii.gz on the pinned grid
#     (storage format mirrors scripts/diff_model_create_training_data.py:
#     latent [C,X,Y,Z] -> NIfTI [X,Y,Z,C]).
#   - Seg remap 1/2/3 -> 129/130/131 (P2 vocabulary) on the same grid.
#   - Emit the 12 ordered-pairs-per-case data list (src != tgt over
#     t1n/t1c/t2w/t2f). Entries reference the per-case embeddings — 4 files
#     per case, 12 entries reuse them (no storage blow-up).
#
# The full 12-pair list is the *real-pipeline* artifact. Smoke training
# filters to the 6 pairs among t1n/t2w/t2f because the v1 DM never saw
# modality 34 (t1c) — that's a P1-planned class embedding, not a v1 one.
#
# Run on gauss:  python prototype/p3_image_cond_controlnet/prep_p3_data.py

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_common import (  # noqa: E402
    GRID,
    LABEL_REMAP,
    MODALITIES,
    ModelBundle,
    OUT_DIR,
    SAMPLES_DIR,
    VolumeLoader,
    list_cases,
)

EMB_DIR = OUT_DIR / "embeddings"
SEG_DIR = OUT_DIR / "segs"
PAIRS_JSON = OUT_DIR / "p3_pairs.json"


def remap_label_values(seg: torch.Tensor) -> torch.Tensor:
    """BraTS 1/2/3 -> 129/130/131 (P2 vocabulary; enters loss only, never the condition)."""
    out = seg.clone()
    for orig, mapped in LABEL_REMAP.items():
        out[seg == orig] = mapped
    return out


def save_latent_nifti(z: torch.Tensor, affine, path: Path) -> None:
    """latent [1,4,X,Y,Z] -> NIfTI [X,Y,Z,C], mirroring diff_model_create_training_data.py:188-191."""
    arr = z.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)  # [X,Y,Z,4]
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def save_seg_nifti(seg: torch.Tensor, affine, path: Path) -> None:
    arr = seg.squeeze(0).short().cpu().numpy()  # [X,Y,Z], values {0,129,130,131}
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    bundle = ModelBundle.load(device)
    print(f"scale_factor from DM ckpt: {bundle.scale_factor}")
    loader = VolumeLoader()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    SEG_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    cases = list_cases()
    print(f"{len(cases)} cases")
    for i, (sub, case) in enumerate(cases):
        case_dir = SAMPLES_DIR / sub / case
        emb_paths = {}
        affine = None
        for mod, (mod_key, _idx) in MODALITIES.items():
            out_path = EMB_DIR / f"{case}-{mod}_emb.nii.gz"
            if not out_path.exists():
                vol = loader.image(case_dir / f"{case}-{mod}.nii.gz")  # [1,X,Y,Z] in [0,1] on GRID
                z = bundle.encode(vol.unsqueeze(0))  # [1,4,64,64,32]
                affine = vol.meta["affine"].numpy() if hasattr(vol, "meta") else affine
                save_latent_nifti(z, affine, out_path)
            emb_paths[mod] = out_path
        seg_path = SEG_DIR / f"{case}-seg129.nii.gz"
        if not seg_path.exists():
            seg = loader.label(case_dir / f"{case}-seg.nii.gz")
            seg = remap_label_values(seg)
            affine = seg.meta["affine"].numpy() if hasattr(seg, "meta") else affine
            save_seg_nifti(seg, affine, seg_path)

        # 12 ordered pairs per case (src != tgt) — the real-pipeline data list.
        for src_mod, (src_key, _s) in MODALITIES.items():
            for tgt_mod, (tgt_key, _t) in MODALITIES.items():
                if src_mod == tgt_mod:
                    continue
                entries.append(
                    {
                        "image": str(emb_paths[tgt_mod]),  # tgt latent (training target)
                        "src_image": str(emb_paths[src_mod]),  # src latent (ControlNet condition)
                        "label": str(seg_path),
                        "modality": tgt_key,  # tgt modality via class_labels path
                        "src_modality": src_key,
                        "spacing": [1.0, 1.0, 1.0],
                        "fold": 0,
                        "sub": sub,
                        "case": case,
                    }
                )
        print(f"[{i + 1}/{len(cases)}] {sub}/{case}: 4 embeddings + seg129 done")

    PAIRS_JSON.write_text(json.dumps({"training": entries}, indent=1))
    n_smoke = sum(1 for e in entries if "t1c" not in e["src_modality"] and "t1c" not in e["modality"])
    print(f"data list: {len(entries)} ordered pairs -> {PAIRS_JSON}")
    print(f"smoke-usable pairs (no t1c): {n_smoke}")


if __name__ == "__main__":
    main()
