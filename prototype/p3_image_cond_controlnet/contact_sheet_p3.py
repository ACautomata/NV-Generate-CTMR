# PROTOTYPE (throwaway, wayfinder #18) — contact sheet for visual inspection.
#
# Rows = ordered pairs (case, src->tgt); columns = real src | real tgt |
# ControlNet-generated tgt | img2img baseline | |tgt - gen| difference.
# Axial slice at the tumor-max z (seg-located, #9 style). Generated volumes
# are in the MR [0,1000] domain -> normalized here for display.
#
# Run on gauss after run_smoke.sh:  python prototype/p3_image_cond_controlnet/contact_sheet_p3.py

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_common import OUT_DIR, SAMPLES_DIR, VolumeLoader  # noqa: E402

GEN_DIR = OUT_DIR / "gen"
SHEET_PATH = OUT_DIR / "contact_sheet_p3.png"


def load_real(loader: VolumeLoader, sub: str, case: str, mod: str) -> np.ndarray:
    vol = loader.image(SAMPLES_DIR / sub / case / f"{case}-{mod}.nii.gz")  # [1,X,Y,Z] in [0,1]
    return vol[0].numpy()


def load_generated(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata(dtype=np.float32)  # [X,Y,Z] in [0,1000]
    arr = np.clip(arr / 1000.0, 0.0, 1.0)
    if arr.ndim == 4:  # defensive: latent-format file would be [X,Y,Z,C]
        arr = arr[..., 0]
    return arr


def tumor_axial_z(seg_path: Path) -> int:
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)  # [X,Y,Z] on GRID
    tumor = seg > 0
    if tumor.any():
        return int(np.argmax(tumor.sum(axis=(0, 1))))
    return seg.shape[2] // 2


def main() -> None:
    loader = VolumeLoader()
    # (sub, case, src, tgt) — covers the two directions the ticket calls out:
    # edema hyperintensity must appear on T2/FLAIR targets; one reverse pair.
    pairs = [
        ("GLI", "BraTS-GLI-00000-000", "t1n", "t2w"),
        ("MEN", "BraTS-MEN-00008-000", "t1n", "t2f"),
        ("SSA", "BraTS-SSA-00002-000", "t2w", "t1n"),
    ]
    cols = ["real src", "real tgt", "ControlNet tgt", "img2img tgt", "|tgt − cnet|"]
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(3.1 * len(cols), 3.1 * len(pairs)))
    if len(pairs) == 1:
        axes = axes[None, :]

    for r, (sub, case, src, tgt) in enumerate(pairs):
        z = tumor_axial_z(OUT_DIR / "segs" / f"{case}-seg129.nii.gz")
        real_src = load_real(loader, sub, case, src)
        real_tgt = load_real(loader, sub, case, tgt)
        gen_cnet = load_generated(GEN_DIR / f"{case}-{src}_to_{tgt}_cnet.nii.gz")
        gen_i2i = load_generated(GEN_DIR / f"{case}-{src}_to_{tgt}_img2img.nii.gz")
        diff = np.abs(real_tgt - gen_cnet)

        imgs = [real_src[:, :, z].T, real_tgt[:, :, z].T, gen_cnet[:, :, z].T, gen_i2i[:, :, z].T, diff[:, :, z].T]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{sub}/{case.split('-')[-2]}\n{src}→{tgt} (z={z})", fontsize=9)

    fig.suptitle("P3 image-conditioned ControlNet smoke — user visual inspection (wayfinder #18)", fontsize=12)
    fig.tight_layout()
    fig.savefig(SHEET_PATH, dpi=130)
    plt.close(fig)
    print(f"saved {SHEET_PATH}")


if __name__ == "__main__":
    main()
