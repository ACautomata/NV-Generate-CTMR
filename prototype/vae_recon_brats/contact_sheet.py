# PROTOTYPE (throwaway) — one contact sheet: t1c axial tumor slice, orig vs both grid plans, all 12 cases.
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vae_recon_smoke import (  # noqa: E402
    SAMPLES_DIR,
    SUBCHALLENGES,
    AutoencoderRunner,
    NETWORK_CONFIG,
    AE_PATH,
    VolumeLoader,
    TumorSliceLocator,
    make_plans,
    pick_device,
)

runner = AutoencoderRunner.from_config(NETWORK_CONFIG, AE_PATH, pick_device())
loader = VolumeLoader()
plans = make_plans()
cases = sorted(p for sub in SUBCHALLENGES for p in (SAMPLES_DIR / sub).glob("BraTS-*"))

fig, axes = plt.subplots(len(cases), 4, figsize=(13, 3.0 * len(cases)))
for r, case_dir in enumerate(cases):
    vol = loader.load_case(case_dir)
    loc = TumorSliceLocator.from_seg(vol["seg"])
    orig = vol["t1c"][0, :, :, loc.axial].numpy()
    tiles = [("original", orig)] + [(f"recon {p.name}", runner.recon(vol["t1c"], p)[0, :, :, loc.axial].numpy()) for p in plans]
    tiles.append(("|diff| A", np.abs(orig - tiles[1][1])))
    for c, (label, img) in enumerate(tiles):
        ax = axes[r, c]
        ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        if r == 0:
            ax.set_title(label, fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{case_dir.parent.name}\n{case_dir.name.replace('BraTS-', '')}", fontsize=8)
fig.suptitle("BraTS2023 VAE recon contact sheet — t1c axial (tumor-max) slices, 12 cases", fontsize=12)
fig.tight_layout()
out = SAMPLES_DIR / "recon_figs" / "_contact_sheet_t1c.png"
fig.savefig(out, dpi=95)
print(f"saved {out}")
