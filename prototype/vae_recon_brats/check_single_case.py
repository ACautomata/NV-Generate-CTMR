# PROTOTYPE (throwaway) — single-case fast check for vae_recon_smoke.py
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "prototype/vae_recon_brats")

from vae_recon_smoke import (  # noqa: E402
    AE_PATH,
    NETWORK_CONFIG,
    AutoencoderRunner,
    VolumeLoader,
    make_plans,
)

from vae_recon_smoke import pick_device  # noqa: E402

device = pick_device()
t0 = time.time()
runner = AutoencoderRunner.from_config(NETWORK_CONFIG, AE_PATH, device)
print(f"AE loaded in {time.time()-t0:.1f}s")

loader = VolumeLoader()
case_dir = Path("datasets/brats2023_samples/GLI/BraTS-GLI-00000-000")
vol = loader.load_case(case_dir)
print("loaded t1c:", vol["t1c"].shape, "range", float(vol["t1c"].min()), float(vol["t1c"].max()))

for plan in make_plans():
    t0 = time.time()
    y = runner.recon(vol["t1c"], plan)
    dt = time.time() - t0
    mae = float((vol["t1c"] - y).abs().mean())
    print(f"{plan.name}: recon {tuple(y.shape)} in {dt:.1f}s, MAE={mae:.4f}")
