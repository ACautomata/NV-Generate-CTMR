# PROTOTYPE (throwaway) — dtype/device ablation on a small patch to isolate the bad-MAE cause.
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import CenterSpatialCrop, Compose

sys.path.insert(0, ".")
sys.path.insert(0, "prototype/vae_recon_brats")

from vae_recon_smoke import AE_PATH, NETWORK_CONFIG, AutoencoderRunner, VolumeLoader  # noqa: E402
from scripts.utils import define_instance  # noqa: E402
import argparse, json  # noqa: E402


def load_ae(device, dtype):
    args = argparse.Namespace(**json.load(open(NETWORK_CONFIG)))
    ae = define_instance(args, "autoencoder_def")
    ckpt = torch.load(AE_PATH, map_location="cpu", weights_only=True)
    if "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    ae.load_state_dict(ckpt)
    return ae.eval().to(device).to(dtype)


loader = VolumeLoader()
vol = loader.load_case(Path("datasets/brats2023_samples/GLI/BraTS-GLI-00000-000"))
crop = Compose([CenterSpatialCrop(roi_size=(96, 96, 96))])
x96 = crop(vol["t1c"])  # [1,96,96,96]

results = {}
configs = [
    ("cpu_fp32", torch.device("cpu"), torch.float32),
    ("cpu_fp16", torch.device("cpu"), torch.float16),
    ("mps_fp16", torch.device("mps"), torch.float16),
]
for name, device, dtype in configs:
    try:
        ae = load_ae(device, dtype)
        with torch.inference_mode():
            t0 = time.time()
            xx = x96.unsqueeze(0).to(device).to(dtype)
            z = ae.encode_stage_2_inputs(xx)
            y = ae.decode_stage_2_outputs(z)
            y = torch.clip(y.float(), 0, 1)[0].cpu()
            dt = time.time() - t0
        mae = float((x96 - y).abs().mean())
        results[name] = (y, mae, dt, tuple(z.shape))
        print(f"{name}: MAE={mae:.4f}  z={tuple(z.shape)}  {dt:.1f}s", flush=True)
    except Exception as e:
        print(f"{name}: FAILED — {type(e).__name__}: {e}", flush=True)
        results[name] = (None, None, None, None)

# figure: original + each successful recon at the tumor-max axial slice
seg = crop(vol["seg"])[0]
axial = int(torch.argmax((seg > 0).sum(dim=(0, 1))))
cols = [("original", x96)] + [(n, r[0]) for n, r in results.items() if r[0] is not None]
fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4))
for ax, (name, t) in zip(axes, cols):
    ax.imshow(t[0, :, :, axial].numpy().T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(name, fontsize=10)
    ax.axis("off")
fig.suptitle(f"t1c 96^3 center crop, axial z={axial}")
fig.tight_layout()
fig.savefig("prototype/vae_recon_brats/ablation_dtype.png", dpi=120)
print("fig saved: prototype/vae_recon_brats/ablation_dtype.png")
