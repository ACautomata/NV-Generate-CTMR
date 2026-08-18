# PROTOTYPE (throwaway, wayfinder #18) — shared constants & loaders for the P3
# image-conditioned ControlNet smoke test.
#
# P3 route (issue #12 resolution): condition = pure src-image latent (4ch, no
# mask); tgt modality label rides the existing class_labels path into both DM
# and ControlNet. Smoke test pairs the 12-case BraTS sample set with the
# original rflow-mr-brain v1 DM.
#
# Run on gauss (A6000):  see run_smoke.sh

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    ResizeWithPadOrCrop,
    ScaleIntensityRangePercentiles,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root for `scripts` package

# ── Paths (gauss layout; #9's working dir reused for data + AE weight) ──────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = Path.home() / "nv-vae-brats" / "datasets" / "brats2023_samples"
MODELS_DIR = Path.home() / "nv-vae-brats" / "models"
OUT_DIR = REPO_ROOT / "prototype" / "p3_image_cond_controlnet" / "out"

AE_PATH = MODELS_DIR / "autoencoder_v1.pt"
DM_PATH = MODELS_DIR / "diff_unet_3d_rflow-mr-brain_v1.pt"
NETWORK_CONFIG = REPO_ROOT / "prototype" / "p3_image_cond_controlnet" / "network_config_p3.json"

# ── Modality table ───────────────────────────────────────────────────────────
# BraTS file suffix -> (modality_mapping key, class-label index).
# v1 DM has only ever seen 29/30/31; 34 (t1c) is the P1-planned addition and
# cannot be smoke-tested against v1 — pairs touching t1c are generated in the
# data list for the real pipeline but excluded from smoke training.
MODALITIES = {
    "t1n": ("mri_t1_skull_stripped", 29),
    "t2w": ("mri_t2_skull_stripped", 30),
    "t2f": ("mri_flair_skull_stripped", 31),
    "t1c": ("mri_t1c_skull_stripped", 34),  # not supported by v1 DM
}
SMOKE_MODALITIES = ["t1n", "t2w", "t2f"]  # 6 ordered pairs among these three

# P1/P2/P3 pinned grid (issue #10/#11/#12): 256x256x128 -> latent 64x64x32.
GRID = (256, 256, 128)
LATENT = (4, 64, 64, 32)

# P2 vocabulary remap (issue #11): BraTS 1/2/3 -> 129/130/131 (8-bit-safe).
# Labels enter ONLY the weighted loss, never the P3 condition (issue #12 §2).
LABEL_REMAP = {1: 129, 2: 130, 3: 131}

SUBCHALLENGES = ["GLI", "MEN", "SSA", "PED"]


def list_cases() -> list[tuple[str, str]]:
    """All 12 sample cases as (subchallenge, case-id), mirroring download_samples.sh."""
    cases = []
    for sub in SUBCHALLENGES:
        for case_dir in sorted((SAMPLES_DIR / sub).glob("BraTS-*")):
            if case_dir.is_dir():
                cases.append((sub, case_dir.name))
    return cases


class VolumeLoader:
    """BraTS volume loading with the repo's MRI normalization (#9 style) on the P3 grid.

    Normalization first (percentile 0-99.5 -> [0,1]), then ResizeWithPadOrCrop
    to 256x256x128 (z: 155->128 center-crop, per the pinned grid).
    """

    def __init__(self):
        self.image_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=False),
                ResizeWithPadOrCrop(spatial_size=GRID),
                EnsureType(dtype=torch.float32),
            ]
        )
        self.label_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                ResizeWithPadOrCrop(spatial_size=GRID),
            ]
        )

    def image(self, path: Path) -> torch.Tensor:
        return self.image_transforms(str(path))

    def label(self, path: Path) -> torch.Tensor:
        return self.label_transforms(str(path))


@dataclass
class ModelBundle:
    """Frozen rflow-mr-brain v1 networks + the 4ch-condition ControlNet definition."""

    autoencoder: torch.nn.Module
    diffusion_unet: torch.nn.Module
    scale_factor: float
    device: torch.device

    @classmethod
    def load(cls, device: torch.device) -> "ModelBundle":
        from monai.utils.enums import TraceKeys
        from scripts.utils import define_instance

        # torch>=2.6 weights_only allowlist: the official AE ckpt carries MONAI
        # MetaTensor trace objects (TraceKeys) — trusted source (nvidia release).
        torch.serialization.add_safe_globals([TraceKeys])

        args = dict_to_namespace(json.loads(NETWORK_CONFIG.read_text()))
        autoencoder = define_instance(args, "autoencoder_def").to(device).eval()
        checkpoint = torch.load(AE_PATH, map_location="cpu", weights_only=True)
        if "unet_state_dict" in checkpoint.keys():
            checkpoint = checkpoint["unet_state_dict"]
        autoencoder.load_state_dict(checkpoint)

        diffusion_unet = define_instance(args, "diffusion_unet_def").to(device).eval()
        dm_ckpt = torch.load(DM_PATH, map_location="cpu", weights_only=True)
        diffusion_unet.load_state_dict(dm_ckpt["unet_state_dict"], strict=False)
        return cls(
            autoencoder=autoencoder,
            diffusion_unet=diffusion_unet,
            scale_factor=float(dm_ckpt["scale_factor"]),
            device=device,
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image [1,1,X,Y,Z] in [0,1] -> latent [1,4,X/4,Y/4,Z/4] (unscaled)."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            z = self.autoencoder.encode_stage_2_inputs(image.to(self.device))
        if isinstance(z, (tuple, list)):
            z = z[0]
        return z.float()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [1,4,...] (unscaled) -> image [1,1,X,Y,Z] in ~[0,1]."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            y = self.autoencoder.decode_stage_2_outputs(latent.to(self.device))
            if isinstance(y, (tuple, list)):
                y = y[0]
        return y.float()


def dict_to_namespace(d: dict):
    """Flat config dict -> argparse-like namespace for define_instance."""
    import argparse

    return argparse.Namespace(**d)
