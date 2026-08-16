# PROTOTYPE (throwaway, wayfinder #9) — VAE reconstruction smoke test on BraTS2023 samples.
#
# Question: is autoencoder_v1 (rflow-mr-brain's frozen VAE) good enough on BraTS?
#   - encoder→decoder round-trip on 12 local samples (GLI/MEN/SSA/PED x3), 4 modalities each
#   - grid handling: 240x240x155 -> multiples of 4, two candidate plans:
#       A) DivisiblePadd(k=4)        -> 240x240x156   (repo val-pipeline default)
#       B) ResizeWithPadOrCrop       -> 256x256x160   (census #3 recommendation, /32-compatible)
#   - outputs per-case comparison figures + a metrics table (MAE / PSNR in [0,1] domain)
#
# Run:  ~/venvs/nv-prototype/bin/python prototype/vae_recon_brats/vae_recon_smoke.py
# Figures land in datasets/brats2023_samples/recon_figs/ (gitignored, local-only per CC BY-NC).

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root for `scripts` package

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    DivisiblePad,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    ResizeWithPadOrCrop,
    ScaleIntensityRangePercentiles,
)

from scripts.utils import define_instance

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SUBCHALLENGES = ["GLI", "MEN", "SSA", "PED"]
SAMPLES_DIR = Path("datasets/brats2023_samples")
FIGS_DIR = SAMPLES_DIR / "recon_figs"
NETWORK_CONFIG = "configs/config_network_rflow.json"
AE_PATH = "models/autoencoder_v1.pt"


class GridPlan(NamedTuple):
    """One candidate handling of the 240x240x155 -> multiple-of-4 grid problem."""

    name: str
    to_grid: Callable  # volume[C,X,Y,Z] float tensor -> padded tensor


class CaseReconResult(NamedTuple):
    """Metrics for one (case, modality, plan) reconstruction."""

    sub: str
    case: str
    modality: str
    plan: str
    mae: float
    psnr: float


def make_plans() -> list[GridPlan]:
    return [
        GridPlan(
            name="pad4_240x240x156",
            to_grid=lambda v: DivisiblePad(k=4)(v),
        ),
        GridPlan(
            name="pad_256x256x160",
            to_grid=lambda v: ResizeWithPadOrCrop(spatial_size=(256, 256, 160))(v),
        ),
    ]


class AutoencoderRunner:
    """Loads autoencoder_v1 and round-trips volumes through encode_stage_2_inputs / decode_stage_2_outputs."""

    def __init__(self, autoencoder: torch.nn.Module, device: torch.device):
        # fp32 weights + CUDA autocast-half forward — mirrors production
        # (diff_model_create_training_data.py wraps encode in torch.amp.autocast("cuda")).
        # MPS/CPU run pure fp32 (no autocast there).
        self.autoencoder = autoencoder.eval().to(device).float()
        self.device = device
        self.use_autocast = device.type == "cuda"

    @classmethod
    def from_config(cls, network_config: str, ae_path: str, device: torch.device) -> "AutoencoderRunner":
        args = argparse.Namespace(**json.load(open(network_config)))
        args.trained_autoencoder_path = ae_path
        autoencoder = define_instance(args, "autoencoder_def")
        checkpoint = torch.load(ae_path, map_location="cpu", weights_only=True)
        if "unet_state_dict" in checkpoint.keys():
            checkpoint = checkpoint["unet_state_dict"]
        autoencoder.load_state_dict(checkpoint)
        return cls(autoencoder, device)

    @torch.inference_mode()
    def recon(self, volume: torch.Tensor, plan: GridPlan) -> torch.Tensor:
        """volume: [C,X,Y,Z] float tensor in [0,1]; returns reconstruction cropped back to original grid."""
        original_size = tuple(volume.shape[1:])
        x = plan.to_grid(volume).unsqueeze(0).to(self.device).float()  # [1,C,X',Y',Z']
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast):
            z = self.autoencoder.encode_stage_2_inputs(x)
            if isinstance(z, (tuple, list)):
                z = z[0]
            y = self.autoencoder.decode_stage_2_outputs(z)
        y = torch.clip(y.float(), 0.0, 1.0)[0].cpu()
        # crop back to the original grid: plan A pads tail (crop from start), plan B center-pads (center-crop)
        if plan.name.startswith("pad_"):
            y = ResizeWithPadOrCrop(spatial_size=original_size)(y)
        else:
            y = y[:, : original_size[0], : original_size[1], : original_size[2]]
        return y


class ReconMetrics:
    """MAE / PSNR in the [0,1] normalized domain."""

    @staticmethod
    def mae(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.mean(torch.abs(a - b)))

    @staticmethod
    def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
        rmse = float(torch.sqrt(torch.mean((a - b) ** 2)))
        return float("inf") if rmse == 0 else 20.0 * np.log10(1.0 / rmse)


class VolumeLoader:
    """BraTS per-case loading with the repo's MRI normalization (percentile 0-99.5 -> [0,1])."""

    def __init__(self):
        self.image_transforms = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Orientation(axcodes="RAS"),
                ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=False),
                EnsureType(dtype=torch.float32),
            ]
        )
        self.label_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Orientation(axcodes="RAS")])

    def load_case(self, case_dir: Path) -> dict[str, torch.Tensor]:
        case = case_dir.name
        volumes = {m: self.image_transforms(str(case_dir / f"{case}-{m}.nii.gz")) for m in MODALITIES}
        volumes["seg"] = self.label_transforms(str(case_dir / f"{case}-seg.nii.gz"))
        return volumes


@dataclass
class TumorSliceLocator:
    """Finds representative slice indices from the seg volume."""

    axial: int
    coronal: int
    sagittal: int

    @classmethod
    def from_seg(cls, seg: torch.Tensor) -> "TumorSliceLocator":
        s = seg[0] > 0  # [X,Y,Z] bool
        if s.any():
            axial = int(torch.argmax(s.sum(dim=(0, 1))))  # z with most tumor
            coronal = int(torch.argmax(s.sum(dim=(0, 2))))  # y
            sagittal = int(torch.argmax(s.sum(dim=(1, 2))))  # x
        else:
            c = np.array(s.shape) // 2
            axial, coronal, sagittal = int(c[2]), int(c[1]), int(c[0])
        return cls(axial=axial, coronal=coronal, sagittal=sagittal)


class ReconPlotter:
    """Renders per-case comparison figures: axial tumor slice (all modalities) + orthogonal triplanar (t1c)."""

    def __init__(self, figs_dir: Path):
        self.figs_dir = figs_dir
        figs_dir.mkdir(parents=True, exist_ok=True)

    def plot_axial(self, sub: str, case: str, vol: dict, recon: dict, locator: TumorSliceLocator, plan_names: list[str]):
        n_mod = len(MODALITIES)
        n_col = 1 + len(plan_names) + 1  # orig | recon per plan | diff(first plan)
        fig, axes = plt.subplots(n_mod, n_col, figsize=(3 * n_col, 3 * n_mod))
        for r, m in enumerate(MODALITIES):
            orig = vol[m][0, :, :, locator.axial].numpy()
            cols = [orig] + [recon[p][m][0, :, :, locator.axial].numpy() for p in plan_names]
            cols.append(np.abs(orig - cols[1]))
            for c, img in enumerate(cols):
                ax = axes[r, c]
                ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.axis("off")
                if r == 0:
                    title = "original" if c == 0 else (f"recon {plan_names[c-1]}" if c < n_col - 1 else f"|diff| ({plan_names[0]})")
                    ax.set_title(title, fontsize=9)
                if c == 0:
                    ax.set_ylabel(m, fontsize=10)
        fig.suptitle(f"{sub} {case} — axial slice z={locator.axial}", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.figs_dir / f"{sub}_{case}_axial.png", dpi=110)
        plt.close(fig)

    def plot_ortho(self, sub: str, case: str, vol: dict, recon: dict, locator: TumorSliceLocator, plan_names: list[str]):
        views = [
            ("axial", lambda v: v[0, :, :, locator.axial].numpy().T),
            ("coronal", lambda v: v[0, :, locator.coronal, :].numpy().T),
            ("sagittal", lambda v: v[0, locator.sagittal, :, :].numpy().T),
        ]
        n_col = 1 + len(plan_names) + 1
        fig, axes = plt.subplots(len(views), n_col, figsize=(3 * n_col, 3 * len(views)))
        for r, (view_name, getter) in enumerate(views):
            orig = getter(vol["t1c"])
            cols = [orig] + [getter(recon[p]["t1c"]) for p in plan_names]
            cols.append(np.abs(orig - cols[1]))
            for c, img in enumerate(cols):
                ax = axes[r, c]
                ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.axis("off")
                if r == 0:
                    title = "original t1c" if c == 0 else (f"recon {plan_names[c-1]}" if c < n_col - 1 else "|diff|")
                    ax.set_title(title, fontsize=9)
                if c == 0:
                    ax.set_ylabel(view_name, fontsize=10)
        fig.suptitle(f"{sub} {case} — t1c orthogonal views (tumor-centered)", fontsize=11)
        fig.tight_layout()
        fig.savefig(self.figs_dir / f"{sub}_{case}_ortho.png", dpi=110)
        plt.close(fig)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = pick_device()
    print(f"device: {device}")
    runner = AutoencoderRunner.from_config(NETWORK_CONFIG, AE_PATH, device)
    plans = make_plans()
    plan_names = [p.name for p in plans]
    loader = VolumeLoader()
    plotter = ReconPlotter(FIGS_DIR)
    metrics = ReconMetrics()
    results: list[CaseReconResult] = []

    case_dirs = sorted(p for sub in SUBCHALLENGES for p in (SAMPLES_DIR / sub).glob("BraTS-*"))
    print(f"{len(case_dirs)} cases")
    for i, case_dir in enumerate(case_dirs):
        sub = case_dir.parent.name
        case = case_dir.name
        print(f"[{i+1}/{len(case_dirs)}] {sub}/{case}", flush=True)
        vol = loader.load_case(case_dir)
        locator = TumorSliceLocator.from_seg(vol["seg"])
        recon: dict[str, dict[str, torch.Tensor]] = {}
        for plan in plans:
            recon[plan.name] = {}
            for m in MODALITIES:
                y = runner.recon(vol[m], plan)
                recon[plan.name][m] = y
                results.append(
                    CaseReconResult(
                        sub,
                        case,
                        m,
                        plan.name,
                        mae=metrics.mae(vol[m], y),
                        psnr=metrics.psnr(vol[m], y),
                    )
                )
        plotter.plot_axial(sub, case, vol, recon, locator, plan_names)
        plotter.plot_ortho(sub, case, vol, recon, locator, plan_names)

    # metrics summary table
    lines = ["| sub | case | modality | " + " | ".join(f"{p} MAE / PSNR" for p in plan_names) + " |",
             "|---|---|" + "---|" * len(plan_names)]
    by_key: dict[tuple, dict] = {}
    for r in results:
        by_key.setdefault((r.sub, r.case, r.modality), {})[r.plan] = (r.mae, r.psnr)
    for (sub, case, m), plans_metrics in by_key.items():
        row = [f"{mm:.4f} / {pp:.2f}" for p in plan_names for mm, pp in [plans_metrics[p]]]
        lines.append(f"| {sub} | {case} | {m} | " + " | ".join(row) + " |")
    table = "\n".join(lines)
    (FIGS_DIR / "metrics.md").write_text(table + "\n")
    print(table)
    print(f"\nFigures: {FIGS_DIR}/")


if __name__ == "__main__":
    main()
