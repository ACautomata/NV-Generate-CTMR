# PROTOTYPE (throwaway, wayfinder #18) — P3 image-conditioned inference wrapper.
#
# The image-conditioned sibling of scripts/infer_image_from_mask.py
# (issue #12 §7 checklist row 4): src image -> VAE latent (4ch) ->
# scripts/utils_infer.run_controlnet_conditioned_image_dm (modality-agnostic
# core, reused unchanged) -> tgt-modality image.
#
# Evidence goals:
#   - the core loop + load_image_models work with a 4ch conditioning tensor
#     (config: conditioning_embedding_in_channels 8->4);
#   - uncond branch = all-zero latent tensor (issue #12 §2) usable via --cfg.
#
# Run on gauss:  python prototype/p3_image_cond_controlnet/infer_p3_controlnet.py \
#     --src <case>-t1n.nii.gz --tgt-modality t2w \
#     --controlnet-ckpt out/train/controlnet_p3_smoke_step120.pt

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_common import GRID, LATENT, MODALITIES, OUT_DIR, VolumeLoader  # noqa: E402
from scripts.utils_infer import load_image_models, run_controlnet_conditioned_image_dm  # noqa: E402

SLIDING_WINDOW_SIZE = (80, 80, 32)  # configs/config_maisi_controlnet_train_rflow-mr.json controlnet_infer
SLIDING_WINDOW_OVERLAP = 0.4


def build_args_namespace(controlnet_ckpt: Path) -> argparse.Namespace:
    """Merge the 4ch network config with checkpoint paths for load_image_models."""
    from p3_common import AE_PATH, DM_PATH

    cfg = json.loads((OUT_DIR.parent / "network_config_p3.json").read_text())
    cfg.update(
        {
            "trained_autoencoder_path": str(AE_PATH),
            "trained_diffusion_path": str(DM_PATH),
            "trained_controlnet_path": str(controlnet_ckpt),
        }
    )
    return argparse.Namespace(**cfg)


def generate_from_image(
    src_nifti: Path,
    tgt_modality: str,
    controlnet_ckpt: Path,
    out_path: Path,
    cfg_scale: float = 0.0,
    num_inference_steps: int = 30,
    seed: int = 0,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    args = build_args_namespace(controlnet_ckpt)
    autoencoder, diffusion_unet, controlnet, scale_factor, noise_scheduler = load_image_models(args, device)

    # src pre-processing (same chain as prep) -> condition latent, encoded with
    # the already-loaded AE (no second model bundle needed)
    loader = VolumeLoader()
    src = loader.image(src_nifti)  # [1,X,Y,Z] in [0,1] on GRID
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        z = autoencoder.encode_stage_2_inputs(src.unsqueeze(0).to(device))
        if isinstance(z, (tuple, list)):
            z = z[0]
    src_latent = z.float() * float(scale_factor)  # [1,4,64,64,32]

    controlnet_cond_tensor = src_latent.half().to(device)
    controlnet_uncond_tensor = None
    if cfg_scale > 0:
        # P3 uncond = all-zero latent tensor (issue #12 §2) — no mask removal needed.
        controlnet_uncond_tensor = torch.zeros_like(controlnet_cond_tensor)

    modality_idx = MODALITIES[tgt_modality][1]
    modality_tensor = torch.tensor([modality_idx], device=device)
    spacing_tensor = torch.tensor([[100.0, 100.0, 100.0]], device=device)  # BraTS 1mm iso, x1e2

    synthetic = run_controlnet_conditioned_image_dm(
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        device=device,
        controlnet_cond_tensor=controlnet_cond_tensor,
        spacing_tensor=spacing_tensor,
        latent_shape=LATENT,
        output_size=GRID,
        noise_factor=1.0,
        modality_tensor=modality_tensor,
        num_inference_steps=num_inference_steps,
        autoencoder_sliding_window_infer_size=SLIDING_WINDOW_SIZE,
        autoencoder_sliding_window_infer_overlap=SLIDING_WINDOW_OVERLAP,
        cfg_guidance_scale=cfg_scale,
        controlnet_uncond_tensor=controlnet_uncond_tensor,
    )

    # MR domain: core maps [0,1] -> [0,1000]. Save NIfTI on the padded grid.
    arr = synthetic[0, 0].float().cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(out_path))
    print(f"saved {out_path}  (modality={tgt_modality} idx={modality_idx}, cfg={cfg_scale})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 image-conditioned ControlNet inference (wayfinder #18)")
    parser.add_argument("--src", type=Path, required=True, help="source-image NIfTI (BraTS raw, any of t1n/t1c/t2w/t2f)")
    parser.add_argument("--tgt-modality", type=str, required=True, choices=list(MODALITIES.keys()))
    parser.add_argument("--controlnet-ckpt", type=Path, required=True)
    parser.add_argument("--cfg", type=float, default=0.0, help="CFG scale; 0 = off (default per issue #12 §2)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (OUT_DIR / "gen" / f"{args.src.stem.split('.')[0]}_to_{args.tgt_modality}_cnet.nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_from_image(args.src, args.tgt_modality, args.controlnet_ckpt, out, cfg_scale=args.cfg, num_inference_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
