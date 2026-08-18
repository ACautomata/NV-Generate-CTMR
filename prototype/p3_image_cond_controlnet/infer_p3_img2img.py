# PROTOTYPE (throwaway, wayfinder #18) — P3 stage-0 baseline: RF img2img.
#
# Zero-training cross-modality baseline (issue #12 §1 route 2): start the RF
# sampling loop from a straight-interpolation point between the src latent and
# noise, x_t0 = (1-t0)*src + t0*noise, with the tgt modality label on the DM;
# then run the normal RFlow Euler loop from t0 down to 0. Mathematically legal
# (training objective lives on this interpolation path) but out-of-distribution
# usage — hence baseline only, never the mainline.
#
# The v1 DM supports modality indices 29/30/31 only (t1c=34 untrained); the
# wrapper checks and refuses t1c targets — same constraint as smoke training.
#
# Run on gauss:  python prototype/p3_image_cond_controlnet/infer_p3_img2img.py \
#     --src <case>-t1n.nii.gz --tgt-modality t2w --strength 0.8

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.networks.schedulers import RFlowScheduler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_common import GRID, LATENT, MODALITIES, ModelBundle, OUT_DIR, VolumeLoader  # noqa: E402
from scripts.utils import define_instance  # noqa: E402

V1_SUPPORTED = {"t1n", "t2w", "t2f"}  # modality 34 (t1c) never seen by the v1 DM


def generate_img2img(
    src_nifti: Path,
    tgt_modality: str,
    out_path: Path,
    strength: float = 0.8,
    num_inference_steps: int = 30,
    seed: int = 0,
) -> Path:
    if tgt_modality not in V1_SUPPORTED:
        raise ValueError(f"v1 DM has no trained class embedding for {tgt_modality}; choose from {sorted(V1_SUPPORTED)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    bundle = ModelBundle.load(device)
    loader = VolumeLoader()
    src = loader.image(src_nifti)  # [1,X,Y,Z] in [0,1] on GRID
    src_latent = (bundle.encode(src.unsqueeze(0)) * bundle.scale_factor).half().to(device)  # [1,4,64,64,32]

    net_cfg = json.loads((OUT_DIR.parent / "network_config_p3.json").read_text())
    scheduler: RFlowScheduler = define_instance(argparse.Namespace(**net_cfg), "noise_scheduler")
    scheduler.set_timesteps(
        num_inference_steps=num_inference_steps,
        input_img_size_numel=torch.prod(torch.tensor(LATENT[1:], dtype=torch.float64)),
    )
    all_t = scheduler.timesteps
    all_next = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

    # pick the starting timestep: first scheduled t <= strength*1000
    t0_raw = strength * scheduler.num_train_timesteps
    start_idx = int(torch.argmax((all_t <= t0_raw).to(torch.int8)).item())
    t0 = all_t[start_idx]
    print(f"img2img start: t0={float(t0):.1f} (idx {start_idx}/{len(all_t)}, strength={strength})")

    noise = torch.randn(src_latent.shape, dtype=src_latent.dtype, device=device)
    latents = scheduler.add_noise(original_samples=src_latent, noise=noise, timesteps=torch.tensor([float(t0)], device=device))

    modality_idx = MODALITIES[tgt_modality][1]
    modality_tensor = torch.tensor([modality_idx], device=device)
    spacing_tensor = torch.tensor([[100.0, 100.0, 100.0]], device=device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for t, next_t in list(zip(all_t, all_next))[start_idx:]:
            model_output = bundle.diffusion_unet(
                x=latents,
                timesteps=torch.tensor([float(t)], device=device),
                spacing_tensor=spacing_tensor,
                class_labels=modality_tensor,
            )
            latents, _ = scheduler.step(model_output, t, latents, next_t)

    image = bundle.decode(latents.float() / bundle.scale_factor)  # [1,1,X,Y,Z] ~[0,1]
    arr = (torch.clip(image, 0.0, 1.0)[0, 0].float().cpu().numpy() * 1000.0).astype(np.float32)  # MR domain [0,1000]
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(out_path))
    print(f"saved {out_path}  (modality={tgt_modality} idx={modality_idx}, strength={strength})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 stage-0 img2img baseline (wayfinder #18)")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--tgt-modality", type=str, required=True, choices=list(MODALITIES.keys()))
    parser.add_argument("--strength", type=float, default=0.8, help="noise level at the interpolation start (1.0 = full noise)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (OUT_DIR / "gen" / f"{args.src.stem.split('.')[0]}_to_{args.tgt_modality}_img2img.nii.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_img2img(args.src, args.tgt_modality, out, strength=args.strength, num_inference_steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
