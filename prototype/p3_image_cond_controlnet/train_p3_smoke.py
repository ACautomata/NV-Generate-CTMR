# PROTOTYPE (throwaway, wayfinder #18) — P3 image-conditioned ControlNet smoke training.
#
# Implements the training-side change of the issue #12 §7 checklist against the
# original rflow-mr-brain v1 DM:
#   - ControlNet condition = src-image latent (4ch) instead of binarize_labels
#     (8ch mask). Labels still enter the weighted loss (100x on 129/130/131),
#     never the condition.
#   - dataloader chain is isomorphic to the planned utils.py change
#     (prepare_maisi_controlnet_json_dataloader gains a src_image key) —
#     LoadImaged(keys=["image","label","src_image"]) over the 4D-latent NIfTIs
#     plus the same spacing/modality Lambdad steps.
#   - no condition dropout (uncond = zero latent exists only at inference;
#     CFG off by default per issue #12 §2).
# Hyperparameters per issue #12 §5 (= P2): lr 1e-5, batch 1, AdamW,
# PolynomialLR(2.0), L1 + weighted_loss, fp16 autocast + GradScaler.
#
# Run on gauss:  python prototype/p3_image_cond_controlnet/train_p3_smoke.py \
#                    --max-steps 120 --snapshot-every 40

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.schedulers import RFlowScheduler
from monai.networks.utils import copy_model_state
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_common import LATENT, MODALITIES, ModelBundle, OUT_DIR  # noqa: E402
from prep_p3_data import PAIRS_JSON  # noqa: E402
from scripts.utils import define_instance  # noqa: E402

MODALITY_INDEX = {key: idx for _suffix, (key, idx) in MODALITIES.items()}
WEIGHTED_LOSS_LABELS = [129, 130, 131]


class PairDataset(torch.utils.data.Dataset):
    """Loads (tgt latent, src latent, seg, spacing, modality) per ordered pair.

    Transform chain is isomorphic to the planned utils.py dataloader change —
    this class is the smoke-test evidence that adding src_image to
    LoadImaged/Orientationd works over the 4D-latent NIfTI format.
    """

    def __init__(self, json_path: Path, entries: list[dict]):
        self.entries = entries
        self.modality_index = MODALITY_INDEX
        self.transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "src_image"], image_only=True, ensure_channel_first=True),
                EnsureTyped(keys="label", dtype=torch.long),
                Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x) * 1e2),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        e = self.entries[i]
        d = self.transforms({"image": e["image"], "label": e["label"], "src_image": e["src_image"], "spacing": e["spacing"]})
        d["modality"] = torch.tensor(self.modality_index[e["modality"]], dtype=torch.long)
        return d


def load_entries(json_path: Path, smoke_only: bool) -> list[dict]:
    entries = json.loads(json_path.read_text())["training"]
    if smoke_only:
        entries = [e for e in entries if "t1c" not in e["src_modality"] and "t1c" not in e["modality"]]
    return entries


def build_controlnet(device: torch.device, dm: torch.nn.Module) -> torch.nn.Module:
    """4ch-condition ControlNet, warm-started from the DM encoder/mid (issue #12 §5)."""
    args = argparse.Namespace(**json.loads((OUT_DIR.parent / "network_config_p3.json").read_text()))
    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, dm.state_dict())
    return controlnet


def compute_model_output(src_latent, images, noise, timesteps, scheduler, controlnet, unet, spacing, modality):
    """P3 rewrite of train_controlnet.py::compute_model_output (:190 condition swap).

    The ONLY structural change vs the original: controlnet_cond = src latent
    (4ch, same space/grid as the noisy latent) instead of binarize_labels.
    """
    noisy_latent = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
    controlnet_inputs = {"x": noisy_latent, "timesteps": timesteps, "controlnet_cond": src_latent, "class_labels": modality}
    down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)
    model_output = unet(
        x=noisy_latent,
        timesteps=timesteps,
        spacing_tensor=spacing,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        class_labels=modality,
    )
    return model_output


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 image-conditioned ControlNet smoke training (wayfinder #18)")
    parser.add_argument("--data-list", type=Path, default=PAIRS_JSON)
    parser.add_argument("--max-steps", type=int, default=120, help="total optimizer steps for the smoke run")
    parser.add_argument("--snapshot-every", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weighted-loss", type=float, default=100.0)
    parser.add_argument("--resume", type=Path, default=None, help="controlnet ckpt to resume from")
    parser.add_argument("--include-t1c", action="store_true", help="train t1c pairs too (v1 DM never saw modality 34 — off by default)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = ModelBundle.load(device)
    unet = bundle.diffusion_unet
    for p in unet.parameters():
        p.requires_grad = False
    unet.eval()

    controlnet = build_controlnet(device, unet)
    if args.resume is not None:
        controlnet.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True)["controlnet_state_dict"])
        print(f"resumed controlnet from {args.resume}")

    scheduler = define_instance(
        argparse.Namespace(**json.loads((OUT_DIR.parent / "network_config_p3.json").read_text())), "noise_scheduler"
    )
    assert isinstance(scheduler, RFlowScheduler)

    entries = load_entries(args.data_list, smoke_only=not args.include_t1c)
    print(f"training entries: {len(entries)} ordered pairs (12/case x cases, src!=tgt{', t1c excluded' if not args.include_t1c else ''})")
    loader = DataLoader(PairDataset(args.data_list, entries), batch_size=1, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_steps, power=2.0)
    scaler = GradScaler("cuda")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loss_log = args.out_dir / "loss.jsonl"

    controlnet.train()
    step = 0
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            if step >= args.max_steps:
                done = True
                break
            images = batch["image"].to(device) * bundle.scale_factor
            src_latent = batch["src_image"].to(device) * bundle.scale_factor
            # LoadImaged reads the 4D latent NIfTI back as (C,X,Y,Z); collate
            # adds the batch dim -> (B, C=4, 64, 64, 32). This is the
            # utils.py src_image-key evidence: channel inference works.
            assert tuple(images.shape) == (1,) + LATENT, f"unexpected latent shape {images.shape}"
            labels = batch["label"].to(device)
            spacing = batch["spacing"].to(device)
            modality = batch["modality"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=True):
                noise = torch.randn_like(images)
                timesteps = scheduler.sample_timesteps(images)
                model_output = compute_model_output(
                    src_latent, images, noise, timesteps, scheduler, controlnet, unet, spacing, modality
                )
                model_gt = images - noise  # RFlow velocity target
                if args.weighted_loss > 1.0:
                    weights = torch.ones_like(images)
                    roi = torch.zeros([1, 1] + list(images.shape[2:]), device=device)
                    interp = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest")
                    for lab in WEIGHTED_LOSS_LABELS:
                        roi[interp == lab] = 1
                    weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = args.weighted_loss
                    loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
                else:
                    loss = F.l1_loss(model_output.float(), model_gt.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            step += 1

            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": lr_scheduler.get_last_lr()[0],
                "modality": int(modality.flatten()[0]),
                "sec": round(time.time() - t0, 1),
            }
            with loss_log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{step}/{args.max_steps}] loss={rec['loss']:.4f} lr={rec['lr']:.2e} {rec['sec']:.0f}s", flush=True)

            if step % args.snapshot_every == 0 or step == args.max_steps:
                ckpt_path = args.out_dir / f"controlnet_p3_smoke_step{step}.pt"
                torch.save({"step": step, "loss": rec["loss"], "controlnet_state_dict": controlnet.state_dict()}, ckpt_path)
                print(f"saved {ckpt_path}", flush=True)

    print("smoke training done.")


if __name__ == "__main__":
    main()
