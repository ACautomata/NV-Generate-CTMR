# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P3 image-conditioned ControlNet candidate training (issue #61, spec #51 decision 4/6/8).

Inter-modality candidate: a ControlNet-only bypass hung off the FROZEN P1-DM (the
registered DM source, ADR-0006). The DM and VAE are untouched; the ControlNet
conditions on the 4ch **src-image latent** (``src_image`` in the ``p3_pairs.json``
list, no mask) and the target modality label rides the existing ``class_labels``
path into both DM and ControlNet — the training-side change of the issue #12 §7
checklist turned into a reusable recipe. Pinned hyperparameters are exactly the
P2 recipe (lr=1e-5, batch=1, <=100 epochs, AdamW, PolynomialLR power 2.0, L1,
cache_rate=0, weighted_loss=100 on 129/130/131, use_region_contrasive_loss OFF,
pure BraTS no MR-RATE replay) plus CFG=0 semantics. The ControlNet is initialized
from the frozen P1-DM encoder/mid (``copy_model_state``) and is NEVER warm-started
from a P2 ControlNet — only the P1-DM checkpoint is read.

Deltas against ``brats_p2_finetune.py``, all pinned:
- ``controlnet_cond`` is the src-image latent (``src_image``, 4ch) instead of the
  binarized 8ch mask (``binarize_labels``); labels only enter the weighted loss;
- the training list is the #52 ``p3_pairs.json`` (fold=1 train / fold=0 dev; the
  val split is DISCARDED — dev-eval selects the candidate, spec #51 decision 7);
- ``P3RecipeGuard`` additionally pins cfg_guidance_scale=0 (the candidate is
  evaluated and selected with CFG off) and refuses to load a ControlNet checkpoint;
- bf16 autocast default (DCU), fp32 fallback via ``--no_amp``.

Usage (torchrun on the DCU):
    torchrun --nproc_per_node=7 -m scripts.brats_p3_finetune \
        -e run/environment.json -c configs/config_brats_p3_train.json \
        -t configs/config_network_p3.json --data-list runs/p3/.../p3_pairs.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
from datetime import datetime, UTC
from pathlib import Path

import monai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.data import CacheDataset
from monai.networks.utils import copy_model_state
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import add_data_dir2path, define_instance, partition_dataset

STOP_FILE = ".early_stop"


def prepare_p3_controlnet_json_dataloader(
    json_data_list,
    data_base_dir,
    batch_size=1,
    fold=0,
    cache_rate=0.0,
    rank=0,
    world_size=1,
    modality_mapping=None,
):
    """P3 dataloader: image (tgt latent), label (loss-only tumour), src_image (4ch src latent).

    Isomorphic to ``utils.prepare_maisi_controlnet_json_dataloader`` but the transforms
    also load ``src_image`` (a 4ch latent, left float — never binarized) and the
    ``src_image`` path is joined to ``data_base_dir`` (``add_data_dir2path`` only
    joins ``image``/``label``). The val split (``fold == fold``) is returned but the
    trainer discards it.
    """
    if isinstance(json_data_list, list):
        list_train, list_valid = [], []
        for data_list, data_root in zip(json_data_list, data_base_dir):
            json_data = json.loads(Path(data_list).read_text())["training"]
            train, val = add_data_dir2path(copy.deepcopy(json_data), data_root, fold)
            # src_image is a latent, not a segmentation: join to the same per-list
            # root that add_data_dir2path used for image/label, keep float.
            for entry in train + val:
                if "src_image" in entry:
                    entry["src_image"] = os.path.join(data_root, entry["src_image"])
            list_train += train
            list_valid += val
    else:
        json_data = json.loads(Path(json_data_list).read_text())["training"]
        list_train, list_valid = add_data_dir2path(copy.deepcopy(json_data), data_base_dir, fold)
        for entry in list_train + list_valid:
            if "src_image" in entry:
                entry["src_image"] = os.path.join(data_base_dir, entry["src_image"])

    common_transform = [
        LoadImaged(keys=["image", "label", "src_image"], image_only=True, ensure_channel_first=True),
        Orientationd(keys=["image", "label", "src_image"], axcodes="RAS"),
        EnsureTyped(keys=["label"], dtype=torch.long, track_meta=True),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["spacing"], func=lambda x: x * 1e2, allow_missing_keys=True),
        Lambdad(keys=["modality"], func=lambda x: modality_mapping[x], allow_missing_keys=True),
        EnsureTyped(keys=["modality"], dtype=torch.long, allow_missing_keys=True),
    ]

    use_ddp = world_size > 1
    if use_ddp:
        list_train = partition_dataset(data=list_train, shuffle=True, num_partitions=world_size, even_divisible=True)[rank]
    train_ds = CacheDataset(data=list_train, transform=Compose(common_transform), cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if use_ddp:
        list_valid = partition_dataset(data=list_valid, shuffle=True, num_partitions=world_size, even_divisible=False)[rank]
    val_ds = CacheDataset(data=list_valid, transform=Compose(common_transform), cache_rate=cache_rate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    return train_loader, val_loader


class P3RecipeGuard:
    """Pinned-recipe guard: rejects any deviation from the frozen P3 image-conditioned recipe (issue #61).

    The recipe is the P2 recipe verbatim (ADR-0007) plus CFG=0 and an explicit
    no-warm-start-from-P2 clause.
    """

    PINNED_LR = 1e-5
    PINNED_BATCH = 1
    PINNED_WEIGHTED_LOSS = 100
    PINNED_WEIGHTED_LABELS = [129, 130, 131]
    PINNED_CACHE_RATE = 0
    MAX_EPOCHS = 100
    PINNED_CFG = 0.0

    def __init__(self, train_config, inference_config, logger):
        self._cfg = train_config
        self._infer = inference_config or {}
        self._logger = logger

    def check(self):
        cfg = self._cfg
        if cfg.get("lr") != self.PINNED_LR:
            raise ValueError(f"pinned P3 lr is {self.PINNED_LR}, got {cfg.get('lr')} (P2-equivalent recipe)")
        if cfg.get("batch_size") != self.PINNED_BATCH:
            raise ValueError(f"pinned P3 batch_size is {self.PINNED_BATCH}, got {cfg.get('batch_size')}")
        if cfg.get("weighted_loss") != self.PINNED_WEIGHTED_LOSS:
            raise ValueError(f"pinned P3 weighted_loss is {self.PINNED_WEIGHTED_LOSS}, got {cfg.get('weighted_loss')}")
        if cfg.get("weighted_loss_label") != self.PINNED_WEIGHTED_LABELS:
            raise ValueError(f"pinned P3 weighted_loss_label is {self.PINNED_WEIGHTED_LABELS}, got {cfg.get('weighted_loss_label')}")
        if cfg.get("use_region_contrasive_loss", False) is not False:
            raise ValueError("P3 recipe forbids use_region_contrasive_loss (must be OFF)")
        if cfg.get("cache_rate") != self.PINNED_CACHE_RATE:
            raise ValueError(f"pinned P3 cache_rate is {self.PINNED_CACHE_RATE}, got {cfg.get('cache_rate')}")
        if cfg.get("n_epochs", self.MAX_EPOCHS) > self.MAX_EPOCHS:
            raise ValueError(f"pinned P3 max n_epochs is {self.MAX_EPOCHS}, got {cfg.get('n_epochs')}")
        if self._infer.get("cfg_guidance_scale", 0.0) != self.PINNED_CFG:
            raise ValueError(
                f"P3 candidate is evaluated/selected with CFG OFF (cfg_guidance_scale=0); got {self._infer.get('cfg_guidance_scale')}"
            )
        self._logger.info(
            f"P3 recipe guard OK: lr={self.PINNED_LR} batch={self.PINNED_BATCH} weighted_loss={self.PINNED_WEIGHTED_LOSS}"
            f"@{self.PINNED_WEIGHTED_LABELS} RCL=off cfg={self.PINNED_CFG}"
        )
        return True


class P3DataCatalog:
    """The P3 image-conditioned training list (pure BraTS, no replay) — one record per ordered (src,tgt) pair."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        payload = json.loads(Path(self._args.json_data_list).read_text())["training"]
        self._logger.info(f"[data] p3 list: {len(payload)} entries from {self._args.json_data_list} (no replay)")
        for entry in payload:
            if "src_image" not in entry:
                raise ValueError(f"P3 list entry missing src-image condition 'src_image': {entry.get('case')}")
            if "src_modality" not in entry:
                raise ValueError(f"P3 list entry missing 'src_modality': {entry.get('case')}")
            if entry["src_modality"] == entry["modality"]:
                raise ValueError(f"P3 list entry must be src!=tgt: {entry.get('case')} src={entry['src_modality']}")
        return payload

    def file_records(self):
        records = []
        for entry in self.load_entries():
            image = os.path.join(self._args.data_base_dir, entry["image"])
            src_image = os.path.join(self._args.data_base_dir, entry["src_image"])
            label = os.path.join(self._args.data_base_dir, entry["label"])
            for path, what in ((src_image, "src-image latent"), (label, "tumour label")):
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{what} missing: {path} (entry {entry.get('sub')}:{entry.get('case')}); "
                        "run the phase encode/labels pipeline before training"
                    )
            records.append(
                {
                    "image": image,
                    "src_image": src_image,
                    "label": label,
                    "spacing": entry["spacing"],
                    "modality": entry["modality"],
                    "src_modality": entry["src_modality"],
                    "fold": entry["fold"],
                    "sub": entry["sub"],
                    "case": entry["case"],
                }
            )
        return records


class P3ControlNetJob:
    """The pinned P3 ControlNet-only loop (upstream-equivalent except the pinned deltas)."""

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank

    def build_loader(self):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (p3, no replay): {len(P3DataCatalog(args, self._logger).file_records())}")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_loader, _val_loader = prepare_p3_controlnet_json_dataloader(
            json_data_list=args.json_data_list,
            data_base_dir=args.data_base_dir,
            batch_size=args.controlnet_train["batch_size"],
            cache_rate=args.controlnet_train["cache_rate"],
            fold=args.controlnet_train["fold"],
            rank=self._local_rank,
            world_size=world_size,
            modality_mapping=args.modality_mapping,
        )
        # The val split (fold==fold) is DISCARDED: P3 selects its candidate by the
        # dev-eval sidecar, never by training/validation loss (spec #51 decision 7).
        return train_loader

    def load_models(self):
        args = self._args
        controlnet = define_instance(args, "controlnet_def").to(self._device)
        unet = define_instance(args, "diffusion_unet_def").to(self._device)
        # The DM-source checkpoint pickles MONAI meta-tensor globals alongside the
        # weights; allowlist them so weights_only stays enabled (trusted source).
        torch.serialization.add_safe_globals([monai.data.meta_tensor.MetaTensor, monai.utils.enums.TraceKeys])
        dm_ckpt = torch.load(args.trained_diffusion_path, map_location=self._device, weights_only=True)
        state = unet.load_state_dict(dm_ckpt["unet_state_dict"], strict=False)
        if state.missing_keys:
            raise ValueError(f"DM source checkpoint missing keys for frozen DM: {state.missing_keys}")
        if state.unexpected_keys:
            self._logger.warning(f"DM source checkpoint unexpected keys (ignored): {state.unexpected_keys}")
        # init ControlNet from the frozen P1-DM encoder/mid — NEVER warm-start from a P2 ControlNet.
        copy_model_state(controlnet, unet.state_dict())
        if dist.is_initialized():
            controlnet = DistributedDataParallel(controlnet, device_ids=[self._device], find_unused_parameters=True)
        scale_factor = float(dm_ckpt["scale_factor"])
        for p in unet.parameters():
            p.requires_grad = False
        unet.eval()
        controlnet.train()
        self._logger.info(f"DM frozen (requires_grad=False); ControlNet init from P1-DM encoder/mid -> {args.trained_diffusion_path}")
        self._logger.info(f"scale_factor reused from P1-DM checkpoint -> {scale_factor}")
        return controlnet, unet, scale_factor

    def train(self):
        args = self._args
        loader = self.build_loader()
        controlnet, unet, scale_factor = self.load_models()
        noise_scheduler = define_instance(args, "noise_scheduler")
        scale_tensor = torch.tensor(scale_factor, device=self._device)

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.controlnet_train["lr"])
        total_steps = (args.controlnet_train["n_epochs"] * len(loader.dataset)) / args.controlnet_train["batch_size"]
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)
        scaler = GradScaler("cuda")
        torch.set_float32_matmul_precision("highest")

        for epoch in range(args.controlnet_train["n_epochs"]):
            if self._stop_requested():
                self._logger.info(f"early-stop file present; halting before epoch {epoch + 1}")
                break
            self._train_one_epoch(epoch, controlnet, unet, loader, optimizer, lr_scheduler, scaler, scale_tensor, noise_scheduler)

        if dist.is_initialized():
            dist.destroy_process_group()

    def _stop_requested(self):
        return (Path(self._args.model_dir) / STOP_FILE).is_file()

    def _weighted_target(self, labels, images):
        """weights = 1, with weighted_loss on the tumour subregions {129,130,131}."""
        args = self._args
        if args.controlnet_train.get("weighted_loss", 1.0) <= 1.0:
            return None
        weights = torch.ones_like(images)
        roi = torch.zeros([images.shape[0], 1] + list(images.shape[2:]), device=self._device)
        interpolate_label = F.interpolate(labels.float(), size=images.shape[2:], mode="nearest").long()
        for label in args.controlnet_train["weighted_loss_label"]:
            roi[interpolate_label == label] = 1
        weights[roi.repeat(1, images.shape[1], 1, 1, 1) == 1] = args.controlnet_train["weighted_loss"]
        return weights

    def _train_one_epoch(self, epoch, controlnet, unet, loader, optimizer, lr_scheduler, scaler, scale_tensor, noise_scheduler):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"Epoch {epoch + 1}, lr {optimizer.param_groups[0]['lr']}.")
        iteration = 0
        loss_totals = torch.zeros(2, dtype=torch.float, device=self._device)
        for batch in loader:
            if self._stop_requested():
                self._logger.info(f"early-stop file present; halting mid-epoch {epoch + 1}")
                return
            iteration += 1
            images = batch["image"].to(self._device) * scale_tensor
            src_latent = batch["src_image"].to(self._device) * scale_tensor
            labels = batch["label"].to(self._device)
            if labels.shape[1] != 1:
                raise ValueError(f"expected labels [B,1,X,Y,Z], got {labels.shape}")
            spacing_tensor = batch["spacing"].to(self._device)
            modality_tensor = batch["modality"].to(self._device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16, enabled=args.amp):
                noise = torch.randn_like(images)
                timesteps = noise_scheduler.sample_timesteps(images)
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                # The ONLY structural change vs P2: condition on the 4ch src latent,
                # not the binarized mask. Labels never enter the condition.
                controlnet_cond = src_latent
                down, mid = controlnet(
                    x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond, class_labels=modality_tensor
                )
                model_output = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    spacing_tensor=spacing_tensor,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    class_labels=modality_tensor,
                )
                model_gt = images - noise
                weights = self._weighted_target(labels, images)
                if weights is not None:
                    loss = (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
                else:
                    loss = F.l1_loss(model_output.float(), model_gt.float())
            if args.amp and args.amp_dtype == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            loss_totals[0] += loss.item()
            loss_totals[1] += 1.0
            if self._local_rank == 0 and iteration % 50 == 0:
                self._logger.info(
                    f"[{str(datetime.now())[:19]}] epoch {epoch + 1}, iter {iteration}/{len(loader)}, "
                    f"loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.12f}."
                )
        if dist.is_initialized():
            dist.all_reduce(loss_totals, op=torch.distributed.ReduceOp.SUM)
        if self._local_rank == 0:
            self._save_checkpoint(epoch, controlnet, loss_totals, scale_tensor)

    def _save_checkpoint(self, epoch, controlnet, loss_totals, scale_tensor):
        path = Path(self._args.model_dir) / f"epoch_{epoch + 1}.pt"
        tmp = path.with_name(path.name + ".tmp")
        controlnet_state = controlnet.module.state_dict() if isinstance(controlnet, DistributedDataParallel) else controlnet.state_dict()
        ckpt = {
            "epoch": epoch + 1,
            "loss": (loss_totals[0] / loss_totals[1]).item(),
            "num_train_timesteps": self._args.noise_scheduler["num_train_timesteps"],
            "scale_factor": scale_tensor,
            "controlnet_state_dict": controlnet_state,
        }
        torch.save(ckpt, tmp)
        tmp.replace(path)
        (Path(self._args.model_dir) / "latest.json").write_text(
            json.dumps({"epoch": epoch + 1, "checkpoint": str(path)}) + "\n"
        )
        self._logger.info(f"epoch {epoch + 1} average loss: {(loss_totals[0] / loss_totals[1]).item():.4f} -> {path}")


class P3TrainProvenanceWriter:
    """Records what the P3 training run consumed (feeds the phase-run contract)."""

    def __init__(self, args, local_rank, logger):
        self._args = args
        self._local_rank = local_rank
        self._logger = logger

    def write(self, path):
        if self._local_rank != 0:
            return None
        provenance = {
            "written_utc": datetime.now(UTC).isoformat(),
            "script": str(Path(__file__).resolve()),
            "env_config": str(Path(self._args.env_config_path).resolve()),
            "model_config": str(Path(self._args.model_config_path).resolve()),
            "model_def": str(Path(self._args.model_def_path).resolve()),
            "data_list": self._args.json_data_list,
            "trained_diffusion_path": self._args.trained_diffusion_path,
            "replay": None,
            "hyperparameters": self._args.controlnet_train,
            "cfg_guidance_scale": self._args.diffusion_unet_inference.get("cfg_guidance_scale") if hasattr(self._args, "diffusion_unet_inference") else None,
            "amp_dtype": self._args.amp_dtype,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "torch_version": torch.__version__,
            "git_commit": self._git_commit(),
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(provenance, indent=2) + "\n")
        self._logger.info(f"train provenance -> {out}")
        return out

    @staticmethod
    def _git_commit():
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=str(Path(__file__).parent)
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--env_config_path", required=True)
    parser.add_argument("-c", "--model_config_path", required=True)
    parser.add_argument("-t", "--model_def_path", required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--data-list", default=None, help="p3_pairs.json (defaults to env json_data_list)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    args = parser.parse_args(argv)

    torchrun_world = int(os.environ["WORLD_SIZE"]) if os.environ.get("WORLD_SIZE") else None
    if torchrun_world is not None and torchrun_world != args.num_gpus:
        raise ValueError(f"--num_gpus {args.num_gpus} disagrees with torchrun WORLD_SIZE {torchrun_world}")

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path
    if args.data_list is not None:
        merged.json_data_list = args.data_list
    with open(merged.modality_mapping_path) as handle:
        merged.modality_mapping = json.load(handle)

    local_rank, _world, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("p3-finetune")
    if local_rank == 0:
        # A P3 run only ever reads the frozen P1-DM; it never loads a ControlNet
        # checkpoint (no warm-start from P2). Guard rejects one if present.
        if getattr(merged, "trained_controlnet_path", None) is not None:
            raise ValueError("P3 recipe forbids warm-starting from a ControlNet checkpoint (P1-DM init only)")
        P3RecipeGuard(
            merged.controlnet_train,
            merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else None,
            logger,
        ).check()
        Path(merged.model_dir).mkdir(parents=True, exist_ok=True)
        P3TrainProvenanceWriter(merged, local_rank, logger).write(Path(merged.model_dir) / "train_provenance.json")
    P3ControlNetJob(merged, device, logger, local_rank).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
