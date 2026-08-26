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

"""P2 mask-conditioned ControlNet candidate training (issue #59, spec #51 decision 7).

Mask-to-image candidate: a ControlNet-only bypass hung off the FROZEN P1-DM (the
registered DM source, ADR-0006). The DM and VAE are untouched; the ControlNet is
initialized from the frozen DM encoder/mid (``copy_model_state``) and learns the
mask -> spatial-layout mapping. Pinned hyperparameters (lr=1e-5, batch=1, <=100
epochs, AdamW, PolynomialLR power 2.0, L1, cache_rate=0, weighted_loss=100 on
129/130/131, use_region_contrasive_loss OFF) and pure BraTS (no MR-RATE replay).

Deltas against the upstream ``train_controlnet.py`` loop, all pinned:
- ``scale_factor`` is REUSED from the frozen P1-DM checkpoint (never recomputed;
  the P2 recipe has no recompute sanity since the P1-DM already froze it);
- the training list is the #52 ``p2_mask_cond.json`` (fold=0 -> train side is
  fold!=0 as the MAISI loader partitions; the val loader is DISCARDED, never used
  to select a checkpoint — the dev-eval sidecar does the selection, spec #51);
- checkpoints persist per epoch as ``epoch_<N>.pt`` (``controlnet_state_dict`` +
  ``scale_factor``) for the dev-eval sidecar and the contract selection;
- the loop polls ``<model_dir>/.early_stop`` at epoch boundaries so the
  pre-recorded early-stop rule (sidecar) can end the run;
- bf16 autocast is the default (DCU), fp32 fallback via ``--no_amp``.

Thin entry since the harness consolidation (ADR-0011, #111): the domain kernel
(``P2TrainKernel``, four-method injection) rides the shared
``ctmr.harness.PhaseHarness`` shell; the CLI face is unchanged.

Usage (torchrun on the DCU):
    torchrun --nproc_per_node=7 -m scripts.brats_p2_finetune \
        -e run/environment.json -c configs/config_brats_p2_train.json \
        -t configs/config_network_rflow.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import monai
import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from torch.nn.parallel import DistributedDataParallel

from ctmr.harness.cli import TrainCli
from ctmr.harness.recipe import P2RecipeSpec
from ctmr.harness.train_shell import PhaseHarness, TrainContext, TrainProvenanceWriter

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import binarize_labels, define_instance, prepare_maisi_controlnet_json_dataloader


class P2DataCatalog:
    """The P2 mask-conditioned training list (pure BraTS, no replay) — one record per (case, modality)."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        payload = json.loads(Path(self._args.json_data_list).read_text())["training"]
        self._logger.info(f"[data] p2 list: {len(payload)} entries from {self._args.json_data_list} (no replay)")
        for entry in payload:
            if "label" not in entry:
                raise ValueError(f"P2 list entry missing mask condition 'label': {entry.get('case')}")
        return payload

    def file_records(self):
        """Maps list entries to {image, label, spacing, modality} loader records (MAISI layout)."""
        records = []
        for entry in self.load_entries():
            image = os.path.join(self._args.data_base_dir, entry["image"])
            label = os.path.join(self._args.data_base_dir, entry["label"])
            if not os.path.exists(label):
                raise FileNotFoundError(
                    f"mask condition missing: {label} (entry {entry.get('sub')}:{entry.get('case')}); "
                    "run the phase labels pipeline before training"
                )
            records.append(
                {
                    "image": image,
                    "label": label,
                    "spacing": entry["spacing"],
                    "modality": entry["modality"],
                    "fold": entry["fold"],
                    "sub": entry["sub"],
                    "case": entry["case"],
                }
            )
        return records


class P2TrainKernel:
    """P2 kernel: mask-conditioned data, frozen-DM ControlNet hook-up, weighted L1.

    The four-method ``PhaseTrainKernel`` boundary. Recipe values live here, not
    in the shell: AdamW + lr + PolynomialLR power 2.0 (ADR-0007).
    """

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank
        self._controlnet = None
        self._unet = None
        self._scale_factor = None
        self._noise_scheduler = None

    def build_loader(self):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (p2, no replay): {len(P2DataCatalog(args, self._logger).file_records())}")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_loader, _val_loader = prepare_maisi_controlnet_json_dataloader(
            json_data_list=args.json_data_list,
            data_base_dir=args.data_base_dir,
            batch_size=args.controlnet_train["batch_size"],
            cache_rate=args.controlnet_train["cache_rate"],
            fold=args.controlnet_train["fold"],
            rank=self._local_rank,
            world_size=world_size,
            modality_mapping=args.modality_mapping,
        )
        # The val split (fold==fold) is DISCARDED: P2 selects its candidate by the
        # dev-eval sidecar, never by training/validation loss (spec #51 decision 7).
        return train_loader

    def load_models(self, loader):
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
        # init ControlNet from the frozen DM encoder/mid (spec #51 decision 7 / ADR-0007).
        copy_model_state(controlnet, unet.state_dict())
        # Only the ControlNet is trained; the DM stays a frozen, non-DDP module
        # (MAISI convention: the trainable bypass is DDP-wrapped, the frozen DM is not).
        if dist.is_initialized():
            controlnet = DistributedDataParallel(controlnet, device_ids=[self._device], find_unused_parameters=True)
        scale_factor = float(dm_ckpt["scale_factor"])
        for p in unet.parameters():
            p.requires_grad = False
        unet.eval()
        controlnet.train()
        self._logger.info(f"DM frozen (requires_grad=False); ControlNet init from DM encoder/mid -> {args.trained_diffusion_path}")
        self._logger.info(f"scale_factor reused from P1-DM checkpoint -> {scale_factor}")
        scale_tensor = torch.tensor(scale_factor, device=self._device)

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.controlnet_train["lr"])
        total_steps = (args.controlnet_train["n_epochs"] * len(loader.dataset)) / args.controlnet_train["batch_size"]
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

        self._controlnet = controlnet
        self._unet = unet
        self._scale_factor = scale_tensor
        self._noise_scheduler = define_instance(args, "noise_scheduler")
        return TrainContext(trainable=controlnet, optimizer=optimizer, scheduler=lr_scheduler, scale=scale_tensor, device=self._device)

    def _weighted_target(self, labels, images):
        """weights = 1, with weighted_loss on the tumor subregions {129,130,131}."""
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

    def train_batch(self, batch):
        images = batch["image"].to(self._device) * self._scale_factor
        labels = batch["label"].to(self._device)
        if labels.shape[1] != 1:
            raise ValueError(f"expected labels [B,1,X,Y,Z], got {labels.shape}")
        spacing_tensor = batch["spacing"].to(self._device)
        modality_tensor = batch["modality"].to(self._device)
        noise = torch.randn_like(images)
        timesteps = self._noise_scheduler.sample_timesteps(images)
        noisy_latent = self._noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
        controlnet_cond = binarize_labels(labels.as_tensor().to(torch.long)).float()
        down, mid = self._controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond, class_labels=modality_tensor)
        model_output = self._unet(
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
            return (F.l1_loss(model_output.float(), model_gt.float(), reduction="none") * weights).mean()
        return F.l1_loss(model_output.float(), model_gt.float())

    def checkpoint_payload(self, epoch, avg_loss, scale):
        controlnet_state = (
            self._controlnet.module.state_dict() if isinstance(self._controlnet, DistributedDataParallel) else self._controlnet.state_dict()
        )
        return {
            "epoch": epoch,
            "loss": avg_loss,
            "num_train_timesteps": self._args.noise_scheduler["num_train_timesteps"],
            "scale_factor": scale,
            "controlnet_state_dict": controlnet_state,
        }


def main(argv=None):
    args = TrainCli(__doc__, stage="p2").parse(argv)

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path
    with open(merged.modality_mapping_path) as handle:
        merged.modality_mapping = json.load(handle)

    local_rank, _world, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("p2-finetune")
    kernel = P2TrainKernel(merged, device, logger, local_rank)
    return PhaseHarness(
        kernel=kernel,
        model_dir=merged.model_dir,
        n_epochs=merged.controlnet_train["n_epochs"],
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        local_rank=local_rank,
        logger=logger,
        recipe_check=P2RecipeSpec(merged.controlnet_train, logger).check,
        provenance=TrainProvenanceWriter(
            merged,
            local_rank,
            logger,
            domain_fields=lambda: {
                "data_list": merged.json_data_list,
                "trained_diffusion_path": merged.trained_diffusion_path,
                "replay": None,
                "hyperparameters": merged.controlnet_train,
            },
            script_path=Path(__file__),
        ),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
