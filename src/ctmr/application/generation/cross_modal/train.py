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

"""Cross-modal (image-conditioned) ControlNet candidate training (issue #61, spec #51 decision 4/6/8).

Inter-modality candidate: a ControlNet-only bypass hung off the FROZEN P1-DM (the
registered DM source, ADR-0006). The DM and VAE are untouched; the ControlNet
conditions on the 4ch **src-image latent** (``src_image`` in the ``p3_pairs.json``
list, no mask) and the target modality label rides the existing ``class_labels``
path into both DM and ControlNet — the training-side change of the issue #12 §7
checklist turned into a reusable recipe. Pinned hyperparameters are exactly the
mask recipe (lr=1e-5, batch=1, <=100 epochs, AdamW, PolynomialLR power 2.0, L1,
cache_rate=0, weighted_loss=100 on 129/130/131, use_region_contrasive_loss OFF,
pure BraTS no MR-RATE replay) plus CFG=0 semantics. The ControlNet is initialized
from the frozen P1-DM encoder/mid (``copy_model_state``) and is NEVER warm-started
from a mask ControlNet — only the P1-DM checkpoint is read.

Deltas against the mask family, all pinned:
- ``controlnet_cond`` is the src-image latent (``src_image``, 4ch) instead of the
  binarized 8ch mask (``binarize_labels``); labels only enter the weighted loss;
- the training list is the #52 ``p3_pairs.json`` (fold=1 train / fold=0 dev; the
  val split is DISCARDED — dev-eval selects the candidate, spec #51 decision 7);
- ``CrossModalRecipeSpec`` additionally pins cfg_guidance_scale=0 (the candidate is
  evaluated and selected with CFG off) and refuses to load a ControlNet checkpoint;
- bf16 autocast default (DCU), fp32 fallback via ``--no_amp``.

Migrated from the retired cross-modal finetune script entry (ticket 08, ADR-0015
§2): the domain kernel (``TrainKernel``, four-method injection) rides the shared
``PhaseHarness`` shell; the CLI face is unchanged.  Per ADR-0016 (issue #174) the
single-batch training math runs as the domain ``DiffusionModel`` +
``ControlNetBypass`` composition (the 4ch src latent rides the ``controlnet_cond``
slot as the bypass condition; labels only enter ``TumourWeightedTarget``); the
application keeps the condition/weight build and the runtime precision strategy
injection.

Usage (CLI, torchrun spawn is derived by the ctmr launcher):
    ctmr generate cross-modal train -e run/environment.json -c configs/config_brats_p3_train.json \
        -t configs/config_network_p3.json --data-list runs/p3/.../p3_pairs.json
    # or directly under torchrun (same argv namespace):
    torchrun --nproc_per_node=7 -m ctmr.application.generation.cross_modal.train ...
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import CacheDataset, partition_dataset
from monai.networks.utils import copy_model_state
from monai.transforms import Compose, EnsureTyped, Lambdad, LoadImaged, Orientationd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ctmr.application.shell import PhaseHarness, TrainContext, TrainProvenanceWriter
from ctmr.application.train_cli import TrainCli
from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import TumourWeightedTarget
from ctmr.domain.recipe import CrossModalRecipeSpec
from ctmr.infrastructure.dataio.list_assembly import add_data_dir2path
from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.diff_model_setting import initialize_distributed, load_config, setup_logging
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance


def prepare_controlnet_json_dataloader(
    json_data_list,
    data_base_dir,
    batch_size=1,
    fold=0,
    cache_rate=0.0,
    rank=0,
    world_size=1,
    modality_mapping=None,
):
    """cross-modal dataloader: image (tgt latent), label (loss-only tumour), src_image (4ch src latent).

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


class DataCatalog:
    """The image-conditioned training list (pure BraTS, no replay) — one record per ordered (src,tgt) pair."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        payload = json.loads(Path(self._args.json_data_list).read_text())["training"]
        self._logger.info(f"[data] cross-modal list: {len(payload)} entries from {self._args.json_data_list} (no replay)")
        for entry in payload:
            if "src_image" not in entry:
                raise ValueError(f"cross-modal list entry missing src-image condition 'src_image': {entry.get('case')}")
            if "src_modality" not in entry:
                raise ValueError(f"cross-modal list entry missing 'src_modality': {entry.get('case')}")
            if entry["src_modality"] == entry["modality"]:
                raise ValueError(f"cross-modal list entry must be src!=tgt: {entry.get('case')} src={entry['src_modality']}")
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
                        f"{what} missing: {path} (entry {entry.get('sub')}:{entry.get('case')}); run the phase encode/labels pipeline before training"
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


class TrainKernel:
    """Image-conditioned kernel: src-image data, frozen-DM ControlNet hook-up, weighted L1, CFG=0.

    The four-method ``PhaseTrainKernel`` boundary. Recipe values live here, not
    in the shell: AdamW + lr + PolynomialLR power 2.0 (mask-equivalent recipe).
    """

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank
        self._controlnet = None
        self._model = None
        # The pinned tumour-region weighting: the domain definition the P2
        # migration introduced (TumourWeightedTarget), driven by the same
        # recipe values (weighted_loss=100 on {129,130,131}).
        self._weighted_target = TumourWeightedTarget(args.controlnet_train["weighted_loss"], args.controlnet_train["weighted_loss_label"])

    def build_loader(self):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (cross-modal family, no replay): {len(DataCatalog(args, self._logger).file_records())}")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        train_loader, _val_loader = prepare_controlnet_json_dataloader(
            json_data_list=args.json_data_list,
            data_base_dir=args.data_base_dir,
            batch_size=args.controlnet_train["batch_size"],
            cache_rate=args.controlnet_train["cache_rate"],
            fold=args.controlnet_train["fold"],
            rank=self._local_rank,
            world_size=world_size,
            modality_mapping=args.modality_mapping,
        )
        # The val split (fold==fold) is DISCARDED: this family selects its
        # candidate by the dev-eval sidecar, never by training/validation loss
        # (spec #51 decision 7).
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
        # init ControlNet from the frozen P1-DM encoder/mid — NEVER warm-start from a mask ControlNet.
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
        scale_tensor = torch.tensor(scale_factor, device=self._device)

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.controlnet_train["lr"])
        total_steps = (args.controlnet_train["n_epochs"] * len(loader.dataset)) / args.controlnet_train["batch_size"]
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

        self._controlnet = controlnet
        # The domain composition carries the training recipe: the frozen P1-DM
        # (behaviour holder), the trainable image-conditioned bypass and the
        # Adam + PolynomialLR session members (ADR-0016, issue #174).  The
        # shell's TrainContext keeps the same handles for checkpoint scale
        # payloads and the shared optimizer/scheduler instances.
        self._model = DiffusionModel(
            unet=unet,
            scale_factor=scale_tensor,
            noise_scheduler=define_instance(args, "noise_scheduler"),
            bypass=ControlNetBypass(controlnet),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        return TrainContext(trainable=controlnet, optimizer=optimizer, scheduler=lr_scheduler, scale=scale_tensor, device=self._device)

    def train_step(self, batch, gradient_executor):
        """The thin batch adapter: scaled-src-latent condition + weight build, then the domain closed update.

        The single-batch training math (RF timesteps, noise, the image-conditioned
        bypass forward and the weighted velocity L1) is the domain
        ``DiffusionModel.train_step`` over the frozen P1-DM + ``ControlNetBypass``
        composition (ADR-0016, issue #174).  The ``controlnet_cond`` slot is the
        bypass condition: the scaled 4ch src latent -- never a binarized mask.
        """
        images = batch["image"].to(self._device)
        src_latent = batch["src_image"].to(self._device)
        labels = batch["label"].to(self._device)
        if labels.shape[1] != 1:
            raise ValueError(f"expected labels [B,1,X,Y,Z], got {labels.shape}")
        spacing_tensor = batch["spacing"].to(self._device)
        modality_tensor = batch["modality"].to(self._device)
        # The ONLY structural difference vs the mask family: condition on the
        # scaled 4ch src latent, not the binarized 8ch mask.  Labels never
        # enter the condition -- they only ride TumourWeightedTarget.
        controlnet_cond = src_latent * self._model.scale_factor
        weights = self._weighted_target.weights(labels, images)
        return self._model.train_step(
            images,
            spacing_tensor,
            modality_tensor,
            gradient_executor,
            controlnet_cond=controlnet_cond,
            target_weights=weights,
        )

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
    args = TrainCli(__doc__, stage="p3").parse(argv)

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
    logger = setup_logging("cross-modal-finetune")
    kernel = TrainKernel(merged, device, logger, local_rank)
    # The application injects the runtime precision strategy (ADR-0016): fp16
    # (scaler), bf16 (DCU default) or non-AMP plain execution.
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = Bf16GradientExecutor()
    else:
        gradient_executor = PlainGradientExecutor()
    infer_cfg = merged.diffusion_unet_inference if hasattr(merged, "diffusion_unet_inference") else None
    return PhaseHarness(
        kernel=kernel,
        model_dir=merged.model_dir,
        n_epochs=merged.controlnet_train["n_epochs"],
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        local_rank=local_rank,
        logger=logger,
        recipe_check=CrossModalRecipeSpec(
            merged.controlnet_train,
            infer_cfg,
            logger,
            trained_controlnet_path=getattr(merged, "trained_controlnet_path", None),
        ).check,
        provenance=TrainProvenanceWriter(
            merged,
            local_rank,
            logger,
            domain_fields=lambda: {
                "data_list": merged.json_data_list,
                "trained_diffusion_path": merged.trained_diffusion_path,
                "replay": None,
                "hyperparameters": merged.controlnet_train,
                "cfg_guidance_scale": merged.diffusion_unet_inference.get("cfg_guidance_scale")
                if hasattr(merged, "diffusion_unet_inference")
                else None,
            },
            script_path=Path(__file__),
        ),
        gradient_executor=gradient_executor,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
