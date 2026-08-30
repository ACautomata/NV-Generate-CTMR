# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mask-conditioned ControlNet candidate training (issue #59, spec #51 decision 7, ADR-0007).

Mask-to-image candidate: a ControlNet-only bypass hung off the FROZEN P1-DM (the
registered DM source, ADR-0006). The DM and VAE are untouched; the ControlNet is
initialized from the frozen DM encoder/mid (``copy_model_state``) and learns the
mask -> spatial-layout mapping. Pinned hyperparameters (lr=1e-5, batch=1, <=100
epochs, AdamW, PolynomialLR power 2.0, L1, cache_rate=0, weighted_loss=100 on
129/130/131, use_region_contrasive_loss OFF) and pure BraTS (no MR-RATE replay).

Deltas against the upstream ``train_controlnet.py`` loop, all pinned:
- ``scale_factor`` is REUSED from the frozen P1-DM checkpoint (never recomputed;
  the mask recipe has no recompute sanity since the P1-DM already froze it);
- the training list is the #52 ``p2_mask_cond.json`` (fold=0 -> train side is
  fold!=0 as the shared bypass loader partitions; the val side is never
  constructed — the dev-eval sidecar selects the candidate, spec #51);
- checkpoints persist per epoch as ``epoch_<N>.pt`` (``controlnet_state_dict`` +
  ``scale_factor``) for the dev-eval sidecar and the contract selection;
- the loop polls ``<model_dir>/.early_stop`` at epoch boundaries so the
  pre-recorded early-stop rule (sidecar) can end the run;
- bf16 autocast is the default (DCU), fp32 fallback via ``--no_amp``.

The ControlNet is initialized from the frozen P1-DM encoder/mid and is NEVER
warm-started from a ControlNet checkpoint — only the P1-DM checkpoint is read.

Migrated from the retired mask finetune script entry (ticket 09, ADR-0015
§2): the domain kernel rides the shared ``PhaseHarness`` shell (checkpoint
publication via ``CheckpointRepository``); the CLI face is unchanged.  Per
ADR-0016 (issue #172) the single-batch training math runs as the domain
``DiffusionModel.train_step`` over a ``ControlNetBypass`` composition (the
runtime bypass object carries no checkpoint identity), and the runtime
precision strategy is injected as a ``GradientExecutor``.

Usage (CLI, torchrun spawn is derived by the ctmr launcher):
    ctmr generate mask train -e run/environment.json -c configs/config_brats_p2_train.json \
        -t configs/config_network_rflow.json
    # or directly under torchrun (same argv namespace):
    torchrun --nproc_per_node=7 -m ctmr.application.generation.mask.train ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ctmr.application.generation.mask.inference import binarize_labels
from ctmr.application.generation.train_loader import BypassTrainLoader
from ctmr.application.shell import PhaseHarness, TrainContext, TrainProvenanceWriter
from ctmr.application.train_cli import TrainCli
from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import TumourWeightedTarget
from ctmr.domain.recipe import MaskRecipeSpec
from ctmr.infrastructure.bypass_mounting import BypassMounting
from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.diff_model_setting import initialize_distributed, load_config, setup_logging


class DataCatalog:
    """The mask-conditioned training list (pure BraTS, no replay) — one record per (case, modality)."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        payload = json.loads(Path(self._args.json_data_list).read_text())["training"]
        self._logger.info(f"[data] mask list: {len(payload)} entries from {self._args.json_data_list} (no replay)")
        for entry in payload:
            if "label" not in entry:
                raise ValueError(f"mask list entry missing mask condition 'label': {entry.get('case')}")
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
                    "generate the required label before training (the phase-label prep pipeline retired to git "
                    "history in #143 pending the `ctmr data` family, ADR-0015)"
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


class TrainKernel:
    """Mask-conditioned kernel: mask data, frozen-DM ControlNet hook-up, weighted L1.

    The four-method ``PhaseTrainKernel`` boundary. Recipe values live here, not
    in the shell: AdamW + lr + PolynomialLR power 2.0 (ADR-0007). The hook-up
    itself is the shared ``BypassMounting`` collaborator -- the kernel injects
    only the recipe values and composes the domain entity from the mount.
    """

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank
        self._controlnet = None
        self._model = None
        self._weighted_target = TumourWeightedTarget(args.controlnet_train["weighted_loss"], args.controlnet_train["weighted_loss_label"])
        self._mounting = BypassMounting(args, device, logger)
        self._train_loader = BypassTrainLoader(load_keys=("image", "label"), companion_keys=("top_region_index", "bottom_region_index"))

    def build_loader(self):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (mask family, no replay): {len(DataCatalog(args, self._logger).file_records())}")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # The loader's contract is train-side only (spec #51 decision 7, BypassTrainLoader).
        return self._train_loader.build(
            json_data_list=args.json_data_list,
            data_base_dir=args.data_base_dir,
            batch_size=args.controlnet_train["batch_size"],
            cache_rate=args.controlnet_train["cache_rate"],
            fold=args.controlnet_train["fold"],
            rank=self._local_rank,
            world_size=world_size,
            modality_mapping=args.modality_mapping,
        )

    def load_models(self, loader):
        args = self._args
        mounted = self._mounting.mount(
            len(loader.dataset),
            lr=args.controlnet_train["lr"],
            n_epochs=args.controlnet_train["n_epochs"],
            batch_size=args.controlnet_train["batch_size"],
        )
        self._controlnet = mounted.trainable
        # The domain composition carries the training recipe: the frozen DM
        # (behaviour holder), the trainable bypass and the Adam + PolynomialLR
        # session members.  The shell's TrainContext keeps the same handles for
        # checkpoint scale payloads and the shared optimizer/scheduler instances.
        self._model = DiffusionModel(
            unet=mounted.dm,
            scale_factor=mounted.scale,
            noise_scheduler=mounted.noise_scheduler,
            bypass=ControlNetBypass(mounted.trainable),
            optimizer=mounted.optimizer,
            lr_scheduler=mounted.scheduler,
        )
        return TrainContext(
            trainable=mounted.trainable, optimizer=mounted.optimizer, scheduler=mounted.scheduler, scale=mounted.scale, device=self._device
        )

    def train_step(self, batch, gradient_executor):
        """The thin batch adapter: mask-binarize + weight-build, then the domain closed update.

        The single-batch training math (RF timesteps, noise, the bypass-conditioned
        forward and the weighted velocity L1) is the domain ``DiffusionModel.train_step``
        over the frozen DM + ``ControlNetBypass`` composition (ADR-0016, issue #172).
        """
        images = batch["image"].to(self._device)
        labels = batch["label"].to(self._device)
        if labels.shape[1] != 1:
            raise ValueError(f"expected labels [B,1,X,Y,Z], got {labels.shape}")
        spacing_tensor = batch["spacing"].to(self._device)
        modality_tensor = batch["modality"].to(self._device)
        # The ONLY structural difference vs the cross-modal family: condition on
        # the binarized 8ch mask, not the src-image latent.
        controlnet_cond = binarize_labels(labels.as_tensor().to(torch.long)).float()
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
        return self._mounting.checkpoint_payload(self._controlnet, epoch, avg_loss, scale)


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
    logger = setup_logging("mask-finetune")
    kernel = TrainKernel(merged, device, logger, local_rank)
    # The application injects the runtime precision strategy (ADR-0016): fp16
    # (scaler), bf16 (DCU default) or non-AMP plain execution.
    if args.amp and args.amp_dtype == "fp16":
        gradient_executor = Fp16GradientExecutor()
    elif args.amp:
        gradient_executor = Bf16GradientExecutor()
    else:
        gradient_executor = PlainGradientExecutor()
    return PhaseHarness(
        kernel=kernel,
        model_dir=merged.model_dir,
        n_epochs=merged.controlnet_train["n_epochs"],
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        local_rank=local_rank,
        logger=logger,
        recipe_check=MaskRecipeSpec(merged.controlnet_train, logger).check,
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
        gradient_executor=gradient_executor,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
