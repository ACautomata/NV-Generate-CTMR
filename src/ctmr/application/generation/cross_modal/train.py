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
  val side is never constructed — dev-eval selects the candidate, spec #51
  decision 7);
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

Layering (ADR-0019 §1-§3, issue #274): the module depends only on domain ports
and injected collaborators -- the engine adapter, the distributed session, the
run logger, the ControlNet mounting and the precision executor are assembled by
the composition root (``ctmr.wiring.generate``, which the main entry consults
directly: the torchrun worker reuses the same assembly).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch.distributed as dist

from ctmr.application.generation.train_loader import BypassTrainLoader
from ctmr.application.shell import PhaseHarness, TrainContext, TrainProvenanceWriter
from ctmr.application.train_cli import TrainCli
from ctmr.domain.generation.bypass import ControlNetBypass
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import TumourWeightedTarget
from ctmr.domain.recipe import CrossModalRecipeSpec
from ctmr.wiring.generate import GenerateRuntime


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
    The hook-up itself is the injected ``BypassMounting`` collaborator (assembled
    by the composition root, ADR-0019 §2) -- the kernel injects only the recipe
    values and composes the domain entity from the mount.
    """

    def __init__(self, args, device, logger, local_rank, mounting):
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
        self._mounting = mounting
        self._train_loader = BypassTrainLoader(load_keys=("image", "label", "src_image"), join_keys=("src_image",))

    def build_loader(self):
        args = self._args
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (cross-modal family, no replay): {len(DataCatalog(args, self._logger).file_records())}")
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
        # The domain composition carries the training recipe: the frozen P1-DM
        # (behaviour holder), the trainable image-conditioned bypass and the
        # Adam + PolynomialLR session members (ADR-0016, issue #174).  The
        # shell's TrainContext keeps the same handles for checkpoint scale
        # payloads and the shared optimizer/scheduler instances.
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
        return self._mounting.checkpoint_payload(self._controlnet, epoch, avg_loss, scale)


def main(argv=None):
    args = TrainCli(__doc__, stage="p3").parse(argv)

    # The composition root assembles every concrete collaborator (ADR-0019 §2):
    # the engine adapter behind the GenerationEngine port, the distributed
    # session, the run logger, the ControlNet mounting and the precision executor.
    runtime = GenerateRuntime()
    engine = runtime.engine()
    merged = engine.load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path
    if args.data_list is not None:
        merged.json_data_list = args.data_list
    with open(merged.modality_mapping_path) as handle:
        merged.modality_mapping = json.load(handle)

    local_rank, _world, device = runtime.train_session(args)
    logger = runtime.logger("cross-modal-finetune")
    kernel = TrainKernel(merged, device, logger, local_rank, mounting=runtime.bypass_mounting(merged, device, logger))
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
        gradient_executor=runtime.gradient_executor(args.amp, args.amp_dtype),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
