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

"""Modality-label-conditioned full-parameter DM continuation (issue #57, spec #51 decision 6).

Full-parameter DM continuation of the frozen rflow-mr-brain v1 checkpoint with
the VAE untouched, pinned hyperparameters (lr=2e-6, batch=1, <=100 epochs,
Rectified Flow uniform timestep scale 1.4, PolynomialLR power 2.0, L1 loss,
augment_modality_label prob 0.1) and the BraTS : MR-RATE 1:1 replay mix
(spec #51 / issue #10 resolution).

Deltas against the retired vendored upstream training driver
(``diff_model_train.py``, git history; deleted with issue #175), all pinned:
- ``scale_factor`` is REUSED from the base checkpoint (never recomputed); the
  recomputed 1/std(z) of the first batch is logged and asserted against it as
  a sanity check (issue #10 §7);
- the training list is the concatenation of the BraTS train list
  (env ``json_data_list``) and the MR-RATE replay list(s) (``--replay-list``);
- checkpoints persist per epoch as ``epoch_<N>.pt`` (upstream key layout) for
  the dev-eval sidecar and the contract selection;
- the loop polls ``<model_dir>/.early_stop`` at epoch boundaries so the
  pre-recorded early-stop rule (sidecar) can end the run without a kill;
- bf16 autocast is the default (DCU), fp32 fallback via --no_amp.

Migrated from the retired modality-label finetune script entry (ticket 10,
ADR-0015 §2): the domain kernel (``TrainKernel``) rides the shared
``PhaseHarness`` shell; per ADR-0016 the single-batch training math is the
domain ``DiffusionModel.train_step`` and the runtime precision strategy is
injected by the application as a ``GradientExecutor``. The CLI face is
unchanged.

Usage (CLI, torchrun spawn is derived by the ctmr launcher):
    ctmr generate modality-label train -e run/environment.json -c configs/config_brats_p1_train.json \
        -t configs/config_network_rflow.json --replay-list run/lists/p1_mrrate_replay.json
    # or directly under torchrun (same argv namespace):
    torchrun --nproc_per_node=7 -m ctmr.application.generation.modality_label.train ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.transforms import Compose
from torch.nn.parallel import DistributedDataParallel

from ctmr.application.shell import PhaseHarness, TrainContext, TrainProvenanceWriter
from ctmr.application.train_cli import TrainCli
from ctmr.domain.generation.model import DiffusionModel
from ctmr.domain.generation.objective import ModalityLabelPerturber
from ctmr.domain.recipe import P1RecipeSpec
from ctmr.infrastructure.bypass_mounting import MonaiCheckpoint
from ctmr.infrastructure.gradient_executors import Bf16GradientExecutor, Fp16GradientExecutor, PlainGradientExecutor
from ctmr.infrastructure.maisi_engine.diff_model_setting import initialize_distributed, load_config, setup_logging
from ctmr.infrastructure.maisi_engine.instance_definition import define_instance

SCALE_FACTOR_RELATIVE_TOLERANCE = 0.5  # issue #10 §7: sanity assert, not a re-pin


class DataCatalog:
    """The 1:1 BraTS + MR-RATE replay training list (spec #51 decision 6)."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        entries = []
        counts = []
        for label, path in (("brats train list", self._args.json_data_list), *(("replay list", p) for p in self._args.replay_list)):
            payload = json.loads(Path(path).read_text())["training"]
            self._logger.info(f"[data] {label}: {len(payload)} entries from {path}")
            counts.append(len(payload))
            entries += payload
        if counts and len(counts) > 1 and counts[0] != counts[1]:
            raise ValueError(
                f"1:1 replay mix violated: brats train {counts[0]} vs replay {counts[1]} (spec #51 decision 6 requires strict 1:1 mixing)"
            )
        return entries

    def file_records(self):
        """Maps list entries to {image, spacing, modality} loader records (upstream layout)."""
        records = []
        for entry in self.load_entries():
            emb = os.path.join(self._args.embedding_base_dir, entry["image"].replace(".nii.gz", "_emb.nii.gz"))
            if not os.path.exists(emb):
                raise FileNotFoundError(
                    f"training embedding missing: {emb} (entry {entry.get('sub')}:{entry.get('case')}); "
                    "VAE-encode the training data first (the phase/replay prep+encode pipelines retired to "
                    "git history in #143 pending the `ctmr data` family, ADR-0015)"
                )
            info = emb + ".json"
            records.append({"image": emb, "spacing": info, "modality": info})
        return records


class ScaleFactorPolicy:
    """Reuses the checkpoint scale_factor (issue #10 §7); recomputation is a sanity check only."""

    def __init__(self, base_ckpt_scale, logger):
        self._checkpoint_value = float(base_ckpt_scale)
        self._logger = logger

    def value(self):
        return self._checkpoint_value

    def sanity_check(self, recomputed, device):
        if dist.is_initialized():
            recomputed = recomputed.clone()
            dist.all_reduce(recomputed, op=torch.distributed.ReduceOp.AVG)
        recomputed_value = float(recomputed)
        relative = abs(recomputed_value - self._checkpoint_value) / self._checkpoint_value
        self._logger.info(
            f"scale_factor sanity: checkpoint={self._checkpoint_value:.6f} recomputed_1/std(z)={recomputed_value:.6f} relative_diff={relative:.4f}"
        )
        if relative > SCALE_FACTOR_RELATIVE_TOLERANCE:
            raise ValueError(
                f"scale_factor sanity assert failed: checkpoint {self._checkpoint_value} vs recomputed "
                f"{recomputed_value} (relative {relative:.3f} > {SCALE_FACTOR_RELATIVE_TOLERANCE})"
            )


class TrainKernel:
    """Modality-label kernel: data composition, DiffusionModel hook-up, payload keys.

    The four-method (``train_step``) ``PhaseTrainKernel`` boundary. Recipe values
    live here, not in the shell: Adam + lr + PolynomialLR power 2.0 (ADR-0005).
    The single-batch training math (modality perturbation, RF timesteps, noise,
    L1 against the velocity target, one parameter update) is the domain
    ``DiffusionModel.train_step``; the shell injects the runtime precision
    strategy via ``GradientExecutor`` (ADR-0016).
    """

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank
        self._unet = None
        self._model = None

    def build_loader(self):
        args = self._args
        catalog = DataCatalog(args, self._logger)
        records = catalog.file_records()
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (brats + replay): {len(records)}")
        if dist.is_initialized():
            records = partition_dataset(data=records, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[self._local_rank]
        transforms = Compose(
            [
                monai.transforms.LoadImaged(keys=["image"]),
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.Lambdad(keys="spacing", func=lambda x: self._load_json_field(x, "spacing")),
                monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
                monai.transforms.Lambdad(keys="modality", func=lambda x: self._token_of(x)),
                monai.transforms.EnsureTyped(keys=["modality"], dtype=torch.long),
            ]
        )
        dataset = monai.data.CacheDataset(data=records, transform=transforms, cache_rate=args.diffusion_unet_train["cache_rate"], num_workers=2)
        return DataLoader(dataset, num_workers=6, batch_size=args.diffusion_unet_train["batch_size"], shuffle=True)

    @staticmethod
    def _load_json_field(path, key):
        with open(path) as handle:
            return torch.FloatTensor(json.load(handle)[key])

    def _token_of(self, path):
        with open(path) as handle:
            return self._args.modality_mapping[json.load(handle)["modality"]]

    def load_unet(self):
        args = self._args
        unet = define_instance(args, "diffusion_unet_def").to(self._device)
        unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
        if dist.is_initialized():
            unet = DistributedDataParallel(unet, device_ids=[self._device], find_unused_parameters=True)
        # The allowlisted weights_only load of the base checkpoint (the one
        # safe_globals point shared with the P2/P3 DM-source hook-up).
        checkpoint = MonaiCheckpoint(args.existing_ckpt_filepath, self._device).load()
        target = unet.module if dist.is_initialized() else unet
        state = target.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        if state.missing_keys:
            raise ValueError(f"base checkpoint missing keys for full-parameter continuation: {state.missing_keys}")
        if state.unexpected_keys:
            self._logger.warning(f"base checkpoint unexpected keys (ignored): {state.unexpected_keys}")
        self._logger.info(f"base checkpoint loaded (full-param continuation): {args.existing_ckpt_filepath}")
        return unet, ScaleFactorPolicy(checkpoint["scale_factor"], self._logger)

    def load_models(self, loader):
        args = self._args
        unet, scale_policy = self.load_unet()
        noise_scheduler = define_instance(args, "noise_scheduler")

        with open(args.modality_mapping_path) as handle:
            args.modality_mapping = json.load(handle)

        # Recompute 1/std(z) on the first batch only as the sanity check; the
        # training scale_factor stays the checkpoint value (issue #10 §7).
        first = next(iter(DataLoader(loader.dataset, num_workers=0, batch_size=1)))
        recomputed = 1 / torch.std(first["image"].to(self._device))
        scale_policy.sanity_check(recomputed, self._device)
        scale_factor = torch.tensor(scale_policy.value(), device=self._device)

        optimizer = torch.optim.Adam(unet.parameters(), lr=args.diffusion_unet_train["lr"])
        total_steps = (args.diffusion_unet_train["n_epochs"] * len(loader.dataset)) / args.diffusion_unet_train["batch_size"]
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

        self._unet = unet
        # The domain entity carries the training recipe: the modality perturber,
        # the RF scheduler shape and the Adam + PolynomialLR session members.
        # The shell's TrainContext keeps the same handles for checkpoint scale
        # payloads and the shared (single) optimizer/scheduler instances.
        self._model = DiffusionModel(
            unet=unet,
            scale_factor=scale_factor,
            noise_scheduler=noise_scheduler,
            perturber=ModalityLabelPerturber(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        return TrainContext(trainable=unet, optimizer=optimizer, scheduler=lr_scheduler, scale=scale_factor, device=self._device)

    def train_step(self, batch, gradient_executor):
        images = batch["image"].to(self._device)
        spacing = batch["spacing"].to(self._device)
        modality = batch["modality"].to(self._device)
        return self._model.train_step(images, spacing, modality, gradient_executor)

    def checkpoint_payload(self, epoch, avg_loss, scale):
        unet_module = self._unet.module if isinstance(self._unet, DistributedDataParallel) else self._unet
        return {
            "epoch": epoch,
            "loss": avg_loss,
            "num_train_timesteps": self._args.noise_scheduler["num_train_timesteps"],
            "scale_factor": scale,
            "unet_state_dict": unet_module.state_dict(),
        }


def main(argv=None):
    args = TrainCli(__doc__, stage="p1").parse(argv)

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.replay_list = args.replay_list
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path

    local_rank, _world, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("modality-label-finetune")
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
        n_epochs=merged.diffusion_unet_train["n_epochs"],
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        local_rank=local_rank,
        logger=logger,
        recipe_check=P1RecipeSpec(merged.diffusion_unet_train, merged.noise_scheduler, logger).check,
        provenance=TrainProvenanceWriter(
            merged,
            local_rank,
            logger,
            domain_fields=lambda: {
                "data_lists": {"brats_train": merged.json_data_list, "replay": list(merged.replay_list)},
                "base_ckpt": merged.existing_ckpt_filepath,
                "hyperparameters": merged.diffusion_unet_train,
            },
            script_path=Path(__file__),
        ),
        gradient_executor=gradient_executor,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
