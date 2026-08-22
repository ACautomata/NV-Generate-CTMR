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

"""P1 image-only tumour candidate training (issue #57, spec #51 decision 6).

Full-parameter DM continuation of the frozen rflow-mr-brain v1 checkpoint with
the VAE untouched, pinned hyperparameters (lr=2e-6, batch=1, <=100 epochs,
Rectified Flow uniform timestep scale 1.4, PolynomialLR power 2.0, L1 loss,
augment_modality_label prob 0.1) and the BraTS : MR-RATE 1:1 replay mix
(spec #51 / issue #10 resolution).

Deltas against the upstream ``diff_model_train.py`` loop, all pinned:
- ``scale_factor`` is REUSED from the base checkpoint (never recomputed); the
  recomputed 1/std(z) of the first batch is logged and asserted against it as
  a sanity check (issue #10 §7);
- the training list is the concatenation of the BraTS P1 train list
  (env ``json_data_list``) and the MR-RATE replay list(s) (``--replay-list``);
- checkpoints persist per epoch as ``epoch_<N>.pt`` (upstream key layout) for
  the dev-eval sidecar and the contract selection;
- the loop polls ``<model_dir>/.early_stop`` at epoch boundaries so the
  pre-recorded early-stop rule (sidecar) can end the run without a kill;
- bf16 autocast is the default (DCU), fp32 fallback via --no_amp.

Usage (torchrun on the DCU):
    torchrun --nproc_per_node=7 -m scripts.brats_p1_finetune \
        -e run/environment.json -c configs/config_brats_p1_train.json \
        -t configs/config_network_rflow.json --replay-list run/lists/p1_mrrate_replay.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, UTC
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.transforms import Compose
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .diff_model_train import augment_modality_label
from .utils import define_instance

STOP_FILE = ".early_stop"
SCALE_FACTOR_RELATIVE_TOLERANCE = 0.5  # issue #10 §7: sanity assert, not a re-pin


class P1TrainDataCatalog:
    """The 1:1 BraTS + MR-RATE replay training list (spec #51 decision 6)."""

    def __init__(self, args, logger):
        self._args = args
        self._logger = logger

    def load_entries(self):
        entries = []
        for label, path in (("brats train list", self._args.json_data_list), * (("replay list", p) for p in self._args.replay_list)):
            payload = json.loads(Path(path).read_text())["training"]
            self._logger.info(f"[data] {label}: {len(payload)} entries from {path}")
            entries += payload
        return entries

    def file_records(self):
        """Maps list entries to {image, spacing, modality} loader records (upstream layout)."""
        records = []
        for entry in self.load_entries():
            emb = os.path.join(self._args.embedding_base_dir, entry["image"].replace(".nii.gz", "_emb.nii.gz"))
            if not os.path.exists(emb):
                raise FileNotFoundError(
                    f"training embedding missing: {emb} (entry {entry.get('sub')}:{entry.get('case')}); "
                    "run the phase/replay pipelines before training"
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
            f"scale_factor sanity: checkpoint={self._checkpoint_value:.6f} recomputed_1/std(z)={recomputed_value:.6f} "
            f"relative_diff={relative:.4f}"
        )
        if relative > SCALE_FACTOR_RELATIVE_TOLERANCE:
            raise ValueError(
                f"scale_factor sanity assert failed: checkpoint {self._checkpoint_value} vs recomputed "
                f"{recomputed_value} (relative {relative:.3f} > {SCALE_FACTOR_RELATIVE_TOLERANCE})"
            )


class P1FinetuneJob:
    """The pinned P1 continuation loop (upstream-equivalent except the pinned deltas)."""

    def __init__(self, args, device, logger, local_rank):
        self._args = args
        self._device = device
        self._logger = logger
        self._local_rank = local_rank

    def build_loader(self):
        args = self._args
        catalog = P1TrainDataCatalog(args, self._logger)
        records = catalog.file_records()
        if self._local_rank == 0:
            self._logger.info(f"num_files_train (brats + replay): {len(records)}")
        if dist.is_initialized():
            records = partition_dataset(
                data=records, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
            )[self._local_rank]
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
        dataset = monai.data.CacheDataset(
            data=records, transform=transforms, cache_rate=args.diffusion_unet_train["cache_rate"], num_workers=2
        )
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
        checkpoint = torch.load(args.existing_ckpt_filepath, map_location=self._device, weights_only=True)
        target = unet.module if dist.is_initialized() else unet
        target.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        self._logger.info(f"base checkpoint loaded (full-param continuation): {args.existing_ckpt_filepath}")
        return unet, ScaleFactorPolicy(checkpoint["scale_factor"], self._logger)

    def train(self):
        args = self._args
        loader = self.build_loader()
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
        loss_pt = torch.nn.L1Loss()
        scaler = GradScaler("cuda")
        torch.set_float32_matmul_precision("highest")

        for epoch in range(args.diffusion_unet_train["n_epochs"]):
            if self._stop_requested():
                self._logger.info(f"early-stop file present; halting before epoch {epoch + 1}")
                break
            self._train_one_epoch(
                epoch, unet, loader, optimizer, lr_scheduler, loss_pt, scaler, scale_factor, noise_scheduler
            )

        if dist.is_initialized():
            dist.destroy_process_group()

    def _stop_requested(self):
        return (Path(self._args.model_dir) / STOP_FILE).is_file()

    def _train_one_epoch(self, epoch, unet, loader, optimizer, lr_scheduler, loss_pt, scaler, scale_factor, noise_scheduler):
        args = self._args
        unet_module = unet.module if isinstance(unet, DistributedDataParallel) else unet
        if self._local_rank == 0:
            self._logger.info(f"Epoch {epoch + 1}, lr {optimizer.param_groups[0]['lr']}.")
        iteration = 0
        loss_totals = torch.zeros(2, dtype=torch.float, device=self._device)
        unet.train()
        for train_data in loader:
            iteration += 1
            images = train_data["image"].to(self._device) * scale_factor
            modality_tensor = augment_modality_label(train_data["modality"].to(self._device)).to(self._device)
            spacing_tensor = train_data["spacing"].to(self._device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16, enabled=args.amp):
                noise = torch.randn_like(images)
                timesteps = noise_scheduler.sample_timesteps(images)
                noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                model_output = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    spacing_tensor=spacing_tensor,
                    class_labels=modality_tensor,
                )
                loss = loss_pt(model_output.float(), (images - noise).float())
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
            self._save_checkpoint(epoch, unet, loss_totals, scale_factor)

    def _save_checkpoint(self, epoch, unet, loss_totals, scale_factor):
        unet_module = unet.module if isinstance(unet, DistributedDataParallel) else unet
        path = Path(self._args.model_dir) / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "loss": (loss_totals[0] / loss_totals[1]).item(),
                "num_train_timesteps": self._args.noise_scheduler["num_train_timesteps"],
                "scale_factor": scale_factor,
                "unet_state_dict": unet_module.state_dict(),
            },
            path,
        )
        (Path(self._args.model_dir) / "latest.json").write_text(
            json.dumps({"epoch": epoch + 1, "checkpoint": str(path)}) + "\n"
        )
        self._logger.info(f"epoch {epoch + 1} average loss: {(loss_totals[0] / loss_totals[1]).item():.4f} -> {path}")


class TrainProvenanceWriter:
    """Records what the training run consumed (feeds the phase-run contract)."""

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
            "data_lists": {
                "brats_train": self._args.json_data_list,
                "replay": list(self._args.replay_list),
            },
            "base_ckpt": self._args.existing_ckpt_filepath,
            "hyperparameters": self._args.diffusion_unet_train,
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
    parser.add_argument(
        "--replay-list", dest="replay_list", action="append", required=True,
        help="MR-RATE replay data list (spec: list-level 1:1 mix; append once per list)",
    )
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"], help="bf16 default (DCU)")
    args = parser.parse_args(argv)

    merged = load_config(args.env_config_path, args.model_config_path, args.model_def_path)
    merged.replay_list = args.replay_list
    merged.amp = args.amp
    merged.amp_dtype = args.amp_dtype
    merged.env_config_path = args.env_config_path
    merged.model_config_path = args.model_config_path
    merged.model_def_path = args.model_def_path

    local_rank, _world, device = initialize_distributed(args.num_gpus)
    logger = setup_logging("p1-finetune")
    if local_rank == 0:
        Path(merged.model_dir).mkdir(parents=True, exist_ok=True)
        TrainProvenanceWriter(merged, local_rank, logger).write(Path(merged.model_dir) / "train_provenance.json")
    P1FinetuneJob(merged, device, logger, local_rank).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
