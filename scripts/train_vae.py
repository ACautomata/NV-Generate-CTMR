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

"""End-to-end VAE training entry (ADR-0015 §8, issue #142).

``train_vae_tutorial.ipynb`` was deleted with the notebook-zeroing batch, which
left ``ctmr.application.vae_train`` without a production caller. This module
rebuilds the caller side in the tutorial's assembly order (cells
10/12/16/18/20/24/26/28/30/31): three config layers onto one namespace, data
lists, VAE_Transform + CacheDataset/DataLoader, network instantiation,
finetune loading, then the per-epoch train_epoch + scheduler step + checkpoint
publication and the val_interval validation pass with best-epoch saving.
TensorBoard/plotting (cells 10/22/31) is not rebuilt -- training summaries go
to stdout; the loss family and loop live in ``ctmr.application.vae_train``.

The data-list files hold ``{"image": path, "class": "ct"|"mri"}`` entries --
the same shape as the tutorial's ``add_assigned_class_to_datalist`` output --
one file per modality, passed via repeatable ``--train-list`` /
``--val-list`` flags.

Usage:
    python -m scripts.train_vae -t configs/config_network_rflow.json \\
        -c configs/config_maisi_vae_train.json \\
        -e configs/environment_maisi_vae_train.json \\
        --train-list train_ct.json --train-list train_mri.json \\
        --val-list val_ct.json --val-list val_mri.json
"""

import argparse
import json
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.utils import set_determinism

from ctmr.application.vae_train import (
    build_adversarial_loss,
    build_amp_scalers,
    build_discriminator,
    build_intensity_loss,
    build_lr_schedulers,
    build_optimizers,
    build_perceptual_loss,
    load_pretrained_weights,
    loss_weighted_sum,
    train_epoch,
    validate_epoch,
)
from scripts.transforms import VAE_Transform
from scripts.utils import define_instance, dynamic_infer

VALID_CLASSES = ("ct", "mri")


def load_settings(network_path: str, config_path: str, environment_path: str) -> argparse.Namespace:
    """The notebook's three config layers merged onto one namespace (cells 10/12)."""
    args = argparse.Namespace()
    for path in (environment_path, network_path):
        with open(path) as source:
            for key, value in json.load(source).items():
                setattr(args, key, value)
    with open(config_path) as source:
        train_config = json.load(source)
    for key, value in train_config["data_option"].items():
        setattr(args, key, value)
    for key, value in train_config["autoencoder_train"].items():
        setattr(args, key, value)
    return args


def load_data_lists(paths: list[str]) -> list[dict]:
    """Cell-16 list shape: ``[{"image": ..., "class": "ct"|"mri"}]``."""
    entries = []
    for path in paths:
        with open(path) as source:
            for item in json.load(source):
                if item.get("class") not in VALID_CLASSES:
                    raise ValueError(f"{path}: every entry needs \"class\" of {VALID_CLASSES}; got {item.get('class')!r}")
                entries.append({"image": item["image"], "class": item["class"]})
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.train_vae",
        description="Train the MAISI 3D autoencoder (VAE) with its adversarial loop.",
    )
    parser.add_argument(
        "-t",
        "--model_def_path",
        type=str,
        default="./configs/config_network_rflow.json",
        help="Path to the network definition file (autoencoder_def).",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="./configs/config_maisi_vae_train.json",
        help="Path to the VAE training configuration file.",
    )
    parser.add_argument(
        "-e",
        "--environment_path",
        type=str,
        default="./configs/environment_maisi_vae_train.json",
        help="Path to the environment configuration file (model_dir / finetune).",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        action="append",
        required=True,
        help='JSON list of {"image", "class"} entries, one file per modality; repeatable.',
    )
    parser.add_argument(
        "--val-list",
        type=str,
        action="append",
        default=[],
        help='JSON list of {"image", "class"} entries for validation; repeatable (optional).',
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.model_def_path, args.config_path, args.environment_path)
    train_files = load_data_lists(args.train_list)
    val_files = load_data_lists(args.val_list) if args.val_list else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cluster default: cuda
    set_determinism(seed=0)
    print(f"device: {device}; finetune: {bool(getattr(settings, 'finetune', False))}")

    # Cells 18/20: per-modality transforms (ct/mri), CacheDataset + DataLoader.
    k = 4  # notebook constant: patches must be divisible by k
    train_transform = VAE_Transform(
        is_train=True,
        random_aug=settings.random_aug,
        k=k,
        patch_size=settings.patch_size,
        val_patch_size=settings.val_patch_size,
        output_dtype=torch.float16,
        spacing_type=settings.spacing_type,
        spacing=settings.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=settings.select_channel,
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=k,
        patch_size=settings.patch_size,
        val_patch_size=settings.val_patch_size,
        output_dtype=torch.float16,
        spacing_type=settings.spacing_type,
        spacing=settings.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=settings.select_channel,
    )
    dataset_train = CacheDataset(data=train_files, transform=train_transform, cache_rate=settings.cache, num_workers=8)
    dataloader_train = DataLoader(dataset_train, batch_size=settings.batch_size, num_workers=4, shuffle=True, drop_last=True)
    dataloader_val = None
    if val_files:
        dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=settings.cache, num_workers=8)
        dataloader_val = DataLoader(dataset_val, batch_size=settings.val_batch_size, num_workers=4, shuffle=False)

    # Cell 24: networks.
    settings.num_splits = 1
    autoencoder = define_instance(settings, "autoencoder_def").to(device)
    discriminator = build_discriminator(spatial_dims=settings.spatial_dims).to(device)

    # Cell 26/28: losses, optimizers, warmup schedulers, AMP pair; finetune load.
    intensity_loss = build_intensity_loss(settings.recon_loss)
    adversarial_loss = build_adversarial_loss()
    perceptual_loss = build_perceptual_loss(device)
    optimizer_g, optimizer_d = build_optimizers(autoencoder, discriminator, lr=settings.lr, amp=settings.amp)
    scheduler_g, scheduler_d = build_lr_schedulers(optimizer_g, optimizer_d)
    scaler_g, scaler_d = build_amp_scalers(amp=settings.amp)
    if getattr(settings, "finetune", False):
        load_pretrained_weights(autoencoder, settings.trained_autoencoder_path)
        print(f"Finetune on pretrained model {settings.trained_autoencoder_path}")
    else:
        print("Train from scratch!")

    # Cell 30/31 tail: val inferer + best-epoch tracking (no TensorBoard here).
    if settings.val_sliding_window_patch_size:
        val_inferer = SlidingWindowInferer(
            roi_size=settings.val_sliding_window_patch_size,
            sw_batch_size=1,
            progress=False,
            overlap=0.0,
            device=torch.device("cpu"),
            sw_device=device,
        )
    else:
        val_inferer = SimpleInferer()

    def infer(images):
        return dynamic_infer(val_inferer, autoencoder, images)

    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    autoencoder_path = model_dir / "autoencoder.pt"
    discriminator_path = model_dir / "discriminator.pt"
    best_val_loss = float("inf")

    for epoch in range(settings.n_epochs):
        train_losses = train_epoch(
            dataloader_train,
            autoencoder=autoencoder,
            discriminator=discriminator,
            intensity_loss=intensity_loss,
            adversarial_loss=adversarial_loss,
            perceptual_loss=perceptual_loss,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            adv_weight=settings.adv_weight,
            kl_weight=settings.kl_weight,
            perceptual_weight=settings.perceptual_weight,
            device=device,
            autocast_device_type=device.type,
            amp=settings.amp,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )
        scheduler_g.step()
        scheduler_d.step()
        torch.save(autoencoder.state_dict(), autoencoder_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        train_sum = loss_weighted_sum(train_losses, kl_weight=settings.kl_weight, perceptual_weight=settings.perceptual_weight)
        print(f"Epoch {epoch} train_vae_loss {train_sum:.4f}: {train_losses}.")
        print(f"Save trained autoencoder to {autoencoder_path}")

        if dataloader_val is not None and epoch % settings.val_interval == 0:
            val_losses = validate_epoch(
                dataloader_val,
                autoencoder=autoencoder,
                intensity_loss=intensity_loss,
                perceptual_loss=perceptual_loss,
                infer=infer,
                device=device,
                autocast_device_type=device.type,
                amp=settings.amp,
            )
            val_sum = loss_weighted_sum(val_losses, kl_weight=settings.kl_weight, perceptual_weight=settings.perceptual_weight)
            print(f"Epoch {epoch} val_vae_loss {val_sum:.4f}: {val_losses}.")
            if val_sum < best_val_loss:
                best_val_loss = val_sum
                best_path = model_dir / f"autoencoder_epoch{epoch}.pt"
                torch.save(autoencoder.state_dict(), best_path)
                print(f"Got best val vae loss; save trained autoencoder to {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
