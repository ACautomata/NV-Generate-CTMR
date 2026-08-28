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

"""DCU smoke for the phase run contract (issue #53, spec #51 decision 12).

Proves, on the sugon DCU stack, the platform half of the contract:

- companion metadata is readable through the same {spacing, modality} companion
  contract the DM trainer consumes (``<emb>.nii.gz`` + ``<emb>.nii.gz.json``,
  spec user story 9);
- a bf16 autocast training step and the fp32 fallback step both run finite
  (bf16 is the pinned recipe dtype, fp32 stays the兜底);
- a persistent checkpoint with the upstream ``diff_model_train`` key layout
  (epoch/loss/num_train_timesteps/scale_factor/unet_state_dict) is written
  under controlled storage and re-loadable;
- under ``torchrun --distributed`` the same path runs through DDP with the
  ``nccl`` backend name (RCCL on DCU), including an all_reduce sanity check;
- the run contract itself opens/selects/freezes/verifies a P1 record from
  the smoke artifacts, including the negative holdout-evidence probe
  (candidate freeze must reject final-holdout selection input).

Usage (from the directory containing ``scripts/``, i.e. repo root or a copy):
    python3 -m scripts.brats_phase_dcu_smoke --root /root/private_data/brats2023_rflow_phase_smoke
    torchrun --standalone --nproc_per_node=2 -m scripts.brats_phase_dcu_smoke \
        --distributed --root /root/private_data/brats2023_rflow_phase_smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ctmr.application.acceptance.contract import (
    ArtifactFingerprinter,
    CandidateFreezer,
    ContractViolationError,
    ManifestSides,
    ReportAttacher,
    RunInitializer,
    RunRecordStore,
    RunVerifier,
    SelectionRecorder,
)

# Pinned generative tokens (spec #51 decision 5, configs/modality_mapping.json).
MODALITY_TOKENS = {"mri_t1_skull_stripped": 29}
LATENT_SHAPE = (4, 8, 8, 8)  # tiny stand-in for the 4x64x64x32 training latent
CKPT_KEYS = ("epoch", "loss", "num_train_timesteps", "scale_factor", "unet_state_dict")
DEFAULT_ROOT = Path("/root/private_data/brats2023_rflow_phase_smoke")


class SmokeFixture:
    """Synthetic non-subject inputs for the smoke: manifest, embeddings, companions, lists."""

    CASES = {
        "GLI": {
            "train": ["FIXGLI-SMK-0000-000"],
            "dev": ["FIXGLI-SMK-0100-000"],
            "holdout": ["FIXGLI-SMK-0200-000"],
        }
    }

    def __init__(self, root):
        self.root = Path(root)

    def root_dir(self):
        return self.root / "fixture"

    def embeddings_dir(self):
        return self.root_dir() / "embeddings"

    def train_case(self):
        return self.CASES["GLI"]["train"][0]

    def write(self):
        fixture = self.root_dir()
        manifest = {"split_id": "dcu-smoke", "challenges": {}}
        for ch, sides in self.CASES.items():
            manifest["challenges"][ch] = {"cases": dict(sides)}
        (fixture / "phase_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (fixture / "phase_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        embeddings = fixture / "embeddings"
        rng = np.random.default_rng(0)  # idempotent fixture: reruns keep identical bytes
        for side in ("train", "dev"):
            for case in self.CASES["GLI"][side]:
                emb = embeddings / f"{case}_t1n_emb.nii.gz"
                emb.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(rng.random(LATENT_SHAPE).astype(np.float32), np.eye(4)), str(emb))
                emb.with_name(emb.name + ".json").write_text(json.dumps({"spacing": [1.0, 1.0, 1.2], "modality": "mri_t1_skull_stripped"}))

        train_case = self.CASES["GLI"]["train"][0]
        dev_case = self.CASES["GLI"]["dev"][0]
        holdout_case = self.CASES["GLI"]["holdout"][0]
        (fixture / "lists").mkdir(parents=True, exist_ok=True)
        (fixture / "lists" / "train.json").write_text(
            json.dumps({"training": [{"image": f"embeddings/{train_case}_t1n_emb.nii.gz", "sub": "GLI", "case": train_case}]})
        )
        (fixture / "env_config.json").write_text('{"lr": 2e-06, "n_epochs": 100, "batch_size": 1}\n')
        (fixture / "base_ckpt.pt").write_bytes(b"rflow-mr-brain-v1-smoke-fixture")
        (fixture / "dev_metrics.json").write_text(json.dumps({"metrics": [{"sub": "GLI", "case": dev_case, "fid_trend": [0.9, 0.7, 0.6]}]}))
        (fixture / "holdout_metrics.json").write_text(json.dumps({"metrics": [{"sub": "GLI", "case": holdout_case, "fid": 0.1}]}))
        (fixture / "samples.json").write_text('{"samples": ["smoke-sample-t1n.nii.gz"]}\n')
        return fixture


class CompanionReader:
    """Reads one batch through the trainer's companion mechanism (spec user story 9)."""

    def __init__(self, modality_tokens):
        self._tokens = modality_tokens

    def read_batch(self, embedding_path):
        """Returns the latent tensor, spacing and modality token from the companion."""
        companion = json.loads(Path(embedding_path + ".json").read_text())
        spacing = companion["spacing"]
        token = self._tokens[companion["modality"]]
        image = torch.from_numpy(np.asanyarray(nib.load(embedding_path).dataobj)).float()
        if tuple(image.shape) != LATENT_SHAPE:
            raise RuntimeError(f"unexpected latent shape {tuple(image.shape)} != {LATENT_SHAPE}")
        return image, spacing, token


class AmpProbe:
    """A tiny 3D conv model driven through one bf16-autocast and one fp32 step."""

    def build_model(self, device):
        model = torch.nn.Sequential(
            torch.nn.Conv3d(LATENT_SHAPE[0], 8, 3, padding=1),
            torch.nn.GroupNorm(4, 8),
            torch.nn.SiLU(),
            torch.nn.Conv3d(8, LATENT_SHAPE[0], 3, padding=1),
        ).to(device)
        return model

    def step(self, model, images, amp_dtype):
        use_amp = amp_dtype is not None
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            target = torch.roll(images, shifts=1, dims=-1)
            loss = torch.nn.functional.l1_loss(model(images), target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss under amp_dtype={amp_dtype}")
        loss.backward()
        optimizer.step()
        return float(loss.item())


class SmokeCheckpoint:
    """Writes and re-loads a persistent checkpoint with the upstream trainer key layout."""

    def __init__(self, root, fingerprinter):
        self._root = Path(root)
        self._fingerprinter = fingerprinter

    def save(self, model, loss, scale_factor, mode):
        ckpt_dir = self._root / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Per-mode filenames: rerunning the other mode must not invalidate this
        # record's pinned checkpoint (the contract hashes the exact bytes).
        path = ckpt_dir / f"smoke_model_{mode}.pt"
        torch.save(
            {
                "epoch": 1,
                "loss": loss,
                "num_train_timesteps": 500,
                "scale_factor": scale_factor,
                "unet_state_dict": model.state_dict(),
            },
            path,
        )
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        missing = [key for key in CKPT_KEYS if key not in reloaded]
        if missing:
            raise RuntimeError(f"checkpoint is missing upstream keys: {missing}")
        return {"path": str(path.resolve()), "sha256": self._fingerprinter.file_sha256(path)}


class DistributedContext:
    """Single-card or torchrun-driven DDP with the nccl backend name (RCCL on DCU)."""

    def __init__(self, distributed):
        self._distributed = distributed

    def enter(self):
        if not self._distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("torch reports no CUDA/DCU device")
            torch.cuda.set_device(0)
            return {"mode": "single", "local_rank": 0, "world_size": 1}
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if world_size < 2 or local_rank < 0:
            raise RuntimeError("launch distributed smoke with torchrun --nproc_per_node=N (N>=2)")
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL/RCCL backend is unavailable")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        probe = torch.tensor([dist.get_rank() + 1], device="cuda", dtype=torch.int64)
        dist.all_reduce(probe)
        expected = world_size * (world_size + 1) // 2
        if probe.item() != expected:
            raise RuntimeError(f"all_reduce sanity failed: {probe.item()} != {expected}")
        return {"mode": "distributed", "local_rank": local_rank, "world_size": world_size, "backend": "nccl", "all_reduce_sum": probe.item()}

    def exit(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class ContractSmokeHarness:
    """Drives the full run contract over the smoke artifacts, including the holdout probe."""

    def __init__(self, root, fingerprinter, mode):
        self._root = Path(root)
        self._fingerprinter = fingerprinter
        self._mode = mode

    def expect_reject(self, action, label, failures):
        try:
            action()
        except ContractViolationError:
            return
        failures.append(f"contract probe not rejected: {label}")

    def run(self, fixture, checkpoint_entry, platform_summary):
        fixture_dir = fixture.root_dir()
        failures = []
        platform_path = self._root / f"platform_{self._mode}.json"
        platform_path.write_text(json.dumps(platform_summary, indent=2, sort_keys=True) + "\n")
        store = RunRecordStore(self._root / "records")
        initializer = RunInitializer(store, self._fingerprinter, ManifestSides.from_path(fixture_dir / "phase_manifest.json"))
        run_id = f"p1-smoke-{self._mode}"
        run_path = initializer.init(
            "P1",
            run_id,
            fixture_dir / "phase_manifest.json",
            [("env", fixture_dir / "env_config.json")],
            [("train", fixture_dir / "lists" / "train.json")],
            fixture_dir / "base_ckpt.pt",
            None,
            platform_path,
        )
        # Negative probe: holdout metrics must never pass as selection evidence.
        self.expect_reject(
            lambda: SelectionRecorder(store, self._fingerprinter).select(
                run_path, checkpoint_entry["path"], "probe", [fixture_dir / "holdout_metrics.json"], None
            ),
            "holdout evidence in checkpoint selection",
            failures,
        )
        SelectionRecorder(store, self._fingerprinter).select(
            run_path,
            checkpoint_entry["path"],
            "dev FID trend smoke rule",
            [fixture_dir / "dev_metrics.json"],
            epoch=1,
        )
        CandidateFreezer(store, self._fingerprinter).freeze(run_path, fixture_dir / "samples.json")
        ReportAttacher(store, self._fingerprinter).attach(run_path, "env", platform_path)
        failures += [f"contract verify: {f}" for f in RunVerifier(self._fingerprinter).verify(store.load_by_path(run_path), record_path=run_path)]
        return failures


class DcuSmoke:
    """Orchestrates fixture -> companion read -> amp steps -> checkpoint -> contract."""

    def __init__(self, root, distributed):
        self._root = Path(root)
        self._distributed = distributed
        self._fingerprinter = ArtifactFingerprinter()
        self._runtime = DistributedContext(distributed)

    def run(self):
        summary = self._runtime.enter()
        rank = summary["local_rank"]
        failures = []
        try:
            fixture = SmokeFixture(self._root)
            if rank == 0:
                fixture.write()
            if summary["world_size"] > 1:
                dist.barrier()

            train_case = fixture.train_case()
            embedding_path = str(fixture.embeddings_dir() / f"{train_case}_t1n_emb.nii.gz")
            images, spacing, token = CompanionReader(MODALITY_TOKENS).read_batch(embedding_path)
            if token != 29 or len(spacing) != 3:
                failures.append(f"companion read broken: token={token} spacing={spacing}")

            device = torch.device("cuda", rank)
            images = images.unsqueeze(0).to(device)
            scale_factor = 1 / torch.std(images)  # mirrors calculate_scale_factor
            probe = AmpProbe()
            model = probe.build_model(device)
            if summary["world_size"] > 1:
                model = DistributedDataParallel(model, device_ids=[device])
            losses = {
                "bf16": probe.step(model, images, torch.bfloat16),
                "fp32_fallback": probe.step(model, images, None),
            }

            checkpoint_entry = None
            if rank == 0:
                checkpoint_entry = SmokeCheckpoint(self._root, self._fingerprinter).save(
                    model.module if isinstance(model, DistributedDataParallel) else model,
                    losses["bf16"],
                    scale_factor.cpu(),
                    summary["mode"],
                )
            if summary["world_size"] > 1:
                dist.barrier()

            if rank == 0:
                platform = {
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "visible_dcus": torch.cuda.device_count(),
                    "nccl_available": dist.is_nccl_available(),
                    "mode": summary["mode"],
                    "world_size": summary["world_size"],
                    "amp": {"bf16": losses["bf16"], "fp32_fallback": losses["fp32_fallback"]},
                    "companion": {"spacing": spacing, "modality_token": token},
                    "checkpoint": checkpoint_entry,
                }
                # Controlled-storage rule: smoke artifacts must stay outside any git work tree.
                if RunVerifier.work_tree_ancestor(self._root) is not None:
                    failures.append(f"smoke root is inside a git work tree: {self._root}")
                failures += ContractSmokeHarness(self._root, self._fingerprinter, summary["mode"]).run(fixture, checkpoint_entry, platform)
                (self._root / "smoke_summary.json").write_text(json.dumps({"failures": failures, **platform}, indent=2, sort_keys=True) + "\n")
                for failure in failures:
                    print("FAIL " + failure, file=sys.stderr)
                verdict = "DCU SMOKE PASS" if not failures else "DCU SMOKE FAIL"
                print(json.dumps({"verdict": verdict, "failures": failures, **platform}, indent=2, sort_keys=True))
            return 0 if rank != 0 or not failures else 1
        finally:
            self._runtime.exit()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="controlled root for smoke artifacts and run records")
    parser.add_argument("--distributed", action="store_true", help="run under torchrun --nproc_per_node=N (DDP, nccl/RCCL)")
    args = parser.parse_args(argv)
    return DcuSmoke(args.root, args.distributed).run()


if __name__ == "__main__":
    sys.exit(main())
