#!/usr/bin/env python3
"""Validate the SSA derived batch-16 plan before 8-card nnU-Net training.

Run once for filesystem/configuration checks, then under ``torchrun`` with
``--distributed`` to prove the eight-rank NCCL/RCCL path.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

PERSISTENT_ROOT = Path("/root/private_data")
DATASET_NAME = "Dataset502_BraTS2023SSA"
PLANS_IDENTIFIER = "nnUNetPlans_SSA_bs16_v1"
PARENT_CONFIGURATION = "3d_fullres"
VARIANT_CONFIGURATION = "3d_fullres_bs16"
GLOBAL_BATCH_SIZE = 16
WORLD_SIZE = 8


class DdpPreflight:
    """Checks the immutable SSA data contract and eight-rank runtime contract."""

    def __init__(
        self,
        raw_root: Path,
        preprocessed_root: Path,
        results_root: Path,
        audit_path: Path,
        distributed: bool,
    ):
        self.raw_root = raw_root
        self.preprocessed_root = preprocessed_root
        self.results_root = results_root
        self.audit_path = audit_path
        self.distributed = distributed

    def verify(self) -> dict:
        self._verify_persistent_paths()
        self._verify_dataset_contract()
        plan_summary = self._verify_plan_contract()
        self._verify_compile_disabled()
        runtime_summary = self._verify_runtime()
        distributed_summary = self._verify_distributed_runtime() if self.distributed else None
        return {
            "dataset": DATASET_NAME,
            "plans": plan_summary,
            "runtime": runtime_summary,
            "distributed": distributed_summary,
            "compile_environment": os.environ.get("nnUNet_compile"),
        }

    def _verify_persistent_paths(self) -> None:
        for label, path in {
            "raw_root": self.raw_root,
            "preprocessed_root": self.preprocessed_root,
            "results_root": self.results_root,
            "audit_path": self.audit_path,
        }.items():
            if not path.resolve().is_relative_to(PERSISTENT_ROOT):
                raise ValueError(f"{label} must be under {PERSISTENT_ROOT}: {path}")
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"raw root not found: {self.raw_root}")
        if not self.preprocessed_root.is_dir():
            raise FileNotFoundError(f"preprocessed root not found: {self.preprocessed_root}")
        if not self.audit_path.is_file():
            raise FileNotFoundError(f"audit record not found: {self.audit_path}")

    def _verify_dataset_contract(self) -> None:
        dataset_dir = self.raw_root / DATASET_NAME
        dataset_json = dataset_dir / "dataset.json"
        splits_path = dataset_dir / "splits_final.json"
        if not dataset_json.is_file() or not splits_path.is_file():
            raise FileNotFoundError("Dataset502 must contain dataset.json and splits_final.json")
        dataset = json.loads(dataset_json.read_text())
        if dataset.get("numTraining") != 42:
            raise ValueError(f"expected SSA training-side cases=42, got {dataset.get('numTraining')!r}")
        expected_channels = {"0000": "t1n", "0001": "t1c", "0002": "t2w", "0003": "t2f"}
        if dataset.get("channel_names") != expected_channels:
            raise ValueError("SSA channel contract is not t1n/t1c/t2w/t2f")
        splits = json.loads(splits_path.read_text())
        if len(splits) != 1 or len(splits[0].get("train", [])) != 34 or len(splits[0].get("val", [])) != 8:
            raise ValueError("expected one SSA fold_0 split with 34 train and 8 validation cases")

    def _verify_plan_contract(self) -> dict:
        dataset_dir = self.preprocessed_root / DATASET_NAME
        source_path = dataset_dir / "nnUNetPlans.json"
        variant_path = dataset_dir / f"{PLANS_IDENTIFIER}.json"
        fingerprint_path = dataset_dir / "dataset_fingerprint.json"
        if not source_path.is_file() or not variant_path.is_file() or not fingerprint_path.is_file():
            raise FileNotFoundError("source plans, derived plans, and fingerprint must exist before DDP training")

        source_bytes = source_path.read_bytes()
        variant_bytes = variant_path.read_bytes()
        source = json.loads(source_bytes)
        variant = json.loads(variant_bytes)
        audit = json.loads(self.audit_path.read_text())
        if audit.get("source_plans_sha256") != hashlib.sha256(source_bytes).hexdigest():
            raise ValueError("source plans hash differs from the approved audit record")
        if audit.get("derived_plans_sha256") != hashlib.sha256(variant_bytes).hexdigest():
            raise ValueError("derived plans hash differs from the approved audit record")
        if variant.get("plans_name") != PLANS_IDENTIFIER:
            raise ValueError("derived plans identifier does not match the approved SSA batch-16 plan")

        parent = self._resolve_configuration(source["configurations"], PARENT_CONFIGURATION)
        derived = self._resolve_configuration(variant["configurations"], VARIANT_CONFIGURATION)
        if parent.get("batch_size") != 2:
            raise ValueError(f"expected source 3d_fullres global batch=2, got {parent.get('batch_size')!r}")
        if derived.get("batch_size") != GLOBAL_BATCH_SIZE:
            raise ValueError(f"expected derived global batch={GLOBAL_BATCH_SIZE}")
        if derived.get("data_identifier") != parent.get("data_identifier"):
            raise ValueError("derived batch-only plan must reuse the source data_identifier")
        if derived.get("data_identifier") != "nnUNetPlans_3d_fullres":
            raise ValueError("unexpected SSA data_identifier")
        if GLOBAL_BATCH_SIZE // WORLD_SIZE != 2:
            raise ValueError("approved batch arithmetic no longer yields local batch 2")
        return {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "derived_sha256": hashlib.sha256(variant_bytes).hexdigest(),
            "configuration": VARIANT_CONFIGURATION,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": 2,
            "data_identifier": derived["data_identifier"],
        }

    def _resolve_configuration(self, configurations: dict, name: str, seen: tuple[str, ...] = ()) -> dict:
        if name in seen:
            raise ValueError(f"cyclic configuration inheritance: {' -> '.join((*seen, name))}")
        configuration = configurations.get(name)
        if not isinstance(configuration, dict):
            raise ValueError(f"configuration not found: {name!r}")
        parent_name = configuration.get("inherits_from")
        resolved = {}
        if parent_name is not None:
            resolved.update(self._resolve_configuration(configurations, parent_name, (*seen, name)))
        resolved.update({key: value for key, value in configuration.items() if key != "inherits_from"})
        return resolved

    def _verify_compile_disabled(self) -> None:
        value = os.environ.get("nnUNet_compile", "").lower()
        if value not in {"f", "false", "0"}:
            raise ValueError("set nnUNet_compile=f before SSA DDP training on this DCU stack")

    def _verify_runtime(self) -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("torch-dcu reports no CUDA/DCU device")
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError(f"expected exactly {WORLD_SIZE} visible DCUs, got {torch.cuda.device_count()}")
        if not dist.is_nccl_available():
            raise RuntimeError("NCCL/RCCL backend is unavailable")
        return {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "visible_dcus": torch.cuda.device_count(),
            "nccl_available": True,
        }

    def _verify_distributed_runtime(self) -> dict:
        expected_world_size = int(os.environ.get("WORLD_SIZE", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if expected_world_size != WORLD_SIZE or local_rank < 0:
            raise RuntimeError("launch distributed preflight with torchrun --nproc_per_node=8")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        try:
            value = torch.tensor([dist.get_rank() + 1], device="cuda", dtype=torch.int64)
            dist.all_reduce(value)
            expected_sum = WORLD_SIZE * (WORLD_SIZE + 1) // 2
            if value.item() != expected_sum:
                raise RuntimeError(f"all_reduce expected {expected_sum}, got {value.item()}")
            return {"rank": dist.get_rank(), "world_size": dist.get_world_size(), "all_reduce_sum": value.item()}
        finally:
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PERSISTENT_ROOT / "brats2023_nnunet")
    parser.add_argument("--preprocessed-root", type=Path, default=PERSISTENT_ROOT / "nnUNet_preprocessed")
    parser.add_argument("--results-root", type=Path, default=PERSISTENT_ROOT / "nnUNet_results_ssa_bs16_ddp8")
    parser.add_argument(
        "--audit",
        type=Path,
        default=PERSISTENT_ROOT / "l2-instrument-audit" / "ssa-bs16-v1" / "plans-variant-audit.json",
    )
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    result = DdpPreflight(
        raw_root=args.raw_root,
        preprocessed_root=args.preprocessed_root,
        results_root=args.results_root,
        audit_path=args.audit,
        distributed=args.distributed,
    ).verify()
    if not args.distributed or result["distributed"]["rank"] == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
