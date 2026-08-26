# PROTOTYPE (throwaway, wayfinder #15) — DCU env/runtime checker.
#
# Runs the *environment* items of the #6 DCU-compatibility verification
# checklist (docs/research/dcu-compatibility.md §6) that need no training data:
#   1  device naming      — torch-dcu keeps the "cuda" namespace
#   2  distributed backend — backend="nccl" resolves (lands on RCCL/HCCL)
#   3  SDPA backend        — scaled_dot_product_attention runs, finite output
#   7  numpy ABI           — numpy stays 1.x (2.x breaks DCU torch)
#   8  visibility var      — which env var picks devices, and device_count
#   11 monai version       — monai >= 1.5.2 and RFlowScheduler importable
#
# The heavier items (4 AMP, 5 SyncBN, 6 op coverage, 9 CacheDataset RAM,
# 10 VRAM) are exercised by the actual smoke train run, not this script.
#
# Run on the DCU node (after sourcing /opt/dtk/env.sh):
#   python prototype/dcu_smoke/dcu_env_check.py

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass

import numpy
import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class CheckResult:
    """One checklist item outcome."""

    item: int
    name: str
    ok: bool
    detail: str


class DcuEnvChecker:
    """Runs the environment-side DCU verification items and collects results."""

    def __init__(self) -> None:
        self._results: list[CheckResult] = []

    def check_device_naming(self) -> CheckResult:
        avail = torch.cuda.is_available()
        count = torch.cuda.device_count()
        hip = torch.version.hip
        name = torch.cuda.get_device_name(0) if avail and count else "<none>"
        ok = avail and count > 0
        detail = f'is_available={avail} count={count} dev0="{name}" torch.version.hip={hip} ("cuda" ns on DCU)'
        return CheckResult(1, "device naming", ok, detail)

    def check_nccl_backend(self) -> CheckResult:
        nccl = dist.is_nccl_available()
        try:
            default = dist.get_default_backend_for_device("cuda")
        except Exception as exc:  # older torch lacks this helper
            default = f"<unavailable: {exc}>"
        detail = f'is_nccl_available={nccl} default_backend_for_cuda={default} (keep backend="nccl" -> RCCL/HCCL)'
        return CheckResult(2, "distributed backend", bool(nccl), detail)

    def check_sdpa(self) -> CheckResult:
        if not torch.cuda.is_available():
            return CheckResult(3, "SDPA", False, "no cuda device")
        q = torch.randn(1, 4, 16, 64, device="cuda")
        out = F.scaled_dot_product_attention(q, q, q)
        finite = bool(torch.isfinite(out).all())
        backends = {
            name: getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
            for name in ("flash", "mem_efficient", "math")
            if hasattr(torch.backends.cuda, f"{name}_sdp_enabled")
        }
        detail = f"sdpa forward finite={finite} backends={backends}"
        return CheckResult(3, "SDPA", finite, detail)

    def check_numpy_abi(self) -> CheckResult:
        major = int(numpy.__version__.split(".")[0])
        ok = major == 1
        detail = f"numpy={numpy.__version__} (DCU torch is built against numpy 1.x; must stay 1.x)"
        return CheckResult(7, "numpy ABI", ok, detail)

    def check_visibility_var(self) -> CheckResult:
        cuda_vd = os.environ.get("CUDA_VISIBLE_DEVICES")
        hip_vd = os.environ.get("HIP_VISIBLE_DEVICES")
        count = torch.cuda.device_count()
        detail = f"CUDA_VISIBLE_DEVICES={cuda_vd} HIP_VISIBLE_DEVICES={hip_vd} -> device_count={count}"
        return CheckResult(8, "device visibility var", count > 0, detail)

    def check_monai(self) -> CheckResult:
        try:
            monai = importlib.import_module("monai")
            sched = importlib.import_module("monai.networks.schedulers")
            has_rflow = hasattr(sched, "RFlowScheduler")
            version = monai.__version__
        except Exception as exc:
            return CheckResult(11, "monai", False, f"import failed: {exc}")
        parts = version.split(".")
        ok = (int(parts[0]), int(parts[1])) >= (1, 5) and version not in ("1.5.0", "1.5.1") and has_rflow
        detail = f"monai={version} RFlowScheduler={has_rflow} (need >=1.5.2 for torch 2.9)"
        return CheckResult(11, "monai", ok, detail)

    def run_all(self) -> list[CheckResult]:
        checks = (
            self.check_device_naming,
            self.check_nccl_backend,
            self.check_sdpa,
            self.check_numpy_abi,
            self.check_visibility_var,
            self.check_monai,
        )
        self._results = [check() for check in checks]
        return self._results

    def print_report(self) -> bool:
        print("=== DCU env/runtime check (wayfinder #15) ===")
        for res in self._results:
            mark = "PASS" if res.ok else "FAIL"
            print(f"[{mark}] item {res.item:<2} {res.name:<22} {res.detail}")
        all_ok = all(res.ok for res in self._results)
        print(f"=> {'ALL PASS' if all_ok else 'SOME FAILED'}")
        return all_ok


def main() -> int:
    checker = DcuEnvChecker()
    checker.run_all()
    return 0 if checker.print_report() else 1


if __name__ == "__main__":
    sys.exit(main())
