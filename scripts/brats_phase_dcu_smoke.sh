#!/usr/bin/env bash
# DCU smoke launcher for the phase run contract (issue #53, spec #51 decision 12).
#
# Runs both platform modes from the directory containing scripts/:
#   1. single card: bf16 autocast step + fp32 fallback + persistent checkpoint
#   2. two-card torchrun DDP with the nccl backend name (RCCL on DCU)
# Both modes read the synthetic companion metadata and drive the full run
# contract (init -> holdout probe -> select -> freeze -> attach -> verify).
#
# Usage (sugon, DTK env sourced):
#   bash scripts/brats_phase_dcu_smoke.sh [ROOT]
# ROOT defaults to /root/private_data/brats2023_rflow_phase_smoke (controlled,
# persistent storage — system disk is volatile on this cluster).

set -euo pipefail

ROOT=${1:-/root/private_data/brats2023_rflow_phase_smoke}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

echo "[1/2] single-card bf16 + fp32 fallback"
python3 -m scripts.brats_phase_dcu_smoke --root "$ROOT"

echo "[2/2] multi-card DDP/RCCL (2 ranks)"
torchrun --standalone --nproc_per_node=2 -m scripts.brats_phase_dcu_smoke \
    --distributed --root "$ROOT"

echo "DCU SMOKE PASS (single bf16/fp32 + DDP/RCCL + run contract)"
