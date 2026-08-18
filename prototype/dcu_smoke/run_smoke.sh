#!/bin/bash
# PROTOTYPE (throwaway, wayfinder #15) — DCU single-card bf16 end-to-end smoke.
#
# Stages: env-check -> build data list -> VAE-encode (data prep) -> 1-epoch train.
# Uses the repo's real pipeline scripts (diff_model_create_training_data.py +
# diff_model_train.py), not a custom loop — the point is proving the *repo*
# training pipeline runs on DCU.
#
# Usage (on the DCU node):
#   bash prototype/dcu_smoke/run_smoke.sh [N_CASES] [AMP_DTYPE]
#     N_CASES   GLI cases to use (default 6 -> 18 entries over t1n/t2w/t2f)
#     AMP_DTYPE bf16 (default, no GradScaler) | fp16 (GradScaler; expect NaN —
#               that contrast *is* checklist item 4)

set -euo pipefail

REPO=/root/private_data/nv-dcu-smoke/NV-Generate-CTMR
SMOKE=$REPO/prototype/dcu_smoke
DATA_BASE=/root/private_data/datasets/ASNR-MICCAI-BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
N_CASES="${1:-6}"
AMP_DTYPE="${2:-bf16}"

cd "$REPO"
# Double source: DTK (compute) + platform proxy (net). Idempotent if bashrc already did it.
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh
[ -f /root/private_data/.ai_user_info/ai_proxy ] && source /root/private_data/.ai_user_info/ai_proxy

echo "=== [1/4] env/runtime check (checklist items 1,2,3,7,8,11) ==="
python "$SMOKE/dcu_env_check.py"

echo "=== [2/4] build smoke data list (n_cases=$N_CASES) ==="
python "$SMOKE/make_smoke_datalist.py" --data-base-dir "$DATA_BASE" --out "$SMOKE/dataset_dcu_smoke.json" --n-cases "$N_CASES"

echo "=== [3/4] data prep: VAE-encode -> *_emb.nii.gz ==="
python -m scripts.diff_model_create_training_data \
    -e "$SMOKE/environment_dcu_smoke.json" -c "$SMOKE/config_dcu_smoke.json" \
    -t ./configs/config_network_rflow.json -g 1

echo "=== [4/4] train: single-card $AMP_DTYPE, 1 epoch (checklist items 4,6,9,10) ==="
python -m scripts.diff_model_train \
    -e "$SMOKE/environment_dcu_smoke.json" -c "$SMOKE/config_dcu_smoke.json" \
    -t ./configs/config_network_rflow.json -g 1 --amp_dtype "$AMP_DTYPE"

echo "SMOKE_DONE (amp_dtype=$AMP_DTYPE, n_cases=$N_CASES)"
