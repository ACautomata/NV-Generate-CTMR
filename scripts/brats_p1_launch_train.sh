#!/bin/bash
# Issue #57 P1 带肿瘤候选训练——sugon DCU 启动器（7 卡 DDP 训练 + 1 卡 dev 评估侧车）。
#
# 在 sugon 上运行（仓库根目录 = phase 脚本所在 controlled 布局）：
#   REPO=/root/nv-phase-57 bash scripts/brats_p1_launch_train.sh
#
# 前置（由执行侧准备，见 docs/training.md P1 一节）：
#   1. $PHASE_ROOT/lists/{p1_image_only.json,p1_image_only_dev.json}（#52 产物）
#   2. $RUN_ROOT/lists/p1_mrrate_replay.json + $EMB_ROOT/MR-RATE/...（brats_p1_replay_prep 产物）
#   3. 基模 ckpt（rflow-mr-brain v1）与 autoencoder_v1.pt
#   4. dev 真实特征参考库已由 `brats_p1_dev_eval reference` 预先构建
set -euo pipefail

REPO="${REPO:?set REPO to the controlled repo copy}"
RUN_ROOT="${RUN_ROOT:-/root/private_data/brats2023_rflow_p1}"
PHASE_ROOT="${PHASE_ROOT:-/root/private_data/brats2023_rflow_phase}"
BASE_CKPT="${BASE_CKPT:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/diff_unet_3d_rflow-mr-brain_v1.pt}"
AUTOENCODER="${AUTOENCODER:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/autoencoder_v1.pt}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6}"
SIDECAR_GPU="${SIDECAR_GPU:-7}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PATIENCE="${PATIENCE:-3}"
MIN_EPOCH="${MIN_EPOCH:-30}"
MAX_EPOCH="${MAX_EPOCH:-100}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{ckpt,logs,records}

# ── 环境配置（实例路径，契约 init 指纹的就是这份） ──
cat > "$RUN_ROOT/environment_brats_p1_train.json" <<EOF
{
    "data_base_dir": "$PHASE_ROOT/raw",
    "embedding_base_dir": "$PHASE_ROOT/embeddings",
    "json_data_list": "$PHASE_ROOT/lists/p1_image_only.json",
    "model_dir": "$RUN_ROOT/ckpt",
    "model_filename": "epoch_N.pt",
    "trained_autoencoder_path": "$AUTOENCODER",
    "existing_ckpt_filepath": "$BASE_CKPT",
    "modality_mapping_path": "$REPO/configs/modality_mapping.json"
}
EOF

# MR-RATE 回放 embeddings 挂载进同一 embedding 根（rsync 落位即无需挂载）
if [ ! -f "$PHASE_ROOT/embeddings/MR-RATE" ] && [ -d "$RUN_ROOT/embeddings/MR-RATE" ]; then
    ln -sfn "$RUN_ROOT/embeddings/MR-RATE" "$PHASE_ROOT/embeddings/MR-RATE"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$RUN_ROOT/logs/train_$STAMP.log"
SIDECAR_LOG="$RUN_ROOT/logs/dev_eval_$STAMP.log"

# ── dev 评估侧车（GPU $SIDECAR_GPU；轮询 epoch_<N>.pt，触发早停） ──
HIP_VISIBLE_DEVICES="$SIDECAR_GPU" nohup python3 -m scripts.brats_p1_dev_eval watch \
    --ckpt-dir "$RUN_ROOT/ckpt" \
    --eval-root "$RUN_ROOT/dev_eval" \
    --dev-list "$PHASE_ROOT/lists/p1_image_only_dev.json" \
    --raw-root "$PHASE_ROOT/raw" \
    --emb-root "$PHASE_ROOT/embeddings" \
    -e "$RUN_ROOT/environment_brats_p1_train.json" \
    -c "$REPO/configs/config_brats_p1_train.json" \
    -t "$REPO/configs/config_network_rflow.json" \
    --eval-every "$EVAL_EVERY" --patience "$PATIENCE" \
    --min-epoch "$MIN_EPOCH" --max-epoch "$MAX_EPOCH" \
    --instrument-results "GLI=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/GLI/attempt-001" \
    --instrument-results "MEN=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/MEN/attempt-001" \
    --instrument-results "METS=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/METS/attempt-001" \
    --instrument-results "PED=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/PED/attempt-001" \
    --instrument-results "SSA=/root/private_data/l2-instrument/be683eefb071022b2b62646234e4f7e469ae8dbc/results/SSA/attempt-001" \
    > "$SIDECAR_LOG" 2>&1 &
SIDECAR_PID=$!
echo "sidecar pid=$SIDECAR_PID gpu=$SIDECAR_GPU log=$SIDECAR_LOG"

# ── 训练主进程（$TRAIN_GPUS 上 torchrun DDP，bf16） ──
GPUS=(${TRAIN_GPUS//,/ })
export HIP_VISIBLE_DEVICES="$TRAIN_GPUS"
nohup torchrun --nproc_per_node="${#GPUS[@]}" -m scripts.brats_p1_finetune \
    -e "$RUN_ROOT/environment_brats_p1_train.json" \
    -c "$REPO/configs/config_brats_p1_train.json" \
    -t "$REPO/configs/config_network_rflow.json" \
    -g "${#GPUS[@]}" \
    --replay-list "$RUN_ROOT/lists/p1_mrrate_replay.json" \
    --amp_dtype bf16 \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "train pid=$TRAIN_PID gpus=$TRAIN_GPUS log=$TRAIN_LOG"
echo "$TRAIN_PID" > "$RUN_ROOT/logs/train.pid"
echo "$SIDECAR_PID" > "$RUN_ROOT/logs/dev_eval.pid"
echo "launched $STAMP: train=$TRAIN_PID sidecar=$SIDECAR_PID"
