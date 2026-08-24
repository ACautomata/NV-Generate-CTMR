#!/bin/bash
# Issue #59 P2 掩码→影像候选训练——sugon DCU 启动器（7 卡 DDP 训练 + 1 卡 dev 评估侧车）。
#
# 在 sugon 上运行（仓库根目录 = phase 脚本所在 controlled 布局）：
#   REPO=/root/nv-phase-59 bash scripts/brats_p2_launch_train.sh
#
# 前置（由执行侧准备，见 docs/training.md P2 一节；部分依赖 #58 结论落地）：
#   1. $PHASE_ROOT/lists/p2_mask_cond.json + $PHASE_ROOT/labels/... + $PHASE_ROOT/embeddings/...（#52 产物）
#   2. 冻结 P1-DM —— DM_SOURCE_CKPT（dm_source.json 注册候选的 epoch_<N>.pt，#58 conclude 产物）
#   3. autoencoder_v1.pt
#   4. dev 真实特征参考库已由 `brats_p2_dev_eval reference` 预先构建
#
# 硬前置：P2 必须在运行契约 `init` 前有已注册 DM source（DmSourceLedger），
# 否则 P2 init 被拒（"no P1 candidate has passed final acceptance yet; ..."）。
set -euo pipefail

REPO="${REPO:?set REPO to the controlled repo copy}"
RUN_ROOT="${RUN_ROOT:-/root/private_data/brats2023_rflow_p2}"
PHASE_ROOT="${PHASE_ROOT:-/root/private_data/brats2023_rflow_phase}"
BASE_CKPT="${BASE_CKPT:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/diff_unet_3d_rflow-mr-brain_v1.pt}"
AUTOENCODER="${AUTOENCODER:-/root/private_data/manifold/models/autoencoder_v1.pt}"
# 冻结 P1-DM（#58 dm_source 注册候选的 checkpoint）。P2 不训练 VAE/DM，只挂接它。
DM_SOURCE_CKPT="${DM_SOURCE_CKPT:?set DM_SOURCE_CKPT to the frozen P1-DM candidate checkpoint from dm_source.json}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6}"
SIDECAR_GPU="${SIDECAR_GPU:-7}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PATIENCE="${PATIENCE:-3}"
MIN_EPOCH="${MIN_EPOCH:-30}"
MAX_EPOCH="${MAX_EPOCH:-100}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{ckpt,logs,records,tblogs}

# ── 环境配置（实例路径，契约 init 指纹的就是这份；P2 无回放、无 base-ckpt） ──
# 幂等：契约 init 已指纹化本文件（--platform-json）。若已存在则不覆盖，
# 避免重写内容（路径/格式差异）导致运行记录的平台 sha 与文件不一致。
if [ ! -f "$RUN_ROOT/environment_brats_p2_train.json" ]; then
cat > "$RUN_ROOT/environment_brats_p2_train.json" <<EOF
{
    "data_base_dir": "$PHASE_ROOT",
    "json_data_list": "$PHASE_ROOT/lists/p2_mask_cond.json",
    "model_dir": "$RUN_ROOT/ckpt",
    "trained_autoencoder_path": "$AUTOENCODER",
    "trained_diffusion_path": "$DM_SOURCE_CKPT",
    "tfevent_path": "$RUN_ROOT/tblogs",
    "exp_name": "p2_mask_cond",
    "modality_mapping_path": "$REPO/configs/modality_mapping.json"
}
EOF
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$RUN_ROOT/logs/train_$STAMP.log"
SIDECAR_LOG="$RUN_ROOT/logs/dev_eval_$STAMP.log"

# ── dev 评估侧车（GPU $SIDECAR_GPU；轮询 epoch_<N>.pt，触发早停） ──
HIP_VISIBLE_DEVICES="$SIDECAR_GPU" nohup python3 -m scripts.brats_p2_dev_eval watch \
    --ckpt-dir "$RUN_ROOT/ckpt" \
    --eval-root "$RUN_ROOT/dev_eval" \
    --dev-list "$PHASE_ROOT/lists/p2_mask_cond.json" \
    --raw-root "$PHASE_ROOT/raw" \
    --label-root "$PHASE_ROOT" \
    -e "$RUN_ROOT/environment_brats_p2_train.json" \
    -c "$REPO/configs/config_brats_p2_train.json" \
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

# ── 训练主进程（$TRAIN_GPUS 上 torchrun DDP，bf16；ControlNet-only，DM 冻结） ──
GPUS=(${TRAIN_GPUS//,/ })
export HIP_VISIBLE_DEVICES="$TRAIN_GPUS"
nohup torchrun --nproc_per_node="${#GPUS[@]}" -m scripts.brats_p2_finetune \
    -e "$RUN_ROOT/environment_brats_p2_train.json" \
    -c "$REPO/configs/config_brats_p2_train.json" \
    -t "$REPO/configs/config_network_rflow.json" \
    -g "${#GPUS[@]}" \
    --amp_dtype bf16 \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "train pid=$TRAIN_PID gpus=$TRAIN_GPUS log=$TRAIN_LOG"
echo "$TRAIN_PID" > "$RUN_ROOT/logs/train.pid"
echo "$SIDECAR_PID" > "$RUN_ROOT/logs/dev_eval.pid"
echo "launched $STAMP: train=$TRAIN_PID sidecar=$SIDECAR_PID"
