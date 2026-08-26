#!/bin/bash
# Phase candidate training launcher — one parameterized template (ADR-0011, #111).
#
# Dispatches P1/P2/P3 from a single skeleton: env validation, RUN_ROOT layout,
# sidecar nohup (dev-eval watch), torchrun nohup, pid files. Stage differences
# ride env vars + condition blocks. The environment-json write is idempotent
# (the pre-#111 P2-only fix is now built in, hence two-sided).
#
# Usage (sugon; REPO = the controlled repo copy that phase scripts run in):
#   PHASE=p1 REPO=/root/nv-phase-57 bash scripts/brats_phase_launch_train.sh
#   PHASE=p2 REPO=/root/nv-phase-59 \
#       DM_SOURCE_CKPT=/root/private_data/brats2023_rflow_p1/ckpt/epoch_20.pt \
#       bash scripts/brats_phase_launch_train.sh
#   PHASE=p3 REPO=/root/nv-phase-61 \
#       DM_SOURCE_CKPT=<frozen P1-DM from dm_source.json> \
#       bash scripts/brats_phase_launch_train.sh
#
# Prerequisites per stage: #52 phase lists/labels/embeddings, the replay cohort
# and base ckpt (P1), the registered frozen P1-DM via DM_SOURCE_CKPT (P2/P3),
# the autoencoder, and the dev real feature bank (`brats_p{1,2,3}_dev_eval reference`).
set -euo pipefail

PHASE="${PHASE:?set PHASE to p1|p2|p3}"
REPO="${REPO:?set REPO to the controlled repo copy}"
PHASE_ROOT="${PHASE_ROOT:-/root/private_data/brats2023_rflow_phase}"

# frozen L2 instrument result flags (ADR-0003 chain) — carried by the p1/p2
# dev-eval sidecars; P3's watch does not have the flag, so p3 clears the array.
INSTRUMENT_ARGS=(
  --instrument-results "GLI=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/GLI/attempt-001"
  --instrument-results "MEN=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/MEN/attempt-001"
  --instrument-results "METS=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/METS/attempt-001"
  --instrument-results "PED=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997/results/PED/attempt-001"
  --instrument-results "SSA=/root/private_data/l2-instrument/be683eefb071022b2b62646234e4f7e469ae8dbc/results/SSA/attempt-001"
)

# ── per-stage dispatch(env var overrides land after the defaults) ──
case "$PHASE" in
  p1)
    RUN_ROOT="${RUN_ROOT:-/root/private_data/brats2023_rflow_p1}"
    TRAIN_MODULE=scripts.brats_p1_finetune
    WATCH_MODULE=scripts.brats_p1_dev_eval
    TRAIN_CONFIG=configs/config_brats_p1_train.json
    NETWORK_CONFIG=configs/config_network_rflow.json
    ENV_JSON="$RUN_ROOT/environment_brats_p1_train.json"
    DEV_LIST="${DEV_LIST:-$PHASE_ROOT/lists/p1_image_only_dev.json}"
    BASE_CKPT="${BASE_CKPT:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/diff_unet_3d_rflow-mr-brain_v1.pt}"
    AUTOENCODER="${AUTOENCODER:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/autoencoder_v1.pt}"
    WATCH_ROOT_FLAGS=(--emb-root "$PHASE_ROOT/embeddings")
    TRAIN_EXTRA_FLAGS=(--replay-list "$RUN_ROOT/lists/p1_mrrate_replay.json")
    ;;
  p2)
    RUN_ROOT="${RUN_ROOT:-/root/private_data/brats2023_rflow_p2}"
    TRAIN_MODULE=scripts.brats_p2_finetune
    WATCH_MODULE=scripts.brats_p2_dev_eval
    TRAIN_CONFIG=configs/config_brats_p2_train.json
    NETWORK_CONFIG=configs/config_network_rflow.json
    ENV_JSON="$RUN_ROOT/environment_brats_p2_train.json"
    DEV_LIST="${DEV_LIST:-$PHASE_ROOT/lists/p2_mask_cond.json}"
    DM_SOURCE_CKPT="${DM_SOURCE_CKPT:?set DM_SOURCE_CKPT to the frozen P1-DM candidate checkpoint from dm_source.json}"
    AUTOENCODER="${AUTOENCODER:-/root/private_data/manifold/models/autoencoder_v1.pt}"
    WATCH_ROOT_FLAGS=(--label-root "$PHASE_ROOT")
    TRAIN_EXTRA_FLAGS=()
    ;;
  p3)
    RUN_ROOT="${RUN_ROOT:-/root/private_data/brats2023_rflow_p3}"
    TRAIN_MODULE=scripts.brats_p3_finetune
    WATCH_MODULE=scripts.brats_p3_dev_eval
    TRAIN_CONFIG=configs/config_brats_p3_train.json
    NETWORK_CONFIG=configs/config_network_p3.json
    ENV_JSON="$RUN_ROOT/environment_brats_p3_train.json"
    DM_SOURCE_CKPT="${DM_SOURCE_CKPT:?set DM_SOURCE_CKPT to the frozen P1-DM candidate checkpoint from dm_source.json}"
    P3_PAIRS_LIST="${P3_PAIRS_LIST:-$PHASE_ROOT/lists/p3_pairs.json}"
    DEV_LIST="${DEV_LIST:-$P3_PAIRS_LIST}"
    AUTOENCODER="${AUTOENCODER:-/root/private_data/manifold/models/autoencoder_v1.pt}"
    WATCH_ROOT_FLAGS=(--phase-root "$PHASE_ROOT")
    TRAIN_EXTRA_FLAGS=()
    INSTRUMENT_ARGS=()  # P3's dev-eval watch has no --instrument-results flag
    ;;
  *)
    echo "unknown PHASE '$PHASE' (expected p1|p2|p3)" >&2
    exit 1
    ;;
esac

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6}"
SIDECAR_GPU="${SIDECAR_GPU:-7}"
EVAL_EVERY="${EVAL_EVERY:-5}"
PATIENCE="${PATIENCE:-3}"
MIN_EPOCH="${MIN_EPOCH:-30}"
MAX_EPOCH="${MAX_EPOCH:-100}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{ckpt,logs,records,tblogs}

# ── 环境配置(实例路径,契约 init 指纹的就是这份) ──
# 幂等(模板内置,双侧):契约 init 已指纹化本文件(--platform-json)。若已存在则不覆盖,
# 避免重写内容(路径/格式差异)导致运行记录的平台 sha 与文件不一致。
if [ ! -f "$ENV_JSON" ]; then
case "$PHASE" in
  p1) cat > "$ENV_JSON" <<EOF
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
  ;;
  p2) cat > "$ENV_JSON" <<EOF
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
  ;;
  p3) cat > "$ENV_JSON" <<EOF
{
    "data_base_dir": "$PHASE_ROOT",
    "json_data_list": "$P3_PAIRS_LIST",
    "model_dir": "$RUN_ROOT/ckpt",
    "trained_autoencoder_path": "$AUTOENCODER",
    "trained_diffusion_path": "$DM_SOURCE_CKPT",
    "tfevent_path": "$RUN_ROOT/tblogs",
    "exp_name": "p3_image_cond",
    "modality_mapping_path": "$REPO/configs/modality_mapping.json"
}
EOF
  ;;
esac
fi

# P1 only: MR-RATE 回放 embeddings 挂载进同一 embedding 根(rsync 落位即无需挂载)
if [ "$PHASE" = p1 ] && [ ! -f "$PHASE_ROOT/embeddings/MR-RATE" ] && [ -d "$RUN_ROOT/embeddings/MR-RATE" ]; then
    ln -sfn "$RUN_ROOT/embeddings/MR-RATE" "$PHASE_ROOT/embeddings/MR-RATE"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$RUN_ROOT/logs/train_$STAMP.log"
SIDECAR_LOG="$RUN_ROOT/logs/dev_eval_$STAMP.log"

# ── dev 评估侧车(GPU $SIDECAR_GPU;轮询 epoch_<N>.pt,触发早停) ──
HIP_VISIBLE_DEVICES="$SIDECAR_GPU" nohup python3 -m "$WATCH_MODULE" watch \
    --ckpt-dir "$RUN_ROOT/ckpt" \
    --eval-root "$RUN_ROOT/dev_eval" \
    --dev-list "$DEV_LIST" \
    --raw-root "$PHASE_ROOT/raw" \
    "${WATCH_ROOT_FLAGS[@]}" \
    -e "$ENV_JSON" \
    -c "$REPO/$TRAIN_CONFIG" \
    -t "$REPO/$NETWORK_CONFIG" \
    --eval-every "$EVAL_EVERY" --patience "$PATIENCE" \
    --min-epoch "$MIN_EPOCH" --max-epoch "$MAX_EPOCH" \
    "${INSTRUMENT_ARGS[@]+"${INSTRUMENT_ARGS[@]}"}" \
    > "$SIDECAR_LOG" 2>&1 &
SIDECAR_PID=$!
echo "sidecar pid=$SIDECAR_PID gpu=$SIDECAR_GPU log=$SIDECAR_LOG"

# ── 训练主进程($TRAIN_GPUS 上 torchrun DDP,bf16) ──
GPUS=(${TRAIN_GPUS//,/ })
export HIP_VISIBLE_DEVICES="$TRAIN_GPUS"
nohup torchrun --nproc_per_node="${#GPUS[@]}" -m "$TRAIN_MODULE" \
    -e "$ENV_JSON" \
    -c "$REPO/$TRAIN_CONFIG" \
    -t "$REPO/$NETWORK_CONFIG" \
    -g "${#GPUS[@]}" \
    "${TRAIN_EXTRA_FLAGS[@]+"${TRAIN_EXTRA_FLAGS[@]}"}" \
    --amp_dtype bf16 \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "train pid=$TRAIN_PID gpus=$TRAIN_GPUS log=$TRAIN_LOG"
echo "$TRAIN_PID" > "$RUN_ROOT/logs/train.pid"
echo "$SIDECAR_PID" > "$RUN_ROOT/logs/dev_eval.pid"
echo "launched $STAMP: train=$TRAIN_PID sidecar=$SIDECAR_PID"
