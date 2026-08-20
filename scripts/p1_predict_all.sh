#!/bin/bash
# Issue #38 仪器推理：五挑战并行 nnUNetv2 预测（每挑战一卡）。
# 用法：bash p1_predict_all.sh [p1|p3]（默认 p1）
# TTA 保持默认开启（与 ADR-0002 校准包络一致；不传 --disable_tta）。
set -u

MODE="${1:-p1}"
BASE=/root/private_data/l2-synth-eval
INPUT=$BASE/${MODE}_nnunet_inputs
PRED=$BASE/${MODE}_predictions
LOGS=$BASE/logs
ENTRY=$BASE/scripts/l2_calibration_predict_entry.py

export nnUNet_raw=/root/private_data/brats2023_nnunet
export nnUNet_preprocessed=/root/private_data/nnUNet_preprocessed
export nnUNet_results=/root/private_data/nnUNet_results
export nnUNet_compile=f
mkdir -p "$PRED" "$LOGS"

# 挑战:GPU:dataset_id:plans:config（SSA 用派生 plans/config，其余默认）
run_pred() {
  local CH=$1 GPU=$2 DSID=$3 PLANS=$4 CONFIG=$5
  echo "PREDICT_${MODE}_${CH}_START $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
  HIP_VISIBLE_DEVICES=$GPU python3 "$ENTRY" \
    -i "$INPUT/$CH" -o "$PRED/$CH" \
    -d "$DSID" -p "$PLANS" -c "$CONFIG" -f 0 \
    -tr nnUNetTrainer250Epochs \
    >> "$LOGS/predict-${MODE}-$CH.log" 2>&1
  echo "PREDICT_${MODE}_${CH}_END rc=$? $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
}

run_pred GLI  0 501 nnUNetPlans               3d_fullres &
run_pred SSA  1 502 nnUNetPlans_SSA_bs16_v1   3d_fullres_bs16 &
run_pred MEN  2 503 nnUNetPlans               3d_fullres &
run_pred METS 3 504 nnUNetPlans               3d_fullres &
run_pred PED  4 505 nnUNetPlans               3d_fullres &
wait
echo "ALL_PREDICTS_${MODE}_DONE $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
