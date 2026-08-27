#!/bin/bash
# Issue #38 仪器推理：五挑战并行 nnUNetv2 预测（每挑战一卡）。
# 用法：bash p1_predict_all.sh [p1|p3]（默认 p1）
# TTA 保持默认开启（与 ADR-0002 校准包络一致；不传 --disable_tta）。
# 调用点已收编 ADR-0009：canonical 入口 python -m ctmr measure predict，
# 每挑战 spec 与 FrozenInstrumentCommand 的 INSTRUMENT_SPECS 一致。
set -u

MODE="${1:-p1}"
BASE=/root/private_data/l2-synth-eval
INPUT=$BASE/${MODE}_nnunet_inputs
PRED=$BASE/${MODE}_predictions
LOGS=$BASE/logs
export PYTHONPATH="$BASE/src${PYTHONPATH:+:$PYTHONPATH}"

export nnUNet_raw=/root/private_data/brats2023_nnunet
export nnUNet_preprocessed=/root/private_data/nnUNet_preprocessed
export nnUNet_results=/root/private_data/nnUNet_results
export nnUNet_compile=f
mkdir -p "$PRED" "$LOGS"

# 挑战:GPU:dataset_id:plans:config（SSA 用派生 plans/config，其余默认）
run_pred() {
  local CH=$1 GPU=$2 DSNAME=$3 PLANS=$4 CONFIG=$5
  echo "PREDICT_${MODE}_${CH}_START $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
  HIP_VISIBLE_DEVICES=$GPU python3 -m ctmr measure predict \
    -i "$INPUT/$CH" -o "$PRED/$CH" \
    -d "$DSNAME" -c "$CONFIG" -p "$PLANS" -tr nnUNetTrainer250Epochs -f 0 \
    >> "$LOGS/predict-${MODE}-$CH.log" 2>&1
  echo "PREDICT_${MODE}_${CH}_END rc=$? $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
}

run_pred GLI  0 Dataset501_BraTS2023GLI nnUNetPlans               3d_fullres &
run_pred SSA  1 Dataset502_BraTS2023SSA nnUNetPlans_SSA_bs16_v1   3d_fullres_bs16 &
run_pred MEN  2 Dataset503_BraTS2023MEN nnUNetPlans               3d_fullres &
run_pred METS 3 Dataset504_BraTS2023METS nnUNetPlans               3d_fullres &
run_pred PED  4 Dataset505_BraTS2023PED nnUNetPlans               3d_fullres &
wait
echo "ALL_PREDICTS_${MODE}_DONE $(date -u +%FT%TZ)" >> "$LOGS/predict-status.txt"
