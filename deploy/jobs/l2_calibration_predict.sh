#!/bin/bash
# Issue #36 校准推理编排：5 子挑战 × 3 次独立重复，4 卡工作槽消费任务队列。
# 每任务 = (challenge, rep)：canonical 入口 python -m ctmr.instrument.predict
# （ADR-0009 #108 收编），冻结配置 = 入口默认（镜像 TTA on、滑窗 overlap/
# step_size 0.5，见 docs/calibration/l2-instrument-calibration-protocol.md §3）。
# 幂等：输出齐全即跳过。ADR-0002 历史包络不重跑。
set -u
CALIB_BASE=${CALIB_BASE:?set CALIB_BASE}
REPO_COMMIT=${REPO_COMMIT:?set REPO_COMMIT}
TRAIN_BASE_DEFAULT=/root/private_data/l2-instrument/52667a345ec9e1885a983bb2b8f063aa0827e997
TRAIN_BASE_SSA=/root/private_data/l2-instrument/be683eefb071022b2b62646234e4f7e469ae8dbc
export PYTHONPATH="$CALIB_BASE/src${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_compile=f
export nnUNet_raw=/root/private_data/brats2023_nnunet
export nnUNet_preprocessed=/root/private_data/nnUNet_preprocessed
LOGS="$CALIB_BASE/logs"; mkdir -p "$LOGS"
STATUS="$LOGS/predict-status.txt"; touch "$STATUS"

declare -A DATASET=( [GLI]=Dataset501_BraTS2023GLI [SSA]=Dataset502_BraTS2023SSA \
  [MEN]=Dataset503_BraTS2023MEN [METS]=Dataset504_BraTS2023METS [PED]=Dataset505_BraTS2023PED )
declare -A RESULTS=( [GLI]="$TRAIN_BASE_DEFAULT/results/GLI/attempt-001" \
  [MEN]="$TRAIN_BASE_DEFAULT/results/MEN/attempt-001" \
  [METS]="$TRAIN_BASE_DEFAULT/results/METS/attempt-001" \
  [PED]="$TRAIN_BASE_DEFAULT/results/PED/attempt-001" \
  [SSA]="$TRAIN_BASE_SSA/results/SSA/attempt-001" )
declare -A PLANS=( [GLI]=nnUNetPlans [MEN]=nnUNetPlans [METS]=nnUNetPlans [PED]=nnUNetPlans \
  [SSA]=nnUNetPlans_SSA_bs16_v1 )
declare -A CONFIG=( [GLI]=3d_fullres [MEN]=3d_fullres [METS]=3d_fullres [PED]=3d_fullres \
  [SSA]=3d_fullres_bs16 )

# 任务文件：大挑战在前（均匀化每卡负载）；重跑脚本只需恢复该文件。
TASKFILE="$CALIB_BASE/predict_tasks.txt"
if [ ! -f "$TASKFILE" ]; then
  : > "$TASKFILE"
  for REP in 1 2 3; do
    for CH in GLI MEN METS SSA PED; do echo "$CH $REP" >> "$TASKFILE"; done
  done
fi

predict_one() {  # $1=challenge $2=rep $3=slot_gpu
  local CH=$1 REP=$2 GPU=$3
  local INPUT="$CALIB_BASE/inputs/$CH"
  local OUT="$CALIB_BASE/predictions/$CH/rep$REP"
  local N_IN N_OUT
  N_IN=$(ls "$INPUT" | grep -c '_0000\.nii\.gz$')
  N_OUT=$(ls "$OUT" 2>/dev/null | grep -c '\.nii\.gz$' || true)
  if [ "$N_OUT" = "$N_IN" ] && [ "$N_IN" -gt 0 ]; then
    echo "SKIP $CH rep$REP (done: $N_OUT) $(date -u +%FT%TZ)" >> "$STATUS"
    return 0
  fi
  mkdir -p "$OUT"
  echo "START $CH rep$REP gpu$GPU $(date -u +%FT%TZ)" >> "$STATUS"
  HIP_VISIBLE_DEVICES=$GPU \
    nnUNet_results="${RESULTS[$CH]}" \
    python3 -m ctmr.instrument.predict \
      -i "$INPUT" -o "$OUT" \
      -d "${DATASET[$CH]}" -c "${CONFIG[$CH]}" -p "${PLANS[$CH]}" \
      -tr nnUNetTrainer250Epochs -f 0 \
      > "$LOGS/predict-$CH-rep$REP.log" 2>&1
  local RC=$?
  N_OUT=$(ls "$OUT" | grep -c '\.nii\.gz$' || true)
  echo "END $CH rep$REP rc=$RC outputs=$N_OUT/$N_IN $(date -u +%FT%TZ)" >> "$STATUS"
}

LOCK="$CALIB_BASE/predict_tasks.lock"
for SLOT in 0 1 2 3; do
  GPU=$((SLOT * 2))  # 0/2/4/6
  (
    while :; do
      TASK=$(flock "$LOCK" bash -c "head -n1 '$TASKFILE' 2>/dev/null; sed -i 1d '$TASKFILE' 2>/dev/null")
      [ -z "$TASK" ] && break
      set -- $TASK
      predict_one "$1" "$2" "$GPU"
    done
  ) &
done
wait
echo "ALL_PREDICT_DONE $(date -u +%FT%TZ)" >> "$STATUS"
