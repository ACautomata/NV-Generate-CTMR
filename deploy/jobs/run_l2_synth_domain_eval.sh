#!/bin/bash
# Issue #38 L2 仪器合成域适用性评估——sugon DCU 集群执行脚本
#
# 用法：在 sugon DCU 集群上执行
#   bash deploy/jobs/run_l2_synth_domain_eval.sh [p1|p3|all]
#
# 前置条件：
#   1. v1 DM 权重已部署到 /root/private_data/models/
#      - diff_unet_3d_rflow-mr-brain_v1.pt
#      - autoencoder_v1.pt
#   2. nnU-Net 仪器已部署到 /root/private_data/nnUNet_results/
#   3. BraTS nnU-Net 数据集在 /root/private_data/brats2023_nnunet/
#   4. 环境已激活（torch-dcu, monai, nnunetv2, SimpleITK 等）
#
set -euo pipefail

# canonical 入口 python -m ctmr.instrument.predict 与 python -m scripts.* 需要
# src 树与仓库根在 sys.path（ADR-0009 #108 收编；两种部署形态的同族 shim）。
# #131 迁入 deploy/jobs/ 后 src 经 ../../ 解析——一层 ../ 会落到 deploy/src。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

EVAL_ROOT="/root/private_data/l2-synth-eval"
NNUNET_ROOT="/root/private_data/brats2023_nnunet"
V1_MODEL_DIR="/root/private_data/models"
RESULTS_ROOT="/root/private_data/nnUNet_results"
CAL_METRICS_DIR="/root/private_data/l2-instrument-calibration"
MODE="${1:-all}"

echo "============================================"
echo "L2 仪器合成域适用性评估 (#38)"
echo "模式: ${MODE}"
echo "评估根目录: ${EVAL_ROOT}"
echo "============================================"

# ── Step 0: 创建病例列表 ──
echo ""
echo "[Step 0] 创建评估病例列表..."
python -m scripts.nnunet_l2_synthetic_domain_eval create-case-lists \
    --nnunet-root "${NNUNET_ROOT}" \
    --output-dir "${EVAL_ROOT}/case_lists"

# ── 函数：评估一个模式 ──
evaluate_mode() {
    local MODE_NAME=$1
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  模式: ${MODE_NAME}"
    echo "═══════════════════════════════════════════"

    # Step 1: 生成 v1 DM 直出样本
    echo ""
    echo "[Step 1] 生成 v1 DM 直出样本 (${MODE_NAME})..."
    python -m scripts.nnunet_l2_synthetic_domain_eval generate \
        --mode "${MODE_NAME}" \
        --case-list "${EVAL_ROOT}/case_lists/${MODE_NAME}_cases.json" \
        --v1-model-dir "${V1_MODEL_DIR}" \
        --output-dir "${EVAL_ROOT}/${MODE_NAME}_samples"

    # Step 2: 组装 nnU-Net 输入
    echo ""
    echo "[Step 2] 组装 nnU-Net 输入..."
    python -m scripts.nnunet_l2_synthetic_domain_eval prep-inputs \
        --sample-dir "${EVAL_ROOT}/${MODE_NAME}_samples" \
        --nnunet-root "${NNUNET_ROOT}" \
        --output-dir "${EVAL_ROOT}/${MODE_NAME}_nnunet_inputs"

    # Step 3: 运行冻结仪器推理
    echo ""
    echo "[Step 3] 运行冻结仪器推理..."
    # 对每个子挑战分别执行 nnU-Net 推理
    for CHALLENGE in GLI SSA MEN METS PED; do
        INPUT_DIR="${EVAL_ROOT}/${MODE_NAME}_nnunet_inputs/${CHALLENGE}"
        PRED_DIR="${EVAL_ROOT}/${MODE_NAME}_predictions/${CHALLENGE}"
        mkdir -p "${PRED_DIR}"

        if [ ! -d "${INPUT_DIR}" ]; then
            echo "  [SKIP] ${CHALLENGE}: input dir not found"
            continue
        fi

        # 每挑战 spec 与 FrozenInstrumentCommand 的 INSTRUMENT_SPECS 逐字一致
        # （ADR-0009 #108 收编：canonical 入口、TTA on 靠省略、无 fatal token）。
        declare -A DATASET_NAME=( [GLI]=Dataset501_BraTS2023GLI [SSA]=Dataset502_BraTS2023SSA \
          [MEN]=Dataset503_BraTS2023MEN [METS]=Dataset504_BraTS2023METS [PED]=Dataset505_BraTS2023PED )
        declare -A PLANS=( [GLI]=nnUNetPlans [SSA]=nnUNetPlans_SSA_bs16_v1 [MEN]=nnUNetPlans \
          [METS]=nnUNetPlans [PED]=nnUNetPlans )
        declare -A CONFIG=( [GLI]=3d_fullres [SSA]=3d_fullres_bs16 [MEN]=3d_fullres \
          [METS]=3d_fullres [PED]=3d_fullres )

        echo "  [PREDICT] ${CHALLENGE} (${DATASET_NAME[$CHALLENGE]})..."
        python3 -m ctmr.instrument.predict \
            -i "${INPUT_DIR}" \
            -o "${PRED_DIR}" \
            -d "${DATASET_NAME[$CHALLENGE]}" \
            -c "${CONFIG[$CHALLENGE]}" \
            -p "${PLANS[$CHALLENGE]}" \
            -tr nnUNetTrainer250Epochs \
            -f 0 \
            2>&1 | tee "${PRED_DIR}/predict.log" || {
                echo "  [WARN] ${CHALLENGE}: prediction failed, continuing..."
            }
    done

    # Step 4: 计算指标 + 生成报告
    echo ""
    echo "[Step 4] 计算指标并生成报告..."
    python -m scripts.nnunet_l2_synthetic_domain_eval evaluate \
        --sample-dir "${EVAL_ROOT}/${MODE_NAME}_samples" \
        --input-dir "${EVAL_ROOT}/${MODE_NAME}_nnunet_inputs" \
        --pred-dir "${EVAL_ROOT}/${MODE_NAME}_predictions" \
        --calibration-summary "${CAL_METRICS_DIR}" \
        --output-dir "${EVAL_ROOT}/report_${MODE_NAME}"

    echo ""
    echo "═══════════════════════════════════════════"
    echo "  ${MODE_NAME} 评估完成"
    echo "  报告: ${EVAL_ROOT}/report_${MODE_NAME}/"
    echo "═══════════════════════════════════════════"
}

# ── 执行 ──
case "${MODE}" in
    p1)
        evaluate_mode "p1"
        ;;
    p3)
        evaluate_mode "p3"
        ;;
    all)
        evaluate_mode "p1"
        evaluate_mode "p3"
        ;;
    *)
        echo "Usage: $0 [p1|p3|all]"
        exit 1
        ;;
esac

# ── 汇总 ──
echo ""
echo "============================================"
echo "  评估完成汇总"
echo "============================================"
for REPORT_DIR in "${EVAL_ROOT}"/report_*/; do
    if [ -d "${REPORT_DIR}" ]; then
        REPORT_FILE="${REPORT_DIR}"*.json
        if [ -f ${REPORT_FILE} ]; then
            echo ""
            echo "报告: ${REPORT_FILE}"
            python3 -c "
import json, sys
with open('${REPORT_FILE}') as f:
    r = json.load(f)
print(f\"  模式: {r['mode']}\")
print(f\"  总体判定: {r['overall_verdict']}\")
for ch, d in r['per_challenge'].items():
    s = d['r_fail_synth']
    print(f\"  {ch}: R_fail_synth={s['point']:.4f} ({s['k']}/{s['n']}) → {d['verdict']}\")
" 2>/dev/null || echo "  (报告解析失败)"
        fi
    fi
done

echo ""
echo "完成时间: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
