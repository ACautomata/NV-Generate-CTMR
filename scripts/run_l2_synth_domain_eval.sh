#!/bin/bash
# Issue #38 L2 仪器合成域适用性评估——sugon DCU 集群执行脚本
#
# 用法：在 sugon DCU 集群上执行
#   bash scripts/run_l2_synth_domain_eval.sh [p1|p3|all]
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

        DATASET_ID=$(case "${CHALLENGE}" in
            GLI) echo 501 ;; SSA) echo 502 ;; MEN) echo 503 ;;
            METS) echo 504 ;; PED) echo 505 ;;
        esac

        EXTRA_FLAGS=""
        if [ "${CHALLENGE}" = "SSA" ]; then
            EXTRA_FLAGS="-p nnUNetPlans_SSA_bs16_v1"
        fi

        echo "  [PREDICT] ${CHALLENGE} (Dataset${DATASET_ID})..."
        nnUNetv2_predict_from_raw_data \
            -i "${INPUT_DIR}" \
            -o "${PRED_DIR}" \
            -d "${DATASET_ID}" \
            -c 3d_fullres \
            -f 0 \
            -tr nnUNetTrainer250Epochs \
            --disable_tta False \
            ${EXTRA_FLAGS} \
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
