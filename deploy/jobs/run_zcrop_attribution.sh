#!/bin/bash
# Issue #206 诊断作业 A:z-crop 补偿重算归因——sugon 集群执行脚本
#
# 用法:bash deploy/jobs/run_zcrop_attribution.sh [工件根目录]
#   默认 /root/private_data/brats2023_rflow_p1/l2_acceptance/evaluate_v1
#
# 前置条件(该目录下需已有 L2 终验工件):
#   plan.json / measurements.csv / predictions/<CH>/<obs_id>.nii.gz
#   l2_final_acceptance_p1.json
#
# 本作业 variant=diagnostic:只读重算与归因读数,不产生任何 L1/L2/L3
# 验收判定;冻结仪器与包络全程不动。原始报告落 sugon 工件区,不入 git。
set -euo pipefail

# 包内模块 python -m 需要本检出的 src 树在 sys.path(与 #38
# run_l2_synth_domain_eval.sh 同族 shim;#131 迁入 deploy/jobs/ 后 src 经
# ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

EVAL_ROOT="${1:-/root/private_data/brats2023_rflow_p1/l2_acceptance/evaluate_v1}"
PLAN="${EVAL_ROOT}/plan.json"
TABLE="${EVAL_ROOT}/measurements.csv"
PREDS="${EVAL_ROOT}/predictions"
REPORT="${EVAL_ROOT}/l2_final_acceptance_p1.json"
OUTPUT_DIR="${EVAL_ROOT}/zcrop_attribution"

for artifact in "$PLAN" "$TABLE" "$REPORT"; do
    [ -f "$artifact" ] || { echo "[FATAL] missing artifact: $artifact" >&2; exit 1; }
done
[ -d "$PREDS" ] || { echo "[FATAL] missing predictions dir: $PREDS" >&2; exit 1; }

echo "============================================"
echo "诊断作业 A:z-crop 补偿重算归因 (#206)"
echo "variant=diagnostic — 不产生验收判定"
echo "工件根目录: ${EVAL_ROOT}"
echo "============================================"

python -m ctmr.application.acceptance.distribution.zcrop_attribution \
    --plan "${PLAN}" \
    --table "${TABLE}" \
    --preds "${PREDS}" \
    --report "${REPORT}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "报告: ${OUTPUT_DIR}/zcrop_attribution_report.json / .md"
echo "完成时间: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
