#!/bin/bash
# 复核作业 A(issue #217,父 #205):z-crop 几何基座复核(以工件 affine 为准)——sugon 只读作业
#
# 用途:作业 C(#208/PR #216)执行时发现 holdout 生成 NIfTI 工件携带单位 1mm affine
#   (sidecar 写出约定 np.diag([1,1,1])),作业 A 注册的「1mm 重采样后 z 174 层、居中裁
#   19 层(crop_start=9)」在该批工件上不发生——实为 1mm 重采样 no-op + z 向 pad 前 13
#   后 14。本作业把同一批 L2 终验逐观测预测掩码按两种几何(注册 crop_start=9 与工件
#   pad 13/14)各重算一遍 centroid_wt_z,产出对照读数、越窗重判清单与视场缺口定量。
#   variant=diagnostic:不产生任何验收判定,不动冻结仪器,不回改已落盘的作业 A。
#
# 用法:
#   bash deploy/jobs/run_zcrop_geometry_audit_217.sh
#
# 环境变量(均可覆写,默认自动探测):
#   L2_RUN_TREE       L2 终验运行树根(默认 /root/private_data/ctmr/brats2023_rflow_p1/l2_acceptance)
#   MEASUREMENTS_CSV  逐观测测量 CSV(受控存储;缺省在运行树下 find measurements*.csv)
#   PREDICT_DIR       逐观测分割 mask 目录(缺省 find -type d -name predictions)
#   OUTPUT_DIR        报告工件输出目录(默认 $L2_RUN_TREE/diagnostics/zcrop_geometry_audit;
#                     sugon 工件区,不入 git)
#   BOOTSTRAP_B       bootstrap 重采样数(默认 10000)
#
# 前置条件:
#   1. L2 终验运行树的逐观测工件仍在(测量 CSV + predictions/ 分割 mask)
#   2. 环境已激活(numpy、scipy、SimpleITK 即可——纯 CPU 重算,不需要 DCU 卡)
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_zcrop_compensation_a.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

L2_RUN_TREE="${L2_RUN_TREE:-/root/private_data/ctmr/brats2023_rflow_p1/l2_acceptance}"
OUTPUT_DIR="${OUTPUT_DIR:-$L2_RUN_TREE/diagnostics/zcrop_geometry_audit}"

# ── 探测测量 CSV(逐观测工件,受控存储)──
if [ -z "${MEASUREMENTS_CSV:-}" ]; then
    MEASUREMENTS_CSV="$(find "$L2_RUN_TREE" -name 'measurements*.csv' -not -path '*/.*' 2>/dev/null | head -1 || true)"
    [ -n "$MEASUREMENTS_CSV" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到测量 CSV — 请以 MEASUREMENTS_CSV=... 显式指定" >&2; exit 1; }
fi
[ -f "$MEASUREMENTS_CSV" ] || { echo "[FATAL] 测量 CSV 不存在: $MEASUREMENTS_CSV" >&2; exit 1; }

# ── 探测预测目录(逐观测分割 mask)──
if [ -z "${PREDICT_DIR:-}" ]; then
    PREDICT_DIR="$(find "$L2_RUN_TREE" -type d -name predictions 2>/dev/null | head -1 || true)"
    [ -n "$PREDICT_DIR" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到 predictions 目录 — 请以 PREDICT_DIR=... 显式指定" >&2; exit 1; }
fi
[ -d "$PREDICT_DIR" ] || { echo "[FATAL] 预测目录不存在: $PREDICT_DIR" >&2; exit 1; }

# ── run id(从终验 json 的 binding 读取,读不到则以未绑定落盘,不阻塞)──
RUN_ID_ARGS=()
ACCEPTANCE_JSON="$L2_RUN_TREE/evaluate_v1/l2_final_acceptance_p1.json"
if [ -f "$ACCEPTANCE_JSON" ]; then
    RUN_ID="$(python -c "import json; print(json.load(open('$ACCEPTANCE_JSON')).get('binding', {}).get('run_id') or '')" 2>/dev/null || true)"
    if [ -n "$RUN_ID" ]; then
        RUN_ID_ARGS=(--run-id "$RUN_ID")
    fi
fi

echo "============================================"
echo "复核作业 A:z-crop 几何基座复核(#217)"
echo "运行树: $L2_RUN_TREE"
echo "测量 CSV: $MEASUREMENTS_CSV"
echo "预测目录: $PREDICT_DIR"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

python -m ctmr.application.acceptance.distribution.zcrop_geometry_audit \
    --measurements "$MEASUREMENTS_CSV" \
    --pred-root "$PREDICT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"}

echo ""
echo "============================================"
echo "  完成。复核读数报告:"
echo "    $OUTPUT_DIR/zcrop_geometry_audit_diagnostic.json"
echo "    $OUTPUT_DIR/zcrop_geometry_audit_diagnostic.md"
echo "  读数供收编票转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
