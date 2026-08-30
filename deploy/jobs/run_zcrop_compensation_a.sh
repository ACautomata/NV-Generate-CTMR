#!/bin/bash
# 诊断作业 A(issue #206,父 #205):z-crop 补偿重算(测量轴归因)——sugon 只读作业
#
# 用途:从模态标签条件生成候选(实验台账阶段代号 P1)的 L2 终验运行树出发,把生成与
#   真实两侧都限制到重叠 z 物理域 [9,155) mm 后重算 WT 相对体积与质心 z,产出「测量轴
#   vs 候选缺陷」归因读数报告。variant=diagnostic:不产生任何验收判定,不动冻结仪器。
#
# 用法:
#   bash deploy/jobs/run_zcrop_compensation_a.sh
#
# 环境变量(均可覆写,默认自动探测):
#   L2_RUN_TREE       L2 终验运行树根(默认 /root/private_data/ctmr/brats2023_rflow_p1/l2_acceptance)
#   MEASUREMENTS_CSV  逐观测测量 CSV(受控存储;缺省在运行树下 find measurements*.csv)
#   PREDICT_DIR       逐观测分割 mask 目录(缺省 find -type d -name predictions)
#   OUTPUT_DIR        报告工件输出目录(默认 $L2_RUN_TREE/diagnostics/zcrop_compensation;
#                     sugon 工件区,不入 git)
#   BOOTSTRAP_B       bootstrap 重采样数(默认 10000)
#
# 前置条件:
#   1. L2 终验运行树的逐观测工件仍在(测量 CSV + predictions/ 分割 mask)
#   2. 环境已激活(numpy、scipy、SimpleITK 即可——纯 CPU 重算,不需要 DCU 卡)
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_l2_synth_domain_eval.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

L2_RUN_TREE="${L2_RUN_TREE:-/root/private_data/ctmr/brats2023_rflow_p1/l2_acceptance}"
OUTPUT_DIR="${OUTPUT_DIR:-$L2_RUN_TREE/diagnostics/zcrop_compensation}"

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
echo "诊断作业 A:z-crop 补偿重算(测量轴归因,#206)"
echo "运行树: $L2_RUN_TREE"
echo "测量 CSV: $MEASUREMENTS_CSV"
echo "预测目录: $PREDICT_DIR"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

python -m ctmr.application.acceptance.distribution.zcrop_compensation \
    --measurements "$MEASUREMENTS_CSV" \
    --pred-root "$PREDICT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"}

echo ""
echo "============================================"
echo "  完成。归因读数报告:"
echo "    $OUTPUT_DIR/zcrop_compensation_diagnostic.json"
echo "    $OUTPUT_DIR/zcrop_compensation_diagnostic.md"
echo "  读数供收编票转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
