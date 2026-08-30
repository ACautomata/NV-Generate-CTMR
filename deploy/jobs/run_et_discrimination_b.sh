#!/bin/bash
# 诊断作业 B(issue #207,父 #205):纯生成样本冻结仪器 ET 甄别——sugon 只读作业
#
# 用途:从模态标签条件生成候选(实验台账阶段代号 P1)的 L2 终验运行树出发,把 holdout
#   530 例(冻结配额 250 GLI + 200 MEN + 48 METS + 20 PED + 12 SSA)生成伪四模态体
#   已产出的逐观测仪器读数(measurements CSV 的 vol_et_ml/pred_empty)重算为逐挑战
#   ET 检出率、ET 体积分布 vs real、空 pred 计数——读数口径与 #38 合成域评估同族,
#   可与 P1 直出(METS 空 pred 2/20)与 img2img 基线(METS 42/80 + MEN 2/80)横向对比。
#   variant=diagnostic:不产生任何验收判定,不动冻结仪器与包络,零推理(纯 CPU 读数)。
#
# 用法:
#   bash deploy/jobs/run_et_discrimination_b.sh
#
# 环境变量(均可覆写,默认自动探测):
#   L2_RUN_TREE       L2 终验运行树根(默认 /root/private_data/ctmr/runs/p1/l2_acceptance)
#   MEASUREMENTS_CSV  逐观测测量 CSV(受控存储;缺省在运行树下 find measurements*.csv)
#   OUTPUT_DIR        报告工件输出目录(默认 $L2_RUN_TREE/diagnostics/et_discrimination;
#                     sugon 工件区,不入 git)
#   BOOTSTRAP_B       bootstrap 重采样数(默认 10000)
#
# 前置条件:
#   1. L2 终验运行树的逐观测测量 CSV 仍在(生成侧已过冻结仪器,无需任何推理)
#   2. 环境已激活(python 3.11+ 与 ctmr 包即可——纯 CPU 读数,不需要 DCU 卡)
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_zcrop_compensation_a.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

L2_RUN_TREE="${L2_RUN_TREE:-/root/private_data/ctmr/runs/p1/l2_acceptance}"
OUTPUT_DIR="${OUTPUT_DIR:-$L2_RUN_TREE/diagnostics/et_discrimination}"

# ── 探测测量 CSV(逐观测工件,受控存储)──
if [ -z "${MEASUREMENTS_CSV:-}" ]; then
    MEASUREMENTS_CSV="$(find "$L2_RUN_TREE" -name 'measurements*.csv' -not -path '*/.*' 2>/dev/null | head -1 || true)"
    [ -n "$MEASUREMENTS_CSV" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到测量 CSV — 请以 MEASUREMENTS_CSV=... 显式指定" >&2; exit 1; }
fi
[ -f "$MEASUREMENTS_CSV" ] || { echo "[FATAL] 测量 CSV 不存在: $MEASUREMENTS_CSV" >&2; exit 1; }

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
echo "诊断作业 B:纯生成样本冻结仪器 ET 甄别(#207)"
echo "运行树: $L2_RUN_TREE"
echo "测量 CSV: $MEASUREMENTS_CSV"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

python -m ctmr.application.acceptance.distribution.et_discrimination \
    --measurements "$MEASUREMENTS_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"}

echo ""
echo "============================================"
echo "  完成。ET 甄别读数报告:"
echo "    $OUTPUT_DIR/et_discrimination_diagnostic.json"
echo "    $OUTPUT_DIR/et_discrimination_diagnostic.md"
echo "  读数供收编票转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
