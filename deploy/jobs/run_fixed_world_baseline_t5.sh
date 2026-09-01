#!/bin/bash
# 序列②T5(issue #252,父 #247):修复后世界基线重跑——sugon 只读 CPU 作业
#
# 用途:#249 写出协议修复(生成 NIfTI sidecar affine 改真实采样 spacing)落地后,对同一
#   holdout 530 生成工件按修复后写出世界重算仪器读数:14 个注册 L2 量族逐 case 重配对
#   (crop 重叠窗 [9,155) mm,声明域 [0,174) mm),作业 B ET 甄别协议复用,作业 A/作业 B
#   记录字面量锚点对账。产出 #247 终验读数与旧 FAIL(pad 世界)读数之间缺失的「修复后
#   世界」历史候选基线。variant=diagnostic:零验收判定,零推理(不重跑冻结仪器分割面),
#   冻结仪器、包络、判定线零接触;上缘声明域外质量预期归零(146 例/431 ml 错配消除)。
#
# 用法:
#   bash deploy/jobs/run_fixed_world_baseline_t5.sh
#
# 环境变量(均可覆写,默认自动探测):
#   L2_RUN_TREE       L2 终验运行树根(默认 /root/private_data/ctmr/runs/p1/l2_acceptance)
#   MEASUREMENTS_CSV  逐观测测量 CSV(受控存储;缺省在运行树下 find measurements*.csv)
#   PREDICT_DIR       逐观测分割 mask 目录(缺省 find -type d -name predictions)
#   INPUTS_DIR        仪器输入目录(wt_brain/et_wt 量族的脑union;缺省 find -type d -name inputs)
#   GEN_ROOT          holdout 生成源工件树(修复后世界前提验证面;缺省 $L2_RUN_TREE/../holdout_generated/generated)
#   OUTPUT_DIR        报告工件输出目录(默认 $L2_RUN_TREE/diagnostics/fixed_world_baseline;
#                     sugon 工件区,不入 git)
#   BOOTSTRAP_B       bootstrap 重采样数(默认 10000;CI 锚点按 B=10000 记录格对账,改动即漂移)
#
# 前置条件:
#   1. L2 终验运行树的逐观测工件仍在(测量 CSV + predictions/ + inputs/)
#   2. 环境已激活(numpy、scipy、SimpleITK 即可——纯 CPU 重算,不需要 DCU 卡)
#   3. 锚点对账失败(退出码 1)= 窗口算术/工件/种子漂移,读数不可作为记录历史对偶
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_zcrop_compensation_a.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

L2_RUN_TREE="${L2_RUN_TREE:-/root/private_data/ctmr/runs/p1/l2_acceptance}"
OUTPUT_DIR="${OUTPUT_DIR:-$L2_RUN_TREE/diagnostics/fixed_world_baseline}"

# ── 探测测量 CSV(逐观测工件,受控存储)──
if [ -z "${MEASUREMENTS_CSV:-}" ]; then
    MEASUREMENTS_CSV="$(find "$L2_RUN_TREE" -maxdepth 1 -name 'measurements*.csv' -not -path '*/.*' 2>/dev/null | head -1 || true)"
    [ -n "$MEASUREMENTS_CSV" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到测量 CSV — 请以 MEASUREMENTS_CSV=... 显式指定" >&2; exit 1; }
fi
[ -f "$MEASUREMENTS_CSV" ] || { echo "[FATAL] 测量 CSV 不存在: $MEASUREMENTS_CSV" >&2; exit 1; }

# ── 探测预测目录(逐观测分割 mask)──
if [ -z "${PREDICT_DIR:-}" ]; then
    PREDICT_DIR="$(find "$L2_RUN_TREE" -type d -name predictions 2>/dev/null | head -1 || true)"
    [ -n "$PREDICT_DIR" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到 predictions 目录 — 请以 PREDICT_DIR=... 显式指定" >&2; exit 1; }
fi
[ -d "$PREDICT_DIR" ] || { echo "[FATAL] 预测目录不存在: $PREDICT_DIR" >&2; exit 1; }

# ── 探测仪器输入目录(wt_brain/et_wt 量族的脑 union)──
if [ -z "${INPUTS_DIR:-}" ]; then
    INPUTS_DIR="$(find "$L2_RUN_TREE" -type d -name inputs 2>/dev/null | head -1 || true)"
    [ -n "$INPUTS_DIR" ] || { echo "[FATAL] $L2_RUN_TREE 下未找到 inputs 目录 — 请以 INPUTS_DIR=... 显式指定" >&2; exit 1; }
fi
[ -d "$INPUTS_DIR" ] || { echo "[FATAL] 仪器输入目录不存在: $INPUTS_DIR" >&2; exit 1; }

# ── 探测 holdout 生成源工件树(修复后世界前提验证面)──
GEN_ROOT="${GEN_ROOT:-$L2_RUN_TREE/../holdout_generated/generated}"
[ -d "$GEN_ROOT" ] || { echo "[FATAL] 生成源工件树不存在: $GEN_ROOT — 请以 GEN_ROOT=... 显式指定" >&2; exit 1; }

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
echo "序列②T5:修复后世界基线重跑(#252,父 #247)"
echo "运行树: $L2_RUN_TREE"
echo "测量 CSV: $MEASUREMENTS_CSV"
echo "预测目录: $PREDICT_DIR"
echo "输入目录: $INPUTS_DIR"
echo "生成源树: $GEN_ROOT"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 零验收判定、零推理"
echo "============================================"

rc=0
python -m ctmr.application.acceptance.distribution.fixed_world_baseline \
    --measurements "$MEASUREMENTS_CSV" \
    --pred-root "$PREDICT_DIR" \
    --inputs-root "$INPUTS_DIR" \
    --gen-root "$GEN_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"} "$@" || rc=$?

echo ""
echo "============================================"
if [ "$rc" -eq 0 ]; then
    echo "  完成。修复后世界基线读数报告:"
else
    echo "  锚点对账漂移(退出码 1)——报告已写出供取证:"
fi
echo "    $OUTPUT_DIR/fixed_world_baseline_diagnostic.json"
echo "    $OUTPUT_DIR/fixed_world_baseline_diagnostic.md"
echo "  读数供 #252 落盘 deploy/experiments/(工件本身不入 git)"
echo "============================================"
exit $rc
