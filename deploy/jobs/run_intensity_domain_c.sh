#!/bin/bash
# 诊断作业 C(issue #208,父 #205):t1c 强度域甄别——sugon 只读作业
#
# 用途:把「t1c 亮核在编码/输出域是否存活」变成定量读数,三个裁决一次做完:
#   ① real / VAE 重建(既有 fp32 训练 embedding 直接 decode,现网协议臂)/ 生成
#     三方直方图,按瘤内(WT) vs 全脑分层读 P99/P99.9/top-0.5% 均值;
#   ② VAE 重建误差按「输入 >1.0 vs [0,1]」体素分层算条件 MAE,clip=True 归一化
#     编码对照臂复用同掩码,直接裁决归一化 clip 取舍(需要 DCU 或 CPU 跑一遍 encode);
#   ③ 生成 t1c 输出 >1000 的 int16 体素占比(全脑/瘤内/ET),检验「亮核在 >1.0
#     输出域、被评估 clip 掉」假说的材料基础。
#   variant=diagnostic:不产生任何验收判定,不动冻结仪器与包络。
#
# 用法:
#   bash deploy/jobs/run_intensity_domain_c.sh            # 两池全量(CPU 慢,DCU 快)
#   LIMIT=200 bash deploy/jobs/run_intensity_domain_c.sh  # 每池均匀抽样 200 例
#   LIMIT=200 GEN_LIMIT= bash deploy/jobs/run_intensity_domain_c.sh  # emb 抽样 200,gen 全量
#   SKIP_EMB_POOL=1 bash deploy/jobs/run_intensity_domain_c.sh   # 只跑 gen 池(零推理)
#
# 环境变量(均可覆写,默认为 P1 台账路径):
#   PHASE_ROOT        阶段数据根(默认 /root/private_data/brats2023_rflow_phase)
#   P1_ROOT           P1 运行根(默认 /root/private_data/brats2023_rflow_p1)
#   TRAIN_LIST        训练 list(默认 $PHASE_ROOT/lists/p1_image_only.json)
#   DATA_ROOT         训练 raw 根(默认 $PHASE_ROOT/raw)
#   EMB_ROOT          训练 embedding 根(默认 $PHASE_ROOT/embeddings)
#   SAMPLES_JSON      holdout 生成清单(默认 $P1_ROOT/holdout_generated/samples.json)
#   REAL_ROOT         holdout real 根(默认 $PHASE_ROOT/raw/ASNR-MICCAI-BraTS2023)
#   PRED_ROOT         L2 仪器预测根(默认 $P1_ROOT/l2_acceptance/plan_v1/predictions)
#   ENV_CONFIG        环境 json(默认 $P1_ROOT/environment_brats_p1_train.json)
#   MODEL_CONFIG      模型配置(默认 /root/nv-phase-57/configs/config_brats_p1_train.json)
#   MODEL_DEF         网络定义(默认 /root/nv-phase-57/configs/config_network_rflow.json)
#   SCALE_FACTOR_PATH 基座 DM checkpoint(复用其 scale_factor;
#                     默认 /root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/diff_unet_3d_rflow-mr-brain_v1.pt)
#   DEVICE            VAE 臂设备(cpu 或 cuda:0;DCU 上用 cuda:0)
#   LIMIT             每池均匀抽样例数(空=全量)
#   GEN_LIMIT         覆盖 gen 池抽样例数(空=跟随 LIMIT;gen 池纯 CPU,建议全量)
#   BOOTSTRAP_B       bootstrap 重采样数(默认 10000)
#   OUTPUT_DIR        报告工件输出目录(默认 $P1_ROOT/l2_acceptance/diagnostics/intensity_domain;
#                     sugon 工件区,不入 git)
#
# 前置条件:
#   1. 训练 list 与 fp32 训练 embedding(_emb.nii.gz)仍在(emb 池)
#   2. holdout 生成样本 + L2 predictions 仍在(gen 池;零推理)
#   3. 环境已激活(torch+monai 跑 emb 池两臂;gen 池纯 numpy)
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_et_discrimination_b.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

PHASE_ROOT="${PHASE_ROOT:-/root/private_data/brats2023_rflow_phase}"
P1_ROOT="${P1_ROOT:-/root/private_data/brats2023_rflow_p1}"
TRAIN_LIST="${TRAIN_LIST:-$PHASE_ROOT/lists/p1_image_only.json}"
DATA_ROOT="${DATA_ROOT:-$PHASE_ROOT/raw}"
EMB_ROOT="${EMB_ROOT:-$PHASE_ROOT/embeddings}"
SAMPLES_JSON="${SAMPLES_JSON:-$P1_ROOT/holdout_generated/samples.json}"
REAL_ROOT="${REAL_ROOT:-$PHASE_ROOT/raw/ASNR-MICCAI-BraTS2023}"
PRED_ROOT="${PRED_ROOT:-$P1_ROOT/l2_acceptance/plan_v1/predictions}"
ENV_CONFIG="${ENV_CONFIG:-$P1_ROOT/environment_brats_p1_train.json}"
MODEL_CONFIG="${MODEL_CONFIG:-/root/nv-phase-57/configs/config_brats_p1_train.json}"
MODEL_DEF="${MODEL_DEF:-/root/nv-phase-57/configs/config_network_rflow.json}"
SCALE_FACTOR_PATH="${SCALE_FACTOR_PATH:-/root/private_data/nv-dcu-smoke/NV-Generate-CTMR/models/diff_unet_3d_rflow-mr-brain_v1.pt}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-$P1_ROOT/l2_acceptance/diagnostics/intensity_domain}"

ARGS=()
if [ -z "${SKIP_EMB_POOL:-}" ]; then
    for f in "$TRAIN_LIST" "$ENV_CONFIG" "$MODEL_CONFIG" "$MODEL_DEF" "$SCALE_FACTOR_PATH"; do
        [ -f "$f" ] || { echo "[FATAL] emb 池输入缺失: $f (或以 SKIP_EMB_POOL=1 只跑 gen 池)" >&2; exit 1; }
    done
    ARGS+=(--train-list "$TRAIN_LIST" --data-root "$DATA_ROOT" --emb-root "$EMB_ROOT"
        -e "$ENV_CONFIG" -c "$MODEL_CONFIG" -t "$MODEL_DEF" --scale-factor-path "$SCALE_FACTOR_PATH"
        --device "$DEVICE")
fi
[ -f "$SAMPLES_JSON" ] || { echo "[FATAL] 生成清单不存在: $SAMPLES_JSON" >&2; exit 1; }
[ -d "$PRED_ROOT" ] || { echo "[FATAL] 预测目录不存在: $PRED_ROOT" >&2; exit 1; }
ARGS+=(--samples "$SAMPLES_JSON" --real-root "$REAL_ROOT" --pred-root "$PRED_ROOT")

# ── run id(从终验 json 的 binding 读取,读不到则以未绑定落盘,不阻塞)──
RUN_ID_ARGS=()
ACCEPTANCE_JSON="$P1_ROOT/l2_acceptance/evaluate_v1/l2_final_acceptance_p1.json"
if [ -f "$ACCEPTANCE_JSON" ]; then
    RUN_ID="$(python -c "import json; print(json.load(open('$ACCEPTANCE_JSON')).get('binding', {}).get('run_id') or '')" 2>/dev/null || true)"
    if [ -n "$RUN_ID" ]; then
        RUN_ID_ARGS=(--run-id "$RUN_ID")
    fi
fi

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
# GEN_LIMIT 非空=按例数抽样;置空串=显式全量(传 0,模块把 <=0 视为不设限)。
if [ -n "${GEN_LIMIT+x}" ]; then
    if [ -n "$GEN_LIMIT" ]; then
        LIMIT_ARGS+=(--gen-limit "$GEN_LIMIT")
    else
        LIMIT_ARGS+=(--gen-limit 0)
    fi
fi

echo "============================================"
echo "诊断作业 C:t1c 强度域甄别(#208)"
echo "训练 list: $TRAIN_LIST"
echo "生成清单: $SAMPLES_JSON"
echo "预测根: $PRED_ROOT"
echo "VAE 设备: $DEVICE(LIMIT=${LIMIT:-全量})"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

python -m ctmr.application.acceptance.distribution.intensity_domain \
    "${ARGS[@]}" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    --output-dir "$OUTPUT_DIR" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"}

echo ""
echo "============================================"
echo "  完成。强度域甄别读数报告:"
echo "    $OUTPUT_DIR/intensity_domain_diagnostic.json"
echo "    $OUTPUT_DIR/intensity_domain_diagnostic.md"
echo "  读数供收编票转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
