#!/bin/bash
# 诊断作业 D(issue #209,父 #205):同 seed 换 token 采样亮核甄别——sugon GPU 作业
#
# 用途:对固定 16 例 dev cohort(DevCohortBuilder,GLI 4/MEN 4/METS 3/PED 3/SSA 2)以
#   冻结 P1 候选 checkpoint 在冻结采样配方(cfg=10、30 步、RFlowScheduler)下做五臂采样
#   ——t1n(29)/t1c(34)/t2w(30)/t2f(31) + 对照「泛 MR」(8,增广把 34 扰到的目标);
#   每例一 seed(sha256(case|t1c),冻结采样规则)五臂共用,噪声逐位一致,输出差异全部
#   归因 token 条件,定量 ~19% 模态标签增广稀释对亮核的实际损伤(RC-2/L3 甄别读数)。
#   亮核统计与甄别报告为纯 CPU 第二步(非零基底 P99/P99.9/前 0.5% 均值/max + 34 对 8
#   增益份额 2/3–1/3 分带)。variant=diagnostic:不产生任何验收判定,checkpoint 只读,
#   训练产物零改动。
#
# 用法:
#   bash deploy/jobs/run_token_dilution_d.sh
#
# 环境变量(均可覆写;EMB_ROOT 无合理默认,必填):
#   P1_ROOT      P1 候选产物基目录(默认 /root/private_data/ctmr/runs/p1)
#   CKPT         冻结候选 checkpoint(默认 $P1_ROOT/ckpt/epoch_20.pt,当选候选)
#   DEV_LIST     dev list(默认 $P1_ROOT/lists/p1_image_only_dev.json)
#   EMB_ROOT     embedding companion 根(含 <case>_t1n_emb.nii.gz.json 的 spacing;
#                必填——embedding 树位置是部署事实,不猜)
#   NV_CONFIGS   训练同源 configs 三件套目录(默认 /root/nv-phase-57/configs)
#   ENV_JSON/MODEL_JSON/NET_JSON  三件套逐文件覆写;env json 的
#                trained_autoencoder_path 须可从执行 cwd 解析(VAE 冻结只读)
#   SAMPLES_DIR  五臂采样产物目录(默认 $P1_ROOT/token_dilution/samples;受控存储不入
#                git;文件存在即跳过,可重入续采)
#   OUTPUT_DIR   甄别报告输出目录(默认 $P1_ROOT/token_dilution)
#   BOOTSTRAP_B  bootstrap 重采样数(默认 10000)
#
# 前置条件:
#   1. DCU 卡一张(16 例 × 5 臂 = 80 次采样,cfg=10 × 30 步,fp16)
#   2. 冻结候选 checkpoint、VAE(autoencoder_v1.pt)、dev list、embedding companion 就位
#   3. 环境已激活(torch-dcu、monai、nibabel)
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_zcrop_compensation_a.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
CKPT="${CKPT:-$P1_ROOT/ckpt/epoch_20.pt}"
DEV_LIST="${DEV_LIST:-$P1_ROOT/lists/p1_image_only_dev.json}"
[ -n "${EMB_ROOT:-}" ] || { echo "[FATAL] EMB_ROOT 必填——embedding companion 根(含 <case>_t1n_emb.nii.gz.json)" >&2; exit 1; }
NV_CONFIGS="${NV_CONFIGS:-/root/nv-phase-57/configs}"
ENV_JSON="${ENV_JSON:-$NV_CONFIGS/environment_maisi_diff_model_rflow-mr-brain.json}"
MODEL_JSON="${MODEL_JSON:-$NV_CONFIGS/config_maisi_diff_model_rflow-mr-brain.json}"
NET_JSON="${NET_JSON:-$NV_CONFIGS/config_network_rflow.json}"
SAMPLES_DIR="${SAMPLES_DIR:-$P1_ROOT/token_dilution/samples}"
OUTPUT_DIR="${OUTPUT_DIR:-$P1_ROOT/token_dilution}"

for f in "$CKPT" "$DEV_LIST" "$ENV_JSON" "$MODEL_JSON" "$NET_JSON"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done

# ── run id(从终验 json 的 binding 读取,读不到则以未绑定落盘,不阻塞)──
RUN_ID_ARGS=()
ACCEPTANCE_JSON="$P1_ROOT/l2_acceptance/evaluate_v1/l2_final_acceptance_p1.json"
if [ -f "$ACCEPTANCE_JSON" ]; then
    RUN_ID="$(python -c "import json; print(json.load(open('$ACCEPTANCE_JSON')).get('binding', {}).get('run_id') or '')" 2>/dev/null || true)"
    if [ -n "$RUN_ID" ]; then
        RUN_ID_ARGS=(--run-id "$RUN_ID")
    fi
fi

echo "============================================"
echo "诊断作业 D:同 seed 换 token 采样亮核甄别(#209)"
echo "checkpoint: $CKPT(冻结,只读)"
echo "dev list: $DEV_LIST"
echo "emb root: $EMB_ROOT"
echo "采样产物: $SAMPLES_DIR"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

# ── 第一步:五臂采样(GPU,文件存在即跳过,可重入)──
python -m ctmr.application.generation.modality_label.token_swap_sampling \
    --dev-list "$DEV_LIST" \
    --emb-root "$EMB_ROOT" \
    --ckpt "$CKPT" \
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON" \
    --samples-dir "$SAMPLES_DIR"

# ── 第二步:亮核统计与甄别报告(纯 CPU,读采样产物)──
python -m ctmr.application.acceptance.distribution.token_dilution \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    --checkpoint "$CKPT" \
    ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"}

echo ""
echo "============================================"
echo "  完成。同 seed 换 token 甄别读数报告:"
echo "    $OUTPUT_DIR/token_dilution_diagnostic.json"
echo "    $OUTPUT_DIR/token_dilution_diagnostic.md"
echo "  采样产物: $SAMPLES_DIR(五臂 × cohort,受控存储)"
echo "  读数供收编票转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
