#!/bin/bash
# dev 监控作业(序列②T6,issue #253,父 #247):dev 选择点 ET/WT 观察线链路——sugon 作业
#
# 用途:候选选择点去盲监控(收编裁决 #5 采纳项)。对 dev list(1060 例的分层样本,
#   GLI 50/MEN 40/METS 24/PED 10/SSA 6 = 130 例 × 4 模态;**绝不可碰 holdout 530**
#   ——选择泄漏即 L2 终验作废)以候选 checkpoint 采样伪四模态体 → 冻结仪器只读路径
#   (final_acceptance predict / measurement_run,plan schema 与终验同构,执行侧零改动)
#   → 作业 B 口径 ET 甄别 + WT 添注读数 → 观察线黄旗判定(METS ET 检出率 <0.9 或
#   任一挑战 vol_et_rel 中位 >2)。选择面、非验收判定;冻结仪器/包络/判定线零改动。
#
# 全链五步(幂等处可断点重入):
#   1. 采样 + 装配 plan(GPU;文件存在即跳过;--plan-only 可只重建 plan)
#   2. 冻结仪器推理脚本写出(predict_all.sh)
#   3. 仪器输入组装(gen 侧重采样 + RAS→LPS 翻转;real 侧原生直通)
#   4. 冻结仪器逐挑战推理(五挑战;nnUNet 环境变量内置)
#   5. 测量 CSV + 观察线报告(CPU;dev_monitor_diagnostic.{json,md})
#
# 用法:
#   bash deploy/jobs/run_dev_monitor_etwt.sh
#
# 环境变量(均可覆写;EMB_ROOT/RAW_ROOT 无合理默认则必填):
#   P1_ROOT       P1 候选产物基目录(默认 /root/private_data/ctmr/runs/p1)
#   CKPT          候选 checkpoint(默认 $P1_ROOT/ckpt/epoch_20.pt,T6 现候选;
#                 T8 重训候选覆写之)
#   RUN_ID        候选 run id(默认 p1-20260822T131947Z,T8 覆写)
#   DEV_LIST      dev list(默认 /root/private_data/ctmr/data/phase/lists/p1_image_only_dev.json,
#                 1060 条;唯一 population 输入)
#   RAW_ROOT      real BraTS 原生数据根(real 侧直通;默认 /root/private_data/ctmr/data/phase/raw)
#   EMB_ROOT      embedding companion 根(<case>_t1n_emb.nii.gz.json 的 spacing;
#                 必填——embedding 树位置是部署事实,不猜)
#   NV_ENV/MODEL_JSON/NET_JSON  训练同源 configs 三件套(默认仓库 configs/ 下同名;
#                 env json 的 trained_autoencoder_path 须可从执行 cwd 解析,或以
#                 VAE_DIR 提供其所在目录,配方据此落绝对路径覆写件)
#   VAE_DIR       VAE 幸存副本目录(默认 /root/private_data/ctmr/instruments/v1_models)
#   NNUNET_RAW/PREPROCESSED/RESULTS  冻结仪器三变量(默认 20260830 聚合布局)
#   MONITOR_ROOT  监控工作根(默认 $P1_ROOT/dev_monitor;工件区不入 git)
#   BOOTSTRAP_B   bootstrap 重采样数(默认 10000)
#   HIP_VISIBLE_DEVICES  选卡(可选)
#
# 前置条件:
#   1. DCU 卡一张(130 例 × 4 模态 = 520 次采样,cfg=10 × 30 步,fp16;约数小时)
#   2. 冻结候选 checkpoint、VAE、dev list、raw real 数据、embedding companion 就位
#   3. 双 source 环境已激活(/opt/dtk/env.sh + ai_proxy);nnunetv2 可用
set -euo pipefail

# 诊断模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_token_dilution_d.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
CKPT="${CKPT:-$P1_ROOT/ckpt/epoch_20.pt}"
RUN_ID="${RUN_ID:-p1-20260822T131947Z}"
DEV_LIST="${DEV_LIST:-/root/private_data/ctmr/data/phase/lists/p1_image_only_dev.json}"
RAW_ROOT="${RAW_ROOT:-/root/private_data/ctmr/data/phase/raw}"
EMB_ROOT="${EMB_ROOT:-/root/private_data/ctmr/data/phase/embeddings}"
VAE_DIR="${VAE_DIR:-/root/private_data/ctmr/models}"
ENV_JSON="${ENV_JSON:-$PROJECT_ROOT/configs/environment_maisi_diff_model_rflow-mr-brain.json}"
MODEL_JSON="${MODEL_JSON:-$PROJECT_ROOT/configs/config_maisi_diff_model_rflow-mr-brain.json}"
NET_JSON="${NET_JSON:-$PROJECT_ROOT/configs/config_network_rflow.json}"
NNUNET_RAW="${NNUNET_RAW:-/root/private_data/ctmr/data/nnunet_raw}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-/root/private_data/ctmr/data/nnunet_preprocessed}"
NNUNET_RESULTS="${NNUNET_RESULTS:-/root/private_data/ctmr/instruments/nnunet_results}"
MONITOR_ROOT="${MONITOR_ROOT:-$P1_ROOT/dev_monitor}"
SAMPLES_DIR="$MONITOR_ROOT/samples"

for f in "$CKPT" "$DEV_LIST" "$ENV_JSON" "$MODEL_JSON" "$NET_JSON"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done
[ -d "$RAW_ROOT" ] || { echo "[FATAL] real 数据根不存在: $RAW_ROOT" >&2; exit 1; }
[ -d "$EMB_ROOT" ] || { echo "[FATAL] embedding companion 根不存在: $EMB_ROOT" >&2; exit 1; }

# ── VAE 路径可解析性(env json 的 trained_autoencoder_path 须可从 cwd 解析)──
# 相对路径不可解析而 $VAE_DIR 下有 VAE 时,落一份绝对路径覆写件(显式日志,不改原件)。
resolve_env_json() {
    local vae_ref
    vae_ref="$(python -c "import json; print(json.load(open('$ENV_JSON')).get('trained_autoencoder_path', ''))")"
    if [ -n "$vae_ref" ] && [ ! -f "$vae_ref" ] && [ -f "$VAE_DIR/$(basename "$vae_ref")" ]; then
        local override="$MONITOR_ROOT/env_vae_override.json"
        mkdir -p "$MONITOR_ROOT"
        python - "$ENV_JSON" "$VAE_DIR/$(basename "$vae_ref")" "$override" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
doc["trained_autoencoder_path"] = sys.argv[2]
open(sys.argv[3], "w").write(json.dumps(doc, indent=2) + "\n")
PY
        echo "[dev-monitor] trained_autoencoder_path 不可解析,已按 VAE_DIR 落绝对路径覆写件: $override(VAE 冻结只读)"
        echo "$override"
    else
        echo "$ENV_JSON"
    fi
}
ENV_JSON_EFFECTIVE="${ENV_JSON_EFFECTIVE:-$(resolve_env_json)}"

echo "============================================"
echo "dev 监控作业:dev 选择点 ET/WT 观察线(#253,父 #247)"
echo "checkpoint: $CKPT(只读)"
echo "dev list:   $DEV_LIST(唯一 population;零 holdout 接触)"
echo "real root:  $RAW_ROOT"
echo "emb root:   $EMB_ROOT"
echo "工作根:     $MONITOR_ROOT(工件区,不入 git)"
echo "run:        $RUN_ID"
echo "variant=diagnostic — 选择面监控,不产生任何验收判定"
echo "============================================"

# ── 第一步:分层样本采样 + 装配 plan(GPU,文件存在即跳过,可重入)──
python -m ctmr.application.generation.modality_label.dev_monitor_sampling \
    --dev-list "$DEV_LIST" \
    --raw-root "$RAW_ROOT" \
    --emb-root "$EMB_ROOT" \
    --ckpt "$CKPT" \
    -e "$ENV_JSON_EFFECTIVE" -c "$MODEL_JSON" -t "$NET_JSON" \
    --samples-dir "$SAMPLES_DIR" \
    --output-dir "$MONITOR_ROOT" \
    --run-id "$RUN_ID"

# ── 第二步:冻结仪器推理脚本写出(plan schema 与终验同构,执行侧零改动)──
python -m ctmr.application.acceptance.distribution.final_acceptance predict \
    --plan "$MONITOR_ROOT/plan.json" \
    --output-dir "$MONITOR_ROOT"

# ── 第三步:仪器输入组装(gen 重采样 + RAS→LPS 翻转;real 原生直通)──
python -m ctmr.application.acceptance.distribution.measurement_run assemble-execute \
    --plan "$MONITOR_ROOT/plan.json" \
    --output-root "$MONITOR_ROOT"

# ── 第四步:冻结仪器逐挑战推理(五挑战;冻结配置,TTA 按冻结口径开启)──
export nnUNet_raw="$NNUNET_RAW" nnUNet_preprocessed="$NNUNET_PREPROCESSED" nnUNet_results="$NNUNET_RESULTS" nnUNet_compile=f
(cd "$MONITOR_ROOT" && bash predict_all.sh)

# ── 第五步:测量 CSV + 观察线报告(纯 CPU)──
python -m ctmr.application.acceptance.distribution.measurement_run measure \
    --plan "$MONITOR_ROOT/plan.json" \
    --input-root "$MONITOR_ROOT/inputs" \
    --pred-root "$MONITOR_ROOT/predictions" \
    --output "$MONITOR_ROOT/measurements_dev.csv"

python -m ctmr.application.acceptance.distribution.dev_monitor \
    --measurements "$MONITOR_ROOT/measurements_dev.csv" \
    --sample-plan "$MONITOR_ROOT/plan.json" \
    --output-dir "$MONITOR_ROOT/report" \
    --bootstrap-b "${BOOTSTRAP_B:-10000}" \
    --run-id "$RUN_ID"

echo ""
echo "============================================"
echo "  完成。dev 监控读数与观察线旗标报告:"
echo "    $MONITOR_ROOT/report/dev_monitor_diagnostic.json"
echo "    $MONITOR_ROOT/report/dev_monitor_diagnostic.md"
echo "  采样协议:cohort.json(配额 + sha256 选择规则)+ plan.json(population=dev)"
echo "  读数供实验记录转写 deploy/experiments/(工件本身不入 git)"
echo "  现候选基线 = T8 重训候选 go/no-go 判读的对照锚"
echo "============================================"
