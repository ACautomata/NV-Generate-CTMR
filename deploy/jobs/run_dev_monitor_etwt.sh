#!/bin/bash
# dev 监控作业(序列②T6,issue #253,父 #247):dev 选择点 ET/WT 观察线链路——sugon 作业
#
# 用途:候选选择点去盲监控(收编裁决 #5 采纳项)。对 dev list(1060 例的分层样本,
#   GLI 50/MEN 40/METS 24/PED 10/SSA 6 = 130 例 × 4 模态;**绝不可碰 holdout 530**
#   ——选择泄漏即 L2 终验作废)以候选 checkpoint 采样伪四模态体 → 冻结仪器只读路径
#   (final_acceptance predict / measurement_run,plan schema 与终验同构;推理脚本内
#   -c/-p/-tr 按 nnUNet_results 树实况覆写,其余冻结执行口径——TTA on、fold 0——零改动)
#   → 作业 B 口径 ET 甄别 + WT 添注读数 → 观察线黄旗判定(METS ET 检出率 <0.9 或
#   任一挑战 vol_et_rel 中位 >2)。选择面、非验收判定;冻结仪器/包络/判定线零改动。
#
# 仪器版本注记(2026-09-03):l2 仪器主本随 2026-08-30 聚合重置丢失,已按 v2 协议
#   全量重训并换树(nnunet_results -> l2-instrument-v2/results;标准 plans 派生
#   nnUNetPlans_v2bs8、4×DCU DDP、BF16,见 deploy/experiments/20260901-仪器主本丢失
#   与重训决策.md)。本链路读数须标注**仪器 v2**,与 T5 历史读数不可直接比仪器版本;
#   仓库 INSTRUMENT_SPECS 冻结锚与新 ADR 同批重钉,不被本配方触碰(实况覆写只改
#   生成物脚本,零包内改动)。
#
# 全链五步 + 一步覆写(幂等处可断点重入):
#   1. 采样 + 装配 plan(GPU;文件存在即跳过;--plan-only 可只重建 plan)
#   2. 冻结仪器推理脚本写出(predict_all.sh)
#   2b. 仪器 spec 实况覆写(按 nnUNet_results 树的 <trainer>__<plans>__<config>
#       目录名改写 predict_*.sh 的 -c/-p/-tr;与实况一致则零改动)
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
#   ENV_JSON/MODEL_JSON/NET_JSON  训练同源 configs 三件套(默认仓库 configs/ 下同名;
#                 env json 的 trained_autoencoder_path 须可从执行 cwd 解析,或以
#                 VAE_DIR 提供其所在目录,配方据此落绝对路径覆写件)
#   VAE_DIR       VAE 幸存副本目录(默认 /root/private_data/ctmr/instruments/v1_models)
#   NNUNET_RAW/PREPROCESSED/RESULTS  冻结仪器三变量(默认 20260830 聚合布局)
#   NNUNET_EXT_TRAINER  包外 trainer 目录(nnUNet_extTrainer;v2 仪器的 BF16
#                 子类 nnUNetTrainer_250epochs_bf16 在此,包内查无——推理与
#                 训练同源消费;默认 l2-instrument-v2/trainer)
#   MONITOR_ROOT  监控工作根(默认 $P1_ROOT/dev_monitor;工件区不入 git)。
#                 采样臂以 sampling_provenance.json 钉住目录的候选 checkpoint
#                 指纹——换新候选(CKPT/RUN_ID)必须改用全新 MONITOR_ROOT
#                 (如 $P1_ROOT/dev_monitor/$RUN_ID),否则指纹不符即响亮失败,
#                 不会静默复用旧候选的采样体去贴新 run_id
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
NNUNET_EXT_TRAINER="${NNUNET_EXT_TRAINER:-/root/private_data/ctmr/instruments/l2-instrument-v2/trainer}"
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
        # 说明走 stderr——本函数的 stdout 被命令替换捕获为 ENV_JSON_EFFECTIVE,
        # 只能含路径一行(双 echo 会把多行串塞进 -e 参数)。
        echo "[dev-monitor] trained_autoencoder_path 不可解析,已按 VAE_DIR 落绝对路径覆写件: $override(VAE 冻结只读)" >&2
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

# ── 第二步b:仪器 spec 实况覆写(零包内改动)──
# PredictScriptWriter 按仓库 INSTRUMENT_SPECS 冻结锚写脚本;仪器 v2 换树后该锚
# 待与校准重跑+新 ADR 同批重钉,而监控是 variant=diagnostic 选择面,须消费现役
# results 树——此处按树实况(<trainer>__<plans>__<config> 恰一个目录,不猜)
# 改写生成物脚本的 -c/-p/-tr,并打印前后对照。与实况一致则零改动(幂等)。
python - "$NNUNET_RESULTS" "$MONITOR_ROOT" <<'PY'
import re
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
monitor_root = Path(sys.argv[2])
for script in sorted(monitor_root.glob("predict_*.sh")):
    if script.name == "predict_all.sh":
        continue
    text = script.read_text()
    challenge = script.stem.removeprefix("predict_")
    dataset_id = re.search(r"-d (\S+)", text).group(1)
    ds_dir = results_root / dataset_id
    if not ds_dir.is_dir():
        raise SystemExit(f"[FATAL] 仪器结果树缺 {ds_dir}——换树/重训未完成,拒绝盲跑")
    trainer_dirs = sorted(p.name for p in ds_dir.iterdir() if p.is_dir())
    if len(trainer_dirs) != 1:
        raise SystemExit(f"[FATAL] {ds_dir} 下应恰一个 trainer 目录,实得 {trainer_dirs}——不猜")
    parts = trainer_dirs[0].split("__")
    if len(parts) != 3:
        raise SystemExit(f"[FATAL] {trainer_dirs[0]!r} 不是 <trainer>__<plans>__<config> 三段式——不猜")
    trainer, plans, config = parts
    spec_old = re.search(r"-c (\S+) -p (\S+) -tr (\S+)", text)
    replacement = f"-c {config} -p {plans} -tr {trainer}"
    if spec_old.group(0) == replacement:
        print(f"[instrument-spec] {challenge}: 脚本 spec 已与 results 树一致({trainer_dirs[0]}),零改动")
        continue
    script.write_text(text.replace(spec_old.group(0), replacement))
    print(f"[instrument-spec] {challenge}: {spec_old.group(3)}__{spec_old.group(2)}__{spec_old.group(1)} -> {trainer_dirs[0]}")
PY

# ── 第三步:仪器输入组装(gen 重采样 + RAS→LPS 翻转;real 原生直通)──
python -m ctmr.application.acceptance.distribution.measurement_run assemble-execute \
    --plan "$MONITOR_ROOT/plan.json" \
    --output-root "$MONITOR_ROOT"

# ── 第四步:冻结仪器逐挑战推理(五挑战;冻结配置,TTA 按冻结口径开启)──
# nnUNet_extTrainer:实况 trainer(v2 BF16 子类)是包外类,训练侧经同变量
# 接入(20260901-仪器主本丢失与重训决策.md §4);推理侧缺它即
# "Could not find requested nnunet trainer" 响亮死。
export nnUNet_raw="$NNUNET_RAW" nnUNet_preprocessed="$NNUNET_PREPROCESSED" nnUNet_results="$NNUNET_RESULTS" nnUNet_compile=f nnUNet_extTrainer="$NNUNET_EXT_TRAINER"
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
