#!/bin/bash
# 序列② T4(issue #251,父 #247):clip=True 编码配方下的训练 embedding 重编码——sugon GPU 作业
#
# 用途:T4 配方改动(mri 归一化旗标 clip=False→True,作业 C 实测:外推输入出冻结 VAE 重建域,
#   自评 MAE 0.8673 vs 域内 0.0062 且伴生瘤内负值伪影)使旧 embedding(clip=False 编码)在
#   clip=True 世界不可复用——本作业把全量 P1 image-only 训练 list 重编码到新 embedding 根:
#   ① 调 vendored 编码链(diff_model_create_training_data,新旗标生效;文件存在即跳过,可重入续跑);
#   ② 新树逐文件 md5/bytes/shape 登记 manifest.jsonl(随 embedding 落盘,供 T7 重训消费与审计);
#   ③ 域内自评 MAE 抽检(均匀抽样,decode 新落盘工件,作业 C clip=True 口径:域内 [0,1] 层
#     对照 0.0062 锚、外推 >1.0 层对照 0.0559 锚——量级参照非判定线)。
#   variant=diagnostic:冻结 VAE 只读;评估链与判定线零改动;旧 embedding 树不动(回退锚)。
#
# 用法:
#   bash deploy/jobs/run_embedding_reencode_t4.sh
#   SELF_CHECK_LIMIT=0 bash deploy/jobs/run_embedding_reencode_t4.sh   # 只重编码+清单,跳过抽检
#
# 环境变量(均可覆写;默认值按 T4 执行实例实测布局登记——2026-09-01 新实例,作业 D 的
#   nv-phase-60 部署树已随系统盘易失;实例换新后执行前必须复核 raw 树与 list 的对应关系):
#   PHASE_ROOT       phase 数据根(默认 /root/private_data/ctmr/data/phase)
#   P1_ROOT          P1 运行根(默认 /root/private_data/ctmr/runs/p1)
#   TRAIN_LIST       训练 list(默认 $PHASE_ROOT/lists/p1_image_only.json;全量 7404 条)
#   DATA_ROOT        训练 raw 根(list 的 image 相对路径相对它解析;默认 $PHASE_ROOT/raw_relinked
#                    ——链接镜像树:phase/raw 的链接目标 brats2023_nnunet 已随重组迁为
#                    ct mr/data/nnunet_raw,故 T4 用重建的链接树指向 nnunet_raw 真实文件)
#   EMB_OUT_ROOT     新 embedding 根(默认 $PHASE_ROOT/embeddings_cliptrue;兄弟目录,不覆盖旧树)
#   NV_CONFIGS       部署树 configs 三件套目录(默认 /root/nv-phase-t4/configs,T4 部署树)
#   VAE_PATH         冻结 VAE(默认 /root/private_data/ctmr/models/autoencoder_v1.pt;
#                    本地上传副本,md5 917cfb1e49631c8a713e3bb7c758fbca 与冻结 canonical 全同,只读)
#   DEVICE           抽检 decode 臂设备(默认 cuda:0)
#   NUM_GPUS         编码链 GPU 数(默认 1)
#   SHARDS           并行分片数(默认 1;>1 时按 idx%SHARDS 均分训练 list,每片一个单卡
#                    worker(--encode-only,cuda:i),全部完成后单卡收尾(--skip-encode:
#                    manifest 对账+抽检+报告)。编码链跳过已存在文件,任意中断后重跑
#                    零浪费;分片 list/env 落 OUTPUT_DIR/,worker 日志 shard_i.log)
#   SELF_CHECK_LIMIT 抽检例数(默认 200,作业 C 同规模;0 关闭)
#   BOOTSTRAP_B      bootstrap 重采样数(默认 10000)
#   OUTPUT_DIR       报告输出目录(默认 $P1_ROOT/reencode_t4)
#
# 前置条件:
#   1. 部署树 src 含 T4 配方(instance_definition mri 臂 clip=True;旧代码跑不出 clip=True 世界)
#   2. 训练 list、raw t1c、冻结 VAE 就位;环境已激活(torch-dcu、monai、nibabel)
#   3. 磁盘余量 ≥ 旧 embedding 树(全量重编码等量写出)
set -euo pipefail

# 诊断/作业模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_token_dilution_d.sh 先例:repo 与平铺部署两种形态的拼写合并;
# deploy/jobs/ 下 src 经 ../../ 解析——一层 ../ 会落到 deploy/src)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

PHASE_ROOT="${PHASE_ROOT:-/root/private_data/ctmr/data/phase}"
P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
TRAIN_LIST="${TRAIN_LIST:-$PHASE_ROOT/lists/p1_image_only.json}"
DATA_ROOT="${DATA_ROOT:-$PHASE_ROOT/raw_relinked}"
EMB_OUT_ROOT="${EMB_OUT_ROOT:-$PHASE_ROOT/embeddings_cliptrue}"
NV_CONFIGS="${NV_CONFIGS:-/root/nv-phase-t4/configs}"
ENV_JSON="${ENV_JSON:-$NV_CONFIGS/environment_maisi_diff_model_rflow-mr-brain.json}"
MODEL_JSON="${MODEL_JSON:-$NV_CONFIGS/config_maisi_diff_model_rflow-mr-brain.json}"
NET_JSON="${NET_JSON:-$NV_CONFIGS/config_network_rflow.json}"
VAE_PATH="${VAE_PATH:-/root/private_data/ctmr/models/autoencoder_v1.pt}"
DEVICE="${DEVICE:-cuda:0}"
NUM_GPUS="${NUM_GPUS:-1}"
SHARDS="${SHARDS:-1}"
SELF_CHECK_LIMIT="${SELF_CHECK_LIMIT:-200}"
BOOTSTRAP_B="${BOOTSTRAP_B:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-$P1_ROOT/reencode_t4}"

for f in "$TRAIN_LIST" "$ENV_JSON" "$MODEL_JSON" "$NET_JSON" "$VAE_PATH"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f(raw 根 $DATA_ROOT 需为目录)" >&2; exit 1; }
done
[ -d "$DATA_ROOT" ] || { echo "[FATAL] raw 根不是目录: $DATA_ROOT(重组后位置待核对,见脚本头注)" >&2; exit 1; }

# ── run id(从终验 json 的 binding 读取,读不到则以未绑定落盘,不阻塞)──
RUN_ID_ARGS=()
ACCEPTANCE_JSON="$P1_ROOT/l2_acceptance/evaluate_v1/l2_final_acceptance_p1.json"
if [ -f "$ACCEPTANCE_JSON" ]; then
    RUN_ID="$(python -c "import json; print(json.load(open('$ACCEPTANCE_JSON')).get('binding', {}).get('run_id') or '')" 2>/dev/null || true)"
    if [ -n "$RUN_ID" ]; then
        RUN_ID_ARGS=(--run-id "$RUN_ID")
    fi
fi

mkdir -p "$OUTPUT_DIR" "$EMB_OUT_ROOT"

# ── 覆写 env json:embedding_base_dir 指新根,数据面指向本次作业的绝对路径 ──
REENCODE_ENV="$OUTPUT_DIR/env_reencode.json"
python - "$ENV_JSON" "$REENCODE_ENV" "$TRAIN_LIST" "$DATA_ROOT" "$EMB_OUT_ROOT" "$VAE_PATH" <<'PY'
import json, sys
src, dst, train_list, data_root, emb_root, vae = sys.argv[1:7]
cfg = json.load(open(src))
cfg["json_data_list"] = train_list
cfg["data_base_dir"] = data_root
cfg["embedding_base_dir"] = emb_root
cfg["trained_autoencoder_path"] = vae
json.dump(cfg, open(dst, "w"), indent=2)
print(f"[reencode] env 覆写落盘: {dst}")
PY

echo "============================================"
echo "序列② T4:训练 embedding 重编码(clip=True)(#251)"
echo "训练 list: $TRAIN_LIST"
echo "raw 根: $DATA_ROOT"
echo "新 embedding 根: $EMB_OUT_ROOT(旧树不动,回退锚)"
echo "VAE: $VAE_PATH(冻结,只读)"
echo "分片: SHARDS=$SHARDS;编码链 GPU 每片 1;收尾(manifest+抽检)设备: $DEVICE(LIMIT=$SELF_CHECK_LIMIT)"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

# ── 单次作业调用(分片 worker 传 --encode-only,收尾传 --skip-encode)──
run_reencode() {
    local env_json="$1"; local train_list="$2"; local device="$3"; shift 3
    python -m ctmr.application.generation.modality_label.reencode \
        --train-list "$train_list" \
        -e "$env_json" \
        -c "$MODEL_JSON" \
        -t "$NET_JSON" \
        --device "$device" \
        --num-gpus 1 \
        --output-dir "$OUTPUT_DIR" \
        ${RUN_ID_ARGS[@]+"${RUN_ID_ARGS[@]}"} \
        "$@"
}

if [ "${SHARDS:-1}" -le 1 ]; then
    run_reencode "$REENCODE_ENV" "$TRAIN_LIST" "$DEVICE" \
        --self-check-limit "$SELF_CHECK_LIMIT" --bootstrap-b "$BOOTSTRAP_B"
else
    # ── 阶段 1:按 idx%SHARDS 均分训练 list,每片一个单卡 worker(--encode-only)──
    python - "$TRAIN_LIST" "$REENCODE_ENV" "$OUTPUT_DIR" "$SHARDS" <<'PY'
import json, sys
train_list, base_env_path, out_dir, n_shards_s = sys.argv[1:5]
n_shards = int(n_shards_s)
listing = json.load(open(train_list))["training"]
shards = [[] for _ in range(n_shards)]
for idx, entry in enumerate(listing):
    shards[idx % n_shards].append(entry)
base_env = json.load(open(base_env_path))
for i, shard in enumerate(shards):
    shard_list = f"{out_dir}/train_list_shard_{i}.json"
    json.dump({"training": shard}, open(shard_list, "w"))
    env = dict(base_env)
    env["json_data_list"] = shard_list
    json.dump(env, open(f"{out_dir}/env_shard_{i}.json", "w"), indent=2)
print(f"[reencode] {n_shards} 片分片落盘: {[len(s) for s in shards]}")
PY
    pids=()
    for i in $(seq 0 $((SHARDS - 1))); do
        run_reencode "$OUTPUT_DIR/env_shard_$i.json" "$OUTPUT_DIR/train_list_shard_$i.json" "cuda:$i" --encode-only \
            > "$OUTPUT_DIR/shard_$i.log" 2>&1 &
        pids+=($!)
    done
    for i in $(seq 0 $((SHARDS - 1))); do
        wait "${pids[$i]}" || { echo "[FATAL] shard $i 退出码非零,见 $OUTPUT_DIR/shard_$i.log" >&2; exit 1; }
    done
    # ── 阶段 2:收尾进程(全量 list)——manifest 对账 + 抽检 + 报告,单卡──
    run_reencode "$REENCODE_ENV" "$TRAIN_LIST" "$DEVICE" \
        --skip-encode --self-check-limit "$SELF_CHECK_LIMIT" --bootstrap-b "$BOOTSTRAP_B"
fi

echo ""
echo "============================================"
echo "  完成。重编码产物与读数报告:"
echo "    manifest: $EMB_OUT_ROOT/manifest.jsonl(md5 清单,供 T7 重训审计)"
echo "    报告:     $OUTPUT_DIR/embedding_reencode_report.{json,md}"
echo "  判读锚:抽检直通臂与作业 C 直通链口径同档(0.006 量级)、工件臂与旧工件同链自评"
echo "  同档(0.08 量级)即配方落地;读数供转写 deploy/experiments/(工件本身不入 git)"
echo "============================================"
