#!/bin/bash
# 序列③ T2(issue #312,父 #310):RAS 编码修复下的全量 embedding 重编码——sugon GPU 作业
#
# 用途:#312 dim 修复(vendored 编码链的 resize 目标改读 RAS 重排后 spatial_shape,替代 NIfTI
#   header 原始轴序 dim;方向审计实测 replay/MR-RATE 臂 ~65% embedding 形状轴序错乱的根因;
#   BraTS 臂方向全为 flip-only 侥幸免疫)使 clip=True 时代 embedding 对换轴方向体积不可复用——
#   本作业把全量训练语料重编码到新 embedding 根(legacy/cliptrue 旧树均不动,回退锚):
#   ① 调 vendored 编码链(diff_model_create_training_data,#312 修复版 + clip=True 工厂;
#     文件存在即跳过,实例易失后可重入续跑,沿 T4 先例);
#   ② 新树逐文件 md5/bytes/shape 登记 manifest.jsonl;T7 sidecar 按多源 fallback 拷贝校验
#     (主+replay 臂源 = embeddings_cliptrue,dev 臂源 = embeddings);
#   ③ 形状守卫:逐条从 raw 推导期望 latent 形状(RAS 重排后 → round_number(128 基)→ ÷4,
#     channels-last),任何违例 = 作业失败(规范形状外即失败);
#   ④ 域内自评 MAE 抽检(主 list t1c 子集,作业 C 口径,量级参照非判定线)。
#   重编码范围:raw 臂 = P1 主 7404 + replay 7404 + dev 1060(P2/P3 臂消费全集);
#   P2/P3 臂 list(p2_mask_cond 8464 / p3_pairs 25392)以 emb 引用形式只对账+守卫,不驱动编码。
#   variant=diagnostic:冻结 VAE 只读;评估链与判定线零改动。
#
# 用法:
#   bash deploy/jobs/run_embedding_reencode_t2.sh
#   SELF_CHECK_LIMIT=0 bash deploy/jobs/run_embedding_reencode_t2.sh   # 只重编码+清单+守卫,跳过抽检
#
# 环境变量(均可覆写;默认按 2026-09-03 实例 crdnotebook-…-43400 持久盘实测布局登记):
#   PHASE_ROOT       phase 数据根(默认 /root/private_data/ctmr/data/phase)
#   P1_ROOT          P1 运行根(默认 /root/private_data/ctmr/runs/p1;replay list 所在)
#   TRAIN_LIST       P1 主训练 list(默认 $PHASE_ROOT/lists/p1_image_only.json;7404 条)
#   REPLAY_LIST      MR-RATE replay list(默认 $P1_ROOT/lists/p1_mrrate_replay.json;7404 条)
#   DEV_LIST         dev cohort list(默认 $PHASE_ROOT/lists/p1_image_only_dev.json;1060 条;
#                    P2/P3 臂消费其 embedding;holdout 判定面零接触——本作业不产生任何判定)
#   EMB_LISTS        P2/P3 臂 emb 引用 list(空格分隔,默认 $PHASE_ROOT/lists/p2_mask_cond.json
#                    $PHASE_ROOT/lists/p3_pairs.json;对账+守卫,不驱动编码)
#   DATA_ROOT        raw 根(list 的 image 相对路径相对它解析;默认 $PHASE_ROOT/raw_relinked
#                    ——链接镜像树,BraTS 与 MR-RATE replay raw 都在此)
#   EMB_OUT_ROOT     新 embedding 根(默认 $PHASE_ROOT/embeddings_ras;兄弟目录,旧树不动)
#   SIDE_CAR_SOURCES sidecar 源旧根(空格分隔,按 fallback 序;默认 "$PHASE_ROOT/embeddings_cliptrue
#                    $PHASE_ROOT/embeddings";设为空串关闭拷贝——此时 sidecar 缺失不记 missing)
#   NV_CONFIGS       部署树 configs 三件套目录(默认 /root/nv-phase-t4/configs;src 须为 #312 修复版)
#   VAE_PATH         冻结 VAE(默认 /root/private_data/ctmr/models/autoencoder_v1.pt,md5
#                    917cfb1e49631c8a713e3bb7c758fbca,只读)
#   DEVICE           抽检 decode 臂设备(默认 cuda:0)
#   NUM_GPUS         编码链 GPU 数(默认 1)
#   SHARDS           并行分片数(默认 4;>1 时按 idx%SHARDS 均分 raw 合并 list,每片一个单卡
#                    worker(--encode-only,CUDA_VISIBLE_DEVICES=i 隔离),全部完成后单卡收尾
#                    (--skip-encode:manifest+sidecar+守卫+对账+抽检+报告)。任意中断重跑零浪费;
#                    分片 list/env 落 OUTPUT_DIR/,worker 日志 shard_i.log)
#   SELF_CHECK_LIMIT 抽检例数(默认 200,作业 C/T4 同规模;0 关闭)
#   BOOTSTRAP_B      bootstrap 重采样数(默认 10000)
#   OUTPUT_DIR       报告输出目录(默认 $P1_ROOT/reencode_t2)
#
# 前置条件:
#   1. 部署树 src 含 #312 修复(create_training_data.new_dim_from_ras_shape;旧代码重现错乱形状)
#   2. 训练/replay/dev list、raw 树、冻结 VAE 就位;双 source 后 torch-dcu/monai/nibabel 可用
#   3. 磁盘余量 ≥ cliptrue 树体量(≈14G+;实测余量 537G)
set -euo pipefail

# 诊断/作业模块在新家包内,src 树与仓库根要在 sys.path(ADR-0009 同族 shim,沿
# run_embedding_reencode_t4.sh 先例)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

# ── 双 source(DTK 算 + 平台代理网;非交互 ssh 需显式,沿 run_p1_retrain_t7.sh)──
set +u
source /opt/dtk/env.sh 2>/dev/null || true
set -u
eval "$(grep -E '^export (HF_TOKEN|HF_ENDPOINT)=' ~/.bashrc 2>/dev/null)" || true

PHASE_ROOT="${PHASE_ROOT:-/root/private_data/ctmr/data/phase}"
P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
TRAIN_LIST="${TRAIN_LIST:-$PHASE_ROOT/lists/p1_image_only.json}"
REPLAY_LIST="${REPLAY_LIST:-$P1_ROOT/lists/p1_mrrate_replay.json}"
DEV_LIST="${DEV_LIST:-$PHASE_ROOT/lists/p1_image_only_dev.json}"
EMB_LISTS="${EMB_LISTS-$PHASE_ROOT/lists/p2_mask_cond.json $PHASE_ROOT/lists/p3_pairs.json}"
DATA_ROOT="${DATA_ROOT:-$PHASE_ROOT/raw_relinked}"
EMB_OUT_ROOT="${EMB_OUT_ROOT:-$PHASE_ROOT/embeddings_ras}"
SIDE_CAR_SOURCES="${SIDE_CAR_SOURCES-$PHASE_ROOT/embeddings_cliptrue $PHASE_ROOT/embeddings}"
NV_CONFIGS="${NV_CONFIGS:-/root/nv-phase-t4/configs}"
ENV_JSON="${ENV_JSON:-$NV_CONFIGS/environment_maisi_diff_model_rflow-mr-brain.json}"
MODEL_JSON="${MODEL_JSON:-$NV_CONFIGS/config_maisi_diff_model_rflow-mr-brain.json}"
NET_JSON="${NET_JSON:-$NV_CONFIGS/config_network_rflow.json}"
VAE_PATH="${VAE_PATH:-/root/private_data/ctmr/models/autoencoder_v1.pt}"
DEVICE="${DEVICE:-cuda:0}"
NUM_GPUS="${NUM_GPUS:-1}"
SHARDS="${SHARDS:-4}"
SELF_CHECK_LIMIT="${SELF_CHECK_LIMIT:-200}"
BOOTSTRAP_B="${BOOTSTRAP_B:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-$P1_ROOT/reencode_t2}"

for f in "$TRAIN_LIST" "$REPLAY_LIST" "$DEV_LIST" "$ENV_JSON" "$MODEL_JSON" "$NET_JSON" "$VAE_PATH"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done
for f in $EMB_LISTS; do
    [ -f "$f" ] || { echo "[FATAL] P2/P3 emb list 不存在: $f" >&2; exit 1; }
done
[ -d "$DATA_ROOT" ] || { echo "[FATAL] raw 根不是目录: $DATA_ROOT" >&2; exit 1; }
grep -q "new_dim_from_ras_shape" "$PROJECT_ROOT/src/ctmr/infrastructure/maisi_engine/create_training_data.py" \
    || { echo "[FATAL] 部署树 src 缺 #312 修复(new_dim_from_ras_shape)——旧代码会重现轴序错乱" >&2; exit 1; }

# ── dev 臂 raw 补链(幂等):raw_relinked 镜像树只覆盖训练 7404 + replay;dev cohort 的
#    raw 在官方源重取树 raw_brats2023_official(T7 reference bank 同源)。把 dev list
#    引用的 raw 以符号链接补进 DATA_ROOT,使编码链与形状守卫以同一 data_base_dir 解析
#    全部 raw 臂。持久盘操作,已存在的链接跳过;缺源即 FATAL。──
python - "$DEV_LIST" "$DATA_ROOT" "$PHASE_ROOT/raw_brats2023_official" <<'PY'
import json, os, sys
dev_list, data_root, official_root = sys.argv[1:4]
entries = json.load(open(dev_list))["training"]
linked = missing = existing = 0
for entry in entries:
    rel = entry["image"]
    dst = os.path.join(data_root, rel)
    if os.path.lexists(dst):
        existing += 1
        continue
    src = os.path.join(official_root, rel)
    if not os.path.isfile(src):
        missing += 1
        print(f"[dev-link] missing source: {rel}")
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    linked += 1
if missing:
    raise SystemExit(f"[FATAL] dev raw 在官方树缺失 {missing} 条,补链中止")
print(f"[dev-link] dev 臂补链完成: 新链 {linked},已存在 {existing}(共 {len(entries)} 条)")
PY

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

# ── 覆写 env json:embedding_base_dir 指新树,数据面指向本次作业的绝对路径;
#    raw 臂(replay/dev)与主 list 拼成合并 list——编码链、分片与对账都吃合并集 ──
REENCODE_ENV="$OUTPUT_DIR/env_reencode_ras.json"
python - "$ENV_JSON" "$REENCODE_ENV" "$TRAIN_LIST" "$REPLAY_LIST" "$DEV_LIST" "$DATA_ROOT" "$EMB_OUT_ROOT" "$VAE_PATH" "$OUTPUT_DIR" <<'PY'
import json, sys
src, dst, train_list, replay_list, dev_list, data_root, emb_root, vae, out_dir = sys.argv[1:10]
cfg = json.load(open(src))
cfg["data_base_dir"] = data_root
cfg["embedding_base_dir"] = emb_root
cfg["trained_autoencoder_path"] = vae
entries = json.load(open(train_list))["training"]
n_train = len(entries)
for path in (replay_list, dev_list):
    entries += json.load(open(path))["training"]
combined = f"{out_dir}/train_list_raw_cohorts.json"
json.dump({"training": entries}, open(combined, "w"))
cfg["json_data_list"] = combined
json.dump(cfg, open(dst, "w"), indent=2)
print(f"[reencode-ras] raw 臂合并 list 落盘: {combined}(主 {n_train} + replay/dev 共 {len(entries)})")
print(f"[reencode-ras] env 覆写落盘: {dst}")
PY

echo "============================================"
echo "序列③ T2:全量 embedding 重编码(RAS 修复版,#312)"
echo "raw 臂: $TRAIN_LIST + $REPLAY_LIST + $DEV_LIST"
echo "P2/P3 臂(只对账+守卫): $EMB_LISTS"
echo "raw 根: $DATA_ROOT"
echo "新 embedding 根: $EMB_OUT_ROOT(legacy/cliptrue 旧树不动)"
echo "VAE: $VAE_PATH(冻结,只读)"
echo "分片: SHARDS=$SHARDS;收尾(sidecar+manifest+守卫+抽检)设备: $DEVICE(LIMIT=$SELF_CHECK_LIMIT)"
echo "报告输出: $OUTPUT_DIR"
echo "variant=diagnostic — 不产生任何验收判定"
echo "============================================"

# ── 单次作业调用(分片 worker 传 --encode-only,收尾传 --skip-encode)──
EXTRA_LIST_ARGS=(--extra-list "$REPLAY_LIST" --extra-list "$DEV_LIST")
EMB_LIST_ARGS=()
for f in $EMB_LISTS; do
    EMB_LIST_ARGS+=(--emb-list "$f")
done
SIDE_CAR_ARGS=()
if [ -n "$SIDE_CAR_SOURCES" ]; then
    for root in $SIDE_CAR_SOURCES; do
        SIDE_CAR_ARGS+=(--sidecar-source "$root")
    done
fi

run_reencode() {
    local env_json="$1"; local train_list="$2"; local device="$3"; shift 3
    python -m ctmr.application.generation.modality_label.reencode_ras \
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

if [ "$SHARDS" -le 1 ]; then
    run_reencode "$REENCODE_ENV" "$TRAIN_LIST" "$DEVICE" \
        --self-check-limit "$SELF_CHECK_LIMIT" --bootstrap-b "$BOOTSTRAP_B" \
        ${EXTRA_LIST_ARGS[@]+"${EXTRA_LIST_ARGS[@]}"} ${EMB_LIST_ARGS[@]+"${EMB_LIST_ARGS[@]}"} \
        ${SIDE_CAR_ARGS[@]+"${SIDE_CAR_ARGS[@]}"}
else
    # ── 阶段 1:按 idx%SHARDS 均分 raw 合并 list,每片一个单卡 worker(--encode-only)。
    # GPU 绑定用 CUDA_VISIBLE_DEVICES 隔离(T4 先例 PR #299 review F1:编码链内部
    # initialize_distributed 在 num_gpus=1 时恒取 cuda:0)。worker 的 --train-list 是
    # 分片 list(env_shard 由模块自派生,链吃分片 list);emb 臂不进分片(不驱动编码)。──
    python - "$REENCODE_ENV" "$OUTPUT_DIR" "$SHARDS" <<'PY'
import json, sys
env_path, out_dir, n_shards_s = sys.argv[1:4]
n_shards = int(n_shards_s)
listing = json.load(open(env_path))["json_data_list"]
entries = json.load(open(listing))["training"]
shards = [[] for _ in range(n_shards)]
for idx, entry in enumerate(entries):
    shards[idx % n_shards].append(entry)
for i, shard in enumerate(shards):
    shard_list = f"{out_dir}/train_list_shard_{i}.json"
    json.dump({"training": shard}, open(shard_list, "w"))
print(f"[reencode-ras] {n_shards} 片分片落盘: {[len(s) for s in shards]}(链吃分片 list)")
PY
    pids=()
    for i in $(seq 0 $((SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES="$i" run_reencode "$REENCODE_ENV" "$OUTPUT_DIR/train_list_shard_$i.json" "cuda:0" --encode-only \
            > "$OUTPUT_DIR/shard_$i.log" 2>&1 &
        pids+=($!)
    done
    for i in $(seq 0 $((SHARDS - 1))); do
        wait "${pids[$i]}" || { echo "[FATAL] shard $i 退出码非零,见 $OUTPUT_DIR/shard_$i.log" >&2; exit 1; }
    done
    # ── 阶段 2:收尾进程(全量 raw 臂 + P2/P3 emb 臂)——sidecar 多源拷贝校验 +
    # manifest 对账 + 形状守卫 + 抽检 + 报告;missing/违例非空即退出码非零 ──
    run_reencode "$REENCODE_ENV" "$TRAIN_LIST" "$DEVICE" \
        --skip-encode --self-check-limit "$SELF_CHECK_LIMIT" --bootstrap-b "$BOOTSTRAP_B" \
        ${EXTRA_LIST_ARGS[@]+"${EXTRA_LIST_ARGS[@]}"} ${EMB_LIST_ARGS[@]+"${EMB_LIST_ARGS[@]}"} \
        ${SIDE_CAR_ARGS[@]+"${SIDE_CAR_ARGS[@]}"}
fi

echo ""
echo "============================================"
echo "  完成。重编码产物与读数报告:"
echo "    新树:     $EMB_OUT_ROOT(legacy/cliptrue 旧树均不动)"
echo "    manifest: $EMB_OUT_ROOT/manifest.jsonl(md5/bytes/shape,供 T5-T7 重训审计)"
echo "    报告:     $OUTPUT_DIR/embedding_reencode_ras_report.{json,md}"
echo "  判读:对账 missing=0 且形状守卫违例=0 即全绿;抽检读数量级参照作业 C 双链锚"
echo "  (直通臂 0.006 量级、工件臂 0.08 量级)。读数供转写 deploy/experiments/(工件不入 git)"
echo "============================================"
