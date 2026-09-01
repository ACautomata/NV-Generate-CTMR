#!/bin/bash
# 序列② T7(issue #254,父 #247):P1 整改重训 run——单臂三改动,sugon DCU 训练作业
#
# 用途:以 P1 候选同款配方(run p1-20260822T131947Z)为基底执行整改重训,改动恰三项:
#   ① 编码 clip=True——训练 embedding 消费 T4(#251)重编码树 embeddings_cliptrue/(BraTS 主
#     list 全量 + MR-RATE replay 经 raw 重获取后同根重编码,均 clip=True;replay raw 已随
#     服务器重组消失,由 mrrate_replay_reacquire.py 重获取——经负责人决策,#254);
#   ② token 34 冻结(T3,#250)——config frozen_modality_tokens=[34],该 token P(保留)=1;
#   ③ 写出 affine(T2,#249)——推理写出侧(dev 侧车 CandidateSampler)以 V1_DM_OUTPUT_GRID
#     实声明真实采样 spacing,已在代码,配方侧零动作。
#   重训定位为假设检验:「协议修正包足以恢复空间保真」。epoch、lr、1:1 replay、CFG=10、
#   采样步数一概不动;不做并行消融(z-pad 留观,触发条件见 #247)。
#
# 【经负责人确认的两项记录在案偏差(#254 会话决议,如实登记)】
#   A. GPU 拓扑:基底 world_size 7(训练)+1(侧车);本作业 4×DCU 实例,取 world_size 4
#      (训练)+侧车共享 GPU0。world_size 决定每 epoch 优化步数与 PolynomialLR total_steps
#      (代码 total_steps = n_epochs × len(per_rank_dataset)),故有效 batch 7→4、schedule
#      长度随之变化——此为拓扑偏差,记入配方 diff 核对表,非协议项。
#   B. replay 编码:基底 replay embedding 为 clip=False;本作业按负责人决策将 replay 与
#      BraTS 一致改为 clip=True(扩展改动①的施用面至全部训练输入)。
#
# 用法:
#   bash deploy/jobs/run_p1_retrain_t7.sh            # 前置校验 + 核对表落盘 + 拉起双进程
#   bash deploy/jobs/run_p1_retrain_t7.sh --dry-run  # 只做校验+核对表,不拉起
#
# 环境变量(均可覆写,默认按 2026-09-01 实例实测持久盘布局):
#   T7_ROOT      T7 运行根(默认 /root/private_data/ctmr/runs/p1_t7;ckpt/dev_eval/logs 落此)
#   DEPLOY_ROOT  部署树(默认 /root/private_data/ctmr/deploy_t7;src+configs,持久盘防易失)
#   DATA_ROOT    phase 数据根(默认 /root/private_data/ctmr/data/phase)
#   P1_ROOT      基底 P1 运行根(默认 /root/private_data/ctmr/runs/p1;读基底 run.json
#                与 reference bank)
#   MODELS_ROOT  模型根(默认 /root/private_data/ctmr/models;底座 DM + 冻结 VAE)
#   NUM_GPUS     训练 world_size(默认 4,偏差 A;基底为 7)
#   SIDE_CAR_GPU 侧车可见 GPU(默认 0,与训练 rank0 共享)
#   IDLE_EXIT    侧车空闲退出秒(默认 7200;早停触发即自行退出,此为兜底)
#   BASE_RUN_ID  基底 run_id(默认 p1-20260822T131947Z)
#
# 前置条件(脚本逐项校验,任一缺失即 FATAL 不拉起):
#   1. 底座 DM checkpoint sha256 与基底 run.json 钉值逐位一致(全参续训的初始化锚)
#   2. 冻结 VAE md5 = 917cfb1e49631c8a713e3bb7c758fbca(冻结 canonical,只读)
#   3. embeddings_cliptrue 覆盖训练 list 全量 7404 + replay list 全量 7404(embedding+sidecar)
#   4. config_brats_p1_train.json 含 frozen_modality_tokens=[34](T3);P1RecipeSpec 启动时再守卫
#   5. dev 侧车 reference bank 可复用(同 dev list 同预处理,确定性等价——拷贝即缓存命中)
#   6. RadImageNet FID 特征网在 torch.hub 缓存(drive.google.com 被代理拦截,须离线就位)
#   7. DCU 卡数 = NUM_GPUS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

T7_ROOT="${T7_ROOT:-/root/private_data/ctmr/runs/p1_t7}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/root/private_data/ctmr/deploy_t7}"
DATA_ROOT="${DATA_ROOT:-/root/private_data/ctmr/data/phase}"
P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
MODELS_ROOT="${MODELS_ROOT:-/root/private_data/ctmr/models}"
NUM_GPUS="${NUM_GPUS:-4}"
SIDE_CAR_GPU="${SIDE_CAR_GPU:-0}"
IDLE_EXIT="${IDLE_EXIT:-7200}"
BASE_RUN_ID="${BASE_RUN_ID:-p1-20260822T131947Z}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

TRAIN_LIST="$DATA_ROOT/lists/p1_image_only.json"
REPLAY_LIST="$P1_ROOT/lists/p1_mrrate_replay.json"
DEV_LIST="$DATA_ROOT/lists/p1_image_only_dev.json"
EMB_ROOT="$DATA_ROOT/embeddings_cliptrue"
EMB_ROOT_LEGACY="$DATA_ROOT/embeddings"            # dev cohort spacing sidecar 所在(旧树,clip 无关)
BASE_CKPT="$MODELS_ROOT/diff_unet_3d_rflow-mr-brain_v1.pt"
VAE_PATH="$MODELS_ROOT/autoencoder_v1.pt"
BASE_RUN_JSON="$P1_ROOT/records/runs/$BASE_RUN_ID/run.json"
# 同仪器口径(#254):服务器重组后预处理栈版本已变,基线 bank(旧栈 real 特征)与
# T7 侧车(新栈)混栈读数带系统偏置;real bank 以当前栈从官方源重取的 dev raw
# 重提(reference_reinstr),T7 侧车与基底 trend 重算共用此 bank,判定线零改动。
REF_BANK_SRC="$P1_ROOT/dev_eval/reference_reinstr/real_reference_bank.pt"
NV_CONFIGS="$DEPLOY_ROOT/configs"
MODEL_JSON="$NV_CONFIGS/config_brats_p1_train.json"
NET_JSON="$NV_CONFIGS/config_network_rflow.json"
VAE_MD5_PIN="917cfb1e49631c8a713e3bb7c758fbca"
# T3 后 train config 终态钉值:与基底 config(基底 run p1-20260822T131947Z 所用,
# git 取证 b78f7b5~1)全文 diff 恰为 frozen_modality_tokens:[34] 新增——协议改动②
# 是 config 面唯一 delta 的机读锚。NETWORK config 不钉此处:launch 时直接对基底
# run.json configs[role=network].sha256 机读对账(当前已验证与钉值逐位一致)。
TRAIN_CFG_SHA_PIN="6c4cdf58eac54a5024130e1ef4e5099b193924e7d37d142118e3c5fcdc495dd2"
# AC4:P3 e39 checkpoint 不动——launch 前存在性 + sha256 观测(记入核对表,不比对历史值)
P3_E39_CKPT="${P3_E39_CKPT:-/root/private_data/ctmr/runs/p3/ckpt/epoch_39.pt}"
T7_CKPT_DIR="$T7_ROOT/ckpt"
T7_EVAL_ROOT="$T7_ROOT/dev_eval"
T7_LOGS="$T7_ROOT/logs"

# ── 双 source(DTK 算 + 平台代理网;非交互 ssh 需显式)──
# env.sh 引用未初始化变量(如 CMAKE_PREFIX_PATH),set -u 下展开级致命(|| true 拦不住),临时关 -u
set +u
source /opt/dtk/env.sh 2>/dev/null || true
set -u
eval "$(grep -E '^export (HF_TOKEN|HF_ENDPOINT)=' ~/.bashrc 2>/dev/null)" || true

echo "============================================"
echo "序列② T7:P1 整改重训(单臂三改动)(#254,父 #247)"
echo "基底: run $BASE_RUN_ID | world_size $NUM_GPUS(偏差 A,基底 7)| 侧车 GPU$SIDE_CAR_GPU 共享"
echo "T7_ROOT=$T7_ROOT"
echo "============================================"

# ── 前置 1-4:底座锚、VAE、embedding 覆盖、冻结 token ──
[ -f "$BASE_RUN_JSON" ] || { echo "[FATAL] 基底 run.json 不存在: $BASE_RUN_JSON" >&2; exit 1; }
[ -f "$BASE_CKPT" ] || { echo "[FATAL] 底座 DM checkpoint 不存在: $BASE_CKPT" >&2; exit 1; }
[ -f "$VAE_PATH" ] || { echo "[FATAL] 冻结 VAE 不存在: $VAE_PATH" >&2; exit 1; }
for f in "$TRAIN_LIST" "$REPLAY_LIST" "$DEV_LIST" "$MODEL_JSON" "$NET_JSON"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done

python3 - "$BASE_RUN_JSON" "$BASE_CKPT" "$VAE_PATH" "$VAE_MD5_PIN" "$TRAIN_LIST" "$REPLAY_LIST" "$EMB_ROOT" "$MODEL_JSON" "$T7_ROOT" "$NUM_GPUS" "$NET_JSON" "$TRAIN_CFG_SHA_PIN" "$P3_E39_CKPT" <<'PY'
import hashlib, json, sys
from pathlib import Path

run_json, base_ckpt, vae, vae_md5_pin, train_list, replay_list, emb_root, model_json, t7_root, num_gpus, net_json, cfg_sha_pin, p3_e39 = sys.argv[1:14]

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

base = json.loads(Path(run_json).read_text())
pinned = base.get("base_ckpt", {}).get("sha256", "")
actual = sha256_of(base_ckpt)
if actual != pinned:
    raise SystemExit(f"[FATAL] 底座 sha256 不匹配: 实算 {actual} vs 钉值 {pinned}")
print(f"[preflight] 底座 sha256 对账一致: {actual[:12]}…")

vae_md5 = md5_of(vae)
if vae_md5 != vae_md5_pin:
    raise SystemExit(f"[FATAL] VAE md5 不匹配: {vae_md5} vs {vae_md5_pin}")
print(f"[preflight] VAE md5 对账一致: {vae_md5}")

cfg = json.loads(Path(model_json).read_text())
train_cfg = cfg.get("diffusion_unet_train", {})
# config 面钉值:T7 train config 必须恰为 T3 终态(基底 + frozen_modality_tokens[34],
# 见 TRAIN_CFG_SHA_PIN 注释的 git 取证链);任何其他 config 改动都使「改动恰三项」失效。
cfg_sha = sha256_of(model_json)
if cfg_sha != cfg_sha_pin:
    raise SystemExit(f"[FATAL] train config sha256 非钉值(T3 终态): {cfg_sha[:16]}… vs {cfg_sha_pin[:16]}…")
freeze = train_cfg.get("frozen_modality_tokens")
if freeze != [34]:
    raise SystemExit(f"[FATAL] frozen_modality_tokens 必须恰为 [34](T3),got {freeze}")
# 逐键断言基底超参真值(git 取证基底 config:lr 2e-6/bs 1/epochs 100/cache 0;ADR-0005 钉 lr)
for key, pin in (("lr", 2e-06), ("batch_size", 1), ("n_epochs", 100), ("cache_rate", 0)):
    if train_cfg.get(key) != pin:
        raise SystemExit(f"[FATAL] 超参 {key} = {train_cfg.get(key)} != 基底真值 {pin}(一概不动)")
# network config 对基底 run.json 机读钉值(run.json configs[role=network].sha256)
net_pin = next((c.get("sha256") for c in base.get("configs", []) if c.get("role") == "network"), "")
if not net_pin or sha256_of(net_json) != net_pin:
    raise SystemExit(f"[FATAL] network config sha256 与基底 run.json 机读钉值不符: {net_pin[:16]}…")
infer_cfg = cfg.get("diffusion_unet_inference", {})
if infer_cfg.get("cfg_guidance_scale") != 10 or infer_cfg.get("num_inference_steps") != 30:
    raise SystemExit(f"[FATAL] dev 采样配方须 cfg=10/30 步(不动),got {infer_cfg}")
print(f"[preflight] config 面:train= T3 终态钉值 + 基底超参逐键一致;network= 基底机读钉值一致;cfg10/30步")
# AC4 观测:P3 e39 checkpoint 在位(不动它,只登记)
p3 = Path(p3_e39)
if not p3.is_file():
    raise SystemExit(f"[FATAL] P3 e39 checkpoint 不存在(AC4 观测对象缺失): {p3}")
print(f"[preflight] P3 e39 在位: {p3} sha256={sha256_of(p3)[:12]}…(观测,不动)")

emb = Path(emb_root)
missing = []
for label, path in (("train", train_list), ("replay", replay_list)):
    entries = json.loads(Path(path).read_text())["training"]
    print(f"[preflight] {label} list {len(entries)} 条")
    for entry in entries:
        rel = entry["image"].replace(".nii.gz", "_emb.nii.gz")
        if not (emb / rel).is_file() or not (emb / (rel + ".json")).is_file():
            missing.append(f"{label}:{entry['image']}")
    if len(entries) != 7404:
        raise SystemExit(f"[FATAL] {label} list 条数 {len(entries)} != 7404(1:1 replay 配方)")
if missing:
    raise SystemExit(f"[FATAL] embedding 树缺 {len(missing)} 条(首3: {missing[:3]})——重编码未完成,不得开训")
print("[preflight] embeddings_cliptrue 覆盖 train+replay 各 7404(embedding+sidecar)")

manifest = emb / "manifest.jsonl"
if manifest.is_file():
    n_rows = sum(1 for _ in open(manifest))
    print(f"[preflight] manifest.jsonl {n_rows} 行(审计面;T7 消费以逐文件对账为准)")
Path(t7_root).mkdir(parents=True, exist_ok=True)
print("[preflight] 全部通过")
PY

# ── 前置 5-7:reference bank、RadImageNet、GPU 数 ──
[ -f "$REF_BANK_SRC" ] || { echo "[FATAL] reference bank 不存在: $REF_BANK_SRC" >&2; exit 1; }
HUB_CKPT_DIR="$(python3 -c 'import torch.hub,os;print(os.path.join(torch.hub.get_dir(),"checkpoints"))' 2>/dev/null || echo "$HOME/.cache/torch/hub/checkpoints")"
[ -f "$HUB_CKPT_DIR/RadImageNet-ResNet50_notop.pth" ] || {
    echo "[FATAL] RadImageNet 权重不在 torch.hub 缓存($HUB_CKPT_DIR)——drive.google.com 被代理拦截,须离线就位后重跑" >&2; exit 1; }
NGPU="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[ "$NGPU" = "$NUM_GPUS" ] || { echo "[FATAL] DCU 卡数 $NGPU != NUM_GPUS $NUM_GPUS(偏差 A 拓扑须与拉起一致)" >&2; exit 1; }
echo "[preflight] reference bank / RadImageNet / DCU×$NGPU 就位"

# ── 配方 diff 核对表(超参逐项与基底一致;三项协议改动 + 两项登记偏差)──
CHECKLIST="$T7_ROOT/recipe_diff_checklist.json"
python3 - "$BASE_RUN_JSON" "$BASE_CKPT" "$MODEL_JSON" "$NET_JSON" "$TRAIN_LIST" "$REPLAY_LIST" "$EMB_ROOT" "$CHECKLIST" "$NUM_GPUS" "$BASE_RUN_ID" "$P3_E39_CKPT" <<'PY'
import hashlib, json, sys
from pathlib import Path

run_json, base_ckpt, model_json, net_json, train_list, replay_list, emb_root, out, num_gpus, base_run_id, p3_e39 = sys.argv[1:12]

base = json.loads(Path(run_json).read_text())
cfg = json.loads(Path(model_json).read_text())
net = json.loads(Path(net_json).read_text())
train_cfg, infer_cfg = cfg["diffusion_unet_train"], cfg["diffusion_unet_inference"]
sched = net["noise_scheduler"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# 基底侧写:train_provenance 的 hyperparameters 为空 dict(基底 run.json 未记超参),
# 故基底超参取值一律来自 git 取证的基底 config(b78f7b5~1,= T3 合入前)与 ADR-0005,
# 且已在 preflight 与当前 config 逐键强校验——此处为记录面,不是校验面。
base_platform = base.get("platform", {})
base_ws = base_platform.get("world_size") or base.get("manifest", {}).get("world_size")
p3_sha = sha(p3_e39)
rows = []
def row(item, base_v, t7_v, klass, note=""):
    rows.append({"item": item, "base": base_v, "t7": t7_v, "class": klass, "note": note})

# —— 不动项(逐项对账;base 侧 = git 取证基底真值,非机读)——
row("lr", 2e-06, train_cfg.get("lr"), "unchanged", "ADR-0005 钉值;preflight 已强校验")
row("batch_size", 1, train_cfg.get("batch_size"), "unchanged", "per-rank;基底 config git 取证;preflight 已强校验")
row("n_epochs(上限)", 100, train_cfg.get("n_epochs"), "unchanged", "早停由侧车预录规则驱动;preflight 已强校验")
row("cache_rate", 0, train_cfg.get("cache_rate"), "unchanged", "基底 config git 取证;preflight 已强校验")
row("RF scheduler.sample_method", "uniform", sched.get("sample_method"), "unchanged", "base:ADR-0005;t7:机读 net json")
row("RF scheduler.scale", 1.4, sched.get("scale"), "unchanged", "base:ADR-0005;t7:机读 net json")
row("network config sha256", next((c.get("sha256", "")[:16] + "…" for c in base.get("configs", []) if c.get("role") == "network"), "?"), sha(net_json)[:16] + "…", "unchanged", "双侧机读:基底 run.json 钉值 vs 当前文件,逐位一致")
row("train config sha256", "6a23f65a23d0b6c5…(基底,git 钉值)", sha(model_json)[:16] + "…(T3 终态)", "protocol_change_note", "差恰 frozen_modality_tokens:[34] 新增(git 取证 b78f7b5~1)")
row("PolynomialLR power", 2.0, 2.0, "unchanged", "代码常量")
row("loss", "L1", "L1", "unchanged", "代码常量")
row("augment_modality_label prob", 0.1, 0.1, "unchanged", "代码常量")
row("replay 混合", "1:1(7404+7404)", "1:1(7404+7404)", "unchanged", "DataCatalog 强校验")
row("dev 侧车 cfg_guidance_scale", 10, infer_cfg.get("cfg_guidance_scale"), "unchanged", "预录采样配方")
row("dev 侧车 num_inference_steps", 30, infer_cfg.get("num_inference_steps"), "unchanged", "预录采样配方")
row("amp", base_platform.get("amp_dtype") or "bf16", "bf16", "unchanged", "DCU 默认")
row("底座 checkpoint sha256", base.get("base_ckpt", {}).get("sha256", "")[:16] + "…", sha(base_ckpt)[:16] + "…", "unchanged", "全参续训初始化锚,双侧机读一致")
row("冻结仪器/包络/判定线", "ADR-0002/0004 冻结", "零改动", "unchanged", "#247 Out of Scope")
row("P3 e39 checkpoint", "不动", "不动", "unchanged", f"#247 Out of Scope;launch 时观测在位 sha256={p3_sha[:12]}…")
# —— 恰三项协议改动 ——
row("①编码 clip", "clip=False(旧 embedding)", "clip=True(T4 重编码树 + replay 同根重编码)", "protocol_change", "改动①;replay 施用面扩展经负责人决策(#254)")
row("②token 34 冻结", "无(增广全分布)", "frozen_modality_tokens=[34]", "protocol_change", "改动②(T3,#250);config 面 git 取证唯一 delta")
row("③写出 affine", "单位 1mm 声明", "V1_DM_OUTPUT_GRID 真实 spacing", "protocol_change", "改动③(T2,#249;代码已合入,dev 侧车写出侧生效)")
# —— 登记偏差(非协议项)——
row("world_size(拓扑)", base_ws or 7, int(num_gpus), "approved_delta", "偏差 A(#254 会话决议):4×DCU 实例;有效 batch 与 LR schedule 长度随拓扑变化")

checklist = {
    "schema": "p1-retrain-recipe-diff/1",
    "base_run_id": base_run_id,
    "t7_run_root": str(Path(out).parent),
    "protocol_changes": [r for r in rows if r["class"] == "protocol_change"],
    "approved_deltas": [r for r in rows if r["class"] == "approved_delta"],
    "unchanged": [r for r in rows if r["class"] == "unchanged"],
    "rows": rows,
}
Path(out).write_text(json.dumps(checklist, indent=2, ensure_ascii=False) + "\n")
n_change = len(checklist["protocol_changes"])
n_delta = len(checklist["approved_deltas"])
print(f"[checklist] 配方 diff:协议改动 {n_change} 项(须=3)、登记偏差 {n_delta} 项(须=1)、不动 {len(checklist['unchanged'])} 项 -> {out}")
if n_change != 3 or n_delta != 1:
    raise SystemExit("[FATAL] 配方 diff 面不符预期(协议改动≠3 或登记偏差≠1)——整改定位失效,不得开训")
PY
echo "[checklist] 核对表: $CHECKLIST"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 校验与核对表完成,未拉起。"
    exit 0
fi

# ── 拉起:先侧车(FID-only,T7 只做 FID 选择;ET/WT 监控是 T6),再训练 ──
mkdir -p "$T7_CKPT_DIR" "$T7_EVAL_ROOT" "$T7_LOGS"
mkdir -p "$T7_EVAL_ROOT/reference"
cp "$REF_BANK_SRC" "$T7_EVAL_ROOT/reference/real_reference_bank.pt"
echo "[launch] reference bank 拷贝至 $T7_EVAL_ROOT/reference/(确定性缓存命中,同 dev list 同预处理)"

ENV_JSON="$T7_ROOT/environment_brats_p1_train_t7.json"
python3 - "$ENV_JSON" "$EMB_ROOT" "$TRAIN_LIST" "$T7_CKPT_DIR" "$VAE_PATH" "$BASE_CKPT" "$NV_CONFIGS" "$DATA_ROOT" <<'PY'
import json, sys
dst, emb, lst, ckpt, vae, base_ckpt, nv, data_root = sys.argv[1:9]
env = {
    "data_base_dir": f"{data_root}/raw_relinked",
    "embedding_base_dir": emb,
    "json_data_list": lst,
    "model_dir": ckpt,
    "model_filename": "epoch_N.pt",
    "trained_autoencoder_path": vae,
    "existing_ckpt_filepath": base_ckpt,
    "modality_mapping_path": f"{nv}/modality_mapping.json",
}
json.dump(env, open(dst, "w"), indent=4)
print(f"[launch] env json 覆写落盘: {dst}(data_base_dir={data_root}/raw_relinked, embedding_base_dir -> {emb})")
PY

TS="$(date -u +%Y%m%dT%H%M%SZ)"
SIDE_LOG="$T7_LOGS/dev_eval_$TS.log"
TRAIN_LOG="$T7_LOGS/train_$TS.log"

setsid nohup env CUDA_VISIBLE_DEVICES="$SIDE_CAR_GPU" python3 -m ctmr generate modality-label dev-eval watch \
    --ckpt-dir "$T7_CKPT_DIR" --eval-root "$T7_EVAL_ROOT" \
    --dev-list "$DEV_LIST" --raw-root "$P1_ROOT/raw" --emb-root "$EMB_ROOT_LEGACY" \
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON" \
    --eval-every 5 --patience 3 --min-epoch 30 --max-epoch 100 \
    --skip-l2 --idle-exit-seconds "$IDLE_EXIT" \
    > "$SIDE_LOG" 2>&1 < /dev/null &
SIDE_PID=$!
echo "[launch] dev 侧车(FID-only,共享 GPU$SIDE_CAR_GPU)pid=$SIDE_PID log=$SIDE_LOG"

setsid nohup python3 -m ctmr generate modality-label train \
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON" --replay-list "$REPLAY_LIST" -g "$NUM_GPUS" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "[launch] 训练(torchrun world_size=$NUM_GPUS)pid=$TRAIN_PID log=$TRAIN_LOG"

echo "============================================"
echo "  T7 已拉起。监控:"
echo "    tail -f $TRAIN_LOG      # 训练"
echo "    tail -f $SIDE_LOG       # dev 侧车(FID trend/早停)"
echo "  完成后(侧车早停或 100 epochs):"
echo "    python3 -m ctmr generate modality-label dev-eval select \\"
echo "      --eval-root $T7_EVAL_ROOT --ckpt-dir $T7_CKPT_DIR --out $T7_ROOT/selection.json"
echo "============================================"
