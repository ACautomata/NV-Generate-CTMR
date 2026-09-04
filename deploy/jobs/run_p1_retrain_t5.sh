#!/bin/bash
# 序列③ T5(issue #315,父 #310):P1 重训 run——干净方向世界首训,sugon DCU 训练作业
#
# 用途:以 T7 整改重训 run(p1_t7,#254)同款配方为载体执行干净方向世界的 P1 重训,
# 相对 T7 的改动恰一项(数据树替换):
#   训练 embedding 消费序列③T2(#312)RAS 修复全量重编码树 embeddings_ras/
#   (训练臂 7404 + replay 臂 7404;dev cohort spacing sidecar 源同树——T2 多源拷贝
#   自旧树,spacing 值与 T7 消费值逐位同)。T7 三改动配方(clip=True 编码 / token 34
#   冻结 / 写出 affine)沿袭;超参逐项不动(lr 2e-6、bs 1、100 epochs 上限、replay 1:1、
#   cfg=10、30 步);world_size=4 沿 T7 偏差 A 登记口径(#310 Implementation Decisions)。
#
# 【方向世界差异声明(#310,本 run 与 T7 的世界差异)】
#   T7 训练世界的 replay 臂 embedding 存在 ~65% 轴序错乱(#310 方向审计:编码链 resize
#   目标误用 NIfTI header 原始轴序 dim),被全卷积 DM 静默吃进训练;本 run 的
#   embeddings_ras 树由修复后编码链(RAS 重排后 spatial_shape 决策 resize 目标)全量
#   重编码,manifest 15868 行对账全绿 + 逐条形状守卫全绿(T2 记录)。labels 树不变:
#   T1(#311)撤销——复核实证旧树已 RAS(重生成空转),且 P1 image-only 训练不消费
#   labels。dev FID trend 与 T7/基底同仪器同 bank(reference_reinstr)可比。
#
# 【dev 监控形态(现役口径,#278/#279,ADR-0019 §5;与 T7 逐字同族)】
#   trainer 内嵌周期验证 --val-every 5(基底评估网格),16 例×4 模态 dev cohort 全卡
#   分片,cfg=10/30 步同钉值;早停规则 ADR-0005 钉值(patience 3/min 30/max=100)经
#   trainer 内嵌评估并经 <ckpt_dir>/.early_stop 停机;reference bank 预拷同仪器重提
#   产物(reference_reinstr),T5 与 T7/基底 trend 全程同栈同 bank 可比。
#
# 用法:
#   bash deploy/jobs/run_p1_retrain_t5.sh            # 前置校验 + 核对表落盘 + 拉起训练
#   bash deploy/jobs/run_p1_retrain_t5.sh --dry-run  # 只做校验+核对表,不拉起
#
# 环境变量(均可覆写,默认按 2026-09-04 实例实测持久盘布局):
#   T5_ROOT      T5 运行根(默认 /root/private_data/ctmr/runs/p1_t5;ckpt/dev_eval/logs 落此)
#   DEPLOY_ROOT  部署树(默认 /root/private_data/ctmr/deploy_t5;src+configs,持久盘防易失)
#   DATA_ROOT    phase 数据根(默认 /root/private_data/ctmr/data/phase)
#   P1_ROOT      基底 P1 运行根(默认 /root/private_data/ctmr/runs/p1;读基底 run.json、
#                replay list、reference bank 与 raw 树)
#   MODELS_ROOT  模型根(默认 /root/private_data/ctmr/models;底座 DM + 冻结 VAE)
#   NUM_GPUS     训练 world_size(默认 4;T7 偏差 A 口径,基底为 7)
#   BASE_RUN_ID  超参基底 run_id(默认 p1-20260822T131947Z)
#
# 前置条件(脚本逐项校验,任一缺失即 FATAL 不拉起):
#   1. 底座 DM checkpoint sha256 与基底 run.json 钉值逐位一致(全参续训的初始化锚)
#   2. 冻结 VAE md5 = 917cfb1e49631c8a713e3bb7c758fbca(冻结 canonical,只读)
#   3. embeddings_ras(T2 新树)覆盖训练 list 全量 7404 + replay list 全量 7404
#      (embedding+sidecar),且 manifest.jsonl = 15868 行(T2 退出条件锚)
#   4. config_brats_p1_train.json = T3 终态钉值(与 T7 同一钉值;config 面相对 T7 零
#      delta);network config 对基底 run.json 机读对账;基底超参逐键断言;
#      P1RecipeSpec 启动时再守卫
#   5. dev 内嵌验证 reference bank(同仪器重提产物)可复用——拷贝即缓存命中
#      (同 dev list 同预处理,确定性等价;eval_root=<ckpt_dir>/dev_eval)
#   6. RadImageNet FID 特征网在 torch.hub 缓存(drive.google.com 被代理拦截,须离线就位)
#   7. DCU 卡数 = NUM_GPUS;rendezvous 29500 端口空闲(T7 先例:旧 worker 残留占端口致拉起失败)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

T5_ROOT="${T5_ROOT:-/root/private_data/ctmr/runs/p1_t5}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/root/private_data/ctmr/deploy_t5}"
DATA_ROOT="${DATA_ROOT:-/root/private_data/ctmr/data/phase}"
P1_ROOT="${P1_ROOT:-/root/private_data/ctmr/runs/p1}"
MODELS_ROOT="${MODELS_ROOT:-/root/private_data/ctmr/models}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_RUN_ID="${BASE_RUN_ID:-p1-20260822T131947Z}"
T7_ROOT="${T7_ROOT:-/root/private_data/ctmr/runs/p1_t7}"   # 配方载体 run(T7)根,核对表记录面用
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

TRAIN_LIST="$DATA_ROOT/lists/p1_image_only.json"
REPLAY_LIST="$P1_ROOT/lists/p1_mrrate_replay.json"
DEV_LIST="$DATA_ROOT/lists/p1_image_only_dev.json"
EMB_ROOT="$DATA_ROOT/embeddings_ras"               # T2(#312)RAS 修复重编码新树——本 run 唯一数据树替换项
BASE_CKPT="$MODELS_ROOT/diff_unet_3d_rflow-mr-brain_v1.pt"
VAE_PATH="$MODELS_ROOT/autoencoder_v1.pt"
BASE_RUN_JSON="$P1_ROOT/records/runs/$BASE_RUN_ID/run.json"
# 同仪器口径(沿 T7):reference bank 用同仪器重提产物,T5 内嵌验证与 T7/基底 trend
# 共用此 bank,判定线零改动。
REF_BANK_SRC="$P1_ROOT/dev_eval/reference_reinstr/real_reference_bank.pt"
NV_CONFIGS="$DEPLOY_ROOT/configs"
MODEL_JSON="$NV_CONFIGS/config_brats_p1_train.json"
NET_JSON="$NV_CONFIGS/config_network_rflow.json"
VAE_MD5_PIN="917cfb1e49631c8a713e3bb7c758fbca"
# T3 后 train config 终态钉值:与 T7 消费的同一钉值(git 取证 b78f7b5~1 基底 config +
# frozen_modality_tokens:[34] 唯一差)——config 面相对 T7 零 delta 的机读锚。NETWORK
# config 不钉此处:launch 时直接对基底 run.json configs[role=network].sha256 机读对账。
TRAIN_CFG_SHA_PIN="6c4cdf58eac54a5024130e1ef4e5099b193924e7d37d142118e3c5fcdc495dd2"
# AC4:#310 Out of Scope——P3 e39 checkpoint 不动,launch 前存在性 + sha256 观测
P3_E39_CKPT="${P3_E39_CKPT:-/root/private_data/ctmr/runs/p3/ckpt/epoch_39.pt}"
# T2 退出条件的 manifest 锚:15868 行 = P1 主 7404 + replay 7404 + dev 1060(#312 记录)
EMB_MANIFEST_ROWS_PIN=15868
T5_CKPT_DIR="$T5_ROOT/ckpt"
T5_LOGS="$T5_ROOT/logs"

# ── 双 source(DTK 算 + 平台代理网;非交互 ssh 需显式)──
# env.sh 引用未初始化变量(如 CMAKE_PREFIX_PATH),set -u 下展开级致命(|| true 拦不住),临时关 -u
set +u
source /opt/dtk/env.sh 2>/dev/null || true
set -u
eval "$(grep -E '^export (HF_TOKEN|HF_ENDPOINT)=' ~/.bashrc 2>/dev/null)" || true

echo "============================================"
echo "序列③ T5:P1 重训(干净方向世界,数据树替换恰一项)(#315,父 #310)"
echo "配方载体:T7 run($T7_ROOT,#254)| 超参基底:$BASE_RUN_ID | world_size $NUM_GPUS(偏差 A 口径沿袭)"
echo "数据树:$EMB_ROOT(T2 #312 RAS 修复重编码)"
echo "T5_ROOT=$T5_ROOT"
echo "============================================"

# ── 前置 1-4:底座锚、VAE、embedding 覆盖、config 钉值 ──
[ -f "$BASE_RUN_JSON" ] || { echo "[FATAL] 基底 run.json 不存在: $BASE_RUN_JSON" >&2; exit 1; }
[ -f "$BASE_CKPT" ] || { echo "[FATAL] 底座 DM checkpoint 不存在: $BASE_CKPT" >&2; exit 1; }
[ -f "$VAE_PATH" ] || { echo "[FATAL] 冻结 VAE 不存在: $VAE_PATH" >&2; exit 1; }
for f in "$TRAIN_LIST" "$REPLAY_LIST" "$DEV_LIST" "$MODEL_JSON" "$NET_JSON"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done

python3 - "$BASE_RUN_JSON" "$BASE_CKPT" "$VAE_PATH" "$VAE_MD5_PIN" "$TRAIN_LIST" "$REPLAY_LIST" "$EMB_ROOT" "$MODEL_JSON" "$T5_ROOT" "$NUM_GPUS" "$NET_JSON" "$TRAIN_CFG_SHA_PIN" "$P3_E39_CKPT" "$EMB_MANIFEST_ROWS_PIN" <<'PY'
import hashlib, json, sys
from pathlib import Path

run_json, base_ckpt, vae, vae_md5_pin, train_list, replay_list, emb_root, model_json, t5_root, num_gpus, net_json, cfg_sha_pin, p3_e39, manifest_rows_pin = sys.argv[1:15]
manifest_rows_pin = int(manifest_rows_pin)

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
# config 面钉值:与 T7 消费的同一钉值(T3 终态 = 基底 + frozen_modality_tokens[34]);
# 任何 config 改动都使「相对 T7 改动恰一项(数据树替换)」失效。
cfg_sha = sha256_of(model_json)
if cfg_sha != cfg_sha_pin:
    raise SystemExit(f"[FATAL] train config sha256 非钉值(T3 终态,与 T7 同一钉值): {cfg_sha[:16]}… vs {cfg_sha_pin[:16]}…")
freeze = train_cfg.get("frozen_modality_tokens")
if freeze != [34]:
    raise SystemExit(f"[FATAL] frozen_modality_tokens 必须恰为 [34](T7 协议改动②沿袭),got {freeze}")
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
print(f"[preflight] config 面:train= T3 终态钉值(与 T7 同钉)+ 基底超参逐键一致;network= 基底机读钉值一致;cfg10/30步")
# AC4 观测:P3 e39 checkpoint 在位(不动它,只登记)
p3 = Path(p3_e39)
if not p3.is_file():
    raise SystemExit(f"[FATAL] P3 e39 checkpoint 不存在(AC4 观测对象缺失): {p3}")
print(f"[preflight] P3 e39 在位: {p3} sha256={sha256_of(p3)[:12]}…(观测,不动)")

emb = Path(emb_root)
manifest = emb / "manifest.jsonl"
if not manifest.is_file():
    raise SystemExit(f"[FATAL] T2 manifest 不存在: {manifest}(embeddings_ras 非完整重编码树,不得开训)")
n_rows = sum(1 for _ in open(manifest))
if n_rows != manifest_rows_pin:
    raise SystemExit(f"[FATAL] T2 manifest {n_rows} 行 != {manifest_rows_pin}(#312 退出条件锚;重编码不完整,不得开训)")
print(f"[preflight] T2 manifest 对账: {n_rows} 行(= 主 7404 + replay 7404 + dev 1060)")

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
    raise SystemExit(f"[FATAL] embeddings_ras 缺 {len(missing)} 条(首3: {missing[:3]})——T2 重编码未覆盖,不得开训")
print("[preflight] embeddings_ras 覆盖 train+replay 各 7404(embedding+sidecar)")

Path(t5_root).mkdir(parents=True, exist_ok=True)
print("[preflight] 全部通过")
PY

# ── 前置 5-7:reference bank、RadImageNet、GPU 数、rendezvous 端口 ──
[ -f "$REF_BANK_SRC" ] || { echo "[FATAL] reference bank 不存在: $REF_BANK_SRC" >&2; exit 1; }
HUB_CKPT_DIR="$(python3 -c 'import torch.hub,os;print(os.path.join(torch.hub.get_dir(),"checkpoints"))' 2>/dev/null || echo "$HOME/.cache/torch/hub/checkpoints")"
[ -f "$HUB_CKPT_DIR/RadImageNet-ResNet50_notop.pth" ] || {
    echo "[FATAL] RadImageNet 权重不在 torch.hub 缓存($HUB_CKPT_DIR)——drive.google.com 被代理拦截,须离线就位后重跑" >&2; exit 1; }
NGPU="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[ "$NGPU" = "$NUM_GPUS" ] || { echo "[FATAL] DCU 卡数 $NGPU != NUM_GPUS $NUM_GPUS(偏差 A 拓扑须与拉起一致)" >&2; exit 1; }
if (exec 3<>/dev/tcp/127.0.0.1/29500) 2>/dev/null; then
    exec 3<&- 3>&-
    echo "[FATAL] rendezvous 端口 29500 已被占用(T7 先例:旧 worker 残留占端口致拉起失败)——清场后重跑" >&2
    exit 1
fi
echo "[preflight] reference bank / RadImageNet / DCU×$NGPU / rendezvous 29500 空闲就位"

# ── 配方 diff 核对表(超参逐项与基底一致;相对 T7 改动恰一项:数据树替换;零新登记偏差)──
CHECKLIST="$T5_ROOT/recipe_diff_checklist.json"
python3 - "$BASE_RUN_JSON" "$BASE_CKPT" "$MODEL_JSON" "$NET_JSON" "$CHECKLIST" "$NUM_GPUS" "$BASE_RUN_ID" "$T7_ROOT" "$P3_E39_CKPT" "$EMB_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path

run_json, base_ckpt, model_json, net_json, out, num_gpus, base_run_id, t7_root, p3_e39, emb_root = sys.argv[1:11]

base = json.loads(Path(run_json).read_text())
cfg = json.loads(Path(model_json).read_text())
net = json.loads(Path(net_json).read_text())
train_cfg, infer_cfg = cfg["diffusion_unet_train"], cfg["diffusion_unet_inference"]
sched = net["noise_scheduler"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# 基底侧写:基底 run.json 的 train_provenance.hyperparameters 为空 dict,基底超参取值
# 一律来自 git 取证的基底 config(b78f7b5~1,= T3 合入前)与 ADR-0005,且已在 preflight
# 与当前 config 逐键强校验——此处为记录面,不是校验面。核对口径:超参基底 = run
# p1-20260822T131947Z;配方载体 = T7 run(p1_t7,#254)——T5 = T7 配方沿袭 + 数据树
# 替换恰一项,diff 的审计主语是「T5 vs T7」。
base_platform = base.get("platform", {})
p3_sha = sha(p3_e39)
rows = []
def row(item, base_v, t5_v, klass, note=""):
    rows.append({"item": item, "base": base_v, "t5": t5_v, "class": klass, "note": note})

# —— 不动项(逐项对账;base 侧 = git 取证基底真值或 T7 沿袭,非机读者已注明)——
row("lr", 2e-06, train_cfg.get("lr"), "unchanged", "ADR-0005 钉值;preflight 已强校验")
row("batch_size", 1, train_cfg.get("batch_size"), "unchanged", "per-rank;基底 config git 取证;preflight 已强校验")
row("n_epochs(上限)", 100, train_cfg.get("n_epochs"), "unchanged", "早停由 trainer 内嵌验证评估(ADR-0005 钉值,#278);preflight 已强校验")
row("cache_rate", 0, train_cfg.get("cache_rate"), "unchanged", "基底 config git 取证;preflight 已强校验")
row("RF scheduler.sample_method", "uniform", sched.get("sample_method"), "unchanged", "base:ADR-0005;t5:机读 net json")
row("RF scheduler.scale", 1.4, sched.get("scale"), "unchanged", "base:ADR-0005;t5:机读 net json")
row("network config sha256", next((c.get("sha256", "")[:16] + "…" for c in base.get("configs", []) if c.get("role") == "network"), "?"), sha(net_json)[:16] + "…", "unchanged", "双侧机读:基底 run.json 钉值 vs 当前文件,逐位一致")
row("train config sha256", "6c4cdf58eac54a50…(T3 终态钉值)", sha(model_json)[:16] + "…(同一钉值)", "unchanged", "与 T7 消费的同一钉值——config 面相对 T7 零 delta;对基底 run 的协议差异(token 34)已在 T7 登记")
row("①编码 clip(T7 沿袭)", "clip=True", "clip=True(embeddings_ras 同为 clip=True 编码)", "unchanged", "T7 协议改动①(#251/T4)沿袭")
row("②token 34 冻结(T7 沿袭)", "frozen_modality_tokens=[34]", "frozen_modality_tokens=[34]", "unchanged", "T7 协议改动②(T3,#250)沿袭;preflight 已强校验")
row("③写出 affine(T7 沿袭)", "V1_DM_OUTPUT_GRID 真实 spacing", "V1_DM_OUTPUT_GRID 真实 spacing", "unchanged", "T7 协议改动③(T2,#249)沿袭;代码面已合入,dev 验证写出侧生效")
row("PolynomialLR power", 2.0, 2.0, "unchanged", "代码常量")
row("loss", "L1", "L1", "unchanged", "代码常量")
row("augment_modality_label prob", 0.1, 0.1, "unchanged", "代码常量")
row("replay 混合", "1:1(7404+7404)", "1:1(7404+7404)", "unchanged", "DataCatalog 强校验")
row("dev 内嵌验证 cfg_guidance_scale", 10, infer_cfg.get("cfg_guidance_scale"), "unchanged", "预录采样配方")
row("dev 内嵌验证 num_inference_steps", 30, infer_cfg.get("num_inference_steps"), "unchanged", "预录采样配方")
row("dev 内嵌验证网格", "val-every 5(基底评估网格)", "val-every 5", "unchanged", "与 T7 逐字同族(#278/#279 现役形态)")
row("早停规则", "ADR-0005:patience 3/min 30/max=100", "ADR-0005:patience 3/min 30/max=n_epochs(100)", "unchanged", "trainer 内嵌经 .early_stop 停机")
row("amp", base_platform.get("amp_dtype") or "bf16", "bf16", "unchanged", "DCU 默认")
row("底座 checkpoint sha256", base.get("base_ckpt", {}).get("sha256", "")[:16] + "…", sha(base_ckpt)[:16] + "…", "unchanged", "全参续训初始化锚,双侧机读一致")
row("world_size(拓扑)", "4(T7 偏差 A 口径)", int(num_gpus), "unchanged", "T7 偏差 A(#254 会话决议)沿袭(#310 Implementation Decisions);对基底 run(world_size 7)的偏差已在 T7 登记,本次无新登记偏差")
row("dev 内嵌验证 reference bank", "reference_reinstr(同仪器重提)", "同 bank 预拷(eval_root 缓存命中)", "unchanged", "T5 与 T7/基底 dev FID trend 同栈同 bank 可比")
row("dev cohort sidecar spacing 值", "旧树 sidecar(raw pixdim 语义)", "新树 sidecar = T2 多源拷贝自旧树,逐位同值", "unchanged", "spacing 条件语义与 T7 消费值逐位同(#312 登记口径)")
row("labels 树", "不变(旧树)", "不变", "unchanged", "T1(#311)撤销:复核实证旧树已 RAS,重生成空转;且 P1 image-only 训练不消费 labels——本 run 与 T7 的 labels 输入逐位同")
row("种子纪律", "内嵌验证 per-(case,modality) sha256 合同种子", "同左", "unchanged", "零 GLOBAL_SEED 判定链接触;零 challenge_registry 诊断槽位消费;holdout 530 零接触")
row("冻结仪器/包络/判定线", "ADR-0002/0004 冻结", "零改动", "unchanged", "#310 Out of Scope")
row("P3 e39 checkpoint", "不动", "不动", "unchanged", f"#310 Out of Scope;launch 时观测在位 sha256={p3_sha[:12]}…")
# —— 恰一项改动:数据树替换(#310 序列③的方向修复项)——
row("embedding 数据树", "embeddings_cliptrue(T4 重编码;replay ~65% 轴序错乱,被 DM 静默吃进 T7)", "embeddings_ras(T2 #312 RAS 修复全量重编码;manifest 15868 对账全绿 + 逐条形状守卫全绿)", "protocol_change", "T5 唯一改动项:数据树替换。施用面:训练臂 7404 + replay 臂 7404 训练消费 + dev cohort spacing sidecar 源同树。世界差异(#310):T7 世界的 replay 轴序错乱在本世界构造性消除;BraTS 臂内容与 cliptrue 同构(轴序本对,x/y 对称侥幸免疫的历史如实登记);新树非 canonical 形状(如 (32,64,64,4))是 MR-RATE 几何多样性的真实反映,T3 形状契约接受")

checklist = {
    "schema": "p1-retrain-recipe-diff/2",
    "base_run_id": base_run_id,
    "recipe_carrier": {"run_root": t7_root, "issue": 254, "note": "T5 = T7 配方沿袭 + 数据树替换恰一项;diff 审计主语是 T5 vs T7"},
    "t5_run_root": str(Path(out).parent),
    "world_declaration": {
        "direction_world": "RAS 全链(#310 序列③);训练 embedding 消费 T2 修复重编码树",
        "vs_t7": "T7 训练世界的 replay 臂 ~65% embedding 轴序错乱(#310 方向审计)在本世界构造性消除;labels 输入逐位同(T1 撤销,#311);BraTS 臂 embedding 内容与 cliptrue 同构",
        "comparability": "dev FID trend 与 T7/基底同仪器(RadImageNet 同权重)同预处理栈同 bank(reference_reinstr)可比",
    },
    "protocol_changes": [r for r in rows if r["class"] == "protocol_change"],
    "approved_deltas": [r for r in rows if r["class"] == "approved_delta"],
    "unchanged": [r for r in rows if r["class"] == "unchanged"],
    "rows": rows,
}
Path(out).write_text(json.dumps(checklist, indent=2, ensure_ascii=False) + "\n")
n_change = len(checklist["protocol_changes"])
n_delta = len(checklist["approved_deltas"])
print(f"[checklist] 配方 diff:协议改动 {n_change} 项(须=1,数据树替换)、新登记偏差 {n_delta} 项(须=0)、不动 {len(checklist['unchanged'])} 项 -> {out}")
if n_change != 1 or n_delta != 0:
    raise SystemExit("[FATAL] 配方 diff 面不符预期(协议改动≠1 或新登记偏差≠0)——「干净方向世界重训=数据树替换恰一项」定位失效,不得开训")
PY
echo "[checklist] 核对表: $CHECKLIST"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 校验与核对表完成,未拉起。"
    exit 0
fi

# ── 拉起:单训练进程,dev 内嵌验证随训练走(#278/#279 现役形态;ET/WT 监控是后续票)──
# 内嵌验证 eval_root = <ckpt_dir>/dev_eval(装配钉死);reference bank 预拷至其下
# (同仪器重提产物,RealReferenceBank.build 缓存命中,不重提)。
mkdir -p "$T5_CKPT_DIR" "$T5_LOGS"
mkdir -p "$T5_CKPT_DIR/dev_eval/reference"
cp "$REF_BANK_SRC" "$T5_CKPT_DIR/dev_eval/reference/real_reference_bank.pt"
echo "[launch] reference bank 拷贝至 $T5_CKPT_DIR/dev_eval/reference/(内嵌验证缓存命中,同 dev list 同预处理)"

ENV_JSON="$T5_ROOT/environment_brats_p1_train_t5.json"
python3 - "$ENV_JSON" "$EMB_ROOT" "$TRAIN_LIST" "$T5_CKPT_DIR" "$VAE_PATH" "$BASE_CKPT" "$NV_CONFIGS" "$DATA_ROOT" <<'PY'
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
TRAIN_LOG="$T5_LOGS/train_$TS.log"

# 内嵌验证:dev cohort 16 例×4 模态=64 shard item 全卡分片;grid 5/epoch(基底网格);
# emb-root 指向新树(dev cohort spacing sidecar,T2 多源拷贝值与旧树逐位同);
# 早停规则装配钉死(ADR-0005: patience 3/min 30/max=trainer n_epochs)。
# 启动盘点期 DataCatalog 对 14808 条 embedding 逐条跑形状契约(header 读,NFS 上需数分钟),
# 违例即响亮死——错乱几何静默进训在启动前拦截(T3 守卫)。
setsid nohup python3 -m ctmr generate modality-label train \
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON" --replay-list "$REPLAY_LIST" -g "$NUM_GPUS" \
    --val-every 5 --dev-list "$DEV_LIST" --raw-root "$P1_ROOT/raw" --emb-root "$EMB_ROOT" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "[launch] 训练(torchrun world_size=$NUM_GPUS,内嵌 dev 验证 grid=5,数据树 embeddings_ras)pid=$TRAIN_PID log=$TRAIN_LOG"

echo "============================================"
echo "  T5 已拉起。监控:"
echo "    tail -f $TRAIN_LOG                              # 训练(loss + 内嵌验证里程碑)"
echo "    tail -f $T5_CKPT_DIR/dev_eval/dev_trend.jsonl   # dev FID trend(逐点 ledger)"
echo "  完成后(内嵌早停触发或 100 epochs):"
echo "    python3 -m ctmr generate modality-label dev-eval select \\"
echo "      --eval-root $T5_CKPT_DIR/dev_eval --ckpt-dir $T5_CKPT_DIR --out $T5_ROOT/selection.json"
echo "  (离线重打分备用: dev-eval watch 单遍对已存 checkpoint 重评,失败点重跑即重试,#279)"
echo "============================================"
