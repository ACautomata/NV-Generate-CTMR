#!/bin/bash
# 序列③ T6(issue #316,父 #310):P2 重训 run——干净 labels 条件,sugon DCU 训练作业
#
# 用途:P2 mask ControlNet 重训(序列③ E2):以 T5(#315)新 P1 候选为 DM 源、
# 干净方向世界数据为条件,冻结配方不动(ControlNet-only、DM/VAE 全冻、lr 1e-5、
# bs 1、≤100 epochs、早停、weighted_loss 100@129-131、RCL off、纯 BraTS 无回放);
# dev FID select 候选落盘。种子纪律与零 holdout 接触延续。
#
# 【相对历史 P2 run(p2-20260824T111958Z)的改动恰两项(协议改动,核对表 gate)】
#   ① DM 源替换:P1 epoch_20(9377f8ba…,方向污染世界)→ T5 候选 epoch_30
#      (p1-20260904T021136Z,干净方向世界 dev select 候选)——#310 序列③整改主项;
#   ② embedding 数据树替换:旧树 embeddings/(legacy,T2 记录其形状分布混有轴序
#      错乱工件)→ T2(#312)RAS 修复重编码树 embeddings_ras/(manifest 15868 行
#      对账全绿 + 逐条形状守卫全绿)——经 run 树 data_view/ 符号链接映射,
#      p2_mask_cond.json 的 `embeddings/` 前缀解析到新树,list 零改动(T2 记录
#      「list 前缀映射到新树根(或重生成 list)」的映射实现)。
#
# 【labels 树:不变(与历史 P2 逐位同,如实声明)】
#   票面(#316)「T1 干净 labels 树为掩码条件」的前提被 T1(#311)撤销改写:
#   三臂复核实证旧 labels 树已是 RAS(原链逐位重建 dice=1.0,P2/P3 训练世界
#   对位 AUC 5/5 as-is 最高),全量重生成空转。本 run 的 labels 输入与历史 P2
#   run 逐位同——「干净 labels 条件」在数据面即既有 RAS labels 树,如实登记。
#
# 【dev 监控形态(现役离线口径,#279,ADR-0019 §5)】
#   mask dev-eval watch 幂等循环(单遍扫 pending checkpoint 后退出,循环驱动),
#   16 例 dev cohort × 4 模态,cfg=10/30 步,per-(case,modality) 合同种子;
#   FID 面与历史 P2 trend 同仪器同预处理(RadImageNet + ras 1mm 预处理,
#   #314 未动 FID 面)可比;L2 trend + round-trip Dice 在 #314 RAS 仪器世界
#   (real 侧统一 RAS + flip 退役)——与历史 P2 的 L2 trend 读数不直接可比
#   (旧世界读数已被 T4 判读修订作废),如实登记。
#   早停规则 ADR-0005 钉值(patience 3 / min_epoch 30 / max=100)经 watch 落
#   <ckpt_dir>/.early_stop,trainer 每 epoch 边界轮询停机(PhaseHarness)。
#
# 【dm_source 溯源落盘(AC1)】
#   run 树 records/dm_source.json,schema brats-dm-source-trace/1:上游 = T5
#   候选(checkpoint sha256 实算 = WeightsRef 对账;md5 对 T5 记录钉值;选择
#   证据 = T5 selection.json 及其 sha256)。诊断声明:T5 候选 variant=
#   diagnostic 未过终验,本文件是 run 树训练溯源面,不是验收链 ledger register
#   (domain 规则只有 final-acceptance-passing P1 才可 register;registered
#   DM source 不动);本 run 不进契约链(与 T5/T7 重训形态一致)。
#
# 用法:
#   bash deploy/jobs/run_p2_retrain_t6.sh            # 前置校验 + 溯源/核对表落盘 + 拉起
#   bash deploy/jobs/run_p2_retrain_t6.sh --dry-run  # 只做校验+落盘,不拉起
#
# 环境变量(均可覆写,默认按 2026-09-05 实例实测持久盘布局):
#   T6_ROOT      T6 运行根(默认 /root/private_data/ctmr/runs/p2_t6;ckpt/dev_eval/records/logs 落此)
#   DEPLOY_ROOT  部署树(默认 /root/private_data/ctmr/deploy_t6;src+configs,持久盘防易失)
#   PHASE_ROOT   phase 数据根(默认 /root/private_data/ctmr/data/phase)
#   P1_T5_ROOT   T5(P1 重训)运行根(默认 /root/private_data/ctmr/runs/p1_t5;DM 源与选择证据)
#   P2_HIST_ROOT 历史 P2 运行根(默认 /root/private_data/ctmr/runs/p2;config 机读钉值源)
#   P1_BASE_ROOT 基底 P1 运行根(默认 /root/private_data/ctmr/runs/p1;base_ckpt 对账源)
#   MODELS_ROOT  模型根(默认 /root/private_data/ctmr/models;底座 DM + 冻结 VAE)
#   NUM_GPUS     训练 world_size(默认 4;T7 偏差 A 口径沿袭,历史 P2 为 7)
#
# 前置条件(脚本逐项校验,任一缺失即 FATAL 不拉起):
#   1. T5 候选 epoch_30 在位:md5 = b0dbb715abb35de3ba2c54c34abd9539(T5 记录
#      钉值)、sha256 实算入溯源;T5 selection.json 在位且 epoch==30、路径对账
#   2. 底座 DM sha256 = 历史 P2 dm_source.json base_ckpt 钉值(90c4a015…);
#      冻结 VAE md5 = 917cfb1e49631c8a713e3bb7c758fbca(冻结 canonical,只读)
#   3. config 面零 delta 机读对账:config_brats_p2_train.json 与
#      config_network_rflow.json 的 sha256 对历史 P2 run.json configs 钉值逐位
#      一致;train config 超参逐键断言(lr 1e-5/bs 1/100/wl 100@[129,130,131]/
#      RCL off/fold 0/cfg 10/30 步);p2_mask_cond.json sha256 对历史 run.json
#      data_lists 钉值
#   4. embeddings_ras(T2 新树)覆盖 p2_mask_cond 全部 image 引用 8464 唯一路径
#      (embedding+sidecar,线程池),manifest.jsonl = 15868 行(T2 退出条件锚);
#      labels 树覆盖全部 label 引用;raw_relinked 覆盖 dev 臂 1060 raw(watch bank)
#   5. DCU 卡数 = NUM_GPUS;rendezvous 29500 端口空闲(T7 先例)
#   6. #310 Out of Scope:P3 e39 checkpoint 在位观测(不动)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PROJECT_ROOT/src" ] || { echo "[FATAL] $PROJECT_ROOT/src missing — run from the repo checkout" >&2; exit 1; }

T6_ROOT="${T6_ROOT:-/root/private_data/ctmr/runs/p2_t6}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/root/private_data/ctmr/deploy_t6}"
PHASE_ROOT="${PHASE_ROOT:-/root/private_data/ctmr/data/phase}"
P1_T5_ROOT="${P1_T5_ROOT:-/root/private_data/ctmr/runs/p1_t5}"
P2_HIST_ROOT="${P2_HIST_ROOT:-/root/private_data/ctmr/runs/p2}"
P1_BASE_ROOT="${P1_BASE_ROOT:-/root/private_data/ctmr/runs/p1}"
MODELS_ROOT="${MODELS_ROOT:-/root/private_data/ctmr/models}"
NUM_GPUS="${NUM_GPUS:-4}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

TRAIN_LIST="$PHASE_ROOT/lists/p2_mask_cond.json"
MODEL_JSON="$DEPLOY_ROOT/configs/config_brats_p2_train.json"
NET_JSON="$DEPLOY_ROOT/configs/config_network_rflow.json"
EMB_RAS="$PHASE_ROOT/embeddings_ras"               # T2(#312)RAS 修复重编码新树——本 run 唯一数据树替换项
RAW_ROOT="$PHASE_ROOT/raw_relinked"                # watch bank 的 dev raw 源(T2 补链后覆盖 dev 1060)
DM_SRC_CKPT="$P1_T5_ROOT/ckpt/epoch_30.pt"         # ①DM 源替换:T5 候选
T5_SELECTION="$P1_T5_ROOT/selection.json"          # T5 候选选择证据(溯源 run_record 面)
BASE_CKPT="$MODELS_ROOT/diff_unet_3d_rflow-mr-brain_v1.pt"
VAE_PATH="$MODELS_ROOT/autoencoder_v1.pt"
HIST_RUN_JSON="$P2_HIST_ROOT/records/runs/p2-20260824T111958Z/run.json"
HIST_DM_SOURCE="$P2_HIST_ROOT/records/dm_source.json"
HIST_P1_DM_SHA="9377f8ba2c9d0ad4d2aaaac8874629dd1ba51f5e237231ea9ebed7bfde12946b"  # 历史 P2 的 DM 源(P1 e20)——被本 run 替换的参照锚
VAE_MD5_PIN="917cfb1e49631c8a713e3bb7c758fbca"
DM_MD5_PIN="b0dbb715abb35de3ba2c54c34abd9539"      # T5 记录钉值(epoch_30 md5)
EMB_MANIFEST_ROWS_PIN=15868                         # T2 退出条件的 manifest 锚(#312 记录)
NNUNET_RESULTS="${NNUNET_RESULTS:-/root/private_data/ctmr/instruments/nnunet_results}"
NNUNET_RAW="${NNUNET_RAW:-/root/private_data/ctmr/data/nnunet_raw}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-/root/private_data/ctmr/data/nnunet_preprocessed}"
P3_E39_CKPT="${P3_E39_CKPT:-/root/private_data/ctmr/runs/p3/ckpt/epoch_39.pt}"
T6_CKPT_DIR="$T6_ROOT/ckpt"
T6_LOGS="$T6_ROOT/logs"

# ── 双 source(DTK 算 + 平台代理网;非交互 ssh 需显式)──
# env.sh 引用未初始化变量(如 CMAKE_PREFIX_PATH),set -u 下展开级致命(|| true 拦不住),临时关 -u
set +u
source /opt/dtk/env.sh 2>/dev/null || true
set -u
eval "$(grep -E '^export (HF_TOKEN|HF_ENDPOINT)=' ~/.bashrc 2>/dev/null)" || true

echo "============================================"
echo "序列③ T6:P2 重训(干净 labels 条件,DM 源 + embedding 树替换恰两项)(#316,父 #310)"
echo "配方基底:历史 P2 run(p2-20260824T111958Z,#59)| DM 源:T5 候选 epoch_30(p1-20260904T021136Z)"
echo "数据树:$EMB_RAS(T2 #312 RAS 修复重编码,经 data_view 符号链接映射)| world_size $NUM_GPUS(偏差 A 口径沿袭)"
echo "T6_ROOT=$T6_ROOT"
echo "============================================"

# ── 前置 1-4:DM 源对账、底座/VAE 锚、config 钉值、数据覆盖 ──
for f in "$DM_SRC_CKPT" "$T5_SELECTION" "$BASE_CKPT" "$VAE_PATH" "$HIST_RUN_JSON" "$HIST_DM_SOURCE" "$TRAIN_LIST" "$MODEL_JSON" "$NET_JSON"; do
    [ -f "$f" ] || { echo "[FATAL] 前置文件不存在: $f" >&2; exit 1; }
done

python3 - "$DM_SRC_CKPT" "$DM_MD5_PIN" "$T5_SELECTION" "$BASE_CKPT" "$VAE_PATH" "$VAE_MD5_PIN" \
    "$HIST_RUN_JSON" "$HIST_DM_SOURCE" "$HIST_P1_DM_SHA" "$MODEL_JSON" "$NET_JSON" "$TRAIN_LIST" \
    "$EMB_RAS" "$RAW_ROOT" "$EMB_MANIFEST_ROWS_PIN" "$P3_E39_CKPT" "$T6_ROOT" "$PHASE_ROOT" \
    "$P1_BASE_ROOT/ckpt/epoch_20.pt" <<'PY'
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

(dm_ckpt, dm_md5_pin, t5_selection, base_ckpt, vae, vae_md5_pin, hist_run_json, hist_dm_source,
 hist_p1_dm_sha, model_json, net_json, train_list, emb_ras, raw_root, manifest_rows_pin,
 p3_e39, t6_root, phase_root, hist_p1_ckpt) = sys.argv[1:20]
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

# ① DM 源:T5 候选 md5 对 T5 记录钉值,sha256 实算(WeightsRef 对账,写溯源)
dm_md5 = md5_of(dm_ckpt)
if dm_md5 != dm_md5_pin:
    raise SystemExit(f"[FATAL] T5 候选 md5 不匹配: {dm_md5} vs T5 记录钉值 {dm_md5_pin}")
dm_sha = sha256_of(dm_ckpt)
print(f"[preflight] DM 源(T5 候选)md5 对账一致: {dm_md5};sha256 实算: {dm_sha}")
sel = json.loads(Path(t5_selection).read_text())
if sel.get("epoch") != 30 or sel.get("checkpoint") != dm_ckpt:
    raise SystemExit(f"[FATAL] T5 selection.json 对账失败: epoch={sel.get('epoch')} checkpoint={sel.get('checkpoint')}")
print(f"[preflight] T5 selection.json 对账一致: epoch 30 argmin mean_fid={sel.get('mean_fid'):.4f}(选择证据 sha256 待溯源步实算)")

# 底座锚 + VAE(冻结只读)
hist_dm = json.loads(Path(hist_dm_source).read_text())
base_pin = hist_dm.get("base_ckpt", {}).get("sha256", "")
actual = sha256_of(base_ckpt)
if actual != base_pin:
    raise SystemExit(f"[FATAL] 底座 sha256 不匹配: 实算 {actual} vs 历史 dm_source 钉值 {base_pin}")
print(f"[preflight] 底座 sha256 对账一致: {actual[:12]}…")
# 被替换 DM 源的三方对账(协议改动①的参照锚):账本字段 == 常量钉值 == 历史 P1 checkpoint 实算
if hist_dm.get("checkpoint", {}).get("sha256") != hist_p1_dm_sha:
    raise SystemExit(f"[FATAL] 历史 dm_source checkpoint sha256 与被替换源钉值不符: {hist_dm.get('checkpoint', {}).get('sha256')}")
if not Path(hist_p1_ckpt).is_file():
    raise SystemExit(f"[FATAL] 被替换 DM 源文件不存在: {hist_p1_ckpt}(superseded 锚须实算对账)")
hist_p1_actual = sha256_of(hist_p1_ckpt)
if hist_p1_actual != hist_p1_dm_sha:
    raise SystemExit(f"[FATAL] 被替换 DM 源实算不符: {hist_p1_actual} vs {hist_p1_dm_sha}")
print(f"[preflight] 被替换 DM 源(历史 P1 e20)三方对账一致: {hist_p1_actual[:12]}…(账本字段=钉值=文件实算)")
vae_md5 = md5_of(vae)
if vae_md5 != vae_md5_pin:
    raise SystemExit(f"[FATAL] VAE md5 不匹配: {vae_md5} vs {vae_md5_pin}")
print(f"[preflight] VAE md5 对账一致: {vae_md5}")

# config 面零 delta 机读对账(对历史 P2 run.json configs 钉值)
hist = json.loads(Path(hist_run_json).read_text())
pins = {c.get("role"): c.get("sha256") for c in hist.get("configs", [])}
if sha256_of(model_json) != pins.get("train"):
    raise SystemExit(f"[FATAL] train config sha256 与历史 P2 run.json 钉值不符(config 面非零 delta)")
if sha256_of(net_json) != pins.get("network"):
    raise SystemExit(f"[FATAL] network config sha256 与历史 P2 run.json 钉值不符(config 面非零 delta)")
cfg = json.loads(Path(model_json).read_text())
train_cfg = cfg.get("controlnet_train", {})
for key, pin in (
    ("lr", 1e-05), ("batch_size", 1), ("n_epochs", 100),
    ("weighted_loss", 100), ("weighted_loss_label", [129, 130, 131]),
    ("use_region_contrasive_loss", False), ("fold", 0), ("cache_rate", 0),
):
    if train_cfg.get(key) != pin:
        raise SystemExit(f"[FATAL] 超参 {key} = {train_cfg.get(key)} != 历史 P2 真值 {pin}(配方冻结不动)")
infer_cfg = cfg.get("diffusion_unet_inference", {})
if infer_cfg.get("cfg_guidance_scale") != 10 or infer_cfg.get("num_inference_steps") != 30:
    raise SystemExit(f"[FATAL] dev 采样配方须 cfg=10/30 步(不动),got {infer_cfg}")
hist_list_sha = next((d.get("sha256") for d in hist.get("data_lists", []) if d.get("side") == "train"), "")
if sha256_of(train_list) != hist_list_sha:
    raise SystemExit(f"[FATAL] p2_mask_cond.json sha256 与历史 run.json data_lists 钉值不符(list 被动过?)")
print(f"[preflight] config 面:train/network/list sha256 对历史 P2 run.json 机读钉值逐位一致;超参逐键一致(冻结)")

# 数据覆盖:embedding(新树,剥 embeddings/ 前缀)、labels、dev raw(线程池逐条存在性)
entries = json.loads(Path(train_list).read_text())["training"]
folds = {}
for entry in entries:
    folds[entry["fold"]] = folds.get(entry["fold"], 0) + 1
if len(entries) != 8464 or folds != {1: 7404, 0: 1060}:
    raise SystemExit(f"[FATAL] p2_mask_cond 条数/折分不符: {len(entries)} {folds} != 8464 {{1:7404, 0:1060}}")
emb_root = Path(emb_ras)
manifest = emb_root / "manifest.jsonl"
if not manifest.is_file():
    raise SystemExit(f"[FATAL] T2 manifest 不存在: {manifest}(embeddings_ras 非完整重编码树,不得开训)")
n_rows = sum(1 for _ in open(manifest))
if n_rows != manifest_rows_pin:
    raise SystemExit(f"[FATAL] T2 manifest {n_rows} 行 != {manifest_rows_pin}(#312 退出条件锚,不得开训)")
print(f"[preflight] list 8464 条(fold1 训练 7404 / fold0 dev 1060);T2 manifest {n_rows} 行对账一致")

emb_paths, label_paths, dev_raw_paths = [], [], []
for entry in entries:
    parts = entry["image"].split("/")
    if not parts or parts[0] != "embeddings":
        raise SystemExit(f"[FATAL] image 路径前缀异常(须 embeddings/): {entry['image']}")
    emb_paths.append(str(emb_root / "/".join(parts[1:])))
    emb_paths.append(emb_paths[-1] + ".json")
    # labels 直查真身根(data_view 符号链接在本脚本后段才构造;preflight 只对数据源)
    label_paths.append(str(Path(phase_root) / entry["label"]))
    if entry["fold"] == 0:
        raw_rel = entry["image"][: -len("_emb.nii.gz")] + ".nii.gz"
        dev_raw_paths.append(str(Path(raw_root) / raw_rel.replace("embeddings/", "", 1)))

def all_exist(paths):
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [p for p, ok in zip(paths, pool.map(lambda q: Path(q).is_file(), paths)) if not ok]

missing = all_exist(emb_paths)
if missing:
    raise SystemExit(f"[FATAL] embeddings_ras 缺 {len(missing)} 条(首3: {missing[:3]})——T2 重编码未覆盖,不得开训")
print(f"[preflight] embeddings_ras 覆盖 image 引用 {len(entries)} 唯一路径(embedding+sidecar 全在)")
missing = all_exist(label_paths)
if missing:
    raise SystemExit(f"[FATAL] labels 树缺 {len(missing)} 条(首3: {missing[:3]})——掩码条件不完整,不得开训")
print(f"[preflight] labels 树覆盖 label 引用 {len(entries)} 条(combined 全在;T1 #311 撤销:树与历史 P2 逐位同)")
missing = all_exist(dev_raw_paths)
if missing:
    raise SystemExit(f"[FATAL] raw_relinked 缺 dev raw {len(missing)} 条(首3: {missing[:3]})——watch bank 数据源不完整")
print(f"[preflight] raw_relinked 覆盖 dev 臂 {len(dev_raw_paths)} raw(watch bank 数据源)")

# AC4 观测:P3 e39 在位(不动它,只登记)
p3 = Path(p3_e39)
if not p3.is_file():
    raise SystemExit(f"[FATAL] P3 e39 checkpoint 不存在(观测对象缺失): {p3}")
print(f"[preflight] P3 e39 在位: sha256={sha256_of(p3)[:12]}…(观测,不动)")

Path(t6_root).mkdir(parents=True, exist_ok=True)
Path(t6_root, "records").mkdir(parents=True, exist_ok=True)
print("[preflight] 全部通过")
PY

# ── 前置 5-6:GPU 数、rendezvous 端口 ──
NGPU="$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
[ "$NGPU" = "$NUM_GPUS" ] || { echo "[FATAL] DCU 卡数 $NGPU != NUM_GPUS $NUM_GPUS(偏差 A 拓扑须与拉起一致)" >&2; exit 1; }
if (exec 3<>/dev/tcp/127.0.0.1/29500) 2>/dev/null; then
    exec 3<&- 3>&-
    echo "[FATAL] rendezvous 端口 29500 已被占用(T7 先例:旧 worker 残留占端口致拉起失败)——清场后重跑" >&2
    exit 1
fi
echo "[preflight] DCU×$NGPU / rendezvous 29500 空闲就位"

# ── dm_source 溯源落盘(AC1:schema brats-dm-source-trace/1,WeightsRef 对账)──
DM_SOURCE_JSON="$T6_ROOT/records/dm_source.json"
python3 - "$DM_SOURCE_JSON" "$DM_SRC_CKPT" "$T5_SELECTION" "$BASE_CKPT" "$HIST_DM_SOURCE" "$MODEL_JSON" "$NET_JSON" "$T6_ROOT" <<'PY'
import hashlib, json, sys
from datetime import UTC, datetime
from pathlib import Path

out, dm_ckpt, t5_selection, base_ckpt, hist_dm_source, model_json, net_json, t6_root = sys.argv[1:9]

def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hist_dm = json.loads(Path(hist_dm_source).read_text())
entry = {
    "schema": "brats-dm-source-trace/1",
    "issue": 316,
    "parent_issue": 310,
    "diagnostic_declaration": (
        "本文件是 P2 重训 run(#316)的训练溯源面,不是验收链 DM-source ledger register:上游 T5 候选 "
        "variant=diagnostic 未过终验,domain 规则(只有 final-acceptance-passing P1 才可 register)不被触发,"
        "registered DM source(历史 P1 epoch_20 账本)零改动。本 run 不进契约链,与 T5(#315)/T7(#254)重训形态一致;"
        "上一轮正式 P2 bypass(p2-20260824T111958Z)钉的是被替换的旧世界 DM 源,按 #310 全链重跑口径一并退役。"
    ),
    "upstream_run_id": "p1-20260904T021136Z",
    "upstream_run_record": str(Path(t5_selection).resolve()),
    "run_record_sha256": sha256_of(t5_selection),
    "run_record_note": "T5 为诊断 run 树形态(无契约 run.json);候选选择证据 = dev-eval select 的 selection.json(argmin mean dev FID,预记录规则)",
    "checkpoint": {
        "epoch": 30,
        "path": str(Path(dm_ckpt).resolve()),
        "sha256": sha256_of(dm_ckpt),
        "md5": hashlib.md5(Path(dm_ckpt).read_bytes()).hexdigest(),
    },
    "md5_note": "md5 与 T5 实验记录候选钉值逐位一致(b0dbb715…,20260904 记录 §3);sha256 为 WeightsRef 对账主键(实算)",
    "base_ckpt": {"path": str(Path(base_ckpt).resolve()), "sha256": hist_dm.get("base_ckpt", {}).get("sha256")},
    "configs": [
        {"path": str(Path(model_json).resolve()), "role": "train", "sha256": sha256_of(model_json)},
        {"path": str(Path(net_json).resolve()), "role": "network", "sha256": sha256_of(net_json)},
    ],
    "superseded_source": {
        "run_id": hist_dm.get("run_id"),
        "checkpoint_sha256": hist_dm.get("checkpoint", {}).get("sha256"),
        "note": "历史 P2(p2-20260824T111958Z)的 DM 源(P1 e20,方向污染世界)——本 run 的替换参照锚,非 ledger supersede",
    },
    "registered_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(out).write_text(json.dumps(entry, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
print(f"[dm_source] 溯源落盘: {out}(schema brats-dm-source-trace/1;upstream sha256={entry['checkpoint']['sha256'][:12]}… run_record sha256={entry['run_record_sha256'][:12]}…)")
PY

# ── 配方 diff 核对表(超参逐项与历史 P2 一致;协议改动恰两项:DM 源 + 数据树;零新登记偏差)──
CHECKLIST="$T6_ROOT/recipe_diff_checklist.json"
python3 - "$HIST_RUN_JSON" "$HIST_DM_SOURCE" "$MODEL_JSON" "$NET_JSON" "$CHECKLIST" "$NUM_GPUS" "$EMB_RAS" "$DM_SRC_CKPT" "$T6_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path

hist_run_json, hist_dm_source, model_json, net_json, out, num_gpus, emb_ras, dm_ckpt, t6_root = sys.argv[1:10]

hist = json.loads(Path(hist_run_json).read_text())
hist_dm = json.loads(Path(hist_dm_source).read_text())
cfg = json.loads(Path(model_json).read_text())
train_cfg, infer_cfg = cfg["controlnet_train"], cfg["diffusion_unet_inference"]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

rows = []
def row(item, base_v, t6_v, klass, note=""):
    rows.append({"item": item, "base": base_v, "t6": t6_v, "class": klass, "note": note})

platform = hist.get("platform", {})
# —— 不动项(逐项对账;base 侧 = 历史 P2 run.json 机读钉值或其 config 真值)——
row("lr", 1e-05, train_cfg.get("lr"), "unchanged", "preflight 已对历史 run.json config 钉值强校验")
row("batch_size", 1, train_cfg.get("batch_size"), "unchanged", "preflight 已强校验")
row("n_epochs(上限)", 100, train_cfg.get("n_epochs"), "unchanged", "早停经 dev-eval watch 落 .early_stop,trainer 轮询停机(ADR-0005 钉值)")
row("cache_rate", 0, train_cfg.get("cache_rate"), "unchanged", "preflight 已强校验")
row("weighted_loss", 100, train_cfg.get("weighted_loss"), "unchanged", "肿瘤加权(P3 加权 loss 同族 129-131)")
row("weighted_loss_label", [129, 130, 131], train_cfg.get("weighted_loss_label"), "unchanged", "")
row("use_region_contrasive_loss", False, train_cfg.get("use_region_contrasive_loss"), "unchanged", "RCL off(历史 P2 配方守卫同值)")
row("fold", 0, train_cfg.get("fold"), "unchanged", "train=fold!=0 7404;dev=fold=0 1060(BypassTrainLoader 折分)")
row("RF scheduler.sample_method", "uniform", "uniform", "unchanged", "network config 与历史 P2 逐位同(sha256 机读一致)")
row("RF scheduler.scale", 1.4, 1.4, "unchanged", "同上")
row("network config sha256", "aea761fe7a915bb8…(历史 run.json 钉值)", sha(net_json)[:16] + "…", "unchanged", "双侧机读:历史 run.json 钉值 vs 当前文件,逐位一致")
row("train config sha256", "d80ae9d6bb6a7385…(历史 run.json 钉值)", sha(model_json)[:16] + "…", "unchanged", "config 面相对历史 P2 零 delta(preflight 机读对账)")
row("p2_mask_cond.json sha256", "93a4914b058dde23…(历史 run.json 钉值)", "逐位一致(preflight 对账)", "unchanged", "list 零改动;embeddings/ 前缀经 data_view 符号链接映射到新树")
row("PolynomialLR power", 2.0, 2.0, "unchanged", "代码常量(BypassMounting)")
row("optimizer", "AdamW", "AdamW", "unchanged", "代码常量")
row("loss", "L1(肿瘤加权)", "L1(肿瘤加权)", "unchanged", "代码常量")
row("replay 混合", "无(纯 BraTS)", "无(纯 BraTS)", "unchanged", "DataCatalog no-replay;mask 链无 replay 面")
row("dev 内嵌/侧车采样 cfg_guidance_scale", 10, infer_cfg.get("cfg_guidance_scale"), "unchanged", "预录采样配方")
row("dev 内嵌/侧车采样 num_inference_steps", 30, infer_cfg.get("num_inference_steps"), "unchanged", "")
row("dev cohort", "16 例(dev_cohort quotas)", "16 例(同 builder 同 list)", "unchanged", "DevCohortBuilder 对同一 p2_mask_cond fold=0 确定性挑选")
row("早停规则", "patience 3 / min_epoch 30 / max=100", "同钉(watch --patience 3 --min-epoch 30 --max-epoch 100)", "unchanged", "ADR-0005 钉值;历史 P2 31 epochs 实训(best e15)")
row("ControlNet 初始化", "冻结 DM encoder/mid copy_model_state", "同左(对 T5 候选 DM)", "unchanged", "BypassMounting 单序列;NEVER warm-start from ControlNet ckpt")
row("DM/VAE 冻结", "全冻(ControlNet-only)", "全冻", "unchanged", "VAE md5 钉值 preflight 对账;训练对象仅 ControlNet")
row("amp", "bf16", "bf16", "unchanged", "DCU 默认")
row("modality_mapping", "mri_t1_skull_stripped=29 等", "同一文件(configs/modality_mapping.json)", "unchanged", "")
row("world_size(拓扑)", "4(T7 偏差 A 口径沿袭;历史 P2 run 实测 7)", int(num_gpus), "unchanged", "本序列已登记拓扑口径(#310 Implementation Decisions「world_size=4 沿 T7 偏差 A 登记口径」);对历史 P2(7)的偏差即该已登记口径本身,非新登记偏差")
row("labels 树", "labels/(RAS 实证树)", "不变(逐位同)", "unchanged", "T1(#311)撤销:三臂复核实证旧树已 RAS、P2 训练世界对位 as-is 最高;票面「T1 干净 labels 树」前提被 #311 改写,本 run labels 输入与历史 P2 逐位同,如实声明")
row("dev FID 仪器", "2.5D RadImageNet(ras 1mm 预处理)", "同仪器(#314 未动 FID 面)", "unchanged", "T6 trend 与历史 P2 trend 同仪器同预处理可比(bank 按 run 树新建,同 dev list 同预处理确定性等价)")
row("种子纪律", "dev 采样 per-(case,modality) 合同种子", "同左", "unchanged", "零 GLOBAL_SEED 判定链接触;零 challenge_registry 诊断槽位消费;holdout 530 零接触")
row("冻结仪器/包络/判定线", "ADR-0002/0004 冻结", "零改动", "unchanged", "#310 Out of Scope(仪器 v2 校准另票)")
row("P3 e39 checkpoint", "不动", "不动", "unchanged", "#310 Out of Scope;launch 时观测在位")
row("holdout 生成", "本票不生成", "不生成", "unchanged", "dev-only 监控延续;holdout 属验收票面")
# —— 协议改动恰两项(#310 序列③的方向修复项)——
row("① DM 源(trained_diffusion_path)", "P1 epoch_20(sha256 9377f8ba…,p1-20260822T131947Z;方向污染世界基底候选)", "T5 候选 epoch_30(sha256 实算入 dm_source;p1-20260904T021136Z;干净方向世界 dev select 候选 m=3.7957)", "protocol_change", "T6 协议改动①:DM 源替换。#310 序列③ E2 主项——P2 掩码条件学习挂到干净方向世界的 DM 上;T5 候选 diagnostic 未过终验,溯源面见 records/dm_source.json(brats-dm-source-trace/1)")
row("② embedding 数据树", "embeddings/(legacy 树;T2 实测其分布混有轴序错乱工件,(32,64,64,4)×2953 等非规范形状)", "embeddings_ras(T2 #312 RAS 修复全量重编码;manifest 15868 对账全绿 + 逐条形状守卫全绿;经 data_view 符号链接映射,list 零改动)", "protocol_change", "T6 协议改动②:数据树替换。施用面:p2_mask_cond 全部 image 引用 8464 唯一路径(训练臂 7404 + dev 臂 1060)。BraTS 臂内容与旧树同构(轴序本对,x/y 对称侥幸免疫,#310 历史如实登记);新树非 canonical 形状是 MR-RATE 几何多样性的真实反映(T3 形状契约接受;P2 训练臂为 BraTS,canonical (64,64,32,4) 为主)")

checklist = {
    "schema": "p2-retrain-recipe-diff/1",
    "base_run_id": "p2-20260824T111958Z",
    "recipe_carrier": {"issue": 59, "note": "P2 无中间配方载体:T6 = 历史 P2 run 配方沿袭 + DM 源与数据树替换恰两项;diff 审计主语是 T6 vs 历史 P2 run"},
    "t6_run_root": str(Path(t6_root)),
    "world_declaration": {
        "direction_world": "RAS 全链(#310 序列③;ADR-0020):DM 源 = T5 干净世界候选,embedding = T2 RAS 修复树,labels = RAS 实证树(不变),仪器链 = #314 RAS 统一后形态",
        "vs_hist_p2": "历史 P2 世界的 DM 源(基底 P1 e20,方向污染世界训练)与 legacy embedding 树在本世界均被替换;labels 与 FID 仪器面不变",
        "comparability": "dev FID trend 与历史 P2 同仪器(RadImageNet 同权重)同预处理栈可比(bank 按 run 新建,同 dev list 同预处理确定性等价);L2 trend/round-trip 面在 #314 RAS 仪器世界,与历史 P2 读数不直接可比(T4 判读修订)",
    },
    "protocol_changes": [r for r in rows if r["class"] == "protocol_change"],
    "approved_deltas": [r for r in rows if r["class"] == "approved_delta"],
    "unchanged": [r for r in rows if r["class"] == "unchanged"],
    "rows": rows,
}
Path(out).write_text(json.dumps(checklist, indent=2, ensure_ascii=False) + "\n")
n_change = len(checklist["protocol_changes"])
n_delta = len(checklist["approved_deltas"])
print(f"[checklist] 配方 diff:协议改动 {n_change} 项(须=2,DM 源 + 数据树)、新登记偏差 {n_delta} 项(须=0)、不动 {len(checklist['unchanged'])} 项 -> {out}")
if n_change != 2 or n_delta != 0:
    raise SystemExit("[FATAL] 配方 diff 面不符预期(协议改动≠2 或新登记偏差≠0)——「T6=历史配方 + DM 源与数据树替换恰两项」定位失效,不得开训")
PY
echo "[checklist] 核对表: $CHECKLIST"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] 校验、溯源与核对表完成,未拉起。"
    exit 0
fi

# ── data_view:embeddings/ 前缀符号链接映射到 embeddings_ras(list 零改动;T2 记录预留方案)──
VIEW_ROOT="$T6_ROOT/data_view"
mkdir -p "$VIEW_ROOT"
ln -sfn "$EMB_RAS" "$VIEW_ROOT/embeddings"
ln -sfn "$PHASE_ROOT/labels" "$VIEW_ROOT/labels"
[ -e "$VIEW_ROOT/embeddings/ASNR-MICCAI-BraTS2023" ] || { echo "[FATAL] data_view/embeddings 解析失败(符号链接断裂)" >&2; exit 1; }
[ -e "$VIEW_ROOT/labels/GLI" ] || { echo "[FATAL] data_view/labels 解析失败(符号链接断裂)" >&2; exit 1; }
echo "[data_view] 符号链接映射就绪: $VIEW_ROOT/embeddings -> $EMB_RAS; $VIEW_ROOT/labels -> $PHASE_ROOT/labels"

# ── 拉起:训练 + dev watch 幂等循环(离线形态,#279;早停经 .early_stop 回传 trainer)──
mkdir -p "$T6_CKPT_DIR" "$T6_LOGS"

ENV_JSON="$T6_ROOT/environment_brats_p2_train_t6.json"
python3 - "$ENV_JSON" "$VIEW_ROOT" "$TRAIN_LIST" "$T6_CKPT_DIR" "$VAE_PATH" "$DM_SRC_CKPT" "$DEPLOY_ROOT" <<'PY'
import json, sys
dst, view, lst, ckpt, vae, dm_src, deploy_root = sys.argv[1:8]
env = {
    "model_dir": ckpt,
    "tfevent_path": str(__import__("pathlib").Path(ckpt).parent / "tblogs"),
    "trained_autoencoder_path": vae,
    "trained_diffusion_path": dm_src,
    "exp_name": "p2_mask_cond_t6",
    "data_base_dir": view,
    "json_data_list": lst,
    "modality_mapping_path": f"{deploy_root}/configs/modality_mapping.json",
}
json.dump(env, open(dst, "w"), indent=4)
print(f"[launch] env json 落盘: {dst}(data_base_dir={view};trained_diffusion_path={dm_src})")
PY

TS="$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="$T6_LOGS/train_$TS.log"
RUN_ID="p2-${TS}"

setsid nohup python3 -m ctmr generate mask train \
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON" -g "$NUM_GPUS" \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "[launch] P2 训练(torchrun world_size=$NUM_GPUS,DM 源=T5 epoch_30,数据树=embeddings_ras view)pid=$TRAIN_PID log=$TRAIN_LOG"

# watch 循环:等首个 eval 点(e5)落盘后周期跑幂等 watch;早停/训练退出即收尾 select。
# 循环自持(setsid nohup),与本脚本生命周期无关;状态经 dev_trend.jsonl / .early_stop 可观测。
WATCH_LOOP="$T6_ROOT/watch_loop.sh"
cat > "$WATCH_LOOP" <<EOF
#!/bin/bash
# 序列③ T6(#316)dev watch 幂等循环:单遍 watch 扫 pending checkpoint 后退出,本循环驱动至终态。
set -u
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
set +u
source /opt/dtk/env.sh 2>/dev/null || true
set -u
CK="$T6_CKPT_DIR"
EVAL_ROOT="$T6_ROOT/dev_eval"
TS="\$(date -u +%Y%m%dT%H%M%SZ)"
WATCH_LOG="$T6_LOGS/watch_\$TS.log"
SELECT_LOG="$T6_LOGS/select_\$TS.log"

echo "[watch-loop] 等 epoch_5.pt(首个 eval 点)…"
until [ -f "\$CK/epoch_5.pt" ]; do sleep 60; done
echo "[watch-loop] epoch_5 就绪,开始 watch 循环(间隔 20 min;幂等,ledger 已有点跳过)"
# watch argv 单点定义:循环遍与收尾扫描共用同一套参数(改 patience/eval-every 只动这里)
WATCH_ARGS=(python3 -m ctmr generate mask dev-eval watch
    --ckpt-dir "\$CK" --eval-root "\$EVAL_ROOT"
    --dev-list "$TRAIN_LIST" --raw-root "$RAW_ROOT" --label-root "$VIEW_ROOT"
    -e "$ENV_JSON" -c "$MODEL_JSON" -t "$NET_JSON"
    --eval-every 5 --patience 3 --min-epoch 30 --max-epoch 100
    --instrument-results "GLI=$NNUNET_RESULTS" --instrument-results "MEN=$NNUNET_RESULTS"
    --instrument-results "METS=$NNUNET_RESULTS" --instrument-results "PED=$NNUNET_RESULTS"
    --instrument-results "SSA=$NNUNET_RESULTS"
    --nnunet-raw "$NNUNET_RAW" --nnunet-preprocessed "$NNUNET_PREPROCESSED"
    # 仪器 v2 换树后冻结 spec 锚待校准重跑同批重钉(#310)——监控面按树实况覆写
    # -tr/-p/-c(etwt 同族机制;恰一 trainer 目录三段式,不猜;冻结锚零改动)
    --instrument-specs-autodiscover)
while true; do
    "\${WATCH_ARGS[@]}" >> "\$WATCH_LOG" 2>&1
    if [ -f "\$CK/.early_stop" ]; then
        echo "[watch-loop] 早停落盘: \$(tr '\n' ' ' < "\$CK/.early_stop")"
        break
    fi
    # 存活判据:launcher(python3 -m ctmr generate mask train)或 worker(-m
    # ctmr.application.generation.mask.train)任一在。字符类 [k] 打破 pgrep
    # 自匹配(T5 执行登记先例:pattern 字面出现在本循环自身命令行会误报 still running)。
    if ! pgrep -f "generate mas[k] train" > /dev/null && ! pgrep -f "generation\.mas[k]\.train" > /dev/null; then
        # 训练进程已退(早停文件缺 = 崩溃面或 100 上限跑满):再跑一遍收尾 pending 后停
        echo "[watch-loop] 训练进程已退出,收尾一遍 watch"
        "\${WATCH_ARGS[@]}" >> "\$WATCH_LOG" 2>&1
        break
    fi
    sleep 1200
done

python3 -m ctmr generate mask dev-eval select \\
    --eval-root "\$EVAL_ROOT" --ckpt-dir "\$CK" --out "$T6_ROOT/selection.json" \\
    > "\$SELECT_LOG" 2>&1
echo "[watch-loop] select 落盘: $T6_ROOT/selection.json(循环结束)"
EOF
chmod +x "$WATCH_LOOP"
setsid nohup bash "$WATCH_LOOP" > "$T6_LOGS/watch_loop_$TS.log" 2>&1 < /dev/null &
echo "[launch] watch 幂等循环 pid=$! loop=$WATCH_LOOP log=$T6_LOGS/watch_loop_$TS.log"

echo "============================================"
echo "  T6 已拉起(run_id 构造 $RUN_ID,UTC 启动戳)。监控:"
echo "    tail -f $TRAIN_LOG                              # 训练(loss)"
echo "    tail -f $T6_ROOT/dev_eval/dev_trend.jsonl       # dev FID trend(逐点 ledger)"
echo "    tail -f $T6_LOGS/watch_loop_$TS.log             # watch 循环状态"
echo "  完成后(watch 循环自动 select):"
echo "    cat $T6_ROOT/selection.json"
echo "============================================"
