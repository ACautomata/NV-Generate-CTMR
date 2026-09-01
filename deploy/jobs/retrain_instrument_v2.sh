#!/bin/bash
# L2 仪器 v2 重训作业(2026-09-01 决策,记录 deploy/experiments/20260901-仪器主本丢失与重训决策.md)
#
# 背景:原冻结仪器主本(l2-instrument/<hash>/results)随 2026-08-30 实例重置丢失,
#   结果目录仅存悬空符号链接农场。负责人选 B(全量重训),新协议 deltas:
#   - SSA 不走 bs16 特例,与其余四挑战同用标准 nnUNetPlans
#   - 全部 4×DCU DDP(-num_gpus 4;实例现役 4 卡)
#   - BF16(job 自有 trainer 子类经 fork 自带 nnUNet_extTrainer 扩展点接入,
#     继承现有 nnUNetTrainer_250epochs,零 nnunetv2 改动)
#   - 数据/plans/fold 与原冻结完全同批(hash 对账见决策记录 §2)
#
# 用法:sugon 上 bash deploy/jobs/retrain_instrument_v2.sh(幂等可重入:
#   completion 审计存在即跳过;checkpoint_latest 在而无 completion 即 --c 续训)
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/private_data/ctmr/instruments/l2-instrument-v2}"
DATA_ROOT="${DATA_ROOT:-/root/private_data/ctmr/data}"
RESULTS_ROOT="$WORK_ROOT/results"
AUDIT_ROOT="$WORK_ROOT/audit"
LOG_DIR="$WORK_ROOT/logs"
export nnUNet_raw="$DATA_ROOT/nnunet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnunet_preprocessed"
export nnUNet_results="$RESULTS_ROOT"
export nnUNet_compile=f

for d in "$nnUNet_raw" "$nnUNet_preprocessed"; do
    [ -d "$d" ] || { echo "[FATAL] 缺 $d" >&2; exit 1; }
done
mkdir -p "$RESULTS_ROOT" "$AUDIT_ROOT" "$LOG_DIR" "$WORK_ROOT/trainer"

# ── BF16 trainer 子类(job 自有物;经 nnUNet_extTrainer 扩展点解析,零包内改动)──
TRAINER_PY="$WORK_ROOT/trainer/nnunet_trainer_v2_bf16.py"
cat > "$TRAINER_PY" <<'PY'
"""L2 仪器 v2 BF16 trainer(2026-09-01 重训协议;决策记录 §3)。

继承现有 nnUNetTrainer_250epochs(250 epochs × 250 iter,62,500 步),唯一
delta 是把 torch 全局 autocast dtype 设为 bfloat16——基类训练/验证循环的
autocast(self.device.type, enabled=True) 因此走 BF16。DDP 为 mp.spawn,每个
worker 独立实例化 trainer,故该设置在每卡各自生效。
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_250epochs,
)


class nnUNetTrainer_250epochs_bf16(nnUNetTrainer_250epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        torch.set_autocast_dtype('cuda', torch.bfloat16)
PY
export nnUNet_extTrainer="$WORK_ROOT/trainer"

echo "============================================"
echo "L2 仪器 v2 重训:五挑战 fold_0,4×DCU DDP,BF16,标准 nnUNetPlans"
echo "results: $RESULTS_ROOT"
echo "audit:   $AUDIT_ROOT"
echo "============================================"

# ── 预检:4 卡 + 数据契约计数(与原 ChallengeRegistry 同值)──
source /opt/dtk/env.sh >/dev/null 2>&1
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
[ "$NGPU" = "4" ] || { echo "[FATAL] 需要 4 张 DCU,现见 $NGPU" >&2; exit 1; }
check_dataset() {
    local ds="$1" cases="$2"
    local n
    n=$(python3 -c "import json; print(json.load(open('$nnUNet_raw/Dataset$ds/dataset.json'))['numTraining'])")
    [ "$n" = "$cases" ] || { echo "[FATAL] Dataset$ds numTraining=$n 预期 $cases" >&2; exit 1; }
}
check_dataset 502_BraTS2023SSA 42
check_dataset 505_BraTS2023PED 68
check_dataset 504_BraTS2023METS 166
check_dataset 503_BraTS2023MEN 700
check_dataset 501_BraTS2023GLI 876

# ── 派生 plans(加法式,冻结母本零改动)──
# nnunetv2 的 plans batch_size 是全局 batch,DDP 断言 global >= world:标准
# 3d_fullres bs=2 在 4 卡下非法(亦是原 SSA bs16 特例的机械成因)。沿 ADR-0001
# 的 batch-only 派生机制,五挑战统一派生 nnUNetPlans_v2bs8:全局 2→8(每卡本地
# 2,与原单卡足迹一致);配置键 3d_fullres_bs8 仅 {batch_size, inherits_from},
# data_identifier 继承 3d_fullres,复用既有预处理数据,零重预处理。
derive_plans() {
    local name="$1"
    python3 - "$nnUNet_preprocessed" "$name" <<'PY'
import json, sys
from pathlib import Path
pre, ds = Path(sys.argv[1]), sys.argv[2]
dst = pre / ds / "nnUNetPlans_v2bs8.json"
if dst.is_file():
    print(f"[skip] {ds} nnUNetPlans_v2bs8 已存在")
else:
    doc = json.loads((pre / ds / "nnUNetPlans.json").read_text())
    parent = doc["configurations"]["3d_fullres"]
    assert parent["batch_size"] == 2, f"{ds}: parent batch != 2"
    doc["plans_name"] = "nnUNetPlans_v2bs8"
    doc["configurations"]["3d_fullres_bs8"] = {"batch_size": 8, "inherits_from": "3d_fullres"}
    dst.write_text(json.dumps(doc, indent=4) + "\n")
    print(f"[derive] {ds}: 3d_fullres(bs=2) -> 3d_fullres_bs8(bs=8), batch-only delta, 母本未动")
PY
}
derive_plans 502_BraTS2023SSA
derive_plans 505_BraTS2023PED
derive_plans 504_BraTS2023METS
derive_plans 503_BraTS2023MEN
derive_plans 501_BraTS2023GLI

# ── 顺序训练(小挑战先行作 shakedown);每挑战一条审计目录 ──
train_one() {
    local name="$1"
    local fold_dir="$RESULTS_ROOT/Dataset$name/nnUNetTrainer_250epochs_bf16__nnUNetPlans_v2bs8__3d_fullres_bs8/fold_0"
    local audit="$AUDIT_ROOT/$name"
    if [ -f "$audit/completion.json" ]; then
        echo "[skip] $name 已有 completion 审计"
        return 0
    fi
    mkdir -p "$audit" "$LOG_DIR"
    echo "[$(date -u +%FT%TZ)] train Dataset$name start"
    # 单卡异常/训练失败以非零退出;set -e 直接终止整链,由人判读后续
    if [ -f "$fold_dir/checkpoint_latest.pth" ]; then
        nnUNetv2_train "Dataset$name" 3d_fullres_bs8 0 \
            -tr nnUNetTrainer_250epochs_bf16 -p nnUNetPlans_v2bs8 -num_gpus 4 --c \
            > "$LOG_DIR/train_$name.log" 2>&1
    else
        nnUNetv2_train "Dataset$name" 3d_fullres_bs8 0 \
            -tr nnUNetTrainer_250epochs_bf16 -p nnUNetPlans_v2bs8 -num_gpus 4 \
            > "$LOG_DIR/train_$name.log" 2>&1
    fi
    # 完成核验:final checkpoint + 日志覆盖 Epoch 249 + hash 证据
    [ -f "$fold_dir/checkpoint_final.pth" ] || { echo "[FATAL] $name 无 checkpoint_final" >&2; exit 1; }
    local log_rank0
    log_rank0=$(ls "$fold_dir"/training_log_*.txt 2>/dev/null | head -1)
    [ -n "$log_rank0" ] && grep -q "Epoch 249" "$log_rank0" || { echo "[FATAL] $name 日志未覆盖 Epoch 249" >&2; exit 1; }
    {
        echo "{"
        echo "  \"dataset\": \"Dataset$name\","
        echo "  \"trainer\": \"nnUNetTrainer_250epochs_bf16\","
        echo "  \"plans\": \"nnUNetPlans_v2bs8\","
        echo "  \"configuration\": \"3d_fullres_bs8\","
        echo "  \"global_batch_size\": 8,"
        echo "  \"fold\": 0,"
        echo "  \"world_size\": 4,"
        echo "  \"precision\": \"bf16-autocast\","
        echo "  \"completed_at_utc\": \"$(date -u +%FT%TZ)\","
        echo "  \"checkpoint_final_sha256\": \"$(sha256sum "$fold_dir/checkpoint_final.pth" | cut -d' ' -f1)\","
        echo "  \"rank0_log_sha256\": \"$(sha256sum "$log_rank0" | cut -d' ' -f1)\""
        echo "}"
    } > "$audit/completion.json"
    echo "[$(date -u +%FT%TZ)] train Dataset$name done -> $audit/completion.json"
}

# SSA(42)→ PED(68)→ METS(166)→ MEN(700)→ GLI(876)
train_one 502_BraTS2023SSA
train_one 505_BraTS2023PED
train_one 504_BraTS2023METS
train_one 503_BraTS2023MEN
train_one 501_BraTS2023GLI

echo "============================================"
echo "  全部完成。五挑战 completion 审计: $AUDIT_ROOT/<CH>/completion.json"
echo "  后续: 结果树替换 nnunet_results → spec 重钉 → 校准重跑 → 新 ADR(见决策记录 §5)"
echo "============================================"
