#!/bin/bash
# Issue #38 P1 式 v1 DM 批量生成（单挑战 × 单 GPU 版本）
# 在 sugon DCU 上运行：CHALLENGE=GLI GPU=0 bash /root/private_data/l2-synth-eval/batch_generate_p1_gpu.sh
# seed 与串行版 batch_generate_p1.sh 完全一致：全表 1-based 位置 IDX，seed = IDX*100 + modality_label
set -uo pipefail

CHALLENGE="${CHALLENGE:?need CHALLENGE env (GLI|SSA|MEN|METS|PED)}"
GPU="${GPU:?need GPU env (0-7)}"
export HIP_VISIBLE_DEVICES="$GPU"

cd /root/private_data/nv-dcu-smoke/NV-Generate-CTMR

CASE_LIST="/root/private_data/l2-synth-eval/case_lists/p1_cases.json"
OUTPUT_BASE="/root/private_data/l2-synth-eval/p1_samples"
LOG_DIR="/root/private_data/l2-synth-eval/logs"
CONFIG_BASE="/tmp/test_smoke_config.json"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# modality labels: t1n=9, t1c=17, t2w=10, t2f=11
declare -A MODLABEL=( [t1n]=9 [t1c]=17 [t2w]=10 [t2f]=11 )

TOTAL=$(python3 -c "import json; print(sum(1 for e in json.load(open('$CASE_LIST')) if e['challenge']=='$CHALLENGE'))")
echo "[$CHALLENGE GPU$GPU] $TOTAL cases | Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

DONE=0
while read -r IDX CASE_ID; do
    DONE=$((DONE + 1))
    CASE_DIR="$OUTPUT_BASE/$CHALLENGE/$CASE_ID"
    mkdir -p "$CASE_DIR"

    for MOD in t1n t1c t2w t2f; do
        OUT="$CASE_DIR/${MOD}.nii.gz"
        [ -f "$OUT" ] && echo "[$CHALLENGE $DONE/$TOTAL] $CASE_ID/$MOD SKIP" && continue

        ML=${MODLABEL[$MOD]}
        SEED=$((IDX * 100 + ML))
        # 临时配置按 挑战×模态 命名，避免并行作业互踩
        CFG="/tmp/_gen_${CHALLENGE}_${MOD}.json"

        python3 -c "
import json
c = json.load(open('$CONFIG_BASE'))
c['diffusion_unet_inference']['modality'] = $ML
c['diffusion_unet_inference']['random_seed'] = $SEED
c['output_dir'] = '$CASE_DIR'
c['output_prefix'] = '$MOD'
json.dump(c, open('$CFG', 'w'))
"
        echo -n "[$CHALLENGE $DONE/$TOTAL] $CASE_ID/$MOD seed=$SEED ... "
        python3 -m scripts.diff_model_infer \
            -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
            -c "$CFG" \
            -t configs/config_network_rflow.json \
            -g 1 \
            >> "$LOG_DIR/${CHALLENGE}.log" 2>&1 || { echo "FAIL"; rm -f "$CFG"; continue; }

        GENERATED=$(find "$CASE_DIR" -maxdepth 1 -name "${MOD}_*_modality${ML}.nii.gz" 2>/dev/null | head -1)
        if [ -n "$GENERATED" ] && [ "$GENERATED" != "$OUT" ]; then
            mv "$GENERATED" "$OUT"
            echo "OK"
        elif [ -f "$OUT" ]; then
            echo "OK (exists)"
        else
            echo "NO_OUTPUT"
        fi
        rm -f "$CFG"
    done
done < <(python3 -c "import json; d=json.load(open('$CASE_LIST')); [print(i+1, e['case_id']) for i, e in enumerate(d) if e['challenge']=='$CHALLENGE']")

echo "[$CHALLENGE GPU$GPU] Complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[$CHALLENGE] files: $(find "$OUTPUT_BASE/$CHALLENGE" -name '*.nii.gz' 2>/dev/null | wc -l)"
