#!/bin/bash
# Issue #38 P1 式 v1 DM 批量生成脚本
# 在 sugon DCU 上运行：bash /root/private_data/l2-synth-eval/batch_generate_p1.sh
set -euo pipefail
cd /root/private_data/nv-dcu-smoke/NV-Generate-CTMR

CASE_LIST="/root/private_data/l2-synth-eval/case_lists/p1_cases.json"
OUTPUT_BASE="/root/private_data/l2-synth-eval/p1_samples"
LOG_DIR="/root/private_data/l2-synth-eval/logs"
CONFIG_BASE="/tmp/test_smoke_config.json"
mkdir -p "$LOG_DIR" "$OUTPUT_BASE"

# modality labels: t1n=9, t1c=17, t2w=10, t2f=11
declare -A MODLABEL=( [t1n]=9 [t1c]=17 [t2w]=10 [t2f]=11 )

TOTAL=$(python3 -c "import json; print(len(json.load(open('$CASE_LIST'))))")
echo "Total cases: $TOTAL | Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

IDX=0
python3 -c "import json; [print(json.dumps(e)) for e in json.load(open('$CASE_LIST'))]" | while read -r ENTRY; do
    IDX=$((IDX + 1))
    CASE_ID=$(echo "$ENTRY" | python3 -c "import sys,json; print(json.load(sys.stdin)['case_id'])")
    CHALLENGE=$(echo "$ENTRY" | python3 -c "import sys,json; print(json.load(sys.stdin)['challenge'])")
    CASE_DIR="$OUTPUT_BASE/$CHALLENGE/$CASE_ID"
    mkdir -p "$CASE_DIR"

    for MOD in t1n t1c t2w t2f; do
        OUT="$CASE_DIR/${MOD}.nii.gz"
        [ -f "$OUT" ] && echo "[$IDX/$TOTAL] $CHALLENGE/$CASE_ID/$MOD SKIP" && continue

        ML=${MODLABEL[$MOD]}
        SEED=$((IDX * 100 + ML))
        CFG="/tmp/_gen_${MOD}.json"

        python3 -c "
import json
c = json.load(open('$CONFIG_BASE'))
c['diffusion_unet_inference']['modality'] = $ML
c['diffusion_unet_inference']['random_seed'] = $SEED
c['output_dir'] = '$CASE_DIR'
c['output_prefix'] = '$MOD'
json.dump(c, open('$CFG', 'w'))
"
        echo -n "[$IDX/$TOTAL] $CHALLENGE/$CASE_ID/$MOD seed=$SEED ... "
        python3 -m scripts.diff_model_infer \
            -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
            -c "$CFG" \
            -t configs/config_network_rflow.json \
            -g 1 \
            >> "$LOG_DIR/${CHALLENGE}.log" 2>&1 || { echo "FAIL"; rm -f "$CFG"; continue; }

        # 重命名 diff_model_infer 的输出文件为标准名
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
done

echo "Complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Output tree:"
find "$OUTPUT_BASE" -name "*.nii.gz" | wc -l
echo "nii.gz files generated"
