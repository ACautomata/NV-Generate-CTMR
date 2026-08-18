# PROTOTYPE (throwaway, wayfinder #18) — one-shot smoke run on gauss.
#
# Usage (from the repo root on gauss, after p3_setup.sh finished):
#   CUDA_VISIBLE_DEVICES=0 bash prototype/p3_image_cond_controlnet/run_smoke.sh [MAX_STEPS]
#
# Stages: prep (VAE encode + 12-pair data list) -> smoke train -> inference
# (ControlNet + img2img on the contact-sheet pairs) -> contact sheet.

#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=~/nv-p3-venv/bin/python
MAX_STEPS="${1:-120}"
SNAP_EVERY=$((MAX_STEPS / 3))
OUT=prototype/p3_image_cond_controlnet/out

echo "=== [1/4] prep: VAE encode + data list ==="
$PY prototype/p3_image_cond_controlnet/prep_p3_data.py

echo "=== [2/4] smoke training (${MAX_STEPS} steps) ==="
$PY prototype/p3_image_cond_controlnet/train_p3_smoke.py --max-steps "$MAX_STEPS" --snapshot-every "$SNAP_EVERY"

echo "=== [3/4] inference on contact-sheet pairs ==="
CKPT=$OUT/train/controlnet_p3_smoke_step${MAX_STEPS}.pt
for pair in "GLI BraTS-GLI-00000-000 t1n t2w" "MEN BraTS-MEN-00008-000 t1n t2f" "SSA BraTS-SSA-00002-000 t2w t1n"; do
  set -- $pair
  sub=$1; case=$2; src=$3; tgt=$4
  src_nii=~/nv-vae-brats/datasets/brats2023_samples/$sub/$case/$case-$src.nii.gz
  $PY prototype/p3_image_cond_controlnet/infer_p3_controlnet.py \
      --src "$src_nii" --tgt-modality "$tgt" --controlnet-ckpt "$CKPT" --seed 0
  $PY prototype/p3_image_cond_controlnet/infer_p3_img2img.py \
      --src "$src_nii" --tgt-modality "$tgt" --strength 0.8 --seed 0
done

echo "=== [4/4] contact sheet ==="
$PY prototype/p3_image_cond_controlnet/contact_sheet_p3.py

echo "SMOKE_DONE — sheet: $OUT/contact_sheet_p3.png, loss: $OUT/train/loss.jsonl"
