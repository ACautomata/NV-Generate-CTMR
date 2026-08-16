# PROTOTYPE (throwaway, wayfinder #9) — BraTS2023 sample downloader
# Re-downloads the 12-case smoke-test sample set from public HF mirrors.
# Idempotent: re-run to fill in missing files only.
#
# Compliance: CC BY-NC 4.0 mirrored data — local one-off smoke test only.
# Never redistribute, never commit under datasets/.

#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/datasets/brats2023_samples"
HF_HOST="${HF_HOST:-https://huggingface.co}"  # set HF_HOST=https://hf-mirror.com behind a firewall

# subchallenge|hf_repo|top_dir|case
CASES="
GLI|MedOtter/brats2023-gli-dataset|ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData|BraTS-GLI-00000-000
GLI|MedOtter/brats2023-gli-dataset|ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData|BraTS-GLI-00002-000
GLI|MedOtter/brats2023-gli-dataset|ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData|BraTS-GLI-00003-000
MEN|MedOtter/brats2023-men-dataset|BraTS-MEN-Train|BraTS-MEN-00004-000
MEN|MedOtter/brats2023-men-dataset|BraTS-MEN-Train|BraTS-MEN-00008-000
MEN|MedOtter/brats2023-men-dataset|BraTS-MEN-Train|BraTS-MEN-00010-000
SSA|MedOtter/brats2023-ssa|ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2|BraTS-SSA-00002-000
SSA|MedOtter/brats2023-ssa|ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2|BraTS-SSA-00007-000
SSA|MedOtter/brats2023-ssa|ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2|BraTS-SSA-00008-000
PED|MedOtter/brats2023-ped-dataset|ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData|BraTS-PED-00002-000
PED|MedOtter/brats2023-ped-dataset|ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData|BraTS-PED-00003-000
PED|MedOtter/brats2023-ped-dataset|ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData|BraTS-PED-00004-000
"

for suffix in t1n t1c t2w t2f seg; do
  while IFS='|' read -r sub repo top case; do
    [ -z "$sub" ] && continue
    out="$BASE_DIR/$sub/$case/$case-$suffix.nii.gz"
    if [ -s "$out" ]; then
      echo "SKIP $out"
      continue
    fi
    mkdir -p "$(dirname "$out")"
    url="$HF_HOST/datasets/$repo/resolve/main/$top/$case/$case-$suffix.nii.gz"
    echo "GET  $url"
    curl -fL --retry 3 -o "$out" "$url"
  done <<< "$CASES"
done

echo "DONE. File count:"
find "$BASE_DIR" -name "*.nii.gz" | wc -l
