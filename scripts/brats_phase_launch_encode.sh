#!/bin/bash
# Issue #52 generative-side VAE encode launcher on sugon DCU.
# Splits the manifest encode list into 3 shards (GLI / MEN / rest), one GPU each.
# Usage: bash launch_encode.sh   (idempotent: skips existing embeddings)
set -uo pipefail

PHASE=/root/private_data/brats2023_rflow_phase
REPO=/root/nv-phase-52
AE=/root/private_data/manifold/models/autoencoder_v1.pt

mkdir -p "$PHASE/encode_runs"

for SHARD in GLI MEN REST; do
  case "$SHARD" in
    GLI) GPU=0 ;;
    MEN) GPU=1 ;;
    REST) GPU=2 ;;
  esac
  python3 - "$PHASE" "$SHARD" <<'PYEOF'
import json, sys
phase, shard = sys.argv[1], sys.argv[2]
entries = json.load(open(f"{phase}/encode_source.json"))["training"]
if shard == "REST":
    picked = [e for e in entries if e["sub"] not in ("GLI", "MEN")]
else:
    picked = [e for e in entries if e["sub"] == shard]
json.dump({"training": picked}, open(f"{phase}/encode_runs/encode_{shard}.json", "w"), indent=1)
env = {
    "data_base_dir": f"{phase}/raw",
    "embedding_base_dir": f"{phase}/embeddings",
    "json_data_list": f"{phase}/encode_runs/encode_{shard}.json",
    "trained_autoencoder_path": "/root/private_data/manifold/models/autoencoder_v1.pt",
}
json.dump(env, open(f"{phase}/encode_runs/env_{shard}.json", "w"), indent=1)
print(shard, len(picked), "entries")
PYEOF
done

cat > "$PHASE/encode_runs/model_config.json" <<'JSONEOF'
{
  "diffusion_unet_train": {"batch_size": 1, "cache_rate": 0, "lr": 2e-6, "n_epochs": 1},
  "diffusion_unet_inference": {"dim": [256, 256, 128], "modality": 29}
}
JSONEOF

for SHARD in GLI MEN REST; do
  case "$SHARD" in
    GLI) GPU=0 ;;
    MEN) GPU=1 ;;
    REST) GPU=2 ;;
  esac
  echo "launching $SHARD on GPU $GPU"
  HIP_VISIBLE_DEVICES=$GPU nohup python3 -m scripts.diff_model_create_training_data \
    -e "$PHASE/encode_runs/env_$SHARD.json" \
    -c "$PHASE/encode_runs/model_config.json" \
    -t "$REPO/configs/config_network_rflow.json" -g 1 \
    > "$PHASE/encode_runs/encode_$SHARD.log" 2>&1 &
  echo "  pid $!"
done
echo "all shards launched; logs under $PHASE/encode_runs/"
