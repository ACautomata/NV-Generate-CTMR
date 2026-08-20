#!/bin/bash
# Issue #38 P3 img2img 分片启动器。用法：bash p3_launch_shards.sh 5 6 7
# 每个 GPU 一个分片（p3_jobs_gpu<k>.jsonl），断点续跑（已存在输出自动 skip）。
set -u
BASE=/root/private_data/l2-synth-eval
LOGS=$BASE/logs
cd /root/private_data/nv-dcu-smoke/NV-Generate-CTMR
for GPU in "$@"; do
  HIP_VISIBLE_DEVICES=$GPU nohup python3 -m scripts.img2img_batch \
    -e configs/environment_maisi_diff_model_rflow-mr-brain.json \
    -c /tmp/test_smoke_config.json \
    -t configs/config_network_rflow.json \
    --jobs $BASE/p3_jobs_gpu$GPU.jsonl \
    --strength 0.9 -g 1 \
    > $LOGS/p3_gpu$GPU.log 2>&1 &
  echo "launched p3 shard gpu$GPU pid=$!"
done
