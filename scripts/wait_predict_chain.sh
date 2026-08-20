#!/bin/bash
# 等 chain2 预测链完成：predict-status.txt 出现第 2 个 ALL_PREDICTS_p3_DONE 或 rc=1 失败
while true; do
  out=$(ssh sugon 'n1=$(grep -c "ALL_PREDICTS_p1_DONE" /root/private_data/l2-synth-eval/logs/predict-status.txt); n3=$(grep -c "ALL_PREDICTS_p3_DONE" /root/private_data/l2-synth-eval/logs/predict-status.txt); p1=$(find /root/private_data/l2-synth-eval/p1_predictions -name "*.nii.gz" 2>/dev/null | wc -l); p3=$(find /root/private_data/l2-synth-eval/p3_predictions -name "*.nii.gz" 2>/dev/null | wc -l); echo "wave1=$n1 wave3=$n3 p1=$p1 p3=$p3"' 2>/dev/null | grep -v WARNING | tail -1)
  n3=$(echo "$out" | grep -oE 'wave3=[0-9]+' | cut -d= -f2)
  if [ -n "$out" ]; then echo "$out"; fi
  if [ "$n3" = "2" ]; then echo "PREDICT_CHAIN_DONE"; exit 0; fi
  sleep 120
done
