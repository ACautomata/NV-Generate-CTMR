#!/bin/bash
# 持久监控 sugon P3 img2img 生成：15 分钟一行，终态后退出
# 终态：1056 文件齐 / 分片进程全退 / 出现失败
while true; do
  out=$(ssh sugon 'n=$(find /root/private_data/l2-synth-eval/p3_samples -name "*.nii.gz" 2>/dev/null | wc -l); alive=$(ps aux | grep img2img_batch | grep -v grep | wc -l); fails=$(grep -h "FAILED" /root/private_data/l2-synth-eval/logs/p3_gpu*.log 2>/dev/null | wc -l); echo "p3=$n alive=$alive fails=$fails"' 2>/dev/null | grep -v WARNING | tail -1)
  n=$(echo "$out" | grep -oE 'p3=[0-9]+' | cut -d= -f2)
  alive=$(echo "$out" | grep -oE 'alive=[0-9]+' | cut -d= -f2)
  fails=$(echo "$out" | grep -oE 'fails=[0-9]+' | cut -d= -f2)
  if [ -n "$out" ] && [ "$fails" != "0" ]; then echo "P3_FAILURES: $out"; exit 1; fi
  if [ "$n" = "1056" ]; then echo "P3_ALL_DONE: $out"; exit 0; fi
  if [ "$alive" = "0" ]; then echo "P3_EXITED_EARLY n=$n: $out"; exit 1; fi
  echo "p3 progress: $out"
  sleep 900
done
