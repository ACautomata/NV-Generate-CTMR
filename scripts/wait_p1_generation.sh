#!/bin/bash
# 轮询 sugon 直到 352 个 P1 文件齐 / 出现失败 / 生成进程全部退出
while true; do
  out=$(ssh sugon 'c=""; for ch in GLI SSA MEN METS PED; do n=$(find /root/private_data/l2-synth-eval/p1_samples/$ch -name "*.nii.gz" 2>/dev/null | wc -l); c="$c $ch=$n"; done; alive=$(ps aux | grep batch_generate_p1_gpu | grep -v grep | wc -l); fails=$(grep -hE "FAIL|NO_OUTPUT" /root/private_data/l2-synth-eval/logs/*_driver.log 2>/dev/null | wc -l); echo "$c alive=$alive fails=$fails"' 2>/dev/null | grep -v WARNING | tail -1)
  total=$(echo "$out" | grep -oE '=[0-9]+' | head -5 | tr -d '=' | paste -sd+ - | bc)
  alive=$(echo "$out" | grep -oE 'alive=[0-9]+' | cut -d= -f2)
  fails=$(echo "$out" | grep -oE 'fails=[0-9]+' | cut -d= -f2)
  if [ "$fails" != "0" ] || [ "$total" = "352" ] || [ "$alive" = "0" ]; then
    echo "FINAL: $out"
    break
  fi
  sleep 120
done
