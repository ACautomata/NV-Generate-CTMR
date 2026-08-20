#!/bin/bash
# 持久监控 sugon P1 生成：每 15 分钟输出一行进度，终态（完成/失败/进程退出）后退出
while true; do
  out=$(ssh sugon 'c=""; for ch in GLI SSA MEN METS PED; do n=$(find /root/private_data/l2-synth-eval/p1_samples/$ch -name "*.nii.gz" 2>/dev/null | wc -l); c="$c $ch=$n"; done; alive=$(ps aux | grep batch_generate_p1_gpu | grep -v grep | wc -l); fails=$(grep -hE "FAIL|NO_OUTPUT" /root/private_data/l2-synth-eval/logs/*_driver.log 2>/dev/null | wc -l); echo "$c alive=$alive fails=$fails"' 2>/dev/null | grep -v WARNING | tail -1)
  total=$(echo "$out" | grep -oE '=[0-9]+' | head -5 | tr -d '=' | paste -sd+ - | bc)
  alive=$(echo "$out" | grep -oE 'alive=[0-9]+' | cut -d= -f2)
  fails=$(echo "$out" | grep -oE 'fails=[0-9]+' | cut -d= -f2)
  if [ -n "$out" ] && [ "$fails" != "0" ]; then echo "GENERATION_FAILURES: $out"; exit 1; fi
  if [ "$total" = "352" ]; then echo "ALL_DONE: $out"; exit 0; fi
  if [ "$alive" = "0" ]; then echo "PROCESSES_EXITED_EARLY total=$total: $out"; exit 1; fi
  echo "progress total=$total: $out"
  sleep 900
done
