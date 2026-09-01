#!/bin/bash
# 仪器 v2 重训监视(会话侧,安静版):15 分钟轮询,仅在挑战起止翻转、错误、
# 死亡、全部完成时输出。提取一律 grep -oE(BSD sed 教训);空探针跳过;
# 连续两次 alive=0 才判死亡。
prev=""
alive_misses=0
while true; do
  out=$(bash "$(dirname "$0")/retrain_v2_probe.sh")
  if [ -z "$out" ]; then
    sleep 900
    continue
  fi
  field() { printf '%s\n' "$out" | grep -oE "$1=[0-9]+" | head -1 | cut -d= -f2; }
  alive=$(field alive)
  done=$(field done)
  err=$(field err)
  mark=$(printf '%s\n' "$out" | sed -n 's/.*mark=//p')
  if [ "${err:-0}" != "0" ]; then
    echo "RETRAIN ERROR: $out"
    exit 1
  fi
  if [ "${done:-0}" -ge 5 ] 2>/dev/null; then
    echo "RETRAIN COMPLETE (5/5): $out"
    exit 0
  fi
  if [ "${alive:-0}" = "0" ]; then
    alive_misses=$((alive_misses + 1))
    if [ "$alive_misses" -ge 2 ]; then
      echo "RETRAIN DIED EARLY (done=$done): $out"
      exit 1
    fi
  else
    alive_misses=0
  fi
  if [ -n "$mark" ] && [ "$mark" != "$prev" ]; then
    echo "retrain: $mark"
    prev=$mark
  fi
  sleep 900
done
