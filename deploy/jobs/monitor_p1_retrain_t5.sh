#!/bin/bash
# 序列③ T5(#315)训练监控:dev FID eval 点里程碑 + 终态(早停/进程死亡)轮询。
# 本机运行,经 ssh sugon 轮询服务器 run 树;每 10 min 一轮,eval 点变化才出一行。
# 终态(.early_stop 落盘,或进程消失而无 .early_stop——崩溃面)即退出,监控流结束。
# 用途:会话侧完成信号;服务器训练本体由 setsid nohup 承载,与本脚本生命周期无关。
set -u
CK=/root/private_data/ctmr/runs/p1_t5/ckpt
LOG=/root/private_data/ctmr/runs/p1_t5/logs/train_20260904T021136Z.log
prev=-1
while true; do
    out=$(ssh -o ConnectTimeout=20 sugon "
        if [ -f '$CK/.early_stop' ]; then
            echo \"TERMINAL early_stop: \$(tr '\n' ' ' < '$CK/.early_stop')\"
        elif ! pgrep -f 'modality_label.train' > /dev/null; then
            echo 'TERMINAL process-gone-without-early-stop — check train log'
        else
            n=\$(wc -l < '$CK/dev_eval/dev_trend.jsonl' 2>/dev/null || echo 0)
            echo \"ALIVE evals=\$n\"
        fi" 2>/dev/null) || out="ALIVE ssh-blip"
    case "$out" in
        TERMINAL*)
            echo "$out"
            ssh -o ConnectTimeout=20 sugon "tail -5 '$CK/dev_eval/dev_trend.jsonl' 2>/dev/null" 2>/dev/null
            exit 0
            ;;
    esac
    if [[ "$out" == ALIVE\ evals=* ]]; then
        n=${out#ALIVE evals=}
        if [ "$n" != "$prev" ]; then
            echo "T5 eval 点 $n/20: $(ssh -o ConnectTimeout=20 sugon "tail -1 '$CK/dev_eval/dev_trend.jsonl' 2>/dev/null" 2>/dev/null | head -c 300)"
            prev=$n
        fi
    fi
    sleep 600
done
