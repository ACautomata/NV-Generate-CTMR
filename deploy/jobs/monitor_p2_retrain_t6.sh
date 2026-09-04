#!/bin/bash
# 序列③ T6(#316)P2 重训监控:epoch checkpoint 里程碑 + dev FID eval 点里程碑 + 终态轮询。
# 本机运行,经 ssh sugon 轮询服务器 run 树;每 10 min 一轮,里程碑变化才出一行。
# 终态(.early_stop 落盘且 select 完成,或训练进程消失而无 .early_stop——崩溃面)即退出。
# 用途:会话侧完成信号;服务器训练/watch 循环由 setsid nohup 承载,与本脚本生命周期无关。
# pgrep 模式用字符类 [k] 打破自匹配(T5 执行登记先例:监控命令自身命令行含 pattern 字面会误报)。
set -u
RT=/root/private_data/ctmr/runs/p2_t6
CK=$RT/ckpt
prev_epochs=-1
prev_evals=-1
while true; do
    out=$(ssh -o ConnectTimeout=20 sugon "
        if [ -f '$CK/.early_stop' ]; then
            es=\$(tr '\n' ' ' < '$CK/.early_stop')
            if [ -f '$RT/selection.json' ]; then
                echo \"TERMINAL early_stop+select: \$es\"
            else
                echo \"ALIVE early_stop-written-awaiting-select: \$es\"
            fi
        elif ! pgrep -f 'generate mas[k] train' > /dev/null && ! pgrep -f 'generation\.mas[k]\.train' > /dev/null; then
            echo 'TERMINAL process-gone-without-early-stop — check train log'
        else
            e=\$(ls '$CK' 2>/dev/null | grep -c '^epoch_[0-9]*\.pt$' || echo 0)
            n=\$(wc -l < '$RT/dev_eval/dev_trend.jsonl' 2>/dev/null || echo 0)
            echo \"ALIVE epochs=\$e evals=\$n\"
        fi" 2>/dev/null) || out="ALIVE ssh-blip"
    case "$out" in
        TERMINAL*)
            echo "$out"
            ssh -o ConnectTimeout=20 sugon "tail -3 '$RT/dev_eval/dev_trend.jsonl' 2>/dev/null; echo ---; cat '$RT/selection.json' 2>/dev/null | head -8" 2>/dev/null
            exit 0
            ;;
    esac
    if [[ "$out" == ALIVE\ epochs=* ]]; then
        e=${out#ALIVE epochs=}; e=${e%% *}
        n=${out##*evals=}
        if [ "$n" != "$prev_evals" ] && [ "$n" != "0" ]; then
            echo "T6 eval 点 $n: $(ssh -o ConnectTimeout=20 sugon "tail -1 '$RT/dev_eval/dev_trend.jsonl' 2>/dev/null" 2>/dev/null | head -c 260)"
            prev_evals=$n
        elif [ "$e" != "$prev_epochs" ]; then
            echo "T6 epochs=$e evals=$n"
            prev_epochs=$e
        fi
    elif [[ "$out" == ALIVE* && "$out" != "$prev_out" ]]; then
        echo "$out"
    fi
    prev_out="$out"
    sleep 600
done
