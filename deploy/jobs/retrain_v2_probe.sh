#!/bin/bash
# 仪器 v2 重训探针(会话侧):进程活性 + 完成数 + 起止标记 + 错误计数,单行输出
PID_FILE=/root/private_data/ctmr/instruments/l2-instrument-v2/logs/retrain.pid
W=/root/private_data/ctmr/instruments/l2-instrument-v2
ssh -o ConnectTimeout=20 sugon "
pid=\$(cat $PID_FILE 2>/dev/null)
alive=\$(ps -p \"\$pid\" -o pid= 2>/dev/null | wc -l)
done=\$(ls $W/audit/*/completion.json 2>/dev/null | wc -l)
mark=\$(grep -E 'train Dataset.*(start|done)|FATAL' $W/logs/retrain.log 2>/dev/null | tail -1 | sed 's/^\\[\\([^]]*\\)\\] //' | cut -c1-60)
err=\$(grep -lE 'Traceback|out of memory|AssertionError|FATAL' $W/logs/train_*.log 2>/dev/null | wc -l)
epoch=\$(grep -oE 'Epoch [0-9]+' \$(ls -t $W/logs/train_*.log 2>/dev/null | head -1) 2>/dev/null | tail -1)
echo \"alive=\$alive done=\$done err=\$err epoch=\$epoch mark=\$mark\"
" 2>/dev/null
