#!/bin/bash
# 仪器 v2 重训探针(会话侧):进程活性 + 完成数 + 起止标记 + 错误计数,单行输出
#
# 路径与 retrain_instrument_v2.sh 同源:WORK_ROOT 可覆写(默认 l2-instrument-v2),
# 否则覆写了 WORK_ROOT 的训练会被本探针当成默认树而死活/完成数误判。
# 错误计数只盯当前活跃 attempt(mtime 最新的 train_*.log):续训虽会以 > 截断
# 本挑战日志,但任何不经本脚本截断的重起路径都不应让陈旧 traceback 顶着 err>0,
# 把健康的续训误报成 RETRAIN ERROR。
W="${WORK_ROOT:-/root/private_data/ctmr/instruments/l2-instrument-v2}"
PID_FILE="$W/logs/retrain.pid"
ssh -o ConnectTimeout=20 sugon "
latest=\$(ls -t $W/logs/train_*.log 2>/dev/null | head -1)
pid=\$(cat $PID_FILE 2>/dev/null)
alive=\$(ps -p \"\$pid\" -o pid= 2>/dev/null | wc -l)
done=\$(ls $W/audit/*/completion.json 2>/dev/null | wc -l)
mark=\$(grep -E 'train Dataset.*(start|done)|FATAL' $W/logs/retrain.log 2>/dev/null | tail -1 | sed 's/^\\[\\([^]]*\\)\\] //' | cut -c1-60)
err=0
[ -n \"\$latest\" ] && err=\$(grep -cE 'Traceback|out of memory|AssertionError|FATAL' \"\$latest\" 2>/dev/null)
epoch=
[ -n \"\$latest\" ] && epoch=\$(grep -oE 'Epoch [0-9]+' \"\$latest\" 2>/dev/null | tail -1)
echo \"alive=\$alive done=\$done err=\$err epoch=\$epoch mark=\$mark\"
" 2>/dev/null
