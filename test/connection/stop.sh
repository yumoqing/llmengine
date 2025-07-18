#!/bin/bash

PORT=8888
PID=$(lsof -t -i:$PORT)

if [ -n "$PID" ]; then
    echo "找到端口 $PORT 的进程: PID=$PID"
    kill -9 $PID
    echo "已终止端口 $PORT 的进程"
else
    echo "未找到端口 $PORT 的进程"
fi