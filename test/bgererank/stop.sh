#!/bin/bash

# 查找并终止运行在端口 8887 上的进程
pid=$(lsof -t -i:8887)
if [ -n "$pid" ]; then
    echo "终止进程: $pid"
    kill -9 $pid
else
    echo "未找到运行在端口 8887 上的进程"
fi