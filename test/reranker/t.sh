#!/usr/bin/bash

curl http://localhost:9997/v1/rerank \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "rerank-001",
  "query": "什么是量子计算？",
  "documents": [
    "量子计算是一种使用量子比特进行计算的方式。",
    "古典计算机使用的是二进制位。",
    "天气预报依赖于统计模型。",
    "量子计算与物理学密切相关。"
  ],
  "top_n": 5
}
EOF
