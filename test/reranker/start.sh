#!/bin/bash

# CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.rerank -p 9997 /d/ymq/models/Qwen/Qwen3-Reranker-0___6B
CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.rerank -p 9997 /share/models/BAAI/bge-reranker-v2-m3
