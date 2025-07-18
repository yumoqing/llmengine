#!/bin/bash

CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.embedding -p 9998 /d/ymq/models/Qwen/Qwen3-Embedding-0.6B
