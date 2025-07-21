#!/bin/bash
CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.db_service -p 8886 Milvus
