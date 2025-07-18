#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=5 /share/vllm-0.8.5/bin/python -m llmengine.server -p 9999 /share/models/google/gemma-3-4b-it &
CUDA_VISIBLE_DEVICES=5 /share/vllm-0.8.5/bin/python -m llmengine.server -p 9999 /share/models/google/gemma-3-4b-it &
