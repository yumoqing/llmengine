#!/bin/bash

CUDA_VISIBLE_DEVICES=4 /share/vllm-0.8.5/bin/python -m llmengine.mm_embedding -p 8882
