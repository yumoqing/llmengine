#!/bin/bash

CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.triple -p 9991 /share/models/Babelscape/mrebel-large