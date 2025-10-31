#!/bin/bash

CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.m2m -p 8883 /share/models/facebook/m2m100_1.2B
