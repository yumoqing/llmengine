#!/bin/bash
CUDA_VISIBLE_DEVICES=7 /share/vllm-0.8.5/bin/python -m llmengine.neo4j -p 8885 Neo4j &
