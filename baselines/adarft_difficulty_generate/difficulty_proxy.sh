#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 /home/fit/lijuanzi1/WORK/dz/SAERL/difficulty_proxy.py

# mean16
# /home/fit/lijuanzi1/WORK/dz/SAERL/stderr.315768
# /home/fit/lijuanzi1/WORK/dz/SAERL/stdout.315768
# 62387s


# mean128
# /home/fit/lijuanzi1/WORK/dz/SAERL/stderr.344003