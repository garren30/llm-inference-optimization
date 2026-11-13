#!/usr/bin/env bash
# CONFIG A — BASELINE (igual que peldaño 2): eager mode.
# --enforce-eager apaga torch.compile y la captura de CUDA graphs.
set -e
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager
