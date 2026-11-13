#!/usr/bin/env bash
# CONFIG B — OPTIMIZADA: se QUITA --enforce-eager.
# Con eso vLLM habilita torch.compile + captura de CUDA graphs, que recortan
# el overhead de lanzamiento de kernels por paso de decode.
#
# OJO (RTX 4050, ~6 GB): la captura de CUDA graphs usa VRAM extra y el primer
# arranque tarda más (compila/captura en el warmup). Si da OOM al arrancar,
# baja --gpu-memory-utilization a 0.75 o --max-model-len a 2048 y vuelve a medir
# (anota el cambio: debe ser el MISMO en ambas configs para comparar justo).
set -e
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80
