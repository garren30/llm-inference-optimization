# llm-inference-optimization

Lowering latency and cost of an LLM inference server, measured **before vs after**.
Builds on [llm-serving-vllm](https://github.com/garren30/llm-serving-vllm) (vLLM +
OpenAI-compatible API, containerized, on Kubernetes), which provides the measured
baseline this repo optimizes against.

> **Headline (fill once experiment 1 is done):** _p99 latency from X s to Y s,
> throughput +Z% — single variable changed (CUDA graphs on)._

## Hardware
RTX 4050 Laptop GPU (Ada, ~6 GB VRAM) · WSL2 Ubuntu · vLLM · Qwen2.5-1.5B-Instruct.

## Baseline (measured in peldaño 2)
| metric | value |
|---|---|
| sequential throughput | ~53.7 tok/s |
| p99 latency (sequential) | ~2.45 s |
| decode cost | ~18.4 ms/token |
| throughput ceiling | ~2250 tok/s (saturates ~64 concurrent) |

## Experiments
See [`EXPERIMENTS.md`](./EXPERIMENTS.md) for the full lab notebook.

1. **compile on/off** — removing `--enforce-eager` (torch.compile + CUDA graphs).
2. **AWQ quantization** — serve quantized, measure latency/throughput/VRAM. _(next)_
3. **prefix/KV caching · model routing.** _(next)_

## How to run
```bash
# Terminal 1 — pick a config
bash serve_eager.sh        # baseline (eager)
bash serve_compiled.sh     # optimized (CUDA graphs on)

# Terminal 2 — benchmark (writes to results.csv)
python bench_serial.py   --tag eager
python bench_parallel.py --tag eager

# Charts
pip install matplotlib
python plot_results.py     # -> latency.png, throughput.png
```

## Files
- `serve_eager.sh` / `serve_compiled.sh` — the two server configs under test.
- `bench_serial.py` — single-request latency p50/p95/p99 + throughput.
- `bench_parallel.py` — throughput across concurrency levels (continuous batching).
- `plot_results.py` — before/after charts from `results.csv`.
- `EXPERIMENTS.md` — lab notebook (hypothesis, steps, results table).
