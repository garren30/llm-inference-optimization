# llm-inference-optimization

Lowering latency and cost of an LLM inference server, measured **before vs after**.
Each experiment changes **one variable** and reports the honest, output-normalized metric.

Builds on [llm-serving-vllm](https://github.com/garren30/llm-serving-vllm) — vLLM behind
an OpenAI-compatible API, containerized, on Kubernetes — which provides the measured
baseline this repo optimizes against.

> **AWQ 4-bit on a 6 GB laptop GPU: served a 3B model — faster *and* 2× bigger
> than the 1.5B fp16 — where the 3B in fp16 can't even allocate its KV cache.**

---

## Hardware & stack
RTX 4050 Laptop GPU (Ada, ~6 GB VRAM) · WSL2 Ubuntu · vLLM 0.33 · Qwen2.5 Instruct.

## Baseline (from the serving project)
| metric | value |
|---|---|
| sequential throughput | ~53.7 tok/s |
| p99 latency (sequential) | ~2.45 s |
| decode cost | ~18.4 ms/token |
| throughput ceiling | ~2250 tok/s (saturates ~64 concurrent) |

---

## Experiment 2 — AWQ 4-bit quantization

**Why:** at batch=1 decode is memory-bound reading weights. AWQ stores weights
in 4-bit (activation-aware: it protects the salient weights), so fewer bytes are
read per token — and the freed VRAM becomes KV cache, i.e. serving capacity.
Served without `--quantization awq` so vLLM auto-selects the fast **Marlin** kernel.

### 2a — Speed (same 1.5B model, fp16 vs AWQ)
| config    | p50 (s) | p99 (s) | tok/s seq | tok/s @ conc32 |
|-----------|---------|---------|-----------|----------------|
| fp16-1.5b | 2.29    | 2.43    | 55.6      | 1473.6         |
| awq-1.5b  | 0.98    | 0.99    | 130.2     | 2710.0         |

### 2a — VRAM & capacity (from vLLM startup logs)
| config    | weights  | free KV cache | max concurrency (4096 tok) |
|-----------|----------|---------------|-----------------------------|
| fp16-1.5b | 2.98 GiB | 1.01 GiB      | 9.19×                       |
| awq-1.5b  | 1.1 GiB  | 2.72 GiB      | 24.84×                      |

![Sequential latency by configuration](latency.png)
![Throughput vs concurrency by configuration](throughput.png)

**Result (2a):** 2.34× lower latency and 2.7× more serving capacity on the same
6 GB card, same model — no quality loss. The speedup *narrows* with concurrency
(2.33× at conc 1 → 1.84× at conc 32): a textbook memory-bound → compute-bound
transition, measured on real hardware.

> Weights shrink 2.7× (not the theoretical 4×) because embeddings/layernorms/
> lm_head stay fp16 and AWQ keeps per-group scales/zeros.

### 2b — Capability unlock
With the VRAM AWQ frees, a bigger model fits on the same 6 GB card:

| model    | weights  | KV cache      | runs? |
|----------|----------|---------------|-------|
| 7B-AWQ   | ~4.5 GiB | negative      | ❌ card's ceiling |
| 3B-fp16  | 5.79 GiB | **−0.73 GiB** | ❌ no room for KV cache |
| 3B-AWQ   | 1.95 GiB | 1.57 GiB      | ✅ 11.16× concurrency |

A 3B in fp16 leaves **−0.73 GiB** for its KV cache and fails to start; the 3B-AWQ
runs comfortably (p50 1.67 s, 75.6 tok/s, up to 1579 tok/s at conc 32).

**The counterintuitive win:** the 3B-AWQ is *faster* than the 1.5B-fp16
(75.6 vs 55.6 tok/s, p50 1.67 vs 2.29 s) despite **2× the parameters** — because
decode speed tracks *bytes read per token*, and 4-bit 3B weights (1.95 GiB) are
fewer bytes than 16-bit 1.5B weights (2.98 GiB). Quantization delivered a bigger,
faster model on fixed hardware.

> `--gpu-memory-utilization` is a fraction of *total* VRAM (6 GB), not free. With
> ~1 GiB taken by the display (4.91 free), the usable max is ~0.80; at 0.92 vLLM
> won't even start (requests 5.52 > 4.91 free).

---

## Experiment 1 — compile on/off (`--enforce-eager`)

Removing `--enforce-eager` lets vLLM capture **CUDA graphs** (replay the kernel
sequence in one launch) and run **torch.compile** (fuse ops), cutting CPU→GPU
dispatch overhead in single-request decode.

| config   | p50 (s) | p99 (s) | tok/s seq | tok/s @ conc32 |
|----------|---------|---------|-----------|----------------|
| eager    | 2.37    | 2.69    | 50.2      | 1241.7         |
| compiled | 2.16    | 2.50    | 53.1      | 1297.6         |

**Result:** +5.8% sequential throughput, up to +8.7% under concurrency, **zero
VRAM cost** — a free win on a 1.5B model, larger on bigger/compute-bound models.

**Honest-metric note:** the raw p50 drop (~9%) was inflated by variable output
length (`temperature 0.7`); throughput normalizes for that. Fixed afterward with
`ignore_eos` (every request emits exactly 128 tokens).

---

## Roadmap
- **Experiment 3 — prefix/KV caching · model routing.**
- Step 4 (separate repo): the platform — autoscaling, observability, CI/CD.

---

## How to run
```bash
# Terminal 1 — serve a config
bash serve_compiled.sh                                   # fp16 1.5B (baseline)
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \              # AWQ 1.5B
  --max-model-len 4096 --gpu-memory-utilization 0.80
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \                # AWQ 3B (capability unlock)
  --max-model-len 4096 --gpu-memory-utilization 0.80

# Terminal 2 — benchmark (appends to results.csv); MODEL must match the served id
MODEL=Qwen/Qwen2.5-1.5B-Instruct     python bench_serial.py   --tag fp16-1.5b
MODEL=Qwen/Qwen2.5-1.5B-Instruct     python bench_parallel.py --tag fp16-1.5b
MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ python bench_serial.py   --tag awq-1.5b
MODEL=Qwen/Qwen2.5-1.5B-Instruct-AWQ python bench_parallel.py --tag awq-1.5b
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ   python bench_serial.py   --tag awq-3b
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ   python bench_parallel.py --tag awq-3b

# Charts
pip install matplotlib
python plot_results.py     # -> latency.png, throughput.png
```

## Method
Benchmarks tag each run (`--tag`) and append to `results.csv` in tidy format
(`tag, test, concurrency, metric, value`); `plot_results.py` renders before/after
charts. One variable per experiment; throughput is the primary metric; `ignore_eos`
fixes output length so latency is directly comparable.

## Files
- `serve_eager.sh` / `serve_compiled.sh` — eager vs CUDA-graphs server configs.
- `bench_serial.py` — single-request latency p50/p95/p99 + throughput.
- `bench_parallel.py` — throughput across concurrency levels (continuous batching).
- `plot_results.py` — before/after charts from `results.csv`.
- `_resultslog.py` — shared CSV logger.
- `EXPERIMENTS.md` — lab notebook (hypothesis, mechanism, results, lessons).