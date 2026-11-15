# llm-inference-optimization

Lowering latency and cost of an LLM inference server, measured **before vs after**.
Each experiment changes **one variable** and reports the honest, output-normalized metric.

Builds on [llm-serving-vllm](https://github.com/garren30/llm-serving-vllm) — vLLM behind
an OpenAI-compatible API, containerized, on Kubernetes — which provides the measured
baseline this repo optimizes against.

> **Experiment 1 — CUDA graphs on (removed `--enforce-eager`): +5.8% sequential
> throughput, up to +8.7% under concurrency, at zero VRAM cost (Qwen2.5-1.5B).**

---

## Hardware & stack
RTX 4050 Laptop GPU (Ada, ~6 GB VRAM) · WSL2 Ubuntu · vLLM 0.33 · Qwen2.5-1.5B-Instruct.

## Baseline (from the serving project)
| metric | value |
|---|---|
| sequential throughput | ~53.7 tok/s |
| p99 latency (sequential) | ~2.45 s |
| decode cost | ~18.4 ms/token |
| throughput ceiling | ~2250 tok/s (saturates ~64 concurrent) |

---

## Experiment 1 — compile on/off (`--enforce-eager`)

**Why:** in single-request decode the GPU finishes each tiny kernel fast, then waits
for the CPU to launch the next one (it's memory-bound). Removing `--enforce-eager`
lets vLLM capture **CUDA graphs** (replay the whole kernel sequence in one launch)
and run **torch.compile** (fuse ops), cutting CPU→GPU dispatch overhead.

| config   | p50 (s) | p99 (s) | tok/s seq | ms/token | tok/s @ conc32 |
|----------|---------|---------|-----------|----------|----------------|
| eager    | 2.37    | 2.69    | 50.2      | 19.9     | 1241.7         |
| compiled | 2.16    | 2.50    | 53.1      | 18.8     | 1297.6         |

![Sequential latency by configuration](latency.png)
![Throughput vs concurrency by configuration](throughput.png)

**Result:** +5.8% sequential throughput, +4.5–8.7% across concurrency, **no VRAM cost**.
The only price is a slower first start (graph capture happens during warmup).

**Honest-metric note:** the raw p50 drop (~9%) is *inflated* — with `temperature 0.7`
each run emitted a different number of tokens (compiled averaged 108.8 vs eager 117.4),
so part of its "lower latency" is simply shorter output. **Throughput (tokens ÷ time)
normalizes for that**, so it's the number I report. Upcoming experiments pin output
length (`ignore_eos`) to make latency directly comparable too.

**Why modest and not 3×:** a 1.5B model is memory-bound on weight loading, so launch
overhead was real but not dominant. The effect scales on larger / more compute-bound models.

---

## Roadmap
- **Experiment 2 — AWQ quantization (core):** serve a 7B-AWQ model that doesn't fit
  unquantized in 6 GB; measure latency, throughput and **VRAM** before/after. The
  headline target: *"quantization fit a 7B where only a 1.5B fit before."*
- **Experiment 3 — prefix/KV caching · model routing.**

---

## How to run
```bash
# Terminal 1 — pick a config
bash serve_eager.sh        # baseline (eager)
bash serve_compiled.sh     # optimized (CUDA graphs on)

# Terminal 2 — benchmark (appends to results.csv)
python bench_serial.py   --tag eager
python bench_parallel.py --tag eager

# Charts
pip install matplotlib
python plot_results.py     # -> latency.png, throughput.png
```

## Method
Benchmarks tag each run (`--tag`) and append to `results.csv` in tidy format
(`tag, test, concurrency, metric, value`); `plot_results.py` renders before/after
charts. One variable changes per experiment; throughput is the primary metric.

## Files
- `serve_eager.sh` / `serve_compiled.sh` — the two server configs under test.
- `bench_serial.py` — single-request latency p50/p95/p99 + throughput.
- `bench_parallel.py` — throughput across concurrency levels (continuous batching).
- `plot_results.py` — before/after charts from `results.csv`.
- `_resultslog.py` — shared CSV logger.
- `EXPERIMENTS.md` — lab notebook (hypothesis, mechanism, results, lessons).