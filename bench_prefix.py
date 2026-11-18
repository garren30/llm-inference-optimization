"""
Benchmark de PREFIX CACHING: mide time-to-first-token (TTFT) con un prefijo
compartido LARGO + una pregunta corta que varía.

Prefix caching ataca el PREFILL, así que la métrica correcta es el TTFT
(cuánto tarda en llegar el primer token), que medimos por streaming.

Con el servidor vLLM levantado en otra terminal:
    # con prefix caching (default de vLLM):
    python bench_prefix.py --tag prefix-on
    # sin prefix caching (servidor arrancado con --no-enable-prefix-caching):
    python bench_prefix.py --tag prefix-off
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

from _resultslog import log_result

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")

# --- Prefijo compartido LARGO (~1.5k tokens) ---
# IDÉNTICO en cada petición -> cacheable. Simula un system prompt / documento RAG.
_PARRAFO = (
    "You are a meticulous operations assistant embedded in a large distributed "
    "inference platform. You must follow the operating manual exactly, cite the "
    "relevant runbook section, prefer safe reversible actions, escalate when a "
    "change is irreversible, and never guess a value you can look up. The platform "
    "serves language models behind an OpenAI-compatible API on Kubernetes, with "
    "autoscaling, observability, and strict latency and cost budgets per tenant. "
)
SYSTEM_PREFIX = _PARRAFO * 18  # ~2k tokens de prefijo fijo (cabe en max-model-len 4096)

# Preguntas cortas que VARÍAN (la parte no compartida, el sufijo).
PREGUNTAS = [
    "Summarize the manual in one sentence.",
    "What should I do before an irreversible change?",
    "Name one latency optimization.",
    "What API shape does the platform expose?",
    "When should I escalate?",
]

N = 20  # peticiones medidas


def ttft(pregunta):
    """Envía una petición en streaming y devuelve el time-to-first-token (s)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": pregunta},
        ],
        "max_tokens": 32,
        "temperature": 0.7,
        "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def percentil(valores, p):
    o = sorted(valores)
    k = (len(o) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(o) - 1)
    if f == c:
        return o[f]
    return o[f] + (o[c] - o[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="prefix-on / prefix-off")
    args = ap.parse_args()

    print(f"[{args.tag}] modelo={MODEL}")
    print("Calentando (puebla el cache del prefijo)...")
    ttft(PREGUNTAS[0])  # primera: cache miss; puebla el prefijo

    print(f"Midiendo TTFT en {N} peticiones (prefijo compartido ~1.5k tok)...\n")
    ttfts = []
    for i in range(N):
        t = ttft(PREGUNTAS[i % len(PREGUNTAS)])
        ttfts.append(t)
        print(f"  petición {i+1:>2}: TTFT {t*1000:7.1f} ms")

    p50 = percentil(ttfts, 50)
    p99 = percentil(ttfts, 99)
    media = statistics.mean(ttfts)

    print("\n" + "=" * 50)
    print(f"RESULTADOS PREFIX — tag={args.tag}")
    print("=" * 50)
    print(f"TTFT media:  {media*1000:.1f} ms")
    print(f"TTFT p50:    {p50*1000:.1f} ms")
    print(f"TTFT p99:    {p99*1000:.1f} ms")
    print("=" * 50)

    for metric, value in [("ttft_mean_s", media), ("ttft_p50_s", p50), ("ttft_p99_s", p99)]:
        log_result(args.tag, "prefix", 1, metric, value)
    print(f"\n-> resultados guardados en results.csv (tag={args.tag})")
    print("Revisa en el log del servidor la línea 'Prefix cache hit rate' como corroboración.")


if __name__ == "__main__":
    main()