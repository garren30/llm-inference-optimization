"""
Benchmark SECUENCIAL (una petición a la vez): latencia p50/p95/p99 + throughput.
Reusa el del peldaño 2, pero ahora:
  - toma una ETIQUETA (--tag) para marcar la config (ej. eager / compiled / awq)
  - escribe los resultados a results.csv para graficar antes/después
  - el modelo sale de la variable de entorno MODEL (así sirve para AWQ después)

Con el servidor vLLM levantado en otra terminal:
    python bench_serial.py --tag eager
    python bench_serial.py --tag compiled
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

from _resultslog import log_result

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPT = "Escribe un párrafo corto explicando qué es Kubernetes."
N = 20


def una_peticion():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0.7,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    t1 = time.perf_counter()
    return (t1 - t0), data["usage"]["completion_tokens"]


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
    ap.add_argument("--tag", required=True, help="etiqueta de la config (ej. eager, compiled, awq)")
    args = ap.parse_args()

    print(f"[{args.tag}] modelo={MODEL}")
    print("Calentando (1 warmup)...")
    una_peticion()

    print(f"Midiendo {N} peticiones...\n")
    latencias = []
    tokens_totales = 0
    t_inicio = time.perf_counter()
    for i in range(N):
        lat, toks = una_peticion()
        latencias.append(lat)
        tokens_totales += toks
        print(f"  petición {i+1:>2}: {lat:6.3f} s, {toks} tokens")
    t_total = time.perf_counter() - t_inicio

    p50 = percentil(latencias, 50)
    p95 = percentil(latencias, 95)
    p99 = percentil(latencias, 99)
    media = statistics.mean(latencias)
    tput = tokens_totales / t_total

    print("\n" + "=" * 50)
    print(f"RESULTADOS SECUENCIAL — tag={args.tag}")
    print("=" * 50)
    print(f"Latencia media:  {media:.3f} s")
    print(f"Latencia p50:    {p50:.3f} s")
    print(f"Latencia p95:    {p95:.3f} s")
    print(f"Latencia p99:    {p99:.3f} s")
    print(f"Throughput:      {tput:.1f} tokens/s")
    print("=" * 50)

    for metric, value in [
        ("mean_s", media), ("p50_s", p50), ("p95_s", p95),
        ("p99_s", p99), ("throughput_tok_s", tput),
    ]:
        log_result(args.tag, "serial", 1, metric, value)
    print(f"\n-> resultados guardados en results.csv (tag={args.tag})")


if __name__ == "__main__":
    main()
