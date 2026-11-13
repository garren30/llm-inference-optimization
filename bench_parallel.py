"""
Benchmark de CONCURRENCIA (continuous batching): throughput a varios niveles.
Reusa el del peldaño 2, ahora con --tag y escritura a results.csv.

Con el servidor vLLM levantado en otra terminal:
    python bench_parallel.py --tag eager
    python bench_parallel.py --tag compiled
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from _resultslog import log_result

URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPT = "Escribe un párrafo corto explicando qué es Kubernetes."

NIVELES = [1, 4, 8, 16, 32]
PETICIONES_POR_NIVEL = 32


def una_peticion(_):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 128,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="etiqueta de la config (ej. eager, compiled, awq)")
    args = ap.parse_args()

    print(f"[{args.tag}] modelo={MODEL}")
    print("Calentando...")
    una_peticion(0)

    print(f"\nCada prueba manda {PETICIONES_POR_NIVEL} peticiones.\n")
    print(f"{'concurrencia':>12} | {'tiempo':>10} | {'throughput':>14}")
    print("-" * 44)

    resultados = []
    for conc in NIVELES:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            tokens = list(ex.map(una_peticion, range(PETICIONES_POR_NIVEL)))
        dt = time.perf_counter() - t0
        tput = sum(tokens) / dt
        resultados.append((conc, tput))
        print(f"{conc:>12} | {dt:>8.2f} s | {tput:>10.1f} tok/s")
        log_result(args.tag, "parallel", conc, "throughput_tok_s", tput)

    base = resultados[0][1]
    mejor = max(resultados, key=lambda r: r[1])
    print("\n" + "=" * 44)
    print(f"Throughput conc=1:   {base:.1f} tok/s")
    print(f"Mejor (conc={mejor[0]}):     {mejor[1]:.1f} tok/s  ({mejor[1]/base:.1f}x)")
    print("=" * 44)
    print(f"\n-> resultados guardados en results.csv (tag={args.tag})")


if __name__ == "__main__":
    main()
