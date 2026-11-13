"""
Genera las gráficas antes/después desde results.csv.
Requiere: pip install matplotlib

    python plot_results.py

Produce:
  - latency.png     : latencia p50/p99 secuencial por config (barras)
  - throughput.png  : throughput vs concurrencia por config (líneas)
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "results.csv"


def load():
    rows = []
    with open(RESULTS_FILE) as f:
        for r in csv.DictReader(f):
            r["concurrency"] = int(r["concurrency"])
            r["value"] = float(r["value"])
            rows.append(r)
    return rows


def plot_latency(rows):
    # p50 y p99 secuencial por tag, en orden de aparición
    tags = list(dict.fromkeys(r["tag"] for r in rows if r["test"] == "serial"))
    p50 = {t: None for t in tags}
    p99 = {t: None for t in tags}
    for r in rows:
        if r["test"] == "serial" and r["metric"] == "p50_s":
            p50[r["tag"]] = r["value"]
        if r["test"] == "serial" and r["metric"] == "p99_s":
            p99[r["tag"]] = r["value"]

    x = range(len(tags))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], [p50[t] for t in tags], w, label="p50")
    ax.bar([i + w / 2 for i in x], [p99[t] for t in tags], w, label="p99")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tags)
    ax.set_ylabel("Latencia (s) — menor es mejor")
    ax.set_title("Latencia secuencial por configuración")
    ax.legend()
    for i, t in enumerate(tags):
        if p99[t] is not None:
            ax.text(i + w / 2, p99[t], f"{p99[t]:.2f}", ha="center", va="bottom", fontsize=8)
        if p50[t] is not None:
            ax.text(i - w / 2, p50[t], f"{p50[t]:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig("latency.png", dpi=130)
    print("-> latency.png")


def plot_throughput(rows):
    series = defaultdict(list)  # tag -> [(conc, tput)]
    for r in rows:
        if r["test"] == "parallel" and r["metric"] == "throughput_tok_s":
            series[r["tag"]].append((r["concurrency"], r["value"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, pts in series.items():
        pts.sort()
        ax.plot([c for c, _ in pts], [v for _, v in pts], marker="o", label=tag)
    ax.set_xlabel("Concurrencia")
    ax.set_ylabel("Throughput (tok/s) — mayor es mejor")
    ax.set_title("Throughput vs concurrencia por configuración")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("throughput.png", dpi=130)
    print("-> throughput.png")


if __name__ == "__main__":
    rows = load()
    if any(r["test"] == "serial" for r in rows):
        plot_latency(rows)
    if any(r["test"] == "parallel" for r in rows):
        plot_throughput(rows)
    print("Listo.")
