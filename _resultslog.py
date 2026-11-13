"""
Helper compartido: escribe filas al results.csv en formato tidy (largo),
fácil de graficar después.

Columnas: tag, test, concurrency, metric, value

- tag:   etiqueta del experimento/config (ej. "eager", "compiled", "awq")
- test:  "serial" o "parallel"
- concurrency: nivel de concurrencia (1 para serial)
- metric: nombre de la métrica (ej. "p99_s", "throughput_tok_s")
- value:  número

Uso:
    from _resultslog import log_result
    log_result("compiled", "serial", 1, "p99_s", 2.11)
"""
import csv
import os
from datetime import datetime, timezone

RESULTS_FILE = os.environ.get("RESULTS_FILE", "results.csv")


def log_result(tag, test, concurrency, metric, value):
    nuevo = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["timestamp", "tag", "test", "concurrency", "metric", "value"])
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tag, test, concurrency, metric, round(float(value), 4),
        ])
