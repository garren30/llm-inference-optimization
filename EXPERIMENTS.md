# Bitácora de experimentos — optimización de inferencia

Cada experimento cambia **una sola variable** y mide contra la línea base.
Todo se guarda en `results.csv` (tag por config) y se grafica con `plot_results.py`.

## Línea base (peldaño 2, Qwen2.5-1.5B-Instruct, RTX 4050)
- Secuencial: ~53.7 tok/s · p99 ~2.45 s · ~18.4 ms/token (decode)
- Paralelo: techo ~2250 tok/s · satura ~64 concurrentes

---

## Experimento 1 — compile on/off (quitar `--enforce-eager`)

**Hipótesis:** quitar `--enforce-eager` habilita torch.compile + CUDA graphs,
que recortan el overhead de lanzar kernels por paso de decode. Debería bajar la
latencia secuencial (single-request) más que el throughput a alta concurrencia.

**Costo esperado:** arranque más lento (compila en el warmup) y algo más de VRAM
por la captura de CUDA graphs.

### Pasos

1. **Config A (baseline, eager).** Terminal 1:
   ```bash
   bash serve_eager.sh
   ```
   Espera a que diga que está escuchando en el puerto 8000. Terminal 2:
   ```bash
   python bench_serial.py   --tag eager
   python bench_parallel.py --tag eager
   ```
   Corta el servidor (Ctrl-C en terminal 1).

2. **Config B (compiled).** Terminal 1:
   ```bash
   bash serve_compiled.sh
   ```
   El primer arranque tarda más (está compilando/capturando). Terminal 2:
   ```bash
   python bench_serial.py   --tag compiled
   python bench_parallel.py --tag compiled
   ```

3. **Gráficas:**
   ```bash
   pip install matplotlib
   python plot_results.py
   ```

### Si da OOM al arrancar la config B
La RTX 4050 va justa. Baja `--gpu-memory-utilization` a 0.75 (o `--max-model-len`
a 2048) **en AMBOS** `serve_*.sh` y repite las dos mediciones, para que la
comparación siga siendo justa. Anótalo aquí.

### Resultados (llenar)

| config   | p50 (s) | p99 (s) | tok/s seq | mejor tok/s (conc) | VRAM | arranque |
|----------|---------|---------|-----------|--------------------|------|----------|
| eager    |         |         |           |                    |      |          |
| compiled |         |         |           |                    |      |          |

**Observaciones:** _(¿dónde ganó más, latencia o throughput? ¿costo en VRAM/arranque valió la pena?)_

---

## Experimento 2 — Cuantización AWQ  (núcleo del peldaño)
_Pendiente. Servir `Qwen2.5-1.5B-Instruct-AWQ` (o 7B-AWQ si entra) y medir
latencia/throughput/**VRAM** antes-después. Mismo benchmark, `--tag awq`._

## Experimento 3 — Prefix/KV caching · model routing
_Pendiente._
