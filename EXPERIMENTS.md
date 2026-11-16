# Bitácora de experimentos — optimización de inferencia

Cada experimento cambia **una sola variable** y mide contra la línea base.
Todo se guarda en `results.csv` (tag por config) y se grafica con `plot_results.py`.
Métrica primaria: throughput (normaliza por tokens). Latencia con `ignore_eos`
(128 tok fijos) para que sea comparable directo.

## Línea base (peldaño 2, Qwen2.5-1.5B-Instruct fp16, RTX 4050)
- Secuencial: ~53.7 tok/s · p99 ~2.45 s · ~18.4 ms/token (decode)
- Paralelo: techo ~2250 tok/s · satura ~64 concurrentes

---

## Experimento 1 — compile on/off (`--enforce-eager`)

**Propósito:** medir si habilitar CUDA graphs + torch.compile (quitando
`--enforce-eager`) compra velocidad, y a qué costo. También calibra el método
tag→CSV→gráfica que reusan los demás experimentos.

**Mecanismo:** en decode con batch=1 la GPU termina cada kernel rápido y espera
a que la CPU lance el siguiente (memory-bound). Los CUDA graphs graban la
secuencia de kernels una vez y la reproducen en un solo lanzamiento → menos
overhead de dispatch CPU→GPU.

| config   | p50 (s) | p99 (s) | tok/s seq | tok/s conc32 |
|----------|---------|---------|-----------|--------------|
| eager    | 2.37    | 2.69    | 50.2      | 1241.7       |
| compiled | 2.16    | 2.50    | 53.1      | 1297.6       |

**Resultado:** +5.8% throughput secuencial, +4.5–8.7% paralelo, **sin costo de
VRAM**. Victoria gratis ~6%. El único precio: arranque más lento (captura de
grafos en el warmup). Compiled queda como baseline.

**Lección de medición (confounder):** la caída de p50 (~9%) estaba inflada —
con `temperature 0.7` cada corrida generó distinto nº de tokens, así que parte
de su "menor latencia" era output más corto. El throughput normaliza por tokens
→ cifra honesta. Fix aplicado desde aquí: `ignore_eos` fija la salida.

**Por qué modesto:** 1.5B memory-bound; el overhead de lanzamiento era real pero
no dominante. Escala en modelos mayores.

---

## Experimento 2a — AWQ 4-bit (mismo modelo, 1.5B)

**Propósito:** aislar el efecto de cuantizar a 4 bits en el mismo modelo.
Benchmark con `ignore_eos`. Servido SIN `--quantization awq` para que vLLM use
el kernel Marlin (rápido, autodetectado en Ada; log: `Using MarlinLinearKernel`).

### Velocidad
| config    | p50 (s) | p99 (s) | tok/s seq | tok/s conc32 |
|-----------|---------|---------|-----------|--------------|
| fp16-1.5b | 2.29    | 2.43    | 55.6      | 1473.6       |
| awq-1.5b  | 0.98    | 0.99    | 130.2     | 2710.0       |

### VRAM (del log de arranque de vLLM)
| config    | pesos    | KV cache libre | concurrencia máx (4096 tok) |
|-----------|----------|----------------|------------------------------|
| fp16-1.5b | 2.98 GiB | 1.01 GiB       | 9.19x                        |
| awq-1.5b  | 1.1 GiB  | 2.72 GiB       | 24.84x                       |

**Resultado:** mismo modelo, 4-bit → **2.34x latencia + 2.7x capacidad**, sin
perder calidad. La VRAM ahorrada en pesos se vuelve KV cache → de ~9 a ~25
peticiones concurrentes de 4K tokens en la misma tarjeta de 6 GB.

**Por qué acelera (no solo ahorra):** a batch=1 el decode es memory-bound
leyendo pesos; pesos 4-bit = menos bytes/token = más rápido. La ventaja se
encoge con la concurrencia (2.33x @conc1 → 1.84x @conc32): a batch alto se
vuelve compute-bound y el des-cuantizar empieza a pesar. Memory-bound →
compute-bound, medido.

**Nota:** pesos 2.7x (no 4x teórico) porque embeddings/layernorms/lm_head quedan
en fp16 + metadatos de grupo AWQ (escalas/ceros, group_size 128).

---

## Experimento 2b — 7B-AWQ (el desbloqueo) — PENDIENTE
Servir `Qwen2.5-7B-Instruct-AWQ` en la 4050 de 6 GB. Objetivo: correr un 7B
donde el 7B-fp16 (~14 GB) daría OOM. Muy apretado; probar `--max-model-len 2048
--gpu-memory-utilization 0.92`, bajar a 1024 si hace falta. Medir latencia/
throughput/VRAM. Que no quepa también es hallazgo (techo de la tarjeta).

## Experimento 3 — prefix/KV caching · model routing — PENDIENTE