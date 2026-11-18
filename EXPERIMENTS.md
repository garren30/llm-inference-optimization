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

## Experimento 2b — AWQ como desbloqueo de capacidad (6 GB)

**Propósito:** usar la VRAM que ahorra AWQ para correr un modelo más grande del
que cabe en fp16, en la misma tarjeta de 6 GB.

### Qué cabe y qué no (gpu-mem-util 0.80)
| modelo   | pesos    | KV cache       | ¿corre? |
|----------|----------|----------------|---------|
| 7B-AWQ   | ~4.5 GiB | negativo       | ❌ techo de la tarjeta |
| 3B-fp16  | 5.79 GiB | **−0.73 GiB**  | ❌ "No available memory for the cache blocks" |
| 3B-AWQ   | 1.95 GiB | 1.57 GiB       | ✅ 11.16× concurrencia |

### 3B-AWQ corriendo
| test                      | valor |
|---------------------------|-------|
| p50 / p99                 | 1.67 s / 1.92 s |
| tok/s seq                 | 75.6 |
| tok/s @ conc 1/4/8/16/32  | 76 / 298 / 577 / 1076 / 1579 |

### Resultado
Un 3B en fp16 (pesos 5.79 GiB) deja el KV cache en **−0.73 GiB** → no arranca.
En AWQ (1.95 GiB) corre con holgura. **La cuantización duplicó el tamaño de
modelo desplegable en hardware fijo.**

### La joya: 3B-AWQ vs 1.5B-fp16
| | 1.5B-fp16 | 3B-AWQ |
|---|---|---|
| parámetros    | 1.5B      | 2× (3B)  |
| pesos en VRAM | 2.98 GiB  | 1.95 GiB |
| tok/s seq     | 55.6      | 75.6     |
| p50           | 2.29 s    | 1.67 s   |

El 3B-AWQ es **más grande y más rápido** que el 1.5B-fp16. ¿Por qué un modelo
mayor es más rápido? Porque el decode es memory-bound: lo que manda es los
**bytes de pesos leídos por token**, no el nº de parámetros — y el 3B en 4-bit
(1.95 GiB) son menos bytes que el 1.5B en 16-bit (2.98 GiB).

### Nota de hardware
`--gpu-memory-utilization` es fracción de la VRAM **total** (6 GB), no de la
libre. Con ~1 GiB tomado por la pantalla (4.91 libres), el máximo usable es
~0.80; a 0.92 vLLM ni arranca (pide 5.52 > 4.91 libres).

---

## Experimento 3 — Prefix caching

**Propósito:** medir cuánto ahorra reusar el KV de un prefijo compartido entre
peticiones (system prompts, RAG, few-shot, chat multi-turno). Ataca el PREFILL,
así que la métrica es TTFT (time-to-first-token), medido por streaming.

**Workload:** prefijo compartido ~2k tokens (fijo) + pregunta corta que varía.
Mismo modelo (awq-1.5b), con prefix caching ON (default) vs OFF
(`--no-enable-prefix-caching`).

| config     | TTFT media | TTFT p50 | hit rate |
|------------|------------|----------|----------|
| prefix-off | 241.9 ms   | 241.4 ms | (off)    |
| prefix-on  | 28.3 ms    | 25.6 ms  | **94.6%** |

**Resultado: TTFT 9.4× más rápido** (241 → 26 ms). `prefix-off` queda clavado
en ~241 ms porque re-prefillea los ~2k tokens en cada petición; `prefix-on` solo
prefillea la pregunta corta y reusa el resto.

**Corroboración doble (logs del servidor):**
- `Prefix cache hit rate: 94.6%` en ON; el campo desaparece en OFF.
- `Avg prompt throughput`: ~1330–1775 tok/s en OFF (prefillea 2k tok cada vez)
  vs ~169 tok/s en ON (94.6% hits) → la GPU hizo **~10× menos prefill**.

**Cuándo aplica (honesto):** el 9.4× es para un workload favorable — prefijo
largo y muy reusado. El efecto escala con *largo del prefijo × hit rate*; con
prompts únicos no ayuda. En producción brilla porque system prompts, RAG,
agentes y chat multi-turno tienen exactamente esa forma.

---

## Resumen — tres partes del stack de latencia
| experimento | ataca | resultado |
|-------------|-------|-----------|
| 1. CUDA graphs | overhead de kernels en decode | +5.8% throughput, gratis |
| 2. AWQ 4-bit | memoria de pesos | 2.3× latencia, 2.7× capacidad, 3B donde no cabía |
| 3. prefix caching | reuso de prefill | 9.4× TTFT en workloads con prefijo compartido |

## Pendiente
- Peldaño 4 (repo aparte): la plataforma — observabilidad, autoscaling, CI/CD.