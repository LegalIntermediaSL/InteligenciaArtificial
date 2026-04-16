# 04 — Despliegue de Modelos Fine-tuneados

> **Bloque:** Fine-tuning avanzado · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Opciones de serving para LLMs propios](#1-opciones-de-serving-para-llms-propios)
2. [vLLM: servidor OpenAI-compatible](#2-vllm-servidor-openai-compatible)
3. [Text Generation Inference (TGI)](#3-text-generation-inference-tgi)
4. [Cuantización para serving](#4-cuantización-para-serving)
5. [API REST con FastAPI sobre vLLM](#5-api-rest-con-fastapi-sobre-vllm)
6. [Docker y docker-compose](#6-docker-y-docker-compose)
7. [Benchmarks de rendimiento](#7-benchmarks-de-rendimiento)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Opciones de serving para LLMs propios

Después de hacer fine-tuning tienes un modelo en disco (HuggingFace format). Para exponerlo en producción necesitas un servidor de inferencia que gestione:

- **Batching**: agrupar múltiples peticiones para aprovechar la GPU
- **Streaming**: enviar tokens al cliente conforme se generan
- **Concurrencia**: atender varias peticiones simultáneas
- **Gestión de memoria KV-cache**: reutilizar cálculos de atención entre tokens

Las dos opciones más utilizadas son:

| Herramienta | Empresa | Puntos fuertes | Limitaciones |
|---|---|---|---|
| **vLLM** | UC Berkeley | PagedAttention, OpenAI-compatible, muy rápido | Requiere GPU NVIDIA con CUDA |
| **TGI** | HuggingFace | Soporte amplio de modelos, Flash Attention 2, Rust | Más complejo de configurar |
| **Ollama** | Ollama | Fácil, multi-plataforma (CPU/GPU) | Menor throughput que vLLM en GPU dedicada |
| **llama.cpp** | Gerganov | CPU, GGUF, embedded | No apto para alta concurrencia |

Para producción con GPU dedicada: **vLLM** es el estándar de facto.

---

## 2. vLLM: servidor OpenAI-compatible

### Instalación

```bash
# Requiere CUDA 12.1+ y GPU NVIDIA
pip install vllm

# Verificar instalación
python -c "import vllm; print(vllm.__version__)"
```

### Servir un modelo fine-tuneado

```bash
# Modelo en disco local
python -m vllm.entrypoints.openai.api_server \
    --model ./mi-modelo-finetuneado \
    --served-model-name mi-modelo \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --dtype bfloat16

# Modelo desde HuggingFace Hub
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3-8b \
    --host 0.0.0.0 \
    --port 8000
```

### Consumir el servidor con el cliente OpenAI

```python
from openai import OpenAI

# El servidor vLLM es compatible con el SDK de OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key-needed"  # vLLM no requiere autenticación por defecto
)

# Completions estándar
response = client.chat.completions.create(
    model="mi-modelo",
    messages=[
        {"role": "system", "content": "Eres un asistente experto en contratos legales."},
        {"role": "user", "content": "Resume los puntos clave de este contrato: ..."}
    ],
    max_tokens=512,
    temperature=0.3
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="mi-modelo",
    messages=[{"role": "user", "content": "Explica la cláusula de responsabilidad limitada."}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Parámetros clave de vLLM

```bash
# Para modelos grandes con cuantización
python -m vllm.entrypoints.openai.api_server \
    --model ./mi-modelo \
    --quantization awq \              # o gptq, bitsandbytes
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \   # % de VRAM a usar (default: 0.90)
    --tensor-parallel-size 2 \        # multi-GPU (requiere múltiples GPUs)
    --max-num-seqs 256 \              # máximo de secuencias en paralelo
    --disable-log-requests            # suprimir logs de cada petición en prod
```

---

## 3. Text Generation Inference (TGI)

TGI es el servidor de HuggingFace, escrito en Rust con backends en Python. Es especialmente bueno para modelos de HuggingFace Hub y tiene soporte nativo para Flash Attention 2.

### Instalación vía Docker (recomendado)

```bash
# Lanzar TGI para un modelo de HF Hub
docker run --gpus all \
    -p 8080:80 \
    -v $HOME/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id mistralai/Mistral-7B-Instruct-v0.3 \
    --max-input-length 4096 \
    --max-total-tokens 8192 \
    --num-shard 1

# Para modelo local fine-tuneado (montar volumen)
docker run --gpus all \
    -p 8080:80 \
    -v /ruta/a/mi-modelo:/model \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /model
```

### Consumir TGI

```python
import requests

# TGI expone una API propia (no OpenAI-compatible por defecto)
def generar_texto(prompt: str, max_tokens: int = 256) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.3,
            "do_sample": True,
            "return_full_text": False
        }
    }
    response = requests.post(
        "http://localhost:8080/generate",
        json=payload
    )
    response.raise_for_status()
    return response.json()["generated_text"]

# Streaming con TGI
def generar_stream(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.3},
        "stream": True
    }
    with requests.post(
        "http://localhost:8080/generate_stream",
        json=payload,
        stream=True
    ) as r:
        for line in r.iter_lines():
            if line:
                import json
                data = json.loads(line.decode("utf-8").removeprefix("data:").strip())
                if token := data.get("token", {}).get("text"):
                    print(token, end="", flush=True)
```

---

## 4. Cuantización para serving

La cuantización reduce el tamaño del modelo y acelera la inferencia a costa de una pequeña pérdida de precisión.

### Comparativa de formatos

| Formato | Bits | Reducción de memoria | Pérdida de calidad | Compatibilidad |
|---|---|---|---|---|
| **bfloat16** | 16 | baseline | ninguna | vLLM, TGI, Transformers |
| **GPTQ** | 4 | ~75% | mínima | vLLM, TGI, AutoGPTQ |
| **AWQ** | 4 | ~75% | mínima (mejor que GPTQ) | vLLM, TGI, AutoAWQ |
| **GGUF** | 2-8 | variable | variable | llama.cpp, Ollama |
| **bitsandbytes NF4** | 4 | ~75% | mínima | solo Transformers (no vLLM) |

### Cuantizar con AWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Cargar modelo fine-tuneado en bfloat16
model_path = "./mi-modelo-finetuneado"
quant_path = "./mi-modelo-awq-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    use_cache=False
)

# Dataset de calibración (pequeño, ~128 samples)
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Cuantizar (tarda ~20-40 min en Llama 7B)
model.quantize(tokenizer, quant_config=quant_config)

# Guardar modelo cuantizado
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Modelo AWQ guardado en {quant_path}")
```

### Cuantizar con GPTQ

```python
from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM

model_path = "./mi-modelo-finetuneado"

# Configuración GPTQ
gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",           # Dataset de calibración
    tokenizer=model_path,
    group_size=128,
    desc_act=False
)

# Cargar y cuantizar
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=gptq_config,
    device_map="auto"
)

# Guardar
model.save_pretrained("./mi-modelo-gptq-4bit")
tokenizer.save_pretrained("./mi-modelo-gptq-4bit")
```

### Servir modelo cuantizado con vLLM

```bash
# AWQ
python -m vllm.entrypoints.openai.api_server \
    --model ./mi-modelo-awq-4bit \
    --quantization awq \
    --max-model-len 4096

# GPTQ
python -m vllm.entrypoints.openai.api_server \
    --model ./mi-modelo-gptq-4bit \
    --quantization gptq \
    --max-model-len 4096
```

---

## 5. API REST con FastAPI sobre vLLM

vLLM ya expone una API OpenAI-compatible, pero a veces necesitas añadir autenticación, logging, rate limiting o transformaciones. Para eso envuelves vLLM con FastAPI:

```python
# server.py
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

VLLM_URL = "http://localhost:8000/v1"
API_KEYS = {"sk-prod-abc123", "sk-dev-xyz789"}  # En prod: cargar desde env/DB


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=VLLM_URL, timeout=120.0)
    yield
    await app.state.client.aclose()


app = FastAPI(title="Mi LLM API", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "mi-modelo"
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


def verify_api_key(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Formato de autorización inválido")
    key = authorization.removeprefix("Bearer ").strip()
    if key not in API_KEYS:
        raise HTTPException(status_code=403, detail="API key no válida")
    return key


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, api_key: str = Header(None, alias="authorization")):
    # Validar API key
    if api_key:
        verify_api_key(api_key)

    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    payload = {
        "model": request.model,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }

    if request.stream:
        async def stream_generator() -> AsyncGenerator[str, None]:
            async with app.state.client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Sin streaming
    resp = await app.state.client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    result = resp.json()

    elapsed = time.time() - start
    print(f"[{request_id}] tokens={result['usage']['total_tokens']} time={elapsed:.2f}s")

    return result


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
```

```bash
# Lanzar FastAPI junto a vLLM
uvicorn server:app --host 0.0.0.0 --port 9000 --workers 2
```

---

## 6. Docker y docker-compose

### Dockerfile para vLLM

```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:latest

# Copiar modelo fine-tuneado al contenedor
# (alternativa: montar como volumen)
COPY ./mi-modelo-awq-4bit /models/mi-modelo

ENV MODEL_PATH=/models/mi-modelo
ENV MODEL_NAME=mi-modelo
ENV MAX_MODEL_LEN=4096
ENV PORT=8000

EXPOSE 8000

CMD python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --quantization awq \
    --max-model-len $MAX_MODEL_LEN \
    --host 0.0.0.0 \
    --port $PORT
```

### docker-compose.yml completo

```yaml
# docker-compose.yml
version: "3.9"

services:
  vllm:
    build:
      context: .
      dockerfile: Dockerfile.vllm
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    ports:
      - "8000:8000"
    volumes:
      - ./mi-modelo-awq-4bit:/models/mi-modelo:ro  # montar en read-only
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "9000:9000"
    environment:
      - VLLM_URL=http://vllm:8000/v1
    depends_on:
      vllm:
        condition: service_healthy
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - api
    restart: unless-stopped
```

```bash
# Construir y lanzar
docker-compose up -d

# Ver logs
docker-compose logs -f vllm

# Escalar la API (sin escalar vLLM — comparten GPU)
docker-compose up -d --scale api=3
```

---

## 7. Benchmarks de rendimiento

Antes de ir a producción, mide el rendimiento para dimensionar correctamente la infraestructura.

```python
# benchmark.py
import asyncio
import time
import statistics
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="no-key")

PROMPT = "Explica en detalle el concepto de atención multi-cabeza en transformers."
MODEL = "mi-modelo"
MAX_TOKENS = 256


async def single_request(semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        start = time.time()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.time() - start
        usage = response.usage
        return {
            "latency": elapsed,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tokens_per_second": usage.completion_tokens / elapsed,
        }


async def benchmark(concurrency: int = 10, total_requests: int = 50):
    semaphore = asyncio.Semaphore(concurrency)
    print(f"Benchmark: {total_requests} peticiones, concurrencia={concurrency}")

    start = time.time()
    tasks = [single_request(semaphore) for _ in range(total_requests)]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start

    latencies = [r["latency"] for r in results]
    tps = [r["tokens_per_second"] for r in results]

    print(f"\n{'='*50}")
    print(f"Total tiempo:      {total_time:.2f}s")
    print(f"Throughput:        {total_requests / total_time:.1f} req/s")
    print(f"Latencia P50:      {statistics.median(latencies):.2f}s")
    print(f"Latencia P95:      {sorted(latencies)[int(0.95 * len(latencies))]:.2f}s")
    print(f"Tokens/s (media):  {statistics.mean(tps):.1f}")
    print(f"Tokens/s (P50):    {statistics.median(tps):.1f}")


if __name__ == "__main__":
    asyncio.run(benchmark(concurrency=10, total_requests=50))
```

### Resultados de referencia (Llama 3 8B AWQ, A10G 24GB)

| Concurrencia | Throughput (req/s) | Latencia P50 | Latencia P95 | Tokens/s |
|---|---|---|---|---|
| 1 | 1.2 | 0.8s | 1.1s | 320 |
| 4 | 3.8 | 1.0s | 1.5s | 1,200 |
| 8 | 6.1 | 1.3s | 2.1s | 1,950 |
| 16 | 8.4 | 1.9s | 3.2s | 2,700 |
| 32 | 9.2 | 3.4s | 5.8s | 2,950 |

> **Regla práctica:** vLLM con PagedAttention escala bien hasta ~16 peticiones concurrentes en una sola GPU. A partir de ahí, el beneficio marginal disminuye y la latencia P95 sube considerablemente.

---

## 8. Extensiones sugeridas

- **Múltiples GPUs**: `--tensor-parallel-size 4` en vLLM para modelos 70B+
- **LoRA adapter switching**: vLLM soporta cargar múltiples adaptadores LoRA sobre un modelo base (`--enable-lora`)
- **Caché de prefijos KV**: `--enable-prefix-caching` para prompts de sistema repetidos
- **Monitorización**: integrar vLLM con Prometheus + Grafana (vLLM expone `/metrics`)
- **Escalado automático**: Kubernetes + KEDA para autoescalar según cola de peticiones

---

**Anterior:** [03 — Evaluación de modelos fine-tuneados](./03-evaluacion-modelos-finetuneados.md) · **Siguiente bloque:** [Bloque 15 — IA Responsable](../ia-responsable/)
