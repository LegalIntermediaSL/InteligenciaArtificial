# 04 — Despliegue de APIs de IA con FastAPI y Docker

> **Bloque:** Producción · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Arquitectura de una API de IA en producción](#1-arquitectura-de-una-api-de-ia-en-producción)
2. [API básica con FastAPI](#2-api-básica-con-fastapi)
3. [Streaming con FastAPI](#3-streaming-con-fastapi)
4. [Rate limiting y gestión de errores](#4-rate-limiting-y-gestión-de-errores)
5. [Dockerfile completo](#5-dockerfile-completo)
6. [docker-compose.yml con logging](#6-docker-composeyml-con-logging)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Arquitectura de una API de IA en producción

Antes de escribir código, es importante entender cómo encajan todas las piezas:

```
                    Internet
                       │
                  ┌────▼─────┐
                  │  Nginx   │  Proxy inverso, TLS, balanceo
                  └────┬─────┘
                       │
              ┌────────▼────────┐
              │   FastAPI App   │  Lógica de negocio, validación
              │  (Uvicorn)      │
              │                 │
              │  /chat          │──────► Anthropic API (Claude)
              │  /chat/stream   │          (llamadas externas)
              │  /health        │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────▼───┐  ┌─────▼────┐  ┌──▼──────┐
     │ Redis  │  │Postgres/ │  │Langfuse │
     │(caché/ │  │SQLite    │  │(traces) │
     │ límite)│  │(historial│  │         │
     └────────┘  └──────────┘  └─────────┘
```

**Componentes del tutorial:**
- **FastAPI + Uvicorn**: framework web asíncrono, ideal para I/O-bound como llamadas a APIs externas
- **Pydantic**: validación de datos de entrada/salida
- **Streaming**: respuestas en tiempo real con Server-Sent Events
- **Rate limiting**: control de abusos y gestión de costes
- **Docker**: empaquetado reproducible y despliegue consistente

---

## 2. API básica con FastAPI

```bash
pip install fastapi uvicorn anthropic python-dotenv pydantic
```

```python
# api/main.py
"""
API de IA con FastAPI — endpoint POST /chat que llama a Claude.
"""
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")

MODELO_DEFAULT = "claude-sonnet-4-6"
MAX_TOKENS_DEFAULT = 1024
MAX_HISTORIAL_MENSAJES = 20

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class Mensaje(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=50_000)


class ChatRequest(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=10_000, description="Mensaje del usuario")
    historial: list[Mensaje] = Field(default=[], description="Historial de la conversación")
    system: str = Field(
        default="Eres un asistente útil que responde en español de forma clara y concisa.",
        max_length=5_000,
        description="System prompt del asistente",
    )
    max_tokens: int = Field(default=MAX_TOKENS_DEFAULT, ge=1, le=4096)
    modelo: str = Field(default=MODELO_DEFAULT)

    model_config = {"json_schema_extra": {
        "example": {
            "mensaje": "¿Qué es el aprendizaje automático?",
            "historial": [],
            "system": "Eres un experto en IA.",
            "max_tokens": 512,
        }
    }}


class ChatResponse(BaseModel):
    respuesta: str
    tokens_entrada: int
    tokens_salida: int
    modelo: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de IA",
    description="API REST para interactuar con Claude de Anthropic",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Permitir peticiones desde el frontend (ajusta los orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Comprueba que la API está en funcionamiento."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=app.version,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Envía un mensaje a Claude y recibe la respuesta completa.

    - Acepta historial de conversación para contexto multi-turno
    - Valida la longitud del mensaje y del historial
    - Retorna tokens consumidos para monitorización de costes
    """
    # Construir el historial de mensajes
    mensajes = [{"role": m.role, "content": m.content} for m in request.historial]

    # Limitar el historial para controlar costes
    if len(mensajes) > MAX_HISTORIAL_MENSAJES:
        mensajes = mensajes[-MAX_HISTORIAL_MENSAJES:]
        logger.warning(f"Historial truncado a {MAX_HISTORIAL_MENSAJES} mensajes")

    # Añadir el mensaje actual
    mensajes.append({"role": "user", "content": request.mensaje})

    logger.info(f"Chat | modelo={request.modelo} | mensajes={len(mensajes)}")

    try:
        respuesta_api = client.messages.create(
            model=request.modelo,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=mensajes,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="API key de Anthropic inválida")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Límite de requests de Anthropic alcanzado. Reintenta en unos segundos.")
    except anthropic.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"Request inválido: {e}")
    except anthropic.APIStatusError as e:
        logger.error(f"Error de la API de Anthropic: {e.status_code} — {e.message}")
        raise HTTPException(status_code=502, detail="Error en el servicio de IA. Inténtalo de nuevo.")

    texto_respuesta = respuesta_api.content[0].text

    logger.info(
        f"Chat completado | "
        f"tokens_in={respuesta_api.usage.input_tokens} | "
        f"tokens_out={respuesta_api.usage.output_tokens}"
    )

    return ChatResponse(
        respuesta=texto_respuesta,
        tokens_entrada=respuesta_api.usage.input_tokens,
        tokens_salida=respuesta_api.usage.output_tokens,
        modelo=respuesta_api.model,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.exception_handler(Exception)
async def manejador_errores_global(request: Request, exc: Exception):
    """Captura errores no controlados para evitar exponer stack traces."""
    logger.exception(f"Error no controlado en {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor. Consulta los logs."},
    )
```

**Arrancar el servidor:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Probar con curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"mensaje": "¿Qué es el machine learning?"}'
```

---

## 3. Streaming con FastAPI

El streaming mejora la experiencia de usuario: el texto aparece progresivamente en lugar de esperar a que la respuesta completa esté lista.

```python
# api/streaming.py
"""
Endpoint de streaming con Server-Sent Events (SSE).
Añadir a main.py o importar como router.
"""
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import anthropic
import os

router = APIRouter(tags=["Streaming"])
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


class StreamRequest(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=10_000)
    system: str = Field(
        default="Eres un asistente útil que responde en español.",
        max_length=5_000,
    )
    max_tokens: int = Field(default=1024, ge=1, le=4096)


async def generar_stream(request: StreamRequest):
    """
    Generador asíncrono que produce eventos SSE a medida que Claude genera texto.

    Formato SSE:
        data: {"tipo": "texto", "contenido": "hola"}
        data: {"tipo": "fin", "tokens_entrada": 50, "tokens_salida": 120}
        data: [DONE]
    """
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.mensaje}],
        ) as stream:
            # Enviar cada fragmento de texto a medida que llega
            for fragmento in stream.text_stream:
                evento = json.dumps({"tipo": "texto", "contenido": fragmento}, ensure_ascii=False)
                yield f"data: {evento}\n\n"

            # Al finalizar, enviar las métricas de uso
            mensaje_final = stream.get_final_message()
            evento_fin = json.dumps({
                "tipo": "fin",
                "tokens_entrada": mensaje_final.usage.input_tokens,
                "tokens_salida": mensaje_final.usage.output_tokens,
                "modelo": mensaje_final.model,
            })
            yield f"data: {evento_fin}\n\n"
            yield "data: [DONE]\n\n"

    except anthropic.RateLimitError:
        error = json.dumps({"tipo": "error", "mensaje": "Límite de rate alcanzado. Reintenta."})
        yield f"data: {error}\n\n"
    except Exception as e:
        error = json.dumps({"tipo": "error", "mensaje": "Error generando respuesta."})
        yield f"data: {error}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """
    Endpoint de chat con streaming.
    La respuesta se entrega como Server-Sent Events (SSE).

    Ejemplo de consumo en JavaScript:
        const evtSource = new EventSource('/chat/stream');
        fetch('/chat/stream', {method: 'POST', body: JSON.stringify({mensaje: 'Hola'})})
            .then(res => {
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                reader.read().then(function process({done, value}) {
                    if (done) return;
                    console.log(decoder.decode(value));
                    reader.read().then(process);
                });
            });
    """
    return StreamingResponse(
        generar_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Desactivar buffering en Nginx
        },
    )


# ---------------------------------------------------------------------------
# Integrar en main.py:
# from api.streaming import router as streaming_router
# app.include_router(streaming_router)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Script de prueba — consumir el stream desde Python
# ---------------------------------------------------------------------------

def probar_stream(mensaje: str = "Explica el teorema de Pitágoras paso a paso.") -> None:
    """Prueba el endpoint de streaming directamente (sin servidor)."""
    import asyncio

    async def consumir():
        request = StreamRequest(mensaje=mensaje)
        print(f"Pregunta: {mensaje}\n")
        print("Respuesta (streaming):")
        print("-" * 40)

        async for evento in generar_stream(request):
            if evento.startswith("data: "):
                datos_str = evento[6:].strip()
                if datos_str == "[DONE]":
                    break
                try:
                    datos = json.loads(datos_str)
                    if datos["tipo"] == "texto":
                        print(datos["contenido"], end="", flush=True)
                    elif datos["tipo"] == "fin":
                        print(f"\n\nTokens: {datos['tokens_entrada']} entrada + {datos['tokens_salida']} salida")
                except json.JSONDecodeError:
                    pass

    asyncio.run(consumir())


if __name__ == "__main__":
    probar_stream()
```

---

## 4. Rate limiting y gestión de errores

El rate limiting protege la API de abusos y controla el gasto. Implementamos tanto un middleware manual como retry automático con backoff exponencial.

```bash
pip install slowapi limits
```

```python
# api/middleware.py
"""
Rate limiting, retry con backoff y gestión centralizada de errores.
"""
import time
import random
import logging
import functools
from collections import defaultdict
from datetime import datetime, timedelta

import anthropic
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("middleware")

# ---------------------------------------------------------------------------
# Rate limiting con slowapi
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)


def configurar_rate_limiting(app: FastAPI) -> None:
    """Añade rate limiting a la aplicación FastAPI."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Uso en endpoints:
# @app.post("/chat")
# @limiter.limit("20/minute")   # 20 requests por minuto por IP
# @limiter.limit("200/hour")    # 200 requests por hora por IP
# async def chat(request: Request, body: ChatRequest):
#     ...


# ---------------------------------------------------------------------------
# Rate limiting manual (sin dependencias externas)
# ---------------------------------------------------------------------------

class RateLimiterManual:
    """
    Rate limiter en memoria con ventana deslizante.
    Válido para un solo proceso; usa Redis en multi-proceso.
    """

    def __init__(self, max_requests: int = 20, ventana_segundos: int = 60):
        self.max_requests = max_requests
        self.ventana_segundos = ventana_segundos
        self._registros: dict[str, list[float]] = defaultdict(list)

    def permitir(self, clave: str) -> tuple[bool, int]:
        """
        Comprueba si la clave (IP, user_id) puede hacer una nueva request.

        Returns:
            (permitido: bool, requests_restantes: int)
        """
        ahora = time.time()
        ventana_inicio = ahora - self.ventana_segundos

        # Limpiar timestamps fuera de la ventana
        self._registros[clave] = [
            ts for ts in self._registros[clave] if ts > ventana_inicio
        ]

        count = len(self._registros[clave])
        if count >= self.max_requests:
            return False, 0

        self._registros[clave].append(ahora)
        return True, self.max_requests - count - 1


rate_limiter = RateLimiterManual(max_requests=30, ventana_segundos=60)


async def middleware_rate_limiting(request: Request, call_next):
    """Middleware de FastAPI para rate limiting manual."""
    ip = request.client.host if request.client else "desconocido"
    permitido, restantes = rate_limiter.permitir(ip)

    if not permitido:
        return JSONResponse(
            status_code=429,
            content={"detail": "Demasiadas requests. Espera un momento."},
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(restantes)
    return response


# ---------------------------------------------------------------------------
# Retry con backoff exponencial
# ---------------------------------------------------------------------------

def retry_con_backoff(
    max_intentos: int = 3,
    espera_base: float = 1.0,
    espera_maxima: float = 60.0,
    excepciones_reintentables: tuple = (
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
    ),
):
    """
    Decorador que reintenta la función en caso de errores transitorios.
    Usa backoff exponencial con jitter para evitar thundering herd.

    Uso:
        @retry_con_backoff(max_intentos=3)
        def llamar_api(...):
            ...
    """
    def decorador(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ultimo_error = None
            for intento in range(1, max_intentos + 1):
                try:
                    return func(*args, **kwargs)
                except excepciones_reintentables as e:
                    ultimo_error = e

                    # No reintentar errores de autenticación o request inválido
                    if isinstance(e, anthropic.APIStatusError):
                        if e.status_code in (400, 401, 403):
                            raise

                    if intento == max_intentos:
                        logger.error(f"Fallaron todos los intentos ({max_intentos}) para {func.__name__}: {e}")
                        raise

                    # Backoff exponencial con jitter aleatorio
                    espera = min(espera_base * (2 ** (intento - 1)), espera_maxima)
                    jitter = random.uniform(0, espera * 0.1)
                    espera_total = espera + jitter

                    logger.warning(
                        f"Intento {intento}/{max_intentos} fallado ({type(e).__name__}). "
                        f"Reintentando en {espera_total:.1f}s..."
                    )
                    time.sleep(espera_total)

            raise ultimo_error
        return wrapper
    return decorador


# ---------------------------------------------------------------------------
# Ejemplo: función con retry integrado
# ---------------------------------------------------------------------------

@retry_con_backoff(max_intentos=3, espera_base=2.0)
def llamar_claude_con_retry(
    client: anthropic.Anthropic,
    mensajes: list[dict],
    system: str,
    max_tokens: int = 512,
) -> anthropic.types.Message:
    """Llamada a Claude con reintentos automáticos en caso de errores de red o rate limit."""
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=mensajes,
    )


# ---------------------------------------------------------------------------
# Integrar middleware en main.py:
#
# from api.middleware import middleware_rate_limiting
# app.middleware("http")(middleware_rate_limiting)
# ---------------------------------------------------------------------------
```

---

## 5. Dockerfile completo

Un Dockerfile de producción con build multi-etapa para minimizar el tamaño de la imagen.

```dockerfile
# Dockerfile
# =========================================================
# Etapa 1: Builder — instalar dependencias
# =========================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================================================
# Etapa 2: Runtime — imagen final ligera
# =========================================================
FROM python:3.12-slim AS runtime

# Crear usuario no-root por seguridad
RUN addgroup --system app && adduser --system --group app

WORKDIR /app

# Copiar dependencias instaladas desde la etapa builder
COPY --from=builder /install /usr/local

# Copiar el código de la aplicación
COPY --chown=app:app api/ ./api/

# Cambiar al usuario no-root
USER app

# Variables de entorno por defecto (sobreescribir en producción)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WORKERS=2

# Exponer el puerto
EXPOSE 8000

# Healthcheck — Docker comprueba que la app responde
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de arranque
CMD uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --no-access-log
```

**requirements.txt** para la imagen:
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
anthropic==0.40.0
pydantic==2.9.0
python-dotenv==1.0.1
slowapi==0.1.9
```

**Construir y ejecutar:**
```bash
# Construir la imagen
docker build -t api-ia:latest .

# Ejecutar con variables de entorno
docker run -d \
  --name api-ia \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e WORKERS=4 \
  api-ia:latest

# Ver logs
docker logs -f api-ia

# Comprobar el healthcheck
docker inspect --format='{{.State.Health.Status}}' api-ia
```

---

## 6. docker-compose.yml con logging

Un `docker-compose.yml` completo que orquesta la API con un servicio de logging centralizado.

```yaml
# docker-compose.yml
version: "3.9"

services:
  # -------------------------------------------------------
  # Servicio principal: API de IA
  # -------------------------------------------------------
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    container_name: api-ia
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - PORT=8000
      - WORKERS=2
      - LOG_LEVEL=info
    env_file:
      - .env
    volumes:
      # Montar logs para que el agente de logging los lea
      - ./logs:/app/logs
    depends_on:
      - vector
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    networks:
      - ia-net
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # -------------------------------------------------------
  # Servicio de logging: Vector
  # Recoge logs de todos los contenedores y los envía
  # a un destino (fichero, Elasticsearch, Loki, etc.)
  # -------------------------------------------------------
  vector:
    image: timberio/vector:0.41.X-alpine
    container_name: vector-logs
    restart: unless-stopped
    volumes:
      - ./vector.toml:/etc/vector/vector.toml:ro
      - ./logs:/logs
      # Acceso a los logs de Docker del host
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - ia-net
    depends_on: []

networks:
  ia-net:
    driver: bridge

volumes:
  logs:
```

**vector.toml** — configuración del agente de logging:

```toml
# vector.toml
# Vector recoge los logs de Docker y los guarda en ficheros JSONL

[sources.docker_logs]
type = "docker_logs"
# Filtrar solo los contenedores de este proyecto
include_labels = { "com.docker.compose.project" = "api-ia" }

[transforms.parsear_json]
type = "remap"
inputs = ["docker_logs"]
source = '''
# Intentar parsear el mensaje como JSON (nuestros logs son JSONL)
parsed, err = parse_json(.message)
if err == null {
  . = merge(., parsed)
}
.contenedor = .container_name
'''

[sinks.archivo_general]
type = "file"
inputs = ["parsear_json"]
path = "/logs/api-%Y-%m-%d.log"
encoding.codec = "json"

[sinks.archivo_errores]
type = "file"
inputs = ["parsear_json"]
path = "/logs/errores-%Y-%m-%d.log"
encoding.codec = "json"

  [sinks.archivo_errores.conditions]
  type = "vrl"
  source = '.level == "error" || .level == "ERROR"'

# Opcional: enviar métricas a Prometheus
# [sinks.prometheus]
# type = "prometheus_exporter"
# inputs = ["parsear_json"]
# address = "0.0.0.0:9598"
```

**Arrancar todo el stack:**
```bash
# Copiar variables de entorno
cp .env.example .env
# Editar .env con tus claves

# Levantar todos los servicios
docker compose up -d

# Ver logs de la API
docker compose logs -f api

# Ver estado de los contenedores
docker compose ps

# Detener todo
docker compose down
```

**.env.example:**
```
ANTHROPIC_API_KEY=sk-ant-TU_CLAVE_AQUI
OPENAI_API_KEY=sk-TU_CLAVE_AQUI
PORT=8000
WORKERS=2
LOG_LEVEL=info
```

**Estructura de archivos del proyecto:**
```
mi-api-ia/
├── api/
│   ├── __init__.py
│   ├── main.py          # App principal FastAPI
│   ├── streaming.py     # Endpoint de streaming
│   └── middleware.py    # Rate limiting y errores
├── logs/                # Volumen para logs (creado por Docker)
├── .env                 # Variables de entorno (no subir a Git)
├── .env.example         # Plantilla pública
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── vector.toml
```

---

## 7. Extensiones sugeridas

- **Autenticación**: añadir API keys propias con `fastapi-security` o JWT para identificar usuarios y aplicar rate limiting por usuario en lugar de por IP.
- **Base de datos**: guardar el historial de conversaciones en PostgreSQL con SQLAlchemy para que los usuarios retomen conversaciones entre sesiones.
- **Caché de respuestas**: usar Redis para cachear respuestas a preguntas frecuentes e idénticas (reducción de costes del 20-40% en muchas aplicaciones).
- **Nginx como proxy**: añadir Nginx al `docker-compose.yml` para TLS (HTTPS), compresión gzip y protección adicional.
- **Kubernetes**: convertir el `docker-compose.yml` en manifiestos de Kubernetes con `kompose convert` para despliegues a mayor escala.
- **CI/CD**: pipeline de GitHub Actions que construye la imagen Docker, ejecuta los tests y despliega automáticamente al hacer push a `main`.

---

**Siguiente:** [Volver al índice del bloque](./README.md)
