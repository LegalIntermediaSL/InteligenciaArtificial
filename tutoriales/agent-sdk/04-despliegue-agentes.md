# Despliegue de Agentes en Producción

## Arquitectura de despliegue

Un agente en producción necesita más que el bucle agéntico básico.
Requiere manejo de sesiones, streaming de eventos, observabilidad y escalado.

```
Cliente (frontend/API)
    │
    ▼
FastAPI Gateway
    ├── Autenticación y rate limiting
    ├── Gestión de sesiones
    │
    ▼
Motor del Agente
    ├── Bucle agéntico con streaming
    ├── Pool de herramientas registradas
    ├── Gestión de memoria (Redis + ChromaDB)
    │
    ▼
Observabilidad
    ├── Logging estructurado
    ├── Métricas (latencia, tokens, coste)
    └── Trazas de ejecución
```

## API FastAPI con streaming de eventos

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import asyncio
import json
import uuid
from datetime import datetime

app = FastAPI(title="Agente API", version="1.0.0")
client = anthropic.Anthropic()

# Almacén de sesiones en memoria (usar Redis en producción)
SESIONES = {}

class SolicitudChat(BaseModel):
    mensaje: str
    session_id: str | None = None
    stream: bool = True

class RespuestaChat(BaseModel):
    session_id: str
    respuesta: str
    tokens_usados: int
    herramientas_usadas: list[str]

# Registro de herramientas del agente
HERRAMIENTAS_AGENTE = [
    {"name": "buscar", "description": "Busca información en la base de conocimiento",
     "input_schema": {"type": "object",
                      "properties": {"consulta": {"type": "string"}},
                      "required": ["consulta"]}},
    {"name": "calcular", "description": "Realiza cálculos",
     "input_schema": {"type": "object",
                      "properties": {"expresion": {"type": "string"}},
                      "required": ["expresion"]}}
]

def ejecutar_herramienta(nombre: str, params: dict) -> str:
    if nombre == "buscar":
        return f"Resultados para '{params['consulta']}': [información relevante encontrada]"
    elif nombre == "calcular":
        try:
            return str(eval(params["expresion"], {"__builtins__": {}}))
        except:
            return "Error en el cálculo"
    return "Herramienta no disponible"

async def bucle_agente_streaming(session_id: str, mensaje: str):
    """Genera eventos SSE del bucle agéntico."""
    sesion = SESIONES.setdefault(session_id, {"mensajes": [], "tokens_total": 0})
    sesion["mensajes"].append({"role": "user", "content": mensaje})

    herramientas_usadas = []
    tokens_sesion = 0

    yield f"data: {json.dumps({'tipo': 'inicio', 'session_id': session_id})}\n\n"

    for iteracion in range(15):
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            tools=HERRAMIENTAS_AGENTE,
            messages=sesion["mensajes"]
        )
        tokens_sesion += resp.usage.input_tokens + resp.usage.output_tokens
        sesion["mensajes"].append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            texto_final = next((b.text for b in resp.content if hasattr(b, "text")), "")
            # Streaming carácter a carácter simulado
            for chunk in [texto_final[i:i+20] for i in range(0, len(texto_final), 20)]:
                yield f"data: {json.dumps({'tipo': 'texto', 'contenido': chunk})}\n\n"
                await asyncio.sleep(0.02)
            break

        if resp.stop_reason == "tool_use":
            resultados = []
            for bloque in resp.content:
                if bloque.type == "tool_use":
                    herramientas_usadas.append(bloque.name)
                    yield f"data: {json.dumps({'tipo': 'herramienta', 'nombre': bloque.name, 'params': bloque.input})}\n\n"
                    resultado = ejecutar_herramienta(bloque.name, bloque.input)
                    yield f"data: {json.dumps({'tipo': 'resultado_herramienta', 'resultado': resultado[:200]})}\n\n"
                    resultados.append({"type": "tool_result", "tool_use_id": bloque.id, "content": resultado})
            sesion["mensajes"].append({"role": "user", "content": resultados})

    sesion["tokens_total"] += tokens_sesion
    yield f"data: {json.dumps({'tipo': 'fin', 'tokens': tokens_sesion, 'herramientas': herramientas_usadas})}\n\n"

@app.post("/chat/stream")
async def chat_stream(solicitud: SolicitudChat):
    """Endpoint de chat con streaming SSE."""
    session_id = solicitud.session_id or str(uuid.uuid4())
    return StreamingResponse(
        bucle_agente_streaming(session_id, solicitud.mensaje),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Session-ID": session_id
        }
    )

@app.delete("/sesion/{session_id}")
async def eliminar_sesion(session_id: str):
    SESIONES.pop(session_id, None)
    return {"eliminado": True}

@app.get("/sesion/{session_id}/info")
async def info_sesion(session_id: str):
    sesion = SESIONES.get(session_id)
    if not sesion:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    return {
        "session_id": session_id,
        "turnos": len(sesion["mensajes"]) // 2,
        "tokens_total": sesion["tokens_total"]
    }
```

## Middleware de observabilidad

```python
import time
import logging
from fastapi import Request
from functools import wraps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("agente")

# Middleware de logging estructurado
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    latencia = time.perf_counter() - inicio

    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status": respuesta.status_code,
        "latencia_ms": round(latencia * 1000),
        "ip": request.client.host if request.client else "unknown"
    }))
    return respuesta

# Decorator de métricas para herramientas
def medir_herramienta(func):
    @wraps(func)
    def wrapper(nombre, params):
        inicio = time.perf_counter()
        try:
            resultado = func(nombre, params)
            latencia = time.perf_counter() - inicio
            logger.info(json.dumps({
                "evento": "herramienta_ok",
                "herramienta": nombre,
                "latencia_ms": round(latencia * 1000)
            }))
            return resultado
        except Exception as e:
            logger.error(json.dumps({
                "evento": "herramienta_error",
                "herramienta": nombre,
                "error": str(e)
            }))
            raise
    return wrapper
```

## Rate limiting por usuario

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 10, ventana_segundos: int = 60):
        self.max_requests = max_requests
        self.ventana = ventana_segundos
        self._historial: dict = defaultdict(list)

    def permitir(self, user_id: str) -> bool:
        ahora = time.time()
        historial = self._historial[user_id]
        # Limpiar entradas viejas
        self._historial[user_id] = [t for t in historial if ahora - t < self.ventana]
        if len(self._historial[user_id]) >= self.max_requests:
            return False
        self._historial[user_id].append(ahora)
        return True

    def tiempo_espera(self, user_id: str) -> int:
        historial = self._historial.get(user_id, [])
        if not historial:
            return 0
        return max(0, int(self.ventana - (time.time() - min(historial))))

limiter = RateLimiter(max_requests=20, ventana_segundos=60)

# Usar en endpoints
from fastapi import Header
@app.post("/chat")
async def chat_con_rate_limit(solicitud: SolicitudChat, x_user_id: str = Header("anonimo")):
    if not limiter.permitir(x_user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit excedido. Espera {limiter.tiempo_espera(x_user_id)}s"
        )
    # ... resto del endpoint
```

## Docker y despliegue

```dockerfile
# Dockerfile para el agente
FROM python:3.11-slim
RUN useradd -m -u 1000 agente
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=agente:agente . .
USER agente
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agente-api:
    build: .
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    ports:
      - "8000:8000"
    restart: unless-stopped
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Checklist de producción para agentes

```
Funcionalidad:
  ✓ Bucle agéntico con límite de iteraciones
  ✓ Manejo de errores en herramientas
  ✓ Timeout por herramienta
  ✓ Streaming de respuesta

Sesiones:
  ✓ Gestión de sesiones con TTL
  ✓ Compresión de historial largo
  ✓ Persistencia de memoria entre sesiones

Seguridad:
  ✓ Autenticación de usuario
  ✓ Rate limiting
  ✓ Validación de inputs
  ✓ Sandbox para herramientas peligrosas

Observabilidad:
  ✓ Logging estructurado (JSON)
  ✓ Métricas: latencia, tokens, coste por sesión
  ✓ Trazas de herramientas ejecutadas
  ✓ Alertas en degradación de calidad

Escalado:
  ✓ Stateless (sesiones en Redis)
  ✓ Multiple workers con uvicorn
  ✓ Health check + readiness
  ✓ Graceful shutdown
```

## Recursos

- [Notebook interactivo](../notebooks/agent-sdk/04-despliegue-agentes.ipynb)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Langfuse — Observabilidad LLMs](https://langfuse.com/docs)
