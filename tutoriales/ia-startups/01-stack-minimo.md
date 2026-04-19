# Stack Mínimo de IA para Startups

## El error más común: sobreingeniería en el día 1

La mayoría de startups de IA fracasan por construir demasiado antes de validar,
no por elegir la tecnología incorrecta. El stack mínimo viable existe para que
salgas al mercado en semanas, no meses.

```
STACK MÍNIMO (día 1 → primeros 100 clientes)
──────────────────────────────────────────────
API de LLM     → Anthropic Claude (claude-haiku-4-5 para vol, sonnet para calidad)
Backend        → FastAPI (Python) o Next.js API routes
Base de datos  → PostgreSQL + pgvector (si necesitas RAG)
Auth           → Clerk o Auth0 (no construyas auth propio)
Pagos          → Stripe (desde el día 1, aunque sea beta)
Hosting        → Railway o Render (más barato que AWS hasta 100k€ ARR)
Monitorización → Langfuse (gratuito hasta 50k events/mes)
```

## Árbol de decisión: ¿qué LLM usar?

```python
import anthropic

client = anthropic.Anthropic()

def elegir_modelo(
    requiere_razonamiento_complejo: bool,
    volumen_diario_llamadas: int,
    presupuesto_mensual_usd: float
) -> dict:
    """
    Guía de selección de modelo para startups.
    Precios aproximados por 1M tokens (input/output combinado):
      - claude-haiku-4-5:   ~$0.40 / $2.00 por 1M tokens
      - claude-sonnet-4-6:  ~$3.00 / $15.00 por 1M tokens
      - claude-opus-4-7:    ~$15.00 / $75.00 por 1M tokens
    """

    # Estimación de tokens por llamada (ajustar a tu caso)
    tokens_por_llamada = 1500  # ~1k input + ~500 output típico
    tokens_diarios = volumen_diario_llamadas * tokens_por_llamada
    tokens_mensuales = tokens_diarios * 30

    coste_haiku_mensual = (tokens_mensuales / 1_000_000) * 1.20
    coste_sonnet_mensual = (tokens_mensuales / 1_000_000) * 9.00

    recomendacion = {}

    if not requiere_razonamiento_complejo and coste_haiku_mensual < presupuesto_mensual_usd:
        recomendacion = {
            "modelo": "claude-haiku-4-5-20251001",
            "razon": "Alta velocidad, bajo coste, suficiente para tareas estructuradas",
            "coste_estimado_mensual_usd": round(coste_haiku_mensual, 2)
        }
    elif coste_sonnet_mensual < presupuesto_mensual_usd:
        recomendacion = {
            "modelo": "claude-sonnet-4-6",
            "razon": "Razonamiento avanzado con coste razonable para startups",
            "coste_estimado_mensual_usd": round(coste_sonnet_mensual, 2)
        }
    else:
        recomendacion = {
            "modelo": "claude-haiku-4-5-20251001",
            "razon": "Presupuesto limitado: Haiku + optimización de prompts es lo correcto",
            "coste_estimado_mensual_usd": round(coste_haiku_mensual, 2),
            "alerta": f"Con {volumen_diario_llamadas} llamadas/día, Sonnet cuesta {round(coste_sonnet_mensual, 0)}$/mes"
        }

    recomendacion["tokens_mensuales_estimados"] = tokens_mensuales
    return recomendacion


# Ejemplo: startup en fase seed con 500 usuarios activos
config = elegir_modelo(
    requiere_razonamiento_complejo=False,
    volumen_diario_llamadas=2000,
    presupuesto_mensual_usd=500
)
for k, v in config.items():
    print(f"  {k}: {v}")
```

## Arquitectura mínima de un producto de IA

```
USUARIO
  │
  ▼
Next.js / React (frontend)
  │
  ├─ /api/chat          → FastAPI backend
  │     │
  │     ├─ Rate limiter (por usuario)
  │     ├─ Auth middleware
  │     │
  │     ▼
  │   Motor de IA
  │     ├─ System prompt con contexto del usuario
  │     ├─ Claude API (streaming)
  │     └─ Tool use si necesitas datos externos
  │
  ├─ PostgreSQL
  │     ├─ Usuarios y suscripciones
  │     ├─ Historial de conversaciones
  │     └─ pgvector (si RAG)
  │
  └─ Langfuse (observabilidad)
        ├─ Trazas de cada llamada
        ├─ Métricas de coste
        └─ Feedback de usuarios
```

## FastAPI mínimo con autenticación y logging

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import json
import time
import uuid

app = FastAPI()
client = anthropic.Anthropic()

# --- Modelos ---
class MensajeChat(BaseModel):
    mensaje: str
    session_id: str | None = None

# --- Auth simplificada (en producción: JWT + base de datos) ---
API_KEYS_VALIDAS = {"test_key_123", "demo_key_456"}  # en producción: consultar DB

def verificar_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS_VALIDAS:
        raise HTTPException(status_code=401, detail="API key inválida")
    return x_api_key

# --- Logging mínimo ---
def log_llamada(user_id: str, tokens: int, latencia_ms: float, modelo: str):
    print(json.dumps({
        "ts": time.time(),
        "user_id": user_id,
        "tokens": tokens,
        "latencia_ms": round(latencia_ms),
        "modelo": modelo
    }))

# --- Endpoint principal ---
SYSTEM_PROMPT = """Eres un asistente de [nombre de tu startup].
Ayudas a los usuarios con [tu propuesta de valor].
Sé conciso y útil. Si no sabes algo, dilo claramente."""

@app.post("/api/chat")
async def chat(
    solicitud: MensajeChat,
    api_key: str = Depends(verificar_api_key)
):
    session_id = solicitud.session_id or str(uuid.uuid4())
    inicio = time.perf_counter()

    async def stream_respuesta():
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": solicitud.mensaje}]
        ) as stream:
            for texto in stream.text_stream:
                yield f"data: {json.dumps({'text': texto})}\n\n"

            # Al terminar, loguear métricas
            msg_final = stream.get_final_message()
            latencia = (time.perf_counter() - inicio) * 1000
            log_llamada(
                api_key, msg_final.usage.input_tokens + msg_final.usage.output_tokens,
                latencia, "claude-haiku-4-5-20251001"
            )
            yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"

    return StreamingResponse(stream_respuesta(), media_type="text/event-stream")

@app.get("/health")
def health():
    return {"status": "ok"}
```

## Qué NO construir en fase pre-PMF

```
NO construyas (todavía):
  ✗ Tu propio modelo fine-tuneado — caro, lento, difícil de mantener
  ✗ Infraestructura de ML propia — usa las APIs
  ✗ Sistema de autenticación propio — usa Clerk/Auth0
  ✗ Pipeline de datos complejo — empieza con PostgreSQL simple
  ✗ Múltiples modelos de diferentes proveedores — elige uno y domínalo

SÍ construye desde el día 1:
  ✓ Logging de TODAS las llamadas a la API
  ✓ Feedback de usuarios (thumbs up/down) por respuesta
  ✓ Rate limiting por usuario desde el primer día
  ✓ Métricas de coste por usuario (para saber cuánto cobrar)
  ✓ System prompt versionado (trátalo como código)
```

## Estructura de proyecto recomendada

```
mi-startup-ia/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── prompts/
│   │   ├── v1.0.txt         # system prompt versionado
│   │   └── v1.1.txt
│   ├── tools/               # herramientas del agente
│   └── requirements.txt
├── frontend/
│   └── (Next.js app)
├── .env.example             # nunca commitear .env real
├── docker-compose.yml       # postgres + redis local
└── README.md
```

## Recursos

- [Notebook interactivo](../notebooks/ia-startups/01-stack-minimo.ipynb)
- [Anthropic — Pricing actualizado](https://www.anthropic.com/pricing)
- [Railway — Hosting económico para startups](https://railway.app)
- [Langfuse — Observabilidad gratuita](https://langfuse.com)
