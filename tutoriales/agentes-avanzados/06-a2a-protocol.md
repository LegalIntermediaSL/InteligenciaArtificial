# 06 — A2A Protocol: Agent-to-Agent Communication

> **Bloque:** Agentes avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Qué es A2A y por qué importa](#1-qué-es-a2a-y-por-qué-importa)
2. [Arquitectura del protocolo](#2-arquitectura-del-protocolo)
3. [Agent Card: identidad y capacidades](#3-agent-card-identidad-y-capacidades)
4. [Servidor A2A con FastAPI](#4-servidor-a2a-con-fastapi)
5. [Cliente A2A: enviar tareas a agentes remotos](#5-cliente-a2a-enviar-tareas-a-agentes-remotos)
6. [Orquestación multi-agente con A2A](#6-orquestación-multi-agente-con-a2a)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Qué es A2A y por qué importa

**A2A (Agent-to-Agent)** es un protocolo abierto propuesto por Google en 2025 para la comunicación estandarizada entre agentes de IA de diferentes sistemas, frameworks y proveedores.

### El problema que resuelve

Antes de A2A, cada sistema multi-agente usaba su propia forma de comunicación:
- CrewAI tiene su formato interno
- AutoGen tiene el suyo
- LangGraph el suyo
- Un agente de Claude y uno de GPT-4 no podían comunicarse directamente

A2A propone un estándar universal basado en HTTP + JSON-RPC, de forma similar a cómo HTTP estandarizó la comunicación entre servicios web.

### Componentes del ecosistema

| Componente | Rol |
|---|---|
| **Agent Card** | JSON que describe las capacidades, autenticación y endpoints de un agente |
| **Task** | Unidad de trabajo enviada de un agente a otro |
| **Message** | Mensaje dentro de una tarea (texto, datos, archivos) |
| **Artifact** | Resultado producido por un agente (texto, código, archivos) |
| **Streaming** | Soporte para respuestas en tiempo real via SSE |

---

## 2. Arquitectura del protocolo

```
┌─────────────────┐     A2A/HTTP     ┌─────────────────┐
│  Agente Cliente │ ───────────────► │  Agente Servidor│
│  (orquestador)  │                  │  (especialista) │
│                 │ ◄─────────────── │                 │
└─────────────────┘   JSON-RPC 2.0   └─────────────────┘
        │                                     │
        │                                     │
   Descubre Agent Card              Expone Agent Card
   en /.well-known/                 en /.well-known/
   agent-card.json                  agent-card.json
```

### Flujo de una tarea

```
1. Cliente descubre Agent Card del servidor
2. Cliente envía Task (POST /tasks/send)
3. Servidor procesa y actualiza estado: submitted → working → completed
4. Cliente consulta estado (GET /tasks/{id}) o recibe streaming (SSE)
5. Cliente recibe Artifacts con los resultados
```

---

## 3. Agent Card: identidad y capacidades

```python
# agent_card.py
from typing import Optional

# Estructura de un Agent Card según spec A2A
AGENT_CARD = {
    "name": "AgentAnalisisContratos",
    "description": "Agente especializado en análisis y extracción de información de contratos legales en español.",
    "url": "https://agentes.miempresa.com/contratos",
    "version": "1.0.0",
    "provider": {
        "organization": "LegalTech SL",
        "url": "https://legaltech.example.com"
    },
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True
    },
    "authentication": {
        "schemes": ["Bearer"]
    },
    "defaultInputModes": ["text/plain", "application/pdf"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "skills": [
        {
            "id": "extraer-partes",
            "name": "Extraer partes del contrato",
            "description": "Identifica y extrae las partes firmantes de un contrato.",
            "tags": ["contratos", "extracción", "legal"],
            "examples": [
                "Extrae las partes de este contrato de compraventa",
                "¿Quiénes son los firmantes?"
            ],
            "inputModes": ["text/plain", "application/pdf"],
            "outputModes": ["application/json"]
        },
        {
            "id": "analizar-clausulas",
            "name": "Analizar cláusulas",
            "description": "Clasifica y resume las cláusulas principales del contrato.",
            "tags": ["contratos", "análisis", "clasificación"],
            "inputModes": ["text/plain"],
            "outputModes": ["application/json", "text/plain"]
        },
        {
            "id": "detectar-riesgos",
            "name": "Detectar riesgos legales",
            "description": "Identifica cláusulas potencialmente problemáticas o inusuales.",
            "tags": ["contratos", "riesgo", "legal"],
            "inputModes": ["text/plain"],
            "outputModes": ["application/json"]
        }
    ]
}
```

---

## 4. Servidor A2A con FastAPI

```python
# a2a_server.py
import uuid
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional
from enum import Enum

import anthropic
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Agente Análisis de Contratos — A2A Server")
claude = anthropic.Anthropic()

# Estado de tareas en memoria (en prod: Redis o BD)
TAREAS: dict[str, dict] = {}


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Modelos A2A ─────────────────────────────────────────────────────────────

class MessagePart(BaseModel):
    type: str  # "text" | "file" | "data"
    text: Optional[str] = None
    mimeType: Optional[str] = None
    data: Optional[dict] = None


class Message(BaseModel):
    role: str  # "user" | "agent"
    parts: list[MessagePart]


class TaskSendRequest(BaseModel):
    id: str
    message: Message
    sessionId: Optional[str] = None
    metadata: Optional[dict] = None


class Artifact(BaseModel):
    index: int
    parts: list[MessagePart]
    lastChunk: bool = True


class TaskStatus(BaseModel):
    state: TaskState
    message: Optional[Message] = None
    timestamp: str = ""

    def __init__(self, **data):
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)


class Task(BaseModel):
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: list[Artifact] = []
    history: list[Message] = []


# ── Endpoints A2A ───────────────────────────────────────────────────────────

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    """Endpoint de descubrimiento — devuelve el Agent Card."""
    return JSONResponse(content=AGENT_CARD)


@app.post("/tasks/send")
async def send_task(request: TaskSendRequest, authorization: str = Header(None)):
    """Recibe una tarea y la procesa de forma síncrona."""
    # Validar auth
    if authorization and not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token no válido")

    task_id = request.id or str(uuid.uuid4())

    # Extraer texto del mensaje
    texto = " ".join(
        p.text for p in request.message.parts
        if p.type == "text" and p.text
    )

    if not texto:
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    # Registrar tarea
    TAREAS[task_id] = {
        "id": task_id,
        "estado": TaskState.WORKING,
        "input": texto,
        "resultado": None,
        "created_at": datetime.now().isoformat()
    }

    # Procesar con Claude
    try:
        resultado = await procesar_contrato(texto)
        TAREAS[task_id]["estado"] = TaskState.COMPLETED
        TAREAS[task_id]["resultado"] = resultado

        return Task(
            id=task_id,
            status=TaskStatus(state=TaskState.COMPLETED),
            artifacts=[
                Artifact(
                    index=0,
                    parts=[MessagePart(type="data", data=resultado, mimeType="application/json")]
                )
            ]
        )
    except Exception as e:
        TAREAS[task_id]["estado"] = TaskState.FAILED
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/sendSubscribe")
async def send_task_subscribe(request: TaskSendRequest):
    """Versión streaming con SSE."""
    texto = " ".join(
        p.text for p in request.message.parts
        if p.type == "text" and p.text
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # Evento: tarea recibida
        yield f"data: {json.dumps({'id': request.id, 'status': {'state': 'working'}})}\n\n"

        # Streaming de Claude
        with claude.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system="Eres un experto en análisis de contratos legales.",
            messages=[{"role": "user", "content": texto}]
        ) as stream:
            buffer = ""
            for text in stream.text_stream:
                buffer += text
                # Enviar chunk
                chunk_event = {
                    "id": request.id,
                    "status": {"state": "working"},
                    "artifact": {
                        "index": 0,
                        "parts": [{"type": "text", "text": text}],
                        "lastChunk": False
                    }
                }
                yield f"data: {json.dumps(chunk_event)}\n\n"

        # Evento final
        final_event = {
            "id": request.id,
            "status": {"state": "completed"},
            "artifact": {
                "index": 0,
                "parts": [{"type": "text", "text": ""}],
                "lastChunk": True
            }
        }
        yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Consulta el estado de una tarea."""
    if task_id not in TAREAS:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    tarea = TAREAS[task_id]
    return Task(
        id=task_id,
        status=TaskStatus(state=tarea["estado"]),
        artifacts=[
            Artifact(
                index=0,
                parts=[MessagePart(type="data", data=tarea["resultado"], mimeType="application/json")]
            )
        ] if tarea["resultado"] else []
    )


async def procesar_contrato(texto: str) -> dict:
    """Lógica de análisis del contrato con Claude."""
    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system="Eres un experto en análisis de contratos. Devuelve JSON.",
        messages=[{
            "role": "user",
            "content": (
                f"Analiza este contrato y devuelve JSON con: "
                f"partes (lista de firmantes), tipo_contrato, "
                f"clausulas_principales (lista), riesgos (lista).\n\n{texto}"
            )
        }]
    )
    import json
    try:
        return json.loads(response.content[0].text)
    except json.JSONDecodeError:
        return {"analisis": response.content[0].text}
```

---

## 5. Cliente A2A: enviar tareas a agentes remotos

```python
# a2a_client.py
import httpx
import json
import uuid
from typing import AsyncGenerator


class A2AClient:
    """Cliente A2A para comunicarse con agentes remotos."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(headers=self.headers, timeout=60)

    async def descubrir_agente(self) -> dict:
        """Obtiene el Agent Card del servidor."""
        r = await self.client.get(f"{self.base_url}/.well-known/agent-card.json")
        r.raise_for_status()
        return r.json()

    async def enviar_tarea(self, mensaje: str, session_id: str = "") -> dict:
        """Envía una tarea y espera el resultado."""
        payload = {
            "id": str(uuid.uuid4()),
            "sessionId": session_id or str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": mensaje}]
            }
        }
        r = await self.client.post(f"{self.base_url}/tasks/send", json=payload)
        r.raise_for_status()
        return r.json()

    async def enviar_tarea_streaming(
        self,
        mensaje: str
    ) -> AsyncGenerator[dict, None]:
        """Envía una tarea con respuesta en streaming."""
        payload = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": mensaje}]
            }
        }
        async with self.client.stream(
            "POST",
            f"{self.base_url}/tasks/sendSubscribe",
            json=payload
        ) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    yield json.loads(line[5:].strip())

    async def cerrar(self):
        await self.client.aclose()


# Ejemplo de uso
import asyncio

async def main():
    cliente = A2AClient("http://localhost:8000", api_key="mi-token")

    # Descubrir capacidades
    card = await cliente.descubrir_agente()
    print(f"Agente: {card['name']}")
    print(f"Habilidades: {[s['name'] for s in card['skills']]}")

    # Enviar tarea
    resultado = await cliente.enviar_tarea(
        "Analiza este contrato: 'CONTRATO DE COMPRAVENTA entre Juan García (vendedor) "
        "y María López (compradora) por importe de 50.000€...'"
    )
    print(f"\nResultado: {json.dumps(resultado, ensure_ascii=False, indent=2)}")

    # Streaming
    print("\nRespuesta en streaming:")
    async for evento in cliente.enviar_tarea_streaming("Extrae las partes de: ..."):
        if "artifact" in evento:
            texto = evento["artifact"]["parts"][0].get("text", "")
            if texto:
                print(texto, end="", flush=True)

    await cliente.cerrar()


asyncio.run(main())
```

---

## 6. Orquestación multi-agente con A2A

```python
# a2a_orchestrator.py
import asyncio
import httpx
from a2a_client import A2AClient

# Registro de agentes disponibles
AGENTES_DISPONIBLES = {
    "analisis_contratos": "http://localhost:8001",
    "traduccion": "http://localhost:8002",
    "resumen_ejecutivo": "http://localhost:8003",
}


class OrquestadorA2A:
    """Orquestador que coordina múltiples agentes A2A."""

    def __init__(self):
        self.clientes: dict[str, A2AClient] = {}

    async def inicializar(self):
        """Descubre y conecta con todos los agentes disponibles."""
        for nombre, url in AGENTES_DISPONIBLES.items():
            cliente = A2AClient(url)
            try:
                card = await cliente.descubrir_agente()
                self.clientes[nombre] = cliente
                print(f"✅ Conectado a {card['name']} en {url}")
            except Exception as e:
                print(f"❌ No se pudo conectar a {nombre}: {e}")

    async def procesar_documento_legal(self, texto: str, idioma_destino: str = "en") -> dict:
        """
        Pipeline multi-agente:
        1. Analizar contrato
        2. Traducir (si se solicita)
        3. Generar resumen ejecutivo
        """
        resultados = {}

        # 1. Análisis en paralelo si hay múltiples análisis a hacer
        tareas = [
            ("analisis", self.clientes["analisis_contratos"].enviar_tarea(
                f"Analiza este contrato en detalle:\n{texto}"
            ))
        ]

        if idioma_destino != "es" and "traduccion" in self.clientes:
            tareas.append(
                ("traduccion", self.clientes["traduccion"].enviar_tarea(
                    f"Traduce al {idioma_destino}:\n{texto[:2000]}"
                ))
            )

        # Ejecutar en paralelo
        nombres, coroutines = zip(*tareas)
        resultados_lista = await asyncio.gather(*coroutines, return_exceptions=True)

        for nombre, resultado in zip(nombres, resultados_lista):
            if isinstance(resultado, Exception):
                resultados[nombre] = {"error": str(resultado)}
            else:
                resultados[nombre] = resultado

        # 2. Resumen ejecutivo basado en el análisis
        if "analisis_contratos" in self.clientes and "analisis" in resultados:
            resumen = await self.clientes["resumen_ejecutivo"].enviar_tarea(
                f"Genera un resumen ejecutivo de este análisis:\n{str(resultados['analisis'])[:2000]}"
            )
            resultados["resumen_ejecutivo"] = resumen

        return resultados

    async def cerrar(self):
        for cliente in self.clientes.values():
            await cliente.cerrar()


# Uso
async def main():
    orquestador = OrquestadorA2A()
    await orquestador.inicializar()

    resultado = await orquestador.procesar_documento_legal(
        texto="CONTRATO DE SERVICIOS entre...",
        idioma_destino="en"
    )
    print(resultado)
    await orquestador.cerrar()

asyncio.run(main())
```

---

## 7. Extensiones sugeridas

- **A2A + MCP**: combinar A2A para comunicación entre agentes con MCP para acceso a herramientas y datos
- **Service mesh para agentes**: usar Istio o Envoy para gestionar comunicación A2A con circuit breakers y observabilidad
- **Agent registry**: implementar un directorio centralizado donde los agentes se registran y son descubribles
- **A2A SDK oficial de Google**: seguir el desarrollo del SDK oficial en github.com/google/a2a

---

**Anterior:** [05 — AutoGen](./05-autogen.md) · **Siguiente bloque:** [Bloque 10 — Casos de uso avanzados](../casos-de-uso-avanzados/)
