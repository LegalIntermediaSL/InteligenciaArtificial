# 03 — Pipelines de negocio con LLMs

> **Bloque:** Automatización con IA · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Automatización de procesos con LLMs: cuándo tiene sentido](#1-automatización-de-procesos-con-llms-cuándo-tiene-sentido)
2. [LangChain para pipelines: chains y LCEL](#2-langchain-para-pipelines-chains-y-lcel)
3. [Procesamiento por lotes: Anthropic Batch API y OpenAI Batch](#3-procesamiento-por-lotes-anthropic-batch-api-y-openai-batch)
4. [Webhook + queue con Celery y Redis](#4-webhook--queue-con-celery-y-redis)
5. [Caso práctico: pipeline de contratos end-to-end](#5-caso-práctico-pipeline-de-contratos-end-to-end)
6. [Monitorización y reintentos](#6-monitorización-y-reintentos)

---

## 1. Automatización de procesos con LLMs: cuándo tiene sentido

No todo proceso merece un LLM. Antes de construir un pipeline, evalúa:

### Criterios de viabilidad

| Criterio | Indicador positivo | Indicador negativo |
|---|---|---|
| **Volumen** | >100 documentos/día | <10 documentos/día (hazlo manualmente) |
| **Uniformidad** | Documentos con estructura similar | Cada documento es completamente distinto |
| **Tolerancia al error** | Un error humano revisa la salida | El error tiene consecuencias legales/financieras directas |
| **Coste actual** | Proceso manual cuesta >2h/día | Proceso ya es rápido |
| **Complejidad del juicio** | Clasificación, extracción, resumen | Decisiones estratégicas, empatía compleja |

### Patrones de pipeline más comunes

```
1. Ingest → Extract → Store
   (procesar documentos entrantes, extraer campos, guardar en BD)

2. Trigger → Enrich → Act
   (evento llega, IA añade contexto, sistema actúa)

3. Schedule → Batch Process → Report
   (cron diario, procesar cola de pendientes, enviar informe)

4. Stream → Classify → Route
   (flujo continuo de datos, clasificar en tiempo real, enrutar)
```

**Instalación de dependencias:**

```bash
pip install anthropic openai langchain langchain-anthropic langchain-openai \
            langchain-community celery redis tenacity pydantic \
            pypdf python-dotenv httpx
```

---

## 2. LangChain para pipelines: chains y LCEL

### LCEL: LangChain Expression Language

LCEL es la forma moderna de componer pipelines en LangChain. Usa el operador `|` (pipe) para encadenar componentes, inspirado en Unix.

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# --- Chain básica: prompt → LLM → parser ---
modelo = ChatAnthropic(model="claude-haiku-3-5-20241022", max_tokens=512)

prompt_resumen = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente que resume textos en español de forma concisa."),
    ("human", "Resume en máximo 3 frases:\n\n{texto}"),
])

chain_resumen = prompt_resumen | modelo | StrOutputParser()

# Uso
resumen = chain_resumen.invoke({"texto": "Texto largo aquí..."})
print(resumen)
```

### Chain con salida estructurada (Pydantic)

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class AnalisisDocumento(BaseModel):
    categoria: Literal["contrato", "factura", "propuesta", "correspondencia", "otro"]
    partes: list[str] = Field(description="Nombres de las partes involucradas")
    fecha: Optional[str] = Field(description="Fecha principal del documento (ISO 8601 o null)")
    valor_economico: Optional[float] = Field(description="Importe principal en euros, o null")
    resumen: str = Field(description="Resumen ejecutivo en 2-3 frases")
    requiere_accion: bool = Field(description="Si requiere alguna acción por parte del receptor")
    accion_recomendada: Optional[str] = Field(default=None)


parser = PydanticOutputParser(pydantic_object=AnalisisDocumento)

prompt_analisis = ChatPromptTemplate.from_messages([
    ("system", "Analiza documentos legales y empresariales. Responde SOLO con JSON válido."),
    ("human", "{format_instructions}\n\nDocumento:\n{documento}"),
]).partial(format_instructions=parser.get_format_instructions())

chain_analisis = prompt_analisis | modelo | parser

# Uso
resultado = chain_analisis.invoke({"documento": "Texto del documento..."})
print(f"Categoría: {resultado.categoria}")
print(f"Partes: {resultado.partes}")
print(f"Requiere acción: {resultado.requiere_accion}")
```

### RunnableParallel: ejecutar ramas en paralelo

```python
from langchain_core.runnables import RunnableParallel

# Dos análisis en paralelo sobre el mismo documento
analisis_paralelo = RunnableParallel(
    resumen=chain_resumen,
    analisis=chain_analisis,
)

resultado = analisis_paralelo.invoke({"texto": "...", "documento": "..."})
print(resultado["resumen"])
print(resultado["analisis"].categoria)
```

### Routing condicional con LCEL

```python
from langchain_core.runnables import RunnableBranch

# Prompt de clasificación rápida
prompt_clasificar = ChatPromptTemplate.from_messages([
    ("human", "Clasifica este texto en una sola palabra: contrato, factura, o email.\n\n{texto}"),
])
chain_clasificar = prompt_clasificar | modelo | StrOutputParser()

# Chains especializadas por tipo
chain_contrato = (prompt_contrato | modelo | parser_contrato)
chain_factura   = (prompt_factura  | modelo | parser_factura)
chain_email     = (prompt_email    | modelo | StrOutputParser())

# Router con RunnableBranch
router = RunnableBranch(
    (lambda x: "contrato" in x["tipo"].lower(), chain_contrato),
    (lambda x: "factura"  in x["tipo"].lower(), chain_factura),
    chain_email,  # default
)

# Pipeline completo: clasificar → enrutar → procesar
def pipeline_documento(texto: str):
    tipo = chain_clasificar.invoke({"texto": texto[:500]})  # clasificar con extracto
    return router.invoke({"tipo": tipo, "texto": texto})
```

### LCEL con streaming

```python
import sys

prompt_streaming = ChatPromptTemplate.from_messages([
    ("human", "Explica detalladamente: {pregunta}"),
])

chain_streaming = prompt_streaming | modelo | StrOutputParser()

# Streaming a stdout
for chunk in chain_streaming.stream({"pregunta": "¿Qué es un pipeline de datos?"}):
    print(chunk, end="", flush=True)
print()  # newline final

# Streaming async (para FastAPI/asyncio)
import asyncio

async def stream_async(pregunta: str):
    async for chunk in chain_streaming.astream({"pregunta": pregunta}):
        print(chunk, end="", flush=True)
    print()

asyncio.run(stream_async("¿Cómo funciona LCEL?"))
```

---

## 3. Procesamiento por lotes: Anthropic Batch API y OpenAI Batch

### Anthropic Message Batches API

La Batch API permite enviar hasta 10.000 peticiones en un solo lote, procesarlas de forma asíncrona y obtener un descuento del 50% en el coste.

```python
import anthropic
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def crear_lote_clasificacion(documentos: list[dict]) -> str:
    """
    Envía un lote de documentos para clasificación.

    Args:
        documentos: lista de dicts con 'id' y 'texto'

    Returns:
        ID del lote creado
    """
    peticiones = []

    for doc in documentos:
        peticion = anthropic.types.message_create_params.Request(
            custom_id=doc["id"],  # ID único para correlacionar resultados
            params=anthropic.types.MessageCreateParamsNonStreaming(
                model="claude-haiku-3-5-20241022",
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Clasifica el siguiente texto en una categoría:\n"
                            "[contrato, factura, propuesta, email, otro]\n"
                            "Responde SOLO con el nombre de la categoría.\n\n"
                            f"Texto:\n{doc['texto'][:2000]}"
                        ),
                    }
                ],
            ),
        )
        peticiones.append(peticion)

    lote = client.messages.batches.create(requests=peticiones)
    print(f"Lote creado: {lote.id}")
    print(f"Estado: {lote.processing_status}")
    return lote.id


def esperar_lote(lote_id: str, intervalo_seg: int = 30) -> dict:
    """
    Espera a que el lote termine y devuelve los resultados.

    Args:
        lote_id: ID del lote
        intervalo_seg: segundos entre comprobaciones

    Returns:
        Diccionario {custom_id: categoria}
    """
    while True:
        lote = client.messages.batches.retrieve(lote_id)
        estado = lote.processing_status

        print(f"Estado: {estado} | "
              f"Procesadas: {lote.request_counts.processing} | "
              f"Completadas: {lote.request_counts.succeeded} | "
              f"Errores: {lote.request_counts.errored}")

        if estado == "ended":
            break

        time.sleep(intervalo_seg)

    # Recoger resultados
    resultados = {}
    for resultado in client.messages.batches.results(lote_id):
        if resultado.result.type == "succeeded":
            texto = resultado.result.message.content[0].text.strip().lower()
            resultados[resultado.custom_id] = texto
        else:
            resultados[resultado.custom_id] = "error"
            print(f"Error en {resultado.custom_id}: {resultado.result.error}")

    return resultados


def guardar_resultados(resultados: dict, ruta: str):
    """Guarda los resultados del lote en JSON."""
    Path(ruta).write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Resultados guardados en: {ruta}")


# Uso completo
if __name__ == "__main__":
    # Simular 20 documentos para clasificar
    documentos = [
        {"id": f"doc_{i:03d}", "texto": f"Texto de prueba número {i}. Contrato de servicios..."}
        for i in range(20)
    ]

    # Enviar lote
    lote_id = crear_lote_clasificacion(documentos)

    # Esperar y recoger resultados
    resultados = esperar_lote(lote_id, intervalo_seg=15)

    # Guardar
    guardar_resultados(resultados, "clasificaciones_lote.json")

    # Resumen
    from collections import Counter
    conteo = Counter(resultados.values())
    print("\nDistribución de categorías:")
    for categoria, cantidad in conteo.most_common():
        print(f"  {categoria}: {cantidad}")
```

### OpenAI Batch API

La API Batch de OpenAI funciona con un fichero JSONL de peticiones:

```python
import openai
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()


def crear_fichero_batch(documentos: list[dict], ruta_jsonl: str) -> str:
    """
    Crea el fichero JSONL de peticiones para el Batch API de OpenAI.

    Returns:
        ID del fichero subido
    """
    lineas = []

    for doc in documentos:
        peticion = {
            "custom_id": doc["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": "Clasifica el texto en: contrato, factura, propuesta, email u otro. Responde solo con la categoría."
                    },
                    {
                        "role": "user",
                        "content": doc["texto"][:2000]
                    }
                ]
            }
        }
        lineas.append(json.dumps(peticion, ensure_ascii=False))

    Path(ruta_jsonl).write_text("\n".join(lineas), encoding="utf-8")

    # Subir el fichero
    with open(ruta_jsonl, "rb") as f:
        fichero = client.files.create(file=f, purpose="batch")

    print(f"Fichero subido: {fichero.id}")
    return fichero.id


def ejecutar_batch_openai(file_id: str) -> str:
    """Crea y monitoriza un batch de OpenAI."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"descripcion": "Clasificacion de documentos"},
    )
    print(f"Batch creado: {batch.id}")

    # Polling hasta que termine
    while True:
        batch = client.batches.retrieve(batch.id)
        print(f"Estado: {batch.status} | "
              f"Completadas: {batch.request_counts.completed}/{batch.request_counts.total}")

        if batch.status in ("completed", "failed", "cancelled"):
            break

        time.sleep(60)

    if batch.status == "completed":
        # Descargar resultados
        contenido = client.files.content(batch.output_file_id)
        resultados = {}
        for linea in contenido.text.splitlines():
            r = json.loads(linea)
            categoria = (
                r["response"]["body"]["choices"][0]["message"]["content"]
                .strip().lower()
            )
            resultados[r["custom_id"]] = categoria
        return resultados

    raise RuntimeError(f"Batch terminó con estado: {batch.status}")
```

---

## 4. Webhook + queue con Celery y Redis

Para procesar documentos de forma asíncrona sin bloquear la API HTTP:

### Arquitectura

```
Cliente HTTP
    │ POST /procesar (retorna job_id inmediatamente)
    ▼
FastAPI / Flask
    │ Encola tarea
    ▼
Redis (broker + backend)
    │ Workers consumen tareas
    ▼
Celery Workers (N workers en paralelo)
    │ Llaman a Claude API
    ▼
PostgreSQL / Redis (almacenar resultado)
    │
    ▼
Cliente consulta GET /resultado/{job_id}
```

### Configuración de Celery

```python
# tasks.py
import anthropic
import json
from celery import Celery
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os

load_dotenv()

# Configurar Celery con Redis como broker y backend
app = Celery(
    "pipeline_documentos",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Madrid",
    enable_utc=True,
    # Configuración de reintentos automáticos
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Límite de tareas por worker (evitar OOM con PDFs grandes)
    worker_max_tasks_per_child=100,
    # Tiempo máximo de ejecución por tarea: 5 minutos
    task_time_limit=300,
    task_soft_time_limit=240,
)

client = anthropic.Anthropic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True,
)
def _llamar_claude(texto: str, instruccion: str) -> str:
    """Llama a Claude con reintentos exponenciales ante errores de API."""
    respuesta = client.messages.create(
        model="claude-haiku-3-5-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"{instruccion}\n\n{texto}"}],
    )
    return respuesta.content[0].text


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def clasificar_documento(self, doc_id: str, texto: str) -> dict:
    """
    Tarea Celery: clasifica un documento con Claude.

    Args:
        doc_id: identificador del documento
        texto: contenido del documento

    Returns:
        Dict con doc_id, categoria, confianza y resumen
    """
    try:
        instruccion = (
            "Analiza el documento y devuelve JSON con:\n"
            '{"categoria": "contrato|factura|propuesta|email|otro",\n'
            ' "confianza": 0.0-1.0,\n'
            ' "resumen": "máximo 2 frases"}\n'
            "Solo JSON, sin texto adicional."
        )

        resultado_raw = _llamar_claude(texto[:4000], instruccion)

        # Limpiar y parsear JSON
        if "```json" in resultado_raw:
            resultado_raw = resultado_raw.split("```json")[1].split("```")[0]
        elif "```" in resultado_raw:
            resultado_raw = resultado_raw.split("```")[1].split("```")[0]

        datos = json.loads(resultado_raw.strip())
        datos["doc_id"] = doc_id
        datos["estado"] = "completado"
        return datos

    except anthropic.RateLimitError as exc:
        # Rate limit: reintento con backoff
        raise self.retry(exc=exc, countdown=60)

    except anthropic.APIStatusError as exc:
        # Error de API: reintento
        raise self.retry(exc=exc, countdown=30)

    except json.JSONDecodeError:
        # LLM devolvió respuesta no parseable
        return {
            "doc_id": doc_id,
            "estado": "error",
            "error": "Respuesta no parseable del LLM",
        }


@app.task(bind=True, max_retries=3)
def procesar_lote(self, documentos: list[dict]) -> list[str]:
    """
    Encola múltiples documentos en paralelo.

    Args:
        documentos: lista de dicts con 'id' y 'texto'

    Returns:
        Lista de task IDs
    """
    task_ids = []
    for doc in documentos:
        tarea = clasificar_documento.apply_async(
            args=[doc["id"], doc["texto"]],
            queue="documentos",
            priority=5,
        )
        task_ids.append(tarea.id)

    return task_ids
```

### API HTTP con FastAPI

```python
# api.py
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from celery.result import AsyncResult
from tasks import app as celery_app, clasificar_documento, procesar_lote

api = FastAPI(title="Pipeline de Documentos")


class DocumentoRequest(BaseModel):
    texto: str
    doc_id: str | None = None


class LoteRequest(BaseModel):
    documentos: list[DocumentoRequest]


@api.post("/procesar")
async def procesar_documento(req: DocumentoRequest):
    """Encola un documento para procesamiento asíncrono."""
    doc_id = req.doc_id or str(uuid.uuid4())
    tarea = clasificar_documento.apply_async(
        args=[doc_id, req.texto],
        queue="documentos",
    )
    return {
        "job_id": tarea.id,
        "doc_id": doc_id,
        "estado": "encolado",
    }


@api.post("/procesar/lote")
async def procesar_documentos_lote(req: LoteRequest):
    """Encola múltiples documentos en paralelo."""
    docs = [{"id": d.doc_id or str(uuid.uuid4()), "texto": d.texto}
            for d in req.documentos]
    tarea_lote = procesar_lote.apply_async(args=[docs], queue="control")
    return {
        "batch_job_id": tarea_lote.id,
        "total_documentos": len(docs),
        "estado": "encolado",
    }


@api.get("/resultado/{job_id}")
async def obtener_resultado(job_id: str):
    """Consulta el estado y resultado de una tarea."""
    resultado = AsyncResult(job_id, app=celery_app)

    if resultado.state == "PENDING":
        return {"job_id": job_id, "estado": "pendiente"}

    if resultado.state == "STARTED":
        return {"job_id": job_id, "estado": "procesando"}

    if resultado.state == "SUCCESS":
        return {"job_id": job_id, "estado": "completado", "resultado": resultado.get()}

    if resultado.state == "FAILURE":
        return {
            "job_id": job_id,
            "estado": "error",
            "error": str(resultado.info),
        }

    return {"job_id": job_id, "estado": resultado.state}
```

### Arrancar el stack completo

```bash
# Terminal 1: Redis
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 2: Celery worker
celery -A tasks worker \
  --loglevel=info \
  --queues=documentos,control \
  --concurrency=4

# Terminal 3: FastAPI
uvicorn api:api --reload --port 8000

# Probar
curl -X POST http://localhost:8000/procesar \
  -H "Content-Type: application/json" \
  -d '{"texto": "CONTRATO DE SERVICIOS entre empresa A y empresa B..."}'
```

---

## 5. Caso práctico: pipeline de contratos end-to-end

Pipeline completo que procesa contratos PDF: ingesta → extracción → validación → informe.

```python
# pipeline_contratos.py
import anthropic
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


# --- Modelos de datos ---

class Parte(BaseModel):
    nombre: str
    tipo: str  # "empresa" o "persona_fisica"
    cif_nif: Optional[str] = None
    direccion: Optional[str] = None
    representante: Optional[str] = None


class ClausulaRelevante(BaseModel):
    tipo: str  # "penalizacion", "confidencialidad", "rescision", "precio", etc.
    descripcion: str
    riesgo: str  # "alto", "medio", "bajo"
    extracto: str  # texto literal de la cláusula


class AnalisisContrato(BaseModel):
    tipo_contrato: str
    fecha_firma: Optional[str] = None
    fecha_inicio: Optional[str] = None
    fecha_fin: Optional[str] = None
    partes: list[Parte]
    objeto: str = Field(description="Descripción del objeto del contrato")
    valor_total: Optional[float] = None
    moneda: str = "EUR"
    clausulas_relevantes: list[ClausulaRelevante]
    obligaciones_parte_a: list[str]
    obligaciones_parte_b: list[str]
    ley_aplicable: Optional[str] = None
    jurisdiccion: Optional[str] = None
    resumen_ejecutivo: str
    alertas: list[str] = Field(default_factory=list)


class InformeContrato(BaseModel):
    fichero: str
    paginas: int
    analisis: AnalisisContrato
    puntuacion_riesgo: int  # 0-100, donde 100 es máximo riesgo
    recomendacion: str  # "firmar", "revisar", "no_firmar"


# --- Pasos del pipeline ---

def paso_1_ingestar(ruta_pdf: str) -> tuple[str, int]:
    """Extrae el texto y el número de páginas de un PDF."""
    lector = PdfReader(ruta_pdf)
    paginas = len(lector.pages)
    texto = ""
    for pagina in lector.pages:
        texto += pagina.extract_text() + "\n"

    # Limpiar artefactos comunes de extracción PDF
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' {2,}', ' ', texto)

    print(f"  [✓] Extraídas {paginas} páginas, {len(texto)} caracteres")
    return texto, paginas


def paso_2_extraer(texto: str) -> AnalisisContrato:
    """Extrae información estructurada del contrato."""
    # Dividir en fragmentos si el contrato es muy largo
    max_chars = 12000
    if len(texto) > max_chars:
        # Procesar primera y última parte (suelen tener las cláusulas clave)
        extracto = texto[:max_chars // 2] + "\n[...]\n" + texto[-(max_chars // 2):]
    else:
        extracto = texto

    prompt = f"""Analiza el siguiente contrato y devuelve ÚNICAMENTE un JSON válido con esta estructura:

{{
  "tipo_contrato": "...",
  "fecha_firma": "YYYY-MM-DD o null",
  "fecha_inicio": "YYYY-MM-DD o null",
  "fecha_fin": "YYYY-MM-DD o null",
  "partes": [
    {{
      "nombre": "...",
      "tipo": "empresa|persona_fisica",
      "cif_nif": "... o null",
      "direccion": "... o null",
      "representante": "... o null"
    }}
  ],
  "objeto": "descripción del objeto del contrato",
  "valor_total": número o null,
  "moneda": "EUR",
  "clausulas_relevantes": [
    {{
      "tipo": "penalizacion|confidencialidad|rescision|precio|exclusividad|otro",
      "descripcion": "...",
      "riesgo": "alto|medio|bajo",
      "extracto": "texto literal relevante (max 200 chars)"
    }}
  ],
  "obligaciones_parte_a": ["obligacion 1", "obligacion 2"],
  "obligaciones_parte_b": ["obligacion 1", "obligacion 2"],
  "ley_aplicable": "... o null",
  "jurisdiccion": "... o null",
  "resumen_ejecutivo": "3-4 frases resumiendo el contrato",
  "alertas": ["alerta 1 si hay cláusulas problemáticas", "..."]
}}

Contrato:
{extracto}"""

    respuesta = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    texto_respuesta = respuesta.content[0].text.strip()

    if "```json" in texto_respuesta:
        texto_respuesta = texto_respuesta.split("```json")[1].split("```")[0]
    elif "```" in texto_respuesta:
        texto_respuesta = texto_respuesta.split("```")[1].split("```")[0]

    datos = json.loads(texto_respuesta.strip())
    print(f"  [✓] Extraídas {len(datos.get('clausulas_relevantes', []))} cláusulas relevantes")
    return AnalisisContrato(**datos)


def paso_3_validar(analisis: AnalisisContrato) -> tuple[int, str, list[str]]:
    """
    Valida el análisis y calcula puntuación de riesgo.

    Returns:
        (puntuacion_riesgo 0-100, recomendacion, alertas_adicionales)
    """
    alertas = list(analisis.alertas)
    puntuacion = 0

    # Cláusulas de alto riesgo
    clausulas_alto = [c for c in analisis.clausulas_relevantes if c.riesgo == "alto"]
    puntuacion += len(clausulas_alto) * 15
    for c in clausulas_alto:
        alertas.append(f"Cláusula de alto riesgo ({c.tipo}): {c.descripcion[:100]}")

    # Penalizaciones detectadas
    penalizaciones = [c for c in analisis.clausulas_relevantes
                      if c.tipo == "penalizacion"]
    puntuacion += len(penalizaciones) * 10

    # Contrato sin fecha fin (riesgo de duración indefinida)
    if not analisis.fecha_fin:
        puntuacion += 10
        alertas.append("El contrato no especifica fecha de finalización")

    # Contrato sin jurisdicción
    if not analisis.jurisdiccion:
        puntuacion += 5
        alertas.append("No se especifica jurisdicción en caso de disputa")

    # Valor económico alto sin penalizaciones claras
    if analisis.valor_total and analisis.valor_total > 50000 and not penalizaciones:
        puntuacion += 10
        alertas.append("Contrato de alto valor sin cláusulas de penalización definidas")

    puntuacion = min(puntuacion, 100)

    if puntuacion < 25:
        recomendacion = "firmar"
    elif puntuacion < 60:
        recomendacion = "revisar"
    else:
        recomendacion = "no_firmar"

    print(f"  [✓] Puntuación de riesgo: {puntuacion}/100 → {recomendacion}")
    return puntuacion, recomendacion, alertas


def paso_4_informe(
    ruta_pdf: str,
    paginas: int,
    analisis: AnalisisContrato,
    puntuacion: int,
    recomendacion: str,
) -> InformeContrato:
    """Genera el informe final estructurado."""
    return InformeContrato(
        fichero=Path(ruta_pdf).name,
        paginas=paginas,
        analisis=analisis,
        puntuacion_riesgo=puntuacion,
        recomendacion=recomendacion,
    )


# --- Pipeline principal ---

def pipeline_contratos(ruta_pdf: str, directorio_salida: str = "output_contratos") -> InformeContrato:
    """
    Ejecuta el pipeline completo de análisis de contratos.

    Args:
        ruta_pdf: Ruta al fichero PDF del contrato
        directorio_salida: Carpeta donde guardar el informe JSON

    Returns:
        InformeContrato con todo el análisis
    """
    Path(directorio_salida).mkdir(exist_ok=True)
    nombre = Path(ruta_pdf).stem

    print(f"\n{'='*60}")
    print(f"PIPELINE DE CONTRATOS: {Path(ruta_pdf).name}")
    print(f"{'='*60}")

    print("\nPASO 1/4 — Ingestando PDF...")
    texto, paginas = paso_1_ingestar(ruta_pdf)

    print("\nPASO 2/4 — Extrayendo información con Claude...")
    analisis = paso_2_extraer(texto)

    print("\nPASO 3/4 — Validando y calculando riesgo...")
    puntuacion, recomendacion, alertas = paso_3_validar(analisis)
    analisis.alertas = alertas

    print("\nPASO 4/4 — Generando informe...")
    informe = paso_4_informe(ruta_pdf, paginas, analisis, puntuacion, recomendacion)

    # Guardar informe en JSON
    ruta_informe = Path(directorio_salida) / f"{nombre}_informe.json"
    ruta_informe.write_text(
        informe.model_dump_json(indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"  Fichero: {informe.fichero}")
    print(f"  Tipo: {analisis.tipo_contrato}")
    print(f"  Partes: {', '.join(p.nombre for p in analisis.partes)}")
    print(f"  Riesgo: {puntuacion}/100")
    print(f"  Recomendación: {recomendacion.upper()}")
    if alertas:
        print(f"\nAlertas ({len(alertas)}):")
        for alerta in alertas:
            print(f"  ⚠  {alerta}")
    print(f"\nInforme guardado: {ruta_informe}")

    return informe


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "contrato_ejemplo.pdf"
    pipeline_contratos(ruta)
```

---

## 6. Monitorización y reintentos

### Estrategia de reintentos con tenacity

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import anthropic
import logging

logger = logging.getLogger(__name__)


@retry(
    # Reintentar 4 veces como máximo
    stop=stop_after_attempt(4),
    # Espera exponencial: 4s, 8s, 16s, 32s
    wait=wait_exponential(multiplier=2, min=4, max=60),
    # Solo reintentar en errores recuperables
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    )),
    # Loggear cada reintento
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # Propagar la última excepción si se agotan los reintentos
    reraise=True,
)
def llamada_robusta(prompt: str, max_tokens: int = 1024) -> str:
    """Llamada a Claude con reintentos automáticos ante errores transitorios."""
    client = anthropic.Anthropic()
    respuesta = client.messages.create(
        model="claude-haiku-3-5-20241022",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.content[0].text
```

### Monitorización con métricas

```python
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MetricasTarea:
    """Métricas de una ejecución del pipeline."""
    tarea_id: str
    inicio: str = field(default_factory=lambda: datetime.now().isoformat())
    fin: Optional[str] = None
    duracion_seg: Optional[float] = None
    tokens_entrada: int = 0
    tokens_salida: int = 0
    coste_usd: float = 0.0
    estado: str = "iniciado"  # iniciado, completado, error
    error: Optional[str] = None
    reintentos: int = 0


class MonitorPipeline:
    """Registra métricas de todas las ejecuciones del pipeline."""

    # Precios por millón de tokens (actualizar según tarifas actuales)
    PRECIOS = {
        "claude-haiku-3-5-20241022": {"entrada": 0.80, "salida": 4.00},
        "claude-sonnet-4-6":         {"entrada": 3.00, "salida": 15.00},
        "claude-opus-4-5":           {"entrada": 15.00, "salida": 75.00},
        "gpt-4o-mini":               {"entrada": 0.15, "salida": 0.60},
        "gpt-4o":                    {"entrada": 2.50, "salida": 10.00},
    }

    def __init__(self, ruta_log: str = "metricas_pipeline.jsonl"):
        self.ruta_log = Path(ruta_log)
        self.metricas_activas: dict[str, MetricasTarea] = {}

    def iniciar(self, tarea_id: str) -> MetricasTarea:
        m = MetricasTarea(tarea_id=tarea_id)
        self.metricas_activas[tarea_id] = m
        return m

    def registrar_uso(
        self,
        tarea_id: str,
        modelo: str,
        tokens_entrada: int,
        tokens_salida: int,
    ):
        """Registra el uso de tokens y calcula el coste."""
        if tarea_id not in self.metricas_activas:
            return

        m = self.metricas_activas[tarea_id]
        m.tokens_entrada += tokens_entrada
        m.tokens_salida += tokens_salida

        if modelo in self.PRECIOS:
            precios = self.PRECIOS[modelo]
            m.coste_usd += (tokens_entrada / 1_000_000) * precios["entrada"]
            m.coste_usd += (tokens_salida / 1_000_000) * precios["salida"]

    def completar(self, tarea_id: str, estado: str = "completado", error: str = None):
        """Cierra la métrica y la persiste en el log."""
        if tarea_id not in self.metricas_activas:
            return

        m = self.metricas_activas.pop(tarea_id)
        m.fin = datetime.now().isoformat()
        m.duracion_seg = (
            datetime.fromisoformat(m.fin) - datetime.fromisoformat(m.inicio)
        ).total_seconds()
        m.estado = estado
        m.error = error

        # Persistir en JSONL (append)
        with open(self.ruta_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

        return m

    def resumen(self, ultimas_n: int = 100) -> dict:
        """Calcula estadísticas de las últimas N ejecuciones."""
        if not self.ruta_log.exists():
            return {}

        registros = []
        with open(self.ruta_log, encoding="utf-8") as f:
            for linea in f:
                registros.append(json.loads(linea))

        registros = registros[-ultimas_n:]
        completados = [r for r in registros if r["estado"] == "completado"]
        errores = [r for r in registros if r["estado"] == "error"]

        if not completados:
            return {"total": len(registros), "completados": 0}

        duraciones = [r["duracion_seg"] for r in completados if r["duracion_seg"]]
        costes = [r["coste_usd"] for r in completados]

        return {
            "total": len(registros),
            "completados": len(completados),
            "errores": len(errores),
            "tasa_exito": len(completados) / len(registros),
            "duracion_media_seg": sum(duraciones) / len(duraciones) if duraciones else 0,
            "duracion_max_seg": max(duraciones) if duraciones else 0,
            "coste_total_usd": sum(costes),
            "coste_medio_usd": sum(costes) / len(costes) if costes else 0,
        }


# Uso integrado en el pipeline
monitor = MonitorPipeline()

def pipeline_con_monitoreo(doc_id: str, texto: str) -> dict:
    m = monitor.iniciar(doc_id)
    try:
        client = anthropic.Anthropic()
        respuesta = client.messages.create(
            model="claude-haiku-3-5-20241022",
            max_tokens=512,
            messages=[{"role": "user", "content": f"Clasifica: {texto[:1000]}"}],
        )
        monitor.registrar_uso(
            doc_id,
            "claude-haiku-3-5-20241022",
            respuesta.usage.input_tokens,
            respuesta.usage.output_tokens,
        )
        resultado = {"categoria": respuesta.content[0].text.strip()}
        metricas = monitor.completar(doc_id, "completado")
        return {**resultado, "coste_usd": metricas.coste_usd}
    except Exception as e:
        monitor.completar(doc_id, "error", str(e))
        raise
```

---

**Anterior:** [02 — Make.com y Zapier con IA](./02-make-zapier-ia.md) · **Siguiente:** [04 — Integración con herramientas de negocio](./04-integracion-herramientas-negocio.md)
