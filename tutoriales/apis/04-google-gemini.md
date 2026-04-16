# 04 — Google Gemini API

> **Bloque:** APIs de IA · **Nivel:** Intermedio · **Tiempo estimado:** 35 min

---

## Índice

1. [Modelos Gemini: familia y capacidades](#1-modelos-gemini-familia-y-capacidades)
2. [Primeros pasos con la SDK de Python](#2-primeros-pasos-con-la-sdk-de-python)
3. [Multimodalidad: texto, imágenes y vídeo](#3-multimodalidad-texto-imágenes-y-vídeo)
4. [Contexto extendido: 1M de tokens](#4-contexto-extendido-1m-de-tokens)
5. [Function calling con Gemini](#5-function-calling-con-gemini)
6. [Vertex AI: Gemini en Google Cloud](#6-vertex-ai-gemini-en-google-cloud)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Modelos Gemini: familia y capacidades

| Modelo | Contexto | Multimodal | Mejor para |
|---|---|---|---|
| **Gemini 1.5 Pro** | 1M tokens | Sí (texto, imagen, audio, vídeo, PDF) | Análisis de documentos largos, vídeo |
| **Gemini 1.5 Flash** | 1M tokens | Sí | Velocidad y costo, alto volumen |
| **Gemini 1.5 Flash-8B** | 1M tokens | Sí | Máxima economía |
| **Gemini 2.0 Flash** | 1M tokens | Sí | Razonamiento mejorado, bajo costo |

**Precio (aproximado, Mayo 2025):**
- Gemini 1.5 Flash: $0.075/M tokens input, $0.30/M output
- Gemini 1.5 Pro: $3.50/M input (hasta 128K), $10.50/M output

---

## 2. Primeros pasos con la SDK de Python

```bash
pip install google-generativeai
```

```python
# gemini_basics.py
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Listar modelos disponibles
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"{m.name} — contexto: {m.input_token_limit:,} tokens")


# ── Texto simple ──────────────────────────────────────────────────────────
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    "Explica en 3 frases qué es un transformer en deep learning."
)
print(response.text)
print(f"Tokens usados: {response.usage_metadata.total_token_count}")


# ── Streaming ────────────────────────────────────────────────────────────
for chunk in model.generate_content("Escribe un poema sobre la IA.", stream=True):
    print(chunk.text, end="", flush=True)
print()


# ── Configuración de generación ──────────────────────────────────────────
config = genai.types.GenerationConfig(
    temperature=0.3,
    max_output_tokens=512,
    top_p=0.95,
    top_k=40
)

response = model.generate_content(
    "Resume los beneficios del aprendizaje automático en medicina.",
    generation_config=config
)
print(response.text)


# ── Chat con historial ───────────────────────────────────────────────────
chat = model.start_chat(history=[])

for turno in [
    "Hola, ¿qué es RAG en el contexto de LLMs?",
    "¿Cuándo usarías RAG en lugar de fine-tuning?",
    "Dame un ejemplo de caso de uso real"
]:
    response = chat.send_message(turno)
    print(f"Usuario: {turno}")
    print(f"Gemini: {response.text}\n")
```

---

## 3. Multimodalidad: texto, imágenes y vídeo

```python
# gemini_multimodal.py
import google.generativeai as genai
from PIL import Image
import httpx
from pathlib import Path

model = genai.GenerativeModel("gemini-1.5-pro")


# ── Análisis de imagen desde URL ─────────────────────────────────────────
def analizar_imagen_url(url: str, pregunta: str) -> str:
    imagen_bytes = httpx.get(url).content
    imagen = Image.open(__import__("io").BytesIO(imagen_bytes))

    response = model.generate_content([pregunta, imagen])
    return response.text


# ── Análisis de imagen desde archivo local ───────────────────────────────
def analizar_imagen_local(ruta: str, pregunta: str) -> str:
    imagen = Image.open(ruta)
    response = model.generate_content([pregunta, imagen])
    return response.text


# ── Análisis de PDF (nativo en Gemini) ──────────────────────────────────
def analizar_pdf(ruta_pdf: str, pregunta: str) -> str:
    """Gemini puede leer PDFs directamente — sin necesidad de extraer texto."""
    pdf_data = Path(ruta_pdf).read_bytes()
    response = model.generate_content([
        {
            "mime_type": "application/pdf",
            "data": pdf_data
        },
        pregunta
    ])
    return response.text


# ── Análisis de vídeo (subir a Files API) ────────────────────────────────
def analizar_video(ruta_video: str, pregunta: str) -> str:
    """Para vídeos, primero se sube a la Files API de Google."""
    import time

    print("Subiendo vídeo...")
    video_file = genai.upload_file(path=ruta_video)

    # Esperar a que el vídeo esté procesado
    while video_file.state.name == "PROCESSING":
        print("Procesando vídeo...")
        time.sleep(5)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Error procesando el vídeo")

    response = model.generate_content([video_file, pregunta])

    # Limpiar el archivo subido
    genai.delete_file(video_file.name)

    return response.text


# Ejemplos de uso
# descripcion = analizar_imagen_url(
#     "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
#     "¿Qué ves en esta imagen? Describe los elementos principales."
# )

# analisis_pdf = analizar_pdf("contrato.pdf", "Extrae las fechas y partes firmantes.")
```

---

## 4. Contexto extendido: 1M de tokens

```python
# gemini_long_context.py
import google.generativeai as genai
from pathlib import Path

model = genai.GenerativeModel("gemini-1.5-pro")


def preguntar_sobre_repositorio(ruta_repo: str, pregunta: str) -> str:
    """
    Carga todos los archivos de código de un repositorio en el contexto.
    Gemini 1.5 Pro soporta hasta 1M tokens — ideal para codebases completos.
    """
    archivos_codigo = list(Path(ruta_repo).rglob("*.py"))[:50]  # limitar para ejemplo

    partes = [
        f"Analiza el siguiente repositorio de código y responde: {pregunta}\n\n"
    ]

    for archivo in archivos_codigo:
        try:
            contenido = archivo.read_text(encoding="utf-8", errors="ignore")
            if len(contenido) > 100:  # ignorar archivos vacíos
                partes.append(f"# Archivo: {archivo}\n```python\n{contenido}\n```\n\n")
        except Exception:
            pass

    prompt_completo = "".join(partes)
    tokens_estimados = len(prompt_completo) // 4
    print(f"Enviando ~{tokens_estimados:,} tokens a Gemini 1.5 Pro")

    response = model.generate_content(prompt_completo)
    return response.text


def cache_contexto_largo(contenido: str, instruccion_sistema: str) -> "genai.CachedContent":
    """
    Context caching de Gemini: cachea el contexto largo para reutilizarlo
    en múltiples llamadas sin pagar tokens de input cada vez.
    Disponible para contenidos de más de 32,768 tokens.
    """
    import datetime

    cached = genai.caching.CachedContent.create(
        model="models/gemini-1.5-pro-001",
        display_name="contexto-largo-cachado",
        system_instruction=instruccion_sistema,
        contents=[contenido],
        ttl=datetime.timedelta(hours=1)
    )
    return cached


def consultar_con_cache(cached_content, pregunta: str) -> str:
    """Usa el contexto cachado para hacer múltiples preguntas."""
    model_cached = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
    response = model_cached.generate_content(pregunta)
    return response.text
```

---

## 5. Function calling con Gemini

```python
# gemini_function_calling.py
import google.generativeai as genai

# Definir herramientas
def obtener_temperatura(ciudad: str, unidad: str = "celsius") -> dict:
    """Función real que consultaría una API del tiempo."""
    temperaturas = {"Madrid": 22, "Barcelona": 25, "Sevilla": 28}
    temp = temperaturas.get(ciudad, 20)
    if unidad == "fahrenheit":
        temp = temp * 9/5 + 32
    return {"ciudad": ciudad, "temperatura": temp, "unidad": unidad}


def buscar_vuelos(origen: str, destino: str, fecha: str) -> dict:
    """Simula búsqueda de vuelos."""
    return {
        "vuelos": [
            {"compania": "Iberia", "precio": 150, "duracion": "2h30"},
            {"compania": "Vueling", "precio": 120, "duracion": "2h45"}
        ]
    }


# Gemini usa su propia API para tool definitions
herramientas_gemini = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="obtener_temperatura",
            description="Obtiene la temperatura actual de una ciudad",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ciudad": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "unidad": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        enum=["celsius", "fahrenheit"]
                    )
                },
                required=["ciudad"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="buscar_vuelos",
            description="Busca vuelos disponibles entre dos ciudades",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "origen": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "destino": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "fecha": genai.protos.Schema(type=genai.protos.Type.STRING)
                },
                required=["origen", "destino", "fecha"]
            )
        )
    ]
)


EJECUTORES = {
    "obtener_temperatura": lambda args: obtener_temperatura(**args),
    "buscar_vuelos": lambda args: buscar_vuelos(**args)
}


def agente_gemini(pregunta: str) -> str:
    """Bucle agentico con Gemini y function calling."""
    model = genai.GenerativeModel("gemini-1.5-pro", tools=[herramientas_gemini])
    chat = model.start_chat()

    response = chat.send_message(pregunta)

    while response.candidates[0].content.parts[0].function_call.name:
        fn_call = response.candidates[0].content.parts[0].function_call
        nombre = fn_call.name
        args = dict(fn_call.args)

        print(f"  → Llamando {nombre}({args})")
        resultado = EJECUTORES[nombre](args)

        response = chat.send_message(
            genai.protos.Content(parts=[
                genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=nombre,
                    response={"result": resultado}
                ))
            ])
        )

    return response.text
```

---

## 6. Vertex AI: Gemini en Google Cloud

```python
# gemini_vertex.py
# Usar Gemini en producción a través de Vertex AI (autenticación GCP, SLAs empresariales)
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# Inicializar Vertex AI
vertexai.init(project="mi-proyecto-gcp", location="europe-west1")

model = GenerativeModel("gemini-1.5-pro-001")

# Uso básico — idéntico a la SDK de Google AI
response = model.generate_content(
    "¿Qué ventajas tiene usar Vertex AI sobre la API directa de Google AI?",
    generation_config=GenerationConfig(temperature=0.3, max_output_tokens=512)
)
print(response.text)

# Ventajas de Vertex AI vs Google AI Studio API:
# - SLA empresarial y soporte técnico
# - Datos no se usan para entrenar modelos de Google
# - Integración con IAM, VPC, logging de GCP
# - Acceso a modelos en regiones específicas (EU para cumplimiento GDPR)
# - Batch prediction para alto volumen
```

---

## 7. Extensiones sugeridas

- **Grounding con Google Search**: activar búsqueda en tiempo real en las respuestas de Gemini
- **Code execution**: Gemini puede ejecutar Python directamente en sandbox
- **Document AI + Gemini**: pipeline OCR + comprensión de documentos enterprise
- **Gemini en Firebase**: integrar Gemini en apps móviles con Firebase Genkit

---

**Anterior:** [03 — Comparativa de proveedores](./03-comparativa-proveedores.md) · **Siguiente:** [05 — Mistral y Cohere](./05-mistral-cohere.md)
