# 01 — Ejecutar modelos locales con Ollama

> **Bloque:** IA local · **Nivel:** Intermedio · **Tiempo estimado:** 45 min

---

## Índice

1. [Qué es Ollama y por qué usarlo](#1-qué-es-ollama-y-por-qué-usarlo)
2. [Instalación](#2-instalación)
3. [Primeros pasos con la CLI](#3-primeros-pasos-con-la-cli)
4. [API REST compatible con OpenAI](#4-api-rest-compatible-con-openai)
5. [SDK nativo de Ollama para Python](#5-sdk-nativo-de-ollama-para-python)
6. [Modelos recomendados](#6-modelos-recomendados)
7. [Multimodalidad local](#7-multimodalidad-local)
8. [Integrar Ollama en un pipeline RAG](#8-integrar-ollama-en-un-pipeline-rag)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Qué es Ollama y por qué usarlo

Ollama es una herramienta de código abierto que permite descargar y ejecutar modelos de lenguaje directamente en tu máquina, con una interfaz simple de línea de comandos y una API REST compatible con el estándar de OpenAI.

### Ventajas frente a las APIs cloud

| Dimensión | Ollama (local) | APIs cloud (OpenAI, Anthropic…) |
|-----------|---------------|--------------------------------|
| **Privacidad** | Los datos nunca salen de tu máquina | Los datos viajan a servidores externos |
| **Coste** | Solo hardware y electricidad | Pago por token (puede escalar) |
| **Disponibilidad** | Funciona offline | Requiere conexión a internet |
| **Latencia** | Depende de tu hardware | Red + cola de la API |
| **Control** | Total: temperatura, contexto, versión del modelo | Limitado por el proveedor |
| **Calidad** | Modelos open-source (muy buenos en 7B+) | Modelos propietarios de vanguardia |

Ollama es ideal cuando:
- Los datos son sensibles (salud, legal, finanzas).
- Necesitas trabajar sin conexión.
- Quieres experimentar sin coste por consulta.
- Buscas integrar un LLM en una aplicación sin depender de terceros.

---

## 2. Instalación

### macOS

```bash
# Opción 1: con el instalador oficial (recomendado)
curl -fsSL https://ollama.ai/install.sh | sh

# Opción 2: con Homebrew
brew install ollama
```

### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

El script detecta automáticamente si tienes drivers NVIDIA y configura soporte GPU.

### Windows

Descarga el instalador desde [https://ollama.ai/download](https://ollama.ai/download) y ejecútalo. Requiere Windows 10 o superior.

### Verificar la instalación

```bash
ollama --version
# ollama version 0.3.x

# Iniciar el servidor (si no arranca automáticamente)
ollama serve
```

El servidor escucha en `http://localhost:11434` por defecto.

---

## 3. Primeros pasos con la CLI

### Descargar un modelo

```bash
# Descargar llama3.2 (versión 3B, ~2 GB)
ollama pull llama3.2

# Descargar una versión específica
ollama pull llama3.2:1b      # 1B parámetros, muy ligero
ollama pull mistral:7b-instruct
ollama pull gemma2:9b
```

### Chatear con un modelo

```bash
# Modo interactivo
ollama run llama3.2

# Prompt de una sola vez (útil en scripts)
ollama run llama3.2 "Explica qué es la cuantización de modelos en 3 líneas"

# Pasar contexto desde un archivo
cat documento.txt | ollama run llama3.2 "Resume este texto"
```

### Gestionar modelos

```bash
# Listar modelos descargados
ollama list

# Ver modelos en ejecución (y su uso de memoria)
ollama ps

# Eliminar un modelo
ollama rm llama3.2

# Ver información de un modelo
ollama show llama3.2
```

---

## 4. API REST compatible con OpenAI

Ollama expone una API REST en `http://localhost:11434` que sigue el mismo formato que la API de OpenAI. Esto permite usar el SDK oficial de OpenAI apuntando a tu servidor local.

```python
from openai import OpenAI

# Apuntar al servidor local de Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama no requiere API key; este valor se ignora
)

# Chat completion (mismo formato que con OpenAI)
respuesta = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "Eres un asistente experto en Python."},
        {"role": "user", "content": "¿Cuál es la diferencia entre una lista y una tupla?"},
    ],
    temperature=0.7,
)

print(respuesta.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Escribe un poema sobre el código limpio"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

Esta compatibilidad significa que puedes cambiar una aplicación que usa OpenAI por Ollama modificando solo dos líneas: `base_url` y `api_key`.

---

## 5. SDK nativo de Ollama para Python

El SDK nativo ofrece características adicionales específicas de Ollama, como embeddings y gestión de modelos.

```bash
pip install ollama
```

### Chat básico

```python
import ollama

# Chat simple
respuesta = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "¿Cuánto es 15% de 340?"},
    ],
)

print(respuesta["message"]["content"])
```

### Conversación con historial

```python
import ollama

def chatbot():
    historial = []
    sistema = {
        "role": "system",
        "content": "Eres un tutor de matemáticas. Explica cada paso con claridad.",
    }

    print("Chatbot de matemáticas (escribe 'salir' para terminar)\n")

    while True:
        entrada = input("Tú: ").strip()
        if entrada.lower() == "salir":
            break

        historial.append({"role": "user", "content": entrada})

        respuesta = ollama.chat(
            model="llama3.2",
            messages=[sistema] + historial,
        )

        mensaje_asistente = respuesta["message"]["content"]
        historial.append({"role": "assistant", "content": mensaje_asistente})

        print(f"\nAsistente: {mensaje_asistente}\n")

chatbot()
```

### Streaming

```python
import ollama

# El streaming devuelve tokens a medida que se generan
stream = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explica el algoritmo de backpropagation"}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
```

### Embeddings

```python
import ollama
import numpy as np

def similitud_coseno(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Generar embeddings
textos = [
    "El aprendizaje automático es una rama de la IA",
    "Los modelos de lenguaje procesan texto",
    "El fútbol es el deporte más popular del mundo",
]

embeddings = []
for texto in textos:
    resultado = ollama.embeddings(model="llama3.2", prompt=texto)
    embeddings.append(resultado["embedding"])

# Comparar similitudes
print("Similitudes entre textos:")
for i in range(len(textos)):
    for j in range(i + 1, len(textos)):
        sim = similitud_coseno(embeddings[i], embeddings[j])
        print(f"  '{textos[i][:40]}...' vs '{textos[j][:40]}...': {sim:.3f}")
```

---

## 6. Modelos recomendados

| Modelo | Tamaño | VRAM aprox. | Uso recomendado |
|--------|--------|-------------|-----------------|
| `llama3.2:1b` | ~1.3 GB | 2 GB | Prototipado rápido, dispositivos limitados |
| `llama3.2:3b` | ~2.0 GB | 3 GB | Uso general en portátil |
| `llama3.2` (7B) | ~4.7 GB | 6 GB | Chat, resúmenes, análisis de texto |
| `mistral` | ~4.1 GB | 6 GB | Instrucciones precisas, razonamiento |
| `gemma2:9b` | ~5.5 GB | 8 GB | Razonamiento, calidad alta |
| `phi3` | ~2.2 GB | 4 GB | Eficiente, bueno en código |
| `codellama` | ~3.8 GB | 6 GB | Generación y análisis de código |
| `qwen2:7b` | ~4.4 GB | 6 GB | Multilingüe, chino/inglés/español |
| `llava` | ~4.5 GB | 6 GB | Análisis de imágenes (multimodal) |

```bash
# Descargar los modelos más útiles para empezar
ollama pull llama3.2
ollama pull mistral
ollama pull phi3
ollama pull codellama
```

---

## 7. Multimodalidad local

El modelo `llava` permite analizar imágenes localmente, sin enviar los datos a ningún servidor externo.

```bash
ollama pull llava
```

```python
import ollama
import base64
from pathlib import Path

def analizar_imagen(ruta_imagen: str, pregunta: str) -> str:
    """Analiza una imagen local con llava."""
    # Leer y codificar la imagen en base64
    imagen_bytes = Path(ruta_imagen).read_bytes()
    imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")

    respuesta = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": pregunta,
                "images": [imagen_b64],
            }
        ],
    )

    return respuesta["message"]["content"]


# Ejemplos de uso
if __name__ == "__main__":
    # Analizar una captura de pantalla
    descripcion = analizar_imagen(
        "captura.png",
        "Describe detalladamente lo que ves en esta imagen",
    )
    print("Descripción:", descripcion)

    # Extraer texto de una imagen (OCR con LLM)
    texto = analizar_imagen(
        "documento.jpg",
        "Extrae todo el texto visible en esta imagen, manteniendo el formato original",
    )
    print("\nTexto extraído:", texto)

    # Analizar un gráfico o diagrama
    analisis = analizar_imagen(
        "grafico.png",
        "Este es un gráfico de datos. ¿Qué tendencias observas? ¿Qué conclusiones puedes extraer?",
    )
    print("\nAnálisis del gráfico:", analisis)
```

---

## 8. Integrar Ollama en un pipeline RAG

RAG (Retrieval-Augmented Generation) combina una base de datos vectorial con un LLM: primero recupera fragmentos relevantes y luego los usa como contexto para la generación. Aquí lo hacemos íntegramente local.

```bash
pip install ollama chromadb
```

```python
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ─── 1. Preparar documentos ────────────────────────────────────────────────
documentos = [
    {
        "id": "doc1",
        "texto": "Python es un lenguaje de programación de alto nivel creado por Guido van Rossum en 1991.",
        "fuente": "Wikipedia Python",
    },
    {
        "id": "doc2",
        "texto": "Las listas en Python son colecciones ordenadas y mutables que permiten duplicados.",
        "fuente": "Docs Python",
    },
    {
        "id": "doc3",
        "texto": "Los decoradores en Python son funciones que modifican el comportamiento de otras funciones.",
        "fuente": "Docs Python",
    },
    {
        "id": "doc4",
        "texto": "El garbage collector de Python gestiona automáticamente la memoria mediante conteo de referencias.",
        "fuente": "Python Internals",
    },
    {
        "id": "doc5",
        "texto": "FastAPI es un framework web moderno para Python, muy rápido y con validación automática de tipos.",
        "fuente": "FastAPI Docs",
    },
]


# ─── 2. Función de embeddings con Ollama ───────────────────────────────────
def embedding_ollama(texto: str) -> list[float]:
    resultado = ollama.embeddings(model="llama3.2", prompt=texto)
    return resultado["embedding"]


# ─── 3. Crear base de datos vectorial con ChromaDB ─────────────────────────
cliente_chroma = chromadb.Client()
coleccion = cliente_chroma.create_collection(
    name="base_conocimiento",
    metadata={"hnsw:space": "cosine"},
)

# Indexar documentos
print("Indexando documentos...")
for doc in documentos:
    embedding = embedding_ollama(doc["texto"])
    coleccion.add(
        ids=[doc["id"]],
        embeddings=[embedding],
        documents=[doc["texto"]],
        metadatas=[{"fuente": doc["fuente"]}],
    )
print(f"  {len(documentos)} documentos indexados.\n")


# ─── 4. Función RAG completa ───────────────────────────────────────────────
def rag_query(pregunta: str, n_resultados: int = 2) -> str:
    # 4a. Recuperar fragmentos relevantes
    embedding_pregunta = embedding_ollama(pregunta)
    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=n_resultados,
    )

    fragmentos = resultados["documents"][0]
    fuentes = [m["fuente"] for m in resultados["metadatas"][0]]

    # 4b. Construir el prompt con contexto
    contexto = "\n".join(
        f"[{fuente}] {fragmento}"
        for fragmento, fuente in zip(fragmentos, fuentes)
    )

    prompt_sistema = (
        "Eres un asistente experto. Responde la pregunta usando SOLO la información "
        "del contexto proporcionado. Si la información no es suficiente, indícalo."
    )

    prompt_usuario = f"""Contexto:
{contexto}

Pregunta: {pregunta}"""

    # 4c. Generar respuesta con Ollama
    respuesta = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
    )

    return respuesta["message"]["content"]


# ─── 5. Probar el sistema RAG ──────────────────────────────────────────────
preguntas = [
    "¿Qué son los decoradores en Python?",
    "¿Cómo gestiona Python la memoria?",
    "¿Qué framework puedo usar para crear una API REST en Python?",
]

for pregunta in preguntas:
    print(f"Pregunta: {pregunta}")
    respuesta = rag_query(pregunta)
    print(f"Respuesta: {respuesta}\n{'─' * 60}\n")
```

---

## 9. Extensiones sugeridas

- **Modelfile personalizado**: crea tu propio modelo con un `Modelfile` para definir el system prompt, temperatura y parámetros por defecto (`ollama create mi-asistente -f Modelfile`).
- **Open WebUI**: interfaz web completa tipo ChatGPT para Ollama ([github.com/open-webui/open-webui](https://github.com/open-webui/open-webui)).
- **LangChain + Ollama**: usa `langchain-ollama` para integrar Ollama en pipelines LangChain complejos.
- **Ollama en Docker**: despliega Ollama en un contenedor para entornos de producción reproducibles.
- **Bench de modelos**: compara velocidad de inferencia entre modelos con `ollama run --verbose`.

---

**Siguiente:** [02 — Inferencia local con Hugging Face Transformers](./02-transformers-local.md)
