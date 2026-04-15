# 09 — Modelos locales con Ollama

> **Bloque:** LLMs · **Nivel:** Intermedio · **Tiempo estimado:** 45 min

---

## Índice

1. [El caso de los modelos locales](#1-el-caso-de-los-modelos-locales)
2. [Ventajas en detalle](#2-ventajas-en-detalle)
3. [Qué es Ollama](#3-qué-es-ollama)
4. [Instalación y primeros pasos](#4-instalación-y-primeros-pasos)
5. [Catálogo de modelos](#5-catálogo-de-modelos)
6. [Integración con Python](#6-integración-con-python)
7. [API compatible con OpenAI](#7-api-compatible-con-openai)
8. [Integración con LangChain](#8-integración-con-langchain)
9. [Rendimiento y optimización](#9-rendimiento-y-optimización)
10. [Limitaciones y cuándo usar cloud](#10-limitaciones-y-cuándo-usar-cloud)
11. [Guía de decisión: local vs cloud](#11-guía-de-decisión-local-vs-cloud)
12. [Resumen](#12-resumen)

---

## 1. El caso de los modelos locales

Desde 2023 el ecosistema de modelos open-source ha madurado radicalmente. Hoy es posible ejecutar en un portátil modelos que rivalizan en muchas tareas con GPT-3.5, y en hardware de gama media superar a GPT-4 en dominios especializados.

Esto no es solo una curiosidad técnica: **cambia qué problemas son viables y para quién**.

### El modelo mental equivocado

La mayoría de personas asume que "modelos locales = peor calidad". La realidad es más matizada:

```
               Calidad en tareas generales
               ────────────────────────────
  GPT-4o       ████████████████████  ← benchmark de referencia
  Claude 3.7   ███████████████████
  Llama 3.1 70B ████████████████     ← casi al nivel, sin coste
  Qwen 2.5 32B  ██████████████
  Llama 3.2 8B  ████████████         ← suficiente para la mayoría de tareas
  Phi-4 Mini    ██████████           ← corre en CPU

  Ventaja cloud: razonamiento complejo, conocimiento actualizado
  Ventaja local: privacidad, coste cero, latencia, control total
```

---

## 2. Ventajas en detalle

### 2.1 Privacidad total — los datos no salen de tu máquina

Esta es la ventaja más importante para muchos casos de uso.

Con un modelo cloud:
```
Tu datos  ──► API cloud ──► Procesamiento en servidores externos
                ↑
        Términos de servicio, logs, posible uso en entrenamiento,
        regulación de datos (GDPR, HIPAA, NDA), jurisdicción extranjera
```

Con un modelo local:
```
Tus datos ──► Ollama en localhost ──► Respuesta
                    ↑
              Todo ocurre en tu hardware.
              Cero transmisión de datos.
```

**Casos donde esto es crítico:**
- Documentos legales confidenciales (contratos, litigios)
- Datos médicos (historiales, diagnósticos)
- Código propietario o secretos industriales
- Datos de clientes bajo GDPR o normativas sectoriales
- Entornos con restricciones de red (finanzas, defensa)
- Empresas con políticas de seguridad que prohíben enviar datos a terceros

### 2.2 Coste cero en inferencia

```
Modelo cloud (estimación para 1 millón de tokens):
  GPT-4o:           ~15 €
  Claude Sonnet:    ~9 €
  GPT-3.5-turbo:    ~0.75 €

Modelo local (Llama 3.1 8B en tu máquina):
  Primer millón de tokens:  0 €
  Siguiente millón:         0 €
  Todos los millones:       0 €
  (solo el coste de electricidad: ~0.01 €/hora)
```

Esto cambia completamente la economía para:
- Procesar grandes volúmenes de texto (miles de documentos)
- Prototipar sin preocuparse por costes
- Aplicaciones con uso intensivo de LLM
- Startups en fase pre-revenue
- Proyectos educativos o de investigación

### 2.3 Latencia: sin red, sin colas, sin throttling

```python
# Modelo cloud: tu petición viaja al datacenter y vuelve
# Latencia típica: 500ms - 5s primer token

# Modelo local: procesamiento en tu propia RAM/GPU
# Latencia típica: 50ms - 500ms primer token (depende del hardware)
```

Sin red no hay:
- Variabilidad por congestión del servidor
- Degradación en horas pico
- Timeouts inesperados
- Rate limiting

Fundamental para: aplicaciones de escritorio, plugins de IDE, asistentes en tiempo real, juegos.

### 2.4 Disponibilidad offline

Un modelo local funciona:
- Sin conexión a internet
- En avión, tren, zonas sin cobertura
- En entornos air-gapped (redes corporativas aisladas)
- Durante outages del proveedor cloud
- En regiones con restricciones de acceso a APIs externas

### 2.5 Sin límites de rate y sin cuotas

Las APIs cloud imponen límites:
```
OpenAI free tier:   3 peticiones/minuto
Anthropic:          varía por tier, puede llegar a 1000/min en enterprise
```

Un modelo local no tiene rate limiting. Puedes:
- Lanzar 100 peticiones simultáneas
- Procesar un corpus de 10.000 documentos de noche sin supervisión
- Hacer benchmark y experimentación intensiva sin preocuparte por costes

### 2.6 Reproducibilidad y control de versiones

Con un modelo cloud:
- La API puede actualizarse sin aviso
- GPT-4 de enero 2024 ≠ GPT-4 de enero 2025
- El comportamiento puede cambiar entre versiones

Con un modelo local:
```bash
ollama pull llama3.2:8b   # Descarga el modelo exacto
# Este modelo nunca cambia a menos que tú lo actualices
# Puedes "congelar" una versión para producción
```

Crucial para: pipelines de evaluación reproducibles, auditorías, compliance, investigación.

### 2.7 Personalización y fine-tuning propio

Puedes:
1. Partir de un modelo base open-source
2. Fine-tunearlo con tus datos propios (con LoRA/QLoRA)
3. Convertirlo al formato GGUF
4. Desplegarlo con Ollama

```bash
# Crear un Modelfile para tu modelo personalizado
cat > Modelfile << 'EOF'
FROM llama3.2

PARAMETER temperature 0.3
SYSTEM """Eres un asistente experto en derecho español. 
Respondes siempre citando artículos del Código Civil cuando es relevante."""
EOF

ollama create mi-asistente-legal -f Modelfile
ollama run mi-asistente-legal
```

### 2.8 Experimentación sin fricción

Cuando el coste por token es cero:
- Pruebas A/B de prompts con miles de ejemplos
- Benchmark entre distintos modelos
- Generación de datasets sintéticos a gran escala
- Iterar rápido sin miedo a la factura

---

## 3. Qué es Ollama

**Ollama** es una herramienta open-source que simplifica radicalmente la ejecución de modelos de lenguaje en local. Antes de Ollama, ejecutar un LLM requería instalar CUDA, compilar llama.cpp, gestionar modelos manualmente. Ahora:

```bash
ollama run llama3.2
# ← esto es todo. Descarga y ejecuta un LLM de 8B parámetros.
```

Ollama proporciona:
- **CLI**: interfaz de línea de comandos para gestionar y ejecutar modelos
- **Servidor REST**: API en `localhost:11434` compatible con la API de OpenAI
- **Biblioteca Python**: cliente oficial `ollama`
- **Gestión de modelos**: descarga, actualización, eliminación
- **Soporte de hardware**: CPU, Apple Silicon (Metal), NVIDIA CUDA, AMD ROCm

### Arquitectura interna

```
┌─────────────────────────────────────────────────┐
│                  Tu aplicación                  │
│   (Python, Node.js, curl, LangChain, etc.)      │
└──────────────────┬──────────────────────────────┘
                   │ HTTP (localhost:11434)
┌──────────────────▼──────────────────────────────┐
│               Servidor Ollama                   │
│  • Gestión de modelos (pull, list, delete)      │
│  • Cola de peticiones                           │
│  • API REST compatible con OpenAI               │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           Motor de inferencia (llama.cpp)       │
│  • CPU: AVX2/AVX512                             │
│  • GPU: CUDA / Metal / ROCm                     │
│  • Cuantización: Q4_K_M, Q8_0, F16...          │
└─────────────────────────────────────────────────┘
```

---

## 4. Instalación y primeros pasos

### 4.1 Instalación

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Descarga el instalador desde https://ollama.com/download/windows

# Verificar instalación
ollama --version
```

### 4.2 Comandos esenciales

```bash
# Descargar un modelo
ollama pull llama3.2

# Ejecutar un modelo (descarga si no está)
ollama run llama3.2

# Ejecutar con un prompt directo
ollama run llama3.2 "¿Cuál es la capital de España?"

# Ver modelos descargados
ollama list

# Información detallada de un modelo
ollama show llama3.2

# Eliminar un modelo
ollama rm llama3.2

# Ver procesos en ejecución
ollama ps
```

### 4.3 El servidor

Ollama inicia automáticamente un servidor en `http://localhost:11434`:

```bash
# Verificar que el servidor está activo
curl http://localhost:11434/api/tags

# Generar texto (API básica)
curl http://localhost:11434/api/generate \
  -d '{"model": "llama3.2", "prompt": "Hola", "stream": false}'
```

### 4.4 Modelfiles — personalizar el comportamiento

```bash
# Crear un asistente especializado
cat > Modelfile.asistente << 'EOF'
FROM llama3.2

# Temperatura más baja = respuestas más deterministas
PARAMETER temperature 0.2
PARAMETER num_ctx 8192

SYSTEM """Eres un asistente técnico especializado en Python y machine learning.
- Responde siempre en español
- Incluye ejemplos de código cuando sea útil
- Si no sabes algo, dilo claramente en lugar de inventar"""
EOF

ollama create asistente-python -f Modelfile.asistente
ollama run asistente-python
```

---

## 5. Catálogo de modelos

### 5.1 Modelos por caso de uso

| Modelo | Tamaño | Ideal para | Hardware mínimo |
|---|---|---|---|
| **Llama 3.2** | 1B / 3B | Tareas ligeras, edge, dispositivos móviles | 4 GB RAM |
| **Llama 3.2** | 8B | Uso general, chat, código | 8 GB RAM |
| **Llama 3.1** | 70B | Tareas complejas, razonamiento | 48 GB RAM / GPU |
| **Mistral** | 7B | Chat rápido, instrucciones | 8 GB RAM |
| **Qwen 2.5** | 7B / 14B / 32B | Código, matemáticas, multilingüe | 8-24 GB RAM |
| **Phi-4** | 3.8B | Código, razonamiento compacto | 4 GB RAM |
| **Gemma 2** | 2B / 9B / 27B | Chat, instrucciones, Google stack | 4-16 GB RAM |
| **DeepSeek-R1** | 7B / 14B / 70B | Razonamiento matemático y lógico | 8-48 GB RAM |
| **CodeLlama** | 7B / 13B / 34B | Generación y análisis de código | 8-24 GB RAM |
| **Nomic Embed** | — | Embeddings para RAG | 4 GB RAM |

### 5.2 Variantes de cuantización

Los modelos se distribuyen en distintos niveles de cuantización. A menor precisión, menos RAM y más velocidad, con cierta pérdida de calidad:

```bash
# Cuantización completa (mayor calidad, más RAM)
ollama pull llama3.1:8b-instruct-fp16    # ~16 GB

# Cuantización 8-bit (buen equilibrio)
ollama pull llama3.1:8b-instruct-q8_0   # ~8 GB

# Cuantización 4-bit (recomendada en la mayoría de casos)
ollama pull llama3.1:8b-instruct-q4_K_M # ~5 GB ← punto dulce

# Cuantización extrema (hardware muy limitado)
ollama pull llama3.1:8b-instruct-q2_K   # ~3 GB
```

**Regla práctica:** Q4_K_M es el punto óptimo entre calidad y eficiencia para la mayoría de modelos y casos.

---

## 6. Integración con Python

### 6.1 Instalación

```bash
pip install ollama
```

### 6.2 Generación básica

```python
import ollama

# Respuesta completa
respuesta = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "¿Qué es un LLM?"}]
)
print(respuesta["message"]["content"])
```

### 6.3 Streaming

```python
import ollama

# Streaming: recibe tokens según se generan
stream = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explica el patrón ReAct en agentes de IA."}],
    stream=True
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()  # salto de línea final
```

### 6.4 Conversación con historial

```python
import ollama

def chat_local(model: str = "llama3.2"):
    """Chat conversacional con modelo local."""
    historial = []
    print(f"Chat con {model} (escribe 'salir' para terminar)\n")

    while True:
        entrada = input("Tú: ").strip()
        if entrada.lower() in ("salir", "exit", "quit"):
            break
        if not entrada:
            continue

        historial.append({"role": "user", "content": entrada})

        respuesta = ollama.chat(model=model, messages=historial)
        mensaje_asistente = respuesta["message"]["content"]
        historial.append({"role": "assistant", "content": mensaje_asistente})

        print(f"\nAsistente: {mensaje_asistente}\n")

chat_local("llama3.2")
```

### 6.5 Sistema de RAG local — completamente offline

```python
import ollama

# Embeddings con un modelo local (sin enviar datos a ningún servidor)
def embedding_local(texto: str, model: str = "nomic-embed-text") -> list[float]:
    respuesta = ollama.embeddings(model=model, prompt=texto)
    return respuesta["embedding"]

# Base de conocimiento en memoria
class BaseConocimientoLocal:
    def __init__(self, model_embed: str = "nomic-embed-text"):
        self.model_embed = model_embed
        self.documentos: list[str] = []
        self.embeddings: list[list[float]] = []

    def añadir(self, texto: str):
        self.documentos.append(texto)
        self.embeddings.append(embedding_local(texto, self.model_embed))

    def buscar(self, consulta: str, top_k: int = 3) -> list[str]:
        """Búsqueda por similitud coseno."""
        import math
        emb_consulta = embedding_local(consulta, self.model_embed)

        def coseno(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x**2 for x in a))
            norm_b = math.sqrt(sum(x**2 for x in b))
            return dot / (norm_a * norm_b + 1e-9)

        puntuaciones = [(coseno(emb_consulta, e), doc)
                       for e, doc in zip(self.embeddings, self.documentos)]
        puntuaciones.sort(reverse=True)
        return [doc for _, doc in puntuaciones[:top_k]]

    def preguntar(self, pregunta: str, model_llm: str = "llama3.2") -> str:
        """RAG completo: recuperar + generar. Todo en local."""
        contextos = self.buscar(pregunta)
        contexto_combinado = "\n\n".join(contextos)

        prompt = f"""Contexto:
{contexto_combinado}

Pregunta: {pregunta}

Responde basándote en el contexto anterior. Si la información no está en el contexto, dilo."""

        respuesta = ollama.chat(
            model=model_llm,
            messages=[{"role": "user", "content": prompt}]
        )
        return respuesta["message"]["content"]

# Ejemplo de uso
base = BaseConocimientoLocal()
base.añadir("LangGraph es un framework para construir agentes con grafos de estado. Permite ciclos, checkpointing y human-in-the-loop.")
base.añadir("Ollama permite ejecutar modelos de lenguaje en local sin enviar datos a servicios externos.")
base.añadir("RAG combina recuperación de información con generación de texto para respuestas más precisas y actualizadas.")

respuesta = base.preguntar("¿Qué ventajas tiene usar Ollama para un sistema RAG?")
print(respuesta)
```

### 6.6 Generación estructurada

```python
import ollama
import json

def generar_estructurado(texto: str, esquema: dict, model: str = "llama3.2") -> dict:
    """Extrae información estructurada con un modelo local."""
    prompt = f"""Analiza el siguiente texto y extrae la información en formato JSON.
Esquema esperado: {json.dumps(esquema, ensure_ascii=False)}

Texto: {texto}

Responde ÚNICAMENTE con el JSON, sin texto adicional."""

    respuesta = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format="json"  # ← fuerza salida JSON
    )
    return json.loads(respuesta["message"]["content"])

# Ejemplo
texto_cv = """
Ana García, 32 años, lleva 5 años trabajando como ingeniera de software en Barcelona.
Especializada en Python y sistemas distribuidos. Habla español e inglés.
"""
esquema = {
    "nombre": "string",
    "edad": "number",
    "ciudad": "string",
    "anos_experiencia": "number",
    "especialidad": "string",
    "idiomas": ["string"]
}

resultado = generar_estructurado(texto_cv, esquema)
print(json.dumps(resultado, indent=2, ensure_ascii=False))
```

---

## 7. API compatible con OpenAI

Ollama expone una API REST compatible con la API de OpenAI. Esto significa que **cualquier código que use la librería openai puede apuntar a Ollama con un cambio mínimo**:

```python
from openai import OpenAI

# Solo cambia base_url y api_key
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # cualquier string, no se valida
)

# El resto del código es idéntico al de la API de OpenAI
respuesta = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "¿Qué es el RAG?"}
    ]
)
print(respuesta.choices[0].message.content)
```

### Migración de OpenAI a Ollama

```python
# ANTES: OpenAI cloud
from openai import OpenAI
client = OpenAI()  # usa OPENAI_API_KEY

# DESPUÉS: Ollama local (2 líneas de cambio)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# El resto de tu código: sin cambios
```

---

## 8. Integración con LangChain

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── LLM local ─────────────────────────────────────────────────────────────────
llm_local = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    num_ctx=4096,       # tamaño de contexto
    num_predict=512,    # máximo de tokens a generar
)

# Uso directo
respuesta = llm_local.invoke([HumanMessage(content="¿Qué es LangGraph?")])
print(respuesta.content)

# ── Cadena LCEL ────────────────────────────────────────────────────────────────
plantilla = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {dominio}. Responde de forma concisa."),
    ("human", "{pregunta}")
])

cadena = plantilla | llm_local | StrOutputParser()

resultado = cadena.invoke({
    "dominio": "inteligencia artificial",
    "pregunta": "¿Cuál es la diferencia entre RAG y fine-tuning?"
})
print(resultado)

# ── Embeddings locales ─────────────────────────────────────────────────────────
embeddings_local = OllamaEmbeddings(model="nomic-embed-text")

# Usar con cualquier vector store de LangChain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

docs = [
    Document(page_content="LangGraph es un framework para agentes con grafos de estado."),
    Document(page_content="Ollama permite ejecutar LLMs en local de forma sencilla."),
    Document(page_content="RAG combina recuperación e inferencia para respuestas precisas."),
]

vectorstore = Chroma.from_documents(docs, embedding=embeddings_local)
resultados = vectorstore.similarity_search("¿Cómo ejecuto un LLM localmente?", k=2)
for doc in resultados:
    print(doc.page_content)
```

### LangGraph con modelo local

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated

# El mismo grafo del tutorial 08 pero con modelo local
llm_local = ChatOllama(model="llama3.2", temperature=0.2)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def nodo_llm(state: State) -> dict:
    return {"messages": [llm_local.invoke(state["messages"])]}

grafo = StateGraph(State)
grafo.add_node("llm", nodo_llm)
grafo.set_entry_point("llm")
grafo.add_edge("llm", END)
app = grafo.compile()

resultado = app.invoke({
    "messages": [HumanMessage(content="Explica qué es un agente de IA.")]
})
print(resultado["messages"][-1].content)
```

---

## 9. Rendimiento y optimización

### 9.1 CPU vs GPU

```
Velocidad de generación (tokens/segundo) — Llama 3.2 8B Q4_K_M

CPU (i7-12700H, 16 cores):       ~10-20 tok/s   → conversación fluida
Apple M2 Pro (Metal):            ~40-60 tok/s   → muy rápido
NVIDIA RTX 3080 (10 GB VRAM):    ~60-90 tok/s   → producción
NVIDIA RTX 4090 (24 GB VRAM):   ~100-130 tok/s  → referencia
```

### 9.2 Parámetros de configuración

```bash
# Configurar Ollama vía variables de entorno
export OLLAMA_NUM_PARALLEL=4      # peticiones paralelas
export OLLAMA_MAX_LOADED_MODELS=2 # modelos en VRAM simultáneos
export OLLAMA_KEEP_ALIVE="10m"    # tiempo que el modelo permanece cargado

# En el Modelfile
PARAMETER num_ctx 8192      # contexto más grande = más RAM
PARAMETER num_gpu 1         # capas en GPU (0 = solo CPU)
PARAMETER num_thread 8      # hilos de CPU
```

### 9.3 Elegir el modelo correcto para tu hardware

```python
GUIA_HARDWARE = {
    "4 GB RAM":  ["phi4-mini", "llama3.2:1b", "llama3.2:3b"],
    "8 GB RAM":  ["llama3.2:8b", "mistral:7b", "gemma2:2b", "qwen2.5:7b"],
    "16 GB RAM": ["llama3.1:8b (fp16)", "qwen2.5:14b", "gemma2:9b"],
    "32 GB RAM": ["llama3.1:70b (q4)", "qwen2.5:32b", "deepseek-r1:14b"],
    "64 GB RAM": ["llama3.1:70b (q8)", "deepseek-r1:70b (q4)"],
}
```

---

## 10. Limitaciones y cuándo usar cloud

Los modelos locales no son superiores en todo. Sus principales limitaciones actuales:

| Limitación | Detalle | Alternativa |
|---|---|---|
| Capacidad de razonamiento | Los modelos grandes cloud (GPT-4o, Claude 3.7) siguen siendo mejores en tareas complejas | Cloud para razonamiento crítico |
| Conocimiento actualizado | Los modelos locales tienen fecha de corte; no navegan internet | RAG + búsqueda web |
| Multimodalidad | Soporte de visión e imagen más limitado | Cloud para visión |
| Tamaño del contexto | La mayoría de modelos locales: 4K-128K tokens | Cloud para contextos muy largos |
| Coste de hardware inicial | GPU de gama alta: 500-2000 € | Cloud para experimentación inicial |
| Instalación y mantenimiento | Requiere gestión del entorno local | Cloud para equipos sin DevOps |

---

## 11. Guía de decisión: local vs cloud

```
¿Los datos son confidenciales o hay restricciones legales?
    SÍ ──► Modelo local (obligatorio)
    NO ──► continúa

¿El volumen de peticiones es muy alto o el presupuesto es ajustado?
    SÍ ──► Modelo local
    NO ──► continúa

¿Necesitas la máxima calidad en razonamiento complejo?
    SÍ ──► Cloud (GPT-4o, Claude 3.7 Sonnet)
    NO ──► continúa

¿La aplicación debe funcionar offline?
    SÍ ──► Modelo local
    NO ──► continúa

¿Estás prototipando o experimentando?
    SÍ ──► Modelo local (coste cero, experimentación libre)
    NO ──► evalúa el caso específico
```

### Combinación híbrida

La estrategia más robusta combina ambos:

```python
def elegir_modelo(tarea: str, datos_sensibles: bool) -> str:
    """Selecciona el modelo según el contexto."""
    if datos_sensibles:
        return "ollama/llama3.1:70b"        # siempre local si hay datos sensibles

    tareas_complejas = ["análisis legal", "diagnóstico médico", "razonamiento matemático avanzado"]
    if any(t in tarea.lower() for t in tareas_complejas):
        return "claude-sonnet-4-6"          # cloud para razonamiento complejo

    return "ollama/llama3.2:8b"             # local para todo lo demás
```

---

## 12. Resumen

Ollama democratiza el acceso a modelos de lenguaje avanzados. Las ventajas clave:

| Ventaja | Impacto |
|---|---|
| **Privacidad total** | Los datos nunca salen de tu máquina |
| **Coste cero** | Sin factura por tokens, sin sorpresas |
| **Sin rate limiting** | Procesa volúmenes arbitrarios |
| **Offline** | Funciona sin internet ni APIs externas |
| **Reproducible** | El modelo no cambia salvo que tú decidas |
| **Personalizable** | Fine-tuning propio + Modelfiles |
| **Compatible** | API OpenAI-compatible, se integra con cualquier stack |

```bash
# En 3 comandos tienes un LLM local:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama run llama3.2
```

### Recursos

- [Ollama — sitio oficial](https://ollama.com)
- [Biblioteca de modelos disponibles](https://ollama.com/library)
- [Repositorio GitHub de Ollama](https://github.com/ollama/ollama)
- [Librería Python oficial](https://github.com/ollama/ollama-python)
- Tutorial anterior: [08 — LangGraph](./08-langgraph.md)
- Tutorial siguiente: [10 — MCP y conexiones con herramientas externas](./10-mcp.md)
- Notebook interactivo: [09-modelos-locales-ollama.ipynb](../notebooks/llms/09-modelos-locales-ollama.ipynb)
