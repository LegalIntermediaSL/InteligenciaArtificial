# 03 — Comparativa de proveedores de IA

> **Bloque:** APIs de IA · **Nivel:** Intermedio · **Tiempo estimado:** 20 min

---

## Índice

1. [Panorama de proveedores](#1-panorama-de-proveedores)
2. [Comparativa técnica](#2-comparativa-técnica)
3. [Comparativa de precios](#3-comparativa-de-precios)
4. [Fortalezas y casos de uso](#4-fortalezas-y-casos-de-uso)
5. [Modelos open source](#5-modelos-open-source)
6. [Consideraciones de privacidad y compliance](#6-consideraciones-de-privacidad-y-compliance)
7. [Cómo elegir proveedor](#7-cómo-elegir-proveedor)
8. [Abstracción multi-proveedor con LiteLLM](#8-abstracción-multi-proveedor-con-litellm)

---

## 1. Panorama de proveedores

El ecosistema de APIs de LLMs está dominado por un puñado de actores principales, con un creciente número de alternativas open source.

```
PROPIETARIOS (API as a Service)
├── Anthropic  → Claude (Opus, Sonnet, Haiku)
├── OpenAI     → GPT-4, o1, o3
├── Google     → Gemini (Ultra, Pro, Flash)
├── Mistral    → Mistral Large, Small, Nemo
└── Cohere     → Command R+, Embed

OPEN SOURCE (autoalojado o vía API)
├── Meta       → Llama 3 (8B, 70B, 405B)
├── Mistral    → Mistral 7B, Mixtral 8x7B (también open)
├── Alibaba    → Qwen 2.5
├── DeepSeek   → DeepSeek V3, R1
└── Microsoft  → Phi-3, Phi-4

PLATAFORMAS DE ACCESO UNIFICADO
├── Together AI  → Múltiples modelos open source
├── Groq         → Inferencia ultra-rápida (LPU)
├── Fireworks    → Open source a alta velocidad
└── AWS Bedrock  → Claude, Llama, Titan y más
```

---

## 2. Comparativa técnica

### Ventana de contexto

| Proveedor / Modelo | Contexto |
|---|---|
| Claude 3 (Anthropic) | **200K tokens** |
| Gemini 1.5 Pro (Google) | **1M tokens** |
| GPT-4o (OpenAI) | 128K tokens |
| Llama 3.1 405B | 128K tokens |
| Mistral Large | 128K tokens |
| GPT-3.5 Turbo | 16K tokens |

> Mayor contexto = mejor para procesar documentos largos, libros, bases de código extensas.

### Multimodalidad

| Capacidad | Anthropic | OpenAI | Google |
|---|---|---|---|
| Texto | ✅ | ✅ | ✅ |
| Imágenes | ✅ | ✅ | ✅ |
| Audio | ❌ (nativo) | ✅ (Whisper/TTS) | ✅ |
| Vídeo | ❌ | ❌ | ✅ (Gemini) |
| Generación de imágenes | ❌ | ✅ (DALL·E) | ✅ (Imagen) |
| Código de ejecución | ❌ | ✅ (Code Interpreter) | ✅ |

### Velocidad de respuesta (latencia aproximada)

| Escenario | Rápido | Medio | Lento |
|---|---|---|---|
| Respuesta primera token | Haiku, GPT-4o mini | Sonnet, GPT-4o | Opus, GPT-4 Turbo |
| Tokens por segundo | Groq (LPU) | Anthropic/OpenAI | Modelos locales |

---

## 3. Comparativa de precios

Precios aproximados en abril 2026 (en USD por millón de tokens):

### Modelos de alta capacidad

| Modelo | Input | Output | Ratio output/input |
|---|---|---|---|
| Claude Opus 4.6 | $15 | $75 | 5x |
| GPT-4o | $2.50 | $10 | 4x |
| Gemini 1.5 Pro | $3.50 | $10.50 | 3x |
| Mistral Large | $4 | $12 | 3x |

### Modelos eficientes (mejor relación calidad/precio)

| Modelo | Input | Output |
|---|---|---|
| Claude Sonnet 4.6 | $3 | $15 |
| GPT-4o mini | $0.15 | $0.60 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Claude Haiku 4.5 | $0.25 | $1.25 |
| Mistral Small | $1 | $3 |

> Los precios cambian frecuentemente. Consulta siempre la documentación oficial.

### Estimación de costes por caso de uso

```
Supuesto: 1.000 consultas/día, prompt promedio 500 tokens, respuesta 300 tokens

Claude Haiku:    1.000 × (500×0.00000025 + 300×0.00000125) = $0.50/día = $15/mes
GPT-4o mini:     1.000 × (500×0.00000015 + 300×0.00000060) = $0.26/día = $7.7/mes
Claude Sonnet:   1.000 × (500×0.000003   + 300×0.000015)   = $6/día   = $180/mes
GPT-4o:          1.000 × (500×0.0000025  + 300×0.00001)    = $4.25/día = $127/mes
```

---

## 4. Fortalezas y casos de uso

### Anthropic (Claude)

**Fortalezas:**
- Seguimiento de instrucciones muy preciso
- Razonamiento en documentos largos (contexto 200K)
- Seguridad y reducción de respuestas dañinas (Constitutional AI)
- Escritura de alta calidad y coherencia
- Excelente para tareas que requieren matiz y cuidado

**Mejores casos de uso:**
- Análisis de documentos largos (contratos, informes, libros)
- Asistentes con instrucciones complejas
- Aplicaciones donde la seguridad es crítica
- Redacción profesional y edición
- Tareas de razonamiento multi-paso

---

### OpenAI (GPT-4)

**Fortalezas:**
- Ecosistema más maduro y amplia adopción
- Multimodalidad completa (texto, imagen, audio, código)
- Function calling muy robusto
- DALL·E para generación de imágenes
- Whisper para transcripción de audio

**Mejores casos de uso:**
- Aplicaciones que necesitan imagen + texto
- Transcripción y procesamiento de audio
- Generación de imágenes integrada
- Function calling en flujos de trabajo complejos
- Proyectos que se benefician de la comunidad más grande

---

### Google (Gemini)

**Fortalezas:**
- Contexto más largo del mercado (1M tokens en Gemini 1.5 Pro)
- Integración nativa con servicios Google (Docs, Drive, Search)
- Modelos Flash muy económicos y rápidos
- Capacidades de vídeo únicas

**Mejores casos de uso:**
- Procesar libros o bases de código completas de una vez
- Integración con workspace de Google
- Aplicaciones que requieren análisis de vídeo
- Proyectos en GCP (créditos, integración)

---

### Mistral

**Fortalezas:**
- Modelos open source de alta calidad (Mistral 7B, Mixtral)
- Opción europea (cumplimiento GDPR más sencillo)
- Modelos pequeños sorprendentemente capaces
- Precio competitivo

**Mejores casos de uso:**
- Proyectos con restricciones de datos europeos
- Fine-tuning (modelos open source)
- Autoalojamiento (privacidad total)
- Presupuesto ajustado

---

## 5. Modelos open source

Si los modelos propietarios no encajan (privacidad, coste, control total), los open source son una alternativa seria.

### Modelos destacados

| Modelo | Parámetros | Destacado |
|---|---|---|
| **Llama 3.1 405B** | 405B | El más capaz; comparable a GPT-4 |
| **Llama 3.1 70B** | 70B | Mejor relación calidad/coste open source |
| **Mistral 7B** | 7B | Increíble para su tamaño |
| **Mixtral 8x7B** | ~47B activos | Arquitectura MoE, muy eficiente |
| **Qwen 2.5 72B** | 72B | Excelente en multilingüe y código |
| **DeepSeek V3** | ~670B | Competitivo con GPT-4, open weights |
| **Phi-4** | 14B | Pequeño pero muy capaz en razonamiento |

### Cómo ejecutarlos

**Opción 1 — Ollama (local, fácil):**
```bash
# Instalar Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar y ejecutar un modelo
ollama run llama3.1:8b
ollama run mistral
```

**Opción 2 — Hugging Face Transformers:**
```python
from transformers import pipeline

generador = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
resultado = generador("¿Qué es la IA?", max_new_tokens=200)
print(resultado[0]["generated_text"])
```

**Opción 3 — API de Together AI (open source en la nube):**
```python
from openai import OpenAI  # Together es compatible con la API de OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://api.together.xyz/v1"
)
completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hola"}]
)
```

---

## 6. Consideraciones de privacidad y compliance

| Consideración | Qué evaluar |
|---|---|
| **Residencia de datos** | ¿Dónde se procesan y almacenan los datos? |
| **Retención de datos** | ¿Cuánto tiempo guarda el proveedor tus prompts? |
| **Uso para entrenamiento** | ¿Usa el proveedor tus datos para entrenar sus modelos? |
| **GDPR** | ¿Tiene DPA (Data Processing Agreement) disponible? |
| **Sector regulado** | Salud (HIPAA), finanzas (SOX/PCI), legal... |

**Para datos sensibles o regulados:**
- Anthropic y OpenAI ofrecen acuerdos de procesamiento de datos (DPA)
- AWS Bedrock, Azure OpenAI y Google Vertex AI añaden capa de cumplimiento empresarial
- Para privacidad total: modelos open source autoalojados

---

## 7. Cómo elegir proveedor

```
¿Tienes restricciones de privacidad o datos sensibles?
    │
    ├── SÍ, datos muy sensibles → Open source autoalojado (Llama, Mistral)
    │
    ├── SÍ, pero puedes usar cloud → AWS Bedrock / Azure OpenAI (con DPA)
    │
    └── NO
         │
         ├── ¿Necesitas procesar documentos muy largos (>100K tokens)?
         │       ├── SÍ → Gemini 1.5 Pro (1M) o Claude (200K)
         │       └── NO → Continúa
         │
         ├── ¿Necesitas imagen, audio o vídeo?
         │       ├── Audio/imagen → OpenAI
         │       ├── Vídeo → Gemini
         │       └── NO → Continúa
         │
         ├── ¿El coste es muy crítico?
         │       ├── SÍ → GPT-4o mini, Gemini Flash, Claude Haiku
         │       └── NO → Claude Sonnet o GPT-4o para uso general
         │
         └── ¿Prioridad? → Calidad razonamiento: Claude Opus / GPT-4o
                           Ecosistema/herramientas: OpenAI
                           Contexto largo: Google Gemini
```

---

## 8. Abstracción multi-proveedor con LiteLLM

Si quieres cambiar de proveedor sin reescribir código, usa **LiteLLM**:

```bash
pip install litellm
```

```python
from litellm import completion

# Mismo código, diferentes proveedores
def llamar_llm(proveedor: str, prompt: str) -> str:
    modelos = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
        "google": "gemini/gemini-1.5-pro",
        "mistral": "mistral/mistral-large-latest"
    }
    
    response = completion(
        model=modelos[proveedor],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Cambiar de proveedor es trivial
print(llamar_llm("anthropic", "Explica qué es RAG"))
print(llamar_llm("openai", "Explica qué es RAG"))
```

---

**Anterior:** [02 — API de OpenAI](./02-api-openai.md) · **Siguiente bloque:** [Python para IA](../python-para-ia/)
