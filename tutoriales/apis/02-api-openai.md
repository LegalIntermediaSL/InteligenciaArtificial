# 02 — API de OpenAI

> **Bloque:** APIs de IA · **Nivel:** Práctico · **Tiempo estimado:** 25 min

---

## Índice

1. [Introducción](#1-introducción)
2. [Configuración inicial](#2-configuración-inicial)
3. [Chat Completions](#3-chat-completions)
4. [System prompt y roles](#4-system-prompt-y-roles)
5. [Parámetros principales](#5-parámetros-principales)
6. [Streaming](#6-streaming)
7. [Function Calling](#7-function-calling)
8. [Embeddings](#8-embeddings)
9. [Modelos de imagen (DALL·E)](#9-modelos-de-imagen-dalle)
10. [Manejo de errores](#10-manejo-de-errores)
11. [Buenas prácticas y costes](#11-buenas-prácticas-y-costes)

---

## 1. Introducción

La API de OpenAI da acceso a los modelos **GPT-4** y variantes, así como a modelos especializados para embeddings, imágenes (DALL·E) y audio (Whisper, TTS).

**Modelos principales (abril 2026):**

| Modelo | ID | Contexto | Uso recomendado |
|---|---|---|---|
| GPT-4o | `gpt-4o` | 128K tokens | Uso general, multimodal |
| GPT-4o mini | `gpt-4o-mini` | 128K tokens | Tareas simples, bajo coste |
| GPT-4 Turbo | `gpt-4-turbo` | 128K tokens | Razonamiento complejo |
| o1 | `o1` | 200K tokens | Razonamiento matemático y científico |
| o3-mini | `o3-mini` | 200K tokens | Razonamiento eficiente |

---

## 2. Configuración inicial

### Instalar el SDK

```bash
pip install openai
```

### Obtener una API key

1. Ve a [platform.openai.com](https://platform.openai.com)
2. Crea una cuenta o inicia sesión
3. En el panel, ve a **API keys** → **Create new secret key**
4. Copia la clave (solo se muestra una vez)

### Configurar la clave

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

client = OpenAI()  # Lee OPENAI_API_KEY automáticamente
```

---

## 3. Chat Completions

El endpoint principal de la API de OpenAI es `chat.completions.create`.

```python
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Explica qué es la inteligencia artificial en 3 frases."}
    ]
)

print(completion.choices[0].message.content)
```

### Estructura de la respuesta

```python
print(completion.id)                          # ID único
print(completion.model)                       # Modelo usado
print(completion.choices[0].finish_reason)    # "stop", "length", "tool_calls"...
print(completion.usage.prompt_tokens)         # Tokens de entrada
print(completion.usage.completion_tokens)     # Tokens de salida
print(completion.usage.total_tokens)          # Total
print(completion.choices[0].message.content)  # Texto de respuesta
```

### Conversación con historial

```python
historial = []

def chat(mensaje):
    historial.append({"role": "user", "content": mensaje})
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=historial
    )
    
    respuesta = completion.choices[0].message.content
    historial.append({"role": "assistant", "content": respuesta})
    return respuesta

print(chat("Hola, cuéntame algo sobre los LLMs"))
print(chat("¿Puedes dar más detalles sobre lo que acabas de decir?"))
```

---

## 4. System prompt y roles

OpenAI usa tres roles en los mensajes:

| Rol | Descripción |
|---|---|
| `system` | Instrucciones globales para el comportamiento del modelo |
| `user` | Mensajes del usuario |
| `assistant` | Respuestas anteriores del modelo (para mantener historial) |

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """Eres un experto en marketing digital.
            Responde siempre en español.
            Usa ejemplos concretos cuando sea posible.
            Sé directo y práctico."""
        },
        {
            "role": "user",
            "content": "¿Qué estrategias de contenido funcionan mejor en LinkedIn?"
        }
    ]
)
```

---

## 5. Parámetros principales

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    max_tokens=2048,          # Máximo tokens en respuesta
    temperature=0.7,          # Aleatoriedad (0.0-2.0 en OpenAI)
    top_p=0.9,                # Nucleus sampling
    frequency_penalty=0.0,    # Penaliza repetición de tokens frecuentes (-2 a 2)
    presence_penalty=0.0,     # Penaliza tokens que ya aparecieron (-2 a 2)
    stop=["###", "\n\n\n"],   # Secuencias de parada
    n=1,                      # Número de respuestas a generar
    seed=42,                  # Para reproducibilidad (best effort)
)
```

| Parámetro | Uso |
|---|---|
| `frequency_penalty > 0` | Reduce repetición de palabras |
| `presence_penalty > 0` | Fomenta hablar de temas nuevos |
| `n > 1` | Genera múltiples respuestas (self-consistency) |
| `seed` | Intenta respuestas reproducibles |

---

## 6. Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Escribe un poema sobre la IA"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 7. Function Calling

El function calling permite que GPT llame funciones definidas por ti.

```python
import json

# Definición de funciones
functions = [
    {
        "name": "buscar_producto",
        "description": "Busca información sobre un producto en el catálogo",
        "parameters": {
            "type": "object",
            "properties": {
                "nombre_producto": {
                    "type": "string",
                    "description": "Nombre o descripción del producto"
                },
                "categoria": {
                    "type": "string",
                    "enum": ["electronica", "ropa", "hogar", "alimentacion"],
                    "description": "Categoría del producto"
                }
            },
            "required": ["nombre_producto"]
        }
    }
]

# Función real
def buscar_producto(nombre_producto: str, categoria: str = None) -> dict:
    # Simulación — en producción esto consultaría una BD real
    return {
        "producto": nombre_producto,
        "precio": 29.99,
        "stock": 15,
        "categoria": categoria or "general"
    }

# Primera llamada
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "¿Cuánto cuesta una camiseta azul?"}],
    tools=[{"type": "function", "function": f} for f in functions],
    tool_choice="auto"
)

# Si GPT quiere llamar una función
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    resultado = buscar_producto(**args)
    
    # Segunda llamada con el resultado
    response_final = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "¿Cuánto cuesta una camiseta azul?"},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(resultado)
            }
        ],
        tools=[{"type": "function", "function": f} for f in functions]
    )
    print(response_final.choices[0].message.content)
```

---

## 8. Embeddings

Los embeddings convierten texto en vectores numéricos para búsqueda semántica y RAG.

```python
# Generar embedding de un texto
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="La inteligencia artificial está transformando la industria"
)

vector = response.data[0].embedding
print(f"Dimensiones del vector: {len(vector)}")  # 1536

# Generar embeddings de múltiples textos (más eficiente)
textos = [
    "Python es un lenguaje de programación",
    "Los gatos son animales domésticos",
    "El machine learning usa datos para aprender"
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=textos
)

vectores = [item.embedding for item in response.data]
```

### Búsqueda de similaridad

```python
import numpy as np

def similitud_coseno(v1: list, v2: list) -> float:
    a, b = np.array(v1), np.array(v2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Buscar el texto más similar a una consulta
consulta = "aprendizaje automático"
consulta_vec = client.embeddings.create(
    model="text-embedding-3-small",
    input=consulta
).data[0].embedding

similaridades = [(i, similitud_coseno(consulta_vec, v)) for i, v in enumerate(vectores)]
mejor = max(similaridades, key=lambda x: x[1])
print(f"Texto más similar: '{textos[mejor[0]]}' (similitud: {mejor[1]:.3f})")
```

**Modelos de embeddings disponibles:**

| Modelo | Dimensiones | Coste (por MTok) |
|---|---|---|
| `text-embedding-3-small` | 1536 | $0.02 |
| `text-embedding-3-large` | 3072 | $0.13 |
| `text-embedding-ada-002` | 1536 | $0.10 (legacy) |

---

## 9. Modelos de imagen (DALL·E)

```python
# Generar imagen
response = client.images.generate(
    model="dall-e-3",
    prompt="Un robot amigable leyendo un libro en una biblioteca futurista, estilo acuarela",
    size="1024x1024",
    quality="standard",
    n=1
)

url_imagen = response.data[0].url
print(f"Imagen generada: {url_imagen}")

# Variaciones de una imagen existente (dall-e-2)
with open("imagen.png", "rb") as f:
    response = client.images.create_variation(
        image=f,
        n=3,
        size="1024x1024"
    )
```

---

## 10. Manejo de errores

```python
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
import time

def llamar_con_reintentos(prompt: str, max_intentos: int = 3) -> str:
    for intento in range(max_intentos):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
            
        except RateLimitError:
            if intento < max_intentos - 1:
                espera = 2 ** intento
                print(f"Rate limit. Esperando {espera}s...")
                time.sleep(espera)
            else:
                raise
                
        except APIConnectionError as e:
            print(f"Error de conexión: {e}")
            raise
            
        except APIStatusError as e:
            print(f"Error {e.status_code}: {e.message}")
            raise
```

---

## 11. Buenas prácticas y costes

### Precios aproximados (abril 2026)

| Modelo | Input (por MTok) | Output (por MTok) |
|---|---|---|
| GPT-4o | $2.50 | $10 |
| GPT-4o mini | $0.15 | $0.60 |
| GPT-4 Turbo | $10 | $30 |

> Consulta [openai.com/pricing](https://openai.com/pricing) para precios actualizados.

### Gestión de contexto

```python
# Truncar historial para no exceder el contexto ni disparar costes
def truncar_historial(historial: list, max_mensajes: int = 20) -> list:
    if len(historial) > max_mensajes:
        # Mantener el system message (si existe) y los últimos N mensajes
        system = [m for m in historial if m["role"] == "system"]
        resto = [m for m in historial if m["role"] != "system"]
        return system + resto[-max_mensajes:]
    return historial
```

---

**Anterior:** [01 — API de Anthropic](./01-api-anthropic-claude.md) · **Siguiente:** [03 — Comparativa de proveedores](./03-comparativa-proveedores.md)
