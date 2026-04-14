# 01 — API de Anthropic (Claude)

> **Bloque:** APIs de IA · **Nivel:** Práctico · **Tiempo estimado:** 30 min

---

## Índice

1. [Introducción](#1-introducción)
2. [Configuración inicial](#2-configuración-inicial)
3. [Primera llamada a la API](#3-primera-llamada-a-la-api)
4. [Mensajes y conversaciones](#4-mensajes-y-conversaciones)
5. [System prompt](#5-system-prompt)
6. [Parámetros principales](#6-parámetros-principales)
7. [Streaming](#7-streaming)
8. [Uso de herramientas (Tool Use)](#8-uso-de-herramientas-tool-use)
9. [Procesamiento de imágenes](#9-procesamiento-de-imágenes)
10. [Manejo de errores](#10-manejo-de-errores)
11. [Buenas prácticas y costes](#11-buenas-prácticas-y-costes)

---

## 1. Introducción

La API de Anthropic da acceso a la familia de modelos **Claude**: los modelos más capaces de Anthropic, conocidos por su razonamiento avanzado, seguimiento de instrucciones y compromiso con la seguridad.

**Modelos disponibles (abril 2026):**

| Modelo | ID | Contexto | Uso recomendado |
|---|---|---|---|
| Claude Opus 4.6 | `claude-opus-4-6` | 200K tokens | Tareas complejas, razonamiento profundo |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 200K tokens | Equilibrio calidad/velocidad/coste |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K tokens | Tareas simples, alta velocidad, bajo coste |

---

## 2. Configuración inicial

### Instalar el SDK

```bash
pip install anthropic
```

### Obtener una API key

1. Ve a [console.anthropic.com](https://console.anthropic.com)
2. Crea una cuenta o inicia sesión
3. En el panel, ve a **API Keys** → **Create Key**
4. Copia la clave (solo se muestra una vez)

### Configurar la clave de forma segura

Nunca escribas la API key directamente en el código. Usa variables de entorno:

```bash
# En tu terminal (o en .env)
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
# En Python — el SDK la lee automáticamente de ANTHROPIC_API_KEY
import anthropic
client = anthropic.Anthropic()  # Lee la variable de entorno automáticamente
```

O explícitamente:

```python
import os
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

---

## 3. Primera llamada a la API

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explica qué es la inteligencia artificial en 3 frases."}
    ]
)

print(message.content[0].text)
```

**Respuesta esperada:**
```
La inteligencia artificial es una rama de la informática que desarrolla sistemas 
capaces de realizar tareas que normalmente requieren inteligencia humana, como 
razonar, aprender y resolver problemas. Se basa en algoritmos y modelos matemáticos 
que aprenden de grandes cantidades de datos. Actualmente abarca desde asistentes 
virtuales hasta sistemas de diagnóstico médico y vehículos autónomos.
```

### Estructura de la respuesta

```python
print(message.id)              # ID único del mensaje
print(message.model)           # Modelo usado
print(message.stop_reason)     # Por qué paró: "end_turn", "max_tokens", etc.
print(message.usage.input_tokens)   # Tokens de entrada
print(message.usage.output_tokens)  # Tokens de salida
print(message.content[0].text)      # Texto de la respuesta
```

---

## 4. Mensajes y conversaciones

La API usa un formato de **mensajes alternados** entre `user` y `assistant`.

### Conversación simple

```python
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "¿Cuál es la capital de Francia?"},
        {"role": "assistant", "content": "La capital de Francia es París."},
        {"role": "user", "content": "¿Y cuál es su población aproximada?"}
    ]
)
```

### Gestionar el historial de conversación

```python
historial = []

def chat(mensaje_usuario):
    historial.append({"role": "user", "content": mensaje_usuario})
    
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=historial
    )
    
    texto = respuesta.content[0].text
    historial.append({"role": "assistant", "content": texto})
    return texto

# Uso
print(chat("Hola, ¿cómo estás?"))
print(chat("¿De qué hablamos antes?"))  # Recuerda el contexto
```

---

## 5. System prompt

El system prompt define el comportamiento, rol y restricciones del asistente para toda la conversación.

```python
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="""Eres un asistente especializado en derecho laboral español.
    - Responde siempre en español formal
    - Cita siempre el artículo legal relevante cuando sea posible
    - Si no conoces algo con certeza, indícalo explícitamente
    - No ofrezcas asesoramiento legal definitivo; recomienda consultar con un abogado""",
    messages=[
        {"role": "user", "content": "¿Cuántos días de vacaciones corresponden por ley?"}
    ]
)
```

---

## 6. Parámetros principales

```python
message = client.messages.create(
    model="claude-sonnet-4-6",   # Modelo a usar
    max_tokens=2048,              # Máximo de tokens en la respuesta
    temperature=0.7,              # Aleatoriedad (0.0-1.0)
    top_p=0.9,                    # Nucleus sampling (alternativa a temperature)
    top_k=50,                     # Limita a los K tokens más probables
    stop_sequences=["###", "FIN"],# Para generación al encontrar estas cadenas
    system="...",                 # System prompt
    messages=[...]
)
```

| Parámetro | Rango | Recomendación |
|---|---|---|
| `temperature` | 0.0–1.0 | 0 para código/datos; 0.7 para redacción |
| `max_tokens` | 1–200K | Pon el mínimo necesario para controlar costes |
| `top_p` | 0.0–1.0 | No combines con temperature; elige uno |

---

## 7. Streaming

Para respuestas largas, el streaming muestra los tokens a medida que se generan (mejor UX).

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Escribe un cuento corto sobre la IA"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Streaming con información de uso

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Resume la historia de la IA"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    
    # Al finalizar
    mensaje_final = stream.get_final_message()
    print(f"\n\nTokens usados: {mensaje_final.usage.input_tokens} entrada, "
          f"{mensaje_final.usage.output_tokens} salida")
```

---

## 8. Uso de herramientas (Tool Use)

El tool use permite que Claude llame a funciones definidas por ti para obtener información externa.

```python
import json

# Definición de herramientas
tools = [
    {
        "name": "obtener_tiempo",
        "description": "Obtiene el tiempo actual en una ciudad",
        "input_schema": {
            "type": "object",
            "properties": {
                "ciudad": {
                    "type": "string",
                    "description": "Nombre de la ciudad"
                }
            },
            "required": ["ciudad"]
        }
    }
]

# Función real que ejecuta la herramienta
def obtener_tiempo(ciudad: str) -> dict:
    # Aquí iría la llamada a una API meteorológica real
    return {"ciudad": ciudad, "temperatura": "22°C", "estado": "Soleado"}

# Primera llamada: Claude decide usar la herramienta
respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}]
)

# Si Claude quiere usar una herramienta
if respuesta.stop_reason == "tool_use":
    tool_use = next(b for b in respuesta.content if b.type == "tool_use")
    
    # Ejecutar la herramienta localmente
    resultado = obtener_tiempo(**tool_use.input)
    
    # Segunda llamada: devolver el resultado a Claude
    respuesta_final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "¿Qué tiempo hace en Madrid?"},
            {"role": "assistant", "content": respuesta.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(resultado)
                }]
            }
        ]
    )
    print(respuesta_final.content[0].text)
```

---

## 9. Procesamiento de imágenes

Claude es multimodal y puede analizar imágenes.

```python
import base64
from pathlib import Path

# Cargar imagen desde archivo
imagen_bytes = Path("diagrama.png").read_bytes()
imagen_b64 = base64.standard_b64encode(imagen_bytes).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": imagen_b64
                }
            },
            {
                "type": "text",
                "text": "Describe lo que ves en esta imagen y extrae cualquier dato o texto visible."
            }
        ]
    }]
)
print(message.content[0].text)
```

**Formatos soportados:** JPEG, PNG, GIF, WebP

---

## 10. Manejo de errores

```python
import anthropic
from anthropic import APIError, APIConnectionError, RateLimitError, APIStatusError

def llamar_claude_con_reintentos(prompt: str, max_intentos: int = 3) -> str:
    for intento in range(max_intentos):
        try:
            respuesta = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return respuesta.content[0].text
            
        except RateLimitError:
            if intento < max_intentos - 1:
                import time
                tiempo_espera = 2 ** intento  # Backoff exponencial: 1s, 2s, 4s
                print(f"Rate limit. Esperando {tiempo_espera}s...")
                time.sleep(tiempo_espera)
            else:
                raise
                
        except APIConnectionError as e:
            print(f"Error de conexión: {e}")
            raise
            
        except APIStatusError as e:
            print(f"Error de API {e.status_code}: {e.message}")
            raise
```

---

## 11. Buenas prácticas y costes

### Controlar costes

```python
# Estimar tokens antes de enviar (regla: ~4 chars = 1 token)
def estimar_tokens(texto: str) -> int:
    return len(texto) // 4

# Limitar max_tokens al mínimo necesario
# Para clasificación: max_tokens=10
# Para resumen: max_tokens=500
# Para generación larga: max_tokens=2048
```

### Precios aproximados (abril 2026)

| Modelo | Input (por MTok) | Output (por MTok) |
|---|---|---|
| Claude Opus 4.6 | $15 | $75 |
| Claude Sonnet 4.6 | $3 | $15 |
| Claude Haiku 4.5 | $0.25 | $1.25 |

> MTok = millón de tokens. Consulta [anthropic.com/pricing](https://www.anthropic.com/pricing) para precios actualizados.

### Prompt caching

Para prompts con partes que se repiten (system prompts largos, documentos de referencia), usa **prompt caching** para reducir costes hasta un 90%:

```python
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "Tu system prompt muy largo aquí...",
        "cache_control": {"type": "ephemeral"}  # Cachear este bloque
    }],
    messages=[{"role": "user", "content": "Pregunta del usuario"}]
)
```

---

**Siguiente:** [02 — API de OpenAI](./02-api-openai.md)
