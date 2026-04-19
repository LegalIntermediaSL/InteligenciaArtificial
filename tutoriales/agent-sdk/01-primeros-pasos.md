# Primeros Pasos con el Agent SDK de Anthropic

## ¿Qué es un agente LLM?

Un agente es un sistema donde el LLM **decide autónomamente qué acciones tomar**
en función del objetivo dado, usando herramientas externas y evaluando el resultado
hasta completar la tarea.

```
Usuario → Objetivo
        → Claude analiza y elige herramienta
        → Ejecutar herramienta → Resultado
        → Claude evalúa resultado → ¿Terminado?
          → No: elegir siguiente herramienta
          → Sí: devolver respuesta final
```

## Estructura mínima de un agente

```python
import anthropic
import json

client = anthropic.Anthropic()

# 1. Definir herramientas disponibles
HERRAMIENTAS = [
    {
        "name": "buscar_web",
        "description": "Busca información actualizada en internet",
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "Términos de búsqueda"}
            },
            "required": ["consulta"]
        }
    },
    {
        "name": "calcular",
        "description": "Realiza cálculos matemáticos",
        "input_schema": {
            "type": "object",
            "properties": {
                "expresion": {"type": "string", "description": "Expresión matemática, ej: 2+2*3"}
            },
            "required": ["expresion"]
        }
    }
]

# 2. Implementar las herramientas
def ejecutar_herramienta(nombre: str, parametros: dict) -> str:
    if nombre == "buscar_web":
        # En producción: llamar a Tavily, SerpAPI, etc.
        return f"[Resultado de búsqueda para '{parametros['consulta']}': Información relevante encontrada...]"
    elif nombre == "calcular":
        try:
            return str(eval(parametros["expresion"], {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
    return "Herramienta no reconocida"

# 3. Bucle agéntico
def agente(objetivo: str, max_iteraciones: int = 10) -> str:
    mensajes = [{"role": "user", "content": objetivo}]
    print(f"Objetivo: {objetivo}\n")

    for i in range(max_iteraciones):
        respuesta = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=HERRAMIENTAS,
            messages=mensajes
        )

        # Añadir respuesta del asistente al historial
        mensajes.append({"role": "assistant", "content": respuesta.content})

        if respuesta.stop_reason == "end_turn":
            # El agente ha terminado
            for bloque in respuesta.content:
                if hasattr(bloque, "text"):
                    return bloque.text
            return "Completado"

        if respuesta.stop_reason == "tool_use":
            # Ejecutar herramientas solicitadas
            resultados_herramientas = []
            for bloque in respuesta.content:
                if bloque.type == "tool_use":
                    print(f"  [{i+1}] Usando herramienta: {bloque.name}({bloque.input})")
                    resultado = ejecutar_herramienta(bloque.name, bloque.input)
                    print(f"       → {resultado[:80]}")
                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": resultado
                    })

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return "Máximo de iteraciones alcanzado"

# Uso
resultado = agente("¿Cuánto es el 15% de 2.847€? Y busca qué es el IVA en España")
print(f"\nRespuesta final: {resultado}")
```

## System prompt para agentes

Un buen system prompt de agente incluye:

```python
SYSTEM_AGENTE = """Eres un agente de análisis de datos con acceso a herramientas.

COMPORTAMIENTO:
- Analiza cuidadosamente la tarea antes de actuar
- Usa herramientas solo cuando sean necesarias
- Verifica los resultados antes de concluir
- Si una herramienta falla, intenta un enfoque alternativo
- Sé conciso en tus respuestas intermedias, detallado en la respuesta final

REGLAS:
- No inventes datos; usa las herramientas para obtenerlos
- Si no puedes completar la tarea, explica por qué
- Confirma resultados calculados antes de usarlos
"""
```

## Tipos de herramientas más comunes

### Búsqueda y recuperación de información
```python
herramienta_busqueda = {
    "name": "buscar_documentos",
    "description": "Busca en la base de conocimiento interna de la empresa",
    "input_schema": {
        "type": "object",
        "properties": {
            "consulta": {"type": "string"},
            "max_resultados": {"type": "integer", "default": 5}
        },
        "required": ["consulta"]
    }
}
```

### Lectura/escritura de datos
```python
herramienta_bd = {
    "name": "consultar_base_datos",
    "description": "Ejecuta una consulta SQL en la base de datos de la empresa",
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "Consulta SQL (solo SELECT)"},
            "limite": {"type": "integer", "description": "Máximo de filas", "default": 100}
        },
        "required": ["sql"]
    }
}
```

### APIs externas
```python
herramienta_email = {
    "name": "enviar_email",
    "description": "Envía un email al destinatario especificado",
    "input_schema": {
        "type": "object",
        "properties": {
            "destinatario": {"type": "string"},
            "asunto": {"type": "string"},
            "cuerpo": {"type": "string"},
            "cc": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["destinatario", "asunto", "cuerpo"]
    }
}
```

## Manejo de errores en el bucle agéntico

```python
def ejecutar_herramienta_segura(nombre: str, parametros: dict) -> dict:
    """Ejecuta herramienta con manejo de errores y timeout."""
    import signal

    def handler(signum, frame):
        raise TimeoutError("Herramienta tardó demasiado")

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(30)  # 30 segundos de timeout

        resultado = ejecutar_herramienta(nombre, parametros)
        signal.alarm(0)  # Cancelar timeout

        return {"success": True, "resultado": resultado}

    except TimeoutError:
        return {"success": False, "error": "Timeout de 30 segundos excedido"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
    finally:
        signal.alarm(0)

# En el bucle: devolver error al agente para que reintente
resultado = ejecutar_herramienta_segura(bloque.name, bloque.input)
if not resultado["success"]:
    contenido_resultado = f"ERROR: {resultado['error']}. Por favor, intenta con un enfoque diferente."
else:
    contenido_resultado = resultado["resultado"]
```

## Parallel tool use

Claude puede solicitar múltiples herramientas simultáneamente:

```python
# Claude devuelve múltiples bloques tool_use en un mismo response
# Hay que ejecutarlos todos y devolver todos los resultados

resultados = []
bloques_tool_use = [b for b in respuesta.content if b.type == "tool_use"]

# Ejecutar en paralelo
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    futuros = {
        executor.submit(ejecutar_herramienta, b.name, b.input): b
        for b in bloques_tool_use
    }
    for futuro, bloque in futuros.items():
        resultado = futuro.result()
        resultados.append({
            "type": "tool_result",
            "tool_use_id": bloque.id,
            "content": str(resultado)
        })

mensajes.append({"role": "user", "content": resultados})
```

## Recursos

- [Documentación Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Notebook interactivo](../notebooks/agent-sdk/01-primeros-pasos.ipynb)
