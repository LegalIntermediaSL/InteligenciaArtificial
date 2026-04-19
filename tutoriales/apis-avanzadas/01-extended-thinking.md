# Extended Thinking: Razonamiento Profundo con Claude

## ¿Qué es Extended Thinking?

Extended Thinking es una funcionalidad de Claude que le permite **razonar en voz alta** antes
de dar una respuesta final. Claude usa un espacio interno de pensamiento (los llamados
"thinking tokens") para explorar el problema en profundidad antes de responder.

Este mecanismo mejora drásticamente la precisión en:

- Problemas matemáticos complejos
- Razonamiento lógico de múltiples pasos
- Análisis de código con bugs sutiles
- Planificación estratégica con múltiples restricciones
- Dilemas éticos o de toma de decisiones

## Cómo funciona internamente

```
Usuario → Claude recibe el prompt
        → Claude "piensa" (bloques de tipo thinking)
        → Claude genera la respuesta final (bloques de tipo text)
        → El cliente recibe ambos (thinking + text)
```

Los bloques `thinking` no se transmiten al modelo en turns siguientes
(se reemplazan por `thinking` encriptado). El usuario puede ver el razonamiento
o mostrarlo solo en modo debug.

## Sintaxis básica

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",       # Requiere claude-sonnet-4-6 o superior
    max_tokens=16000,                 # Debe ser mayor que budget_tokens
    thinking={
        "type": "enabled",
        "budget_tokens": 10000        # Tokens reservados para el razonamiento
    },
    messages=[{
        "role": "user",
        "content": "¿Cuántos ceros tiene 100! (factorial de 100)?"
    }]
)

# Separar bloques thinking de la respuesta final
for bloque in response.content:
    if bloque.type == "thinking":
        print("RAZONAMIENTO INTERNO:")
        print(bloque.thinking[:500])
    elif bloque.type == "text":
        print("\nRESPUESTA FINAL:")
        print(bloque.text)
```

## Parámetro budget_tokens

El `budget_tokens` controla cuántos tokens puede usar Claude para pensar.
No es un mínimo garantizado sino un límite máximo:

| Complejidad tarea | budget_tokens recomendado | Ejemplo |
|-------------------|--------------------------|---------|
| Simple | 1.000 – 2.000 | Suma de fracciones |
| Media | 5.000 – 8.000 | Análisis de algoritmo |
| Alta | 10.000 – 16.000 | Demostración matemática |
| Muy alta | 20.000 – 32.000 | Prueba formal, estrategia multi-paso |

**Regla:** `max_tokens` siempre debe ser > `budget_tokens` (necesitas espacio para la respuesta).

## Casos de uso con ejemplos

### 1. Matemáticas y lógica

```python
problema = """
Una empresa tiene 3 fábricas (A, B, C) y 4 almacenes (1, 2, 3, 4).
Costes de transporte (€/unidad):
  A→1: 2, A→2: 3, A→3: 1, A→4: 4
  B→1: 5, B→2: 4, B→3: 8, B→4: 2
  C→1: 3, C→2: 6, C→3: 2, C→4: 5
Capacidad: A=120, B=80, C=100 unidades
Demanda: 1=90, 2=70, 3=80, 4=60 unidades
¿Cómo distribuir para minimizar el coste total?
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={"type": "enabled", "budget_tokens": 15000},
    messages=[{"role": "user", "content": problema}]
)
```

### 2. Revisión de código con razonamiento

```python
codigo_con_bug = """
def calcular_media_movil(datos, ventana):
    resultado = []
    for i in range(len(datos)):
        inicio = max(0, i - ventana)
        resultado.append(sum(datos[inicio:i]) / ventana)
    return resultado
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=[{
        "role": "user",
        "content": f"Encuentra todos los bugs en este código y corrígelos:\n{codigo_con_bug}"
    }]
)
```

### 3. Decisiones estratégicas

```python
escenario = """
Somos una startup SaaS con 500K€ de caja para 8 meses.
Opciones:
A) Contratar 2 ingenieros senior (240K€/año): acelerar producto
B) Invertir en marketing (150K€): crecer MRR más rápido
C) Lanzar en EEUU ahora (100K€): mercado más grande
D) Buscar ronda Serie A: diluir pero no morir

MRR actual: 30K€. Churn: 5%/mes. CAC: 800€. LTV: 4.000€.
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=12000,
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=[{
        "role": "user",
        "content": f"Analiza las opciones y recomienda la estrategia óptima:\n{escenario}"
    }]
)
```

## Extended Thinking en conversaciones multi-turn

En conversaciones, los bloques `thinking` previos se pasan como `thinking` encriptado:

```python
# Primera vuelta: Claude piensa
resp1 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10000,
    thinking={"type": "enabled", "budget_tokens": 6000},
    messages=[{"role": "user", "content": "Planifica un sistema de caché distribuido"}]
)

# Construir historial correctamente (incluir bloques thinking)
historial = [
    {"role": "user", "content": "Planifica un sistema de caché distribuido"},
    {"role": "assistant", "content": resp1.content}  # Incluye thinking + text
]

# Segunda vuelta: Claude recuerda su razonamiento previo
resp2 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=10000,
    thinking={"type": "enabled", "budget_tokens": 6000},
    messages=historial + [{"role": "user", "content": "¿Cómo gestionar la invalidación de caché?"}]
)
```

## Streaming con Extended Thinking

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=10000,
    thinking={"type": "enabled", "budget_tokens": 6000},
    messages=[{"role": "user", "content": "Demuestra que √2 es irracional"}]
) as stream:
    for event in stream:
        if hasattr(event, "type"):
            if event.type == "content_block_start":
                if hasattr(event.content_block, "type"):
                    bloque_tipo = event.content_block.type
                    print(f"\n--- Bloque: {bloque_tipo} ---")
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "thinking"):
                    print(event.delta.thinking, end="", flush=True)
                elif hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)
```

## Métricas de uso y costes

```python
response = client.messages.create(...)

# Tokens de thinking se muestran en usage
print(f"Tokens entrada: {response.usage.input_tokens}")
print(f"Tokens thinking: {response.usage.thinking_tokens if hasattr(response.usage, 'thinking_tokens') else 'N/A'}")
print(f"Tokens salida: {response.usage.output_tokens}")

# Coste estimado (Sonnet 4.6 a fecha 2025)
coste = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
print(f"Coste estimado: ${coste:.4f}")
```

## Cuándo NO usar Extended Thinking

- Tareas simples de generación de texto (respuestas directas)
- Clasificación binaria o extracción de datos estructurados
- Chatbots de atención al cliente (latencia demasiado alta)
- Pipelines de alto volumen con coste sensible

## Comparativa: con y sin thinking

| Aspecto | Sin thinking | Con thinking |
|---------|-------------|--------------|
| Latencia | 1-3s | 5-30s |
| Precisión (tareas complejas) | 60-70% | 85-95% |
| Coste | Base | +30-200% |
| Transparencia | Ninguna | Razonamiento visible |
| Streaming | Sí | Sí |
| Multi-turn | Sí | Sí (con limitaciones) |

## Recursos

- [Documentación oficial Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Notebook interactivo](../notebooks/apis-avanzadas/01-extended-thinking.ipynb)
