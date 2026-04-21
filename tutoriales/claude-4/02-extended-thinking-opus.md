# Extended Thinking con Claude Opus 4.7: razonamiento profundo

## ¿Qué es Extended Thinking?

Extended Thinking permite a Claude Opus 4.7 "pensar en voz alta" antes de responder. El modelo genera un bloque de razonamiento interno (thinking tokens) que no forma parte de la respuesta final pero sí influye en su calidad. Es el equivalente a pedirle a un experto que trabaje el problema en papel antes de darte la respuesta.

### ¿Cuándo usarlo?

- Problemas matemáticos de varios pasos
- Razonamiento lógico y deductivo
- Análisis de argumentos complejos
- Generación de código con múltiples requisitos
- Planificación de proyectos con restricciones
- Decisiones con trade-offs múltiples

**No uses Extended Thinking** para respuestas cortas, clasificaciones simples o conversaciones informales — añade latencia y coste sin beneficio.

## Parámetros clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `thinking.type` | `"enabled"` | Activa el modo de pensamiento extendido |
| `thinking.budget_tokens` | 1024–32000 | Tokens máximos para el bloque de thinking |
| `betas` | `["interleaved-thinking-2025-05-14"]` | Header beta requerido |
| `max_tokens` | ≥ budget_tokens | Debe ser mayor que budget_tokens |

### Recomendaciones de budget_tokens

- **1024–4096** — problemas moderados, balance velocidad/calidad
- **8000–16000** — problemas complejos, análisis profundo
- **32000** — máximo, para casos extremadamente difíciles

## Ejemplo básico

```python
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-opus-4-7",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,
    },
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{
        "role": "user",
        "content": "Un tren sale de Madrid a las 8:00 a 200 km/h hacia Barcelona (620 km). Otro tren sale de Barcelona a las 9:30 a 160 km/h hacia Madrid. ¿A qué hora se cruzan y a qué distancia de Madrid?"
    }],
)

for block in response.content:
    if block.type == "thinking":
        print("=== RAZONAMIENTO INTERNO ===")
        print(block.thinking)
        print("============================\n")
    elif block.type == "text":
        print("=== RESPUESTA FINAL ===")
        print(block.text)
```

## Streaming con Extended Thinking

Para aplicaciones en tiempo real, puedes hacer streaming tanto del bloque de thinking como de la respuesta:

```python
import anthropic

client = anthropic.Anthropic()

with client.beta.messages.stream(
    model="claude-opus-4-7",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 8000,
    },
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{
        "role": "user",
        "content": "Diseña un algoritmo eficiente para detectar ciclos en un grafo dirigido. Explica la complejidad temporal y espacial."
    }],
) as stream:
    current_block_type = None
    for event in stream:
        if hasattr(event, "type"):
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "thinking":
                    current_block_type = "thinking"
                    print("\n[Pensando...]", end="", flush=True)
                elif block.type == "text":
                    current_block_type = "text"
                    print("\n[Respuesta]\n", end="", flush=True)
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "thinking"):
                    print(".", end="", flush=True)  # Indicador visual
                elif hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)
    print()
```

## Caso de uso: análisis de contratos legales

```python
import anthropic

client = anthropic.Anthropic()

contrato = """
CLÁUSULA 7.3 — LIMITACIÓN DE RESPONSABILIDAD
El proveedor no será responsable por daños indirectos, incidentales o consecuentes,
incluyendo pérdida de beneficios, incluso si ha sido advertido de la posibilidad de
dichos daños. La responsabilidad total del proveedor no excederá el importe pagado
en los 12 meses anteriores al evento que dio lugar a la reclamación.

CLÁUSULA 12.1 — FUERZA MAYOR
Ninguna parte será responsable del incumplimiento por causas fuera de su control
razonable. La parte afectada deberá notificar a la otra dentro de las 48 horas.
"""

response = client.beta.messages.create(
    model="claude-opus-4-7",
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 12000,
    },
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{
        "role": "user",
        "content": f"""Analiza este contrato e identifica:
1. Riesgos principales para el cliente
2. Cláusulas ambiguas o problemáticas
3. Recomendaciones de negociación

Contrato:
{contrato}"""
    }],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
```

## Costes y optimización

Extended Thinking genera tokens adicionales (los thinking tokens) que **sí se cobran** como tokens de output:

```python
# Estimar coste de una llamada con thinking
def estimar_coste_opus(input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    # Precios aproximados por millón de tokens
    precio_input = 15.0 / 1_000_000
    precio_output = 75.0 / 1_000_000  # thinking + output se cobran igual

    total_output = output_tokens + thinking_tokens
    return input_tokens * precio_input + total_output * precio_output

# Ejemplo
coste = estimar_coste_opus(
    input_tokens=500,
    output_tokens=800,
    thinking_tokens=8000
)
print(f"Coste estimado: ${coste:.4f}")
# ~$0.68 por llamada con 8K thinking tokens
```

### Estrategia de coste

1. Usa `budget_tokens` pequeño (1024-2048) para validar que mejora la calidad antes de subir
2. Aplica Extended Thinking solo en el paso final de un pipeline, no en todos
3. Combina: Haiku para filtrar → Sonnet para procesar → Opus+thinking para casos difíciles

## Resumen

- Extended Thinking activa razonamiento interno antes de responder
- El parámetro `budget_tokens` controla cuánto puede "pensar" el modelo
- Los thinking tokens se cobran como output — gestiona el budget con cuidado
- Mejoras más notables en: matemáticas, lógica, análisis complejos, código difícil
- Para streaming, los thinking tokens se emiten como bloques separados
