# La familia Claude 4.X: Opus 4.7, Sonnet 4.6 y Haiku 4.5

## Introducción

En 2025, Anthropic lanzó la familia Claude 4.X, la generación más capaz de sus modelos. Incluye tres variantes diseñadas para diferentes equilibrios entre potencia, velocidad y coste:

| Modelo | ID API | Contexto | Mejor para |
|--------|--------|----------|-----------|
| Claude Opus 4.7 | `claude-opus-4-7` | 200K tokens | Razonamiento complejo, análisis profundo |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 200K tokens | Balance ideal calidad/coste/velocidad |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200K tokens | Tareas rápidas, alta frecuencia, bajo coste |

## Claude Opus 4.7

Opus 4.7 es el modelo más capaz de Anthropic. Destaca en:

- **Razonamiento matemático y lógico avanzado** — resolución de problemas de varios pasos
- **Extended Thinking** — razonamiento interno extendido antes de responder
- **Análisis de documentos largos** — procesa hasta 200K tokens con alta coherencia
- **Generación de código complejo** — arquitecturas, refactorizaciones, debugging profundo
- **Investigación y síntesis** — cruza información de múltiples fuentes con precisión

**Cuándo usarlo:** tareas que requieren el máximo nivel de razonamiento y donde el coste no es el factor limitante. No lo uses para respuestas cortas o clasificaciones simples.

## Claude Sonnet 4.6

Sonnet 4.6 es el modelo de referencia para la mayoría de aplicaciones en producción:

- Excelente calidad de respuesta en prácticamente todos los dominios
- Velocidad significativamente mayor que Opus
- Coste ~5x menor que Opus
- Soporte completo de tool use, vision y structured output
- Recomendado para **Computer Use** por Anthropic

**Cuándo usarlo:** chatbots, asistentes, procesamiento de documentos, generación de contenido, análisis de datos, aplicaciones de producción con tráfico real.

## Claude Haiku 4.5

Haiku 4.5 prioriza velocidad y eficiencia económica:

- Respuestas en milisegundos para casos de uso de baja latencia
- Coste hasta 20x menor que Opus
- Ideal para pipelines de alto volumen
- Clasificación, resumen corto, extracción de campos simples

**Cuándo usarlo:** primer paso de un pipeline multi-modelo, clasificación de emails, etiquetado de datos, respuestas FAQ, cualquier tarea que puedas definir con un prompt simple y repetitivo.

## Árbol de decisión

```
¿Necesitas razonamiento muy complejo o Extended Thinking?
├── Sí → claude-opus-4-7
└── No → ¿Es una tarea de producción con tráfico real?
    ├── Sí → ¿Importa la latencia o el coste?
    │   ├── Sí → claude-haiku-4-5-20251001 (si tarea simple)
    │   │         claude-sonnet-4-6 (si tarea moderada)
    │   └── No → claude-sonnet-4-6
    └── No (prototipo/exploración) → claude-sonnet-4-6
```

## Precios (aproximados, por millón de tokens)

| Modelo | Input | Output |
|--------|-------|--------|
| Opus 4.7 | $15 | $75 |
| Sonnet 4.6 | $3 | $15 |
| Haiku 4.5 | $0.80 | $4 |

## Código: usando los tres modelos

```python
import anthropic

client = anthropic.Anthropic()

MODELS = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

def chat(model_key: str, prompt: str) -> str:
    response = client.messages.create(
        model=MODELS[model_key],
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# Tarea simple → Haiku
respuesta_haiku = chat("haiku", "Clasifica este email como urgente o no urgente: 'Reunión mañana a las 9'")
print(f"Haiku: {respuesta_haiku}")

# Tarea moderada → Sonnet
respuesta_sonnet = chat("sonnet", "Explica las ventajas y desventajas del fine-tuning vs RAG en 3 puntos cada uno.")
print(f"Sonnet: {respuesta_sonnet}")

# Tarea compleja → Opus
respuesta_opus = chat("opus", "Diseña la arquitectura de un sistema multi-agente para automatizar la due diligence legal de contratos, incluyendo gestión de errores, logging y escalado horizontal.")
print(f"Opus: {respuesta_opus}")
```

## Router de modelos por complejidad

Un patrón habitual en producción es enrutar automáticamente las peticiones al modelo más económico que pueda manejar la tarea:

```python
import anthropic

client = anthropic.Anthropic()

def router(prompt: str, max_tokens: int = 1024) -> tuple[str, str]:
    """Selecciona el modelo óptimo según la longitud y complejidad del prompt."""
    words = len(prompt.split())

    # Heurísticas simples (en producción, usarías un clasificador)
    if words < 50 and not any(kw in prompt.lower() for kw in ["analiza", "diseña", "compara", "razona"]):
        model = "claude-haiku-4-5-20251001"
    elif words < 300:
        model = "claude-sonnet-4-6"
    else:
        model = "claude-opus-4-7"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return model, response.content[0].text

model_usado, respuesta = router("¿Cuánto es 2+2?")
print(f"Modelo: {model_usado}")
print(f"Respuesta: {respuesta}")
```

## Resumen

- **Opus 4.7** para razonamiento profundo y Extended Thinking
- **Sonnet 4.6** para producción generalista (la opción por defecto)
- **Haiku 4.5** para volumen alto y tareas simples
- Un router de modelos puede reducir el coste total un 60-80% sin pérdida de calidad percibida
