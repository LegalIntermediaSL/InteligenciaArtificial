# Producción con LLMs

Tutoriales prácticos para llevar aplicaciones de IA al entorno de producción: evaluación de modelos, observabilidad, optimización de costos y despliegue. Cada tutorial incluye código completo, listo para usar en proyectos reales.

## Contenido

| # | Tutorial | Descripción |
|---|---|---|
| 01 | [Evaluación de LLMs](./01-evaluacion-llms.md) | Medir la calidad de las respuestas con métricas automáticas y LLM-as-judge |
| 02 | [Observabilidad y tracing](./02-observabilidad.md) | Instrumentar aplicaciones con Langfuse para rastrear tokens, latencia y coste |
| 03 | [Optimización de costos](./03-optimizacion-costos.md) | Prompt caching, Batch API y estrategias para reducir el gasto en APIs |
| 04 | [Despliegue con FastAPI y Docker](./04-despliegue.md) | Publicar una API de IA con streaming, rate limiting y contenedores |

## Requisitos del bloque

```bash
pip install langfuse anthropic openai fastapi uvicorn docker
```

Cada tutorial especifica sus propias dependencias adicionales al inicio.

## Orden recomendado

01 → 02 → 03 → 04. Los conceptos de evaluación y observabilidad son útiles antes de optimizar y desplegar.

---

> Bloque anterior: [Casos de uso](../casos-de-uso/)
