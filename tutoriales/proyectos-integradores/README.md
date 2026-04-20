# Bloque 25 — Proyectos integradores

Proyectos completos de principio a fin que combinan los conceptos de bloques anteriores
en productos reales con código listo para producción.

## Proyectos

| # | Proyecto | Bloques que combina | Dificultad |
|---|----------|---------------------|------------|
| 01 | [SaaS de análisis de contratos](01-saas-contratos.md) | APIs, Agent SDK, seguridad, producción, BD vectoriales | ⭐⭐⭐ Avanzado |
| 02 | [Chatbot de soporte con memoria](02-chatbot-soporte.md) | LLMs, memoria, n8n, Customer Success, producción | ⭐⭐ Intermedio |
| 03 | [Pipeline de contenido con IA](03-pipeline-contenido.md) | Multimodalidad, automatización, prompts, startups | ⭐⭐ Intermedio |
| 04 | [Asistente de análisis de datos](04-asistente-datos.md) | Tool use, agentes, structured output, MLOps | ⭐⭐⭐ Avanzado |

## ¿Cómo usar estos proyectos?

Cada proyecto incluye:
- **Arquitectura completa** con diagrama de componentes
- **Código funcional** con FastAPI, SQLite/PostgreSQL y la API de Anthropic
- **Notebook interactivo** para explorar cada capa del sistema
- **Checklist de producción** para llevar el proyecto a deploy real

## Requisitos previos recomendados

```
Proyecto 01 (Contratos): Bloques 3, 13, 18, 19
Proyecto 02 (Chatbot):   Bloques 2, 3, 19, 22
Proyecto 03 (Contenido): Bloques 8, 17, 21
Proyecto 04 (Datos):     Bloques 5, 9, 10, 16
```

## Stack común a todos los proyectos

```bash
pip install anthropic fastapi uvicorn python-dotenv
```
