# Bloque 19 — Claude Agent SDK: Construcción de Agentes

Aprende a construir agentes robustos con el SDK de Python de Anthropic:
desde el bucle agéntico básico hasta orquestadores multi-agente en producción.

## Tutoriales

| # | Título | Descripción |
|---|--------|-------------|
| 01 | [Primeros pasos con el Agent SDK](01-primeros-pasos.md) | Estructura básica, herramientas, bucle agéntico |
| 02 | [Agentes compuestos](02-agentes-compuestos.md) | Orquestador + subagentes especializados |
| 03 | [Agentes con memoria](03-agentes-con-memoria.md) | Memoria corto/largo plazo, gestión de contexto |
| 04 | [Despliegue de agentes](04-despliegue-agentes.md) | FastAPI, streaming, monitorización, escalado |

## Requisitos

```bash
pip install anthropic>=0.40.0 fastapi uvicorn chromadb
```

## ¿Qué es el Agent SDK?

El Agent SDK de Anthropic no es una librería separada: es un conjunto de
**patrones y utilidades** dentro del SDK oficial de Python que facilitan
la construcción de sistemas agénticos robustos:

- `client.messages.create()` con `tools` → bucle agéntico manual
- `anthropic.lib.streaming` → streaming de eventos de agentes
- Integración nativa con MCP (Model Context Protocol)
- Soporte para computer use, extended thinking y batch processing
