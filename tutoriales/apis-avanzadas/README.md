# Bloque 18 — Funcionalidades Avanzadas de la API Anthropic

Explora las capacidades más potentes de la API de Anthropic: razonamiento extendido,
procesamiento por lotes, gestión de archivos, citas automáticas y caché de prompts.

## Tutoriales

| # | Título | Descripción |
|---|--------|-------------|
| 01 | [Extended Thinking](01-extended-thinking.md) | Razonamiento extendido con presupuesto de tokens |
| 02 | [Batch API](02-batch-api.md) | Procesamiento masivo asíncrono con ahorro de costes |
| 03 | [Files API](03-files-api.md) | Subida y reutilización de documentos entre llamadas |
| 04 | [Citations API](04-citations-api.md) | Citas automáticas y grounding factual |
| 05 | [Prompt Caching](05-prompt-caching.md) | Caché de contexto estático para reducir costes hasta 90% |

## Requisitos

```bash
pip install anthropic>=0.40.0
```

## Cuándo usar cada funcionalidad

| Funcionalidad | Caso de uso ideal | Ahorro/Beneficio |
|---------------|-------------------|------------------|
| Extended Thinking | Matemáticas, lógica, planificación compleja | Mejor precisión en tareas difíciles |
| Batch API | Clasificación masiva, embeddings, evaluaciones | Hasta 50% descuento en costes |
| Files API | PDFs grandes, datasets, reutilización entre usuarios | Reducción de tokens de entrada |
| Citations | RAG con atribución, fact-checking, investigación | Respuestas verificables |
| Prompt Caching | System prompts largos, RAG con contexto fijo | Hasta 90% reducción en costes |
