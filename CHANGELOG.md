# Changelog

Todos los cambios relevantes del proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.0.0/),
y el proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0] — 2026-04-20

### Añadido
- **Bloque 22 — Workflows con n8n y Claude**: 4 artículos + 4 notebooks
  - `01-introduccion-n8n.md` — conceptos, primer workflow Email→Claude→Slack
  - `02-workflows-documentos.md` — facturas, contratos, resumen largo, leads
  - `03-workflows-negocio.md` — triaje emails, CRM, Slack bot, informe semanal
  - `04-workflows-avanzados.md` — sub-workflows, webhooks, reintentos, Docker Compose

- **Bloque 23 — Claude Code y desarrollo asistido**: 4 artículos + 4 notebooks
  - `01-claude-code-cli.md` — instalación, slash commands, CLAUDE.md, configuración
  - `02-mcp-servers.md` — servidores MCP en Python, recursos, prompts, HTTP mode
  - `03-hooks-automatizacion.md` — hooks de ciclo de vida, linting, auditoría, CI/CD
  - `04-flujos-desarrollo-ia.md` — flujos completos, debug, code review, comparativa

- **Bloque 25 — Proyectos integradores**: 4 proyectos completos + 4 notebooks
  - `01-saas-contratos.md` — SaaS legal: Files API + ChromaDB + FastAPI + Pydantic
  - `02-chatbot-soporte.md` — chatbot con KB vectorial y enrutamiento Haiku→Sonnet
  - `03-pipeline-contenido.md` — multi-formato con LLM-as-judge y A/B de prompts
  - `04-asistente-datos.md` — tool use nativo, sandbox Pandas, bucle agéntico

- `README.md` — actualizado con bloques 18-23 y 25, estructura y orden de lectura

---

## [0.7.0] — 2026-04-16

### Añadido
- **Bloque 9 — Agentes Avanzados (ampliación)**: 4 nuevos tutoriales que extienden el bloque de agentes:
  - `07-langgraph-agentes.md` — StateGraph con TypedDict reducers, ToolNode, checkpointing con MemorySaver y SqliteSaver, human-in-the-loop con interrupt_before/after, subgrafos y paralelismo, streaming de tokens y eventos, patrones de producción (reintentos, timeout)
  - `08-evaluacion-agentes.md` — Métricas de trayectoria (task success rate, tool selection accuracy, trajectory efficiency), LLM-as-judge con rúbrica JSON ponderada, trazas instrumentadas, tracing con Langfuse, golden dataset, tests unitarios con mocks y red teaming automatizado
  - `09-agentes-codigo.md` — Ejecución con exec() y captura de stdout/stderr, sandboxing seguro con E2B, agente analista de datos (pandas/matplotlib), agente de debugging con pytest, generador de tests, validación AST y patrones de producción
  - `10-agentes-especializados.md` — Research Agent (DuckDuckGo, lectura web, PDF), Financial Agent (yfinance, métricas de riesgo; solo análisis, nunca ejecución de trades), Code Review Agent (ruff, mypy, rúbrica estructurada), guía de selección y patrones comunes de especialización
- **Aviso legal** añadido al `README.md` principal

### Modificado
- `tutoriales/agentes-avanzados/README.md` — actualizado con los 4 nuevos tutoriales, nuevas dependencias y tiempo estimado ampliado a 8–10 horas
- `TODO.md` — bloque 9 actualizado con los 4 nuevos tutoriales

---

## [0.6.0] — 2026-04-16

### Añadido
- **Bloque 10 — Casos de uso avanzados** (`tutoriales/casos-de-uso-avanzados/`):
  - `01-generacion-revision-codigo.md` — revisor de código con rúbrica JSON, generador de tests con pytest, documentador automático con AST
  - `02-busqueda-semantica.md` — embeddings, índice local sin ChromaDB, chatbot con RAG sobre ficheros propios, buscador con Streamlit
  - `03-structured-output-instructor.md` — extracción robusta con Instructor y Pydantic, validadores personalizados, modelos anidados
  - `04-analisis-datos-ia.md` — agente analista de DataFrames con tool_use, generación de gráficas con `exec()`, detección de anomalías
- **Bloque 11 — IA local** (`tutoriales/ia-local/`):
  - `01-ollama.md` — instalación, CLI, API REST compatible con OpenAI, SDK nativo, modelos recomendados, visión y RAG local
  - `02-transformers-local.md` — pipeline de inferencia, quantización 4-bit con bitsandbytes, embeddings con sentence-transformers
  - `03-transformers-js-navegador.md` — Transformers.js en Node.js y navegador, Web Workers, buscador semántico offline
  - `04-comparativa-local-cloud.md` — análisis de costes, privacidad y compliance, arquitectura híbrida con router
- **Bloque 12 — Seguridad en IA** (`tutoriales/seguridad/`):
  - `01-prompt-injection.md` — ataques directos e indirectos, detector con Claude, clase GuardrailsManager
  - `02-jailbreaking-guardrails.md` — guardrails de entrada y salida, moderación con OpenAI y Claude, middleware FastAPI
  - `03-datos-sensibles-pii.md` — Presidio de Microsoft, anonimización antes de API, logger seguro sin PII
  - `04-auditoria-seguridad.md` — AuditLogger con redacción PII, detección de anomalías con pandas, dashboard, Monitor con threading
- **Bloque 13 — Bases de datos vectoriales** (`tutoriales/bases-de-datos-vectoriales/`):
  - `01-pgvector.md` — extensión PostgreSQL, inserción con psycopg2, búsqueda coseno, índice HNSW, pipeline RAG completo
  - `02-pinecone-weaviate.md` — Pinecone (upsert, filtros por metadata), Weaviate (esquema, búsqueda híbrida BM25+vectorial), comparativa
  - `03-reranking.md` — Cross-Encoders con sentence-transformers, Cohere Rerank multilingüe, métricas NDCG/MRR, pipeline con reranking
  - `04-rag-avanzado.md` — HyDE, Parent-Child chunks, Self-Query Retriever, RAG multi-documento, evaluación con ragas
- **Mejoras transversales**:
  - `javascript/03-nextjs-ia.md` — chatbot de producción con Next.js 15, App Router, streaming SSE, Tailwind, rate limiting con Upstash
  - `tutoriales/llms/08-dspy.md` — Signatures, módulos (Predict, ChainOfThought, ReAct), compilación con BootstrapFewShot, RAG con DSPy
  - `tutoriales/llms/09-langgraph.md` — StateGraph, aristas condicionales, persistencia con MemorySaver, human-in-the-loop
- **Notebooks nuevos** (complementan los bloques 7, 8 y 9):
  - `notebooks/produccion/` — 3 notebooks: evaluación LLMs, observabilidad, optimización de costos
  - `notebooks/multimodalidad/` — 3 notebooks: visión, generación de imágenes, voz
  - `notebooks/agentes-avanzados/` — 3 notebooks: multi-agente, MCP, memoria a largo plazo
- **README.md** actualizado con todos los bloques, árbol de directorios completo y orden de lectura recomendado

---

## [0.5.0] — 2026-04-15

### Añadido
- **Bloque 7 — Producción y evaluación** (`tutoriales/produccion/`): evaluación de LLMs, observabilidad con Langfuse, optimización de costos (Prompt Caching, Batch API), despliegue con FastAPI y Docker
- **Bloque 8 — Multimodalidad** (`tutoriales/multimodalidad/`): visión con LLMs, generación de imágenes con DALL-E 3 y Stable Diffusion, voz (Whisper + TTS + pipeline completo)
- **Bloque 9 — Agentes avanzados** (`tutoriales/agentes-avanzados/`): multi-agente con CrewAI, Model Context Protocol, Computer Use, memoria a largo plazo con ChromaDB

---

## [0.4.0] — 2026-04-14

### Añadido
- **CI/CD**: workflow de GitHub Actions para validar formato y estructura de notebooks (`.github/workflows/validate-notebooks.yml`)
- **GitHub Pages**: workflow de despliegue automático con MkDocs Material (`.github/workflows/gh-pages.yml` + `mkdocs.yml`)
- **Bloque JavaScript**: carpeta `javascript/` con `README.md`, `package.json` y dos tutoriales completos:
  - `01-langchain-js.js` / `01-langchain-js.md` — LangChain.js con Claude: llamadas, cadenas LCEL, clasificación, streaming
  - `02-vercel-ai-sdk.js` / `02-vercel-ai-sdk.md` — Vercel AI SDK: `generateText`, `streamText`, `generateObject` con Zod, tool use con `maxSteps`
- **Notebooks avanzados LLMs**: tres nuevos cuadernos Jupyter interactivos:
  - `notebooks/llms/04-agentes-ia.ipynb` — Bucle agéntico completo con 3 herramientas
  - `notebooks/llms/05-rag-chromadb.ipynb` — Pipeline RAG con ChromaDB y SentenceTransformers
  - `notebooks/llms/06-finetuning-lora.ipynb` — Dataset sintético con Claude, QLoRA y evaluación baseline

---

## [0.3.0] — 2026-04-14

### Añadido
- Bloque 2 LLMs: 6 tutoriales Markdown completos (LLMs, prompt engineering, fine-tuning vs RAG, agentes, RAG con ChromaDB, fine-tuning con LoRA)
- Bloque 3 APIs: 3 tutoriales (Anthropic Claude, OpenAI, comparativa de proveedores)
- Bloque 4 Python para IA: 3 tutoriales (intro, librerías esenciales, Jupyter Notebooks)
- Bloque 5 Casos de uso: 4 tutoriales (chatbot, clasificación, resumen, extracción de PDFs)
- Bloque 6 Notebooks: 7 cuadernos Jupyter interactivos (fundamentos, prompt engineering, API Anthropic, chatbot, clasificación, resumen, extracción PDFs)
- `requirements.txt` por cada carpeta de bloque

---

## [0.2.0] — 2026-04-14

### Añadido
- Estructura inicial del repositorio
- `README.md` con objetivo y descripción del proyecto
- `CHANGELOG.md` para el historial de cambios
- `BITACORA.md` como diario de decisiones y aprendizajes
- `TODO.md` con la hoja de ruta inicial
- Carpeta `tutoriales/` con estructura por bloques temáticos
- Bloque 1 Fundamentos: 3 tutoriales Markdown (qué es la IA, tipos, historia)

---

## [0.1.0] — 2026-04-14

### Añadido
- Commit inicial con `LICENSE` y `README.md` básico
