# Bitácora del Proyecto

Registro cronológico de decisiones, aprendizajes, problemas encontrados y soluciones adoptadas durante el desarrollo del proyecto.

---

## 2026-04-20 (v1.1.0)

### Bloque 24, ampliación de Fundamentos y Multimodalidad, relleno de notebooks

- Se crea el **Bloque 24 — PydanticAI y frameworks de agentes**: el espacio reservado desde v1.0.0 se rellena ahora que PydanticAI ha madurado. Se incluyen cuatro artículos: introducción tipada, patrones avanzados (TestModel, ModelRetry, nested agents), Mastra.ai (TypeScript framework) y comparativa de frameworks. Decisión clave: Mastra se cubre en Python simulado + referencia TypeScript porque el repositorio es predominantemente Python, pero los patrones son transferibles. El árbol de decisión de frameworks es el núcleo del artículo de comparativa — responde la pregunta más frecuente en proyectos nuevos.

- **Ampliación Bloque 1 — Fundamentos** con dos artículos que cubrían un hueco crítico: `04-redes-neuronales.md` (neurona → MLP → backpropagation → PyTorch) y `05-transformers-atencion.md` (Q/K/V → multi-cabeza → PE → Chinchilla). Decisión: incluirlos en Fundamentos (no en LLMs) porque son la base matemática que hace que el resto del repositorio tenga sentido. Ambos incluyen implementaciones NumPy desde cero para reforzar la comprensión, con notebooks interactivos que visualizan mapas de atención y PE sinusoidal.

- **Ampliación Bloque 8 — Multimodalidad** con `04-video-ia.md`: cubre Gemini nativo (video hasta 1h), Claude Vision por fotogramas con ffmpeg, transcripción Whisper con chunking para archivos >25MB, generación Runway ML y subtítulos SRT automáticos. Decisión: documentar los dos caminos (Gemini nativo vs Claude+ffmpeg) porque el acceso a Gemini no está garantizado en todos los proyectos.

- **14 notebooks de relleno**: auditoría detectó que bloques 2 (LLMs), 3 (APIs) y 4 (Python) tenían artículos sin notebook. Se crean los 14 faltantes para cerrar la brecha. Patrón consistente: cada notebook arranca con `# pip install ...` y una celda de verificación de dependencias, usa `claude-haiku-4-5-20251001` como modelo por defecto para minimizar costes, e incluye estimaciones de coste en los casos relevantes.

---

## 2026-04-20 (v1.0.0)

### Bloques 22, 23 y 25 — n8n, Claude Code y Proyectos integradores

- Se añade el **Bloque 22 — Workflows con n8n y Claude**: patrón de orquestación visual donde n8n gestiona las integraciones (Gmail, Slack, HubSpot, webhooks) y Claude aporta el razonamiento. Todos los Code nodes usan `$http.request`, `$env.VAR` y `$input.first().json` — la API exacta de n8n. Los notebooks simulan los mismos patrones en Python para que el lector entienda la lógica antes de implementarlos visualmente.

- Se añade el **Bloque 23 — Claude Code y desarrollo asistido**: primer bloque centrado en herramientas para desarrolladores en lugar de en la API. Decisión clave: incluir los tres niveles (CLI básico → MCP servers → hooks) en lugar de solo el uso básico, porque el valor diferencial de Claude Code frente a Copilot está en MCP y hooks. Se incluye comparativa honesta Cursor vs Copilot vs Claude Code sin sesgo hacia Anthropic.

- Se añade el **Bloque 25 — Proyectos integradores**: cuatro proyectos completos que combinan bloques anteriores. Numerados como "25" dejando espacio para un futuro Bloque 24 (PydanticAI/frameworks modernos) cuando maduren. Cada proyecto tiene arquitectura, código funcional y checklist de producción — no son demos sino puntos de partida reales. El proyecto de análisis de datos (04) usa sandbox de ejecución de Pandas con lista de operaciones prohibidas, patrón que evita inyección de código.

- Se alcanza **v1.0.0**: el repositorio cubre ahora el ciclo completo desde fundamentos hasta proyectos de producción reales, con 23+ bloques, 100+ notebooks y múltiples proyectos integradores.

- `README.md` actualizado con todos los bloques, estructura de directorios completa y orden de lectura revisado.

---

## 2026-04-16 (v0.6.0)

### Bloques 10–13, mejoras transversales y notebooks

- Se añade el **Bloque 10 — Casos de uso avanzados** (`tutoriales/casos-de-uso-avanzados/`): revisión de código con rúbrica JSON y AST, búsqueda semántica con embeddings sin dependencia de vector DB, structured output con la librería `instructor`, agente analista de DataFrames con tool_use y generación de gráficas.

- Se añade el **Bloque 11 — IA local** (`tutoriales/ia-local/`): Ollama con API REST compatible OpenAI y soporte visión (LLaVA), Transformers con quantización 4-bit (bitsandbytes), Transformers.js en navegador con Web Workers, y análisis de coste/privacidad para decidir entre local y cloud. Decisión: incluir el tutorial de Transformers.js porque cierra el gap de IA sin backend, caso de uso clave para aplicaciones con datos sensibles.

- Se añade el **Bloque 12 — Seguridad** (`tutoriales/seguridad/`): enfoque exclusivamente defensivo. Se documenta prompt injection directa e indirecta, guardrails de entrada y salida, detección y anonimización de PII con Microsoft Presidio, y sistema completo de auditoría con threading. Decisión: no incluir técnicas ofensivas completas, solo ejemplos mínimos para entender qué hay que defenderse.

- Se añade el **Bloque 13 — Bases de datos vectoriales** (`tutoriales/bases-de-datos-vectoriales/`): pgvector con PostgreSQL (sin infraestructura nueva para quien ya usa Postgres), Pinecone y Weaviate con búsqueda híbrida BM25+vectorial, reranking con Cross-Encoders y Cohere, y técnicas avanzadas de RAG (HyDE, Parent-Child, Self-Query). Decisión: pgvector primero porque es la opción más accesible para proyectos que ya tienen PostgreSQL.

- **Mejoras transversales**:
  - `javascript/03-nextjs-ia.md`: chatbot de producción con Next.js 15 App Router, Vercel AI SDK v4, streaming SSE, rate limiting con Upstash Redis. Se elige Upstash por ser serverless-friendly (compatible con Vercel Edge).
  - `tutoriales/llms/08-dspy.md`: DSPy como alternativa a prompt engineering manual; se incluye compilación con BootstrapFewShot que muestra la diferencia de precisión real.
  - `tutoriales/llms/09-langgraph.md`: LangGraph para flujos con estado, aristas condicionales y human-in-the-loop; se incluye `MemorySaver` para persistencia y nota sobre `SqliteSaver` para producción.

- **9 notebooks nuevos** para los bloques 7, 8 y 9 en `tutoriales/notebooks/produccion/`, `tutoriales/notebooks/multimodalidad/` y `tutoriales/notebooks/agentes-avanzados/`. Todos usan `claude-haiku-4-5-20251001` para minimizar el coste durante la práctica.

- Se actualiza `README.md` con árbol de directorios completo, tabla de los 13 bloques y diagrama de orden de lectura recomendado.

- Se actualiza `mkdocs.yml` con navegación completa de todos los bloques y artículos nuevos.

---

## 2026-04-15 (v0.5.0)

### Bloques 7, 8 y 9 — Producción, Multimodalidad y Agentes avanzados

- Se añade el **Bloque 7 — Producción y evaluación** (`tutoriales/produccion/`):
  - `01-evaluacion-llms.md`: métricas automáticas (BLEU/ROUGE), patrón LLM-as-judge con rúbrica JSON, framework de evaluación con dataset de referencia, evaluación de RAG con ragas.
  - `02-observabilidad.md`: decorador de logging JSONL, tracing con Langfuse, dashboard de costes con pandas, sistema de alertas con ventana deslizante.
  - `03-optimizacion-costos.md`: contador de tokens, Prompt Caching de Anthropic con `cache_control`, Batch API con `MessageBatch`, compresión de historial, calculadora de costes comparativa.
  - `04-despliegue.md`: API REST con FastAPI + Pydantic, streaming SSE, rate limiting con slowapi, retry con backoff exponencial, Dockerfile multi-stage, docker-compose.

- Se añade el **Bloque 8 — Multimodalidad** (`tutoriales/multimodalidad/`):
  - `01-vision-llms.md`: análisis de imágenes locales (base64) y por URL con Claude, extractor de facturas a JSON con Pydantic, comparación de imágenes, transcripción de documentos escaneados.
  - `02-generacion-imagenes.md`: DALL-E 3 (generación y edición/inpainting), prompt engineering para imágenes, Stable Diffusion local con diffusers, pipeline Claude→DALL-E.
  - `03-voz-ia.md`: Whisper local y vía API (con partición de audio largo), TTS con OpenAI, pipeline completo voz→Claude→voz, asistente de reuniones con Pydantic.

- Se añade el **Bloque 9 — Agentes avanzados** (`tutoriales/agentes-avanzados/`):
  - `01-multi-agente.md`: arquitecturas (orquestador-trabajadores, par a par, jerárquica), implementación manual con `tool_use` de Anthropic, CrewAI, comunicación asíncrona con `asyncio.Queue`.
  - `02-model-context-protocol.md`: arquitectura MCP, servidor con herramientas (`@servidor.list_tools`) y recursos (`@servidor.list_resources`), configuración de Claude Desktop.
  - `03-computer-use.md`: bucle de control completo con PyAutoGUI, automatización de formularios con Playwright, scraping visual, lista blanca de seguridad.
  - `04-memoria-largo-plazo.md`: memoria episódica JSON con extracción de hechos por Claude, memoria semántica con ChromaDB + sentence-transformers, compresión automática de historial, clase `MemoryManager` integrada.

- Se actualiza `mkdocs.yml` con las 3 nuevas secciones de navegación.
- Se actualiza `TODO.md` con los nuevos bloques marcados como completados.

---

## 2026-04-14 (v0.4.0)

### CI/CD y GitHub Pages

- Se añade `.github/workflows/validate-notebooks.yml`: valida que todos los notebooks tengan formato JSON correcto (nbformat 4) y estructura mínima (primera celda Markdown con título). Se ejecuta en cada push que toque archivos `.ipynb`.
- Se añade `.github/workflows/gh-pages.yml`: despliega automáticamente la documentación con **MkDocs Material** en GitHub Pages al hacer push a `main`. Requiere activar GitHub Pages en el repositorio con "GitHub Actions" como fuente.
- Se crea `mkdocs.yml` con la navegación completa del proyecto, tema Material en español con modo claro/oscuro, búsqueda en español y copy de código.

### Bloque JavaScript

- Decisión: incluir ejemplos en JavaScript nativo (Node.js ESM) para cubrir el stack web más común.
- Se elige **LangChain.js** para pipelines complejos y **Vercel AI SDK** para streaming y structured output con Zod.
- Se documenta la diferencia principal: LangChain.js es más potente para RAG y agentes; Vercel AI SDK es más limpio para Next.js y para `generateObject` con validación automática.
- El `package.json` usa `"type": "module"` para ESM moderno.

### Notebooks avanzados de LLMs

- Se crean los 3 notebooks pendientes de la hoja de ruta:
  - `04-agentes-ia.ipynb`: muestra la progresión de una herramienta simple a un bucle agéntico completo con trace visible.
  - `05-rag-chromadb.ipynb`: usa `EphemeralClient` de ChromaDB (sin persistencia) para que funcione en cualquier entorno sin configuración extra.
  - `06-finetuning-lora.ipynb`: usa Claude Haiku para generar el propio dataset de entrenamiento, demostrando el patrón "IA para preparar datos de IA".

---

## 2026-04-14

### Inicio del proyecto

- Se crea el repositorio `InteligenciaArtificial` bajo la organización `LegalIntermediaSL`.
- Objetivo: construir un recurso de referencia y aprendizaje sobre IA con enfoque práctico y profesional.
- Licencia elegida: **MIT**, para permitir uso libre del contenido y ejemplos.
- Se define la estructura base del repositorio: `tutoriales/`, `README.md`, `CHANGELOG.md`, `BITACORA.md`, `TODO.md`.

### Decisiones de diseño

- Se opta por separar el contenido en carpetas temáticas dentro de `tutoriales/` para facilitar la navegación.
- Se usará Jupyter Notebooks como formato principal para tutoriales interactivos.
- Cada subcarpeta tendrá su propio `README.md` explicando el contenido y requisitos.

---

<!-- Añade nuevas entradas al principio, con fecha formato YYYY-MM-DD -->
