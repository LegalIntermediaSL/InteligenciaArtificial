# TODO — Hoja de ruta

Lista de tareas pendientes y contenido por desarrollar en el proyecto.

---

## Estructura y organización ✅

- [x] Crear estructura base del repositorio
- [x] Redactar `README.md` con objetivo del proyecto
- [x] Crear `CHANGELOG.md`
- [x] Crear `BITACORA.md`
- [x] Crear carpeta `tutoriales/`
- [x] Definir subcarpetas por temática dentro de `tutoriales/`
- [x] Añadir `requirements.txt` por carpeta

---

## Bloque 1 — Fundamentos ✅

- [x] ¿Qué es la Inteligencia Artificial? Conceptos clave
- [x] Tipos de IA: narrow AI, AGI, machine learning, deep learning
- [x] Historia y evolución de la IA
- [x] Notebook: 01-que-es-la-ia.ipynb

---

## Bloque 2 — LLMs y modelos de lenguaje ✅

- [x] ¿Qué es un LLM? Cómo funcionan los modelos de lenguaje
- [x] Prompt engineering: técnicas y buenas prácticas
- [x] Fine-tuning vs RAG: cuándo usar cada uno
- [x] Agentes de IA: ReAct, tool use, bucle agéntico
- [x] RAG completo con ChromaDB
- [x] Fine-tuning con LoRA y QLoRA
- [x] Notebook: 02-prompt-engineering.ipynb

---

## Bloque 3 — APIs de IA ✅

- [x] Primeros pasos con la API de Anthropic (Claude)
- [x] Primeros pasos con la API de OpenAI
- [x] Comparativa de proveedores de IA
- [x] Notebook: 01-api-anthropic-claude.ipynb

---

## Bloque 4 — Python para IA ✅

- [x] Introducción a Python para proyectos de IA
- [x] Librerías esenciales: NumPy, Pandas, scikit-learn, HuggingFace
- [x] Jupyter Notebooks: configuración y uso

---

## Bloque 5 — Casos de uso prácticos ✅

- [x] Chatbot con Claude API
- [x] Clasificación de texto (zero-shot, few-shot, ML clásico)
- [x] Resumen automático de documentos (map-reduce, refine)
- [x] Extracción de información de PDFs con IA
- [x] Notebook: 01-chatbot-claude-api.ipynb
- [x] Notebook: 02-clasificacion-texto.ipynb
- [x] Notebook: 03-resumen-documentos.ipynb
- [x] Notebook: 04-extraccion-pdfs.ipynb

---

## Bloque 6 — Cuadernos Jupyter Notebook ✅

- [x] notebooks/fundamentos/01-que-es-la-ia.ipynb
- [x] notebooks/llms/02-prompt-engineering.ipynb
- [x] notebooks/apis/01-api-anthropic-claude.ipynb
- [x] notebooks/casos-de-uso/01-chatbot-claude-api.ipynb
- [x] notebooks/casos-de-uso/02-clasificacion-texto.ipynb
- [x] notebooks/casos-de-uso/03-resumen-documentos.ipynb
- [x] notebooks/casos-de-uso/04-extraccion-pdfs.ipynb

---

## Mejoras futuras ✅

- [x] Añadir CI/CD para validar notebooks (`.github/workflows/validate-notebooks.yml`)
- [x] Publicar como GitHub Pages (`.github/workflows/gh-pages.yml` + `mkdocs.yml`)
- [x] Añadir ejemplos en JavaScript (`javascript/` — LangChain.js, Vercel AI SDK)
- [x] Notebook para agentes de IA (`notebooks/llms/04-agentes-ia.ipynb`)
- [x] Notebook para RAG con ChromaDB (`notebooks/llms/05-rag-chromadb.ipynb`)
- [x] Notebook para fine-tuning con LoRA (`notebooks/llms/06-finetuning-lora.ipynb`)

---

## Bloque 7 — Producción y evaluación ✅

- [x] `tutoriales/produccion/README.md`
- [x] `01-evaluacion-llms.md` — LLM-as-judge, métricas automáticas, ragas para RAG
- [x] `02-observabilidad.md` — Logging, tracing con Langfuse, alertas
- [x] `03-optimizacion-costos.md` — Prompt Caching, Batch API, compresión de contexto
- [x] `04-despliegue.md` — FastAPI, streaming, rate limiting, Docker

---

## Bloque 8 — Multimodalidad ✅

- [x] `tutoriales/multimodalidad/README.md`
- [x] `01-vision-llms.md` — Análisis de imágenes, extracción de facturas, comparación
- [x] `02-generacion-imagenes.md` — DALL-E 3, Stable Diffusion, pipeline LLM+imágenes
- [x] `03-voz-ia.md` — Whisper, TTS, pipeline voz→IA→voz, asistente de reuniones

---

## Bloque 9 — Agentes avanzados ✅

- [x] `tutoriales/agentes-avanzados/README.md`
- [x] `01-multi-agente.md` — Orquestador manual, CrewAI, comunicación asíncrona
- [x] `02-model-context-protocol.md` — Servidor MCP, herramientas, recursos, Claude Desktop
- [x] `03-computer-use.md` — Bucle de control, automatización de formularios, scraping visual
- [x] `04-memoria-largo-plazo.md` — Memoria episódica JSON, semántica ChromaDB, compresión
