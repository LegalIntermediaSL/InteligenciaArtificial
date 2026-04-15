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

## Bloque 2 — Extensiones avanzadas (Agentes y Modelos) ✅

- [x] Tutorial LangGraph: StateGraph, ciclos, checkpointing, human-in-the-loop, multi-agente (`tutoriales/llms/08-langgraph.md`)
- [x] Tutorial Modelos locales con Ollama: ventajas privacidad/coste, catálogo, integración Python/LangChain, RAG local (`tutoriales/llms/09-modelos-locales-ollama.md`)
- [x] Tutorial MCP — Model Context Protocol: servidores, primitivas, Claude Desktop, NotebookLM, seguridad (`tutoriales/llms/10-mcp.md`)
- [x] Notebook LangGraph (`notebooks/llms/08-langgraph.ipynb`)
- [x] Notebook Ollama (`notebooks/llms/09-modelos-locales-ollama.ipynb`)

---

## Bloque 1 — Extensiones técnicas (Fundamentos) ✅

- [x] Tutorial Arquitectura Transformer en profundidad: Q/K/V, Multi-Head, positional encoding, scaling laws (`tutoriales/fundamentos/04-arquitectura-transformers.md`)
- [x] Tutorial Algoritmos fundamentales: gradient descent, backprop, activaciones, CNN, RNN, RLHF, embeddings (`tutoriales/fundamentos/05-algoritmos-fundamentales.md`)

---

## Próximas extensiones planificadas

- [ ] Tutorial evaluación de agentes (LLM-as-judge, benchmarks, LangSmith)
- [ ] Tutorial memoria persistente (Mem0, episodic memory, retrieval memory)
- [ ] Tutorial structured outputs con Pydantic + instructor
- [ ] Tutorial modelos multimodales (visión, análisis de imágenes con Claude)
- [ ] Tutorial agentes con CrewAI (multi-agente declarativo)
- [ ] Notebook MCP (`notebooks/llms/10-mcp.ipynb`)

---

## Mejoras futuras ✅

- [x] Añadir CI/CD para validar notebooks (`.github/workflows/validate-notebooks.yml`)
- [x] Publicar como GitHub Pages (`.github/workflows/gh-pages.yml` + `mkdocs.yml`)
- [x] Añadir ejemplos en JavaScript (`javascript/` — LangChain.js, Vercel AI SDK)
- [x] Notebook para agentes de IA (`notebooks/llms/04-agentes-ia.ipynb`)
- [x] Notebook para RAG con ChromaDB (`notebooks/llms/05-rag-chromadb.ipynb`)
- [x] Notebook para fine-tuning con LoRA (`notebooks/llms/06-finetuning-lora.ipynb`)
