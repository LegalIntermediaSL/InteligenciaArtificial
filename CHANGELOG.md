# Changelog

Todos los cambios relevantes del proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es/1.0.0/),
y el proyecto sigue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
