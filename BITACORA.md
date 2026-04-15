# Bitácora del Proyecto

Registro cronológico de decisiones, aprendizajes, problemas encontrados y soluciones adoptadas durante el desarrollo del proyecto.

---

## 2026-04-15 (v0.5.0)

### Extensión: LangGraph, Ollama y MCP

**Motivación:** el proyecto cubría bien la teoría de agentes (07-tipos-agentes.md) pero carecía de implementaciones con frameworks modernos y del caso de uso de modelos locales, cada vez más relevante para privacidad y coste.

**LangGraph (tutorial 08):**
- Decisión: cubrir LangGraph como el estándar de facto para agentes stateful, por encima del patrón bucle-while ya presente en el tutorial 04.
- Se prioriza explicar el "por qué" (grafos vs cadenas) antes del "cómo", porque el salto conceptual es significativo.
- `ToolNode` + `tools_condition` como patrón canónico del ciclo ReAct en LangGraph.
- `MemorySaver` para demos; se menciona `SqliteSaver` para persistencia real, evitando dependencias externas en el notebook.
- Se incluye el patrón Supervisor + subagentes como implementación práctica de la teoría de multi-agente del tutorial 07.

**Modelos locales con Ollama (tutorial 09):**
- Decisión: dedicar una sección completa a las ventajas de los modelos locales porque es un tema que va más allá de la técnica: privacidad, coste, reproducibilidad, soberanía de datos.
- Se estructura la sección 2 con cada ventaja desarrollada en profundidad, con ejemplos de código y cifras concretas.
- API compatible con OpenAI de Ollama: destacada porque permite migrar código existente con 2 líneas de cambio.
- RAG 100% local como caso de uso estrella: demuestra que toda una pipeline de IA puede funcionar sin internet ni costes.
- Tabla de hardware/modelos: añadida como referencia práctica que faltaba en el proyecto.

**MCP — Model Context Protocol (tutorial 10):**
- Decisión: cubrir MCP porque es la capa de integración emergente más importante del ecosistema Anthropic y cada vez más adoptada.
- Se explica la diferencia conceptual entre MCP y tool use directo, para que el lector sepa cuándo usar cada opción.
- Se documenta el flujo con NotebookLM: no hay integración directa (NotebookLM es producto cerrado), pero se describe el flujo complementario real.
- Seguridad tratada como sección propia: mínimo privilegio, validación de inputs, no exponer credenciales en config files.

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
