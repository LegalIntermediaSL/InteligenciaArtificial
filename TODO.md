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
- [x] `05-autogen.md` — Microsoft AutoGen, GroupChat, orquestación multi-agente
- [x] `06-a2a-protocol.md` — A2A Protocol (Google 2025), servidor/cliente FastAPI
- [x] `07-langgraph-agentes.md` — StateGraph, checkpointing, human-in-the-loop, subgrafos, streaming
- [x] `08-evaluacion-agentes.md` — Métricas de trayectoria, LLM-as-judge, tracing Langfuse, red teaming
- [x] `09-agentes-codigo.md` — exec(), sandboxing E2B, agente analista, debugging automatizado, generador de tests
- [x] `10-agentes-especializados.md` — Research Agent, Financial Agent, Code Review Agent

---

## Bloque 10 — Casos de uso avanzados ✅

- [x] `tutoriales/casos-de-uso-avanzados/README.md`
- [x] `01-generacion-revision-codigo.md` — Revisor con rúbrica JSON, generador de tests, documentador AST
- [x] `02-busqueda-semantica.md` — Embeddings, índice local, chatbot RAG, buscador Streamlit
- [x] `03-structured-output-instructor.md` — Instructor + Pydantic, validadores, modelos anidados
- [x] `04-analisis-datos-ia.md` — Agente analista, tool_use, gráficas, detección de anomalías

---

## Bloque 11 — IA local ✅

- [x] `tutoriales/ia-local/README.md`
- [x] `01-ollama.md` — CLI, API REST, SDK nativo, visión local, RAG local
- [x] `02-transformers-local.md` — Pipeline, quantización 4-bit, embeddings locales
- [x] `03-transformers-js-navegador.md` — Node.js, navegador, Web Workers, buscador offline
- [x] `04-comparativa-local-cloud.md` — Costes, privacidad, router híbrido

---

## Bloque 12 — Seguridad en IA ✅

- [x] `tutoriales/seguridad/README.md`
- [x] `01-prompt-injection.md` — Ataques directos/indirectos, detector, GuardrailsManager
- [x] `02-jailbreaking-guardrails.md` — Guardrails entrada/salida, moderación, middleware FastAPI
- [x] `03-datos-sensibles-pii.md` — Presidio, anonimización, logger seguro
- [x] `04-auditoria-seguridad.md` — AuditLogger, detección anomalías, dashboard, Monitor

---

## Bloque 13 — Bases de datos vectoriales ✅

- [x] `tutoriales/bases-de-datos-vectoriales/README.md`
- [x] `01-pgvector.md` — PostgreSQL, psycopg2, búsqueda coseno, HNSW, RAG completo
- [x] `02-pinecone-weaviate.md` — Pinecone, Weaviate, búsqueda híbrida, comparativa
- [x] `03-reranking.md` — Cross-Encoders, Cohere Rerank, NDCG/MRR, pipeline con reranking
- [x] `04-rag-avanzado.md` — HyDE, Parent-Child, Self-Query, multi-documento, ragas

---

## Mejoras transversales ✅

- [x] `javascript/03-nextjs-ia.md` — Next.js 15, App Router, streaming, Tailwind, rate limiting
- [x] `tutoriales/llms/08-dspy.md` — Signatures, módulos, BootstrapFewShot, RAG con DSPy
- [x] `tutoriales/llms/09-langgraph.md` — StateGraph, condicionales, MemorySaver, human-in-the-loop
- [x] Notebooks bloque 7 — `produccion/01-evaluacion-llms.ipynb`, `02-observabilidad.ipynb`, `03-optimizacion-costos.ipynb`
- [x] Notebooks bloque 8 — `multimodalidad/01-vision-llms.ipynb`, `02-generacion-imagenes.ipynb`, `03-voz-ia.ipynb`
- [x] Notebooks bloque 9 — `agentes-avanzados/01-multi-agente.ipynb`, `02-mcp.ipynb`, `04-memoria-largo-plazo.ipynb`

---

## Ampliaciones — LLMs, APIs, Agentes, Seguridad ✅

- [x] `tutoriales/llms/10-gestion-contexto.md` — Context window, compresión, sliding window, prompt caching
- [x] `tutoriales/llms/11-tokenizacion.md` — BPE, WordPiece, SentencePiece, costos por idioma
- [x] `tutoriales/llms/12-function-calling-avanzado.md` — Tool use, parallel tools, Pydantic, patrones agénticos
- [x] `tutoriales/apis/04-google-gemini.md` — Familia Gemini, contexto 1M, multimodalidad, Vertex AI
- [x] `tutoriales/apis/05-mistral-cohere.md` — Mistral (open weights), Cohere (embeddings, reranking)
- [x] `tutoriales/agentes-avanzados/05-autogen.md` — Microsoft AutoGen, GroupChat, orquestación multi-agente
- [x] `tutoriales/agentes-avanzados/06-a2a-protocol.md` — A2A Protocol (Google 2025), servidor/cliente FastAPI
- [x] `tutoriales/seguridad/05-red-teaming.md` — Red teaming manual y automatizado, evaluación adversarial

---

## Bloque 14 — Fine-tuning avanzado ✅

- [x] `tutoriales/finetuning-avanzado/README.md`
- [x] `01-dpo-rlhf.md` — DPO, RLHF, TRL DPOTrainer, dataset de preferencias
- [x] `02-instruction-tuning.md` — SFTTrainer, formatos Alpaca/ShareGPT/ChatML, chat templates
- [x] `03-evaluacion-modelos-finetuneados.md` — ROUGE, BLEU, LLM-as-judge, lm-eval, W&B
- [x] `04-despliegue-modelos-propios.md` — vLLM, TGI, AWQ/GPTQ, FastAPI, Docker, benchmarks

---

## Bloque 15 — IA responsable ✅

- [x] `tutoriales/ia-responsable/README.md`
- [x] `01-sesgos-fairness.md` — Tipos de sesgo, fairlearn, aif360, mitigación, visualización
- [x] `02-interpretabilidad.md` — SHAP, LIME, attention visualization, saliency maps
- [x] `03-model-cards-datasheets.md` — Model cards (Mitchell), datasheets (Gebru), model-card-toolkit, HF Hub
- [x] `04-cumplimiento-gdpr.md` — GDPR Art. 22, EU AI Act, minimización, anonimización, machine unlearning

---

## Bloque 16 — MLOps ✅

- [x] `tutoriales/mlops/README.md`
- [x] `01-registro-modelos.md` — MLflow, Model Registry, W&B Artifacts, HuggingFace Hub
- [x] `02-ab-testing-llms.md` — Shadow deployment, router A/B, significancia estadística, canary
- [x] `03-deteccion-drift.md` — PSI, KS test, Evidently, alertas Slack/email, dashboard Streamlit
- [x] `04-pipelines-automatizados.md` — Prefect, pipeline completo, triggers, Docker, MLflow integrado

---

## Bloque 17 — Automatización con IA ✅

- [x] `tutoriales/automatizacion/README.md`
- [x] `01-n8n-ia.md` — n8n Docker, AI Agent node, casos prácticos, self-hosted vs cloud
- [x] `02-make-zapier-ia.md` — Make.com escenarios, Zapier zaps, casos prácticos, guía de decisión
- [x] `03-pipelines-negocio.md` — LangChain LCEL, Batch API, Celery/Redis, pipeline de contratos
- [x] `04-integracion-herramientas-negocio.md` — HubSpot, Google Workspace, Microsoft 365, Slack bot, Notion

---

## Notebooks pendientes de crear ✅

- [x] `notebooks/mlops/01-registro-modelos.ipynb`
- [x] `notebooks/mlops/02-ab-testing-llms.ipynb`
- [x] `notebooks/mlops/03-deteccion-drift.ipynb`
- [x] `notebooks/mlops/04-pipelines-automatizados.ipynb`
- [x] `notebooks/automatizacion/01-n8n-ia.ipynb`
- [x] `notebooks/automatizacion/02-make-zapier-ia.ipynb`
- [x] `notebooks/automatizacion/03-pipelines-negocio.ipynb`
- [x] `notebooks/automatizacion/04-integracion-herramientas-negocio.ipynb`
- [x] `notebooks/produccion/04-despliegue.ipynb`
- [x] `notebooks/agentes-avanzados/03-computer-use.ipynb`

---

## Bloque 18 — Funcionalidades avanzadas de la API Anthropic ✅

- [x] `tutoriales/apis-avanzadas/01-extended-thinking.md` — Razonamiento extendido (claude-sonnet-4-6), budget_tokens, casos de uso matemáticos y lógicos
- [x] `tutoriales/apis-avanzadas/02-batch-api.md` — Batch API para procesamiento masivo asíncrono, ahorro de costes, polling y webhooks
- [x] `tutoriales/apis-avanzadas/03-files-api.md` — Files API para subida de documentos grandes, reutilización entre llamadas, formatos soportados
- [x] `tutoriales/apis-avanzadas/04-citations-api.md` — Citas automáticas de fuentes, grounding factual, RAG con atribución
- [x] `tutoriales/apis-avanzadas/05-prompt-caching.md` — Cache de prompts estáticos, reducción de costes hasta 90%, patrones de uso
- [x] `notebooks/apis-avanzadas/01-extended-thinking.ipynb`
- [x] `notebooks/apis-avanzadas/02-batch-api.ipynb`
- [x] `notebooks/apis-avanzadas/03-files-api.ipynb`
- [x] `notebooks/apis-avanzadas/04-citations-api.ipynb`
- [x] `notebooks/apis-avanzadas/05-prompt-caching.ipynb`

---

## Bloque 19 — Claude Agent SDK y construcción de agentes ✅

- [x] `tutoriales/agent-sdk/01-primeros-pasos.md` — Agent SDK de Anthropic, estructura básica, herramientas, bucle agéntico
- [x] `tutoriales/agent-sdk/02-agentes-compuestos.md` — Orquestador + subagentes especializados, paso de contexto
- [x] `tutoriales/agent-sdk/03-agentes-con-memoria.md` — Memoria a corto y largo plazo, gestión del contexto entre sesiones
- [x] `tutoriales/agent-sdk/04-despliegue-agentes.md` — FastAPI wrapper, streaming de eventos, monitorización, escalado
- [x] `notebooks/agent-sdk/01-primeros-pasos.ipynb`
- [x] `notebooks/agent-sdk/02-agentes-compuestos.ipynb`
- [x] `notebooks/agent-sdk/03-agentes-con-memoria.ipynb`
- [x] `notebooks/agent-sdk/04-despliegue-agentes.ipynb`

---

## Bloque 20 — IA en la empresa: casos reales ✅

- [x] `tutoriales/ia-empresarial/01-estrategia-ia.md` — Cómo evaluar e implementar IA en una empresa, ROI, change management
- [x] `tutoriales/ia-empresarial/02-ia-legal-compliance.md` — IA en despachos legales, contratos, due diligence, límites legales
- [x] `tutoriales/ia-empresarial/03-ia-customer-success.md` — IA para retención de clientes, CSAT, predicción de churn
- [x] `tutoriales/ia-empresarial/04-ia-producto.md` — IA en el proceso de producto: user research, priorización, A/B testing
- [x] `notebooks/ia-empresarial/01-estrategia-ia.ipynb`
- [x] `notebooks/ia-empresarial/02-ia-legal-compliance.ipynb`
- [x] `notebooks/ia-empresarial/03-ia-customer-success.ipynb`
- [x] `notebooks/ia-empresarial/04-ia-producto.ipynb`
- [x] `REFERENCIAS.md` — Referencias, vínculos y documentación relacionada

---

## Bloque 21 — IA para Startups ✅

- [x] `tutoriales/ia-startups/README.md`
- [x] `tutoriales/ia-startups/01-stack-minimo.md` — Stack mínimo, selector de modelo, backend MVP con FastAPI
- [x] `tutoriales/ia-startups/02-mvp-dos-semanas.md` — Metodología 2 semanas, versionado de prompts, pipeline y pricing
- [x] `tutoriales/ia-startups/03-costes-y-escalado.md` — Monitor de costes, router de modelos, prompt caching, proyecciones
- [x] `tutoriales/ia-startups/04-ia-ventaja-competitiva.md` — Data flywheel, evaluación de moat, narrativa inversores
- [x] `notebooks/ia-startups/01-stack-minimo.ipynb`
- [x] `notebooks/ia-startups/02-mvp-dos-semanas.ipynb`
- [x] `notebooks/ia-startups/03-costes-y-escalado.ipynb`
- [x] `notebooks/ia-startups/04-ia-ventaja-competitiva.ipynb`

---

## Bloque 22 — Workflows con n8n y Claude ✅

- [x] `tutoriales/n8n-workflows/README.md`
- [x] `tutoriales/n8n-workflows/01-introduccion-n8n.md` — Instalación, conceptos, primer workflow con IA
- [x] `tutoriales/n8n-workflows/02-workflows-documentos.md` — Procesamiento de facturas, contratos y PDFs con Claude
- [x] `tutoriales/n8n-workflows/03-workflows-negocio.md` — CRM, email marketing, Slack bot y reportes automáticos
- [x] `tutoriales/n8n-workflows/04-workflows-avanzados.md` — Sub-workflows, webhooks, manejo de errores y producción
- [x] `notebooks/n8n-workflows/01-introduccion-n8n.ipynb`
- [x] `notebooks/n8n-workflows/02-workflows-documentos.ipynb`
- [x] `notebooks/n8n-workflows/03-workflows-negocio.ipynb`
- [x] `notebooks/n8n-workflows/04-workflows-avanzados.ipynb`

---

## Bloque 25 — Proyectos integradores ✅

- [x] `tutoriales/proyectos-integradores/README.md`
- [x] `tutoriales/proyectos-integradores/01-saas-contratos.md` — SaaS análisis legal: Files API, ChromaDB, FastAPI, rate limiting
- [x] `tutoriales/proyectos-integradores/02-chatbot-soporte.md` — Chatbot soporte: KB vectorial, memoria, Haiku→Sonnet→escalado
- [x] `tutoriales/proyectos-integradores/03-pipeline-contenido.md` — Pipeline multi-formato: blog/Twitter/LinkedIn/newsletter + evaluación
- [x] `tutoriales/proyectos-integradores/04-asistente-datos.md` — Asistente datos: tool use, sandbox Pandas, bucle agéntico
- [x] `notebooks/proyectos-integradores/01-saas-contratos.ipynb`
- [x] `notebooks/proyectos-integradores/02-chatbot-soporte.ipynb`
- [x] `notebooks/proyectos-integradores/03-pipeline-contenido.ipynb`
- [x] `notebooks/proyectos-integradores/04-asistente-datos.ipynb`

---

## Bloque 23 — Claude Code y desarrollo asistido por IA ✅

- [x] `tutoriales/claude-code/README.md`
- [x] `tutoriales/claude-code/01-claude-code-cli.md` — Instalación, slash commands, CLAUDE.md, configuración avanzada
- [x] `tutoriales/claude-code/02-mcp-servers.md` — Servidores MCP en Python, recursos, prompts reutilizables, HTTP mode
- [x] `tutoriales/claude-code/03-hooks-automatizacion.md` — Hooks de ciclo de vida, linting, auditoría, CI/CD con GitHub Actions
- [x] `tutoriales/claude-code/04-flujos-desarrollo-ia.md` — Flujos completos, debug, code review, comparativa Cursor/Copilot
- [x] `notebooks/claude-code/01-claude-code-cli.ipynb`
- [x] `notebooks/claude-code/02-mcp-servers.ipynb`
- [x] `notebooks/claude-code/03-hooks-automatizacion.ipynb`
- [x] `notebooks/claude-code/04-flujos-desarrollo-ia.ipynb`

---

## Bloque 24 — PydanticAI y frameworks de agentes ✅

- [x] `tutoriales/pydanticai/README.md`
- [x] `tutoriales/pydanticai/01-pydanticai-intro.md` — Agent tipado, result_type, deps_type, streaming, historial
- [x] `tutoriales/pydanticai/02-pydanticai-avanzado.md` — TestModel, ModelRetry, nested agents, FastAPI, Logfire
- [x] `tutoriales/pydanticai/03-mastra-ai.md` — TypeScript agent framework, createTool, Workflow, Memory, evals
- [x] `tutoriales/pydanticai/04-comparativa-frameworks.md` — PydanticAI vs LangGraph vs CrewAI vs AutoGen vs Mastra
- [x] `notebooks/pydanticai/01-pydanticai-intro.ipynb`
- [x] `notebooks/pydanticai/02-pydanticai-avanzado.ipynb`
- [x] `notebooks/pydanticai/03-mastra-ai.ipynb`
- [x] `notebooks/pydanticai/04-comparativa-frameworks.ipynb`

---

## Bloque 1 ampliado — Fundamentos (redes neuronales y Transformers) ✅

- [x] `tutoriales/fundamentos/04-redes-neuronales.md` — Neurona, activaciones, MLP desde cero, backpropagation, PyTorch
- [x] `tutoriales/fundamentos/05-transformers-atencion.md` — Atención Q/K/V, multi-cabeza, PE sinusoidal, ley Chinchilla
- [x] `notebooks/fundamentos/04-redes-neuronales.ipynb`
- [x] `notebooks/fundamentos/05-transformers-atencion.ipynb`

---

## Bloque 8 ampliado — Multimodalidad (video e IA) ✅

- [x] `tutoriales/multimodalidad/04-video-ia.md` — Gemini video, Claude Vision frames, Whisper, Runway ML, SRT
- [x] `notebooks/multimodalidad/04-video-ia.ipynb`

---

## Notebooks pendientes bloque 2 (LLMs) ✅

- [x] `notebooks/llms/01-que-es-un-llm.ipynb`
- [x] `notebooks/llms/03-finetuning-vs-rag.ipynb`
- [x] `notebooks/llms/08-dspy.ipynb`
- [x] `notebooks/llms/09-langgraph.ipynb`
- [x] `notebooks/llms/10-gestion-contexto.ipynb`
- [x] `notebooks/llms/11-tokenizacion.ipynb`
- [x] `notebooks/llms/12-function-calling-avanzado.ipynb`

---

## Notebooks pendientes bloque 3 (APIs) ✅

- [x] `notebooks/apis/02-api-openai.ipynb`
- [x] `notebooks/apis/03-comparativa-proveedores.ipynb`
- [x] `notebooks/apis/04-google-gemini.ipynb`
- [x] `notebooks/apis/05-mistral-cohere.ipynb`

---

## Notebooks pendientes bloque 4 (Python para IA) ✅

- [x] `notebooks/python-para-ia/01-intro-python-ia.ipynb`
- [x] `notebooks/python-para-ia/02-librerias-esenciales.ipynb`
- [x] `notebooks/python-para-ia/03-jupyter-notebooks.ipynb`
