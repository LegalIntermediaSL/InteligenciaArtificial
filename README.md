# InteligenciaArtificial

> Tutoriales, sistemas y modelos de Inteligencia Artificial — por [LegalIntermediaSL](https://github.com/LegalIntermediaSL)

---

## Objetivo

Este repositorio es un recurso de aprendizaje y referencia sobre Inteligencia Artificial, con especial enfoque en su aplicación práctica en entornos profesionales y empresariales.

El proyecto nace con la vocación de ser una guía progresiva: desde los conceptos más básicos hasta la construcción de sistemas reales con IA. Está dirigido tanto a perfiles técnicos que quieren iniciarse en IA como a equipos que buscan integrar estas tecnologías en sus procesos.

---

## ¿Qué encontrarás aquí?

| Bloque | Descripción |
|---|---|
| **1 — Fundamentos** | Conceptos clave de IA, ML y DL explicados de forma clara |
| **2 — LLMs** | Modelos de lenguaje, prompt engineering, RAG, fine-tuning, DSPy, LangGraph, tokenización, function calling |
| **3 — APIs de IA** | Claude (Anthropic), OpenAI, Google Gemini, Mistral, Cohere — guías y comparativa |
| **4 — Python para IA** | Librerías esenciales, notebooks y pipelines de datos |
| **5 — Casos de uso** | Proyectos prácticos: chatbots, clasificación, resumen, extracción de PDFs |
| **6 — JavaScript** | LangChain.js, Vercel AI SDK, chatbot en producción con Next.js |
| **7 — Producción y evaluación** | LLM-as-judge, observabilidad con Langfuse, optimización de costos, despliegue con Docker |
| **8 — Multimodalidad** | Visión con LLMs, generación de imágenes (DALL-E, Stable Diffusion), voz e IA |
| **9 — Agentes avanzados** | Multi-agente, MCP, Computer Use, memoria, AutoGen, A2A Protocol |
| **10 — Casos de uso avanzados** | Revisión de código, búsqueda semántica, structured output con Instructor, análisis de datos |
| **11 — IA local** | Ollama, Transformers locales, Transformers.js en el navegador, comparativa local vs cloud |
| **12 — Seguridad en IA** | Prompt injection, jailbreaking, PII, auditoría, red teaming y evaluación adversarial |
| **13 — Bases de datos vectoriales** | pgvector, Pinecone, Weaviate, reranking, RAG avanzado |
| **14 — Fine-tuning avanzado** | DPO, RLHF, instruction tuning, evaluación de modelos, despliegue con vLLM y TGI |
| **15 — IA responsable** | Sesgos y fairness, interpretabilidad (SHAP, LIME), model cards, GDPR y EU AI Act |
| **16 — MLOps** | Registro de modelos, A/B testing, detección de drift, pipelines automatizados con Prefect |
| **17 — Automatización con IA** | n8n, Make, Zapier, pipelines de negocio, HubSpot, Slack bot, Google Workspace |
| **18 — APIs avanzadas Anthropic** | Extended Thinking, Batch API, Files API, Citations API, Prompt Caching |
| **19 — Claude Agent SDK** | Primeros pasos, agentes compuestos, memoria, despliegue con FastAPI |
| **20 — IA en la empresa** | Estrategia, legal/compliance, Customer Success, producto |
| **21 — IA para startups** | Stack mínimo, MVP en 2 semanas, costes y escalado, ventaja competitiva |
| **22 — Workflows con n8n** | Orquestación visual con Claude: documentos, negocio, webhooks, producción |
| **23 — Claude Code** | CLI, MCP servers personalizados, hooks, flujos de desarrollo con IA |
| **24 — PydanticAI y frameworks** | PydanticAI (tipado, TestModel, nested agents), Mastra.ai (TypeScript), comparativa LangGraph/CrewAI/AutoGen |
| **25 — Proyectos integradores** | 4 proyectos completos: SaaS contratos, chatbot soporte, pipeline contenido, asistente datos |

---

## Estructura del repositorio

```
InteligenciaArtificial/
├── tutoriales/
│   ├── fundamentos/                  # Bloque 1 — Conceptos base
│   ├── llms/                         # Bloque 2 — Modelos de lenguaje (12 tutoriales)
│   ├── apis/                         # Bloque 3 — APIs de IA (5 proveedores)
│   ├── python-para-ia/               # Bloque 4 — Python aplicado a IA
│   ├── casos-de-uso/                 # Bloque 5 — Proyectos prácticos básicos
│   ├── produccion/                   # Bloque 7 — Producción y evaluación
│   ├── multimodalidad/               # Bloque 8 — Visión, imágenes, voz
│   ├── agentes-avanzados/            # Bloque 9 — Multi-agente, MCP, AutoGen, A2A
│   ├── casos-de-uso-avanzados/       # Bloque 10 — Casos de uso avanzados
│   ├── ia-local/                     # Bloque 11 — Modelos sin APIs externas
│   ├── seguridad/                    # Bloque 12 — Seguridad + red teaming
│   ├── bases-de-datos-vectoriales/   # Bloque 13 — Vector DBs y RAG avanzado
│   ├── finetuning-avanzado/          # Bloque 14 — DPO, RLHF, vLLM, TGI
│   ├── ia-responsable/               # Bloque 15 — Fairness, GDPR, EU AI Act
│   ├── mlops/                        # Bloque 16 — MLOps, drift, pipelines
│   ├── automatizacion/               # Bloque 17 — n8n, Make, Zapier, integraciones
│   ├── apis-avanzadas/               # Bloque 18 — APIs avanzadas Anthropic
│   ├── agent-sdk/                    # Bloque 19 — Claude Agent SDK
│   ├── ia-empresarial/               # Bloque 20 — IA en la empresa
│   ├── ia-startups/                  # Bloque 21 — IA para startups
│   ├── n8n-workflows/                # Bloque 22 — Workflows con n8n y Claude
│   ├── claude-code/                  # Bloque 23 — Claude Code y desarrollo asistido
│   ├── pydanticai/                   # Bloque 24 — PydanticAI, Mastra.ai y frameworks de agentes
│   ├── proyectos-integradores/       # Bloque 25 — Proyectos completos de principio a fin
│   └── notebooks/                    # Notebooks Jupyter interactivos (todos los bloques)
├── javascript/                       # Bloque 6 — Ejemplos JS/TS
├── README.md                         # Este archivo
├── CHANGELOG.md                      # Historial de cambios
├── BITACORA.md                       # Diario de decisiones y aprendizajes
└── TODO.md                           # Hoja de ruta
```

Cada carpeta dentro de `tutoriales/` incluye su propio `README.md` con descripción, requisitos y orden de lectura recomendado.

---

## Orden de lectura recomendado

```
Bloque 1 (Fundamentos)
  └─► Bloque 2 (LLMs)
        └─► Bloque 3 (APIs de IA)
              └─► Bloque 4 (Python para IA)
                    └─► Bloque 5 (Casos de uso)
                          ├─► Bloque 6 (JavaScript)
                          ├─► Bloque 7 (Producción)
                          ├─► Bloque 8 (Multimodalidad)
                          ├─► Bloque 9 (Agentes avanzados)
                          ├─► Bloque 10 (Casos avanzados)
                          ├─► Bloque 11 (IA local)
                          ├─► Bloque 12 (Seguridad)
                          ├─► Bloque 13 (Vector DBs)
                          ├─► Bloque 14 (Fine-tuning avanzado)
                          ├─► Bloque 15 (IA responsable)
                          ├─► Bloque 16 (MLOps)
                          ├─► Bloque 17 (Automatización)
                          ├─► Bloque 18 (APIs avanzadas Anthropic)
                          ├─► Bloque 19 (Agent SDK)
                          ├─► Bloque 20 (IA empresarial)
                          ├─► Bloque 21 (IA para startups)
                          ├─► Bloque 22 (Workflows n8n)
                          ├─► Bloque 23 (Claude Code)
                          ├─► Bloque 24 (PydanticAI y frameworks)
                          └─► Bloque 25 (Proyectos integradores)
```

Los bloques del 7 al 23 son independientes entre sí y pueden leerse en cualquier orden una vez completados los 5 primeros. El Bloque 25 (Proyectos integradores) combina conocimientos de varios bloques y se recomienda como cierre.

---

## Requisitos generales

Cada tutorial especifica sus propias dependencias, pero en general el proyecto trabaja con:

- **Python** 3.10 o superior
- **Jupyter Notebooks** (recomendado: JupyterLab o VS Code con extensión Jupyter)
- **Node.js** 18+ (para los tutoriales de JavaScript)
- **Variables de entorno**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` según el tutorial

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/LegalIntermediaSL/InteligenciaArtificial.git
cd InteligenciaArtificial

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Instalar dependencias de un bloque concreto
pip install -r tutoriales/apis/requirements.txt

# Fichero .env en la raíz (o en cada carpeta)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

---

## Notebooks interactivos

Todos los bloques incluyen cuadernos Jupyter que permiten ejecutar el código directamente:

```bash
pip install jupyterlab
jupyter lab
```

También puedes abrirlos en **VS Code** (extensión Jupyter de Microsoft) o en **Google Colab** subiéndolos directamente.

---

## Cómo contribuir

Este repositorio es de uso interno y aprendizaje, pero las sugerencias son bienvenidas:

1. Abre un [issue](https://github.com/LegalIntermediaSL/InteligenciaArtificial/issues) describiendo la mejora o error
2. Haz un fork del repositorio
3. Crea una rama con tu aportación (`git checkout -b mejora/nombre`)
4. Abre un Pull Request

---

## Recursos externos recomendados

- [Documentación oficial de Anthropic](https://docs.anthropic.com)
- [OpenAI Cookbook](https://cookbook.openai.com)
- [fast.ai — Practical Deep Learning](https://course.fast.ai)
- [Hugging Face Learn](https://huggingface.co/learn)
- [Papers With Code](https://paperswithcode.com)
- [LangChain Docs](https://python.langchain.com)
- [DSPy Documentation](https://dspy.ai)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Mistral AI Docs](https://docs.mistral.ai)
- [Cohere Documentation](https://docs.cohere.com)
- [Google Gemini API](https://ai.google.dev/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [Evidently AI](https://docs.evidentlyai.com)
- [Prefect Docs](https://docs.prefect.io)
- [AutoGen](https://microsoft.github.io/autogen/)
- [A2A Protocol](https://github.com/google/A2A)

---

## Aviso legal

> El contenido de este repositorio se facilita "tal cual es", puede contener errores o inexactitudes. Legal Intermedia S.L. no se responsabiliza del uso que se haga de esta información que se facilita con fines instructivos. Se han utilizado asistentes digitales para su confección. Lea [LICENSE](./LICENSE) para mayor detalle.

---

## Licencia

Distribuido bajo licencia **MIT** — ver [LICENSE](./LICENSE) para más detalles.

---

*Mantenido por [LegalIntermediaSL](https://github.com/LegalIntermediaSL) · 17 bloques · 84+ tutoriales · Última actualización: abril 2026*
