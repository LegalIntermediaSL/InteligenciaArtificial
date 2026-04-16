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
| **2 — LLMs** | Cómo funcionan los modelos de lenguaje, prompt engineering, RAG, fine-tuning, DSPy, LangGraph |
| **3 — APIs de IA** | Integración con Anthropic (Claude), OpenAI y comparativa de proveedores |
| **4 — Python para IA** | Librerías esenciales, notebooks y pipelines de datos |
| **5 — Casos de uso** | Proyectos prácticos: chatbots, clasificación, resumen, extracción de PDFs |
| **6 — JavaScript** | LangChain.js, Vercel AI SDK, chatbot en producción con Next.js |
| **7 — Producción y evaluación** | LLM-as-judge, observabilidad con Langfuse, optimización de costos, despliegue con Docker |
| **8 — Multimodalidad** | Visión con LLMs, generación de imágenes (DALL-E, Stable Diffusion), voz e IA |
| **9 — Agentes avanzados** | Multi-agente, Model Context Protocol, Computer Use, memoria a largo plazo |
| **10 — Casos de uso avanzados** | Revisión de código, búsqueda semántica, structured output con Instructor, análisis de datos |
| **11 — IA local** | Ollama, Transformers locales, Transformers.js en el navegador, comparativa local vs cloud |
| **12 — Seguridad en IA** | Prompt injection, jailbreaking y guardrails, PII, auditoría y trazabilidad |
| **13 — Bases de datos vectoriales** | pgvector, Pinecone, Weaviate, reranking, RAG avanzado |

---

## Estructura del repositorio

```
InteligenciaArtificial/
├── tutoriales/
│   ├── fundamentos/                  # Bloque 1 — Conceptos base
│   ├── llms/                         # Bloque 2 — Modelos de lenguaje
│   ├── apis/                         # Bloque 3 — APIs de IA
│   ├── python-para-ia/               # Bloque 4 — Python aplicado a IA
│   ├── casos-de-uso/                 # Bloque 5 — Proyectos prácticos básicos
│   ├── produccion/                   # Bloque 7 — Producción y evaluación
│   ├── multimodalidad/               # Bloque 8 — Visión, imágenes, voz
│   ├── agentes-avanzados/            # Bloque 9 — Multi-agente, MCP, Computer Use
│   ├── casos-de-uso-avanzados/       # Bloque 10 — Casos de uso avanzados
│   ├── ia-local/                     # Bloque 11 — Modelos sin APIs externas
│   ├── seguridad/                    # Bloque 12 — Seguridad defensiva
│   ├── bases-de-datos-vectoriales/   # Bloque 13 — Vector DBs y RAG avanzado
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
                          └─► Bloque 13 (Vector DBs)
```

Los bloques del 7 al 13 son independientes entre sí y pueden leerse en cualquier orden una vez completados los 5 primeros.

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

---

## Licencia

Distribuido bajo licencia **MIT** — ver [LICENSE](./LICENSE) para más detalles.

---

*Mantenido por [LegalIntermediaSL](https://github.com/LegalIntermediaSL) · Última actualización: abril 2026*
