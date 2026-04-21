# Referencias, Vínculos y Recursos de Apoyo

Recopilación de documentación oficial, cursos, papers y herramientas relevantes
para todos los bloques del repositorio.

---

## Documentación oficial

### Anthropic / Claude
| Recurso | URL | Descripción |
|---------|-----|-------------|
| Documentación Claude | https://docs.anthropic.com | Referencia completa de la API |
| Extended Thinking | https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking | Razonamiento extendido |
| Batch API | https://docs.anthropic.com/en/docs/build-with-claude/message-batches | Procesamiento asíncrono |
| Files API | https://docs.anthropic.com/en/docs/build-with-claude/files | Gestión de documentos |
| Citations | https://docs.anthropic.com/en/docs/build-with-claude/citations | Citas verificables |
| Prompt Caching | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching | Reducción de costes |
| Computer Use | https://docs.anthropic.com/en/docs/build-with-claude/computer-use | Control del ordenador |
| MCP (Model Context Protocol) | https://modelcontextprotocol.io | Protocolo de herramientas |
| Tool Use | https://docs.anthropic.com/en/docs/build-with-claude/tool-use | Herramientas y function calling |
| SDK Python | https://github.com/anthropics/anthropic-sdk-python | SDK oficial Python |
| Agent SDK | https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/lib/streaming | SDK de agentes |
| Cookbook | https://github.com/anthropics/anthropic-cookbook | Ejemplos oficiales |

### OpenAI
| Recurso | URL | Descripción |
|---------|-----|-------------|
| Documentación API | https://platform.openai.com/docs | Referencia completa |
| Cookbook | https://github.com/openai/openai-cookbook | Ejemplos y recetas |
| Fine-tuning | https://platform.openai.com/docs/guides/fine-tuning | Guía de fine-tuning |

### HuggingFace
| Recurso | URL | Descripción |
|---------|-----|-------------|
| Hub | https://huggingface.co/models | Catálogo de modelos |
| Transformers | https://huggingface.co/docs/transformers | Librería Transformers |
| TRL (Fine-tuning) | https://huggingface.co/docs/trl | DPO, SFT, RLHF |
| PEFT / LoRA | https://huggingface.co/docs/peft | Fine-tuning eficiente |
| Datasets | https://huggingface.co/docs/datasets | Gestión de datasets |
| Accelerate | https://huggingface.co/docs/accelerate | Entrenamiento distribuido |

### Google
| Recurso | URL | Descripción |
|---------|-----|-------------|
| Gemini API | https://ai.google.dev/docs | Documentación Gemini |
| Vertex AI | https://cloud.google.com/vertex-ai/docs | Plataforma MLOps de Google |
| A2A Protocol | https://google.github.io/A2A | Agent-to-Agent protocol |

### Modelos Claude 4.X
| Recurso | URL | Descripción |
|---------|-----|-------------|
| Claude claude-opus-4-7 | https://docs.anthropic.com/en/docs/about-claude/models | claude-opus-4-7 — máximo razonamiento |
| Claude claude-sonnet-4-6 | https://docs.anthropic.com/en/docs/about-claude/models | claude-sonnet-4-6 — equilibrio rendimiento/coste |
| Claude claude-haiku-4-5 | https://docs.anthropic.com/en/docs/about-claude/models | claude-haiku-4-5-20251001 — velocidad y bajo coste |
| Extended Thinking (betas) | https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking | interleaved-thinking-2025-05-14 |

---

## Frameworks y librerías

### Orquestación de agentes
| Librería | URL | Descripción |
|----------|-----|-------------|
| LangChain | https://python.langchain.com | Framework de agentes y RAG |
| LangGraph | https://langchain-ai.github.io/langgraph | Grafos de agentes con estado |
| AutoGen | https://microsoft.github.io/autogen | Multi-agente Microsoft |
| CrewAI | https://crewai.com | Equipos de agentes colaborativos |
| DSPy | https://dspy.ai | Programación declarativa para LLMs |
| PydanticAI | https://ai.pydantic.dev | Agentes tipados con Pydantic |
| Mastra | https://mastra.ai | Framework de agentes TypeScript |

### RAG y búsqueda vectorial
| Librería | URL | Descripción |
|----------|-----|-------------|
| ChromaDB | https://docs.trychroma.com | Base de datos vectorial embebida |
| Weaviate | https://weaviate.io/developers/weaviate | Base de datos vectorial cloud/local |
| Qdrant | https://qdrant.tech/documentation | Base de datos vectorial Rust |
| Pinecone | https://docs.pinecone.io | Base de datos vectorial serverless |
| pgvector | https://github.com/pgvector/pgvector | Extensión PostgreSQL para vectores |
| FAISS | https://github.com/facebookresearch/faiss | Búsqueda vectorial de Facebook |
| sentence-transformers | https://www.sbert.net | Embeddings multilingues |

### Despliegue y producción
| Herramienta | URL | Descripción |
|-------------|-----|-------------|
| FastAPI | https://fastapi.tiangolo.com | API REST en Python |
| vLLM | https://vllm.ai | Servidor LLM de alto rendimiento |
| TGI (HuggingFace) | https://huggingface.co/docs/text-generation-inference | Servidor LLM HuggingFace |
| Ollama | https://ollama.com | LLMs locales simplificados |
| Langfuse | https://langfuse.com/docs | Observabilidad para LLMs |
| MLflow | https://mlflow.org | Tracking de experimentos ML |

### Seguridad y privacidad
| Herramienta | URL | Descripción |
|-------------|-----|-------------|
| Presidio | https://microsoft.github.io/presidio | Detección y anonimización PII |
| Guardrails AI | https://guardrailsai.com | Validación de outputs LLM |

### Automatización
| Herramienta | URL | Descripción |
|-------------|-----|-------------|
| n8n | https://n8n.io | Automatización visual open-source |
| Make (Integromat) | https://make.com | Automatización low-code |
| Zapier | https://zapier.com | Automatización con 6000+ apps |
| Prefect | https://prefect.io | Orquestación de pipelines Python |

---

## Papers y lecturas recomendadas

### Fundamentos
| Paper | Año | Descripción |
|-------|-----|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Arquitectura Transformer original |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | BERT de Google |
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | Base de los LLMs actuales |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | Alineación de Anthropic |

### RAG y recuperación
| Paper | Año | Descripción |
|-------|-----|-------------|
| [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 2020 | Paper original de RAG |
| [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496) | 2022 | RAG con documentos hipotéticos |
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | 2023 | Métricas de evaluación RAG |

### Agentes
| Paper | Año | Descripción |
|-------|-----|-------------|
| [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629) | 2022 | Agentes con razonamiento y acción |
| [Toolformer](https://arxiv.org/abs/2302.04761) | 2023 | Modelos que aprenden a usar herramientas |
| [Auto-GPT y agentes autónomos](https://arxiv.org/abs/2306.02224) | 2023 | Survey de agentes LLM |

### Fine-tuning y alineación
| Paper | Año | Descripción |
|-------|-----|-------------|
| [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) | 2021 | Fine-tuning eficiente |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | QLoRA con 4-bit |
| [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | Alternativa a RLHF |
| [RLHF: Training Language Models to Follow Instructions](https://arxiv.org/abs/2203.02155) | 2022 | InstructGPT de OpenAI |

### IA responsable
| Paper | Año | Descripción |
|-------|-----|-------------|
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) | 2019 | Mitchell et al. — Model Cards |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | 2018 | Gebru et al. — Datasheets |
| [Fairness and Abstraction in Sociotechnical Systems](https://dl.acm.org/doi/10.1145/3287560.3287598) | 2019 | Fairness en IA |

---

## Cursos y formación

### Gratuitos
| Curso | Plataforma | Descripción |
|-------|-----------|-------------|
| [Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | DeepLearning.AI | Andrew Ng + OpenAI |
| [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | DeepLearning.AI | LangChain oficial |
| [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) | DeepLearning.AI | Pipelines con LLMs |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | HuggingFace | NLP completo con Transformers |
| [Fast.ai Practical Deep Learning](https://course.fast.ai) | Fast.ai | Deep Learning práctico |
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Google | Machine Learning desde cero |

### De pago (recomendados)
| Curso | Plataforma | Descripción |
|-------|-----------|-------------|
| [LLMOps Specialization](https://www.deeplearning.ai/courses/llmops-specialization/) | DeepLearning.AI | Operaciones con LLMs |
| [Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms) | Coursera/AWS | Fine-tuning y despliegue |

---

## Normativa y cumplimiento

| Recurso | URL | Descripción |
|---------|-----|-------------|
| EU AI Act (texto oficial) | https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32024R1689 | Reglamento IA de la UE |
| GDPR (texto oficial) | https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32016R0679 | GDPR en español |
| AEPD — IA y protección de datos | https://www.aepd.es/es/prensa-y-comunicacion/blog/ia-y-proteccion-de-datos | Guía AEPD |
| NIST AI RMF | https://airc.nist.gov/RMF_Overview | Framework de gestión de riesgos IA |
| ISO/IEC 42001 | https://www.iso.org/standard/81230.html | Estándar internacional sistemas IA |

---

## Comunidades y blogs

| Recurso | URL | Descripción |
|---------|-----|-------------|
| Anthropic Blog | https://www.anthropic.com/research | Investigación y anuncios |
| OpenAI Blog | https://openai.com/blog | Novedades OpenAI |
| HuggingFace Blog | https://huggingface.co/blog | Posts técnicos de la comunidad |
| The Batch (DeepLearning.AI) | https://www.deeplearning.ai/the-batch | Newsletter semanal de IA |
| r/MachineLearning | https://www.reddit.com/r/MachineLearning | Comunidad Reddit ML |
| Papers With Code | https://paperswithcode.com | Papers con implementaciones |
| Towards Data Science | https://towardsdatascience.com | Artículos técnicos accesibles |
| Sebastian Raschka Blog | https://sebastianraschka.com/blog | Fine-tuning y LLMs explicados |
| Lilian Weng Blog | https://lilianweng.github.io | Posts profundos sobre RL y LLMs |

---

## Herramientas de evaluación

| Herramienta | URL | Descripción |
|-------------|-----|-------------|
| RAGAS | https://docs.ragas.io | Evaluación automática de RAG |
| LM Evaluation Harness | https://github.com/EleutherAI/lm-evaluation-harness | Benchmarks de modelos |
| PromptFoo | https://promptfoo.dev | Testing de prompts |
| Weights & Biases | https://wandb.ai | Tracking de experimentos |
| Arize Phoenix | https://phoenix.arize.com | Observabilidad LLMs open-source |

### Benchmarks públicos

| Benchmark | URL | Descripción |
|-----------|-----|-------------|
| MMLU | https://paperswithcode.com/dataset/mmlu | 57 dominios de conocimiento |
| HumanEval | https://github.com/openai/human-eval | Generación de código Python |
| GSM8K | https://github.com/openai/grade-school-math | Razonamiento matemático |
| LMSYS Chatbot Arena | https://lmarena.ai | Comparativa humana de modelos |
| BIG-Bench | https://github.com/google/BIG-bench | Suite de 204 tareas |

### Computer Use y automatización visual

| Herramienta | URL | Descripción |
|-------------|-----|-------------|
| Computer Use (Anthropic) | https://docs.anthropic.com/en/docs/build-with-claude/computer-use | API de control visual |
| Playwright | https://playwright.dev | Automatización web determinista |
| PyAutoGUI | https://pyautogui.readthedocs.io | Control GUI multiplataforma |
| E2B | https://e2b.dev | Sandbox de código seguro en la nube |

---

*Última actualización: abril 2026. Los precios y disponibilidad de APIs pueden cambiar.*
