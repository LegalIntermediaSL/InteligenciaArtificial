# JavaScript para IA

Ejemplos prácticos de integración con LLMs desde JavaScript/Node.js.

## Contenido

| # | Tutorial | Librería | Nivel |
|---|----------|----------|-------|
| 01 | [LangChain.js](./01-langchain-js.md) | `@langchain/anthropic` | Intermedio |
| 02 | [Vercel AI SDK](./02-vercel-ai-sdk.md) | `ai` + `@ai-sdk/anthropic` | Intermedio |

---

## Requisitos

- Node.js ≥ 18
- Variables de entorno en `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # solo para ejemplos OpenAI
```

## Instalación rápida

```bash
cd javascript/
npm install
```

---

## ¿LangChain.js o Vercel AI SDK?

| Criterio | LangChain.js | Vercel AI SDK |
|----------|-------------|---------------|
| **Ecosistema** | Chains, agentes, RAG completo | Streaming UI, Next.js first |
| **Curva de aprendizaje** | Media-alta | Baja |
| **Mejor para** | Pipelines complejos, backends | Apps React/Next.js |
| **Streaming** | Sí | Sí (primera clase) |
| **Multi-proveedor** | Sí | Sí |

> Para backends Node.js puros, también puedes usar el **SDK oficial de Anthropic** (`@anthropic-ai/sdk`) directamente.
