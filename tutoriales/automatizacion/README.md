# Bloque 17 — Automatización con IA

> **Nivel:** Avanzado · **Prerequisitos:** Bloques 3 (APIs de IA), 5 (Casos de uso), 9 (Agentes avanzados)

Automatizar procesos de negocio con LLMs: desde herramientas no-code como n8n, Make y Zapier hasta pipelines Python con colas asíncronas, procesamiento por lotes y conexiones con CRMs, suites ofimáticas y aplicaciones de colaboración empresarial.

---

## Tutoriales

| # | Tutorial | Descripción | Tiempo estimado |
|---|---|---|---|
| 01 | [n8n con IA](./01-n8n-ia.md) | Workflows visuales con nodos OpenAI/Anthropic, AI Agent node, casos prácticos | 50 min |
| 02 | [Make.com y Zapier con IA](./02-make-zapier-ia.md) | Escenarios Make, Zaps Zapier, pipelines no-code con LLMs, guía de decisión | 45 min |
| 03 | [Pipelines de negocio con LLMs](./03-pipelines-negocio.md) | LangChain LCEL, Batch API, Celery/Redis, pipeline de contratos end-to-end | 60 min |
| 04 | [Integración con herramientas de negocio](./04-integracion-herramientas-negocio.md) | CRM, Google Workspace, Microsoft 365, Slack bot, Notion + IA | 55 min |

---

## Descripción del bloque

La automatización con IA va más allá del prompt engineering puntual: consiste en encadenar LLMs con sistemas reales —bases de datos, CRMs, calendarios, bandejas de correo— de forma robusta, observable y económicamente eficiente.

El bloque avanza de lo más accesible (herramientas no-code) a lo más potente (código Python con colas y procesamiento por lotes):

1. **n8n** — plataforma open-source de automatización con nodos nativos de IA. Perfecta para equipos técnicos que quieren control total sin escribir servidores.
2. **Make y Zapier** — automatización orientada a operaciones: conectar SaaS con IA en minutos, sin código.
3. **Pipelines Python** — LCEL, Batch API de Anthropic/OpenAI, Celery + Redis para procesar cientos de documentos de forma asíncrona.
4. **Herramientas de negocio** — integrar IA directamente en el flujo de trabajo: Salesforce, Google Sheets, Slack, Notion.

---

## Requisitos

```bash
# Dependencias Python del bloque
pip install anthropic openai langchain langchain-anthropic langchain-openai \
            celery redis slack-bolt notion-client google-api-python-client \
            google-auth-httplib2 google-auth-oauthlib pydantic python-dotenv \
            httpx tenacity
```

Herramientas externas:
- **Docker** (para n8n self-hosted): [docs.docker.com](https://docs.docker.com)
- **n8n**: `docker pull n8nio/n8n`
- **Redis**: `docker pull redis:7`
- Cuentas en Make.com y/o Zapier (planes gratuitos suficientes para los ejemplos)

Variables de entorno recomendadas en `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
NOTION_TOKEN=secret_...
GOOGLE_SERVICE_ACCOUNT_FILE=credentials.json
```

---

## Orden de lectura

```
01-n8n-ia.md
    └── 02-make-zapier-ia.md
            └── 03-pipelines-negocio.md
                    └── 04-integracion-herramientas-negocio.md
```

Los tutoriales 01 y 02 son independientes entre sí (no-code vs no-code). El 03 y 04 son código Python puro y se apoyan en los conceptos del Bloque 9 (Agentes avanzados).

---

**Bloque anterior:** [16 — IA responsable](../ia-responsable/) · **Siguiente bloque:** en desarrollo
