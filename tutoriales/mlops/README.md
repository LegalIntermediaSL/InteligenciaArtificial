# MLOps para LLMs

Tutoriales prácticos de MLOps aplicados a modelos de lenguaje: registro y versionado de modelos, experimentación controlada con A/B testing, detección de drift en producción y pipelines de ML automatizados de extremo a extremo. Cada tutorial incluye código Python completo y funcional.

## Contenido

| # | Tutorial | Descripción |
|---|---|---|
| 01 | [Registro y versionado de modelos](./01-registro-modelos.md) | MLflow Model Registry, HuggingFace Hub y W&B Artifacts para gestionar el ciclo de vida de modelos |
| 02 | [A/B testing de LLMs](./02-ab-testing-llms.md) | Diseño de experimentos, shadow deployment, router A/B con FastAPI y análisis estadístico |
| 03 | [Detección de drift en producción](./03-deteccion-drift.md) | PSI, KS test, Jensen-Shannon, Evidently y alertas automáticas de drift |
| 04 | [Pipelines de ML automatizados](./04-pipelines-automatizados.md) | Prefect flows completos con ingest → train → evaluate → register → deploy y triggers automáticos |

## Requisitos del bloque

```bash
pip install mlflow prefect evidently scipy fastapi uvicorn redis \
            huggingface-hub wandb pandas scikit-learn httpx python-dotenv
```

Cada tutorial especifica sus propias dependencias adicionales al inicio.

## Orden recomendado

01 → 02 → 03 → 04. El registro de modelos sienta las bases del resto: el pipeline del tutorial 04 referencia el registry del 01, y las métricas de drift del 03 actúan como trigger de reentrenamiento.

---

> Bloque anterior: [Producción con LLMs](../produccion/)
