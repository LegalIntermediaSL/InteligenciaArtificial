# 04 — Pipelines de ML Automatizados

> **Bloque:** MLOps · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Orquestadores de ML: panorama](#1-orquestadores-de-ml-panorama)
2. [Prefect: conceptos fundamentales](#2-prefect-conceptos-fundamentales)
3. [Pipeline completo con Prefect](#3-pipeline-completo-con-prefect)
4. [Triggers automáticos](#4-triggers-automáticos)
5. [Manejo de fallos y reintentos](#5-manejo-de-fallos-y-reintentos)
6. [Integración con MLflow y Docker](#6-integración-con-mlflow-y-docker)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Orquestadores de ML: panorama

| Herramienta | Punto fuerte | Curva de aprendizaje | Mejor para |
|---|---|---|---|
| **Prefect** | Pythónico, UI moderna, fácil despliegue | Baja | Equipos pequeños/medianos, rapid iteration |
| **Airflow** | Ecosistema maduro, muchos operadores | Alta | Grandes organizaciones con pipelines complejos |
| **Kedro** | Estructura de proyecto opinada, reproducibilidad | Media | Proyectos de data science colaborativos |
| **ZenML** | MLOps-first, stack de herramientas pluggable | Media | Pipelines ML con múltiples artefactos |
| **Metaflow** | Netflix, integración nativa con AWS | Baja | Datos grandes, equipos en AWS |

**Recomendación para este tutorial:** Prefect 2 — la curva de aprendizaje mínima permite centrarse en la lógica del pipeline, no en la infraestructura.

---

## 2. Prefect: conceptos fundamentales

```bash
pip install prefect mlflow scikit-learn pandas
```

### Bloques de construcción

```python
# conceptos_prefect.py
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta


# TASK: unidad atómica de trabajo, con reintentos, caché y logging automáticos
@task(
    name="cargar-datos",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,    # Cachear resultado si los inputs no cambian
    cache_expiration=timedelta(hours=1),
    log_prints=True
)
def cargar_datos(ruta: str) -> "pd.DataFrame":
    import pandas as pd
    print(f"Cargando datos desde {ruta}")
    return pd.read_csv(ruta)


# FLOW: grafo de tasks con contexto, scheduling y observabilidad
@flow(
    name="pipeline-entrenamiento",
    description="Pipeline completo de entrenamiento del modelo de contratos"
)
def pipeline_entrenamiento(ruta_datos: str, experimento: str = "default"):
    df = cargar_datos(ruta_datos)
    # ... más tasks
    return df
```

---

## 3. Pipeline completo con Prefect

```python
# pipeline_ml.py
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import joblib


# ── TASKS ──────────────────────────────────────────────────────────────────

@task(name="ingest-datos", retries=2, retry_delay_seconds=60, log_prints=True)
def ingest(ruta_fuente: str, ruta_destino: str) -> str:
    """Descarga o copia datos crudos al directorio de trabajo."""
    import shutil
    Path(ruta_destino).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ruta_fuente, ruta_destino)
    print(f"Datos copiados: {ruta_fuente} → {ruta_destino}")
    return ruta_destino


@task(name="preprocesar", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=2))
def preprocesar(ruta_datos: str, features: list[str], target: str) -> tuple:
    """Limpia, escala y divide en train/test."""
    df = pd.read_csv(ruta_datos)

    # Eliminar nulos
    df = df.dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Guardar artefactos de preprocesamiento
    artefactos = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
        "n_train": len(X_train), "n_test": len(X_test)
    }
    joblib.dump(artefactos, "/tmp/preprocesado.pkl")
    return "/tmp/preprocesado.pkl", {"n_train": len(X_train), "n_test": len(X_test)}


@task(name="entrenar-modelo", log_prints=True)
def entrenar(artefactos_path: str, params: dict, experimento: str) -> str:
    """Entrena el modelo y registra en MLflow."""
    artefactos = joblib.load(artefactos_path)
    X_train, y_train = artefactos["X_train"], artefactos["y_train"]
    X_test, y_test = artefactos["X_test"], artefactos["y_test"]
    scaler = artefactos["scaler"]

    mlflow.set_experiment(experimento)

    with mlflow.start_run() as run:
        # Entrenar
        modelo = GradientBoostingClassifier(**params, random_state=42)
        modelo.fit(X_train, y_train)

        # Evaluar
        y_pred = modelo.predict(X_test)
        metricas = {
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }

        # Registrar en MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, "modelo")
        mlflow.sklearn.log_model(scaler, "scaler")

        print(f"Run ID: {run.info.run_id}")
        print(f"Métricas: {metricas}")

    return run.info.run_id


@task(name="evaluar-y-registrar", log_prints=True)
def evaluar_y_registrar(
    run_id: str,
    umbral_f1: float,
    nombre_modelo: str
) -> bool:
    """Promueve el modelo a Staging si supera el umbral de F1."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    f1 = run.data.metrics.get("f1", 0)

    print(f"F1 score: {f1:.4f} (umbral: {umbral_f1})")

    if f1 >= umbral_f1:
        # Registrar en Model Registry
        model_uri = f"runs:/{run_id}/modelo"
        mv = mlflow.register_model(model_uri, nombre_modelo)

        # Transicionar a Staging
        client.transition_model_version_stage(
            name=nombre_modelo,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False
        )
        print(f"✅ Modelo v{mv.version} promovido a Staging")
        return True
    else:
        print(f"❌ Modelo no promovido — F1 {f1:.4f} < umbral {umbral_f1}")
        return False


# ── FLOW PRINCIPAL ──────────────────────────────────────────────────────────

@flow(name="pipeline-ml-completo", log_prints=True)
def pipeline_ml(
    ruta_fuente: str = "data/contratos_raw.csv",
    features: list[str] = None,
    target: str = "categoria",
    params: dict = None,
    umbral_f1: float = 0.88,
    experimento: str = "ClasificadorContratos",
    nombre_modelo: str = "clasificador-contratos"
):
    if features is None:
        features = ["longitud_clausula", "n_partes", "tipo_contrato_encoded"]
    if params is None:
        params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}

    # 1. Ingest
    ruta_local = ingest(ruta_fuente, "/tmp/datos_raw.csv")

    # 2. Preprocesar
    artefactos_path, stats = preprocesar(ruta_local, features, target)
    print(f"Dataset: {stats}")

    # 3. Entrenar
    run_id = entrenar(artefactos_path, params, experimento)

    # 4. Evaluar y registrar
    promovido = evaluar_y_registrar(run_id, umbral_f1, nombre_modelo)

    return {"run_id": run_id, "promovido": promovido}


if __name__ == "__main__":
    resultado = pipeline_ml()
    print(json.dumps(resultado, indent=2))
```

---

## 4. Triggers automáticos

```python
# deployment.py — Desplegar el pipeline con schedule y triggers
from prefect import serve
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
from datetime import timedelta

# ── Despliegue con schedule periódico (todos los lunes a las 2am) ──
deployment_semanal = Deployment.build_from_flow(
    flow=pipeline_ml,
    name="entrenamiento-semanal",
    schedule=CronSchedule(cron="0 2 * * 1", timezone="Europe/Madrid"),
    parameters={
        "ruta_fuente": "gs://mi-bucket/datos/contratos_latest.csv",
        "umbral_f1": 0.88
    },
    tags=["produccion", "semanal"]
)

# ── Despliegue trigger-based (lanzado cuando se detecta drift) ──
deployment_drift = Deployment.build_from_flow(
    flow=pipeline_ml,
    name="reentrenamiento-por-drift",
    parameters={
        "ruta_fuente": "gs://mi-bucket/datos/contratos_latest.csv",
        "umbral_f1": 0.85  # umbral más permisivo en reentrenamiento urgente
    },
    tags=["drift", "triggered"]
)

# Aplicar deployments
deployment_semanal.apply()
deployment_drift.apply()
print("Deployments registrados en Prefect")
```

```python
# trigger_desde_monitor.py — Lanzar pipeline cuando se detecta drift
import httpx
import os


def lanzar_reentrenamiento(deployment_name: str = "reentrenamiento-por-drift"):
    """Llama a la API de Prefect para crear una flow run del deployment."""
    prefect_api_url = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")

    # Buscar el deployment por nombre
    r = httpx.get(f"{prefect_api_url}/deployments/name/pipeline-ml-completo/{deployment_name}")
    r.raise_for_status()
    deployment_id = r.json()["id"]

    # Crear flow run
    payload = {
        "name": "drift-triggered-run",
        "parameters": {"ruta_fuente": "data/contratos_latest.csv"}
    }
    r2 = httpx.post(
        f"{prefect_api_url}/deployments/{deployment_id}/create_flow_run",
        json=payload
    )
    r2.raise_for_status()
    flow_run = r2.json()
    print(f"Flow run creada: {flow_run['id']} — {flow_run['name']}")
    return flow_run["id"]
```

---

## 5. Manejo de fallos y reintentos

```python
# resilience.py
from prefect import flow, task
from prefect.tasks import task_input_hash
import random


@task(
    name="tarea-con-reintentos",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),  # 2s, 4s, 8s
    retry_jitter_factor=0.5,   # añade aleatoriedad para evitar thundering herd
    log_prints=True
)
def tarea_fragil(x: int) -> int:
    """Simula una tarea que falla aleatoriamente."""
    if random.random() < 0.5:
        raise ConnectionError("Servicio externo no disponible")
    return x * 2


@task(name="tarea-con-timeout", timeout_seconds=300)
def tarea_lenta() -> str:
    """Fallará con TimeoutError si tarda más de 5 minutos."""
    import time
    time.sleep(10)  # En test fallará con timeout_seconds=5
    return "completado"


@flow(name="pipeline-resiliente", log_prints=True)
def pipeline_resiliente():
    try:
        resultado = tarea_fragil(42)
        print(f"Resultado: {resultado}")
    except Exception as e:
        print(f"Tarea fallida definitivamente: {e}")
        # Aquí se puede: notificar, guardar estado parcial, lanzar pipeline alternativo
        raise

    return resultado


# Importar la función de backoff
from prefect.tasks import exponential_backoff
```

### Notificaciones de fallo

```python
# notifications.py
from prefect import flow
from prefect.blocks.notifications import SlackWebhook


async def notificar_fallo(flow, flow_run, state):
    """Hook que se ejecuta cuando el flow falla."""
    slack = await SlackWebhook.load("slack-mlops-alerts")
    await slack.notify(
        body=f"❌ Pipeline fallido: {flow.name}\nRun: {flow_run.name}\nError: {state.message}"
    )


@flow(
    name="pipeline-con-notificaciones",
    on_failure=[notificar_fallo],
    on_completion=[lambda f, fr, s: print(f"✅ {f.name} completado")]
)
def pipeline_con_notificaciones():
    pass
```

---

## 6. Integración con MLflow y Docker

### docker-compose completo

```yaml
# docker-compose.yml
version: "3.9"

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow-artifacts:/mlflow/artifacts
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      mlflow server
        --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow
        --default-artifact-root /mlflow/artifacts
        --host 0.0.0.0
        --port 5000

  prefect:
    image: prefecthq/prefect:2-latest
    ports:
      - "4200:4200"
    environment:
      - PREFECT_API_DATABASE_CONNECTION_URL=postgresql+asyncpg://prefect:prefect@postgres/prefect
    depends_on:
      postgres:
        condition: service_healthy
    command: prefect server start --host 0.0.0.0

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - PREFECT_API_URL=http://prefect:4200/api
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - prefect
      - mlflow
    command: prefect worker start --pool ml-pool

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  mlflow-artifacts:
  postgres-data:
```

```sql
-- init-db.sql
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
CREATE USER prefect WITH PASSWORD 'prefect';
CREATE DATABASE prefect OWNER prefect;
```

```dockerfile
# Dockerfile.worker
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

```bash
# Lanzar el stack completo
docker-compose up -d

# Registrar el work pool
docker-compose exec worker prefect work-pool create ml-pool --type process

# Desplegar los flows
docker-compose exec worker python deployment.py

# Ver UI
open http://localhost:4200  # Prefect
open http://localhost:5000  # MLflow
```

---

## 7. Extensiones sugeridas

- **Kedro + Prefect**: usar Kedro para estructura del proyecto y Prefect como orquestador
- **Great Expectations**: añadir validación de datos como task antes del entrenamiento
- **DVC**: versionado de datos y pipelines reproducibles integrado con Git
- **Terraform**: provisionar la infraestructura (postgres, S3/GCS) como código

---

**Anterior:** [03 — Detección de drift](./03-deteccion-drift.md) · **Siguiente bloque:** [Bloque 17 — Automatización](../automatizacion/)
