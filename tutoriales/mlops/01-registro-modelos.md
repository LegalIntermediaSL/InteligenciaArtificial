# 01 — Registro y versionado de modelos

> **Bloque:** MLOps · **Nivel:** Práctico · **Tiempo estimado:** 45 min

---

## Índice

1. [Por qué necesitas un model registry](#1-por-qué-necesitas-un-model-registry)
2. [MLflow: tracking de experimentos](#2-mlflow-tracking-de-experimentos)
3. [Model Registry: etapas y transiciones](#3-model-registry-etapas-y-transiciones)
4. [Comparativa de runs y selección del mejor modelo](#4-comparativa-de-runs-y-selección-del-mejor-modelo)
5. [HuggingFace Hub como registry](#5-huggingface-hub-como-registry)
6. [Weights & Biases Artifacts para LLMs](#6-weights--biases-artifacts-para-llms)
7. [Automatización de promoción con Python API](#7-automatización-de-promoción-con-python-api)

---

## 1. Por qué necesitas un model registry

Sin un sistema de registro, el versionado de modelos se convierte en caos: carpetas con nombres como `modelo_final_v3_bueno_ESTE.pt`, pérdida de trazabilidad entre código y artefacto, imposibilidad de hacer rollback y ausencia de auditoría.

```
Sin registry                         Con registry
──────────────────────────────────   ─────────────────────────────────────
modelo_v1.pkl                        models:/mi-modelo/1  →  Archived
modelo_v2_mejorado.pkl               models:/mi-modelo/2  →  Staging
modelo_final.pkl                     models:/mi-modelo/3  →  Production
modelo_final_REAL.pkl                            ▲
modelo_final_REAL_2.pkl              metadatos: métricas, commit git,
                                     dataset hash, parámetros, autor
```

**Los tres problemas que resuelve un registry:**

- **Reproducibilidad**: cada versión lleva asociados los hiperparámetros, el hash del dataset y el commit de git que la generó.
- **Gobernanza**: transiciones de estado explícitas (Staging → Production) que requieren aprobación o criterios automatizados.
- **Rollback**: volver a una versión anterior es cambiar un alias, no desplegar código nuevo.

---

## 2. MLflow: tracking de experimentos

MLflow es la herramienta de facto para tracking de experimentos en Python. Se instala localmente y funciona sin infraestructura adicional.

```bash
pip install mlflow scikit-learn pandas numpy python-dotenv
```

```python
# mlops/mlflow_tracking.py
"""
Entrena varios modelos con distintos hiperparámetros y registra
cada run en MLflow con métricas, parámetros y artefactos.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Configuración de MLflow
# ---------------------------------------------------------------------------

# Para MLflow local, los experimentos se guardan en ./mlruns por defecto.
# Para un servidor remoto: mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # BD local para persistencia
mlflow.set_experiment("clasificacion-clientes")  # Crea el experimento si no existe


# ---------------------------------------------------------------------------
# Datos de ejemplo
# ---------------------------------------------------------------------------

def generar_datos(n_samples: int = 2000, seed: int = 42):
    """Genera un dataset sintético de clasificación binaria."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        random_state=seed,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return train_test_split(X, y, test_size=0.2, random_state=seed), feature_names


# ---------------------------------------------------------------------------
# Función de entrenamiento con logging completo
# ---------------------------------------------------------------------------

def entrenar_y_registrar(
    modelo,
    nombre_modelo: str,
    params: dict,
    X_train, X_test, y_train, y_test,
    tags: dict | None = None,
) -> str:
    """
    Entrena un modelo y registra todo en MLflow.
    Devuelve el run_id para referencia posterior.
    """
    with mlflow.start_run(run_name=nombre_modelo) as run:
        # --- 1. Registrar parámetros del modelo ---
        mlflow.log_params(params)

        # --- 2. Registrar metadatos del experimento como tags ---
        mlflow.set_tags({
            "modelo_clase": type(modelo).__name__,
            "dataset_size": len(X_train) + len(X_test),
            "sklearn_version": "1.3",
            **(tags or {}),
        })

        # --- 3. Entrenar ---
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        # --- 4. Calcular y registrar métricas ---
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metricas)

        # --- 5. Registrar métricas por epoch (simulando entrenamiento iterativo) ---
        # Para modelos que emiten métricas por iteración (GBM, redes, etc.)
        if hasattr(modelo, "train_score_"):
            for step, score in enumerate(modelo.train_score_):
                mlflow.log_metric("train_deviance", score, step=step)

        # --- 6. Guardar artefactos adicionales ---
        # Importancias de características
        if hasattr(modelo, "feature_importances_"):
            importancias = pd.DataFrame({
                "feature": [f"feature_{i}" for i in range(len(modelo.feature_importances_))],
                "importance": modelo.feature_importances_,
            }).sort_values("importance", ascending=False)

            ruta_csv = Path("feature_importances.csv")
            importancias.to_csv(ruta_csv, index=False)
            mlflow.log_artifact(str(ruta_csv), artifact_path="analysis")
            ruta_csv.unlink()  # limpiar temporal

        # Guardar configuración del experimento como JSON
        config = {"params": params, "metricas": metricas}
        ruta_config = Path("config.json")
        ruta_config.write_text(json.dumps(config, indent=2))
        mlflow.log_artifact(str(ruta_config), artifact_path="config")
        ruta_config.unlink()

        # --- 7. Registrar el modelo con firma automática ---
        # La firma captura el esquema de entrada/salida para validación
        signature = mlflow.models.infer_signature(X_train, modelo.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"clasificacion-{nombre_modelo.lower().replace(' ', '-')}",
        )

        run_id = run.info.run_id
        print(f"Run '{nombre_modelo}' completado: {run_id[:8]}... | "
              f"AUC={metricas['roc_auc']:.4f} | F1={metricas['f1_score']:.4f}")
        return run_id


# ---------------------------------------------------------------------------
# Ejecutar múltiples experimentos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), feature_names = generar_datos()

    experimentos = [
        {
            "modelo": LogisticRegression(C=0.1, max_iter=1000),
            "nombre": "LogReg-C0.1",
            "params": {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
            "tags": {"tipo": "lineal"},
        },
        {
            "modelo": LogisticRegression(C=1.0, max_iter=1000),
            "nombre": "LogReg-C1.0",
            "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
            "tags": {"tipo": "lineal"},
        },
        {
            "modelo": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
            "nombre": "GBM-100est",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            "tags": {"tipo": "ensemble"},
        },
        {
            "modelo": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4),
            "nombre": "GBM-200est",
            "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
            "tags": {"tipo": "ensemble"},
        },
    ]

    run_ids = []
    for exp in experimentos:
        rid = entrenar_y_registrar(
            modelo=exp["modelo"],
            nombre_modelo=exp["nombre"],
            params=exp["params"],
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            tags=exp["tags"],
        )
        run_ids.append(rid)

    print(f"\n{len(run_ids)} runs registrados.")
    print("Abre la UI con: mlflow ui --backend-store-uri sqlite:///mlflow.db")
```

---

## 3. Model Registry: etapas y transiciones

El Model Registry de MLflow gestiona el ciclo de vida de los modelos con tres etapas: **Staging** (candidato a producción), **Production** (versión activa) y **Archived** (retirado).

```
Flujo de un modelo:
None → Staging → Production → Archived
         ↑           ↑
    (validación) (aprobación)
```

```python
# mlops/model_registry.py
"""
Gestión del ciclo de vida de modelos en MLflow Model Registry.
"""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

NOMBRE_MODELO = "clasificacion-gbm-200est"


def listar_versiones(nombre_modelo: str) -> None:
    """Muestra todas las versiones registradas de un modelo con sus metadatos."""
    print(f"\n=== Versiones de '{nombre_modelo}' ===")
    try:
        versiones = client.search_model_versions(f"name='{nombre_modelo}'")
    except Exception:
        print("  Modelo no encontrado. Ejecuta primero mlflow_tracking.py")
        return

    for v in sorted(versiones, key=lambda x: int(x.version)):
        print(f"  v{v.version} | etapa: {v.current_stage:12} | "
              f"run: {v.run_id[:8]}... | creado: {v.creation_timestamp}")


def promover_a_staging(nombre_modelo: str, version: int) -> None:
    """Mueve una versión específica a Staging."""
    client.transition_model_version_stage(
        name=nombre_modelo,
        version=str(version),
        stage="Staging",
        archive_existing_versions=False,  # Permitir múltiples en Staging
    )
    # Añadir descripción a la transición
    client.update_model_version(
        name=nombre_modelo,
        version=str(version),
        description="Candidato a producción. Pendiente de validación en entorno de staging.",
    )
    print(f"  v{version} promovido a Staging")


def promover_a_production(nombre_modelo: str, version: int) -> None:
    """
    Mueve una versión a Production y archiva automáticamente
    la versión anterior que estaba en Production.
    """
    client.transition_model_version_stage(
        name=nombre_modelo,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,  # Archivar la versión anterior
    )
    client.update_model_version(
        name=nombre_modelo,
        version=str(version),
        description="Versión en producción. Validada en staging con AUC > 0.90.",
    )
    print(f"  v{version} promovido a Production (versiones anteriores archivadas)")


def archivar_version(nombre_modelo: str, version: int, razon: str = "") -> None:
    """Retira una versión moviéndola a Archived."""
    client.transition_model_version_stage(
        name=nombre_modelo,
        version=str(version),
        stage="Archived",
    )
    if razon:
        client.update_model_version(
            name=nombre_modelo,
            version=str(version),
            description=f"Archivado: {razon}",
        )
    print(f"  v{version} archivado. Razón: {razon}")


def cargar_modelo_de_produccion(nombre_modelo: str):
    """
    Carga la versión actualmente en Production sin conocer el número de versión.
    Esto es lo que usan los servicios de inferencia en producción.
    """
    uri = f"models:/{nombre_modelo}/Production"
    modelo = mlflow.sklearn.load_model(uri)
    print(f"  Modelo cargado desde: {uri}")
    return modelo


def añadir_etiquetas_al_modelo(nombre_modelo: str, version: int, etiquetas: dict) -> None:
    """Añade tags clave-valor a una versión de modelo para búsquedas y auditoría."""
    for key, value in etiquetas.items():
        client.set_model_version_tag(
            name=nombre_modelo,
            version=str(version),
            key=key,
            value=str(value),
        )
    print(f"  Tags añadidos a v{version}: {etiquetas}")


# ---------------------------------------------------------------------------
# Demo del flujo completo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    listar_versiones(NOMBRE_MODELO)

    # Típicamente tienes varias versiones tras varios runs de entrenamiento.
    # Simulamos el flujo: la v1 pasa a Staging, luego a Production.
    print("\n--- Promoviendo versiones ---")
    promover_a_staging(NOMBRE_MODELO, version=1)
    añadir_etiquetas_al_modelo(
        NOMBRE_MODELO,
        version=1,
        etiquetas={
            "validado_por": "equipo-ml",
            "dataset_version": "2024-Q1",
            "auc_staging": "0.923",
        }
    )
    promover_a_production(NOMBRE_MODELO, version=1)

    listar_versiones(NOMBRE_MODELO)

    # Simular rollback: archivar la versión en producción
    print("\n--- Simulando rollback ---")
    archivar_version(NOMBRE_MODELO, version=1, razon="Degradación de métricas detectada")
    listar_versiones(NOMBRE_MODELO)
```

---

## 4. Comparativa de runs y selección del mejor modelo

```python
# mlops/seleccion_mejor_modelo.py
"""
Consulta la API de MLflow para comparar todos los runs de un experimento
y seleccionar automáticamente el mejor modelo según una métrica.
"""
import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


def comparar_runs(
    nombre_experimento: str,
    metrica_objetivo: str = "roc_auc",
    maximizar: bool = True,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Compara todos los runs de un experimento y devuelve un DataFrame ordenado.

    Args:
        nombre_experimento: Nombre del experimento en MLflow.
        metrica_objetivo: Métrica por la que ordenar los runs.
        maximizar: True si la métrica es mejor cuanto mayor (AUC, F1, Accuracy).
                   False si es mejor cuanto menor (RMSE, MAE, Loss).
        top_n: Número de runs a mostrar.

    Returns:
        DataFrame con los mejores runs y sus métricas/parámetros.
    """
    experimento = client.get_experiment_by_name(nombre_experimento)
    if experimento is None:
        raise ValueError(f"Experimento '{nombre_experimento}' no encontrado")

    runs = client.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="status = 'FINISHED'",  # Solo runs completados
        order_by=[f"metrics.{metrica_objetivo} {'DESC' if maximizar else 'ASC'}"],
        max_results=top_n,
    )

    if not runs:
        print("No se encontraron runs finalizados.")
        return pd.DataFrame()

    registros = []
    for run in runs:
        registro = {
            "run_id": run.info.run_id[:8],
            "run_name": run.info.run_name,
            **{f"param_{k}": v for k, v in run.data.params.items()},
            **run.data.metrics,
            "duracion_s": (run.info.end_time - run.info.start_time) / 1000
            if run.info.end_time
            else None,
        }
        registros.append(registro)

    df = pd.DataFrame(registros)
    return df


def seleccionar_mejor_run(
    nombre_experimento: str,
    metrica_objetivo: str = "roc_auc",
    maximizar: bool = True,
) -> tuple[str, float]:
    """
    Devuelve el run_id completo y el valor de la métrica del mejor run.
    """
    experimento = client.get_experiment_by_name(nombre_experimento)
    if experimento is None:
        raise ValueError(f"Experimento '{nombre_experimento}' no encontrado")

    runs = client.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metrica_objetivo} {'DESC' if maximizar else 'ASC'}"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No hay runs completados en el experimento.")

    mejor_run = runs[0]
    valor = mejor_run.data.metrics.get(metrica_objetivo, float("nan"))
    return mejor_run.info.run_id, valor


def registrar_mejor_modelo(
    nombre_experimento: str,
    nombre_modelo_registro: str,
    metrica_objetivo: str = "roc_auc",
    umbral_minimo: float = 0.85,
) -> str | None:
    """
    Selecciona el mejor run y lo registra en el Model Registry solo si
    supera el umbral mínimo de calidad.

    Returns:
        Nombre del modelo registrado o None si no superó el umbral.
    """
    run_id, valor_metrica = seleccionar_mejor_run(nombre_experimento, metrica_objetivo)

    print(f"Mejor run: {run_id[:8]}... | {metrica_objetivo}={valor_metrica:.4f}")

    if valor_metrica < umbral_minimo:
        print(f"RECHAZADO: {valor_metrica:.4f} < umbral mínimo {umbral_minimo}")
        return None

    # Registrar en el Model Registry
    uri_modelo = f"runs:/{run_id}/model"
    resultado = mlflow.register_model(
        model_uri=uri_modelo,
        name=nombre_modelo_registro,
    )
    print(f"REGISTRADO: '{nombre_modelo_registro}' v{resultado.version}")
    return nombre_modelo_registro


# ---------------------------------------------------------------------------
# Ejemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXPERIMENTO = "clasificacion-clientes"

    print("=== TOP 5 RUNS ===")
    df = comparar_runs(EXPERIMENTO, metrica_objetivo="roc_auc", top_n=5)
    if not df.empty:
        cols_mostrar = [c for c in df.columns if not c.startswith("param_")]
        print(df[cols_mostrar].to_string(index=False))

    print("\n=== REGISTRAR MEJOR MODELO ===")
    registrar_mejor_modelo(
        nombre_experimento=EXPERIMENTO,
        nombre_modelo_registro="mejor-clasificador-produccion",
        metrica_objetivo="roc_auc",
        umbral_minimo=0.85,
    )
```

---

## 5. HuggingFace Hub como registry

HuggingFace Hub es el registry estándar para modelos de lenguaje. Permite versionar modelos fine-tuned, compartirlos con el equipo y hacer `model_id` pointing al commit exacto.

```bash
pip install huggingface-hub transformers
```

```python
# mlops/hf_hub_registry.py
"""
Uso de HuggingFace Hub como model registry para LLMs fine-tuned.
Cubre: subida, descarga, tags de versión y carga por commit hash.
"""
import os
from pathlib import Path
from huggingface_hub import (
    HfApi,
    Repository,
    hf_hub_download,
    snapshot_download,
    ModelCard,
    ModelCardData,
)

# Inicializar cliente (requiere HF_TOKEN en el entorno)
api = HfApi(token=os.getenv("HF_TOKEN"))

REPO_ID = "mi-organizacion/clasificador-reviews"  # Sustituye por tu repo


# ---------------------------------------------------------------------------
# 1. Crear y configurar un repositorio de modelo
# ---------------------------------------------------------------------------

def crear_repo_modelo(repo_id: str, privado: bool = True) -> str:
    """
    Crea un repositorio en HF Hub para un modelo.
    Devuelve la URL del repositorio.
    """
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=privado,
        exist_ok=True,  # No falla si ya existe
    )
    print(f"Repositorio listo: {url}")
    return url


# ---------------------------------------------------------------------------
# 2. Subir artefactos de modelo
# ---------------------------------------------------------------------------

def subir_modelo(
    repo_id: str,
    directorio_local: str,
    mensaje_commit: str = "Actualizar modelo",
    rama: str = "main",
) -> str:
    """
    Sube todos los archivos de un directorio al repositorio.
    Devuelve el commit hash para referencia futura.
    """
    # Subir carpeta completa de una vez
    commit_info = api.upload_folder(
        folder_path=directorio_local,
        repo_id=repo_id,
        repo_type="model",
        commit_message=mensaje_commit,
    )
    commit_hash = commit_info.oid
    print(f"Subido a {repo_id}@{commit_hash[:8]}")
    return commit_hash


def subir_archivo_individual(
    repo_id: str,
    ruta_local: str,
    ruta_en_repo: str,
    mensaje_commit: str = "Actualizar archivo",
) -> str:
    """Sube un solo archivo (útil para actualizar métricas o configs)."""
    commit_info = api.upload_file(
        path_or_fileobj=ruta_local,
        path_in_repo=ruta_en_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=mensaje_commit,
    )
    return commit_info.oid


# ---------------------------------------------------------------------------
# 3. Gestión de versiones con tags
# ---------------------------------------------------------------------------

def etiquetar_version(repo_id: str, commit_hash: str, tag: str) -> None:
    """
    Crea un tag de versión semántica en un commit específico.
    Convención: v1.0.0, v1.1.0, etc.
    """
    api.create_tag(
        repo_id=repo_id,
        tag=tag,
        target_commitish=commit_hash,
        repo_type="model",
    )
    print(f"Tag '{tag}' creado en commit {commit_hash[:8]}")


def listar_versiones(repo_id: str) -> list[dict]:
    """Lista todos los commits del repositorio como historial de versiones."""
    commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="model"))
    versiones = []
    for c in commits:
        versiones.append({
            "commit": c.commit_id[:8],
            "mensaje": c.title,
            "fecha": c.created_at,
        })
        print(f"  {c.commit_id[:8]} | {c.created_at:%Y-%m-%d} | {c.title}")
    return versiones


# ---------------------------------------------------------------------------
# 4. Cargar modelo por versión exacta
# ---------------------------------------------------------------------------

def cargar_por_revision(repo_id: str, revision: str, directorio_local: str = "./modelo") -> str:
    """
    Descarga el snapshot del modelo en una revisión específica.
    'revision' puede ser un commit hash, un tag o una rama.

    Ejemplos:
        cargar_por_revision(REPO_ID, "v1.0.0")
        cargar_por_revision(REPO_ID, "abc12345")
        cargar_por_revision(REPO_ID, "main")
    """
    ruta = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=directorio_local,
    )
    print(f"Modelo descargado en: {ruta}")
    return ruta


# ---------------------------------------------------------------------------
# 5. Model Card — documentación estandarizada
# ---------------------------------------------------------------------------

def crear_model_card(
    repo_id: str,
    descripcion: str,
    metricas: dict,
    lenguajes: list[str] = ["es"],
) -> None:
    """
    Crea y sube una Model Card con metadatos estructurados.
    Las Model Cards son indexadas por HF Hub para búsqueda y comparación.
    """
    card_data = ModelCardData(
        language=lenguajes,
        license="apache-2.0",
        model_name=repo_id.split("/")[-1],
        eval_results=[
            {
                "task": {"type": "text-classification"},
                "dataset": {"name": "reviews-es", "type": "custom"},
                "metrics": [
                    {"type": nombre, "value": valor}
                    for nombre, valor in metricas.items()
                ],
            }
        ],
    )

    contenido = f"""---
{card_data.to_yaml()}
---

# {repo_id.split('/')[-1]}

{descripcion}

## Métricas de evaluación

| Métrica | Valor |
|---------|-------|
""" + "\n".join(f"| {k} | {v:.4f} |" for k, v in metricas.items())

    card = ModelCard(contenido)
    card.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))
    print(f"Model Card subida a {repo_id}")


# ---------------------------------------------------------------------------
# Ejemplo de flujo completo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Este ejemplo asume que tienes HF_TOKEN configurado y un repo válido.
    # Para probarlo en local sin subir nada real, usa repo_id con tu usuario.

    # Paso 1: Crear el repo
    # crear_repo_modelo(REPO_ID, privado=True)

    # Paso 2: Crear directorio de modelo de ejemplo
    directorio = Path("./modelo_ejemplo")
    directorio.mkdir(exist_ok=True)
    (directorio / "config.json").write_text('{"model_type": "bert", "version": "1.0"}')
    (directorio / "metricas.json").write_text('{"accuracy": 0.923, "f1": 0.918}')

    print("Directorio de modelo creado en ./modelo_ejemplo")
    print("Para subir al Hub: descomenta las llamadas en el __main__")
    print(f"  subir_modelo('{REPO_ID}', './modelo_ejemplo', 'v1.0.0 - modelo inicial')")
    print(f"  etiquetar_version('{REPO_ID}', <commit_hash>, 'v1.0.0')")
```

---

## 6. Weights & Biases Artifacts para LLMs

W&B Artifacts es especialmente útil para LLMs: permite versionar no solo los pesos del modelo sino también los datasets de fine-tuning, los prompts y los resultados de evaluación.

```bash
pip install wandb
```

```python
# mlops/wandb_artifacts.py
"""
Uso de W&B Artifacts para versionar modelos LLM, datasets y evaluaciones.
Requiere: wandb login (o WANDB_API_KEY en el entorno)
"""
import wandb
import json
import tempfile
from pathlib import Path
import numpy as np

# Configuración del proyecto
PROYECTO = "llm-produccion"
ENTIDAD = "mi-equipo"  # Tu usuario o equipo en W&B


# ---------------------------------------------------------------------------
# 1. Registrar un modelo como Artifact
# ---------------------------------------------------------------------------

def registrar_modelo_artifact(
    nombre_modelo: str,
    directorio_pesos: str,
    metricas: dict,
    hiperparametros: dict,
    descripcion: str = "",
) -> str:
    """
    Registra un modelo LLM como W&B Artifact con sus metadatos.
    Devuelve la versión del artifact creado.
    """
    with wandb.init(
        project=PROYECTO,
        entity=ENTIDAD,
        job_type="registro-modelo",
        config=hiperparametros,
    ) as run:
        # Crear el artifact de tipo 'model'
        artifact = wandb.Artifact(
            name=nombre_modelo,
            type="model",
            description=descripcion,
            metadata={
                "metricas": metricas,
                "hiperparametros": hiperparametros,
                "framework": "transformers",
            },
        )

        # Añadir todos los archivos del directorio de pesos
        artifact.add_dir(directorio_pesos, name="weights")

        # Registrar métricas del run
        run.log(metricas)

        # Subir el artifact
        run.log_artifact(artifact, aliases=["latest"])

        version = artifact.version
        print(f"Modelo registrado: {nombre_modelo}:{version}")
        return version


# ---------------------------------------------------------------------------
# 2. Versionar el dataset de fine-tuning
# ---------------------------------------------------------------------------

def versionar_dataset(
    nombre: str,
    datos: list[dict],
    descripcion: str = "",
    split: str = "train",
) -> None:
    """
    Guarda un dataset de fine-tuning como W&B Artifact.
    Permite vincular el dataset exacto con cada run de entrenamiento.
    """
    with wandb.init(
        project=PROYECTO,
        entity=ENTIDAD,
        job_type="versionar-dataset",
    ) as run:
        artifact = wandb.Artifact(
            name=nombre,
            type="dataset",
            description=descripcion,
            metadata={
                "num_ejemplos": len(datos),
                "split": split,
                "campos": list(datos[0].keys()) if datos else [],
            },
        )

        # Guardar dataset como JSONL
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for ejemplo in datos:
                f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
            ruta_temporal = f.name

        artifact.add_file(ruta_temporal, name=f"{split}.jsonl")
        run.log_artifact(artifact)
        Path(ruta_temporal).unlink()
        print(f"Dataset '{nombre}' versionado con {len(datos)} ejemplos")


# ---------------------------------------------------------------------------
# 3. Linaje: vincular dataset → modelo (artifact lineage)
# ---------------------------------------------------------------------------

def entrenar_con_linaje(
    nombre_modelo: str,
    nombre_dataset: str,
    version_dataset: str = "latest",
) -> None:
    """
    Entrena un modelo descargando el dataset de W&B, creando linaje
    automático entre dataset y modelo.
    """
    with wandb.init(
        project=PROYECTO,
        entity=ENTIDAD,
        job_type="entrenamiento",
    ) as run:
        # Descargar y usar el artifact del dataset (crea el linaje)
        artifact_dataset = run.use_artifact(
            f"{nombre_dataset}:{version_dataset}",
            type="dataset",
        )
        directorio_dataset = artifact_dataset.download()
        print(f"Dataset descargado en: {directorio_dataset}")

        # --- Aquí iría el entrenamiento real del LLM ---
        # trainer = Trainer(model=model, train_dataset=dataset, ...)
        # trainer.train()
        metricas_simuladas = {
            "eval_loss": 0.234,
            "eval_accuracy": 0.921,
            "perplexity": 12.4,
        }
        run.log(metricas_simuladas)

        # Registrar el modelo entrenado
        artifact_modelo = wandb.Artifact(
            name=nombre_modelo,
            type="model",
            metadata={
                "metricas": metricas_simuladas,
                "dataset_fuente": f"{nombre_dataset}:{version_dataset}",
            },
        )

        # En producción aquí añadirías los pesos reales:
        # artifact_modelo.add_dir("./output/model")

        # Crear archivo de metadatos de ejemplo
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(metricas_simuladas, f, indent=2)
            ruta_meta = f.name
        artifact_modelo.add_file(ruta_meta, name="metricas.json")
        Path(ruta_meta).unlink()

        run.log_artifact(artifact_modelo)
        print(f"Modelo '{nombre_modelo}' registrado con linaje al dataset")


# ---------------------------------------------------------------------------
# 4. Comparar versiones de modelos
# ---------------------------------------------------------------------------

def comparar_versiones_modelo(nombre_modelo: str, versiones: list[str]) -> None:
    """
    Descarga y compara los metadatos de distintas versiones de un modelo.
    """
    api = wandb.Api()

    print(f"\n=== Comparativa de versiones: {nombre_modelo} ===")
    for version in versiones:
        try:
            artifact = api.artifact(
                f"{ENTIDAD}/{PROYECTO}/{nombre_modelo}:{version}"
            )
            metricas = artifact.metadata.get("metricas", {})
            print(f"  {version:10} | {artifact.created_at:%Y-%m-%d} | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in metricas.items()))
        except Exception as e:
            print(f"  {version:10} | Error: {e}")


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Datos de fine-tuning de ejemplo
    datos_ejemplo = [
        {"instruccion": "Resume el siguiente texto:", "entrada": "...", "salida": "..."},
        {"instruccion": "Clasifica el sentimiento:", "entrada": "Me encanta", "salida": "positivo"},
    ]

    print("Ejemplo de W&B Artifacts para MLOps de LLMs")
    print("Requiere: wandb login o WANDB_API_KEY configurada\n")
    print("Llamadas disponibles:")
    print("  versionar_dataset('dataset-reviews-v1', datos_ejemplo, split='train')")
    print("  registrar_modelo_artifact('llm-clasificador', './weights/', metricas={...}, hiperparametros={...})")
    print("  entrenar_con_linaje('llm-v2', 'dataset-reviews-v1')")
    print("  comparar_versiones_modelo('llm-clasificador', ['v1', 'v2', 'latest'])")
```

---

## 7. Automatización de promoción con Python API

Combina todo lo anterior: un script que evalúa automáticamente si un nuevo modelo supera al de producción y lo promociona sin intervención manual.

```python
# mlops/auto_promocion.py
"""
Pipeline de promoción automática: evalúa el mejor modelo de Staging
contra el de Production y lo promueve si supera los criterios.
"""
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from dataclasses import dataclass
from typing import Optional

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()


@dataclass
class CriteriosPromocion:
    """Define los criterios que debe cumplir un modelo para ser promovido."""
    metrica_principal: str = "roc_auc"
    mejora_minima_relativa: float = 0.005   # 0.5% de mejora mínima
    umbral_absoluto_minimo: float = 0.85    # No promover si AUC < 0.85
    comparar_con: str = "Production"        # Etapa contra la que comparar


def obtener_metricas_version(nombre_modelo: str, etapa: str) -> Optional[dict]:
    """
    Carga el modelo en una etapa y evalúa sus métricas en el dataset de validación.
    Devuelve None si no hay modelo en esa etapa.
    """
    versiones = client.get_latest_versions(nombre_modelo, stages=[etapa])
    if not versiones:
        return None

    version = versiones[0]
    uri = f"models:/{nombre_modelo}/{etapa}"

    try:
        modelo = mlflow.sklearn.load_model(uri)
    except Exception as e:
        print(f"  Error cargando {uri}: {e}")
        return None

    # Generar datos de validación (en producción usarías tu dataset real)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=99)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.5, random_state=99)

    y_pred = modelo.predict(X_val)
    y_proba = modelo.predict_proba(X_val)[:, 1]

    return {
        "version": version.version,
        "run_id": version.run_id,
        "roc_auc": roc_auc_score(y_val, y_proba),
        "f1_score": f1_score(y_val, y_pred),
    }


def decidir_promocion(
    nombre_modelo: str,
    criterios: CriteriosPromocion,
) -> tuple[bool, str]:
    """
    Evalúa si el modelo en Staging debe ser promovido a Production.

    Returns:
        (promover: bool, razon: str)
    """
    print(f"\n=== Evaluando promoción de '{nombre_modelo}' ===")

    # Obtener métricas del candidato en Staging
    metricas_staging = obtener_metricas_version(nombre_modelo, "Staging")
    if metricas_staging is None:
        return False, "No hay modelo en Staging"

    metrica = criterios.metrica_principal
    valor_staging = metricas_staging[metrica]
    print(f"  Staging  v{metricas_staging['version']}: {metrica}={valor_staging:.4f}")

    # Verificar umbral mínimo absoluto
    if valor_staging < criterios.umbral_absoluto_minimo:
        razon = (f"Staging ({valor_staging:.4f}) no supera el umbral "
                 f"mínimo de {criterios.umbral_absoluto_minimo}")
        print(f"  RECHAZADO: {razon}")
        return False, razon

    # Obtener métricas del modelo en Production
    metricas_prod = obtener_metricas_version(nombre_modelo, criterios.comparar_con)
    if metricas_prod is None:
        # No hay modelo en producción → promover directamente
        razon = "No hay modelo en Production. Promoviendo sin comparación."
        print(f"  APROBADO: {razon}")
        return True, razon

    valor_prod = metricas_prod[metrica]
    print(f"  {criterios.comparar_con} v{metricas_prod['version']}: {metrica}={valor_prod:.4f}")

    # Calcular mejora relativa
    mejora = (valor_staging - valor_prod) / (valor_prod + 1e-10)
    print(f"  Mejora relativa: {mejora*100:+.2f}% (mínima: {criterios.mejora_minima_relativa*100:.1f}%)")

    if mejora >= criterios.mejora_minima_relativa:
        razon = (f"Staging ({valor_staging:.4f}) supera a Production ({valor_prod:.4f}) "
                 f"por {mejora*100:.2f}%")
        print(f"  APROBADO: {razon}")
        return True, razon
    else:
        razon = (f"Staging ({valor_staging:.4f}) no supera a Production ({valor_prod:.4f}) "
                 f"en el mínimo requerido ({criterios.mejora_minima_relativa*100:.1f}%)")
        print(f"  RECHAZADO: {razon}")
        return False, razon


def ejecutar_pipeline_promocion(
    nombre_modelo: str,
    criterios: CriteriosPromocion | None = None,
) -> bool:
    """
    Pipeline completo de promoción automática.
    Retorna True si el modelo fue promovido.
    """
    if criterios is None:
        criterios = CriteriosPromocion()

    promover, razon = decidir_promocion(nombre_modelo, criterios)

    if promover:
        versiones_staging = client.get_latest_versions(nombre_modelo, stages=["Staging"])
        if versiones_staging:
            version_num = versiones_staging[0].version
            client.transition_model_version_stage(
                name=nombre_modelo,
                version=version_num,
                stage="Production",
                archive_existing_versions=True,
            )
            client.update_model_version(
                name=nombre_modelo,
                version=version_num,
                description=f"Promovido automáticamente. Razón: {razon}",
            )
            print(f"\n  v{version_num} promovido a Production exitosamente.")
        return True
    else:
        print(f"\n  Modelo NO promovido. Razón: {razon}")
        return False


if __name__ == "__main__":
    MODELO = "mejor-clasificador-produccion"
    criterios = CriteriosPromocion(
        metrica_principal="roc_auc",
        mejora_minima_relativa=0.003,   # 0.3%
        umbral_absoluto_minimo=0.85,
    )
    ejecutar_pipeline_promocion(MODELO, criterios)
```

---

**Anterior:** [Producción con LLMs](../produccion/04-despliegue.md) · **Siguiente:** [A/B testing de LLMs](./02-ab-testing-llms.md)
