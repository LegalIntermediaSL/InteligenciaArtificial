# 03 — Detección de Drift en Producción

> **Bloque:** MLOps · **Nivel:** Avanzado · **Tiempo estimado:** 55 min

---

## Índice

1. [Data drift vs concept drift](#1-data-drift-vs-concept-drift)
2. [Métricas estadísticas de drift](#2-métricas-estadísticas-de-drift)
3. [Detección con Evidently](#3-detección-con-evidently)
4. [Alertas automáticas](#4-alertas-automáticas)
5. [Dashboard de monitorización](#5-dashboard-de-monitorización)
6. [Estrategias de reentrenamiento](#6-estrategias-de-reentrenamiento)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Data drift vs concept drift

En producción, los modelos de ML degradan con el tiempo porque el mundo cambia. Hay dos tipos principales de drift:

| Tipo | Qué cambia | Ejemplo | Consecuencia |
|---|---|---|---|
| **Data drift** | Distribución de los inputs X | Los usuarios empiezan a enviar consultas más largas | El modelo puede seguir siendo correcto para inputs similares a los de entrenamiento, pero recibe más inputs fuera de distribución |
| **Concept drift** | Relación entre X e y | Las palabras que antes indicaban spam ahora son legítimas | El modelo hace predicciones incorrectas aunque los inputs parezcan normales |
| **Label drift** | Distribución de las etiquetas y | Aumenta la proporción de clase positiva | Métricas de negocio caen aunque el modelo no haya cambiado |
| **Prediction drift** | Distribución de las predicciones | El modelo predice cada vez más clase negativa | Señal indirecta de data o concept drift |

### ¿Cuándo hay que preocuparse?

```python
# reglas_drift.py — umbral heurístico según impacto del sistema

UMBRALES_DRIFT = {
    "critico": {
        "descripcion": "Sistema de decisión automática (crédito, salud, contratación)",
        "alerta_data_drift": 0.05,   # PSI > 0.05 → revisar
        "alerta_concept_drift": 0.02  # caída F1 > 2% → reentrenar
    },
    "moderado": {
        "descripcion": "Sistema de recomendación, clasificación de contenido",
        "alerta_data_drift": 0.10,
        "alerta_concept_drift": 0.05
    },
    "bajo": {
        "descripcion": "Análisis exploratorio, dashboards internos",
        "alerta_data_drift": 0.20,
        "alerta_concept_drift": 0.10
    }
}
```

---

## 2. Métricas estadísticas de drift

### Population Stability Index (PSI)

El PSI mide cuánto ha cambiado la distribución de una variable entre el periodo de entrenamiento (referencia) y producción (actual).

```python
import numpy as np
import pandas as pd


def calcular_psi(referencia: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    
    Interpretación:
    - PSI < 0.10: sin cambio significativo
    - 0.10 ≤ PSI < 0.20: cambio moderado — monitorizar
    - PSI ≥ 0.20: cambio significativo — investigar y considerar reentrenamiento
    """
    # Crear bins a partir de la distribución de referencia
    breakpoints = np.linspace(
        min(referencia.min(), actual.min()),
        max(referencia.max(), actual.max()),
        bins + 1
    )

    ref_counts = np.histogram(referencia, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]

    # Normalizar y evitar divisiones por cero
    ref_pct = np.where(ref_counts == 0, 0.001, ref_counts / len(referencia))
    act_pct = np.where(act_counts == 0, 0.001, act_counts / len(actual))

    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return float(psi)


def calcular_psi_dataframe(
    df_ref: pd.DataFrame,
    df_actual: pd.DataFrame,
    columnas: list[str]
) -> pd.DataFrame:
    """Calcula PSI para múltiples columnas y retorna un resumen."""
    resultados = []
    for col in columnas:
        if col not in df_ref.columns or col not in df_actual.columns:
            continue
        psi = calcular_psi(df_ref[col].dropna().values, df_actual[col].dropna().values)
        estado = (
            "estable" if psi < 0.10
            else "advertencia" if psi < 0.20
            else "crítico"
        )
        resultados.append({"columna": col, "psi": round(psi, 4), "estado": estado})

    return pd.DataFrame(resultados).sort_values("psi", ascending=False)


# Kolmogorov-Smirnov para variables continuas
from scipy.stats import ks_2samp

def detectar_drift_ks(referencia: np.ndarray, actual: np.ndarray, alpha: float = 0.05) -> dict:
    """Test KS — detecta si dos muestras vienen de la misma distribución."""
    stat, p_value = ks_2samp(referencia, actual)
    return {
        "estadistico_ks": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detectado": p_value < alpha,
        "nivel_confianza": f"{(1 - alpha) * 100:.0f}%"
    }


# Jensen-Shannon Divergence para variables categóricas
from scipy.spatial.distance import jensenshannon

def detectar_drift_categorico(
    ref_counts: dict,
    act_counts: dict
) -> dict:
    """Jensen-Shannon divergence para variables categóricas."""
    categorias = list(set(ref_counts) | set(act_counts))
    p = np.array([ref_counts.get(c, 0) for c in categorias], dtype=float)
    q = np.array([act_counts.get(c, 0) for c in categorias], dtype=float)

    # Normalizar
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q

    js = jensenshannon(p, q)
    return {
        "js_divergence": round(float(js), 4),
        "drift_detectado": js > 0.1,
        "categorias": categorias
    }
```

---

## 3. Detección con Evidently

Evidently es una librería open-source especializada en monitorización de ML que genera informes HTML y JSON sobre drift de datos y rendimiento del modelo.

```bash
pip install evidently
```

```python
# evidently_monitor.py
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
)


def generar_informe_drift(
    df_referencia: pd.DataFrame,
    df_produccion: pd.DataFrame,
    columnas_features: list[str],
    output_path: str = "drift_report.html"
) -> dict:
    """Genera informe de drift completo con Evidently."""

    # Informe de drift de datos
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetMissingValuesMetric(),
    ])

    report.run(
        reference_data=df_referencia[columnas_features],
        current_data=df_produccion[columnas_features]
    )

    # Guardar HTML
    report.save_html(output_path)
    print(f"Informe guardado en: {output_path}")

    # Extraer métricas como dict
    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]

    return {
        "share_drifted_features": drift_metric.get("share_of_drifted_columns", 0),
        "n_drifted_features": drift_metric.get("number_of_drifted_columns", 0),
        "dataset_drift": drift_metric.get("dataset_drift", False),
    }


def suite_tests_drift(
    df_referencia: pd.DataFrame,
    df_produccion: pd.DataFrame,
    columnas_criticas: list[str],
    max_share_drift: float = 0.30
) -> bool:
    """
    Ejecuta un test suite y retorna True si todo pasa.
    Útil en CI/CD o pipelines de monitorización automática.
    """
    tests = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=max_share_drift),
        *[TestColumnDrift(column_name=col) for col in columnas_criticas]
    ])

    tests.run(reference_data=df_referencia, current_data=df_produccion)
    results = tests.as_dict()

    todos_pasan = all(
        t["status"] == "SUCCESS"
        for t in results["tests"]
    )

    for test in results["tests"]:
        estado = "✅" if test["status"] == "SUCCESS" else "❌"
        print(f"{estado} {test['name']}: {test.get('description', '')}")

    return todos_pasan


# Informe completo con métricas de clasificación (si tienes etiquetas reales)
def informe_rendimiento_modelo(
    df_ref_con_predicciones: pd.DataFrame,
    df_actual_con_predicciones: pd.DataFrame,
    target_col: str = "target",
    pred_col: str = "prediction",
    output_path: str = "performance_report.html"
):
    """Genera informe de degradación del rendimiento del modelo."""
    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=df_ref_con_predicciones[[target_col, pred_col]],
        current_data=df_actual_con_predicciones[[target_col, pred_col]]
    )
    report.save_html(output_path)
    print(f"Informe de rendimiento guardado en: {output_path}")
```

---

## 4. Alertas automáticas

```python
# alert_manager.py
import smtplib
import json
import httpx
from email.mime.text import MIMEText
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AlertaDrift:
    sistema: str
    tipo: str  # "data_drift" | "concept_drift" | "label_drift"
    severidad: str  # "advertencia" | "crítico"
    metricas: dict
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def mensaje(self) -> str:
        return (
            f"⚠️ DRIFT DETECTADO — {self.sistema}\n"
            f"Tipo: {self.tipo}\n"
            f"Severidad: {self.severidad}\n"
            f"Timestamp: {self.timestamp.isoformat()}\n"
            f"Métricas: {json.dumps(self.metricas, ensure_ascii=False, indent=2)}"
        )


class AlertManager:
    def __init__(self):
        self.canales: list = []

    def agregar_slack(self, webhook_url: str):
        self.canales.append(("slack", webhook_url))

    def agregar_email(self, smtp_host: str, from_addr: str, to_addrs: list[str], password: str):
        self.canales.append(("email", {
            "smtp_host": smtp_host,
            "from": from_addr,
            "to": to_addrs,
            "password": password
        }))

    def enviar(self, alerta: AlertaDrift):
        for canal, config in self.canales:
            if canal == "slack":
                self._enviar_slack(config, alerta)
            elif canal == "email":
                self._enviar_email(config, alerta)

    def _enviar_slack(self, webhook_url: str, alerta: AlertaDrift):
        emoji = "🔴" if alerta.severidad == "crítico" else "🟡"
        payload = {
            "text": f"{emoji} *Drift detectado en {alerta.sistema}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{alerta.mensaje()}```"}
                }
            ]
        }
        try:
            r = httpx.post(webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            print(f"Alerta Slack enviada: {alerta.sistema}")
        except Exception as e:
            print(f"Error enviando alerta Slack: {e}")

    def _enviar_email(self, config: dict, alerta: AlertaDrift):
        msg = MIMEText(alerta.mensaje())
        msg["Subject"] = f"[Drift {alerta.severidad.upper()}] {alerta.sistema}"
        msg["From"] = config["from"]
        msg["To"] = ", ".join(config["to"])
        try:
            with smtplib.SMTP_SSL(config["smtp_host"], 465) as server:
                server.login(config["from"], config["password"])
                server.sendmail(config["from"], config["to"], msg.as_string())
            print(f"Alerta email enviada: {alerta.sistema}")
        except Exception as e:
            print(f"Error enviando alerta email: {e}")


# Pipeline de monitorización con alertas
def monitorizar(
    df_ref: pd.DataFrame,
    df_actual: pd.DataFrame,
    features: list[str],
    sistema: str,
    alert_manager: AlertManager,
    umbral_psi: float = 0.10
):
    resultados_psi = calcular_psi_dataframe(df_ref, df_actual, features)
    criticos = resultados_psi[resultados_psi["estado"] == "crítico"]

    if len(criticos) > 0:
        alerta = AlertaDrift(
            sistema=sistema,
            tipo="data_drift",
            severidad="crítico" if len(criticos) > 2 else "advertencia",
            metricas={
                "features_criticas": criticos["columna"].tolist(),
                "psi_max": float(criticos["psi"].max()),
                "n_features_drift": len(criticos)
            }
        )
        alert_manager.enviar(alerta)
```

---

## 5. Dashboard de monitorización

```python
# dashboard.py — Dashboard con Streamlit
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Simular datos históricos de PSI
@st.cache_data
def cargar_historico_psi():
    fechas = pd.date_range(end=datetime.now(), periods=90, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "fecha": fechas,
        "feature_A_psi": np.random.exponential(0.05, 90).cumsum() * 0.01 + np.random.normal(0.03, 0.01, 90),
        "feature_B_psi": np.random.normal(0.06, 0.015, 90),
        "feature_C_psi": np.where(fechas > fechas[70], 0.18, np.random.normal(0.04, 0.01, 90)),
        "f1_score": np.where(fechas > fechas[70], 0.87 - np.arange(20) * 0.003, 0.94) [:90],
    })


def render_dashboard():
    st.set_page_config(page_title="MLOps — Drift Monitor", layout="wide")
    st.title("📊 Monitor de Drift en Producción")

    df = cargar_historico_psi()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    ultimo = df.iloc[-1]
    col1.metric("PSI Feature A", f"{ultimo.feature_A_psi:.3f}", delta=f"{ultimo.feature_A_psi - df.iloc[-8].feature_A_psi:.3f}")
    col2.metric("PSI Feature B", f"{ultimo.feature_B_psi:.3f}")
    col3.metric("PSI Feature C", f"{ultimo.feature_C_psi:.3f}", delta_color="inverse")
    col4.metric("F1 Score", f"{ultimo.f1_score:.3f}", delta=f"{ultimo.f1_score - df.iloc[-8].f1_score:.3f}", delta_color="normal")

    # Gráfico PSI temporal
    st.subheader("Evolución PSI por feature (90 días)")
    fig = px.line(
        df.melt(id_vars="fecha", value_vars=["feature_A_psi", "feature_B_psi", "feature_C_psi"],
                var_name="feature", value_name="psi"),
        x="fecha", y="psi", color="feature"
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Advertencia (0.10)")
    fig.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Crítico (0.20)")
    st.plotly_chart(fig, use_container_width=True)

    # F1 Score temporal
    st.subheader("Rendimiento del modelo (F1 Score)")
    fig2 = px.line(df, x="fecha", y="f1_score", title="F1 Score en producción")
    fig2.add_hline(y=0.90, line_dash="dash", line_color="orange", annotation_text="Umbral mínimo (0.90)")
    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
```

```bash
streamlit run dashboard.py
```

---

## 6. Estrategias de reentrenamiento

```python
# retraining_strategy.py
from enum import Enum
from dataclasses import dataclass


class EstrategiaReentrenamiento(Enum):
    SCHEDULED = "programado"          # Periódico independientemente del drift
    TRIGGER_BASED = "por_trigger"     # Solo cuando se detecta drift
    CONTINUOUS = "continuo"           # Online learning — actualización incremental
    HUMAN_IN_LOOP = "human_in_loop"   # El drift activa una revisión humana


@dataclass
class PoliticaReentrenamiento:
    sistema: str
    estrategia: EstrategiaReentrenamiento
    umbral_psi: float = 0.15
    umbral_f1_drop: float = 0.03
    min_nuevos_datos: int = 1000
    ventana_datos_dias: int = 90

    def evaluar(self, psi_actual: float, f1_actual: float, f1_referencia: float, n_nuevos: int) -> str:
        if self.estrategia == EstrategiaReentrenamiento.SCHEDULED:
            return "REENTRENAR_SEGÚN_CALENDARIO"

        if n_nuevos < self.min_nuevos_datos:
            return f"ESPERAR — datos insuficientes ({n_nuevos}/{self.min_nuevos_datos})"

        psi_trigger = psi_actual > self.umbral_psi
        f1_trigger = (f1_referencia - f1_actual) > self.umbral_f1_drop

        if psi_trigger and f1_trigger:
            return "REENTRENAR_URGENTE — drift de datos Y degradación de rendimiento"
        elif f1_trigger:
            return "REENTRENAR — degradación de rendimiento (concept drift probable)"
        elif psi_trigger:
            return "INVESTIGAR — drift de datos sin degradación de rendimiento aún"
        else:
            return "SIN_ACCIÓN — sistema estable"


# Ejemplo
politica = PoliticaReentrenamiento(
    sistema="ClasificadorContratos",
    estrategia=EstrategiaReentrenamiento.TRIGGER_BASED,
    umbral_psi=0.15,
    umbral_f1_drop=0.03,
    min_nuevos_datos=500
)

accion = politica.evaluar(
    psi_actual=0.22,
    f1_actual=0.88,
    f1_referencia=0.94,
    n_nuevos=1200
)
print(f"Acción recomendada: {accion}")
```

---

## 7. Extensiones sugeridas

- **Drift en embeddings**: comparar distribuciones en espacio latente para LLMs con UMAP + distancia de Wasserstein
- **Drift en texto**: detectar cambios en la distribución de tópicos (LDA, BERTopic) entre periodos
- **Integración con MLflow**: registrar métricas de drift como experimentos en MLflow junto al modelo
- **Reentrenamiento automático con Prefect**: usar el trigger de drift como evento que lanza un flow

---

**Anterior:** [02 — A/B testing de LLMs](./02-ab-testing-llms.md) · **Siguiente:** [04 — Pipelines automatizados](./04-pipelines-automatizados.md)
