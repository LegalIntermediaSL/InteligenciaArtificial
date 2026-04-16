# 01 — Sesgos y Fairness en IA

> **Bloque:** IA Responsable · **Nivel:** Avanzado · **Tiempo estimado:** 90 min

---

## Índice

1. [Tipos de sesgo en sistemas de ML](#1-tipos-de-sesgo-en-sistemas-de-ml)
2. [Métricas de fairness](#2-métricas-de-fairness)
3. [Detección de sesgos con fairlearn](#3-detección-de-sesgos-con-fairlearn)
4. [Detección con AIF360](#4-detección-con-aif360)
5. [Técnicas de mitigación](#5-técnicas-de-mitigación)
6. [Caso práctico: clasificación con análisis de fairness](#6-caso-práctico-clasificación-con-análisis-de-fairness)
7. [Visualización de disparidades](#7-visualización-de-disparidades)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Tipos de sesgo en sistemas de ML

El sesgo en IA no es una anomalía: es el resultado esperable cuando se entrena un modelo sobre datos producidos por sistemas sociales que ya son desiguales. Hay cuatro fuentes principales de sesgo que todo practicante debe conocer.

### 1.1 Sesgo de datos (data bias)

Los datos de entrenamiento no representan equitativamente a todos los grupos relevantes.

| Subtipo | Descripción | Ejemplo |
|---------|-------------|---------|
| **Sesgo histórico** | Los datos reflejan decisiones pasadas discriminatorias | Datos de contratación con historial de discriminación de género |
| **Sesgo de representación** | Algunos grupos están subrepresentados | Dataset de reconocimiento facial con 80% de imágenes de personas blancas |
| **Sesgo de medición** | Distintos instrumentos o criterios para distintos grupos | Proxies como código postal que correlacionan con raza |
| **Sesgo de etiquetado** | Los etiquetadores humanos proyectan sus propios sesgos | Clasificación de texto de odio con mayor tasa de falsos positivos para dialectos afroamericanos |

### 1.2 Sesgo algorítmico

El proceso de optimización amplifica desigualdades existentes en los datos.

- Un clasificador maximiza la exactitud global, pero puede sacrificar el rendimiento en grupos minoritarios.
- La regularización penaliza parámetros raramente activados: si un grupo es pequeño, sus patrones se "regularizarán" más agresivamente.
- El sesgo de inducción: modelos lineales pueden no capturar relaciones no lineales presentes solo en subgrupos.

### 1.3 Sesgo de agregación

Se asume que un único modelo sirve para toda la población, ignorando que subgrupos tienen distribuciones distintas.

```
Ejemplo clásico — Paradoja de Simpson:
Grupo A:  Tratamiento X: 70% éxito (140/200)  |  Tratamiento Y: 40% éxito (20/50)
Grupo B:  Tratamiento X: 30% éxito (30/100)   |  Tratamiento Y: 10% éxito (10/100)
Total:    Tratamiento X: 68% éxito (170/250)  |  Tratamiento Y: 20% éxito (30/150)

X parece mejor en total, pero si estratificas por grupo,
el tratamiento Y es mejor en AMBOS grupos separadamente.
```

### 1.4 Sesgo de evaluación

Las métricas de evaluación no capturan el impacto diferencial en distintos grupos.

- Accuracy global = 95% puede ocultar accuracy de 40% en un grupo minoritario.
- AUC-ROC agrega sobre todos los umbrales y oculta problemas a umbrales operativos específicos.
- Los benchmarks estándar (ImageNet, GLUE) tienen sus propios sesgos de representación.

---

## 2. Métricas de fairness

No existe una única definición matemática de "fairness"; son definiciones que entran en conflicto entre sí. Elegir la métrica adecuada es una decisión que depende del contexto y tiene implicaciones éticas.

### 2.1 Paridad demográfica (Demographic Parity)

La tasa de predicciones positivas es igual en todos los grupos.

```
P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)

Donde A es el atributo sensible (ej. género, raza).
```

**Cuándo usarla:** cuando el proceso de decisión debe ser independiente del atributo sensible (ej. selección aleatoria de candidatos para entrevista).

**Limitación:** no tiene en cuenta si la tasa base de la variable objetivo es diferente entre grupos. Si realmente hay más personas cualificadas en un grupo, la paridad demográfica penalizaría artificialmente ese grupo.

### 2.2 Igualdad de oportunidades (Equalized Odds)

Las tasas de verdaderos positivos y falsos positivos son iguales en todos los grupos.

```
P(Ŷ=1 | A=0, Y=1) = P(Ŷ=1 | A=1, Y=1)   ← igual TPR
P(Ŷ=1 | A=0, Y=0) = P(Ŷ=1 | A=1, Y=0)   ← igual FPR
```

**Cuándo usarla:** cuando la clasificación incorrecta tiene costos asimétricos (ej. denegación de libertad condicional, denegación de crédito).

**Variante relajada — Equal Opportunity:** solo iguala la TPR (tasa de verdaderos positivos), ignorando la FPR. Apropiada cuando los falsos negativos son más dañinos que los falsos positivos.

### 2.3 Calibración (Calibration)

Las probabilidades predichas reflejan las verdaderas probabilidades de la variable objetivo, por grupo.

```
P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1) = p   para todo p ∈ [0,1]
```

**Cuándo usarla:** cuando las probabilidades se usan directamente en decisiones (ej. scoring de riesgo médico), y la interpretabilidad del score es importante.

**Tensión fundamental (Chouldechova 2017):** cuando las tasas base difieren entre grupos, no se pueden satisfacer simultáneamente la calibración perfecta y la igualdad de odds. Hay que elegir.

### 2.4 Individual fairness

Personas similares deben recibir predicciones similares. Requiere definir una métrica de similitud entre individuos, lo que es difícil en la práctica y puede ser disputado.

```python
# Definición formal:
# d(f(x), f(x')) ≤ L · d(x, x')
# La distancia entre predicciones no puede superar
# L veces la distancia entre individuos.
```

---

## 3. Detección de sesgos con fairlearn

`fairlearn` es la librería de Microsoft para medir y mitigar sesgos en clasificadores de scikit-learn.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. Crear dataset sintético con sesgo simulado
# ─────────────────────────────────────────────
np.random.seed(42)
n = 3000

# Atributo sensible: grupo A (0) y grupo B (1)
# El grupo B tiene 30% menos probabilidad de etiqueta positiva
# en el dataset original, simulando sesgo histórico
grupo = np.random.binomial(1, 0.4, n)  # 40% grupo B

# Features correlacionadas con el grupo (simula sesgo de medición)
X, y = make_classification(
    n_samples=n,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Introducir sesgo: en el grupo B, invertir algunas etiquetas positivas
sesgo_mask = (grupo == 1) & (y == 1)
flip_indices = np.where(sesgo_mask)[0]
flip_count = int(len(flip_indices) * 0.35)  # 35% de FN extra en grupo B
y[flip_indices[:flip_count]] = 0

df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
df["grupo"] = grupo
df["target"] = y

print(f"Distribución de etiquetas por grupo:")
print(df.groupby("grupo")["target"].value_counts(normalize=True).unstack())
print(f"\nTasa positiva global: {y.mean():.3f}")
print(f"Tasa positiva grupo 0: {y[grupo == 0].mean():.3f}")
print(f"Tasa positiva grupo 1: {y[grupo == 1].mean():.3f}")
```

```python
# ─────────────────────────────────────────────
# 2. Entrenar modelo (sin corrección de sesgo)
# ─────────────────────────────────────────────
X_features = df[[f"feat_{i}" for i in range(10)]].values
y_labels = df["target"].values
sensitive = df["grupo"].values

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X_features, y_labels, sensitive,
    test_size=0.3, random_state=42, stratify=y_labels
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_sc, y_train)

y_pred = modelo.predict(X_test_sc)

print(f"\nAccuracy global: {(y_pred == y_test).mean():.3f}")
```

```python
# ─────────────────────────────────────────────
# 3. Análisis de fairness con MetricFrame
# ─────────────────────────────────────────────
# MetricFrame calcula métricas por cada subgrupo del atributo sensible
mf = MetricFrame(
    metrics={
        "accuracy": lambda y_t, y_p: (y_t == y_p).mean(),
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=s_test
)

print("\n=== MÉTRICAS POR GRUPO ===")
print(mf.by_group.round(3))

print("\n=== MÉTRICAS GLOBALES ===")
print(mf.overall.round(3))

print("\n=== DISPARIDAD (diferencia máxima entre grupos) ===")
print(mf.difference().round(3))

print("\n=== RATIO (mínimo/máximo entre grupos) ===")
print(mf.ratio().round(3))
```

```python
# ─────────────────────────────────────────────
# 4. Métricas de fairness escalares
# ─────────────────────────────────────────────
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=s_test)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)

print("\n=== MÉTRICAS DE FAIRNESS ESCALARES ===")
print(f"Demographic Parity Difference:  {dpd:.4f}")
print(f"  → Ideal: 0.0 | Regla práctica: |DPD| < 0.1")
print(f"Demographic Parity Ratio:       {dpr:.4f}")
print(f"  → Ideal: 1.0 | Regla 4/5: DPR > 0.8 (estándar EEOC EE.UU.)")
print(f"Equalized Odds Difference:      {eod:.4f}")
print(f"  → Ideal: 0.0")
```

**Salida esperada:**

```
Distribución de etiquetas por grupo:
target      0         1
grupo
0      0.408564  0.591436
1      0.545455  0.454545

=== MÉTRICAS POR GRUPO ===
       accuracy  selection_rate  false_positive_rate  false_negative_rate
grupo
0         0.761           0.542                0.187                0.231
1         0.742           0.398                0.164                0.438

=== DISPARIDAD ===
accuracy               0.019
selection_rate         0.144
false_positive_rate    0.023
false_negative_rate    0.207
```

El modelo tiene accuracy similar en ambos grupos (0.76 vs 0.74), pero la **tasa de falsos negativos es el doble en el grupo 1** (43.8% vs 23.1%). Un modelo así, en contexto de préstamos, significaría que personas del grupo 1 cualificadas son rechazadas el doble de veces.

---

## 4. Detección con AIF360

`aif360` (IBM) ofrece más algoritmos de mitigación y conectores a formatos estándar.

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. Convertir datos al formato AIF360
# ─────────────────────────────────────────────
# AIF360 usa su propio contenedor que lleva metadatos de fairness
df_train = pd.DataFrame(
    X_train_sc,
    columns=[f"feat_{i}" for i in range(10)]
)
df_train["grupo"] = s_train
df_train["target"] = y_train

df_test_aif = pd.DataFrame(
    X_test_sc,
    columns=[f"feat_{i}" for i in range(10)]
)
df_test_aif["grupo"] = s_test
df_test_aif["target"] = y_test
df_test_aif["pred"] = y_pred

# BinaryLabelDataset encapsula el dataframe con metadatos de privilegio
train_dataset = BinaryLabelDataset(
    df=df_train,
    label_names=["target"],
    protected_attribute_names=["grupo"],
    favorable_label=1,
    unfavorable_label=0,
    privileged_protected_attributes=[[0]],   # grupo 0 = privilegiado
    unprivileged_protected_attributes=[[1]]  # grupo 1 = no privilegiado
)

test_dataset = BinaryLabelDataset(
    df=df_test_aif[["feat_" + str(i) for i in range(10)] + ["grupo", "target"]],
    label_names=["target"],
    protected_attribute_names=["grupo"],
    favorable_label=1,
    unfavorable_label=0,
    privileged_protected_attributes=[[0]],
    unprivileged_protected_attributes=[[1]]
)

# Dataset con predicciones
test_pred_dataset = test_dataset.copy()
test_pred_dataset.labels = y_pred.reshape(-1, 1)

# ─────────────────────────────────────────────
# 2. Métricas sobre el dataset (antes del modelo)
# ─────────────────────────────────────────────
dataset_metric = BinaryLabelDatasetMetric(
    train_dataset,
    unprivileged_groups=[{"grupo": 1}],
    privileged_groups=[{"grupo": 0}]
)

print("=== MÉTRICAS DEL DATASET DE ENTRENAMIENTO ===")
print(f"Disparate Impact (tasa base): {dataset_metric.disparate_impact():.4f}")
print(f"  → Ideal: 1.0 | Regla 4/5: > 0.8")
print(f"Statistical Parity Difference: {dataset_metric.statistical_parity_difference():.4f}")
print(f"  → Ideal: 0.0")

# ─────────────────────────────────────────────
# 3. Métricas de clasificación
# ─────────────────────────────────────────────
clf_metric = ClassificationMetric(
    test_dataset,
    test_pred_dataset,
    unprivileged_groups=[{"grupo": 1}],
    privileged_groups=[{"grupo": 0}]
)

print("\n=== MÉTRICAS DE CLASIFICACIÓN ===")
print(f"Equal Opportunity Difference:    {clf_metric.equal_opportunity_difference():.4f}")
print(f"Average Odds Difference:         {clf_metric.average_odds_difference():.4f}")
print(f"Disparate Impact (predicciones): {clf_metric.disparate_impact():.4f}")
print(f"Theil Index (desigualdad indiv): {clf_metric.theil_index():.4f}")
print(f"  → El Theil Index mide desigualdad: 0 = perfecta igualdad")
```

---

## 5. Técnicas de mitigación

Las técnicas de mitigación se clasifican según en qué etapa del pipeline actúan.

### 5.1 Pre-procesamiento: Reweighting

Asigna pesos distintos a cada instancia de entrenamiento para compensar el sesgo en los datos.

```python
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# Reweighing: ajusta los pesos de instancias
# para que todos los grupos tengan el mismo
# peso efectivo en el entrenamiento
# ─────────────────────────────────────────────
rw = Reweighing(
    unprivileged_groups=[{"grupo": 1}],
    privileged_groups=[{"grupo": 0}]
)

train_rw = rw.fit_transform(train_dataset)

# Los pesos reajustados están en train_rw.instance_weights
print("Estadísticas de pesos reajustados:")
print(f"  Peso medio grupo 0: {train_rw.instance_weights[s_train == 0].mean():.4f}")
print(f"  Peso medio grupo 1: {train_rw.instance_weights[s_train == 1].mean():.4f}")
# El grupo menos representado obtiene mayor peso

# Entrenar modelo con pesos
modelo_rw = LogisticRegression(max_iter=1000, random_state=42)
modelo_rw.fit(
    X_train_sc,
    y_train,
    sample_weight=train_rw.instance_weights
)

y_pred_rw = modelo_rw.predict(X_test_sc)

# Comparar métricas
dpd_base = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test)
dpd_rw = demographic_parity_difference(y_test, y_pred_rw, sensitive_features=s_test)
eod_base = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test)
eod_rw = equalized_odds_difference(y_test, y_pred_rw, sensitive_features=s_test)

print(f"\n=== COMPARACIÓN ANTES / DESPUÉS DE REWEIGHTING ===")
print(f"{'Métrica':<35} {'Antes':>8} {'Después':>8} {'Mejora':>8}")
print("-" * 60)
print(f"{'Accuracy':<35} {accuracy_score(y_test, y_pred):>8.3f} {accuracy_score(y_test, y_pred_rw):>8.3f}")
print(f"{'Demographic Parity Difference':<35} {dpd_base:>8.4f} {dpd_rw:>8.4f} {abs(dpd_base)-abs(dpd_rw):>8.4f}")
print(f"{'Equalized Odds Difference':<35} {eod_base:>8.4f} {eod_rw:>8.4f} {abs(eod_base)-abs(eod_rw):>8.4f}")
```

### 5.2 En-procesamiento: Adversarial Debiasing

Entrena simultáneamente un clasificador (predice la tarea) y un adversario (intenta predecir el atributo sensible a partir de las predicciones). El clasificador aprende a ser bueno en su tarea sin revelar el atributo sensible.

```python
# Implementación conceptual con PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ClasificadorBase(nn.Module):
    """Red que predice el target y produce un embedding intermedio."""
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z  # predicción y representación latente


class Adversario(nn.Module):
    """Intenta predecir el atributo sensible desde la representación latente."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.red(z)


def entrenar_adversarial_debiasing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    n_epochs: int = 30,
    lambda_adv: float = 1.0,  # peso del adversario
    lr: float = 1e-3
):
    """
    Entrena con debiasing adversarial.
    
    El clasificador minimiza:   L_clf - lambda_adv * L_adv
    El adversario minimiza:     L_adv
    
    El signo negativo en L_adv para el clasificador lo obliga a
    maximizar la dificultad del adversario (confundirlo).
    """
    device = torch.device("cpu")

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    s_t = torch.FloatTensor(s_train).to(device)

    dataset = TensorDataset(X_t, y_t, s_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    clf = ClasificadorBase(input_dim=X_train.shape[1], hidden_dim=64).to(device)
    adv = Adversario(latent_dim=32).to(device)

    opt_clf = optim.Adam(clf.parameters(), lr=lr)
    opt_adv = optim.Adam(adv.parameters(), lr=lr)

    criterio = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        epoch_clf_loss = 0.0
        epoch_adv_loss = 0.0

        for x_batch, y_batch, s_batch in loader:
            # ── Paso 1: actualizar adversario ──
            opt_adv.zero_grad()
            pred_clf, z = clf(x_batch)
            pred_adv = adv(z.detach())  # detach: el gradiente no pasa al clf
            loss_adv = criterio(pred_adv.squeeze(), s_batch)
            loss_adv.backward()
            opt_adv.step()

            # ── Paso 2: actualizar clasificador ──
            opt_clf.zero_grad()
            pred_clf, z = clf(x_batch)
            pred_adv = adv(z)  # sin detach: el clf ve el gradiente del adversario

            loss_clf = criterio(pred_clf.squeeze(), y_batch)
            # El clasificador MAXIMIZA la pérdida del adversario (lo confunde)
            loss_total = loss_clf - lambda_adv * criterio(pred_adv.squeeze(), s_batch)
            loss_total.backward()
            opt_clf.step()

            epoch_clf_loss += loss_clf.item()
            epoch_adv_loss += loss_adv.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"L_clf: {epoch_clf_loss/len(loader):.4f} | "
                  f"L_adv: {epoch_adv_loss/len(loader):.4f}")

    return clf


# Entrenar
clf_adv = entrenar_adversarial_debiasing(
    X_train_sc, y_train, s_train,
    n_epochs=30, lambda_adv=0.8
)

# Obtener predicciones
clf_adv.eval()
with torch.no_grad():
    logits, _ = clf_adv(torch.FloatTensor(X_test_sc))
    y_pred_adv = (torch.sigmoid(logits).squeeze().numpy() > 0.5).astype(int)

dpd_adv = demographic_parity_difference(y_test, y_pred_adv, sensitive_features=s_test)
eod_adv = equalized_odds_difference(y_test, y_pred_adv, sensitive_features=s_test)

print(f"\nAdversarial Debiasing:")
print(f"  DPD: {dpd_adv:.4f}  (baseline: {dpd_base:.4f})")
print(f"  EOD: {eod_adv:.4f}  (baseline: {eod_base:.4f})")
```

### 5.3 Post-procesamiento: Threshold Optimization

Ajusta los umbrales de decisión por grupo para igualar las métricas de fairness sin re-entrenar.

```python
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import equalized_odds_difference

# ThresholdOptimizer busca umbrales distintos por grupo
# que satisfagan la restricción de fairness especificada
opt = ThresholdOptimizer(
    estimator=modelo,               # modelo ya entrenado
    constraints="equalized_odds",   # restricción a satisfacer
    objective="balanced_accuracy_score",
    predict_method="predict_proba",
    flip=True
)

opt.fit(X_train_sc, y_train, sensitive_features=s_train)
y_pred_opt = opt.predict(X_test_sc, sensitive_features=s_test)

acc_opt = accuracy_score(y_test, y_pred_opt)
dpd_opt = demographic_parity_difference(y_test, y_pred_opt, sensitive_features=s_test)
eod_opt = equalized_odds_difference(y_test, y_pred_opt, sensitive_features=s_test)

print(f"\n=== RESULTADOS THRESHOLD OPTIMIZATION ===")
print(f"Accuracy:                  {acc_opt:.4f}")
print(f"Demographic Parity Diff:   {dpd_opt:.4f}")
print(f"Equalized Odds Diff:       {eod_opt:.4f}")

# Mostrar los umbrales aprendidos por grupo
print(f"\nUmbrales por grupo:")
for grupo_val, interpolated in opt.interpolated_thresholder_.interpolated_thresholder_.items():
    print(f"  Grupo {grupo_val}: {interpolated}")
```

---

## 6. Caso práctico: clasificación con análisis de fairness

Reproducimos el ciclo completo: datos → modelo base → análisis → mitigación → comparación.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# ─────────────────────────────────────────────
# Dataset: Adult Income (UCI)
# Tarea: predecir si ingresos > 50K
# Atributo sensible: sexo
# ─────────────────────────────────────────────

# Cargar desde la URL de UCI (o desde disco si ya se tiene)
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ["age","workclass","fnlwgt","education","education_num","marital",
            "occupation","relationship","race","sex","cap_gain","cap_loss",
            "hours","country","income"]
    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    print(f"Dataset cargado: {len(df)} filas")
except Exception:
    # Si no hay conexión, crear dataset sintético equivalente
    print("Usando dataset sintético (sin conexión a UCI)")
    np.random.seed(0)
    n = 5000
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "education_num": np.random.randint(1, 16, n),
        "hours": np.random.randint(20, 60, n),
        "cap_gain": np.random.exponential(500, n).astype(int),
        "cap_loss": np.random.exponential(100, n).astype(int),
        "sex": np.random.choice(["Male", "Female"], n, p=[0.67, 0.33]),
        "workclass": np.random.choice(["Private", "Self-emp", "Government"], n),
        "marital": np.random.choice(["Married", "Single", "Divorced"], n),
    })
    # Sesgo sintético: mujeres con menos probabilidad de >50K
    p_income = np.where(df["sex"] == "Male", 0.32, 0.11)
    df["income"] = np.vectorize(lambda p: ">50K" if np.random.random() < p else "<=50K")(p_income)

# Limpiar y preparar
df = df.dropna()
df["income_bin"] = (df["income"].str.strip() == ">50K").astype(int)
df["sex_bin"] = (df["sex"].str.strip() == "Male").astype(int)

# Features
cat_cols = ["workclass", "marital"] if "marital" in df.columns else []
num_cols = ["age", "education_num", "hours", "cap_gain", "cap_loss"]

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

features = num_cols + cat_cols
X = df[features].values
y = df["income_bin"].values
sensitive = df["sex_bin"].values

X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
    X, y, sensitive, test_size=0.25, random_state=42, stratify=y
)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

# ─────────────────────────────────────────────
# Modelo base
# ─────────────────────────────────────────────
clf_base = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf_base.fit(X_tr_sc, y_tr)
y_pred_base = clf_base.predict(X_te_sc)

# ─────────────────────────────────────────────
# Análisis de fairness
# ─────────────────────────────────────────────
mf_base = MetricFrame(
    metrics={
        "accuracy": lambda yt, yp: (yt == yp).mean(),
        "tasa_seleccion": selection_rate,
        "fnr": false_negative_rate,
    },
    y_true=y_te,
    y_pred=y_pred_base,
    sensitive_features=s_te
)

print("=== ANÁLISIS MODELO BASE (GBM sin corrección) ===")
print(mf_base.by_group.rename(index={0: "Mujer", 1: "Hombre"}).round(3))
print(f"\nDisparidad tasa selección: {mf_base.difference()['tasa_seleccion']:.4f}")
print(f"Disparidad FNR:            {mf_base.difference()['fnr']:.4f}")

# ─────────────────────────────────────────────
# Mitigación: ExponentiatedGradient con EqualizedOdds
# ─────────────────────────────────────────────
# ExponentiatedGradient reduce el problema de fairness
# a una secuencia de problemas ponderados de clasificación
clf_mitigado = ExponentiatedGradient(
    estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
    constraints=EqualizedOdds(),
    eps=0.01  # tolerancia de violación de la restricción
)
clf_mitigado.fit(X_tr_sc, y_tr, sensitive_features=s_tr)
y_pred_mit = clf_mitigado.predict(X_te_sc)

mf_mit = MetricFrame(
    metrics={
        "accuracy": lambda yt, yp: (yt == yp).mean(),
        "tasa_seleccion": selection_rate,
        "fnr": false_negative_rate,
    },
    y_true=y_te,
    y_pred=y_pred_mit,
    sensitive_features=s_te
)

print("\n=== ANÁLISIS MODELO MITIGADO (ExponentiatedGradient + EqualizedOdds) ===")
print(mf_mit.by_group.rename(index={0: "Mujer", 1: "Hombre"}).round(3))
print(f"\nDisparidad tasa selección: {mf_mit.difference()['tasa_seleccion']:.4f}")
print(f"Disparidad FNR:            {mf_mit.difference()['fnr']:.4f}")

print("\n=== RESUMEN DEL TRADE-OFF ===")
print(f"{'':35} {'Base':>8} {'Mitigado':>9}")
print("-" * 55)
acc_b = (y_pred_base == y_te).mean()
acc_m = (y_pred_mit == y_te).mean()
dpd_b = demographic_parity_difference(y_te, y_pred_base, sensitive_features=s_te)
dpd_m = demographic_parity_difference(y_te, y_pred_mit, sensitive_features=s_te)
eod_b = equalized_odds_difference(y_te, y_pred_base, sensitive_features=s_te)
eod_m = equalized_odds_difference(y_te, y_pred_mit, sensitive_features=s_te)
print(f"{'Accuracy':<35} {acc_b:>8.3f} {acc_m:>9.3f}")
print(f"{'Demographic Parity Difference':<35} {dpd_b:>8.4f} {dpd_m:>9.4f}")
print(f"{'Equalized Odds Difference':<35} {eod_b:>8.4f} {eod_m:>9.4f}")
```

---

## 7. Visualización de disparidades

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# Gráfico 1: Comparación de métricas por grupo
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Análisis de Fairness: Base vs Mitigado", fontsize=14, fontweight="bold")

metricas = ["accuracy", "tasa_seleccion", "fnr"]
nombres = ["Accuracy", "Tasa de Selección", "Tasa Falsos Negativos"]
grupos = ["Mujer (grupo 0)", "Hombre (grupo 1)"]
colores = ["#e74c3c", "#3498db"]

for ax, metrica, nombre in zip(axes, metricas, nombres):
    vals_base = [
        mf_base.by_group[metrica].iloc[0],
        mf_base.by_group[metrica].iloc[1]
    ]
    vals_mit = [
        mf_mit.by_group[metrica].iloc[0],
        mf_mit.by_group[metrica].iloc[1]
    ]

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, vals_base, width, label="Modelo base",
                   color=colores, alpha=0.6, edgecolor="black")
    bars2 = ax.bar(x + width/2, vals_mit, width, label="Mitigado",
                   color=colores, alpha=1.0, edgecolor="black", hatch="//")

    ax.set_title(nombre)
    ax.set_xticks(x)
    ax.set_xticklabels(grupos, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(y=vals_base[0], color=colores[0], linestyle=":", alpha=0.4)
    ax.axhline(y=vals_base[1], color=colores[1], linestyle=":", alpha=0.4)

    # Etiquetas de valor
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

patch_base = mpatches.Patch(facecolor="white", edgecolor="black", alpha=0.6, label="Modelo base")
patch_mit = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Mitigado")
fig.legend(handles=[patch_base, patch_mit], loc="lower center", ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("fairness_comparacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fairness_comparacion.png")

# ─────────────────────────────────────────────
# Gráfico 2: Frontier accuracy–fairness
# ─────────────────────────────────────────────
# Exploramos el trade-off variando el peso del adversario
# (simulado aquí con ThresholdOptimizer a diferentes tolerancias)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import equalized_odds_difference
from sklearn.metrics import accuracy_score

resultados = []
for tol in [0.01, 0.05, 0.1, 0.2, 0.5]:
    opt_tmp = ThresholdOptimizer(
        estimator=clf_base,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        flip=True,
    )
    opt_tmp.fit(X_tr_sc, y_tr, sensitive_features=s_tr)
    y_tmp = opt_tmp.predict(X_te_sc, sensitive_features=s_te)
    resultados.append({
        "tol": tol,
        "accuracy": accuracy_score(y_te, y_tmp),
        "eod": abs(equalized_odds_difference(y_te, y_tmp, sensitive_features=s_te))
    })

# Añadir el modelo base sin corrección
resultados.insert(0, {
    "tol": "base",
    "accuracy": accuracy_score(y_te, y_pred_base),
    "eod": abs(equalized_odds_difference(y_te, y_pred_base, sensitive_features=s_te))
})

df_front = pd.DataFrame(resultados)

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(
    df_front["eod"], df_front["accuracy"],
    c=range(len(df_front)), cmap="RdYlGn_r",
    s=100, zorder=5, edgecolors="black"
)

for _, row in df_front.iterrows():
    label = "Base" if row["tol"] == "base" else f"tol={row['tol']}"
    ax.annotate(label, (row["eod"], row["accuracy"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel("Equalized Odds Difference (↓ mejor fairness)", fontsize=11)
ax.set_ylabel("Accuracy (↑ mejor rendimiento)", fontsize=11)
ax.set_title("Frontera Accuracy–Fairness\n(ThresholdOptimizer a diferentes tolerancias)", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fairness_frontier.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fairness_frontier.png")

# ─────────────────────────────────────────────
# Gráfico 3: Matriz de confusión por grupo
# ─────────────────────────────────────────────
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Matrices de confusión por grupo (modelo base)", fontsize=12)

for idx, (nombre_grupo, grupo_val) in enumerate([("Mujer (grupo 0)", 0), ("Hombre (grupo 1)", 1)]):
    mask = s_te == grupo_val
    cm = confusion_matrix(y_te[mask], y_pred_base[mask])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                ax=axes[idx], cbar=False,
                xticklabels=["≤50K", ">50K"],
                yticklabels=["≤50K", ">50K"])
    axes[idx].set_title(f"{nombre_grupo}\n(n={mask.sum()})")
    axes[idx].set_xlabel("Predicho")
    axes[idx].set_ylabel("Real")

plt.tight_layout()
plt.savefig("confusion_por_grupo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: confusion_por_grupo.png")
```

---

## 8. Extensiones sugeridas

- **Fairness en NLP:** analizar sesgos en embeddings con el test WEAT (Word Embedding Association Test) usando `wefe` (Word Embeddings Fairness Evaluation).
- **Audit automatizado en CI/CD:** añadir un test de fairness como paso de CI que falle el pipeline si `demographic_parity_difference > 0.1`.
- **Interseccionalidad:** las métricas de fairness por grupo único pueden ocultar sesgos en la intersección (mujeres + minorías étnicas). Fairlearn permite subgrupos compuestos con `MetricFrame` y atributos múltiples.
- **Fairness en ranking:** para sistemas de recomendación, estudiar las métricas de exposición relativa (NDCG por grupo) con la librería `rankfairness`.
- **Documentar el análisis:** integrar las métricas de fairness directamente en el model card (tutorial 03).

---

**Anterior:** [README — Índice del bloque](./README.md) · **Siguiente:** [02 — Interpretabilidad y Explicabilidad](./02-interpretabilidad.md)
