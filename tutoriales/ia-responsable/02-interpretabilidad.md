# 02 — Interpretabilidad y Explicabilidad en IA

> **Bloque:** IA Responsable · **Nivel:** Avanzado · **Tiempo estimado:** 75 min

---

## Índice

1. [Interpretabilidad global vs local](#1-interpretabilidad-global-vs-local)
2. [SHAP: fundamentos y uso práctico](#2-shap-fundamentos-y-uso-práctico)
3. [LIME: explicaciones locales agnósticas](#3-lime-explicaciones-locales-agnósticas)
4. [Attention visualization con BertViz](#4-attention-visualization-con-bertviz)
5. [Saliency maps para modelos de visión](#5-saliency-maps-para-modelos-de-visión)
6. [Caso práctico: explicar predicciones de clasificador de texto](#6-caso-práctico-explicar-predicciones-de-clasificador-de-texto)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Interpretabilidad global vs local

La interpretabilidad no es una propiedad binaria: un modelo puede ser explicable de formas distintas a distintos niveles.

### Taxonomía de explicaciones

| Dimensión | Global | Local |
|-----------|--------|-------|
| **Alcance** | Comportamiento del modelo completo | Una predicción individual |
| **Pregunta** | ¿Qué features importan en general? | ¿Por qué el modelo dijo X para esta instancia? |
| **Audiencia** | Investigadores, reguladores, equipos ML | Usuarios finales, auditores de decisión |
| **Herramienta** | Feature importances, SHAP global, PDP | SHAP local, LIME, counterfactuals |
| **Ejemplo** | "La edad es el predictor más importante" | "Para Juan, la denegación se debe a su historial de crédito" |

### Modelos intrínsecamente interpretables vs caja negra

```
Interpretabilidad intrínseca          Caja negra (post-hoc)
────────────────────────────          ──────────────────────
Regresión lineal/logística            Gradient Boosting
Árboles de decisión (pequeños)        Random Forest (profundo)
Redes bayesianas simples              Redes neuronales profundas
Reglas de clasificación               Transformers / LLMs

↑ Mayor interpretabilidad,            ↑ Mayor capacidad predictiva,
  menor capacidad                       menor interpretabilidad nativa
```

### El problema del "rashomon set"

Para casi cualquier dataset hay cientos de modelos con accuracy estadísticamente equivalente pero explicaciones radicalmente distintas. Elegir un modelo porque "es más explicable" sin comprobar que su accuracy es comparable es una falsa optimización.

---

## 2. SHAP: fundamentos y uso práctico

SHAP (SHapley Additive exPlanations) es el estándar de facto para explicabilidad en ML clásico. Asigna a cada feature una contribución aditiva a la predicción, fundamentada en la teoría de juegos cooperativos (valores de Shapley).

### 2.1 Fundamento matemático

La predicción para una instancia x se descompone como:

```
f(x) = φ₀ + Σᵢ φᵢ

Donde:
  φ₀     = predicción media del modelo (baseline)
  φᵢ     = contribución de la feature i para esta instancia
  Σᵢ φᵢ = desviación total respecto al baseline
```

Los valores φᵢ se calculan considerando todas las posibles coaliciones de features, garantizando que la suma de contribuciones iguala exactamente la predicción. Esto los hace **consistentes** y **globalmente exactos**.

### 2.2 TreeExplainer (modelos basados en árboles)

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Preparar datos y modelo
# ─────────────────────────────────────────────
datos = load_breast_cancer()
X = pd.DataFrame(datos.data, columns=datos.feature_names)
y = datos.target  # 0=maligno, 1=benigno

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

print(f"Accuracy: {modelo.score(X_test, y_test):.3f}")

# ─────────────────────────────────────────────
# 2. TreeExplainer: rápido y exacto para árboles
# ─────────────────────────────────────────────
explainer = shap.TreeExplainer(modelo)

# Calcular SHAP values para el test set
# shap_values.shape = (n_muestras, n_features)
shap_values = explainer.shap_values(X_test)

print(f"\nForma de SHAP values: {shap_values.shape}")
print(f"Baseline (expected_value): {explainer.expected_value:.4f}")

# Para clasificación binaria en GBM, shap_values son para la clase positiva
# La suma de cada fila + expected_value = log-odds predicho
fila = 0
suma = explainer.expected_value + shap_values[fila].sum()
logodds_pred = modelo.predict_proba(X_test.iloc[[fila]])[0][1]
print(f"\nVerificación instancia 0:")
print(f"  φ₀ + Σφᵢ = {suma:.4f}")
print(f"  log-odds predicho (modelo): {np.log(logodds_pred/(1-logodds_pred)):.4f}")
# Deben ser iguales (o muy cercanos)
```

```python
# ─────────────────────────────────────────────
# 3. Plots de importancia global
# ─────────────────────────────────────────────

# Summary plot: muestra importancia global y distribución de valores
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",       # barras con importancia media |SHAP|
    show=False,
    max_display=15
)
plt.title("Importancia global de features (|SHAP| medio)")
plt.tight_layout()
plt.savefig("shap_global_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# Beeswarm: distribución de SHAP values por feature
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="dot",       # beeswarm
    show=False,
    max_display=15
)
plt.title("Distribución de valores SHAP (beeswarm)")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráficos guardados.")
```

```python
# ─────────────────────────────────────────────
# 4. Explicación local: waterfall plot
# ─────────────────────────────────────────────
# Elegir una instancia para explicar
idx = 5
instancia = X_test.iloc[idx]
pred_prob = modelo.predict_proba(X_test.iloc[[idx]])[0][1]
pred_clase = modelo.predict(X_test.iloc[[idx]])[0]

print(f"\nInstancia {idx}:")
print(f"  Predicción: {'Benigno' if pred_clase == 1 else 'Maligno'} (p={pred_prob:.3f})")
print(f"  Real:       {'Benigno' if y_test.iloc[idx] == 1 else 'Maligno'}")

# Waterfall plot: muestra paso a paso cómo cada feature
# lleva la predicción desde el baseline hasta el valor final
shap_exp = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=instancia.values,
    feature_names=list(X.columns)
)

plt.figure()
shap.plots.waterfall(shap_exp, show=False)
plt.title(f"Explicación local — Instancia {idx} (p={pred_prob:.3f})")
plt.tight_layout()
plt.savefig(f"shap_waterfall_{idx}.png", dpi=150, bbox_inches="tight")
plt.show()
```

```python
# ─────────────────────────────────────────────
# 5. Dependence plots: interacción entre features
# ─────────────────────────────────────────────
# Muestra cómo el SHAP value de una feature varía
# con su valor numérico, coloreada por otra feature

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Feature más importante (índice 0 del bar plot)
top_feature = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X.columns
).nlargest(2).index.tolist()

for ax, feat in zip(axes, top_feature):
    shap.dependence_plot(
        feat, shap_values, X_test,
        ax=ax, show=False
    )
    ax.set_title(f"Dependence plot: {feat}")

plt.tight_layout()
plt.savefig("shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 2.3 LinearExplainer (modelos lineales)

Para regresión/clasificación lineal, LinearExplainer es exacto y muy rápido.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

modelo_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
modelo_lr.fit(X_train_sc, y_train)

# LinearExplainer usa directamente los coeficientes del modelo
# Mucho más rápido que KernelExplainer para modelos lineales
explainer_lr = shap.LinearExplainer(
    modelo_lr,
    X_train_sc,
    feature_perturbation="interventional"
)

X_test_df_sc = pd.DataFrame(X_test_sc, columns=X.columns)
shap_values_lr = explainer_lr.shap_values(X_test_df_sc)

# Para clasificación binaria con LogisticRegression, hay 2 sets de SHAP values
# shap_values_lr[0] = clase 0, shap_values_lr[1] = clase 1
if isinstance(shap_values_lr, list):
    sv = shap_values_lr[1]  # clase positiva
else:
    sv = shap_values_lr

# Verificar consistencia con los coeficientes
print("Comparación SHAP medio vs coeficientes normalizados:")
shap_importancia = np.abs(sv).mean(axis=0)
coef_abs = np.abs(modelo_lr.coef_[0])
for feat, s, c in sorted(
    zip(X.columns, shap_importancia, coef_abs),
    key=lambda x: x[1], reverse=True
)[:5]:
    print(f"  {feat:<35} SHAP: {s:.4f}  |coef|: {c:.4f}")
```

### 2.4 SHAP values para modelos de texto (transformers)

```python
import shap
import transformers
import torch

# Pipeline de clasificación de sentimiento
pipeline_nlp = transformers.pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True,
    device=-1  # CPU
)

# Textos de ejemplo
textos = [
    "This product is absolutely amazing and I love it",
    "The service was terrible and I want a refund",
    "It was okay, not great but not awful either",
]

# PartitionExplainer para texto: más eficiente que KernelExplainer
# Particiona recursivamente los tokens y mide su contribución
explainer_text = shap.Explainer(pipeline_nlp)
shap_values_text = explainer_text(textos)

# Plot de texto con highlighting (positivo=verde, negativo=rojo)
shap.plots.text(shap_values_text[0, :, "POSITIVE"], display=False)

# Acceder a los valores directamente
print("\nSHAP values para texto 0 (clase POSITIVE):")
sv_texto = shap_values_text[0, :, "POSITIVE"]
tokens = shap_values_text.data[0]
for token, val in sorted(
    zip(tokens, sv_texto.values),
    key=lambda x: abs(x[1]), reverse=True
)[:8]:
    signo = "+" if val > 0 else ""
    print(f"  '{token}': {signo}{val:.4f}")
```

---

## 3. LIME: explicaciones locales agnósticas

LIME (Local Interpretable Model-agnostic Explanations) construye un modelo lineal simple que aproxima el modelo complejo en la vecindad de una instancia. No requiere acceso a los internos del modelo: solo necesita hacer predicciones.

```python
import lime
import lime.lime_tabular
import lime.lime_text
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# LIME tabular
# ─────────────────────────────────────────────
datos = load_breast_cancer()
X = datos.data
y = datos.target
nombres_features = datos.feature_names
nombres_clases = datos.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Crear el explainer LIME
# LIME genera perturbaciones alrededor de la instancia,
# obtiene predicciones del modelo y ajusta un modelo lineal local
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=nombres_features,
    class_names=nombres_clases,
    mode="classification",
    discretize_continuous=True,  # discretiza features continuas para el modelo lineal
    random_state=42
)

# Explicar una instancia concreta
idx = 10
instancia = X_test[idx]
pred_prob = clf_rf.predict_proba([instancia])[0]
pred_clase = clf_rf.predict([instancia])[0]

print(f"Instancia {idx}:")
print(f"  Predicción: {nombres_clases[pred_clase]} (p={pred_prob[pred_clase]:.3f})")
print(f"  Real:       {nombres_clases[y_test[idx]]}")

# explain_instance genera perturbaciones y construye el modelo local
exp = explainer_lime.explain_instance(
    instancia,
    clf_rf.predict_proba,
    num_features=10,         # cuántas features incluir en la explicación
    num_samples=1000,        # cuántas perturbaciones generar
    labels=[pred_clase]      # clase a explicar
)

print(f"\nExplicación LIME (clase '{nombres_clases[pred_clase]}'):")
for feature, peso in exp.as_list(label=pred_clase):
    signo = "→ POSITIVO" if peso > 0 else "→ NEGATIVO"
    print(f"  {feature:<45} peso: {peso:+.4f}  {signo}")

# Guardar como HTML interactivo
exp.save_to_file("lime_explicacion.html")
print("\nExplicación guardada en lime_explicacion.html")
```

```python
# ─────────────────────────────────────────────
# LIME para texto
# ─────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lime.lime_text

# Dataset simple de sentimiento
textos_train = [
    "great product love it amazing",
    "excellent quality highly recommend",
    "wonderful experience very happy",
    "terrible quality waste of money",
    "awful product very disappointed",
    "horrible experience never again",
    "okay product nothing special",
]
etiquetas_train = [1, 1, 1, 0, 0, 0, 1]

textos_test = [
    "amazing quality but terrible support service",
    "not great but acceptable for the price",
]

# Pipeline sklearn (TF-IDF + LR)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(random_state=42))
])
pipe.fit(textos_train, etiquetas_train)

# LIME text explainer
explainer_text = lime.lime_text.LimeTextExplainer(
    class_names=["negativo", "positivo"],
    random_state=42
)

for texto in textos_test:
    pred = pipe.predict([texto])[0]
    pred_prob = pipe.predict_proba([texto])[0]

    exp = explainer_text.explain_instance(
        texto,
        pipe.predict_proba,
        num_features=6,
        labels=[pred]
    )

    print(f"\nTexto: '{texto}'")
    print(f"Predicción: {'POSITIVO' if pred == 1 else 'NEGATIVO'} "
          f"(confianza: {pred_prob[pred]:.2f})")
    print("Palabras clave:")
    for palabra, peso in exp.as_list(label=pred):
        direc = "positivo" if peso > 0 else "negativo"
        print(f"  '{palabra}': {peso:+.4f} ({direc})")
```

### Diferencias clave SHAP vs LIME

| Aspecto | SHAP | LIME |
|---------|------|------|
| **Base teórica** | Teoría de juegos (Shapley values) | Modelo sustituto lineal local |
| **Consistencia** | Garantizada matemáticamente | No garantizada (varía con la semilla) |
| **Velocidad** | Rápido con TreeExplainer; lento con KernelExplainer | Depende de `num_samples` |
| **Fidelidad global** | Alta (exacta para árboles) | Solo local (por diseño) |
| **Interacciones** | Detecta interacciones entre features | No captura interacciones |
| **Uso recomendado** | Análisis sistemático, auditorías | Explicaciones rápidas, modelos caja negra |

---

## 4. Attention visualization con BertViz

Los pesos de atención en transformers no son valores SHAP y no explican directamente las predicciones, pero visualizarlos revela qué pares de tokens el modelo relaciona, útil para debugging y auditoría.

```python
# Instalación: pip install bertviz transformers torch
# BertViz requiere Jupyter para las visualizaciones interactivas
# Aquí mostramos la API y cómo extraer las matrices

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# 1. Cargar modelo y extraer atenciones
# ─────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True  # crucial: activa la salida de atenciones
)
model.eval()

texto = "The bank can guarantee deposits will eventually cover future bank losses"
inputs = tokenizer(texto, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tupla de (n_capas,) tensores
# Cada tensor: (batch, n_heads, seq_len, seq_len)
attentions = outputs.attentions
n_layers = len(attentions)
n_heads = attentions[0].shape[1]
seq_len = len(tokens)

print(f"Modelo: {n_layers} capas, {n_heads} cabezas de atención")
print(f"Tokens: {tokens}")

# ─────────────────────────────────────────────
# 2. Visualizar una cabeza de atención específica
# ─────────────────────────────────────────────
# Cabeza conocida por capturar relaciones sintácticas
capa_idx = 5
cabeza_idx = 2

attn_matrix = attentions[capa_idx][0, cabeza_idx].numpy()  # (seq_len, seq_len)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    attn_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="Blues",
    ax=ax,
    fmt=".2f",
    annot=len(tokens) < 15,
    linewidths=0.5
)
ax.set_title(f"Pesos de atención — Capa {capa_idx+1}, Cabeza {cabeza_idx+1}")
ax.set_xlabel("Token destino (qué se atiende)")
ax.set_ylabel("Token origen (quién atiende)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"attention_L{capa_idx+1}_H{cabeza_idx+1}.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# 3. Atención media por capa
# ─────────────────────────────────────────────
# Promedio sobre todas las cabezas de cada capa
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Atención media por capa (promedio sobre cabezas)", fontsize=12)

for capa, ax in enumerate(axes.flat):
    if capa < n_layers:
        mean_attn = attentions[capa][0].mean(dim=0).numpy()
        sns.heatmap(
            mean_attn, ax=ax,
            cmap="Blues", cbar=False,
            xticklabels=False, yticklabels=False
        )
        ax.set_title(f"Capa {capa+1}", fontsize=9)
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig("attention_todas_capas.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# 4. Visualización interactiva con BertViz (Jupyter)
# ─────────────────────────────────────────────
# En Jupyter Notebook, usar:
try:
    from bertviz import head_view, model_view

    # head_view: visualiza las cabezas de una capa
    # head_view(attentions, tokens)

    # model_view: panorama de todas las capas y cabezas
    # model_view(attentions, tokens)

    print("BertViz disponible. Ejecutar head_view(attentions, tokens) en Jupyter.")
except ImportError:
    print("BertViz no instalado. pip install bertviz")
    print("Funciona mejor en entornos Jupyter Notebook.")
```

---

## 5. Saliency maps para modelos de visión

Los saliency maps indican qué píxeles de la imagen input contribuyen más a la predicción, calculando el gradiente de la salida respecto a los píxeles.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os

# ─────────────────────────────────────────────
# 1. Cargar modelo e imagen
# ─────────────────────────────────────────────
modelo_vision = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modelo_vision.eval()

# Transformaciones estándar de ImageNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Usar una imagen de ejemplo (o crear una sintética)
img_path = "imagen_prueba.jpg"
if not os.path.exists(img_path):
    # Imagen sintética: gradiente de colores
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            img_arr[i, j] = [i, j, (i + j) // 2]
    img = Image.fromarray(img_arr)
    img.save(img_path)
    print("Imagen sintética creada.")
else:
    img = Image.open(img_path).convert("RGB")

# Preparar tensor con requires_grad=True para calcular gradientes
img_tensor = transform(img).unsqueeze(0)
img_tensor.requires_grad_(True)

# ─────────────────────────────────────────────
# 2. Vanilla Gradient Saliency
# ─────────────────────────────────────────────
logits = modelo_vision(img_tensor)
clase_predicha = logits.argmax().item()
score_clase = logits[0, clase_predicha]

# Backpropagation hasta la imagen
modelo_vision.zero_grad()
score_clase.backward()

# El gradiente indica la sensibilidad de cada píxel
gradientes = img_tensor.grad.data[0]  # (3, H, W)

# Saliency map: magnitud del gradiente por canal, luego máximo entre canales
saliency = gradientes.abs().max(dim=0)[0].numpy()

# Normalizar
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# ─────────────────────────────────────────────
# 3. Guided Backpropagation
# ─────────────────────────────────────────────
class GuidedReLU(nn.Module):
    """
    ReLU modificada para guided backprop:
    solo propaga gradientes donde TANTO la activación
    como el gradiente son positivos.
    """
    def forward(self, x):
        return torch.relu(x)

    def backward_hook(self, module, grad_in, grad_out):
        # Solo dejar pasar gradientes positivos donde la activación fue positiva
        return (torch.clamp(grad_in[0], min=0),)


def registrar_guided_relu(model):
    """Reemplaza ReLUs por GuidedReLU para guided backpropagation."""
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hook = module.register_backward_hook(
                lambda m, gi, go: (torch.clamp(gi[0], min=0),)
            )
            hooks.append(hook)
    return hooks


modelo_guided = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modelo_guided.eval()
hooks = registrar_guided_relu(modelo_guided)

img_tensor_g = transform(img).unsqueeze(0).requires_grad_(True)
logits_g = modelo_guided(img_tensor_g)
clase_g = logits_g.argmax().item()
modelo_guided.zero_grad()
logits_g[0, clase_g].backward()

gradientes_guided = img_tensor_g.grad.data[0].numpy()
guided_saliency = np.abs(gradientes_guided).max(axis=0)
guided_saliency = (guided_saliency - guided_saliency.min()) / (guided_saliency.max() - guided_saliency.min() + 1e-8)

# Limpiar hooks
for h in hooks:
    h.remove()

# ─────────────────────────────────────────────
# 4. Grad-CAM (Gradient-weighted Class Activation Maps)
# ─────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM: usa los gradientes de la última capa convolucional
    para producir un mapa de calor de alta resolución.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activaciones = None
        self.gradientes = None
        self._registrar_hooks()

    def _registrar_hooks(self):
        def guardar_activaciones(module, input, output):
            self.activaciones = output.detach()

        def guardar_gradientes(module, grad_in, grad_out):
            self.gradientes = grad_out[0].detach()

        self.target_layer.register_forward_hook(guardar_activaciones)
        self.target_layer.register_backward_hook(guardar_gradientes)

    def generar_cam(self, input_tensor, clase=None):
        logits = self.model(input_tensor)
        if clase is None:
            clase = logits.argmax().item()

        self.model.zero_grad()
        logits[0, clase].backward()

        # Peso por gradiente global average pooling
        pesos = self.gradientes.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (pesos * self.activaciones).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam).squeeze().numpy()

        # Redimensionar al tamaño del input
        cam_resized = np.array(
            Image.fromarray(cam).resize((224, 224), Image.BILINEAR)
        )
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        return cam_norm, clase


modelo_cam = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
modelo_cam.eval()

# Target: última capa convolucional de ResNet50
grad_cam = GradCAM(modelo_cam, modelo_cam.layer4[-1])

img_tensor_cam = transform(img).unsqueeze(0).requires_grad_(True)
cam, clase_cam = grad_cam.generar_cam(img_tensor_cam)

# ─────────────────────────────────────────────
# 5. Visualización comparativa
# ─────────────────────────────────────────────
img_original = np.array(img.resize((224, 224)))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f"Explicabilidad visual — clase predicha: {clase_predicha}", fontsize=12)

axes[0].imshow(img_original)
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(saliency, cmap="hot")
axes[1].set_title("Vanilla Gradient\nSaliency")
axes[1].axis("off")

axes[2].imshow(guided_saliency, cmap="hot")
axes[2].set_title("Guided\nBackpropagation")
axes[2].axis("off")

# Grad-CAM superpuesto
axes[3].imshow(img_original)
axes[3].imshow(cam, cmap="jet", alpha=0.5)
axes[3].set_title("Grad-CAM\n(superpuesto)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("saliency_comparacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: saliency_comparacion.png")
```

---

## 6. Caso práctico: explicar predicciones de un clasificador de texto

Integramos SHAP y LIME en un pipeline completo de clasificación de texto con reporte de explicabilidad por instancia.

```python
import shap
import lime.lime_text
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────
# 1. Dataset: 20 Newsgroups (4 categorías)
# ─────────────────────────────────────────────
categorias = [
    "sci.med",
    "sci.space",
    "talk.politics.guns",
    "talk.religion.misc"
]

datos = fetch_20newsgroups(
    subset="all",
    categories=categorias,
    remove=("headers", "footers", "quotes"),  # remover metadatos para hacer la tarea más difícil
    random_state=42
)

X_raw = datos.data
y_labels = datos.target
nombres_clases = datos.target_names

# Solo primeras 2000 muestras para velocidad
X_raw = X_raw[:2000]
y_labels = y_labels[:2000]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# ─────────────────────────────────────────────
# 2. Pipeline TF-IDF + Regresión Logística
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5.0,
        multi_class="multinomial",
        random_state=42
    ))
])

pipeline.fit(X_train_raw, y_train)
y_pred = pipeline.predict(X_test_raw)

print("=== RENDIMIENTO DEL CLASIFICADOR ===")
print(classification_report(
    y_test, y_pred,
    target_names=nombres_clases
))

# ─────────────────────────────────────────────
# 3. Explicabilidad con SHAP
# ─────────────────────────────────────────────
# Para pipelines sklearn, PartitionExplainer funciona bien
# Usamos una muestra del test para velocidad
muestra_test = X_test_raw[:50]

explainer_pipeline = shap.Explainer(
    pipeline.predict_proba,
    masker=shap.maskers.Text(tokenizer=r"\W+")  # tokenizar por no-palabras
)

# Calcular SHAP values
shap_vals = explainer_pipeline(muestra_test[:10])  # solo 10 para demo rápida

print("\nSHAP values calculados.")
print(f"Shape: {shap_vals.shape}")  # (n_textos, n_tokens, n_clases)

# ─────────────────────────────────────────────
# 4. Explicabilidad con LIME + reporte por instancia
# ─────────────────────────────────────────────
explainer_lime = lime.lime_text.LimeTextExplainer(
    class_names=nombres_clases,
    random_state=42
)

def generar_reporte_instancia(
    texto: str,
    pipeline: Pipeline,
    explainer_lime: lime.lime_text.LimeTextExplainer,
    nombres_clases: list,
    n_features: int = 8,
    num_samples: int = 500
) -> dict:
    """
    Genera un reporte de interpretabilidad completo para una instancia.
    
    Returns:
        dict con predicción, probabilidades y explicación LIME.
    """
    pred_prob = pipeline.predict_proba([texto])[0]
    pred_clase_idx = pred_prob.argmax()
    pred_clase = nombres_clases[pred_clase_idx]

    # LIME explanation
    exp = explainer_lime.explain_instance(
        texto,
        pipeline.predict_proba,
        num_features=n_features,
        num_samples=num_samples,
        labels=[pred_clase_idx]
    )

    palabras_positivas = [
        (w, p) for w, p in exp.as_list(label=pred_clase_idx) if p > 0
    ]
    palabras_negativas = [
        (w, p) for w, p in exp.as_list(label=pred_clase_idx) if p < 0
    ]

    return {
        "texto_preview": texto[:200] + "..." if len(texto) > 200 else texto,
        "prediccion": pred_clase,
        "confianza": pred_prob[pred_clase_idx],
        "probabilidades": dict(zip(nombres_clases, pred_prob)),
        "palabras_a_favor": palabras_positivas,
        "palabras_en_contra": palabras_negativas,
        "fidelidad_lime": exp.score
    }


# ─────────────────────────────────────────────
# 5. Generar reportes para instancias de prueba
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("REPORTES DE INTERPRETABILIDAD")
print("=" * 70)

for i in range(3):
    reporte = generar_reporte_instancia(
        X_test_raw[i], pipeline, explainer_lime, nombres_clases
    )

    print(f"\n--- Instancia {i} ---")
    print(f"Texto (preview): {reporte['texto_preview']}")
    print(f"Predicción:      {reporte['prediccion']} (confianza: {reporte['confianza']:.3f})")
    print(f"Clase real:      {nombres_clases[y_test[i]]}")
    print(f"Fidelidad LIME:  {reporte['fidelidad_lime']:.3f}")

    print("\nProbabilidades por clase:")
    for clase, prob in sorted(reporte["probabilidades"].items(),
                               key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 20)
        print(f"  {clase:<30} {prob:.3f} {barra}")

    if reporte["palabras_a_favor"]:
        print("\nPalabras que APOYAN la predicción:")
        for w, p in reporte["palabras_a_favor"][:5]:
            print(f"  '{w}': +{p:.4f}")

    if reporte["palabras_en_contra"]:
        print("Palabras que CONTRADICEN la predicción:")
        for w, p in reporte["palabras_en_contra"][:5]:
            print(f"  '{w}': {p:.4f}")

# ─────────────────────────────────────────────
# 6. Análisis de casos conflictivos
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANÁLISIS DE PREDICCIONES CON BAJA CONFIANZA")
print("=" * 70)

# Encontrar instancias donde el modelo está más inseguro
probs_test = pipeline.predict_proba(X_test_raw)
max_probs = probs_test.max(axis=1)
indices_bajos = np.argsort(max_probs)[:5]  # 5 instancias con menor confianza

print(f"\nInstancias con menor confianza en el test set:")
for idx in indices_bajos:
    pred_clase_idx = probs_test[idx].argmax()
    print(f"  Idx {idx:4d}: predicción={nombres_clases[pred_clase_idx]:<30} "
          f"real={nombres_clases[y_test[idx]]:<30} "
          f"confianza={max_probs[idx]:.3f}")
```

---

## 7. Extensiones sugeridas

- **Integrated Gradients:** alternativa a vanilla gradient que soluciona el problema del saturamiento. Disponible en `captum` de PyTorch. Especialmente útil para embeddings.
- **SHAP interaction values:** cuantificar el efecto conjunto de pares de features, no solo efectos marginales. `shap.TreeExplainer` lo soporta con `shap_interaction_values()`.
- **Counterfactual explanations:** dado un input X que produce predicción Y no deseada, ¿cuál es el cambio mínimo en X para obtener Y'? La librería `alibi` implementa `CounterfactualProto` y `Wachter`.
- **Global surrogates:** entrenar un árbol de decisión interpretable sobre las predicciones de un modelo complejo para obtener una visión global aproximada. Se puede combinar con SHAP para validar que el árbol es fiel al modelo original.
- **Monitoreo de deriva de explicaciones:** los SHAP values de producción pueden cambiar con deriva de datos incluso si la accuracy se mantiene estable. Monitorear la distribución de `|SHAP|` por feature como señal de alerta temprana.

---

**Anterior:** [01 — Sesgos y Fairness](./01-sesgos-fairness.md) · **Siguiente:** [03 — Model Cards y Documentación Responsable](./03-model-cards-datasheets.md)
