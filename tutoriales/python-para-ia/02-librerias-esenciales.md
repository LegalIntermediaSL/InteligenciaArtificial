# 02 — Librerías esenciales para IA

> **Bloque:** Python para IA · **Nivel:** Introductorio-Intermedio · **Tiempo estimado:** 35 min

---

## Índice

1. [El stack de IA en Python](#1-el-stack-de-ia-en-python)
2. [NumPy — computación numérica](#2-numpy--computación-numérica)
3. [Pandas — manipulación de datos](#3-pandas--manipulación-de-datos)
4. [Matplotlib y Seaborn — visualización](#4-matplotlib-y-seaborn--visualización)
5. [scikit-learn — machine learning clásico](#5-scikit-learn--machine-learning-clásico)
6. [Hugging Face Transformers](#6-hugging-face-transformers)
7. [LangChain — orquestación de LLMs](#7-langchain--orquestación-de-llms)
8. [Otras librerías relevantes](#8-otras-librerías-relevantes)

---

## 1. El stack de IA en Python

```
DATOS
├── NumPy        — arrays numéricos y álgebra lineal
├── Pandas       — tablas de datos (DataFrames)
└── Pillow       — imágenes

VISUALIZACIÓN
├── Matplotlib   — gráficos base
├── Seaborn      — gráficos estadísticos
└── Plotly       — gráficos interactivos

ML CLÁSICO
└── scikit-learn — algoritmos de ML, preprocesado, evaluación

DEEP LEARNING
├── PyTorch      — framework de DL (domina en investigación)
├── TensorFlow   — framework de DL (domina en producción)
└── Keras        — API de alto nivel sobre TF/PyTorch

LLMs / NLP
├── Transformers (Hugging Face) — modelos preentrenados
├── LangChain    — orquestación de pipelines LLM
├── LlamaIndex   — RAG y conectores de datos
└── anthropic / openai — SDKs de las APIs

UTILIDADES
├── python-dotenv — variables de entorno
├── pydantic      — validación de datos
├── tqdm          — barras de progreso
└── requests      — peticiones HTTP
```

Instalación del stack básico:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tqdm requests python-dotenv
```

---

## 2. NumPy — computación numérica

NumPy proporciona el tipo de dato fundamental de la IA: el **array multidimensional** (ndarray). Es la base de casi todas las demás librerías.

### Arrays básicos

```python
import numpy as np

# Crear arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz 2x3
c = np.zeros((3, 4))                   # Matriz de ceros 3x4
d = np.ones((2, 3))                    # Matriz de unos
e = np.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
f = np.linspace(0, 1, 5)              # [0, 0.25, 0.5, 0.75, 1.0]
g = np.random.randn(3, 3)             # Matriz aleatoria normal

print(b.shape)   # (2, 3)
print(b.dtype)   # int64
print(b.ndim)    # 2
```

### Operaciones vectorizadas

```python
# Las operaciones se aplican elemento a elemento (sin bucles)
x = np.array([1.0, 2.0, 3.0, 4.0])

print(x * 2)         # [2. 4. 6. 8.]
print(x ** 2)        # [1. 4. 9. 16.]
print(np.sqrt(x))    # [1. 1.41 1.73 2.]
print(np.log(x))     # [0. 0.69 1.09 1.38]

# Operaciones entre arrays
y = np.array([10, 20, 30, 40])
print(x + y)         # [11. 22. 33. 44.]
print(np.dot(x, y))  # Producto escalar: 300.0
```

### Álgebra lineal (clave para embeddings)

```python
# Similitud coseno entre dos embeddings
def similitud_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

embedding_a = np.random.randn(1536)  # Dimensión típica de embeddings
embedding_b = np.random.randn(1536)
print(f"Similitud: {similitud_coseno(embedding_a, embedding_b):.4f}")

# Multiplicación de matrices
A = np.random.randn(4, 3)
B = np.random.randn(3, 5)
C = A @ B   # Resultado: (4, 5)

# Valores propios (usado en PCA)
valores, vectores = np.linalg.eig(np.cov(A.T))
```

### Indexado y slicing

```python
datos = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(datos[1, 2])      # 7 (fila 1, columna 2)
print(datos[:, 0])      # [1, 5, 9] (toda la columna 0)
print(datos[0, :])      # [1, 2, 3, 4] (toda la fila 0)
print(datos[1:, 2:])    # [[7, 8], [11, 12]]

# Indexado booleano (filtrar)
print(datos[datos > 6]) # [7, 8, 9, 10, 11, 12]
```

---

## 3. Pandas — manipulación de datos

Pandas es imprescindible para cargar, limpiar y explorar datasets antes de entrenar modelos.

### DataFrame básico

```python
import pandas as pd

# Crear DataFrame desde dict
df = pd.DataFrame({
    "texto": ["El servicio es bueno", "Muy mala experiencia", "Normal, sin más"],
    "sentimiento": ["positivo", "negativo", "neutro"],
    "score": [0.85, 0.12, 0.50]
})

print(df.head())           # Primeras filas
print(df.shape)            # (3, 3)
print(df.dtypes)           # Tipos de cada columna
print(df.describe())       # Estadísticas de columnas numéricas
```

### Cargar y guardar datos

```python
# CSV
df = pd.read_csv("datos.csv", encoding="utf-8")
df.to_csv("resultado.csv", index=False)

# Excel
df = pd.read_excel("datos.xlsx", sheet_name="Hoja1")

# JSON
df = pd.read_json("datos.json")

# Parquet (formato eficiente para grandes datasets)
df = pd.read_parquet("datos.parquet")
df.to_parquet("resultado.parquet")
```

### Operaciones esenciales

```python
# Selección
df["texto"]                     # Serie (una columna)
df[["texto", "score"]]          # DataFrame (varias columnas)
df.loc[0]                       # Fila por índice
df.iloc[0:2]                    # Filas por posición

# Filtrado
positivos = df[df["sentimiento"] == "positivo"]
alta_confianza = df[df["score"] > 0.7]
combinado = df[(df["sentimiento"] != "neutro") & (df["score"] > 0.6)]

# Ordenar
df_ordenado = df.sort_values("score", ascending=False)

# Agrupar
por_sentimiento = df.groupby("sentimiento")["score"].mean()
conteo = df["sentimiento"].value_counts()

# Aplicar función
df["texto_limpio"] = df["texto"].str.lower().str.strip()
df["longitud"] = df["texto"].apply(len)
df["tokens_est"] = df["texto"].apply(lambda t: len(t.split()))
```

### Limpieza de datos

```python
# Valores nulos
print(df.isnull().sum())          # Nulos por columna
df.dropna()                       # Eliminar filas con nulos
df.fillna({"score": 0.5})         # Rellenar nulos con valor

# Duplicados
df.drop_duplicates(subset=["texto"], inplace=True)

# Cambiar tipos
df["score"] = df["score"].astype(float)

# Resetear índice tras filtrar
df = df[df["score"] > 0.5].reset_index(drop=True)
```

---

## 4. Matplotlib y Seaborn — visualización

### Matplotlib (base)

```python
import matplotlib.pyplot as plt
import numpy as np

# Gráfico de líneas
x = np.linspace(0, 10, 100)
plt.figure(figsize=(10, 4))
plt.plot(x, np.sin(x), label="sin(x)", color="blue")
plt.plot(x, np.cos(x), label="cos(x)", color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Funciones trigonométricas")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("grafico.png", dpi=150)
plt.show()

# Histograma de scores
scores = np.random.beta(2, 5, 1000)
plt.figure(figsize=(8, 4))
plt.hist(scores, bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Score")
plt.ylabel("Frecuencia")
plt.title("Distribución de scores del modelo")
plt.show()
```

### Seaborn (estadístico)

```python
import seaborn as sns
import pandas as pd

df = pd.DataFrame({
    "modelo": ["claude", "gpt-4", "gemini", "llama"] * 50,
    "score": np.random.beta([8, 7, 6, 5], [2, 3, 4, 5], (50, 4)).flatten()
})

# Boxplot comparativo
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="modelo", y="score", palette="Set2")
plt.title("Comparativa de scores por modelo")
plt.show()

# Heatmap de correlaciones
corr = pd.DataFrame(np.random.randn(5, 5), columns=list("ABCDE")).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.show()
```

---

## 5. scikit-learn — machine learning clásico

scikit-learn ofrece algoritmos de ML listos para usar con una API consistente.

### Pipeline típico de ML

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Cargar datos
df = pd.read_csv("emails.csv")
X = df.drop("spam", axis=1)
y = df["spam"]

# 2. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocesar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transform, NO fit_transform

# 4. Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train)

# 5. Evaluar
y_pred = modelo.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

### Pipelines de scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Pipeline que encadena preprocesado + modelo
pipe = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train_texto, y_train)
y_pred = pipe.predict(X_test_texto)
```

### Métricas de evaluación

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)

# Clasificación
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1:        {f1_score(y_test, y_pred, average='weighted'):.3f}")

# Regresión
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

---

## 6. Hugging Face Transformers

La librería `transformers` de Hugging Face da acceso a miles de modelos preentrenados.

```bash
pip install transformers torch
```

### Pipelines de alto nivel

```python
from transformers import pipeline

# Clasificación de sentimiento
clasificador = pipeline("sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment")
resultado = clasificador("Este producto es fantástico")
print(resultado)  # [{'label': '5 stars', 'score': 0.89}]

# Resumen automático
resumidor = pipeline("summarization", model="facebook/bart-large-cnn")
texto_largo = "..." # texto a resumir
resumen = resumidor(texto_largo, max_length=150, min_length=50)
print(resumen[0]["summary_text"])

# Traducción
traductor = pipeline("translation_es_to_en",
                     model="Helsinki-NLP/opus-mt-es-en")
traduccion = traductor("La inteligencia artificial es fascinante")
print(traduccion[0]["translation_text"])

# Generación de embeddings
from transformers import AutoTokenizer, AutoModel
import torch

modelo_nombre = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
modelo = AutoModel.from_pretrained(modelo_nombre)

def obtener_embedding(texto: str) -> list:
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

vec = obtener_embedding("La IA transforma los negocios")
print(f"Dimensión del embedding: {len(vec)}")
```

---

## 7. LangChain — orquestación de LLMs

LangChain facilita la construcción de aplicaciones complejas con LLMs: cadenas, agentes y RAG.

```bash
pip install langchain langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Modelo
llm = ChatAnthropic(model="claude-sonnet-4-6")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {dominio}. Responde en español."),
    ("human", "{pregunta}")
])

# Chain (encadenar prompt + modelo)
chain = prompt | llm

respuesta = chain.invoke({
    "dominio": "machine learning",
    "pregunta": "¿Qué es el overfitting?"
})
print(respuesta.content)
```

---

## 8. Otras librerías relevantes

| Librería | Uso | Instalación |
|---|---|---|
| `tqdm` | Barras de progreso | `pip install tqdm` |
| `pydantic` | Validación de datos y modelos | `pip install pydantic` |
| `requests` | Llamadas HTTP | `pip install requests` |
| `httpx` | HTTP asíncrono | `pip install httpx` |
| `faiss-cpu` | Búsqueda vectorial eficiente (Meta) | `pip install faiss-cpu` |
| `chromadb` | Base de datos vectorial para RAG | `pip install chromadb` |
| `pypdf` / `pymupdf` | Leer PDFs | `pip install pypdf` |
| `python-docx` | Leer/escribir Word | `pip install python-docx` |
| `Pillow` | Procesamiento de imágenes | `pip install Pillow` |
| `opencv-python` | Visión por computador | `pip install opencv-python` |

```python
# Ejemplo tqdm — barra de progreso en bucles
from tqdm import tqdm
import time

textos = ["texto1", "texto2", "texto3"] * 100

resultados = []
for texto in tqdm(textos, desc="Procesando"):
    # Simular llamada a API
    time.sleep(0.01)
    resultados.append(texto.upper())
```

---

**Anterior:** [01 — Intro a Python para IA](./01-intro-python-ia.md) · **Siguiente:** [03 — Jupyter Notebooks](./03-jupyter-notebooks.md)
