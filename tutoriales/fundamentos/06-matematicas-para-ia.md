# Matemáticas Esenciales para IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/fundamentos/06-matematicas-para-ia.ipynb)

Las matemáticas que hay detrás de la IA se reducen a tres pilares: álgebra lineal, probabilidad y cálculo.
Este artículo los explica desde cero con NumPy y los conecta con aplicaciones concretas en LLMs.

---

## 1. Álgebra lineal

### Vectores

Un vector es una lista ordenada de números. En IA representa cualquier cosa: una palabra, una imagen, un documento.

```python
import numpy as np

# Embeddings simplificados de dos palabras
gato  = np.array([0.9, 0.1, 0.8, 0.2])
perro = np.array([0.8, 0.2, 0.7, 0.3])
coche = np.array([0.1, 0.9, 0.2, 0.8])

# Dot product: mide cuánto se "alinean" dos vectores
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# Norma L2: "longitud" del vector
def norma(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

# Cosine similarity: similitud independiente de la magnitud
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot_product(a, b) / (norma(a) * norma(b))

print("Dot products:")
print(f"  gato · perro = {dot_product(gato, perro):.3f}")
print(f"  gato · coche = {dot_product(gato, coche):.3f}")

print("\nCosine similarities (rango -1 a 1, mayor = más similar):")
print(f"  gato ↔ perro = {cosine_similarity(gato, perro):.3f}")
print(f"  gato ↔ coche = {cosine_similarity(gato, coche):.3f}")

# En LLMs: cosine similarity es la base de la búsqueda semántica
# Dos chunks de texto se comparan por la similitud de sus embeddings
```

### Matrices y operaciones esenciales

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])    # shape (3, 2)

# Multiplicación de matrices: (2,3) × (3,2) → (2,2)
C = A @ B
print("A @ B =")
print(C)

# Transpuesta
print(f"\nA.T shape: {A.T.shape}")
print(A.T)

# Inversa (solo matrices cuadradas)
M = np.array([[2.0, 1.0],
              [1.0, 3.0]])
M_inv = np.linalg.inv(M)
print(f"\nM_inv @ M (debería ser identidad):")
print((M_inv @ M).round(10))

# En los Transformers: Q @ K.T es multiplicación de matrices
# Las matrices de pesos W_Q, W_K, W_V son matrices que se aprenden
```

### Valores y vectores propios

```python
import numpy as np

# Matriz de covarianza de 3 features
cov = np.array([[3.0, 1.5, 0.5],
                [1.5, 2.0, 0.3],
                [0.5, 0.3, 1.0]])

eigenvalues, eigenvectors = np.linalg.eig(cov)

# Ordenar por importancia (mayor eigenvalue = más varianza explicada)
orden = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[orden]
eigenvectors = eigenvectors[:, orden]

varianza_total = eigenvalues.sum()
print("Eigenvalues y varianza explicada:")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    pct = val / varianza_total * 100
    print(f"  PC{i+1}: {val:.3f} ({pct:.1f}%) — vector: {vec.round(3)}")

# En LLMs: PCA es la forma conceptual de entender por qué los embeddings
# de alta dimensión capturan semántica — cada dirección principal
# corresponde a una "dimensión semántica" aprendida
```

---

## 2. Probabilidad y estadística

### Distribuciones fundamentales

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Normal (Gaussiana) ---
mu, sigma = 0.0, 1.0
muestras_normal = rng.normal(mu, sigma, 5000)

# --- Bernoulli (clasificación binaria) ---
p = 0.7
muestras_bernoulli = rng.binomial(1, p, 1000)
print(f"Bernoulli(p=0.7): media empírica = {muestras_bernoulli.mean():.3f}")

# --- Softmax (distribución de probabilidad sobre clases) ---
def softmax(logits: np.ndarray) -> np.ndarray:
    # Restar el máximo para estabilidad numérica
    e = np.exp(logits - logits.max())
    return e / e.sum()

logits = np.array([2.1, 0.5, -0.3, 1.8, 0.9])
probs  = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax: {probs.round(4)}")
print(f"Suma:    {probs.sum():.6f}")

# En LLMs: la última capa aplica softmax sobre el vocabulario (~100k tokens)
# El token con mayor probabilidad es el que se genera en modo greedy
```

### Regla de Bayes

```python
import numpy as np

# Clasificación de spam con Naive Bayes
# P(spam | palabra) = P(palabra | spam) * P(spam) / P(palabra)

p_spam  = 0.30   # prior: 30% de los correos son spam
p_ham   = 0.70

# Verosimilitudes: ¿con qué frecuencia aparece "oferta" en cada clase?
p_oferta_dado_spam = 0.80
p_oferta_dado_ham  = 0.05

# Regla de Bayes
p_oferta = p_oferta_dado_spam * p_spam + p_oferta_dado_ham * p_ham
p_spam_dado_oferta = (p_oferta_dado_spam * p_spam) / p_oferta

print("Clasificación de spam — Regla de Bayes:")
print(f"  P(spam)              = {p_spam:.2f}")
print(f"  P('oferta' | spam)   = {p_oferta_dado_spam:.2f}")
print(f"  P('oferta' | ham)    = {p_oferta_dado_ham:.2f}")
print(f"  P(spam | 'oferta')   = {p_spam_dado_oferta:.3f}")

# En LLMs: el preentrenamiento maximiza P(token_t | token_1...token_{t-1})
# — es decir, aprende la distribución condicional de cada token
```

### Entropía y KL divergence

```python
import numpy as np

def entropia(p: np.ndarray, eps: float = 1e-12) -> float:
    """Entropía de Shannon en bits."""
    p = p + eps
    return float(-np.sum(p * np.log2(p)))

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q): cuánto difiere Q de la distribución "real" P."""
    p, q = p + eps, q + eps
    return float(np.sum(p * np.log(p / q)))

# Distribución uniforme (máxima incertidumbre) vs concentrada
uniforme    = np.array([0.25, 0.25, 0.25, 0.25])
concentrada = np.array([0.97, 0.01, 0.01, 0.01])
media       = np.array([0.40, 0.30, 0.20, 0.10])

print("Entropía (bits):")
print(f"  Uniforme:    {entropia(uniforme):.3f}")
print(f"  Concentrada: {entropia(concentrada):.3f}")
print(f"  Media:       {entropia(media):.3f}")

print("\nKL divergence (KL(media || uniforme)):")
print(f"  {kl_divergence(media, uniforme):.4f}")
print(f"  (0 = distribuciones idénticas)")

# En LLMs: el fine-tuning con RLHF minimiza la KL divergence entre
# la política actual y el modelo base para no alejarse demasiado
```

---

## 3. Cálculo para IA

### Gradiente y derivadas parciales

```python
import numpy as np

def funcion_perdida(w: float, b: float, x: float, y_real: float) -> float:
    """MSE para un solo ejemplo."""
    y_pred = w * x + b
    return (y_pred - y_real) ** 2

# Derivadas parciales calculadas analíticamente
def gradiente(w: float, b: float, x: float, y_real: float) -> tuple[float, float]:
    y_pred = w * x + b
    error  = y_pred - y_real
    dL_dw = 2 * error * x   # ∂L/∂w
    dL_db = 2 * error       # ∂L/∂b
    return dL_dw, dL_db

# Verificación numérica del gradiente (diferencias finitas)
h = 1e-5
w, b, x, y_real = 0.5, 0.1, 2.0, 3.0

grad_w_analitico, grad_b_analitico = gradiente(w, b, x, y_real)
grad_w_numerico = (funcion_perdida(w + h, b, x, y_real) - funcion_perdida(w - h, b, x, y_real)) / (2 * h)

print(f"∂L/∂w analítico: {grad_w_analitico:.6f}")
print(f"∂L/∂w numérico:  {grad_w_numerico:.6f}")
print(f"Error relativo:  {abs(grad_w_analitico - grad_w_numerico) / abs(grad_w_numerico):.2e}")
```

### Backpropagation en 1 neurona — ejemplo completo

```python
import numpy as np

# Red de 1 neurona: x → [w,b] → z → sigmoid → ŷ → loss (BCE)
# Forward pass
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def bce_loss(y_hat: float, y: float, eps: float = 1e-12) -> float:
    return -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

# Parámetros y dato
w, b = 0.8, -0.2
x, y = 1.5, 1.0    # ejemplo positivo

# --- Forward ---
z     = w * x + b
y_hat = sigmoid(z)
loss  = bce_loss(y_hat, y)

print("=== Forward pass ===")
print(f"  z     = w*x + b = {w}*{x} + {b} = {z:.4f}")
print(f"  y_hat = σ(z)    = {y_hat:.4f}")
print(f"  loss  = BCE     = {loss:.4f}")

# --- Backward (regla de la cadena) ---
# ∂loss/∂y_hat
dL_dyhat = -(y / y_hat) + (1 - y) / (1 - y_hat)

# ∂y_hat/∂z = σ(z) * (1 - σ(z))
dyhat_dz = y_hat * (1 - y_hat)

# ∂z/∂w = x, ∂z/∂b = 1
dz_dw = x
dz_db = 1.0

# Regla de la cadena
dL_dw = dL_dyhat * dyhat_dz * dz_dw
dL_db = dL_dyhat * dyhat_dz * dz_db

print("\n=== Backward pass ===")
print(f"  ∂L/∂ŷ  = {dL_dyhat:.4f}")
print(f"  ∂ŷ/∂z  = {dyhat_dz:.4f}")
print(f"  ∂L/∂w  = {dL_dw:.4f}")
print(f"  ∂L/∂b  = {dL_db:.4f}")

# --- Gradient descent step ---
lr = 0.1
w_nuevo = w - lr * dL_dw
b_nuevo = b - lr * dL_db

loss_nueva = bce_loss(sigmoid(w_nuevo * x + b_nuevo), y)
print(f"\n=== Después de un paso (lr={lr}) ===")
print(f"  w: {w:.4f} → {w_nuevo:.4f}")
print(f"  b: {b:.4f} → {b_nuevo:.4f}")
print(f"  loss: {loss:.4f} → {loss_nueva:.4f}  ({'↓ mejoró' if loss_nueva < loss else '↑ empeoró'})")
```

---

## 4. Por qué importa: conexión con LLMs

| Concepto matemático | Dónde aparece en LLMs |
|---|---|
| **Cosine similarity** | Búsqueda semántica en RAG; comparar embeddings de chunks |
| **Multiplicación de matrices** | Cada capa del Transformer: `Q @ K.T`, proyecciones `W_Q`, `W_V` |
| **Eigenvalores / PCA** | Interpretabilidad: encontrar "direcciones semánticas" en el espacio de embeddings |
| **Softmax** | Capa final: convierte logits en probabilidades sobre el vocabulario |
| **Entropía** | Temperatura del muestreo: temp alta → distribución más uniforme → más creatividad |
| **KL divergence** | RLHF y fine-tuning: evitar que el modelo se aleje demasiado del modelo base |
| **Gradiente** | Entrenamiento: actualiza ~70B parámetros en cada paso de backprop |
| **Regla de la cadena** | Backpropagation a través de decenas de capas anidadas |

```python
# Resumen visual: los 3 pilares y sus aplicaciones
pilares = {
    "Álgebra lineal": [
        "Embeddings como vectores de alta dimensión",
        "Atención: Q @ K.T / sqrt(d_k)",
        "Proyecciones lineales en cada capa",
    ],
    "Probabilidad": [
        "LLM = distribución P(token_t | contexto)",
        "Softmax sobre vocabulario en cada paso",
        "RLHF minimiza KL divergence del policy",
    ],
    "Cálculo": [
        "Backprop calcula gradientes por regla de cadena",
        "Adam optimizer usa gradientes de 1er y 2o orden",
        "Fine-tuning = gradient descent sobre tus datos",
    ],
}

for pilar, apps in pilares.items():
    print(f"\n{pilar}:")
    for app in apps:
        print(f"  • {app}")
```

---

→ Anterior: [Transformers y atención](05-transformers-atencion.md) | → Siguiente: [Bloque 2 — LLMs](../llms/01-que-es-un-llm.md)
