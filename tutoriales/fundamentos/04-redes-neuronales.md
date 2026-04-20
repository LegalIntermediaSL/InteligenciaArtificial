# Redes Neuronales — De la neurona al perceptrón multicapa

Las redes neuronales artificiales son la base de toda la IA moderna.
Entender cómo funcionan explica por qué los LLMs hacen lo que hacen.

---

## 1. La neurona artificial

Una neurona artificial es una función matemática que toma múltiples entradas,
las pondera y produce una salida:

```
entrada₁ × peso₁ ┐
entrada₂ × peso₂ ├─→ [SUMA + SESGO] → [ACTIVACIÓN] → salida
entrada₃ × peso₃ ┘
```

```python
import numpy as np

def neurona(entradas: np.ndarray, pesos: np.ndarray, sesgo: float) -> float:
    suma = np.dot(entradas, pesos) + sesgo
    return suma  # sin activación = regresión lineal

# Neurona con activación ReLU (Rectified Linear Unit)
def relu(x: float) -> float:
    return max(0.0, x)

# Ejemplo: clasificar si una temperatura es "caliente"
entradas = np.array([37.5])   # temperatura en °C
pesos = np.array([1.0])       # peso (importancia)
sesgo = -36.5                 # umbral de "caliente"

suma = neurona(entradas, pesos, sesgo)
activacion = relu(suma)

print(f'Temperatura: {entradas[0]}°C')
print(f'Suma ponderada: {suma}')
print(f'Activación ReLU: {activacion}')
print(f'Decisión: {"caliente" if activacion > 0 else "normal"}')
```

---

## 2. Funciones de activación

La función de activación introduce **no-linealidad** — sin ella, una red neuronal
de 100 capas sería equivalente a una sola capa lineal.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

activaciones = {
    'ReLU':    np.maximum(0, x),
    'Sigmoid': 1 / (1 + np.exp(-x)),
    'Tanh':    np.tanh(x),
    'GELU':    x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) / 2,
}

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, (nombre, valores) in zip(axes.flat, activaciones.items()):
    ax.plot(x, valores, linewidth=2, color='#2196F3')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(nombre, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)

plt.suptitle('Funciones de activación más comunes', fontsize=13)
plt.tight_layout()
plt.show()

print('Usos en la práctica:')
print('  ReLU:    Capas ocultas de CNNs y redes clásicas')
print('  GELU:    Transformers (GPT, BERT, Claude)')
print('  Sigmoid: Clasificación binaria (capa final)')
print('  Tanh:    RNNs y LSTMs')
```

---

## 3. Perceptrón multicapa (MLP)

```python
import numpy as np

class RedNeuronal:
    """MLP simple con una capa oculta — implementación desde cero."""

    def __init__(self, n_entrada: int, n_oculta: int, n_salida: int):
        # Pesos inicializados con distribución Xavier
        self.W1 = np.random.randn(n_entrada, n_oculta) * np.sqrt(2 / n_entrada)
        self.b1 = np.zeros(n_oculta)
        self.W2 = np.random.randn(n_oculta, n_salida) * np.sqrt(2 / n_oculta)
        self.b2 = np.zeros(n_salida)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def adelante(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagación hacia adelante."""
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return a1, a2

    def predecir(self, x: np.ndarray) -> int:
        _, prob = self.adelante(x)
        return np.argmax(prob)

# Clasificador de tipo de texto (demo)
red = RedNeuronal(n_entrada=5, n_oculta=8, n_salida=3)

# Características de ejemplo: [longitud, n_preguntas, n_palabras_negativas, formal, urgente]
textos_ejemplo = [
    (np.array([0.2, 0.8, 0.1, 0.9, 0.1]), 'consulta_legal'),    # consulta formal
    (np.array([0.9, 0.1, 0.8, 0.1, 0.9]), 'queja'),              # queja urgente
    (np.array([0.5, 0.3, 0.2, 0.5, 0.3]), 'informacion'),        # petición info
]

clases = ['consulta_legal', 'queja', 'informacion']
print('Predicciones (pesos aleatorios — no entrenados):')
for features, etiqueta_real in textos_ejemplo:
    prediccion = red.predecir(features)
    _, probs = red.adelante(features)
    print(f'  Real: {etiqueta_real:<15} Predicho: {clases[prediccion]:<15} Confianza: {max(probs):.1%}')
```

---

## 4. Backpropagation — cómo aprende una red neuronal

El aprendizaje ocurre ajustando los pesos para minimizar el error.
El algoritmo de **backpropagation** calcula cómo afecta cada peso al error total.

```python
import numpy as np

# Implementación simplificada de entrenamiento
def entrenar_epoch(red, X: np.ndarray, y: np.ndarray, tasa_aprendizaje: float = 0.01) -> float:
    """Una pasada completa de entrenamiento con gradiente descendente."""
    perdida_total = 0.0

    for xi, yi in zip(X, y):
        # 1. Propagación hacia adelante
        a1, a2 = red.adelante(xi)

        # 2. Calcular pérdida (cross-entropy)
        y_one_hot = np.eye(3)[yi]
        perdida = -np.sum(y_one_hot * np.log(a2 + 1e-8))
        perdida_total += perdida

        # 3. Backpropagation (gradientes)
        dL_da2 = a2 - y_one_hot                         # gradiente capa salida
        dL_dW2 = np.outer(a1, dL_da2)                   # gradiente pesos W2
        dL_db2 = dL_da2

        dL_da1 = dL_da2 @ red.W2.T                      # gradiente capa oculta
        dL_da1[a1 <= 0] = 0                              # gradiente de ReLU

        dL_dW1 = np.outer(xi, dL_da1)
        dL_db1 = dL_da1

        # 4. Actualizar pesos (gradiente descendente)
        red.W2 -= tasa_aprendizaje * dL_dW2
        red.b2 -= tasa_aprendizaje * dL_db2
        red.W1 -= tasa_aprendizaje * dL_dW1
        red.b1 -= tasa_aprendizaje * dL_db1

    return perdida_total / len(X)

# Datos de entrenamiento sintéticos
np.random.seed(42)
X_train = np.random.randn(150, 5)
# Clase 0: alta formalidad, clase 1: alta negatividad, clase 2: resto
y_train = np.where(X_train[:, 3] > 0.5, 0, np.where(X_train[:, 2] > 0.3, 1, 2))

red = RedNeuronal(5, 16, 3)
perdidas = []

for epoch in range(50):
    perdida = entrenar_epoch(red, X_train, y_train, tasa_aprendizaje=0.05)
    perdidas.append(perdida)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d}: pérdida = {perdida:.4f}')

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(perdidas, color='#4CAF50', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Pérdida (cross-entropy)')
plt.title('Curva de entrenamiento')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5. De MLP a LLMs: la escala cambia todo

```python
# Comparativa de tamaño para entender la escala
modelos = {
    'MLP (ejemplo anterior)': {
        'parametros': 5*16 + 16 + 16*3 + 3,  # W1 + b1 + W2 + b2
        'fecha': '1986',
        'tarea': 'Clasificación simple',
    },
    'LeNet-5 (CNN)': {
        'parametros': 60_000,
        'fecha': '1998',
        'tarea': 'Reconocimiento de dígitos escritos a mano',
    },
    'ResNet-50': {
        'parametros': 25_000_000,
        'fecha': '2015',
        'tarea': 'Clasificación de imágenes (1000 clases)',
    },
    'GPT-2': {
        'parametros': 1_500_000_000,
        'fecha': '2019',
        'tarea': 'Generación de texto',
    },
    'GPT-3': {
        'parametros': 175_000_000_000,
        'fecha': '2020',
        'tarea': 'Generación, razonamiento, código',
    },
    'Claude 3.5 Sonnet (estimado)': {
        'parametros': 200_000_000_000,
        'fecha': '2024',
        'tarea': 'Razonamiento, código, multimodal',
    },
}

print(f'{"Modelo":<35} {"Parámetros":>20} {"Año":>6}')
print('-' * 65)
for nombre, info in modelos.items():
    p = info['parametros']
    if p >= 1e12:
        p_str = f'{p/1e12:.1f}T'
    elif p >= 1e9:
        p_str = f'{p/1e9:.1f}B'
    elif p >= 1e6:
        p_str = f'{p/1e6:.1f}M'
    elif p >= 1e3:
        p_str = f'{p/1e3:.1f}K'
    else:
        p_str = str(p)
    print(f'{nombre:<35} {p_str:>20} {info["fecha"]:>6}')

print('\nLos LLMs son MLPs enormes con la arquitectura Transformer.')
print('La diferencia clave: mecanismo de atención (ver artículo 05-transformers).')
```

---

## 6. Redes neuronales en producción con PyTorch

```python
import torch
import torch.nn as nn

# El mismo clasificador de texto, ahora con PyTorch
class ClasificadorTexto(nn.Module):
    def __init__(self, n_entrada: int, n_oculta: int, n_clases: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_entrada, n_oculta),
            nn.ReLU(),
            nn.Dropout(0.2),          # regularización
            nn.Linear(n_oculta, n_oculta // 2),
            nn.ReLU(),
            nn.Linear(n_oculta // 2, n_clases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)

modelo = ClasificadorTexto(n_entrada=768, n_oculta=256, n_clases=5)
print(f'Parámetros del modelo: {sum(p.numel() for p in modelo.parameters()):,}')

# Inferencia
with torch.no_grad():
    x_ejemplo = torch.randn(1, 768)   # embedding de 768 dimensiones (BERT/Anthropic)
    logits = modelo(x_ejemplo)
    probs = torch.softmax(logits, dim=-1)
    print(f'Probabilidades: {probs.numpy().round(3)}')
    print(f'Clase predicha: {probs.argmax().item()}')
```

---

→ Siguiente: [Transformers y mecanismo de atención](05-transformers-atencion.md)
