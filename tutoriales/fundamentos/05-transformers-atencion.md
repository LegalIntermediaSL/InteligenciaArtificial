# Transformers y mecanismo de atención

La arquitectura Transformer (2017) es la base de todos los LLMs modernos.
Este artículo explica el mecanismo de atención desde primeros principios.

---

## 1. El problema que resuelve el Transformer

Antes de los Transformers, los modelos de lenguaje usaban RNNs/LSTMs que procesaban
texto de forma **secuencial** — token por token. Esto tenía dos problemas:

1. **Memoria limitada**: el modelo "olvidaba" información de tokens lejanos
2. **No paralelizable**: cada token dependía del anterior → entrenamiento lento

El Transformer soluciona ambos con el **mecanismo de atención**: cada token puede
"prestar atención" a cualquier otro token en la secuencia, independientemente de la distancia.

---

## 2. Embeddings — convertir palabras en vectores

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Cada palabra se representa como un vector de alta dimensión
# En la práctica: 768 (BERT), 1536 (GPT-3), 4096 (LLaMA-2-7B)
DIMENSION = 8  # pequeña para visualizar

# Vocabulario de ejemplo
VOCABULARIO = {
    'contrato': 0, 'cláusula': 1, 'precio': 2, 'vencimiento': 3,
    'legal': 4, 'empresa': 5, 'proveedor': 6, 'cliente': 7,
    'tecnología': 8, 'software': 9, 'API': 10, 'datos': 11,
}

np.random.seed(42)
# Embedding matrix: cada fila es el vector de una palabra
matriz_embeddings = np.random.randn(len(VOCABULARIO), DIMENSION)

# Normalizar para que las similitudes sean comparables
norms = np.linalg.norm(matriz_embeddings, axis=1, keepdims=True)
matriz_embeddings = matriz_embeddings / norms

def similitud(palabra1: str, palabra2: str) -> float:
    v1 = matriz_embeddings[VOCABULARIO[palabra1]]
    v2 = matriz_embeddings[VOCABULARIO[palabra2]]
    return float(np.dot(v1, v2))

# En un modelo real los embeddings aprendidos muestran agrupaciones semánticas
print('Similitudes entre palabras (embeddings aleatorios — sin entrenar):')
pares = [
    ('contrato', 'cláusula'),
    ('contrato', 'precio'),
    ('empresa', 'proveedor'),
    ('empresa', 'tecnología'),
    ('legal', 'cláusula'),
]
for p1, p2 in pares:
    sim = similitud(p1, p2)
    print(f'  {p1} ↔ {p2}: {sim:.3f}')
```

---

## 3. El mecanismo de atención — Query, Key, Value

La atención funciona como un sistema de búsqueda:
- **Query (Q)**: "¿qué estoy buscando?" — cada token pregunta qué información necesita
- **Key (K)**: "¿qué tengo yo?" — cada token anuncia qué información puede ofrecer
- **Value (V)**: "¿cuál es mi valor?" — la información real que se transfiere

```python
import numpy as np

def atencion(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention.
    Q, K, V: matrices [n_tokens, d_modelo]
    Devuelve: (output, pesos_atencion)
    """
    d_k = K.shape[-1]

    # 1. Calcular puntuaciones de similitud
    scores = Q @ K.T / np.sqrt(d_k)      # escalar para estabilidad numérica

    # 2. Softmax → pesos que suman 1
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    pesos = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    # 3. Suma ponderada de valores
    output = pesos @ V

    return output, pesos

# Ejemplo: frase de 4 tokens "el contrato legal vence"
np.random.seed(0)
n_tokens = 4
d_modelo = 6

# Matrices Q, K, V (en la práctica se aprenden durante el entrenamiento)
Q = np.random.randn(n_tokens, d_modelo)
K = np.random.randn(n_tokens, d_modelo)
V = np.random.randn(n_tokens, d_modelo)

output, pesos = atencion(Q, K, V)

tokens = ['el', 'contrato', 'legal', 'vence']
print('Pesos de atención (cada fila muestra a qué presta atención cada token):')
print(f'{"":8}', end='')
for t in tokens:
    print(f'{t:>10}', end='')
print()
for i, fila in enumerate(pesos):
    print(f'{tokens[i]:8}', end='')
    for p in fila:
        print(f'{p:>10.3f}', end='')
    print()

print(f'\nOutput shape: {output.shape} (misma que input: [{n_tokens}, {d_modelo}])')
```

---

## 4. Atención multi-cabeza

En lugar de una sola atención, los Transformers usan **múltiples cabezas** en paralelo.
Cada cabeza aprende a prestar atención a aspectos diferentes del texto:

```python
import numpy as np

class AtencionMultiCabeza:
    def __init__(self, d_modelo: int, n_cabezas: int):
        assert d_modelo % n_cabezas == 0
        self.d_modelo = d_modelo
        self.n_cabezas = n_cabezas
        self.d_k = d_modelo // n_cabezas

        # Matrices de proyección (se aprenden durante entrenamiento)
        np.random.seed(42)
        self.W_Q = np.random.randn(d_modelo, d_modelo) * 0.1
        self.W_K = np.random.randn(d_modelo, d_modelo) * 0.1
        self.W_V = np.random.randn(d_modelo, d_modelo) * 0.1
        self.W_O = np.random.randn(d_modelo, d_modelo) * 0.1

    def atencion_individual(self, Q, K, V):
        scores = Q @ K.T / np.sqrt(self.d_k)
        exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
        pesos = exp_s / exp_s.sum(axis=-1, keepdims=True)
        return pesos @ V, pesos

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        n_tokens = x.shape[0]

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        outputs = []
        todos_pesos = []

        # Procesar cada cabeza por separado
        for i in range(self.n_cabezas):
            inicio = i * self.d_k
            fin = (i + 1) * self.d_k
            output_i, pesos_i = self.atencion_individual(
                Q[:, inicio:fin],
                K[:, inicio:fin],
                V[:, inicio:fin],
            )
            outputs.append(output_i)
            todos_pesos.append(pesos_i)

        # Concatenar y proyectar
        concatenado = np.concatenate(outputs, axis=-1)
        return concatenado @ self.W_O, todos_pesos

# Demo
n_tokens, d_modelo, n_cabezas = 6, 8, 2
x = np.random.randn(n_tokens, d_modelo)

mha = AtencionMultiCabeza(d_modelo=d_modelo, n_cabezas=n_cabezas)
output, pesos = mha.forward(x)

print(f'Entrada: [{n_tokens}, {d_modelo}]')
print(f'Salida:  {output.shape}')
print(f'Cabezas: {n_cabezas}')
print(f'D por cabeza: {d_modelo // n_cabezas}')
print(f'\nPesos cabeza 1 (primera fila = token 0 prestando atención a todos):')
print(pesos[0][0].round(3))
print(f'\nPesos cabeza 2 (misma fila, diferentes pesos = diferente aspecto aprendido):')
print(pesos[1][0].round(3))
```

---

## 5. Arquitectura completa del Transformer

```
Entrada (texto) → Tokenización → Embeddings + Positional Encoding
                                       ↓
                              ┌─── Encoder Block (× N) ───┐
                              │  Multi-Head Attention       │
                              │  Add & LayerNorm            │
                              │  Feed-Forward Network       │
                              │  Add & LayerNorm            │
                              └────────────────────────────┘
                                       ↓
                                  Representación
                                       ↓
                              ┌─── Decoder Block (× N) ───┐  ← (solo en modelos seq2seq)
                              │  Masked Multi-Head Attn    │
                              │  Cross-Attention           │
                              │  Feed-Forward              │
                              └────────────────────────────┘
                                       ↓
                              Linear + Softmax → Probabilidades
                                       ↓
                                  Token generado
```

```python
# Los LLMs modernos como Claude/GPT usan solo el Decoder (autoregresivo)
# Cada token generado se añade a la entrada para generar el siguiente

def ilustrar_generacion_autoregresiva(prompt: str, n_tokens: int = 5) -> list[str]:
    """Ilustra cómo un LLM genera tokens uno a uno."""
    tokens_generados = prompt.split()
    vocabulario_demo = ['contrato', 'válido', 'hasta', 'fecha', 'límite',
                        'el', 'la', 'es', 'de', 'en', 'y', 'con', '.']
    import random
    random.seed(42)

    for paso in range(n_tokens):
        contexto_actual = ' '.join(tokens_generados)
        nuevo_token = random.choice(vocabulario_demo)  # en la práctica: softmax sobre el vocabulario
        tokens_generados.append(nuevo_token)
        print(f'Paso {paso + 1}: "{contexto_actual}" → genera "{nuevo_token}"')

    return tokens_generados

print('Generación autoregresiva (tokens aleatorios — demo):')
tokens_finales = ilustrar_generacion_autoregresiva('El contrato')
print(f'\nTexto completo: "{" ".join(tokens_finales)}"')
```

---

## 6. Codificación posicional — el orden importa

A diferencia de las RNNs, los Transformers procesan todos los tokens **en paralelo**.
Para que el modelo sepa el orden, se añade una **codificación posicional**:

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(n_tokens: int, d_modelo: int) -> np.ndarray:
    """Codificación posicional sinusoidal (artículo original de Attention is All You Need)."""
    PE = np.zeros((n_tokens, d_modelo))
    posiciones = np.arange(n_tokens)[:, np.newaxis]
    denominadores = np.power(10000, np.arange(0, d_modelo, 2) / d_modelo)

    PE[:, 0::2] = np.sin(posiciones / denominadores)  # dimensiones pares: seno
    PE[:, 1::2] = np.cos(posiciones / denominadores)  # dimensiones impares: coseno

    return PE

n_tokens, d_modelo = 50, 64
PE = positional_encoding(n_tokens, d_modelo)

plt.figure(figsize=(10, 4))
plt.pcolormesh(PE.T, cmap='RdBu')
plt.colorbar()
plt.xlabel('Posición del token')
plt.ylabel('Dimensión del embedding')
plt.title('Codificación posicional sinusoidal')
plt.tight_layout()
plt.show()

print('Cada posición tiene un "fingerprint" único gracias a las frecuencias sinusoidales.')
print('Los modelos modernos (LLaMA, Claude) usan RoPE (Rotary Position Embedding),')
print('que generaliza mejor a secuencias más largas que las del entrenamiento.')
```

---

## 7. Por qué los Transformers escalan tan bien

```python
# Ley de escalado de Chinchilla (DeepMind, 2022)
# Pérdida ≈ A/N^α + B/D^β + C (N=parámetros, D=datos)

import numpy as np

def perdida_estimada(n_params: float, n_tokens_entrenamiento: float) -> float:
    """Aproximación simplificada de la ley de Chinchilla."""
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    C = 1.69  # irreducible
    return A / (n_params ** alpha) + B / (n_tokens_entrenamiento ** beta) + C

# Comparativa de modelos
configs = [
    ('GPT-2 (1.5B)',         1.5e9,  300e9),   # 300B tokens
    ('GPT-3 (175B)',         175e9,  300e9),
    ('LLaMA-2-7B',           7e9,    2e12),    # 2T tokens
    ('Optimal 70B Chinchilla', 70e9, 1.4e12),  # ratio óptimo ~20 tokens/param
]

print(f'{"Modelo":<30} {"Parámetros":>12} {"Tokens train":>14} {"Pérdida est.":>13}')
print('-' * 72)
for nombre, n_params, n_tokens in configs:
    perdida = perdida_estimada(n_params, n_tokens)
    p_str = f'{n_params/1e9:.0f}B'
    t_str = f'{n_tokens/1e9:.0f}B' if n_tokens < 1e12 else f'{n_tokens/1e12:.1f}T'
    print(f'{nombre:<30} {p_str:>12} {t_str:>14} {perdida:>13.3f}')

print('\nRegla de Chinchilla: para mejor pérdida, escalar parámetros y datos por igual.')
print('Ratio óptimo: ~20 tokens de entrenamiento por parámetro.')
```

---

→ Anterior: [Redes neuronales](04-redes-neuronales.md) | → Siguiente: [Bloque 2 — LLMs](../llms/01-que-es-un-llm.md)
