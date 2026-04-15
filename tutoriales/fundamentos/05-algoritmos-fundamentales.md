# 05 — Algoritmos fundamentales de la IA

> **Bloque:** Fundamentos · **Nivel:** Intermedio · **Tiempo estimado:** 55 min

---

## Índice

1. [El aprendizaje automático: la idea central](#1-el-aprendizaje-automático-la-idea-central)
2. [Funciones de pérdida](#2-funciones-de-pérdida)
3. [Descenso de gradiente](#3-descenso-de-gradiente)
4. [Retropropagación (Backpropagation)](#4-retropropagación-backpropagation)
5. [Funciones de activación](#5-funciones-de-activación)
6. [Regularización — combatir el sobreajuste](#6-regularización--combatir-el-sobreajuste)
7. [Optimizadores modernos](#7-optimizadores-modernos)
8. [Redes convolucionales (CNN)](#8-redes-convolucionales-cnn)
9. [Redes recurrentes (RNN / LSTM)](#9-redes-recurrentes-rnn--lstm)
10. [El mecanismo de atención — de RNN a Transformer](#10-el-mecanismo-de-atención--de-rnn-a-transformer)
11. [Aprendizaje por refuerzo y RLHF](#11-aprendizaje-por-refuerzo-y-rlhf)
12. [Embeddings y representaciones vectoriales](#12-embeddings-y-representaciones-vectoriales)
13. [Mapa conceptual de la IA](#13-mapa-conceptual-de-la-ia)

---

## 1. El aprendizaje automático: la idea central

En la programación clásica, el programador escribe **reglas explícitas**:

```
SI el email contiene "ganaste un premio" Y "haz clic aquí" ENTONCES → spam
```

El problema: las reglas se vuelven infinitamente complejas. Para clasificar imágenes de perros no puedes escribir todas las reglas posibles.

**El aprendizaje automático invierte el paradigma:**

```
Programación clásica:
  Reglas + Datos → Respuestas

Machine Learning:
  Datos + Respuestas → Reglas (el modelo aprende las reglas)
```

Un modelo de ML es esencialmente una **función parametrizable**:

```
y = f(x; θ)

Donde:
  x = entrada (texto, imagen, audio...)
  y = salida (predicción, clasificación, generación...)
  θ = parámetros (pesos del modelo, millones o miles de millones)
```

El objetivo del entrenamiento: encontrar los valores de `θ` que hacen que `f(x; θ)` dé la respuesta correcta para los datos de entrenamiento.

---

## 2. Funciones de pérdida

La **función de pérdida** (loss function) mide **cuánto se equivoca el modelo**. Es el número que el entrenamiento intenta minimizar.

### 2.1 Error Cuadrático Medio (MSE) — para predicción numérica

```
MSE = (1/n) · Σ (y_predicho - y_real)²

Ejemplo:
  Predicción del precio de una casa: 250.000 €
  Precio real:                       280.000 €
  Error:                             (250.000 - 280.000)² = 900.000.000
```

El cuadrado penaliza más los errores grandes y siempre es positivo.

### 2.2 Entropía cruzada (Cross-Entropy) — para clasificación y LLMs

```
Cross-Entropy = -Σ y_real · log(y_predicho)
```

Para clasificación binaria (spam / no spam):

```
Real: spam (1.0)
Predicción: 0.9 → pérdida = -log(0.9)  = 0.105  ← bien
Predicción: 0.1 → pérdida = -log(0.1)  = 2.303  ← muy mal
```

**Para LLMs**: el modelo predice una distribución sobre todo el vocabulario. La pérdida es el logaritmo negativo de la probabilidad asignada al token correcto:

```
Frase: "El gato come [???]"
Token correcto: "pescado"

Si modelo predice P("pescado") = 0.7  → pérdida = -log(0.7)  = 0.36
Si modelo predice P("pescado") = 0.01 → pérdida = -log(0.01) = 4.60
```

La entropía cruzada se relaciona con la **perplejidad** (perplexity), la métrica estándar para evaluar LLMs:

```
Perplejidad = e^(cross-entropy promedio)
```

Un LLM perfecto tendría perplejidad = 1. GPT-3 en texto estándar: ~20-30.

### 2.3 Otras funciones de pérdida relevantes

| Pérdida | Uso |
|---|---|
| **Binary Cross-Entropy** | Clasificación binaria |
| **Categorical Cross-Entropy** | Clasificación multiclase |
| **MSE** | Regresión, generación de imágenes |
| **Contrastive Loss** | Embeddings, similitud semántica |
| **RLHF Reward** | Ajuste de preferencias humanas |

---

## 3. Descenso de gradiente

### 3.1 El problema de optimización

Minimizar la pérdida sobre todos los datos de entrenamiento:

```
θ* = argmin_θ  L(θ) = (1/n) Σᵢ loss(f(xᵢ; θ), yᵢ)
```

Si el modelo tiene miles de millones de parámetros, no hay forma analítica de encontrar el mínimo. Se usa **descenso de gradiente**: moverse iterativamente en la dirección que más reduce la pérdida.

### 3.2 El gradiente

El **gradiente** `∇L(θ)` es un vector que indica, para cada parámetro, en qué dirección y cuánto cambia la pérdida:

```
Intuición en 2D: la función de pérdida es como un paisaje de montañas.
El gradiente apunta "cuesta arriba".
El descenso de gradiente camina "cuesta abajo".

  L(θ)  ↑
        │     .
        │    . .
        │   .   .   .
        │  .       .   .
        │ .            .  ← mínimo (objetivo)
        └──────────────────► θ
```

### 3.3 La actualización de parámetros

```
θ_nuevo = θ_actual - η · ∇L(θ)

Donde:
  η (eta) = learning rate (tasa de aprendizaje)
  ∇L(θ)  = gradiente de la pérdida respecto a θ
```

### 3.4 Variantes de descenso de gradiente

| Variante | Calcula gradiente sobre | Ventaja | Desventaja |
|---|---|---|---|
| **Batch GD** | Todo el dataset | Gradiente exacto | Muy lento; no escala |
| **Stochastic GD (SGD)** | 1 ejemplo | Muy rápido | Muy ruidoso |
| **Mini-batch GD** | Lote de N ejemplos (32-512) | Equilibrio | El estándar actual |

Los LLMs se entrenan con mini-batch GD, con lotes de hasta millones de tokens.

### 3.5 El learning rate: el hiperparámetro más crítico

```
η demasiado grande → los pasos son enormes → oscila, no converge
η demasiado pequeño → los pasos son minúsculos → converge demasiado lento

Solución: learning rate scheduling
  - Warmup: empezar con lr pequeño y crecer gradualmente
  - Decay: ir reduciéndolo a medida que avanza el entrenamiento
  - Cosine annealing: lr sigue una curva de coseno
```

---

## 4. Retropropagación (Backpropagation)

### 4.1 El problema del gradiente en redes profundas

Una red neuronal profunda es una composición de funciones:

```
salida = f_N( f_{N-1}( ... f_2( f_1(entrada) ) ... ) )
```

Para actualizar los pesos de la capa 1, necesitamos saber cómo la pérdida final depende de esos pesos, a través de todas las capas intermedias.

### 4.2 La regla de la cadena

La retropropagación aplica la **regla de la cadena** del cálculo diferencial:

```
Función compuesta: y = f(g(x))
Derivada: dy/dx = dy/dg · dg/dx

Para una red de N capas:
∂L/∂θ₁ = ∂L/∂a_N · ∂a_N/∂a_{N-1} · ... · ∂a_2/∂a_1 · ∂a_1/∂θ₁
```

### 4.3 El algoritmo paso a paso

```
FORWARD PASS:
  1. Pasar la entrada a través de todas las capas, de la 1 a la N
  2. Calcular la pérdida L al final

BACKWARD PASS (backpropagation):
  3. Calcular ∂L/∂a_N (gradiente en la última capa)
  4. Propagar hacia atrás capa por capa: ∂L/∂a_{i-1} = ∂L/∂a_i · ∂a_i/∂a_{i-1}
  5. Calcular ∂L/∂θ_i para cada capa (el gradiente de los pesos)

ACTUALIZAR PESOS:
  6. θᵢ ← θᵢ - η · ∂L/∂θᵢ  para cada capa i
```

### 4.4 El problema del gradiente que desaparece

En redes muy profundas, los gradientes pueden hacerse exponencialmente pequeños al propagarse hacia atrás:

```
Si cada capa multiplica el gradiente por 0.5:
  Capa 10:  gradiente × 0.5¹⁰  = × 0.001    ← muy pequeño
  Capa 50:  gradiente × 0.5⁵⁰  = × 10⁻¹⁵   ← prácticamente cero
```

Las primeras capas "dejan de aprender" porque su gradiente es despreciable.

**Soluciones:**
- **Conexiones residuales** (ResNet, Transformer): el gradiente fluye directamente
- **Funciones de activación modernas** (ReLU, GeLU) que no saturan
- **Batch/Layer Normalization**: estabiliza las activaciones
- **Gradient clipping**: limita el máximo del gradiente para evitar explosiones

---

## 5. Funciones de activación

Las funciones de activación introducen **no linealidad** en la red. Sin ellas, una red de N capas sería equivalente a una sola capa lineal.

### 5.1 Sigmoide

```
σ(x) = 1 / (1 + e^(-x))

Rango: (0, 1)
Uso: capas de salida para clasificación binaria

Problemas:
  - Satura en valores grandes/pequeños → gradiente ≈ 0
  - Gradiente que desaparece en redes profundas
```

### 5.2 Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Rango: (-1, 1)
Ventaja sobre sigmoide: salida centrada en 0
Mismo problema: satura en los extremos
```

### 5.3 ReLU (Rectified Linear Unit) — el estándar durante años

```
ReLU(x) = max(0, x)

Para x > 0: f(x) = x     (gradiente = 1, no desaparece)
Para x < 0: f(x) = 0     (neurona "muerta")

Ventaja: simple, eficiente, gradiente limpio para x > 0
Problema: "neuronas muertas" — si el input siempre es negativo, la neurona nunca aprende
```

### 5.4 GeLU (Gaussian Error Linear Unit) — estándar en LLMs

```
GeLU(x) = x · Φ(x)

Donde Φ(x) es la función de distribución acumulada de la distribución normal.
Aproximación: GeLU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
```

GeLU es suave (diferenciable en todo punto) y permite valores negativos pequeños, lo que mejora el aprendizaje. Usado en BERT, GPT-2/3, la mayoría de LLMs modernos.

### 5.5 SwiGLU — Llama, PaLM

```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
Swish(x) = x · σ(x)
```

SwiGLU combina SiLU (sigmoid linear unit) con una compuerta multiplicativa. En la práctica supera a GeLU en modelos grandes. Llama, PaLM y muchos modelos modernos lo usan en la FFN.

---

## 6. Regularización — combatir el sobreajuste

El **sobreajuste** (overfitting) ocurre cuando el modelo aprende el dataset de entrenamiento de memoria, pero no generaliza a datos nuevos.

```
        Alto sesgo (underfitting)       Correcto      Alto varianza (overfitting)
              __________               ____            _______
             |                        |    |          |  ↑ memoriza ruido
  Error      |  el modelo es          |    |          |  
  en         |  demasiado simple      |    |          |  perfecto en train,
  test ──────|  para los datos        |    |          |  malo en test
             |________________________|    |__________|
```

### 6.1 Dropout

Durante el entrenamiento, **desactiva aleatoriamente** una fracción `p` de neuronas en cada forward pass:

```
Con dropout p=0.1:
  10% de neuronas → output = 0 (ignoradas)
  90% de neuronas → output normal

En inferencia: todas las neuronas activas, pero pesos escalados × (1-p)
```

Efecto: el modelo no puede depender demasiado de ninguna neurona individual → aprende representaciones más robustas.

### 6.2 Weight Decay (L2 Regularization)

Añade a la pérdida un término proporcional a la magnitud de los pesos:

```
L_total = L_original + λ · Σ θᵢ²

Efecto: durante el entrenamiento, los pesos tienden a ser pequeños
        Un peso grande "cuesta" más → el modelo prefiere soluciones simples
```

### 6.3 Early Stopping

Detener el entrenamiento cuando la pérdida en el conjunto de validación empieza a subir (aunque la pérdida de entrenamiento siga bajando):

```
Epoch 1-50:  train loss ↓, val loss ↓  → seguir entrenando
Epoch 51-60: train loss ↓, val loss →  → zona peligrosa
Epoch 61-70: train loss ↓, val loss ↑  → sobreajuste — DETENER
```

### 6.4 Data Augmentation

Generar variaciones artificiales de los datos de entrenamiento:
- **Imágenes**: rotaciones, recortes, cambios de brillo, volteo
- **Texto**: parafraseo, sinónimos, traducción de ida y vuelta, máscaras aleatorias (BERT)

---

## 7. Optimizadores modernos

### 7.1 SGD con Momentum

```
v_t = β · v_{t-1} + (1-β) · ∇L(θ)   ← velocidad acumulada
θ_t = θ_{t-1} - η · v_t

β típico: 0.9  (recuerda 90% del momento anterior)
```

El momentum permite "pasar por encima" de mínimos locales pequeños y converger más rápido.

### 7.2 Adam — el estándar para LLMs

Adam (Adaptive Moment Estimation) adapta el learning rate individualmente para cada parámetro:

```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L(θ)        ← 1er momento (media)
v_t = β₂ · v_{t-1} + (1-β₂) · (∇L(θ))²     ← 2º momento (varianza)

m̂_t = m_t / (1-β₁ᵗ)    ← corrección de sesgo
v̂_t = v_t / (1-β₂ᵗ)

θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)

Hiperparámetros por defecto: β₁=0.9, β₂=0.999, ε=1e-8
```

**Ventaja clave:** si un parámetro tiene gradientes grandes y consistentes, su learning rate efectivo se reduce. Si tiene gradientes pequeños e inconsistentes, su learning rate aumenta.

### 7.3 AdamW

Adam con weight decay separado del gradiente (la combinación correcta que Adam original no implementaba bien):

```
θ_t = θ_{t-1} - η · [m̂_t / (√v̂_t + ε) + λ · θ_{t-1}]
                                              ↑
                                    weight decay aplicado directamente
```

AdamW es el optimizador estándar para entrenar LLMs (GPT, Llama, Claude, etc.).

---

## 8. Redes convolucionales (CNN)

Aunque los LLMs usan Transformers, las CNNs son fundamentales para entender la historia y para tareas de visión que los modelos multimodales integran.

### 8.1 La operación de convolución

Una CNN aplica **filtros** (kernels) pequeños que se deslizan sobre la imagen:

```
Imagen (6×6):          Filtro (3×3):      Feature Map (4×4):
┌─────────────┐        ┌───────┐           ┌─────────────┐
│1 0 1 0 1 0  │   ×   │1 0 -1 │    =      │ resultado   │
│0 1 0 1 0 1  │        │1 0 -1 │           │ de aplicar  │
│1 0 1 0 1 0  │        │1 0 -1 │           │ el filtro   │
│0 1 0 1 0 1  │        └───────┘           │ en cada     │
│1 0 1 0 1 0  │                            │ posición    │
│0 1 0 1 0 1  │                            └─────────────┘
└─────────────┘
```

**Ventajas clave de las CNN:**
- **Compartición de pesos**: el mismo filtro se aplica en toda la imagen → muchos menos parámetros que una red densa
- **Invarianza translacional**: un gato en la esquina y un gato en el centro producen activaciones similares
- **Jerarquía de características**: capas tempranas detectan bordes → capas medias detectan texturas → capas profundas detectan objetos

### 8.2 Capas pooling

Reducen las dimensiones espaciales y añaden invarianza:

```
Max Pooling (2×2):
┌───────────────┐     ┌───────┐
│ 1  3 | 2  4  │     │  3  4 │
│ 5  6 | 1  2  │ →   │  6  4 │
│ 7  8 | 3  4  │     │  8  5 │
│ 1  2 | 4  5  │     └───────┘
└───────────────┘
    Se queda con el máximo de cada ventana 2×2
```

### 8.3 Arquitecturas CNN históricas

| Arquitectura | Año | Innovación |
|---|---|---|
| **LeNet** | 1989 | Primera CNN funcional (dígitos escritos a mano) |
| **AlexNet** | 2012 | GPU + ReLU + Dropout → revolución ImageNet |
| **VGG** | 2014 | Profundidad consistente (3×3 kernels) |
| **ResNet** | 2015 | Conexiones residuales → redes de 100+ capas |
| **EfficientNet** | 2019 | Escalado eficiente de anchura/profundidad/resolución |

**ResNet** es especialmente relevante porque las **conexiones residuales** que introdujo son ahora omnipresentes en los Transformers.

---

## 9. Redes recurrentes (RNN / LSTM)

Antes de los Transformers, las RNN eran el estándar para secuencias. Entender sus limitaciones explica por qué los Transformers las reemplazaron.

### 9.1 RNN básica

```
Procesa tokens de forma secuencial, manteniendo un estado oculto:

  x₁ → [RNN] → h₁
  x₂ → [RNN] → h₂   (h₁ se propaga a h₂)
  x₃ → [RNN] → h₃
  ...

Ecuaciones:
  hₜ = tanh(Wₕ · hₜ₋₁ + Wₓ · xₜ + b)
  yₜ = Wᵧ · hₜ + bᵧ
```

**Problema:** el estado `h` es un vector de tamaño fijo que debe comprimir toda la historia de la secuencia. Después de ~50 tokens, la información de los primeros se pierde.

### 9.2 LSTM (Long Short-Term Memory)

LSTM introduce compuertas que controlan explícitamente qué información guardar, descartar o leer:

```
Compuerta de olvido (forget gate):
  fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)   → "¿qué olvido del pasado?"

Compuerta de entrada (input gate):
  iₜ = σ(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)   → "¿qué información nueva retengo?"
  c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc) → "nueva información candidata"

Actualización del estado celular:
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ       → "estado de memoria a largo plazo"

Compuerta de salida (output gate):
  oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)   → "¿qué expongo al exterior?"
  hₜ = oₜ ⊙ tanh(cₜ)
```

LSTM es significativamente mejor que RNN básica para dependencias largas. Sin embargo, sigue siendo **inherentemente secuencial** y tiene dificultades con dependencias de cientos de tokens.

### 9.3 Por qué los Transformers reemplazaron a las RNN

| Criterio | RNN/LSTM | Transformer |
|---|---|---|
| Paralelismo en entrenamiento | No (secuencial) | Sí (total) |
| Dependencias largas | Débil a partir de ~50-100 tokens | Directo, sin degradación |
| Velocidad de entrenamiento | Lento | Muy rápido en GPU |
| Velocidad de inferencia | Rápida (por token) | Más lenta (cuadrática) |
| Escalabilidad | Limitada | Escala excelentemente |

Las RNN aún se usan en edge computing y aplicaciones donde la inferencia secuencial es crítica (audio, señales de tiempo real).

---

## 10. El mecanismo de atención — de RNN a Transformer

### 10.1 Atención en modelos seq2seq (2015)

La atención surgió como mejora para los modelos de traducción RNN (Bahdanau et al., 2015):

```
PROBLEMA: el encoder RNN comprime toda la frase a un único vector.
          Para frases largas, este vector no puede capturar todo el contenido.

SOLUCIÓN: que el decoder pueda "mirar" todos los estados ocultos del encoder
          y decidir en qué enfocarse en cada paso de la decodificación.
```

```
Frase fuente: "The cat sat on the mat"
              h₁   h₂   h₃  h₄  h₅   h₆

Al generar "gato" en español:
  Atención ──► h₂ ("cat") recibe peso alto 0.85
               h₁, h₃-h₆ reciben pesos bajos

Al generar "mat" → "alfombra":
  Atención ──► h₆ ("mat") recibe peso alto 0.78
```

### 10.2 Self-Attention — la innovación del Transformer

El paper "Attention Is All You Need" tomó la atención y la aplicó **dentro de la propia secuencia** (self-attention), eliminando las RNN por completo:

```
En lugar de: encoder RNN + atención → decoder RNN
El Transformer usa: solo atención (en todas las capas, en paralelo)
```

Esto permite que cada token, en cada capa, se relacione directamente con cualquier otro token de la secuencia, independientemente de la distancia.

Para una explicación completa del mecanismo Q/K/V y Multi-Head Attention, ver el [Tutorial 04 — Arquitectura Transformer](./04-arquitectura-transformers.md).

---

## 11. Aprendizaje por refuerzo y RLHF

### 11.1 Aprendizaje por refuerzo (RL)

El RL es un paradigma donde un **agente** aprende por **recompensas** y **penalizaciones**:

```
┌──────────────────────────────────────────┐
│                ENTORNO                   │
│                                          │
│  Estado s_t                              │
│       ↓                                  │
│  AGENTE ──── acción a_t ────►           │
│       ↑                                  │
│  Recompensa r_t, Estado s_{t+1}          │
└──────────────────────────────────────────┘

Objetivo: maximizar la recompensa acumulada a largo plazo
```

Algoritmos clave:
- **Q-learning**: aprende el "valor" de cada (estado, acción)
- **PPO (Proximal Policy Optimization)**: el estándar moderno, estable y eficiente
- **DPO (Direct Preference Optimization)**: variante más simple usada en LLMs

### 11.2 RLHF — cómo se alinean los LLMs

RLHF (Reinforcement Learning from Human Feedback) es el proceso que convierte un LLM capaz pero "crudo" en un asistente útil y seguro:

```
FASE 1 — Preentrenamiento (ya visto):
  Aprende a predecir texto. Capaz, pero sin seguir instrucciones ni valores.

FASE 2 — Fine-tuning supervisado (SFT):
  Entrenado con ejemplos curados de (instrucción, respuesta ideal)
  → Aprende el formato de conversación

FASE 3 — Entrenamiento del Reward Model:
  Humanos ven pares de respuestas para el mismo prompt y votan cuál es mejor
  → Se entrena un modelo separado que predice la preferencia humana

FASE 4 — RL con el Reward Model (PPO):
  El LLM genera respuestas
  El Reward Model puntúa cada respuesta
  PPO actualiza el LLM para maximizar la puntuación
  KL-divergence penalty: evita que se aleje demasiado del modelo base
```

### 11.3 DPO — la alternativa más simple

DPO elimina el modelo de recompensa separado. Optimiza directamente la política usando pares de preferencias:

```
Dataset: {prompt, respuesta_ganadora, respuesta_perdedora}

DPO Loss:
  -log σ(β · log π(ganadora|prompt) - β · log π(perdedora|prompt))
         ↑
   β controla la fuerza de la preferencia
```

Más estable que PPO, más fácil de implementar. Usado por Llama 3, Mistral Instruct y muchos modelos open-source.

---

## 12. Embeddings y representaciones vectoriales

### 12.1 ¿Qué es un embedding?

Un **embedding** es una representación densa y continua de un objeto discreto (token, frase, imagen) en un espacio vectorial de alta dimensión.

```
"París"  → [0.82, -0.41,  0.13,  0.67, ..., -0.22]   (768 dimensiones)
"Madrid" → [0.79, -0.38,  0.15,  0.64, ..., -0.19]   ← cercano a París
"perro"  → [-0.12,  0.93, -0.55,  0.03, ...,  0.44]  ← lejos de París
```

La clave: **la geometría importa**. Objetos semánticamente relacionados tienen vectores cercanos.

### 12.2 Word2Vec — el punto de partida histórico

Word2Vec (Mikolov et al., 2013) fue el primer método exitoso para aprender embeddings de palabras:

```
Idea: entrenar un modelo para predecir palabras vecinas en el texto

"El gato [???] sobre la alfombra"   → el contexto predice la palabra oculta

Resultado: palabras con contextos similares → vectores similares
```

Propiedades famosas de Word2Vec:
```
vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")
vector("París") - vector("Francia") + vector("Alemania") ≈ vector("Berlín")
```

### 12.3 Embeddings contextuales (BERT, LLMs)

Word2Vec asigna un único vector a cada palabra. Pero "banco" (asiento vs. entidad financiera) debería tener vectores distintos según el contexto.

Los LLMs generan embeddings **contextuales**: el mismo token tiene representaciones distintas según lo que lo rodea:

```
"El banco donde me senté"  → embedding("banco") apunta hacia asiento
"El banco tiene sede en Madrid" → embedding("banco") apunta hacia institución
```

### 12.4 Embeddings para búsqueda semántica

Los modelos de embeddings (como `text-embedding-3-small` de OpenAI o `nomic-embed-text` de Nomic) transforman frases completas en vectores para búsqueda semántica:

```
Consulta: "¿Cómo cocinar pasta?"
                ↓ embedding
         [0.42, -0.31, ..., 0.18]
                ↓ búsqueda coseno
Resultado más cercano: "Receta de espaguetis carbonara" [similitud: 0.94]
                       "Tiempo de cocción de la pasta"  [similitud: 0.89]
                       "Historia de la pizza"           [similitud: 0.61]
```

Esta es la base de los sistemas **RAG** (Retrieval-Augmented Generation) — ver tutorial 05.

---

## 13. Mapa conceptual de la IA

```
INTELIGENCIA ARTIFICIAL
│
├── Machine Learning
│   ├── Aprendizaje supervisado
│   │   ├── Clasificación (SVM, árboles, redes neuronales)
│   │   └── Regresión (regresión lineal, random forest)
│   ├── Aprendizaje no supervisado
│   │   ├── Clustering (K-means, DBSCAN)
│   │   └── Reducción de dimensionalidad (PCA, t-SNE)
│   └── Aprendizaje por refuerzo (Q-learning, PPO)
│
└── Deep Learning (redes neuronales profundas)
    │
    ├── CNN (visión, audio)
    │   ├── LeNet → AlexNet → VGG → ResNet → EfficientNet
    │   └── Base de modelos de visión (CLIP, ViT)
    │
    ├── RNN / LSTM (secuencias, pre-2017)
    │   └── Atención en seq2seq (2015)
    │
    └── Transformer (2017 → presente)
        ├── Solo Encoder: BERT, RoBERTa (comprensión)
        ├── Encoder-Decoder: T5, BART (traducción, resumen)
        └── Solo Decoder: GPT, Claude, Llama (generación)
            │
            ├── Preentrenamiento (predicción de siguiente token)
            ├── Fine-tuning supervisado (SFT)
            └── RLHF / DPO (alineación con preferencias humanas)

ALGORITMOS FUNDAMENTALES (transversales a toda la taxonomía):
  ├── Descenso de gradiente (batch, stochastic, mini-batch)
  ├── Retropropagación (regla de la cadena)
  ├── Funciones de activación (ReLU, GeLU, SwiGLU)
  ├── Funciones de pérdida (MSE, cross-entropy)
  ├── Regularización (dropout, weight decay)
  ├── Optimizadores (SGD, Adam, AdamW)
  └── Embeddings y representaciones vectoriales
```

---

## Resumen

| Concepto | Qué es | Por qué importa |
|---|---|---|
| **Función de pérdida** | Mide el error del modelo | Define qué queremos minimizar |
| **Descenso de gradiente** | Optimización iterativa por el gradiente | Cómo aprenden todos los modelos |
| **Backpropagation** | Regla de la cadena sobre la red | Calcula gradientes en todas las capas |
| **ReLU / GeLU** | Funciones de activación no lineales | Permiten aprender funciones complejas |
| **Dropout / Weight Decay** | Regularización | Evitan el sobreajuste |
| **Adam / AdamW** | Optimizadores adaptativos | Estándar para entrenar LLMs |
| **CNN** | Convoluciones sobre datos espaciales | Visión, señales, audio |
| **LSTM** | RNN con compuertas de memoria | Historia del procesamiento de secuencias |
| **Atención** | Ponderación dinámica de tokens | Corazón del Transformer |
| **RLHF / DPO** | RL con feedback humano | Alineación de LLMs |
| **Embeddings** | Representaciones vectoriales densas | Semántica, RAG, búsqueda |

### Recursos de referencia

- *"Deep Learning"* — Goodfellow, Bengio, Courville (libro) — referencia académica completa
- *"Neural Networks and Deep Learning"* — Michael Nielsen (online, gratuito) — pedagogía clara
- *"CS231n: Convolutional Neural Networks for Visual Recognition"* — Stanford (YouTube)
- *"The spelled-out intro to neural networks and backpropagation"* — Andrej Karpathy (YouTube)
- Tutorial anterior: [04 — Arquitectura Transformer](./04-arquitectura-transformers.md)
- Tutorial relacionado: [01 — ¿Qué es un LLM?](../llms/01-que-es-un-llm.md)
