# 04 — Arquitectura Transformer en profundidad

> **Bloque:** Fundamentos · **Nivel:** Intermedio · **Tiempo estimado:** 60 min

---

## Índice

1. [El problema que resuelven los Transformers](#1-el-problema-que-resuelven-los-transformers)
2. [Visión general de la arquitectura](#2-visión-general-de-la-arquitectura)
3. [Embeddings y codificación posicional](#3-embeddings-y-codificación-posicional)
4. [El mecanismo de atención](#4-el-mecanismo-de-atención)
5. [Multi-Head Attention](#5-multi-head-attention)
6. [Feed-Forward Network](#6-feed-forward-network)
7. [Normalización y conexiones residuales](#7-normalización-y-conexiones-residuales)
8. [Encoder vs Decoder vs Encoder-Decoder](#8-encoder-vs-decoder-vs-encoder-decoder)
9. [Enmascaramiento (Masking)](#9-enmascaramiento-masking)
10. [De Transformer a LLM: preentrenamiento y ajuste fino](#10-de-transformer-a-llm-preentrenamiento-y-ajuste-fino)
11. [Scaling laws y los límites del escalado](#11-scaling-laws-y-los-límites-del-escalado)
12. [Variantes y evoluciones](#12-variantes-y-evoluciones)
13. [Resumen](#13-resumen)

---

## 1. El problema que resuelven los Transformers

Para entender por qué los Transformers son revolucionarios hay que entender qué se usaba antes y por qué no era suficiente.

### Redes recurrentes (RNN / LSTM): el estado anterior

Antes de 2017, el estándar para procesar texto eran las **redes neuronales recurrentes (RNN)** y su variante mejorada **LSTM** (Long Short-Term Memory).

```
RNN: procesa tokens uno a uno, izquierda a derecha

Entrada: "El   gato   se   sienta   sobre   el   tejado"
          ↓      ↓      ↓      ↓       ↓      ↓      ↓
        [h₁] → [h₂] → [h₃] → [h₄] → [h₅] → [h₆] → [h₇]
         ↑ estado oculto que "recuerda" lo procesado
```

**Problemas:**
1. **Secuencial por diseño**: no se puede paralelizar el entrenamiento. Hay que esperar a procesar el token N para procesar el N+1.
2. **Memoria a corto plazo**: aunque LSTM mejora esto, el contexto de hace 100 tokens sigue siendo débil. La información se diluye.
3. **Gradientes que desaparecen**: durante el entrenamiento, la señal de error se degrada al propagarse hacia atrás en secuencias largas.

**Consecuencia práctica:** en la frase *"El banco donde me senté estaba mojado"*, una RNN ya habrá "olvidado" parte del contexto de "banco" cuando llega a "mojado".

### La propuesta de los Transformers

El paper seminal *"Attention Is All You Need"* (Vaswani et al., Google, 2017) propuso eliminar la recurrencia y basarse **exclusivamente en atención**:

```
Transformer: todos los tokens se procesan en paralelo y
            cada token puede "atender" a cualquier otro token directamente

Entrada: "El   gato   se   sienta   sobre   el   tejado"
          ↓ ↗↘  ↓ ↗↘  ↓ ↗↘  ↓ ↗↘   ↓ ↗↘  ↓ ↗↘  ↓
         cada token se relaciona con todos los demás simultáneamente
```

**Ventajas inmediatas:**
- Paralelización total → entrenamiento masivamente más rápido en GPU
- Contexto global → cualquier token puede atender a cualquier otro directamente
- Sin degradación de gradiente por distancia

---

## 2. Visión general de la arquitectura

El Transformer original tiene dos componentes principales: un **Encoder** (para entender) y un **Decoder** (para generar). Los LLMs modernos como Claude, GPT y Llama usan **solo el Decoder**.

```
TRANSFORMER COMPLETO (arquitectura original):

     Entrada (texto fuente)          Salida (texto destino)
           ↓                               ↓ (desplazada)
   ┌───────────────────┐          ┌────────────────────────┐
   │    Embedding +    │          │     Embedding +        │
   │  Pos. Encoding    │          │   Pos. Encoding        │
   └─────────┬─────────┘          └──────────┬─────────────┘
             │                               │
   ┌─────────▼─────────┐          ┌──────────▼─────────────┐
   │  Multi-Head       │          │  Masked Multi-Head     │
   │  Self-Attention   │          │  Self-Attention        │
   ├───────────────────┤          ├────────────────────────┤
   │  Feed-Forward     │──────►   │  Cross-Attention       │
   │  Network          │          │  (Encoder-Decoder)     │
   └─────────┬─────────┘          ├────────────────────────┤
             │ × N capas          │  Feed-Forward Network  │
             │                   └──────────┬─────────────┘
         ENCODER                            │ × N capas
                                        DECODER
                                            │
                                     Proyección lineal
                                            │
                                      Softmax → token
```

Para un LLM solo-decoder (Claude, GPT, Llama), el encoder desaparece y el decoder procesa tanto la entrada como la generación:

```
SOLO DECODER (LLMs modernos):

Tokens de entrada
        ↓
   Embedding + Positional Encoding
        ↓
┌────────────────────────┐
│  Masked Self-Attention │  ← solo ve tokens anteriores
├────────────────────────┤
│  Feed-Forward Network  │
└──────────┬─────────────┘
           │ × N capas (32, 64, 96... según el modelo)
           ↓
    Proyección lineal
           ↓
    Softmax sobre vocabulario
           ↓
    Probabilidad de cada token
```

---

## 3. Embeddings y codificación posicional

### 3.1 Embeddings de tokens

Antes de entrar al Transformer, cada token se convierte en un **vector denso** de dimensión `d_model` (típicamente 768, 1024, 2048, 4096...).

```
Vocabulario = {... "gato": 4521, "perro": 7832, ...}

Token "gato" (ID: 4521)
        ↓
Matriz de embeddings E[4521]
        ↓
[0.82, -0.41, 0.13, 0.67, ..., -0.22]  ← vector de d_model dimensiones
```

Esta matriz de embeddings se **aprende durante el entrenamiento**. Tokens semánticamente relacionados acaban con vectores cercanos en el espacio vectorial.

### 3.2 Codificación posicional (Positional Encoding)

El mecanismo de atención es por naturaleza **independiente del orden**: no distingue si "gato come ratón" o "ratón come gato". Para inyectar información de posición, se añade al embedding una **codificación posicional**.

**Codificación sinusoidal (paper original):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Donde:
  pos = posición del token en la secuencia (0, 1, 2, ...)
  i   = dimensión del vector
  d_model = dimensión del embedding
```

Visualmente:

```
Posición 0: [sin(0/1),    cos(0/1),    sin(0/100),  cos(0/100),  ...]
Posición 1: [sin(1/1),    cos(1/1),    sin(1/100),  cos(1/100),  ...]
Posición 2: [sin(2/1),    cos(2/1),    sin(2/100),  cos(2/100),  ...]
```

Cada posición produce un vector único, y las posiciones cercanas producen vectores similares, lo que el modelo puede aprender a interpretar.

**Alternativas modernas:**
- **RoPE (Rotary Position Embedding)**: LLaMA, Qwen. Codifica la posición relativa rotando los vectores Q y K.
- **ALiBi**: penaliza la atención entre tokens distantes añadiendo un sesgo negativo proporcional a la distancia.
- **Learnable embeddings**: simplemente una matriz de posiciones que se aprende (GPT-2, GPT-3).

---

## 4. El mecanismo de atención

### 4.1 Intuición

La atención responde a la pregunta: **"Para representar este token, ¿qué otros tokens debo tener en cuenta y en qué medida?"**

```
Frase: "El banco donde me senté estaba mojado"

Para representar "banco":
  - "senté"   → peso alto  (0.45) ← contexto relevante
  - "mojado"  → peso alto  (0.38) ← contexto relevante
  - "donde"   → peso medio (0.12)
  - "El"      → peso bajo  (0.03)
  - "me"      → peso bajo  (0.02)

La representación de "banco" = suma ponderada de todos los vectores,
con los pesos aprendidos arriba.
```

### 4.2 Queries, Keys y Values (Q, K, V)

El mecanismo de atención usa tres matrices aprendibles: **Q (Query)**, **K (Key)** y **V (Value)**.

La analogía más clara es un motor de búsqueda:
- **Query (Q)**: *"¿Qué estoy buscando?"* — la pregunta que hace el token actual
- **Key (K)**: *"¿Qué ofrezco?"* — la descripción que cada token tiene de sí mismo
- **Value (V)**: *"¿Qué información aporto?"* — el contenido real de cada token

```
Para cada token x en la secuencia:
  Q = x · Wq     (proyección lineal aprendida)
  K = x · Wk
  V = x · Wv

  Donde Wq, Wk, Wv ∈ ℝ^(d_model × d_k) son matrices de pesos
```

### 4.3 Scaled Dot-Product Attention

El cálculo de atención entre una consulta Q y las claves/valores K, V:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

Paso a paso:

```
1. Similitud: Q·Kᵀ
   → Producto escalar entre la query y todas las keys
   → Resultado: vector de puntuaciones de similitud (uno por cada token)

2. Escalado: / √d_k
   → √d_k previene que los productos escalares crezcan demasiado
   → Sin escalado, el softmax saturaría en gradientes muy pequeños

3. Softmax(...)
   → Convierte las puntuaciones en una distribución de probabilidad
   → Suma = 1; cada valor indica "cuánta atención" prestar a ese token

4. · V
   → Multiplica los pesos de atención por los vectores Value
   → Resultado: combinación ponderada de información de toda la secuencia
```

Ejemplo numérico simplificado (d_k = 2):

```
Tokens: ["gato", "come", "pescado"]
Q[0] = [1.0, 0.5]   (query de "gato")
K[0] = [1.0, 0.5]   (key  de "gato")    → similitud alta con sí mismo
K[1] = [0.2, 0.8]   (key  de "come")    → similitud media
K[2] = [0.6, 0.3]   (key  de "pescado") → similitud baja

Q[0]·K[0]ᵀ = 1.0×1.0 + 0.5×0.5 = 1.25
Q[0]·K[1]ᵀ = 1.0×0.2 + 0.5×0.8 = 0.6
Q[0]·K[2]ᵀ = 1.0×0.6 + 0.5×0.3 = 0.75

Escalado (√2 ≈ 1.41): [0.89, 0.43, 0.53]
Softmax:               [0.48, 0.24, 0.28]  ← distribución de atención

Representación de "gato" = 0.48·V[0] + 0.24·V[1] + 0.28·V[2]
```

---

## 5. Multi-Head Attention

Un solo mecanismo de atención solo puede capturar un tipo de relación a la vez. Con **múltiples cabezas** en paralelo, cada una aprende a capturar relaciones distintas.

```
Multi-Head Attention con h cabezas:

Input X ──┬──► [Proyección Q₁, K₁, V₁] ──► Atención₁ ──┐
          ├──► [Proyección Q₂, K₂, V₂] ──► Atención₂ ──┤
          ├──► [Proyección Q₃, K₃, V₃] ──► Atención₃ ──┤
          │                    ...                       │
          └──► [Proyección Qₕ, Kₕ, Vₕ] ──► Atenciónₕ ──┘
                                                         │
                                               Concatenar y proyectar
                                                         │
                                               Output de Multi-Head
```

**¿Qué aprende cada cabeza?**

Las investigaciones de interpretabilidad han encontrado que distintas cabezas en distintas capas especializan en:
- **Sintaxis**: relaciones sujeto-verbo, concordancia de número
- **Correferencia**: "ella" → quién es "ella" en el texto
- **Semántica posicional**: tokens adyacentes, siguiente oración
- **Relaciones de dominio**: términos técnicos relacionados en el campo del texto
- **Estructura**: apertura/cierre de paréntesis, etiquetas HTML

Formalmente:

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wₒ

head_i = Attention(Q · Wᵢq, K · Wᵢk, V · Wᵢv)
```

**Dimensiones típicas:**

| Modelo | d_model | Nº cabezas | d_k por cabeza | Nº capas |
|---|---|---|---|---|
| BERT-base | 768 | 12 | 64 | 12 |
| GPT-2 small | 768 | 12 | 64 | 12 |
| GPT-3 175B | 12288 | 96 | 128 | 96 |
| Llama 3.1 8B | 4096 | 32 | 128 | 32 |
| Claude 3 Sonnet | ~4096* | ~32* | ~128* | ~48* |

*Estimados; Anthropic no publica las arquitecturas exactas.

---

## 6. Feed-Forward Network

Después de la atención, cada capa Transformer incluye una **red feed-forward** (FFN) que se aplica de forma independiente a cada posición:

```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
          ↑
        ReLU (o GeLU en modelos modernos)
```

La FFN tiene tipicamente una dimensión interna 4× mayor que `d_model`:

```
Input:  d_model = 4096
Capa 1: d_ff    = 16384  (×4)
Capa 2: d_model = 4096   (vuelve a la dimensión original)
```

**¿Qué hace la FFN?**

Mientras la atención captura **relaciones entre tokens** (quién atiende a quién), la FFN almacena y transforma **información factual dentro de cada token**. Investigaciones recientes sugieren que las capas FFN actúan como *memorias clave-valor* donde se almacena conocimiento factual.

```
Analogía: la atención es el "razonamiento contextual" del modelo;
          la FFN es la "memoria de hechos".
```

---

## 7. Normalización y conexiones residuales

### 7.1 Conexiones residuales (Residual Connections)

Cada subcapa (Atención y FFN) usa conexiones residuales:

```
Output = LayerNorm(x + Sublayer(x))
```

El `+ x` es la conexión residual: el input se suma directamente al output de la subcapa. Esto:
- Permite que el gradiente fluya directamente hacia atrás durante el entrenamiento (sin degradarse)
- Permite que las capas aprendan la **diferencia** respecto a la entrada en lugar de una función completa

Visualmente:

```
Entrada x
    ├──────────────────────────┐
    ↓                          │
[Multi-Head Attention]         │
    ↓                          │
  output                       │
    +  ←───────────────────────┘ (skip connection)
    ↓
[Layer Norm]
    ↓
    ├──────────────────────────┐
    ↓                          │
[Feed-Forward Network]         │
    ↓                          │
  output                       │
    +  ←───────────────────────┘
    ↓
[Layer Norm]
```

### 7.2 Layer Normalization

Normaliza los activaciones de cada capa para que tengan media ≈ 0 y desviación ≈ 1, estabilizando el entrenamiento:

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Donde μ, σ = media y desviación estándar del vector x
      γ, β = parámetros aprendibles (escala y desplazamiento)
      ε    = constante pequeña para estabilidad numérica
```

**Pre-norm vs Post-norm:**
- Paper original: post-norm (LayerNorm después de la suma residual)
- Modelos modernos (GPT, Llama): pre-norm (LayerNorm antes de la subcapa) → más estable para modelos muy profundos

---

## 8. Encoder vs Decoder vs Encoder-Decoder

### 8.1 Solo Encoder

```
Modelos: BERT, RoBERTa, DeBERTa

Propósito: entender texto, no generarlo.
           Representa texto de entrada con vectores contextuales.

Atención: bidireccional — cada token atiende a todos los demás
          (hacia adelante Y hacia atrás)

Uso:
  ✓ Clasificación de texto
  ✓ Análisis de sentimiento
  ✓ Named Entity Recognition
  ✓ Question Answering extractivo
  ✗ No genera texto nuevo
```

### 8.2 Solo Decoder

```
Modelos: GPT-2, GPT-3, GPT-4, Claude, Llama, Mistral, Gemini

Propósito: generar texto token por token.

Atención: causal (o enmascarada) — cada token solo atiende
          a los tokens ANTERIORES (no puede "ver el futuro")

Uso:
  ✓ Generación de texto
  ✓ Completado de código
  ✓ Conversación (chat)
  ✓ Razonamiento y resolución de problemas
  ✗ Representaciones bidireccionales menos ricas
```

### 8.3 Encoder-Decoder

```
Modelos: T5, BART, MarianMT, mT5

Propósito: transformar una secuencia en otra (seq2seq).

El encoder procesa la entrada bidirecccionalmente.
El decoder genera la salida y atiende al encoder via cross-attention.

Uso:
  ✓ Traducción automática
  ✓ Resumen abstractivo
  ✓ Generación de preguntas
  ✓ Transformación de código
```

---

## 9. Enmascaramiento (Masking)

### 9.1 Causal Mask (Decoder)

Los modelos generativos usan una máscara triangular inferior que impide que un token en la posición `i` atienda a tokens en posiciones `> i`:

```
Secuencia: [El, gato, come, pescado]
           posición: 0    1     2      3

Máscara causal (1 = atención permitida, 0 = bloqueada):

         El  gato  come  pescado
El      [ 1,   0,    0,      0 ]
gato    [ 1,   1,    0,      0 ]
come    [ 1,   1,    1,      0 ]
pescado [ 1,   1,    1,      1 ]
```

Antes del softmax, las posiciones bloqueadas reciben `-∞` para que tras softmax su peso sea 0.

**¿Por qué?** Durante el entrenamiento, el modelo aprende a predecir el siguiente token. Si pudiera "ver el futuro", haría trampa y no aprendería las dependencias reales.

### 9.2 Padding Mask

Cuando se procesan lotes (batches) de secuencias de distinta longitud, se rellenan con tokens de padding `[PAD]`. La máscara de padding evita que el modelo atienda a estos tokens vacíos.

---

## 10. De Transformer a LLM: preentrenamiento y ajuste fino

Un Transformer por sí solo es solo la arquitectura. Convertirlo en un LLM útil requiere tres fases:

### Fase 1: Preentrenamiento (Pre-training)

```
Dataset: billones de tokens de texto web, libros, código, papers...
Tarea: predecir el siguiente token (autoregresión)
Objetivo: minimizar la pérdida de entropía cruzada

Loss = -log P(token_t+1 | token_1, ..., token_t)
```

Esta fase es la más costosa:
- GPT-3: ~$4.6M en compute
- Llama 3.1 70B: ~$500M+ en compute estimado
- Dura semanas o meses en clústeres de miles de GPUs

El resultado es un **modelo base**: muy capaz de completar texto, pero sin seguir instrucciones ni mantener un tono útil/seguro.

### Fase 2: Fine-tuning supervisado (SFT)

```
Dataset: pares (instrucción, respuesta ideal) curados por humanos
Tarea: aprender a responder de la forma que se le muestra
```

Transforma el modelo base en un modelo que sigue instrucciones.

### Fase 3: RLHF (Reinforcement Learning from Human Feedback)

```
1. Generar múltiples respuestas para cada prompt
2. Humanos ranquean las respuestas de mejor a peor
3. Entrenar un "reward model" que predice el ranking humano
4. Usar PPO (algoritmo de RL) para que el LLM maximice la recompensa
```

Resultado: modelo más útil, honesto y seguro. Es lo que separa un GPT base de ChatGPT, o un Claude base de Claude.

**Variante moderna: DPO (Direct Preference Optimization)**

DPO simplifica RLHF eliminando el modelo de recompensa separado y optimizando directamente sobre preferencias humanas. Más estable y fácil de entrenar. Usado por Llama 3, Mistral y otros.

---

## 11. Scaling laws y los límites del escalado

### 11.1 Las Leyes de Scaling (Kaplan et al., 2020)

OpenAI descubrió que el rendimiento de los LLMs sigue leyes de potencia predecibles:

```
Pérdida ≈ C · N^α   (en función del número de parámetros N)
Pérdida ≈ C · D^β   (en función del tamaño del dataset D)
Pérdida ≈ C · F^γ   (en función del compute F)
```

**Implicación:** duplicar el compute siempre mejora el modelo de forma predecible, antes de que el modelo converja.

### 11.2 Chinchilla Scaling Laws (Hoffmann et al., DeepMind, 2022)

Refinamiento crucial: para un compute dado, la distribución óptima entre parámetros y datos es:

```
Tokens de entrenamiento ≈ 20 × Número de parámetros
```

GPT-3 (175B parámetros) fue entrenado con ~300B tokens → **infraentrenado en datos**.

El modelo "Chinchilla" (70B parámetros, 1.4T tokens) superó a Gopher (280B) con la mitad de parámetros. **Más datos > más parámetros** para el mismo compute.

**Consecuencia en modelos modernos:** Llama 3.1 8B fue entrenado con 15 billones de tokens — mucho más que los ~160B que Chinchilla recomendaría para ese tamaño. Los modelos actuales se entrenan intencionalmente con más datos del "óptimo" para producir modelos más eficientes en inferencia.

### 11.3 ¿Seguirá escalando?

El debate actual (2025-2026):
- **A favor del scaling:** modelos como GPT-4o, Claude 3.7, Gemini Ultra siguen mejorando
- **Señales de saturación:** algunos benchmarks se saturan, el coste crece cuadráticamente
- **Nuevas fronteras:** razonamiento via chain-of-thought extendido (OpenAI o1, DeepSeek-R1), test-time compute

---

## 12. Variantes y evoluciones

### 12.1 Arquitecturas especializadas

| Variante | Innovación | Ejemplo |
|---|---|---|
| **Sparse MoE** | Mixture of Experts: solo se activan N de M sub-redes por token | Mixtral, GPT-4 (estimado) |
| **Grouped Query Attention (GQA)** | Varias cabezas Q comparten las mismas K, V → menos memoria | Llama 2/3, Mistral |
| **Flash Attention** | Implementación eficiente en CUDA que evita materializar la matriz de atención completa | Todos los modelos modernos |
| **Rotary Position Embedding (RoPE)** | Codificación posicional relativa via rotaciones → extrapolación de contexto | LLaMA, Qwen, Deepseek |
| **SwiGLU** | Variante de activación en la FFN que mejora el rendimiento | Llama, PaLM |

### 12.2 Mecanismos de contexto largo

El problema: la complejidad de la atención es **O(n²)** en longitud de secuencia. Para contextos de 1M de tokens, esto es prohibitivo.

Soluciones:
- **Sliding Window Attention**: atiende solo a una ventana local + algunos tokens globales (Mistral)
- **Ring Attention**: distribuye la secuencia entre múltiples GPUs
- **Linear Attention**: reduce la complejidad a O(n) con aproximaciones
- **State Space Models (Mamba)**: alternativa a Transformers para secuencias muy largas

---

## 13. Resumen

El Transformer es la base de toda la IA generativa moderna. Sus componentes clave:

```
┌─────────────────────────────────────────────────────────────────┐
│                   BLOQUE TRANSFORMER                            │
│                                                                 │
│  Token → Embedding → + Pos.Encoding                            │
│                           ↓                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Multi-Head Self-Attention                             │    │
│  │  • Queries, Keys, Values (Q, K, V)                    │    │
│  │  • Similitud: QKᵀ/√d_k → softmax → pesos             │    │
│  │  • h cabezas en paralelo, cada una aprende algo       │    │
│  └────────────────────────────────────────────────────────┘    │
│                      + residual + LayerNorm                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Feed-Forward Network                                  │    │
│  │  • Proyección a 4×d_model y vuelta                    │    │
│  │  • Almacena conocimiento factual                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                      + residual + LayerNorm                    │
│                           ↓                                    │
│               (repetido N veces)                               │
└─────────────────────────────────────────────────────────────────┘
```

| Componente | Propósito |
|---|---|
| Embeddings | Convertir tokens en vectores de alta dimensión |
| Positional Encoding | Inyectar información de posición (RoPE, sinusoidal) |
| Self-Attention (Q/K/V) | Capturar relaciones entre tokens |
| Multi-Head Attention | Capturar múltiples tipos de relaciones en paralelo |
| Feed-Forward Network | Transformar y almacenar conocimiento |
| Residual + LayerNorm | Estabilidad numérica y flujo de gradiente |
| Causal Masking | Evitar que el modelo "vea el futuro" al generar |

### Recursos de referencia

- *"Attention Is All You Need"* — Vaswani et al. (2017) — el paper original
- *"Scaling Laws for Neural Language Models"* — Kaplan et al. (2020)
- *"Training Compute-Optimal Large Language Models"* — Hoffmann et al. (2022) — Chinchilla
- *"The Illustrated Transformer"* — Jay Alammar (blog) — la mejor visualización disponible
- Tutorial siguiente: [05 — Algoritmos fundamentales de IA](./05-algoritmos-fundamentales.md)
- Tutorial relacionado en LLMs: [01 — ¿Qué es un LLM?](../llms/01-que-es-un-llm.md)
