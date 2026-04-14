# 02 — Tipos de Inteligencia Artificial

> **Bloque:** Fundamentos · **Nivel:** Introductorio · **Tiempo estimado:** 20 min

---

## Índice

1. [Clasificación por capacidad](#1-clasificación-por-capacidad)
2. [Clasificación por enfoque técnico](#2-clasificación-por-enfoque-técnico)
3. [Machine Learning en detalle](#3-machine-learning-en-detalle)
4. [Deep Learning en detalle](#4-deep-learning-en-detalle)
5. [Comparativa de enfoques](#5-comparativa-de-enfoques)
6. [Resumen](#6-resumen)

---

## 1. Clasificación por capacidad

### IA Estrecha (Narrow AI / ANI)

Es la única forma de IA que existe hoy en producción. Está diseñada y entrenada para una tarea específica o un conjunto acotado de tareas.

**Características:**
- Supera al humano en su tarea específica
- Incapaz de generalizar a tareas fuera de su dominio
- Toda la IA comercial actual pertenece a esta categoría

**Ejemplos:**
- Modelos de lenguaje (Claude, GPT-4, Gemini)
- AlphaGo / AlphaZero (juegos de tablero)
- Sistemas de reconocimiento facial
- Motores de recomendación (Spotify, Netflix)
- Detección de fraude bancario
- Diagnóstico médico por imagen

---

### IA General (Artificial General Intelligence / AGI)

Un sistema AGI podría realizar cualquier tarea intelectual que un humano pueda hacer, y aprender nuevas tareas con la misma facilidad.

**Estado actual:** No existe. Es objeto de debate sobre si es posible, cómo conseguirla y cuándo.

**¿Están cerca los LLMs de la AGI?**
Los modelos actuales son sorprendentemente versátiles (código, texto, análisis, traducción...), pero carecen de razonamiento causal robusto, memoria persistente real y capacidad de actuar en el mundo físico de forma autónoma.

---

### Superinteligencia Artificial (ASI)

Una hipotética IA que superaría la inteligencia humana en *todos* los aspectos: creatividad, resolución de problemas, juicio social, etc.

**Estado actual:** Completamente especulativa. Es el objeto de estudio de la seguridad en IA a largo plazo (Alignment Problem).

---

## 2. Clasificación por enfoque técnico

Más allá de la capacidad, la IA puede clasificarse según *cómo* el sistema aprende o razona:

### IA Simbólica (GOFAI — Good Old-Fashioned AI)

Basada en reglas explícitas y lógica formal. Un humano codifica el conocimiento directamente.

```
SI temperatura > 38.5 Y síntoma = "tos"
  ENTONCES diagnóstico = "posible gripe"
```

- **Ventaja:** Explicable, predecible, determinista
- **Desventaja:** No escala bien; no maneja la ambigüedad del mundo real
- **Uso actual:** Sistemas expertos en entornos muy controlados, motores de reglas en banca/seguros

---

### Machine Learning (ML)

El sistema *aprende* patrones a partir de datos, sin que un humano programe las reglas explícitamente.

```
Datos de entrada → Algoritmo de aprendizaje → Modelo → Predicciones
```

- **Ventaja:** Generaliza a datos nuevos, escala con más datos
- **Desventaja:** Requiere grandes cantidades de datos etiquetados, puede ser opaco

---

### Deep Learning (DL)

Subcampo de ML que usa **redes neuronales artificiales** con múltiples capas ocultas.

```
Entrada → [Capa 1] → [Capa 2] → ... → [Capa N] → Salida
```

- **Ventaja:** Aprende representaciones muy complejas (imágenes, sonido, texto)
- **Desventaja:** Necesita aún más datos y cómputo; menos interpretable

---

### Aprendizaje por Refuerzo (Reinforcement Learning / RL)

Un agente aprende mediante **prueba y error**, recibiendo recompensas o penalizaciones por sus acciones.

```
Agente → Acción → Entorno → Recompensa → Agente (ajusta política)
```

- **Uso destacado:** AlphaGo, sistemas de control robótico, optimización de datacenter
- **Relevante en LLMs:** RLHF (Reinforcement Learning from Human Feedback) se usa para alinear modelos como Claude o GPT con las preferencias humanas

---

## 3. Machine Learning en detalle

### Aprendizaje supervisado

El modelo aprende de datos **etiquetados**: pares (entrada, salida esperada).

| Tarea | Entrada | Salida |
|---|---|---|
| Clasificación de spam | Email (texto) | Spam / No spam |
| Diagnóstico | Imagen médica | Enfermedad detectada |
| Predicción de precios | Características de una casa | Precio estimado |
| Traducción | Texto en español | Texto en inglés |

**Algoritmos comunes:** Regresión lineal/logística, árboles de decisión, SVM, redes neuronales.

---

### Aprendizaje no supervisado

El modelo trabaja con datos **sin etiquetar**. Busca estructura oculta por sí solo.

**Tareas principales:**

- **Clustering:** Agrupar elementos similares (segmentación de clientes, detección de anomalías)
- **Reducción de dimensionalidad:** Comprimir datos manteniendo la información relevante (PCA, t-SNE, UMAP)
- **Modelos generativos:** Aprender la distribución de los datos para generar nuevos ejemplos (VAE, GANs)

---

### Aprendizaje semi-supervisado

Combina una pequeña cantidad de datos etiquetados con una gran cantidad de datos sin etiquetar. Muy útil cuando etiquetar es costoso.

---

### Aprendizaje por refuerzo (RL)

Ya descrito arriba. El agente aprende interactuando con un entorno.

**Conceptos clave:**
- **Política (policy):** la estrategia del agente (qué acción tomar en cada estado)
- **Recompensa (reward):** señal de feedback del entorno
- **Valor (value):** recompensa esperada a largo plazo

---

## 4. Deep Learning en detalle

### Redes Neuronales Artificiales (ANN)

Inspiradas (muy vagamente) en el cerebro humano. Compuestas por **neuronas artificiales** organizadas en capas:

```
Capa de entrada → Capas ocultas → Capa de salida
    [x1, x2, x3]  →  [oculta1] [oculta2]  →  [ŷ]
```

Cada neurona aplica una función de activación a la suma ponderada de sus entradas. El entrenamiento ajusta los pesos mediante **retropropagación** y **descenso de gradiente**.

---

### Tipos de arquitecturas de Deep Learning

| Arquitectura | Siglas | Especialidad | Ejemplos de uso |
|---|---|---|---|
| Red Neuronal Convolucional | CNN | Imágenes, vídeo | Clasificación de imágenes, detección de objetos |
| Red Neuronal Recurrente | RNN / LSTM | Secuencias, series temporales | Traducción (pre-Transformer), predicción de series |
| Red Generativa Adversarial | GAN | Generación de contenido | DeepFakes, generación de imágenes |
| Autoencoder Variacional | VAE | Compresión, generación | Generación de imágenes, reducción de ruido |
| **Transformer** | — | Secuencias, lenguaje, visión | **LLMs (Claude, GPT), BERT, ViT** |

---

### Transformers: la arquitectura dominante

Introducida en 2017 por Google en el paper *"Attention Is All You Need"*, la arquitectura **Transformer** ha revolucionado la IA.

**Mecanismo clave: la atención (attention)**

En lugar de procesar una secuencia token a token (como las RNN), el Transformer procesa toda la secuencia a la vez y calcula qué partes son más relevantes entre sí.

```
"El banco donde me senté estaba mojado"
       ↑
  "banco" presta atención a "senté" y "mojado"
  para desambiguar el significado (banco = asiento, no entidad financiera)
```

Esto permite capturar dependencias a larga distancia con mucha más eficiencia.

---

## 5. Comparativa de enfoques

| Característica | IA Simbólica | ML clásico | Deep Learning |
|---|---|---|---|
| Conocimiento | Codificado manualmente | Aprendido de datos | Aprendido de datos |
| Datos necesarios | Pocos | Miles-millones | Millones-billones |
| Interpretabilidad | Alta | Media | Baja |
| Rendimiento en tareas complejas | Bajo | Medio | Alto |
| Coste computacional | Bajo | Medio | Alto |
| Ejemplo | Sistema experto médico | Random Forest para fraude | GPT-4, Claude |

---

## 6. Resumen

```
IA
├── Por capacidad
│   ├── Narrow AI       ← Todo lo que existe hoy
│   ├── AGI             ← No existe todavía
│   └── ASI             ← Especulativa
│
└── Por enfoque técnico
    ├── IA Simbólica    ← Reglas explícitas
    ├── Machine Learning
    │   ├── Supervisado     ← Datos etiquetados
    │   ├── No supervisado  ← Sin etiquetas
    │   └── Por refuerzo    ← Prueba y error (RLHF)
    └── Deep Learning
        ├── CNN             ← Imágenes
        ├── RNN/LSTM        ← Secuencias
        ├── GAN             ← Generación
        └── Transformer     ← Lenguaje, visión, audio
```

---

**Anterior:** [01 — ¿Qué es la IA?](./01-que-es-la-ia.md) · **Siguiente:** [03 — Historia y evolución de la IA](./03-historia-de-la-ia.md)
