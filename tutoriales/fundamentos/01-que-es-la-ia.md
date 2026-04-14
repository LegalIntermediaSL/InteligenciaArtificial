# 01 — ¿Qué es la Inteligencia Artificial?

> **Bloque:** Fundamentos · **Nivel:** Introductorio · **Tiempo estimado:** 15 min

---

## Índice

1. [Definición](#1-definición)
2. [¿Por qué ahora?](#2-por-qué-ahora)
3. [IA estrecha vs IA general](#3-ia-estrecha-vs-ia-general)
4. [Áreas principales de la IA](#4-áreas-principales-de-la-ia)
5. [IA, Machine Learning y Deep Learning](#5-ia-machine-learning-y-deep-learning)
6. [Conceptos erróneos frecuentes](#6-conceptos-erróneos-frecuentes)
7. [Resumen](#7-resumen)

---

## 1. Definición

La **Inteligencia Artificial (IA)** es la rama de la informática que estudia y desarrolla sistemas capaces de realizar tareas que, cuando las realiza un humano, requieren inteligencia.

El término fue acuñado en **1956** por John McCarthy, quien la definió como:

> *"La ciencia e ingeniería de crear máquinas inteligentes."*

Con el tiempo la definición ha evolucionado. Hoy se entiende la IA de forma más operativa:

> Un sistema es inteligente si puede **percibir su entorno**, **procesar información** y **actuar** de forma que maximice la probabilidad de alcanzar un objetivo.

Esto incluye tareas como:

- Reconocer imágenes o voz
- Traducir idiomas
- Jugar al ajedrez o al Go
- Mantener una conversación
- Escribir código o redactar textos
- Conducir un vehículo de forma autónoma

---

## 2. ¿Por qué ahora?

La IA lleva décadas en desarrollo, pero el salto reciente se explica por tres factores convergentes:

### 2.1 Datos masivos (Big Data)

Los modelos de IA aprenden de ejemplos. El crecimiento exponencial de datos digitales (texto, imágenes, vídeos, transacciones) ha proporcionado el "combustible" que los sistemas modernos necesitan.

### 2.2 Potencia de cómputo

Las **GPUs** (unidades de procesamiento gráfico), diseñadas originalmente para videojuegos, resultaron ideales para entrenar redes neuronales. Hoy existen chips especializados (TPUs de Google, chips de Nvidia) dedicados exclusivamente a IA.

### 2.3 Avances en algoritmos

La aparición del **aprendizaje profundo** (deep learning) en torno a 2012 y, más recientemente, la arquitectura **Transformer** (2017), han permitido construir modelos de una capacidad antes impensable.

---

## 3. IA estrecha vs IA general

Toda la IA que existe hoy en producción es **IA estrecha** (narrow AI o ANI):

| Tipo | Descripción | Ejemplos |
|---|---|---|
| **IA Estrecha (ANI)** | Diseñada para una tarea concreta | GPT-4, AlphaGo, Siri, recomendadores de Netflix |
| **IA General (AGI)** | Capaz de realizar cualquier tarea intelectual humana | No existe todavía |
| **Superinteligencia (ASI)** | Supera la inteligencia humana en todos los ámbitos | Teórica, muy especulativa |

> Un modelo de lenguaje como Claude puede escribir código, resumir documentos y mantener conversaciones, pero no puede conducir un coche ni controlar un robot. Sigue siendo IA estrecha, aunque muy versátil.

---

## 4. Áreas principales de la IA

La IA es un campo amplio que abarca múltiples disciplinas:

```
Inteligencia Artificial
│
├── Machine Learning (ML)
│   ├── Aprendizaje supervisado
│   ├── Aprendizaje no supervisado
│   └── Aprendizaje por refuerzo
│
├── Deep Learning (DL)
│   ├── Redes neuronales convolucionales (CNN)
│   ├── Redes neuronales recurrentes (RNN)
│   └── Transformers
│
├── Procesamiento del Lenguaje Natural (NLP / PLN)
│   ├── Modelos de lenguaje (LLMs)
│   ├── Traducción automática
│   └── Análisis de sentimiento
│
├── Visión por Computador (Computer Vision)
│   ├── Clasificación de imágenes
│   ├── Detección de objetos
│   └── Generación de imágenes (DALL·E, Stable Diffusion)
│
├── Robótica e IA encarnada
│
└── Sistemas expertos y IA simbólica
```

---

## 5. IA, Machine Learning y Deep Learning

Estos tres términos se confunden con frecuencia. La relación entre ellos es de **inclusión**:

```
┌─────────────────────────────────────┐
│         Inteligencia Artificial     │
│  ┌───────────────────────────────┐  │
│  │       Machine Learning        │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │      Deep Learning      │  │  │
│  │  │  ┌───────────────────┐  │  │  │
│  │  │  │   Transformers /  │  │  │  │
│  │  │  │      LLMs         │  │  │  │
│  │  │  └───────────────────┘  │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

- **IA** es el campo general: cualquier sistema que simule comportamiento inteligente.
- **Machine Learning** es una subdisciplina de la IA: sistemas que *aprenden de datos* sin ser programados explícitamente para cada caso.
- **Deep Learning** es una subdisciplina de ML: usa redes neuronales con muchas capas para aprender representaciones complejas.
- **Transformers / LLMs** son una arquitectura concreta de Deep Learning, especializada en procesar secuencias (texto, audio, código).

---

## 6. Conceptos erróneos frecuentes

**"La IA entiende lo que dice"**
Los LLMs procesan y generan texto de forma estadísticamente coherente, pero no "comprenden" en el sentido humano. No tienen experiencias, intenciones ni conciencia.

**"La IA siempre tiene razón"**
Los modelos cometen errores, inventan datos (alucinaciones) y pueden reproducir sesgos presentes en sus datos de entrenamiento. Siempre hay que verificar outputs críticos.

**"La IA va a reemplazar todos los empleos de inmediato"**
La IA automatiza tareas específicas, no trabajos completos. Transforma los roles más que eliminarlos, aunque el impacto varía mucho por sector.

**"La IA es una caja negra ininteligible"**
Es un área activa de investigación (IA explicable / XAI). Algunos modelos son perfectamente interpretables; otros, como las redes neuronales profundas, lo son menos, pero hay técnicas para analizarlos.

---

## 7. Resumen

| Concepto | Clave |
|---|---|
| IA | Sistemas que realizan tareas que requieren inteligencia |
| Narrow AI | Todo lo que existe hoy: especializada en tareas concretas |
| AGI | IA de propósito general: no existe todavía |
| ML | Subdisciplina de IA: aprende de datos |
| Deep Learning | Subdisciplina de ML: redes neuronales profundas |
| LLMs | Arquitectura Transformer aplicada a lenguaje |

---

**Siguiente:** [02 — Tipos de IA](./02-tipos-de-ia.md)
