# 01 — ¿Qué es un LLM?

> **Bloque:** LLMs · **Nivel:** Introductorio-Intermedio · **Tiempo estimado:** 25 min

---

## Índice

1. [Definición](#1-definición)
2. [De texto a números: tokens y embeddings](#2-de-texto-a-números-tokens-y-embeddings)
3. [La arquitectura Transformer](#3-la-arquitectura-transformer)
4. [Cómo genera texto un LLM](#4-cómo-genera-texto-un-llm)
5. [Fases de entrenamiento](#5-fases-de-entrenamiento)
6. [Parámetros y escala](#6-parámetros-y-escala)
7. [Capacidades y limitaciones](#7-capacidades-y-limitaciones)
8. [Modelos destacados](#8-modelos-destacados)
9. [Resumen](#9-resumen)

---

## 1. Definición

Un **Large Language Model (LLM)** es un modelo de aprendizaje profundo entrenado sobre enormes cantidades de texto para aprender a predecir y generar lenguaje natural.

"Large" hace referencia a dos cosas:
- **Datos de entrenamiento:** cientos de miles de millones de palabras (libros, webs, código, artículos...)
- **Parámetros del modelo:** miles de millones de valores numéricos que codifican el "conocimiento" aprendido

La tarea fundamental de un LLM es aparentemente simple:

> Dado un texto de entrada, predecir cuál es el siguiente token más probable.

De esta tarea simple emergen capacidades sorprendentes: razonamiento, traducción, resumen, escritura creativa, generación de código y mucho más.

---

## 2. De texto a números: tokens y embeddings

Los ordenadores trabajan con números. Antes de procesar texto, hay que convertirlo.

### Tokenización

El texto se divide en **tokens**, que son las unidades mínimas de procesamiento. Un token puede ser una palabra completa, una parte de palabra, o un carácter de puntuación.

```
"Inteligencia Artificial" → ["Intelig", "encia", " Artif", "icial"]
"hola mundo"              → ["hola", " mundo"]
"ChatGPT"                 → ["Chat", "G", "PT"]
```

Cada token recibe un **ID numérico** único del vocabulario del modelo.

**Regla práctica:** 1 token ≈ 0.75 palabras en inglés / ≈ 0.6 palabras en español.

Un modelo con "contexto de 128.000 tokens" puede procesar aproximadamente 96.000 palabras (~un libro completo) de una vez.

### Embeddings

Cada token se convierte en un **vector de alta dimensión** (embedding): una lista de números que captura el significado semántico del token en relación con todos los demás.

```
"rey"    → [0.82, -0.41, 0.13, ..., 0.67]  # vector de ~768-4096 dimensiones
"reina"  → [0.79, -0.38, 0.15, ..., 0.71]  # similar a "rey"
"coche"  → [-0.12, 0.93, -0.55, ..., 0.03] # muy diferente
```

La geometría de estos vectores captura relaciones semánticas:

```
embedding("rey") - embedding("hombre") + embedding("mujer") ≈ embedding("reina")
```

---

## 3. La arquitectura Transformer

Los LLMs modernos se basan en la arquitectura **Transformer**, introducida por Google en 2017.

### Componentes principales

```
Texto de entrada
      ↓
Tokenización + Embeddings
      ↓
┌─────────────────────────────┐
│     Bloque Transformer ×N   │
│  ┌───────────────────────┐  │
│  │  Multi-Head Attention  │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Feed-Forward Network  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
      ↓
Distribución de probabilidad sobre tokens
      ↓
Token generado
```

### Atención (Attention): el mecanismo clave

El mecanismo de **atención** permite al modelo decidir, para cada token, qué otros tokens de la secuencia son más relevantes para entenderlo.

Ejemplo:

```
"El banco donde me senté estaba mojado"
```

Para entender "banco" (asiento vs. entidad financiera), el modelo presta atención a "senté" y "mojado", que confirman que es un asiento.

### Atención multi-cabeza (Multi-Head Attention)

En lugar de un solo mecanismo de atención, el Transformer usa varios en paralelo (**cabezas**). Cada cabeza aprende a atender a aspectos diferentes:

- Una cabeza puede capturar relaciones sintácticas (sujeto-verbo)
- Otra, relaciones semánticas (sinónimos, antónimos)
- Otra, referencias pronominales ("él" → quién es "él" en el texto)

### Encoders y Decoders

| Arquitectura | Ejemplo | Uso |
|---|---|---|
| Solo encoder | BERT | Clasificación, comprensión de texto |
| Solo decoder | GPT, Claude, Llama | Generación de texto |
| Encoder-decoder | T5, BART | Traducción, resumen |

Los LLMs conversacionales (Claude, GPT-4, Gemini) usan **solo el decoder**.

---

## 4. Cómo genera texto un LLM

La generación es un proceso **autoregresivo**: el modelo genera un token a la vez, y cada token generado se añade al contexto para generar el siguiente.

```
Prompt: "La capital de Francia es"
  → Paso 1: predice " París"      (alta probabilidad)
  → Paso 2: predice "."           (alta probabilidad)
  → Paso 3: predice " La"         (menos probable) ← aquí puede variar
  ...
```

### Temperatura y muestreo

El modelo produce una **distribución de probabilidad** sobre todos los tokens del vocabulario. ¿Cuál se elige?

**Temperatura (`temperature`):** controla la aleatoriedad.

| Temperatura | Efecto | Uso recomendado |
|---|---|---|
| 0.0 | Siempre el token más probable (determinista) | Código, datos estructurados, preguntas factuales |
| 0.3–0.7 | Equilibrio entre coherencia y variedad | Resúmenes, análisis |
| 0.8–1.2 | Más creativo y variado | Escritura creativa, brainstorming |
| > 1.5 | Muy aleatorio, puede ser incoherente | Raramente útil |

**Top-p (nucleus sampling):** en lugar de considerar todos los tokens, solo considera los tokens cuyas probabilidades acumuladas alcanzan el umbral `p`.

```python
# Ejemplo con la API de Anthropic
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    temperature=0.7,   # creatividad moderada
    messages=[{"role": "user", "content": "Escribe un haiku sobre la IA"}]
)
```

---

## 5. Fases de entrenamiento

Un LLM moderno pasa por varias fases de entrenamiento:

### Fase 1: Pre-entrenamiento

El modelo aprende a predecir el siguiente token sobre un corpus masivo y diverso:

- Páginas web (CommonCrawl, C4)
- Libros digitalizados
- Artículos académicos (ArXiv)
- Código fuente (GitHub)
- Wikipedia
- Foros y conversaciones

**Objetivo:** que el modelo aprenda gramática, hechos del mundo, razonamiento, código, y todo tipo de conocimiento implícito en el texto humano.

**Coste:** miles de GPUs durante meses. GPT-3 costó ~$5M en cómputo. Los modelos actuales cuestan mucho más.

### Fase 2: Instruction Fine-tuning (SFT)

El modelo pre-entrenado sabe predecir texto, pero no "sabe seguir instrucciones". En esta fase se entrena con ejemplos de pares (instrucción, respuesta ideal) creados por humanos.

```
Instrucción: "Resume este artículo en 3 puntos"
Respuesta ideal: [escrita por un humano]
```

### Fase 3: RLHF (Reinforcement Learning from Human Feedback)

Se entrena un **modelo de recompensa** que aprende a puntuar respuestas según las preferencias humanas. Luego se usa esa señal para ajustar el LLM mediante aprendizaje por refuerzo.

Esto es lo que hace que Claude, GPT o Gemini sean útiles, seguros y alineados con las preferencias del usuario.

```
Pregunta → LLM genera 2 respuestas → Humano elige la mejor → Modelo de recompensa aprende → LLM mejora
```

**Anthropic** ha desarrollado **Constitutional AI (CAI)**, una variante que usa principios escritos (una "constitución") en lugar de solo feedback humano para alinear el modelo.

---

## 6. Parámetros y escala

Los **parámetros** son los valores numéricos (pesos) que el modelo aprende durante el entrenamiento. Determinan el "conocimiento" y capacidad del modelo.

| Modelo | Parámetros (aprox.) | Contexto |
|---|---|---|
| GPT-2 (2019) | 1.5B | 1.024 tokens |
| GPT-3 (2020) | 175B | 4.096 tokens |
| Llama 3 8B (2024) | 8B | 128K tokens |
| Claude 3 Haiku | ~20B (est.) | 200K tokens |
| GPT-4 (2023) | ~1T (est.) | 128K tokens |
| Claude 3 Opus | desconocido | 200K tokens |

### Leyes de escala (Scaling Laws)

Investigadores de OpenAI y DeepMind descubrieron que el rendimiento de los LLMs mejora de forma predecible al aumentar:
- El número de parámetros
- La cantidad de datos de entrenamiento
- El cómputo utilizado

Esto ha impulsado la carrera por modelos cada vez más grandes. Sin embargo, a partir de 2023 el foco se desplaza hacia **modelos más eficientes** (Mistral 7B, Llama 3 8B) que consiguen rendimientos sorprendentes con menos parámetros.

---

## 7. Capacidades y limitaciones

### Capacidades emergentes

A partir de cierta escala, los modelos desarrollan capacidades que no fueron entrenadas explícitamente:

- **Razonamiento en cadena (Chain-of-Thought):** resolver problemas paso a paso
- **Aprendizaje en contexto (In-Context Learning):** aprender de ejemplos dados en el prompt sin reentrenar
- **Traducción zero-shot:** traducir entre pares de idiomas no vistos explícitamente
- **Generación de código:** escribir, explicar y depurar código en múltiples lenguajes

### Limitaciones importantes

| Limitación | Descripción |
|---|---|
| **Alucinaciones** | El modelo puede generar texto plausible pero factualmente incorrecto con total confianza |
| **Fecha de corte** | El conocimiento está limitado a la fecha de entrenamiento (no sabe qué pasó después) |
| **Sin memoria persistente** | Cada conversación empieza desde cero (sin herramientas externas) |
| **Contexto limitado** | Aunque grande, el contexto tiene un límite; no puede procesar libros enteros por defecto |
| **Sesgos** | Refleja los sesgos presentes en los datos de entrenamiento |
| **No razona causalmente** | Es muy bueno en patrones estadísticos, pero el razonamiento causal profundo es débil |

---

## 8. Modelos destacados

| Modelo | Organización | Tipo | Destacado por |
|---|---|---|---|
| **Claude 3.5 / 4** | Anthropic | Propietario | Seguridad, razonamiento largo, contexto 200K |
| **GPT-4o** | OpenAI | Propietario | Multimodalidad (texto, imagen, audio) |
| **Gemini Ultra** | Google | Propietario | Integración con servicios Google |
| **Llama 3** | Meta | Open source | Libre para uso comercial, muy eficiente |
| **Mistral** | Mistral AI | Open source | Modelos pequeños y potentes |
| **Qwen** | Alibaba | Open source | Excelente en chino y multilingüe |
| **DeepSeek** | DeepSeek | Open source | Muy capaz, entrenado a bajo coste |

---

## 9. Resumen

| Concepto | Clave |
|---|---|
| LLM | Modelo entrenado para predecir el siguiente token |
| Token | Unidad mínima de texto (~0.75 palabras) |
| Embedding | Representación vectorial del significado |
| Transformer | Arquitectura basada en mecanismo de atención |
| Atención | Permite relacionar tokens entre sí en la secuencia |
| Temperatura | Controla la aleatoriedad de la generación |
| Pre-entrenamiento | Aprender de texto masivo sin supervisión |
| RLHF | Alineación con preferencias humanas mediante RL |
| Alucinación | Texto plausible pero factualmente incorrecto |

---

**Siguiente:** [02 — Prompt Engineering](./02-prompt-engineering.md)
