# 03 — Historia y evolución de la Inteligencia Artificial

> **Bloque:** Fundamentos · **Nivel:** Introductorio · **Tiempo estimado:** 20 min

---

## Índice

1. [Los orígenes (1940–1955)](#1-los-orígenes-19401955)
2. [El nacimiento oficial (1956)](#2-el-nacimiento-oficial-1956)
3. [El primer invierno de la IA (1974–1980)](#3-el-primer-invierno-de-la-ia-19741980)
4. [La era de los sistemas expertos (1980–1987)](#4-la-era-de-los-sistemas-expertos-19801987)
5. [El segundo invierno (1987–1993)](#5-el-segundo-invierno-19871993)
6. [El renacimiento: ML y Big Data (1993–2012)](#6-el-renacimiento-ml-y-big-data-19932012)
7. [La revolución del Deep Learning (2012–2017)](#7-la-revolución-del-deep-learning-20122017)
8. [La era de los Transformers y LLMs (2017–hoy)](#8-la-era-de-los-transformers-y-llms-2017hoy)
9. [Línea del tiempo](#9-línea-del-tiempo)

---

## 1. Los orígenes (1940–1955)

La IA no surge de la nada. Sus raíces están en varias disciplinas que convergieron a mediados del siglo XX.

### Alan Turing y la computación (1936–1950)

**Alan Turing** publicó en 1936 el concepto de la *Máquina de Turing*, un modelo teórico de computación universal que sentó las bases matemáticas de la informática moderna.

En 1950 publicó el influyente paper *"Computing Machinery and Intelligence"*, donde planteaba la pregunta:

> *"¿Pueden las máquinas pensar?"*

Para evitar la ambigüedad filosófica del término "pensar", propuso el **Test de Turing**: si un humano no puede distinguir, en una conversación escrita, si su interlocutor es humano o máquina, la máquina puede considerarse "inteligente".

### McCulloch & Pitts: la primera neurona artificial (1943)

Warren McCulloch y Walter Pitts publicaron el primer modelo matemático de una neurona artificial, inspirado en la neurociencia. Demostraron que redes de neuronas simples podían computar cualquier función lógica.

### Norbert Wiener y la Cibernética (1948)

Norbert Wiener publicó *Cybernetics*, que estudia los sistemas de control y comunicación en animales y máquinas. Introdujo conceptos como **retroalimentación (feedback)** que serían fundamentales para la IA.

---

## 2. El nacimiento oficial (1956)

### La conferencia de Dartmouth

En el verano de **1956**, John McCarthy organizó una conferencia en el Dartmouth College (New Hampshire, EEUU) junto a Marvin Minsky, Claude Shannon y otros investigadores.

McCarthy propuso el nombre **"Inteligencia Artificial"** y la conferencia estableció los objetivos fundacionales del campo:

> *"Todo aspecto del aprendizaje o cualquier característica de la inteligencia puede ser descrita con suficiente precisión como para que se pueda construir una máquina que la simule."*

### Los años de optimismo (1956–1974)

Los primeros programas fueron sorprendentes para la época:

| Año | Hito |
|---|---|
| 1956 | Logic Theorist (Newell & Simon): primer programa de IA, demostraba teoremas matemáticos |
| 1957 | Perceptrón (Frank Rosenblatt): primer modelo de neurona artificial entrenable |
| 1958 | LISP: lenguaje de programación diseñado para IA, usado durante décadas |
| 1965 | ELIZA (Joseph Weizenbaum, MIT): primer chatbot, simulaba a un psicoterapeuta |
| 1969 | Shakey: primer robot móvil con razonamiento |

Las predicciones eran tremendamente optimistas. Minsky llegó a afirmar en 1967:

> *"En una generación, el problema de la creación de inteligencia artificial estará resuelto."*

---

## 3. El primer invierno de la IA (1974–1980)

La realidad no cumplió las expectativas. Los problemas resultaron mucho más difíciles de lo previsto.

### Causas del primer invierno

**Limitaciones del Perceptrón:** En 1969, Minsky y Papert publicaron *Perceptrons*, demostrando que el modelo no podía resolver problemas no lineales simples (como la función XOR). La financiación para redes neuronales se desplomó.

**Explosión combinatoria:** Los algoritmos de búsqueda funcionaban en ejemplos simples pero no escalaban. El número de posibilidades a explorar crecía exponencialmente.

**Limitaciones de hardware:** Los ordenadores de la época eran demasiado lentos y tenían poca memoria.

**Promesas incumplidas:** Los organismos financiadores (DARPA en EEUU, SRC en Reino Unido) cortaron fondos tras años de resultados pobres.

---

## 4. La era de los sistemas expertos (1980–1987)

La IA resurgió con un enfoque diferente: en lugar de buscar inteligencia general, se construyeron sistemas especializados.

### ¿Qué es un sistema experto?

Un programa que codifica el conocimiento de expertos humanos en reglas IF-THEN para resolver problemas en un dominio específico.

```
SI el paciente tiene fiebre > 38.5°C
Y tiene rigidez de nuca
Y tiene fotofobia
ENTONCES sospechar meningitis (confianza: 0.85)
```

### Sistemas expertos destacados

| Sistema | Año | Dominio |
|---|---|---|
| MYCIN | 1976 | Diagnóstico de infecciones bacterianas |
| XCON (R1) | 1980 | Configuración de ordenadores DEC (ahorraba $40M/año) |
| PROSPECTOR | 1978 | Exploración geológica |
| DENDRAL | 1965–1983 | Análisis de estructuras moleculares |

El mercado de IA alcanzó los **$1.000 millones** en 1985.

---

## 5. El segundo invierno (1987–1993)

Los sistemas expertos tenían limitaciones críticas:

- **Fragilidad:** Funcionaban bien en su dominio pero fallaban ante casos ligeramente distintos
- **Mantenimiento costoso:** Actualizar las reglas era caro y lento
- **Sin aprendizaje:** No mejoraban solos con la experiencia
- **Colapso del mercado LISP:** Las máquinas especializadas en LISP perdieron frente a los PCs convencionales

En 1987 el mercado de hardware especializado para IA colapsó, arrastrando la financiación del sector.

---

## 6. El renacimiento: ML y Big Data (1993–2012)

### El enfoque estadístico gana terreno

En lugar de codificar reglas, los investigadores apostaron por dejar que los algoritmos aprendieran de datos. Varios avances clave:

| Año | Hito |
|---|---|
| 1997 | **Deep Blue** (IBM) derrota a Garry Kasparov en ajedrez |
| 1998 | **LeNet** (Yann LeCun): CNN para reconocimiento de dígitos escritos a mano (base de los cajeros ATM) |
| 2002 | **Roomba**: primer robot doméstico de éxito comercial |
| 2006 | Geoffrey Hinton publica el paper que relanza las redes neuronales profundas (Deep Belief Networks) |
| 2009 | **ImageNet**: base de datos de 14 millones de imágenes etiquetadas, clave para el despegue del DL |
| 2011 | **Watson** (IBM) gana a humanos en Jeopardy! |
| 2011 | **Siri** (Apple): primer asistente de voz en un smartphone masivo |

### El papel del Big Data e Internet

La explosión de Internet generó cantidades masivas de datos etiquetados de forma natural (búsquedas, clicks, imágenes, texto). Esto proporcionó el "combustible" que el ML necesitaba.

---

## 7. La revolución del Deep Learning (2012–2017)

### AlexNet: el punto de inflexión (2012)

En el concurso **ImageNet LSVRC 2012**, el modelo **AlexNet** (Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton) redujo el error de clasificación de imágenes del 26% al **15,3%**, una mejora sin precedentes.

La clave fue:
- Redes neuronales profundas (8 capas)
- Entrenamiento en GPUs (mucho más rápido)
- Técnicas nuevas: ReLU, Dropout

Desde entonces, el Deep Learning dominó la IA.

### Hitos del período

| Año | Hito |
|---|---|
| 2012 | AlexNet revoluciona la visión por computador |
| 2014 | **GANs** (Ian Goodfellow): generación de imágenes realistas |
| 2014 | **Word2Vec** (Google): representaciones vectoriales de palabras |
| 2015 | **ResNet** (Microsoft): redes de 152 capas, supera a humanos en ImageNet |
| 2016 | **AlphaGo** (DeepMind) derrota al campeón mundial de Go, Lee Sedol |
| 2016 | TensorFlow (Google) y PyTorch (Facebook) se popularizan |

---

## 8. La era de los Transformers y LLMs (2017–hoy)

### "Attention Is All You Need" (2017)

El paper publicado por investigadores de Google introdujo la arquitectura **Transformer**, basada en el mecanismo de **atención**. Cambió para siempre el campo del NLP y, posteriormente, de toda la IA.

### La escalada de los modelos de lenguaje

| Año | Modelo | Parámetros | Organización |
|---|---|---|---|
| 2018 | BERT | 340M | Google |
| 2019 | GPT-2 | 1.5B | OpenAI |
| 2020 | GPT-3 | 175B | OpenAI |
| 2021 | PaLM | 540B | Google |
| 2022 | ChatGPT | ~175B | OpenAI |
| 2023 | GPT-4 | ~1T (est.) | OpenAI |
| 2023 | Claude 2 | — | Anthropic |
| 2024 | Gemini Ultra | — | Google |
| 2024 | Claude 3 Opus | — | Anthropic |
| 2025 | Claude 3.5 / GPT-4o / Llama 3 | — | Varios |

### La democratización de la IA (2022–hoy)

**Noviembre 2022:** El lanzamiento de **ChatGPT** marcó un antes y un después. En 5 días alcanzó 1 millón de usuarios; en 2 meses, 100 millones. Nunca una tecnología había sido adoptada tan rápido.

A partir de 2023 se produce una aceleración sin precedentes:

- **Modelos open-source** (Llama, Mistral, Falcon) democratizan el acceso
- **Multimodalidad:** los modelos procesan texto, imagen, audio y vídeo
- **Agentes de IA:** sistemas que planifican y ejecutan tareas autónomamente
- **Integración empresarial:** copilots en Office, GitHub, Salesforce, etc.
- **IA en dispositivos:** modelos pequeños que corren en móviles y portátiles

---

## 9. Línea del tiempo

```
1936  Turing: Máquina de Turing
1943  McCulloch & Pitts: neurona artificial
1950  Turing: Test de Turing
1956  Dartmouth: nace oficialmente la IA
1957  Perceptrón (Rosenblatt)
1965  ELIZA: primer chatbot
      │
      ▼ PRIMER INVIERNO (1974–1980)
      │
1980  Sistemas expertos: boom comercial
      │
      ▼ SEGUNDO INVIERNO (1987–1993)
      │
1997  Deep Blue vence a Kasparov
2006  Hinton relanza el Deep Learning
2009  ImageNet
2011  Watson, Siri
      │
2012  AlexNet ← PUNTO DE INFLEXIÓN
2014  GANs
2016  AlphaGo vence a Lee Sedol
      │
2017  Transformer ("Attention Is All You Need") ← CAMBIO DE ERA
2020  GPT-3
      │
2022  ChatGPT ← ADOPCIÓN MASIVA
2023  GPT-4, Claude 2, Gemini
2024  Claude 3, multimodalidad generalizada
2025  Agentes, modelos en dispositivos, IA en todo
```

---

**Anterior:** [02 — Tipos de IA](./02-tipos-de-ia.md) · **Siguiente bloque:** [LLMs y modelos de lenguaje](../llms/)
