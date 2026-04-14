# 03 — Fine-tuning vs RAG

> **Bloque:** LLMs · **Nivel:** Intermedio · **Tiempo estimado:** 25 min

---

## Índice

1. [El problema: adaptar un LLM a tu dominio](#1-el-problema-adaptar-un-llm-a-tu-dominio)
2. [¿Qué es el fine-tuning?](#2-qué-es-el-fine-tuning)
3. [¿Qué es RAG?](#3-qué-es-rag)
4. [Comparativa detallada](#4-comparativa-detallada)
5. [Cuándo usar cada uno](#5-cuándo-usar-cada-uno)
6. [Otras estrategias](#6-otras-estrategias)
7. [Flujo de decisión](#7-flujo-de-decisión)
8. [Resumen](#8-resumen)

---

## 1. El problema: adaptar un LLM a tu dominio

Un LLM de propósito general sabe muchas cosas, pero no sabe:

- El contenido de tus documentos internos
- La terminología específica de tu sector
- Los procedimientos concretos de tu empresa
- Información posterior a su fecha de corte

Para que el modelo trabaje con esta información, tienes dos grandes estrategias: **Fine-tuning** y **RAG**. Elegir bien entre ellas puede marcar la diferencia entre un proyecto exitoso y uno fallido.

---

## 2. ¿Qué es el fine-tuning?

El **fine-tuning** (ajuste fino) consiste en continuar el entrenamiento de un modelo pre-entrenado con un conjunto de datos específico de tu dominio.

```
Modelo base (pre-entrenado)
       ↓
   + tus datos
       ↓
Modelo ajustado (fine-tuned)
```

### Cómo funciona

1. Preparas un dataset de pares (instrucción → respuesta ideal) en tu dominio
2. Entrenas el modelo con ese dataset durante algunas épocas
3. Los pesos del modelo se ajustan para priorizar el comportamiento deseado
4. Despliegas el nuevo modelo ajustado

### Tipos de fine-tuning

| Tipo | Descripción | Coste |
|---|---|---|
| **Full fine-tuning** | Se actualizan todos los parámetros del modelo | Muy alto |
| **LoRA** (Low-Rank Adaptation) | Solo se entrenan matrices de bajo rango añadidas | Bajo-Medio |
| **QLoRA** | LoRA con cuantización (4-bit) | Muy bajo |
| **Instruction tuning** | Se entrena para seguir instrucciones específicas | Medio |

**LoRA** es el método más popular para fine-tuning de modelos grandes: es eficiente, barato, y los resultados son muy buenos.

### Ejemplo de dataset para fine-tuning

```json
[
  {
    "instruction": "¿Cuál es el proceso de reclamación de garantía?",
    "output": "Para reclamar la garantía de un producto, debe seguir estos pasos: 1) Contactar con el SAT en el plazo de 15 días desde la detección del defecto..."
  },
  {
    "instruction": "¿Qué documentación necesito para dar de alta a un empleado?",
    "output": "Para dar de alta a un empleado necesita: DNI o NIE, número de afiliación a la Seguridad Social, datos bancarios..."
  }
]
```

### Cuánto datos se necesita

- Para cambiar el **estilo o formato:** 100-500 ejemplos
- Para aprender **terminología específica:** 1.000-5.000 ejemplos
- Para cambiar **comportamiento profundo:** 10.000+ ejemplos

---

## 3. ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es una técnica que combina un sistema de búsqueda con un LLM: en lugar de que el modelo "recuerde" la información, la **recupera** en tiempo real y la incluye en el contexto.

```
Pregunta del usuario
       ↓
Sistema de búsqueda (vector store)
       ↓
Fragmentos relevantes de tus documentos
       ↓
Prompt = [Instrucción] + [Fragmentos recuperados] + [Pregunta]
       ↓
LLM genera respuesta basada en el contexto
```

### Componentes de un sistema RAG

#### 1. Ingesta de documentos (indexing)

```
Documentos PDF, Word, HTML...
       ↓
Chunking (dividir en fragmentos de ~500 tokens)
       ↓
Embedding de cada fragmento (vector numérico)
       ↓
Almacenamiento en una base de datos vectorial
```

#### 2. Recuperación (retrieval)

```
Pregunta del usuario
       ↓
Embedding de la pregunta
       ↓
Búsqueda por similitud semántica en la BD vectorial
       ↓
Top-K fragmentos más relevantes
```

#### 3. Generación (generation)

```python
prompt = f"""
Responde la pregunta basándote únicamente en el siguiente contexto.
Si la respuesta no está en el contexto, dilo explícitamente.

Contexto:
{fragmentos_recuperados}

Pregunta: {pregunta_usuario}
"""

respuesta = llm.generate(prompt)
```

### Bases de datos vectoriales populares

| Base de datos | Tipo | Destacado por |
|---|---|---|
| **ChromaDB** | Open source, embebida | Facilidad de uso, ideal para prototipos |
| **Pinecone** | SaaS gestionado | Escala, sin mantenimiento |
| **Weaviate** | Open source | Filtros híbridos (semántico + keyword) |
| **Qdrant** | Open source | Alto rendimiento, muy activo |
| **pgvector** | Extensión PostgreSQL | Si ya tienes Postgres |
| **FAISS** | Librería (Meta) | Muy eficiente para búsqueda en memoria |

---

## 4. Comparativa detallada

| Criterio | Fine-tuning | RAG |
|---|---|---|
| **Conocimiento actualizable** | No (hay que re-entrenar) | Sí (se actualizan los documentos) |
| **Conocimiento verificable** | Difícil (está en los pesos) | Fácil (se pueden citar las fuentes) |
| **Coste inicial** | Alto (entrenamiento) | Medio (infraestructura) |
| **Coste por consulta** | Bajo (modelo cargado) | Medio (búsqueda + LLM) |
| **Latencia** | Baja | Media (búsqueda añade tiempo) |
| **Calidad con datos escasos** | Baja | Media-alta |
| **Alucinaciones** | No se reducen necesariamente | Se reducen (contexto real) |
| **Privacidad de datos** | Los datos se "meten" en el modelo | Los datos permanecen en tu BD |
| **Complejidad técnica** | Alta | Media |
| **Control de fuentes** | No | Sí (puedes citar el documento exacto) |

---

## 5. Cuándo usar cada uno

### Usa Fine-tuning cuando...

- Necesitas **cambiar el estilo o formato** de las respuestas de forma consistente
- Quieres que el modelo hable con la **voz y tono de tu marca**
- Necesitas que el modelo conozca **terminología muy específica** y poco común
- La **latencia es crítica** y no puedes permitirte el overhead de la búsqueda
- El conocimiento a incorporar **no cambia** (o cambia muy raramente)
- Tienes un **dataset grande y de calidad** (mínimo 1.000 ejemplos buenos)

**Ejemplos típicos:**
- Modelo de atención al cliente que responde siempre en el tono de la empresa
- Clasificador de documentos legales con terminología específica
- Asistente de codificación entrenado en el estilo de código de tu empresa

---

### Usa RAG cuando...

- Tu conocimiento **cambia frecuentemente** (precios, normativas, documentación)
- Necesitas que el modelo **cite fuentes** verificables
- Trabajas con **muchos documentos** que no cabrían en el contexto
- Quieres **reducir alucinaciones** anclando las respuestas a documentos reales
- Tienes **pocos ejemplos** para fine-tuning
- Necesitas **control de acceso** por documento o usuario
- Quieres **auditar** qué información usó el modelo para responder

**Ejemplos típicos:**
- Chatbot sobre base de conocimiento interna (manual de empleados, procedimientos)
- Asistente jurídico sobre legislación actualizada
- Q&A sobre catálogo de productos
- Soporte técnico basado en documentación

---

### ¿Y si necesito los dos?

**RAG + Fine-tuning** es posible y a veces la mejor opción:

- Fine-tuning para el **estilo y comportamiento** del modelo
- RAG para el **conocimiento actualizable**

Ejemplo: un asistente legal que habla siempre con el tono formal de tu despacho (fine-tuning) y consulta la legislación vigente (RAG).

---

## 6. Otras estrategias

### Prompt Engineering (sin fine-tuning ni RAG)

Para casos simples, un buen system prompt puede ser suficiente:

```
Eres el asistente virtual de ACME Corp. Respondes siempre en español formal.
Solo respondes preguntas relacionadas con nuestros productos y servicios.
Si no sabes algo, remite al usuario a soporte@acme.com
```

**Cuándo es suficiente:** comportamiento simple, volumen bajo, sin necesidad de conocimiento específico extenso.

### Few-shot en el contexto

Para dominios pequeños, incluir ejemplos directamente en el prompt puede ser más barato que el fine-tuning.

### Function Calling / Tool Use

En lugar de que el modelo "sepa" los datos, le das herramientas para consultarlos en tiempo real (APIs, bases de datos, calculadoras...). Similar a RAG pero con mayor flexibilidad.

---

## 7. Flujo de decisión

```
¿El problema se resuelve con un buen prompt?
      │
      ├── SÍ → Usa prompt engineering (más simple y barato)
      │
      └── NO
           │
           ├── ¿El conocimiento cambia frecuentemente?
           │        │
           │        ├── SÍ → RAG
           │        │
           │        └── NO
           │              │
           │              ├── ¿Necesitas cambiar el estilo/comportamiento base?
           │              │        │
           │              │        ├── SÍ → Fine-tuning (o Fine-tuning + RAG)
           │              │        │
           │              │        └── NO → RAG (más flexible)
           │
           └── ¿Tienes más de 1.000 ejemplos de calidad?
                    │
                    ├── SÍ → Fine-tuning es viable
                    └── NO → RAG o prompt engineering
```

---

## 8. Resumen

| | Fine-tuning | RAG |
|---|---|---|
| **Qué hace** | Modifica los pesos del modelo | Recupera contexto externo en tiempo real |
| **Mejor para** | Estilo, tono, terminología fija | Conocimiento dinámico y verificable |
| **Requisito clave** | Dataset grande y de calidad | Base de datos vectorial + documentos |
| **Actualizar conocimiento** | Re-entrenar | Actualizar documentos |
| **Citar fuentes** | No | Sí |

---

**Anterior:** [02 — Prompt Engineering](./02-prompt-engineering.md) · **Siguiente bloque:** [APIs de IA](../apis/)
