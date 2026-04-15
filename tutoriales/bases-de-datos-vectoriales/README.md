# Bloque 13 — Bases de Datos Vectoriales

> **Bloque:** Bases de datos vectoriales · **Nivel:** Avanzado · **Tiempo estimado:** 15 min

---

## Índice

1. [¿Qué son las bases de datos vectoriales?](#qué-son-las-bases-de-datos-vectoriales)
2. [Vectores vs SQL tradicional](#vectores-vs-sql-tradicional)
3. [Tutoriales del bloque](#tutoriales-del-bloque)
4. [Requisitos de instalación](#requisitos-de-instalación)
5. [Cómo está organizado este bloque](#cómo-está-organizado-este-bloque)

---

## ¿Qué son las bases de datos vectoriales?

Una base de datos vectorial almacena representaciones numéricas de alta dimensión —llamadas **embeddings**— que codifican el significado semántico de textos, imágenes o cualquier otro tipo de dato. A diferencia de las bases de datos tradicionales, que buscan coincidencias exactas, las bases de datos vectoriales permiten encontrar el contenido *más similar* a una consulta, incluso cuando no existe ninguna palabra en común.

Este bloque cubre las herramientas más utilizadas en producción: **pgvector** (extensión para PostgreSQL), **Pinecone** y **Weaviate** en la nube, junto con técnicas avanzadas como **reranking** y patrones sofisticados de RAG.

---

## Vectores vs SQL tradicional

Elegir entre una base de datos vectorial y una base de datos relacional tradicional depende del tipo de consulta que necesitas hacer:

| Criterio | SQL tradicional | Base de datos vectorial |
|---|---|---|
| **Tipo de búsqueda** | Coincidencia exacta (`=`, `LIKE`, `IN`) | Similitud semántica (coseno, producto escalar) |
| **Pregunta que responde** | "¿Qué registros tienen exactamente este valor?" | "¿Qué contenido tiene un significado similar a esto?" |
| **Ejemplo de consulta** | `SELECT * FROM facturas WHERE cliente_id = 42` | "Encuentra contratos parecidos a este párrafo" |
| **Escalabilidad** | Excelente para datos estructurados y filtros exactos | Necesita índices especializados (HNSW, IVF) para millones de vectores |
| **Coste de infraestructura** | Bajo si ya tienes PostgreSQL | Variable: pgvector es gratuito; Pinecone/Weaviate tienen costes por uso |
| **Joins y transacciones** | Soporte completo (ACID) | Limitado o inexistente en soluciones cloud especializadas |
| **Casos de uso ideales** | ERP, finanzas, inventarios, reportes | Buscadores semánticos, RAG, recomendaciones, detección de duplicados |

**Regla práctica:**

- Si tus datos son **estructurados** y buscas por **valores exactos** → SQL.
- Si necesitas buscar por **significado** o **similitud** → vectores.
- Si necesitas **ambas cosas** (filtrar por fecha Y buscar por similitud) → **pgvector** (combina lo mejor de los dos mundos dentro de PostgreSQL) o bases de datos vectoriales con soporte de filtros de metadata como Pinecone o Weaviate.

---

## Tutoriales del bloque

| # | Archivo | Tema | Tiempo estimado |
|---|---|---|---|
| 01 | [01-pgvector.md](./01-pgvector.md) | Búsqueda vectorial en PostgreSQL con pgvector | 45 min |
| 02 | [02-pinecone-weaviate.md](./02-pinecone-weaviate.md) | Bases de datos vectoriales en cloud — Pinecone y Weaviate | 40 min |
| 03 | [03-reranking.md](./03-reranking.md) | Reranking — mejorar la calidad del RAG | 35 min |
| 04 | [04-rag-avanzado.md](./04-rag-avanzado.md) | Técnicas avanzadas de RAG | 50 min |

---

## Requisitos de instalación

Instala todas las dependencias del bloque con un solo comando:

```bash
pip install chromadb pgvector psycopg2-binary pinecone-client weaviate-client
```

Dependencias adicionales según el tutorial:

```bash
# Para reranking
pip install sentence-transformers cohere

# Para métricas de evaluación de RAG
pip install ragas datasets

# Para embeddings de OpenAI
pip install openai
```

Variables de entorno necesarias:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export PINECONE_API_KEY="..."        # Solo para tutorial 02
export COHERE_API_KEY="..."          # Solo para tutorial 03
```

---

## Cómo está organizado este bloque

Cada tutorial es independiente, pero el orden recomendado es lineal:

1. **pgvector** es el punto de entrada más sencillo si ya tienes PostgreSQL. No requiere infraestructura adicional.
2. **Pinecone y Weaviate** muestran cuándo vale la pena pasar a una solución cloud especializada.
3. **Reranking** mejora la calidad de cualquier pipeline RAG existente con un paso adicional de puntuación.
4. **RAG avanzado** reúne todas las técnicas en patrones de producción completos.

Si eres nuevo en bases de datos vectoriales, empieza por el tutorial 01. Si ya tienes experiencia con RAG básico, puedes ir directamente al 03 o al 04.

---

**Siguiente:** [01 — Búsqueda vectorial con pgvector](./01-pgvector.md)
