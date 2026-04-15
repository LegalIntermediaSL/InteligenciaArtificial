# 03 — Reranking: mejorar la calidad del RAG

> **Bloque:** Bases de datos vectoriales · **Nivel:** Avanzado · **Tiempo estimado:** 35 min

---

## Índice

1. [El problema del RAG naïve](#1-el-problema-del-rag-naïve)
2. [Qué es el reranking](#2-qué-es-el-reranking)
3. [Cross-Encoders con sentence-transformers](#3-cross-encoders-con-sentence-transformers)
4. [Reranking con Cohere](#4-reranking-con-cohere)
5. [Métricas de calidad](#5-métricas-de-calidad)
6. [Pipeline RAG completo con reranking](#6-pipeline-rag-completo-con-reranking)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. El problema del RAG naïve

En un sistema RAG básico, la recuperación de documentos funciona así:

1. El usuario escribe una pregunta.
2. Se genera el embedding de la pregunta.
3. Se buscan los K documentos con mayor similitud coseno.
4. Se envían esos K documentos a Claude como contexto.

Este enfoque tiene un fallo fundamental: **la similitud coseno entre embeddings no es lo mismo que la relevancia para responder una pregunta específica**.

**Ejemplo concreto:**

Supón que tienes una base de documentos legales y el usuario pregunta:

> "¿Cuáles son las consecuencias de incumplir un contrato de arrendamiento?"

Los embeddings podrían devolver, en este orden:

| Posición | Documento | Similitud coseno |
|---|---|---|
| 1 | "Tipos de contratos de arrendamiento en España" | 0.89 |
| 2 | "Obligaciones del arrendatario" | 0.87 |
| 3 | "Penalizaciones por rescisión anticipada" | 0.85 |
| 4 | "Consecuencias legales del impago de renta" | 0.84 |
| 5 | "Historia del derecho inmobiliario en España" | 0.82 |

El documento más relevante para responder la pregunta es el **número 4** (consecuencias legales del impago), pero está en cuarta posición. El documento 5 es casi irrelevante, pero tiene una similitud coseno razonablemente alta porque comparte vocabulario del dominio legal.

Si enviamos a Claude solo los top 3, **excluimos el documento más útil**. Si enviamos los 5, incluimos ruido que puede confundir la respuesta.

El **reranking** soluciona esto.

---

## 2. Qué es el reranking

El reranking añade una segunda etapa de puntuación más precisa pero más costosa computacionalmente:

```
ETAPA 1 — Retrieve (rápido, aproximado)
┌─────────────────────────────────────────┐
│  Base de datos vectorial                │
│  Millones de documentos                 │
│                                         │
│  Query ──embedding──► similitud coseno  │
│                        └──► top 20      │
└─────────────────────────────────────────┘
                    │
                    │ 20 candidatos
                    ▼
ETAPA 2 — Rerank (lento, preciso)
┌─────────────────────────────────────────┐
│  Cross-Encoder o modelo de reranking    │
│                                         │
│  Evalúa cada par (query, documento)     │
│  Score de relevancia 0.0 → 1.0          │
│                        └──► top 5       │
└─────────────────────────────────────────┘
                    │
                    │ 5 documentos reordenados por relevancia real
                    ▼
ETAPA 3 — Generate
┌─────────────────────────────────────────┐
│  Claude con contexto de alta calidad    │
└─────────────────────────────────────────┘
```

La clave: en la etapa 2, el modelo ve **la query y el documento juntos** en el mismo input, lo que le permite evaluar la relevancia real del par, no solo la similitud independiente de sus embeddings.

---

## 3. Cross-Encoders con sentence-transformers

Los Cross-Encoders son modelos que reciben `[query, documento]` concatenados y devuelven un score de relevancia. Son más lentos que los embeddings, pero mucho más precisos:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Modelo pre-entrenado para reranking (MS MARCO — entrenado en búsqueda web)
# Opciones: ms-marco-MiniLM-L-6-v2 (rápido), ms-marco-electra-base (más preciso)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def reranking_con_cross_encoder(
    query: str,
    documentos: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Aplica reranking a una lista de documentos usando un Cross-Encoder.

    Args:
        query: La pregunta del usuario.
        documentos: Lista de dicts con al menos la clave 'contenido'.
        top_k: Cuántos documentos devolver tras el reranking.

    Returns:
        Lista de documentos ordenados por relevancia real, con su nuevo score.
    """
    # Crear pares (query, contenido_del_documento)
    pares = [(query, doc["contenido"]) for doc in documentos]

    # El Cross-Encoder puntúa cada par independientemente
    scores = reranker.predict(pares)

    # Asociar scores a documentos y ordenar de mayor a menor
    documentos_con_score = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(documentos, scores)
    ]

    documentos_con_score.sort(key=lambda x: x["rerank_score"], reverse=True)

    return documentos_con_score[:top_k]


# Ejemplo: 10 candidatos recuperados por similitud vectorial
candidatos = [
    {
        "titulo": "Tipos de contratos de arrendamiento",
        "contenido": "Existen varios tipos de contratos de arrendamiento en España: vivienda, local de negocio y arrendamiento de temporada.",
        "similitud_vectorial": 0.89
    },
    {
        "titulo": "Obligaciones del arrendatario",
        "contenido": "El arrendatario debe pagar la renta puntualmente, conservar el inmueble y no realizar obras sin permiso.",
        "similitud_vectorial": 0.87
    },
    {
        "titulo": "Penalizaciones por rescisión anticipada",
        "contenido": "Si el arrendatario rescinde el contrato antes del plazo, puede deber hasta dos mensualidades al arrendador.",
        "similitud_vectorial": 0.85
    },
    {
        "titulo": "Consecuencias legales del impago de renta",
        "contenido": "El impago de renta da derecho al arrendador a iniciar un procedimiento de desahucio y reclamar las rentas adeudadas con intereses.",
        "similitud_vectorial": 0.84
    },
    {
        "titulo": "Historia del derecho inmobiliario",
        "contenido": "El derecho inmobiliario español tiene sus raíces en el derecho romano y ha evolucionado significativamente desde el siglo XIX.",
        "similitud_vectorial": 0.82
    },
]

query = "¿Cuáles son las consecuencias de incumplir un contrato de arrendamiento?"

print("ANTES del reranking (orden por similitud vectorial):")
for i, doc in enumerate(candidatos, 1):
    print(f"  {i}. [{doc['similitud_vectorial']:.2f}] {doc['titulo']}")

resultados_reranked = reranking_con_cross_encoder(query, candidatos, top_k=3)

print(f"\nDESPUÉS del reranking (top 3 por relevancia real):")
for i, doc in enumerate(resultados_reranked, 1):
    print(f"  {i}. [{doc['rerank_score']:.4f}] {doc['titulo']}")
```

**Resultado esperado**: el documento "Consecuencias legales del impago de renta" sube a la primera posición, y "Historia del derecho inmobiliario" desaparece del top 3.

---

## 4. Reranking con Cohere

Cohere ofrece un servicio de reranking en la nube, especialmente útil cuando no quieres gestionar modelos locales y necesitas soporte multilingüe:

```bash
pip install cohere
```

```python
import os
import cohere

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])


def reranking_con_cohere(
    query: str,
    documentos: list[dict],
    top_k: int = 5,
    modelo: str = "rerank-multilingual-v3.0"
) -> list[dict]:
    """
    Reranking usando la API de Cohere.

    Modelos disponibles:
    - rerank-multilingual-v3.0: soporta 100+ idiomas, ideal para español
    - rerank-english-v3.0: solo inglés, ligeramente más preciso en ese idioma
    - rerank-multilingual-v2.0: versión anterior, más barata

    Args:
        query: La pregunta del usuario.
        documentos: Lista de dicts con clave 'contenido'.
        top_k: Número de documentos a devolver.
        modelo: Modelo de reranking de Cohere.

    Returns:
        Lista de documentos reordenados con score de relevancia.
    """
    textos = [doc["contenido"] for doc in documentos]

    respuesta = co.rerank(
        query=query,
        documents=textos,
        model=modelo,
        top_n=top_k,
        return_documents=True
    )

    resultados = []
    for resultado in respuesta.results:
        doc_original = documentos[resultado.index]
        resultados.append({
            **doc_original,
            "rerank_score": resultado.relevance_score,
            "posicion_original": resultado.index
        })

    return resultados


def comparar_rerankers(query: str, candidatos: list[dict]):
    """Compara los resultados de Cross-Encoder local vs Cohere cloud."""
    print(f"Query: '{query}'\n")

    print("Cross-Encoder local (ms-marco-MiniLM-L-6-v2):")
    local = reranking_con_cross_encoder(query, candidatos, top_k=3)
    for i, doc in enumerate(local, 1):
        print(f"  {i}. [{doc['rerank_score']:.4f}] {doc['titulo']}")

    print("\nCohere rerank-multilingual-v3.0:")
    cloud = reranking_con_cohere(query, candidatos, top_k=3)
    for i, doc in enumerate(cloud, 1):
        print(f"  {i}. [{doc['rerank_score']:.4f}] {doc['titulo']}")


if __name__ == "__main__":
    comparar_rerankers(query, candidatos)
```

**Ventajas de Cohere reranking:**

- Sin gestión de modelos locales ni GPU.
- `rerank-multilingual-v3.0` funciona directamente en español sin ajustes.
- La latencia de red suele ser aceptable (100-300ms por llamada).
- Precio basado en el número de pares query-documento evaluados.

---

## 5. Métricas de calidad

Para saber si el reranking mejora, necesitamos medir con datos reales. Las dos métricas estándar son:

**NDCG** (Normalized Discounted Cumulative Gain): valora tanto la relevancia como la posición. Un documento relevante en posición 1 vale más que en posición 5.

**MRR** (Mean Reciprocal Rank): la media del recíproco de la posición del primer resultado relevante. Simple pero eficaz para búsquedas con una sola respuesta correcta.

```python
import numpy as np
from typing import Any


# Dataset de prueba: lista de (query, documentos_candidatos, id_relevante)
# En producción, este dataset lo construyen evaluadores humanos o anotaciones previas
dataset_prueba = [
    {
        "query": "¿Cuáles son las consecuencias de incumplir un contrato?",
        "candidatos": [
            {"id": "doc-1", "titulo": "Tipos de contratos", "contenido": "Existen varios tipos de contratos en derecho civil."},
            {"id": "doc-2", "titulo": "Penalizaciones por incumplimiento", "contenido": "El incumplimiento contractual puede acarrear indemnización por daños y perjuicios."},
            {"id": "doc-3", "titulo": "Historia del derecho", "contenido": "El derecho civil tiene raíces en el derecho romano."},
            {"id": "doc-4", "titulo": "Rescisión de contratos", "contenido": "La rescisión es la anulación de un contrato por causas previstas en la ley."},
        ],
        "ids_relevantes": ["doc-2", "doc-4"]   # documentos que responden la pregunta
    },
    {
        "query": "¿Cómo protege la empresa los datos personales?",
        "candidatos": [
            {"id": "doc-5", "titulo": "Política de cookies", "contenido": "Usamos cookies para mejorar la experiencia de usuario."},
            {"id": "doc-6", "titulo": "Protección de datos RGPD", "contenido": "Protegemos tus datos según el Reglamento General de Protección de Datos."},
            {"id": "doc-7", "titulo": "Términos de servicio", "contenido": "Al usar nuestro servicio, aceptas nuestros términos y condiciones."},
            {"id": "doc-8", "titulo": "Cifrado de datos", "contenido": "Todos los datos se almacenan con cifrado AES-256."},
        ],
        "ids_relevantes": ["doc-6", "doc-8"]
    },
]


def calcular_ndcg(ids_ordenados: list[str], ids_relevantes: set[str], k: int) -> float:
    """Calcula NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(ids_ordenados[:k]):
        relevancia = 1.0 if doc_id in ids_relevantes else 0.0
        dcg += relevancia / np.log2(i + 2)   # +2 porque i empieza en 0

    # IDCG: el mejor orden posible (todos los relevantes primero)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ids_relevantes), k)))

    return dcg / idcg if idcg > 0 else 0.0


def calcular_mrr(ids_ordenados: list[str], ids_relevantes: set[str]) -> float:
    """Calcula MRR (Mean Reciprocal Rank)."""
    for i, doc_id in enumerate(ids_ordenados):
        if doc_id in ids_relevantes:
            return 1.0 / (i + 1)
    return 0.0


def evaluar_sistema(
    dataset: list[dict],
    usar_reranking: bool = False,
    top_k: int = 4
) -> dict[str, float]:
    """
    Evalúa un sistema de recuperación con y sin reranking.
    Devuelve NDCG@k y MRR promedio sobre el dataset.
    """
    ndcg_scores = []
    mrr_scores = []

    for ejemplo in dataset:
        query = ejemplo["query"]
        candidatos = ejemplo["candidatos"]
        ids_relevantes = set(ejemplo["ids_relevantes"])

        if usar_reranking:
            # Con reranking: reordenamos los candidatos
            reranked = reranking_con_cross_encoder(query, candidatos, top_k=len(candidatos))
            ids_ordenados = [doc["id"] for doc in reranked]
        else:
            # Sin reranking: orden original (simula el orden por similitud vectorial)
            ids_ordenados = [doc["id"] for doc in candidatos]

        ndcg = calcular_ndcg(ids_ordenados, ids_relevantes, k=top_k)
        mrr = calcular_mrr(ids_ordenados, ids_relevantes)

        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)

    return {
        "ndcg_at_k": round(np.mean(ndcg_scores), 4),
        "mrr": round(np.mean(mrr_scores), 4)
    }


if __name__ == "__main__":
    print("Evaluando sistema SIN reranking...")
    sin_reranking = evaluar_sistema(dataset_prueba, usar_reranking=False)
    print(f"  NDCG@4: {sin_reranking['ndcg_at_k']}")
    print(f"  MRR:    {sin_reranking['mrr']}")

    print("\nEvaluando sistema CON reranking (Cross-Encoder)...")
    con_reranking = evaluar_sistema(dataset_prueba, usar_reranking=True)
    print(f"  NDCG@4: {con_reranking['ndcg_at_k']}")
    print(f"  MRR:    {con_reranking['mrr']}")

    mejora_ndcg = ((con_reranking["ndcg_at_k"] - sin_reranking["ndcg_at_k"]) / sin_reranking["ndcg_at_k"]) * 100
    mejora_mrr  = ((con_reranking["mrr"] - sin_reranking["mrr"]) / sin_reranking["mrr"]) * 100

    print(f"\nMejora con reranking:")
    print(f"  NDCG@4: {mejora_ndcg:+.1f}%")
    print(f"  MRR:    {mejora_mrr:+.1f}%")
```

---

## 6. Pipeline RAG completo con reranking

Este es el pipeline de producción recomendado: ChromaDB para la recuperación inicial, Cross-Encoder para el reranking, Claude para la generación.

```python
import os
import chromadb
import anthropic
from openai import OpenAI
from sentence_transformers import CrossEncoder

# Clientes
openai_client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
claude_client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
chroma_client  = chromadb.Client()
cross_encoder  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Colección en ChromaDB
coleccion = chroma_client.get_or_create_collection(
    name="documentos",
    metadata={"hnsw:space": "cosine"}
)


# ── INDEXACIÓN ────────────────────────────────────────────────────────────────

def indexar_documentos(documentos: list[dict]) -> None:
    """Genera embeddings e indexa documentos en ChromaDB."""
    ids        = [doc["id"] for doc in documentos]
    textos     = [f"{doc['titulo']}\n{doc['contenido']}" for doc in documentos]
    metadatas  = [{"titulo": doc["titulo"], "categoria": doc.get("categoria", "")} for doc in documentos]

    # Embeddings en lote (más eficiente)
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )
    embeddings = [item.embedding for item in respuesta.data]

    coleccion.add(
        ids=ids,
        embeddings=embeddings,
        documents=[doc["contenido"] for doc in documentos],
        metadatas=metadatas
    )
    print(f"{len(documentos)} documentos indexados.")


# ── PIPELINE RAG + RERANKING ──────────────────────────────────────────────────

def recuperar_candidatos(query: str, n_candidatos: int = 20) -> list[dict]:
    """Etapa 1: recupera N candidatos de ChromaDB por similitud vectorial."""
    respuesta_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = respuesta_embedding.data[0].embedding

    resultados = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=n_candidatos,
        include=["documents", "metadatas", "distances"]
    )

    candidatos = []
    for i in range(len(resultados["ids"][0])):
        candidatos.append({
            "id":         resultados["ids"][0][i],
            "contenido":  resultados["documents"][0][i],
            "titulo":     resultados["metadatas"][0][i]["titulo"],
            "categoria":  resultados["metadatas"][0][i]["categoria"],
            "distancia":  resultados["distances"][0][i]
        })

    return candidatos


def aplicar_reranking(query: str, candidatos: list[dict], top_k: int = 5) -> list[dict]:
    """Etapa 2: reordena los candidatos por relevancia real."""
    pares  = [(query, doc["contenido"]) for doc in candidatos]
    scores = cross_encoder.predict(pares)

    documentos_con_score = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(candidatos, scores)
    ]

    documentos_con_score.sort(key=lambda x: x["rerank_score"], reverse=True)
    return documentos_con_score[:top_k]


def generar_respuesta(query: str, documentos_reranked: list[dict]) -> str:
    """Etapa 3: genera la respuesta con Claude usando el contexto de alta calidad."""
    contexto_partes = []
    for i, doc in enumerate(documentos_reranked, 1):
        contexto_partes.append(
            f"[Documento {i} — {doc['titulo']} | score: {doc['rerank_score']:.3f}]\n"
            f"{doc['contenido']}"
        )
    contexto = "\n\n---\n\n".join(contexto_partes)

    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="""Eres un asistente experto. Responde basándote únicamente en los 
documentos proporcionados. Cita la fuente entre corchetes [Documento N]. 
Si la información no está en los documentos, indícalo claramente.""",
        messages=[
            {
                "role": "user",
                "content": f"Documentos de referencia:\n\n{contexto}\n\n---\n\nPregunta: {query}"
            }
        ]
    )

    return respuesta.content[0].text


def rag_con_reranking(query: str, n_candidatos: int = 20, top_k_final: int = 5) -> str:
    """
    Pipeline RAG completo con reranking.

    Flujo:
    1. Recuperar N=20 candidatos de ChromaDB (rápido, aproximado)
    2. Reranking con Cross-Encoder → top K=5 (lento, preciso)
    3. Generar respuesta con Claude usando los top K documentos
    """
    print(f"\n[1/3] Recuperando {n_candidatos} candidatos...")
    candidatos = recuperar_candidatos(query, n_candidatos)

    print(f"[2/3] Aplicando reranking → seleccionando top {top_k_final}...")
    top_docs = aplicar_reranking(query, candidatos, top_k=top_k_final)

    print("[3/3] Generando respuesta con Claude...\n")
    return generar_respuesta(query, top_docs)


# ── DEMO ──────────────────────────────────────────────────────────────────────

documentos_ejemplo = [
    {"id": "d1", "titulo": "Contrato arrendamiento",   "contenido": "El arrendador cede el inmueble por 12 meses. El impago da derecho a desahucio.", "categoria": "legal"},
    {"id": "d2", "titulo": "Política de privacidad",   "contenido": "Protegemos tus datos según el RGPD. Cifrado AES-256 en reposo.", "categoria": "legal"},
    {"id": "d3", "titulo": "Informe financiero Q3",    "contenido": "Ingresos de 2.3M€ en Q3, un 15% más que el año anterior.", "categoria": "finanzas"},
    {"id": "d4", "titulo": "Manual de onboarding",     "contenido": "Bienvenido al equipo. Herramientas: Slack, Notion, GitHub.", "categoria": "rrhh"},
    {"id": "d5", "titulo": "Tipos de contratos",       "contenido": "Existen contratos de arrendamiento, prestación de servicios y compraventa.", "categoria": "legal"},
    {"id": "d6", "titulo": "Penalizaciones legales",   "contenido": "El incumplimiento contractual puede conllevar indemnización y resolución del contrato.", "categoria": "legal"},
]

if __name__ == "__main__":
    indexar_documentos(documentos_ejemplo)

    pregunta = "¿Qué ocurre si no pago el alquiler?"
    respuesta = rag_con_reranking(pregunta, n_candidatos=6, top_k_final=3)

    print("=" * 60)
    print(f"Pregunta: {pregunta}")
    print("-" * 60)
    print(respuesta)
```

---

## 7. Extensiones sugeridas

- **Reranking en tiempo real vs por lotes**: para baja latencia, aplica reranking solo al momento de la query. Para pipelines offline (indexación de informes nocturnos), pre-rankea y almacena el orden.
- **Fine-tuning del Cross-Encoder**: si tienes datos de relevancia propios (por ejemplo, clicks de usuarios o evaluaciones humanas), entrena un Cross-Encoder específico para tu dominio con `sentence-transformers`.
- **Reranking con LLM**: en lugar de un Cross-Encoder, usa Claude para puntuar la relevancia de cada par (query, documento). Más caro pero más flexible para dominios muy específicos.
- **Colbert / RAGatouille**: ColBERT es un modelo de reranking que genera múltiples embeddings por token (en lugar de uno por documento), ofreciendo precisión cercana a los Cross-Encoders con mayor velocidad. La biblioteca RAGatouille facilita su integración en Python.
- **Umbral de score mínimo**: filtra documentos con `rerank_score < 0.3` antes de enviarlos a Claude. Si todos los documentos tienen score bajo, responde que no tienes información suficiente.

---

**Siguiente:** [04 — Técnicas avanzadas de RAG](./04-rag-avanzado.md)
