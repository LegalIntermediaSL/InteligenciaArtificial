# 02 — Bases de datos vectoriales en cloud: Pinecone y Weaviate

> **Bloque:** Bases de datos vectoriales · **Nivel:** Avanzado · **Tiempo estimado:** 40 min

---

## Índice

1. [Cuándo usar vectorDB cloud vs pgvector](#1-cuándo-usar-vectordb-cloud-vs-pgvector)
2. [Pinecone: setup y primeros pasos](#2-pinecone-setup-y-primeros-pasos)
3. [Pinecone: búsqueda con metadatos y filtros](#3-pinecone-búsqueda-con-metadatos-y-filtros)
4. [Weaviate: setup local con Docker y cloud](#4-weaviate-setup-local-con-docker-y-cloud)
5. [Weaviate: búsqueda híbrida](#5-weaviate-búsqueda-híbrida)
6. [Comparativa práctica Pinecone vs Weaviate vs ChromaDB](#6-comparativa-práctica-pinecone-vs-weaviate-vs-chromadb)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Cuándo usar vectorDB cloud vs pgvector

La elección entre una base de datos vectorial cloud especializada y pgvector no es trivial. Esta tabla resume los factores más relevantes:

| Criterio | pgvector | Pinecone | Weaviate |
|---|---|---|---|
| **Escala de vectores** | Hasta ~50M con HNSW | Cientos de millones | Decenas de millones |
| **Latencia de búsqueda** | 10-100ms (depende de hardware) | <10ms (infraestructura propia) | 10-50ms |
| **Coste inicial** | Gratuito (usa tu PostgreSQL) | Desde $0 (tier gratuito muy limitado) | Gratuito en local; cloud de pago |
| **Operaciones DevOps** | Alta (tú gestionas Postgres) | Ninguna (SaaS gestionado) | Media (Docker en local o SaaS) |
| **Búsqueda híbrida** | Manual (combinar con `tsvector`) | No nativa (requiere workarounds) | Nativa (vectorial + BM25) |
| **Filtros de metadata** | SQL completo | Filtros JSON con limitaciones | GraphQL con filtros avanzados |
| **Datos relacionales** | Integrado (misma BD) | No | No |
| **Soporte local / offline** | Sí | No (requiere internet) | Sí (Docker) |
| **Ideal para** | Apps ya en PostgreSQL, <50M docs | Startups que escalan rápido, SaaS | Búsqueda híbrida, grafos de conocimiento |

**Regla de decisión:**

- Empezaste con pgvector pero ya superas los 20-30M de vectores y la latencia es inaceptable → **Pinecone**.
- Necesitas búsqueda híbrida (vectorial + texto completo) de forma nativa → **Weaviate**.
- Prototipas localmente o en CI/CD sin internet → **ChromaDB** o **Weaviate en Docker**.
- Tienes datos relacionales que deben vivir junto a los vectores → **pgvector**.

---

## 2. Pinecone: setup y primeros pasos

### Instalación y configuración

```bash
pip install pinecone-client openai
```

Crea una cuenta en [pinecone.io](https://pinecone.io) y obtén tu API key desde el dashboard.

```python
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Clientes
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

NOMBRE_INDICE = "documentos-empresa"
DIMENSION = 1536   # text-embedding-3-small


def crear_indice_si_no_existe():
    """Crea el índice en Pinecone si no existe todavía."""
    indices_existentes = [idx.name for idx in pc.list_indexes()]

    if NOMBRE_INDICE not in indices_existentes:
        pc.create_index(
            name=NOMBRE_INDICE,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Índice '{NOMBRE_INDICE}' creado.")
    else:
        print(f"Índice '{NOMBRE_INDICE}' ya existe.")

    return pc.Index(NOMBRE_INDICE)
```

### Generar embeddings e insertar vectores (upsert)

```python
def generar_embedding(texto: str) -> list[float]:
    """Genera un embedding con OpenAI text-embedding-3-small."""
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding


def insertar_documentos(indice, documentos: list[dict]) -> None:
    """
    Inserta documentos en Pinecone.
    Cada documento necesita un id único, el vector y metadata opcional.
    """
    vectores = []

    for doc in documentos:
        texto = f"{doc['titulo']}\n\n{doc['contenido']}"
        embedding = generar_embedding(texto)

        vectores.append({
            "id": doc["id"],
            "values": embedding,
            "metadata": {          # metadata para filtros posteriores
                "titulo": doc["titulo"],
                "contenido": doc["contenido"],
                "categoria": doc.get("categoria", ""),
                "autor": doc.get("autor", ""),
                "año": doc.get("año", 2024)
            }
        })

    # Pinecone recomienda insertar en lotes de 100
    tamaño_lote = 100
    for i in range(0, len(vectores), tamaño_lote):
        lote = vectores[i:i + tamaño_lote]
        indice.upsert(vectors=lote)
        print(f"Insertados {min(i + tamaño_lote, len(vectores))}/{len(vectores)} vectores")

    print("Upsert completado.")


# Datos de ejemplo
documentos = [
    {
        "id": "doc-001",
        "titulo": "Contrato de arrendamiento 2024",
        "contenido": "El arrendador cede el uso del inmueble por 12 meses.",
        "categoria": "legal",
        "autor": "García López",
        "año": 2024
    },
    {
        "id": "doc-002",
        "titulo": "Informe financiero Q3",
        "contenido": "Los ingresos del tercer trimestre alcanzaron 2.3M€.",
        "categoria": "finanzas",
        "autor": "Ana Martínez",
        "año": 2024
    },
    {
        "id": "doc-003",
        "titulo": "Política de privacidad RGPD",
        "contenido": "Esta política describe cómo protegemos tus datos según el RGPD.",
        "categoria": "legal",
        "autor": "Departamento Legal",
        "año": 2023
    },
]


def buscar_en_pinecone(indice, query: str, top_k: int = 5) -> list[dict]:
    """Búsqueda semántica básica sin filtros."""
    query_embedding = generar_embedding(query)

    resultados = indice.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "id": match.id,
            "score": round(match.score, 4),
            "titulo": match.metadata.get("titulo", ""),
            "contenido": match.metadata.get("contenido", ""),
            "categoria": match.metadata.get("categoria", "")
        }
        for match in resultados.matches
    ]


if __name__ == "__main__":
    indice = crear_indice_si_no_existe()
    insertar_documentos(indice, documentos)

    resultados = buscar_en_pinecone(indice, "información sobre contratos")
    for r in resultados:
        print(f"[{r['score']}] {r['titulo']} ({r['categoria']})")
```

---

## 3. Pinecone: búsqueda con metadatos y filtros

Pinecone permite filtrar por metadata antes de aplicar la búsqueda vectorial. Esto reduce el espacio de búsqueda y mejora la relevancia:

```python
def buscar_con_filtros(
    indice,
    query: str,
    categoria: str | None = None,
    año_minimo: int | None = None,
    top_k: int = 5
) -> list[dict]:
    """
    Búsqueda semántica con filtros de metadata en Pinecone.
    Los filtros usan sintaxis MongoDB-like.
    """
    query_embedding = generar_embedding(query)

    # Construir el filtro dinámicamente
    filtro = {}

    if categoria:
        filtro["categoria"] = {"$eq": categoria}

    if año_minimo:
        filtro["año"] = {"$gte": año_minimo}

    # Si hay múltiples filtros, Pinecone los combina con AND por defecto
    if len(filtro) > 1:
        filtro = {"$and": [{k: v} for k, v in filtro.items()]}

    resultados = indice.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filtro if filtro else None   # None = sin filtro
    )

    return [
        {
            "id": match.id,
            "score": round(match.score, 4),
            "titulo": match.metadata.get("titulo", ""),
            "categoria": match.metadata.get("categoria", ""),
            "autor": match.metadata.get("autor", ""),
            "año": match.metadata.get("año", "")
        }
        for match in resultados.matches
    ]


def demo_filtros(indice):
    """Demuestra distintas combinaciones de filtros."""
    print("=== Búsqueda en categoría 'legal' ===")
    resultados = buscar_con_filtros(
        indice,
        query="documentos sobre protección de datos",
        categoria="legal"
    )
    for r in resultados:
        print(f"  [{r['score']}] {r['titulo']} — {r['autor']} ({r['año']})")

    print("\n=== Búsqueda con año mínimo 2024 ===")
    resultados = buscar_con_filtros(
        indice,
        query="informes y documentos recientes",
        año_minimo=2024
    )
    for r in resultados:
        print(f"  [{r['score']}] {r['titulo']} ({r['año']})")

    print("\n=== Búsqueda combinada: legal + 2024 ===")
    resultados = buscar_con_filtros(
        indice,
        query="contratos y acuerdos",
        categoria="legal",
        año_minimo=2024
    )
    for r in resultados:
        print(f"  [{r['score']}] {r['titulo']} ({r['año']})")


if __name__ == "__main__":
    indice = pc.Index(NOMBRE_INDICE)
    demo_filtros(indice)
```

Operadores de filtro disponibles en Pinecone:

| Operador | Significado | Ejemplo |
|---|---|---|
| `$eq` | Igual | `{"categoria": {"$eq": "legal"}}` |
| `$ne` | Distinto | `{"categoria": {"$ne": "borrador"}}` |
| `$gte` / `$lte` | Mayor/menor o igual | `{"año": {"$gte": 2023}}` |
| `$in` | Está en la lista | `{"autor": {"$in": ["García", "Martínez"]}}` |
| `$and` / `$or` | Combinaciones lógicas | `{"$and": [{"cat": "legal"}, {"año": {"$gte": 2024}}]}` |

---

## 4. Weaviate: setup local con Docker y cloud

### Setup local con Docker

Crea el archivo `docker-compose.yml`:

```yaml
# docker-compose.yml
version: '3.4'
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.1
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "none"      # usaremos vectores propios de OpenAI
      ENABLE_MODULES: ""
      CLUSTER_HOSTNAME: "node1"
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

```bash
docker-compose up -d
```

### Instalación y conexión Python

```bash
pip install weaviate-client openai
```

```python
import os
import weaviate
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Conexión local
cliente = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

# Conexión a Weaviate Cloud Services (WCS)
# cliente = weaviate.connect_to_wcs(
#     cluster_url=os.environ["WCS_URL"],
#     auth_credentials=weaviate.auth.AuthApiKey(os.environ["WCS_API_KEY"])
# )

print("Conectado a Weaviate:", cliente.is_ready())
```

### Definir esquema de datos (colección)

```python
from weaviate.classes.config import Configure, Property, DataType


def crear_coleccion(cliente) -> None:
    """Crea la colección 'Documento' en Weaviate si no existe."""
    colecciones_existentes = [c.name for c in cliente.collections.list_all().values()]

    if "Documento" in colecciones_existentes:
        print("La colección 'Documento' ya existe.")
        return

    cliente.collections.create(
        name="Documento",
        # none = sin vectorizador automático (nosotros proporcionamos los vectores)
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="titulo",    data_type=DataType.TEXT),
            Property(name="contenido", data_type=DataType.TEXT),
            Property(name="categoria", data_type=DataType.TEXT),
            Property(name="autor",     data_type=DataType.TEXT),
            Property(name="año",       data_type=DataType.INT),
        ]
    )
    print("Colección 'Documento' creada.")


def insertar_documentos_weaviate(cliente, documentos: list[dict]) -> None:
    """Inserta documentos con vectores propios en Weaviate."""
    coleccion = cliente.collections.get("Documento")

    with coleccion.batch.dynamic() as batch:
        for doc in documentos:
            texto = f"{doc['titulo']}\n\n{doc['contenido']}"
            embedding = generar_embedding_weaviate(texto)

            batch.add_object(
                properties={
                    "titulo":    doc["titulo"],
                    "contenido": doc["contenido"],
                    "categoria": doc.get("categoria", ""),
                    "autor":     doc.get("autor", ""),
                    "año":       doc.get("año", 2024),
                },
                vector=embedding
            )

    print(f"{len(documentos)} documentos insertados en Weaviate.")


def generar_embedding_weaviate(texto: str) -> list[float]:
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding


documentos = [
    {
        "titulo": "Contrato de arrendamiento 2024",
        "contenido": "El arrendador cede el uso del inmueble por 12 meses a partir del 1 de enero.",
        "categoria": "legal",
        "autor": "García López",
        "año": 2024
    },
    {
        "titulo": "Manual de onboarding",
        "contenido": "Bienvenido al equipo. Este manual describe los procesos internos.",
        "categoria": "rrhh",
        "autor": "Recursos Humanos",
        "año": 2024
    },
    {
        "titulo": "Política de privacidad RGPD",
        "contenido": "Describimos cómo recopilamos y protegemos tus datos según el RGPD europeo.",
        "categoria": "legal",
        "autor": "Departamento Legal",
        "año": 2023
    },
]

crear_coleccion(cliente)
insertar_documentos_weaviate(cliente, documentos)
```

---

## 5. Weaviate: búsqueda híbrida

La búsqueda híbrida combina búsqueda vectorial (semántica) y BM25 (coincidencia léxica). Es especialmente útil cuando la consulta contiene términos técnicos específicos que deben aparecer literalmente en los resultados:

```python
from weaviate.classes.query import HybridFusion, MetadataQuery


def busqueda_hibrida(
    cliente,
    query: str,
    alpha: float = 0.5,
    top_k: int = 5,
    categoria: str | None = None
) -> list[dict]:
    """
    Búsqueda híbrida en Weaviate: vectorial + BM25.

    Parámetro alpha:
    - alpha=1.0 → solo vectorial (semántico)
    - alpha=0.0 → solo BM25 (léxico/exacto)
    - alpha=0.5 → combinación 50/50 (recomendado como punto de partida)
    """
    coleccion = cliente.collections.get("Documento")

    # Embedding de la consulta para la parte vectorial
    query_embedding = generar_embedding_weaviate(query)

    # Construir filtros de metadata si se especifican
    filtros = None
    if categoria:
        from weaviate.classes.query import Filter
        filtros = Filter.by_property("categoria").equal(categoria)

    resultados = coleccion.query.hybrid(
        query=query,                          # para BM25
        vector=query_embedding,               # para búsqueda vectorial
        alpha=alpha,                          # balance entre los dos modos
        limit=top_k,
        fusion_type=HybridFusion.RELATIVE_SCORE,   # normaliza scores antes de combinar
        filters=filtros,
        return_metadata=MetadataQuery(score=True, explain_score=True),
        return_properties=["titulo", "contenido", "categoria", "autor", "año"]
    )

    return [
        {
            "titulo":    obj.properties["titulo"],
            "contenido": obj.properties["contenido"],
            "categoria": obj.properties["categoria"],
            "autor":     obj.properties["autor"],
            "año":       obj.properties["año"],
            "score":     round(obj.metadata.score, 4) if obj.metadata.score else None
        }
        for obj in resultados.objects
    ]


def comparar_modos_busqueda(cliente, query: str):
    """Muestra la diferencia entre búsqueda vectorial pura, BM25 puro e híbrida."""
    print(f"Query: '{query}'\n")

    print("--- Solo vectorial (alpha=1.0) ---")
    for r in busqueda_hibrida(cliente, query, alpha=1.0, top_k=3):
        print(f"  [{r['score']}] {r['titulo']}")

    print("\n--- Solo BM25 (alpha=0.0) ---")
    for r in busqueda_hibrida(cliente, query, alpha=0.0, top_k=3):
        print(f"  [{r['score']}] {r['titulo']}")

    print("\n--- Híbrida (alpha=0.5) ---")
    for r in busqueda_hibrida(cliente, query, alpha=0.5, top_k=3):
        print(f"  [{r['score']}] {r['titulo']}")


if __name__ == "__main__":
    comparar_modos_busqueda(cliente, "RGPD datos personales")
    print()
    comparar_modos_busqueda(cliente, "¿cómo empiezo en la empresa?")

    cliente.close()
```

---

## 6. Comparativa práctica Pinecone vs Weaviate vs ChromaDB

| Característica | Pinecone | Weaviate | ChromaDB |
|---|---|---|---|
| **Precio** | Freemium (muy limitado gratis); desde ~$70/mes en producción | Gratuito en local; WCS cloud de pago | Completamente gratuito |
| **Escalabilidad** | Excelente (diseñado para escala) | Buena (millones de vectores) | Limitada (ideal para prototipado) |
| **Búsqueda híbrida** | No nativa | Nativa (vectorial + BM25) | No nativa |
| **Filtros de metadata** | Sí (operadores MongoDB-like) | Sí (GraphQL, muy expresivos) | Sí (básicos) |
| **Soporte local/offline** | No (solo SaaS) | Sí (Docker) | Sí (en memoria o disco) |
| **Facilidad de setup** | Muy fácil (API key + SDK) | Media (Docker o WCS) | Muy fácil (`pip install`) |
| **Grafos de conocimiento** | No | Sí (objetos con referencias cruzadas) | No |
| **Ecosistema Python** | SDK oficial maduro | SDK oficial maduro | SDK oficial maduro |
| **Ideal para** | Producción a escala, SaaS, startups | Búsqueda híbrida, grafos, uso local | Prototipado rápido, desarrollo local, CI/CD |

**Recomendación por caso de uso:**

- **Prototipo o MVP**: ChromaDB — cero infraestructura, `pip install` y listo.
- **Producción con escala rápida y sin DevOps**: Pinecone.
- **Búsqueda híbrida o datos relacionados entre sí**: Weaviate.
- **Control total del stack y datos en PostgreSQL**: pgvector (ver tutorial 01).

---

## 7. Extensiones sugeridas

- **Pinecone namespaces**: separa vectores por tenant o idioma dentro de un mismo índice usando espacios de nombres, sin crear índices adicionales.
- **Weaviate generative search**: Weaviate tiene módulos para invocar LLMs directamente desde la query GraphQL, convirtiendo la búsqueda en RAG de una sola llamada.
- **Weaviate multi-vector**: a partir de la versión 1.24, Weaviate soporta múltiples vectores por objeto (por ejemplo, un vector para el título y otro para el cuerpo).
- **Pinecone + metadata filtering en RAG**: combina filtros de metadata (cliente, idioma, fecha) con búsqueda semántica para RAG multi-tenant.
- **Benchmarks**: el proyecto [ANN Benchmarks](https://ann-benchmarks.com/) ofrece comparativas objetivas de rendimiento entre diferentes bases de datos vectoriales.

---

**Siguiente:** [03 — Reranking: mejorar la calidad del RAG](./03-reranking.md)
