# 01 — Búsqueda vectorial en PostgreSQL con pgvector

> **Bloque:** Bases de datos vectoriales · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Por qué pgvector](#1-por-qué-pgvector)
2. [Instalación](#2-instalación)
3. [Crear tabla con columna vector](#3-crear-tabla-con-columna-vector)
4. [Insertar documentos con embeddings](#4-insertar-documentos-con-embeddings)
5. [Búsqueda por similitud coseno](#5-búsqueda-por-similitud-coseno)
6. [Índice HNSW para escalar](#6-índice-hnsw-para-escalar)
7. [Integrar pgvector en un pipeline RAG](#7-integrar-pgvector-en-un-pipeline-rag)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Por qué pgvector

La mayoría de aplicaciones ya tienen una base de datos PostgreSQL. Migrar a una base de datos vectorial especializada implica mantener dos sistemas sincronizados, duplicar datos y gestionar una infraestructura adicional.

**pgvector** resuelve esto añadiendo un tipo de dato `vector` directamente a PostgreSQL. Con él puedes:

- Guardar embeddings en la misma tabla que el resto de los datos del documento (título, autor, fecha, categoría).
- Hacer búsquedas semánticas **combinadas con filtros SQL** exactos en una sola consulta.
- Aprovechar toda la maquinaria de PostgreSQL: transacciones, backups, roles, índices existentes.
- Escalar con índices **HNSW** o **IVFFlat** cuando el volumen crece.

Es la elección natural para proyectos que no superan los 10-50 millones de vectores y que ya viven en el ecosistema PostgreSQL.

---

## 2. Instalación

### Opción A — Docker (recomendada para desarrollo)

```bash
docker run \
  --name pgvector-dev \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

### Opción B — PostgreSQL existente

Si ya tienes PostgreSQL 15 o 16, instala la extensión:

```bash
# En sistemas Debian/Ubuntu
sudo apt install postgresql-16-pgvector

# Con Homebrew en macOS
brew install pgvector
```

### Activar la extensión

Conéctate a tu base de datos y ejecuta:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Verifica que funcionó:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Dependencias Python

```bash
pip install psycopg2-binary pgvector openai
```

---

## 3. Crear tabla con columna vector

### SQL puro

```sql
-- Tabla para almacenar documentos con su embedding
CREATE TABLE documentos (
    id          SERIAL PRIMARY KEY,
    titulo      TEXT NOT NULL,
    contenido   TEXT NOT NULL,
    categoria   TEXT,
    autor       TEXT,
    fecha       DATE DEFAULT CURRENT_DATE,
    embedding   vector(1536)   -- dimensiones de text-embedding-3-small
);
```

El número `1536` corresponde a las dimensiones del modelo `text-embedding-3-small` de OpenAI. Si usas otro modelo, ajusta ese valor (por ejemplo, `3072` para `text-embedding-3-large`).

### Crear la tabla desde Python

```python
import psycopg2
from psycopg2.extras import execute_values

# Conexión a la base de datos
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="vectordb",
    user="postgres",
    password="pass"
)
conn.autocommit = True
cur = conn.cursor()

# Activar la extensión (idempotente)
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Crear la tabla
cur.execute("""
    CREATE TABLE IF NOT EXISTS documentos (
        id        SERIAL PRIMARY KEY,
        titulo    TEXT NOT NULL,
        contenido TEXT NOT NULL,
        categoria TEXT,
        autor     TEXT,
        fecha     DATE DEFAULT CURRENT_DATE,
        embedding vector(1536)
    );
""")

print("Tabla 'documentos' creada correctamente.")
cur.close()
conn.close()
```

---

## 4. Insertar documentos con embeddings

Primero generamos los embeddings con OpenAI y luego los insertamos en PostgreSQL:

```python
import os
import psycopg2
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Documentos de ejemplo
documentos = [
    {
        "titulo": "Contrato de arrendamiento 2024",
        "contenido": "El arrendador cede el uso del inmueble ubicado en Calle Mayor 10 por un periodo de 12 meses.",
        "categoria": "legal",
        "autor": "García López"
    },
    {
        "titulo": "Política de privacidad",
        "contenido": "Esta política describe cómo recopilamos, usamos y protegemos sus datos personales.",
        "categoria": "legal",
        "autor": "Departamento Legal"
    },
    {
        "titulo": "Informe financiero Q3",
        "contenido": "Los ingresos del tercer trimestre alcanzaron 2.3 millones de euros, un 15% más que el año anterior.",
        "categoria": "finanzas",
        "autor": "Ana Martínez"
    },
    {
        "titulo": "Manual de onboarding",
        "contenido": "Bienvenido al equipo. Este manual describe los procesos internos y herramientas que usarás en tu día a día.",
        "categoria": "rrhh",
        "autor": "Recursos Humanos"
    },
]


def generar_embedding(texto: str) -> list[float]:
    """Genera un embedding con OpenAI text-embedding-3-small."""
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding


def insertar_documentos(documentos: list[dict]) -> None:
    """Genera embeddings e inserta los documentos en PostgreSQL."""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="vectordb",
        user="postgres",
        password="pass"
    )
    cur = conn.cursor()

    for doc in documentos:
        # El texto que se embebe es título + contenido para mayor contexto
        texto_a_embeber = f"{doc['titulo']}\n\n{doc['contenido']}"
        embedding = generar_embedding(texto_a_embeber)

        cur.execute(
            """
            INSERT INTO documentos (titulo, contenido, categoria, autor, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                doc["titulo"],
                doc["contenido"],
                doc.get("categoria"),
                doc.get("autor"),
                embedding  # psycopg2 + pgvector acepta listas de float directamente
            )
        )
        print(f"Insertado: {doc['titulo']}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n{len(documentos)} documentos insertados correctamente.")


if __name__ == "__main__":
    insertar_documentos(documentos)
```

---

## 5. Búsqueda por similitud coseno

### SQL directo

El operador `<=>` calcula la **distancia coseno** entre dos vectores (0 = idénticos, 2 = opuestos). Ordenar por este valor de menor a mayor devuelve los más similares primero:

```sql
-- Buscar los 5 documentos más similares a un embedding dado
-- (sustituye [0.1, 0.2, ...] por un vector real de 1536 dimensiones)
SELECT
    id,
    titulo,
    categoria,
    autor,
    1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similitud
FROM documentos
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 5;
```

Otros operadores disponibles:

| Operador | Métrica | Cuándo usarlo |
|---|---|---|
| `<=>` | Distancia coseno | Texto e información semántica (lo más común) |
| `<->` | Distancia euclidiana | Datos espaciales o numéricos |
| `<#>` | Producto escalar negativo | Cuando los vectores están normalizados |

### Búsqueda desde Python

```python
import os
import psycopg2
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def buscar_similares(
    query: str,
    top_k: int = 5,
    categoria: str | None = None
) -> list[dict]:
    """
    Busca los documentos más similares a la consulta.
    Opcionalmente filtra por categoría (combinación vectorial + SQL exacto).
    """
    # 1. Generar embedding de la consulta
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = respuesta.data[0].embedding

    # 2. Construir la consulta SQL
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="vectordb",
        user="postgres",
        password="pass"
    )
    cur = conn.cursor()

    if categoria:
        # Filtro exacto por categoría + búsqueda semántica
        cur.execute(
            """
            SELECT
                id,
                titulo,
                contenido,
                categoria,
                autor,
                1 - (embedding <=> %s::vector) AS similitud
            FROM documentos
            WHERE categoria = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, categoria, query_embedding, top_k)
        )
    else:
        cur.execute(
            """
            SELECT
                id,
                titulo,
                contenido,
                categoria,
                autor,
                1 - (embedding <=> %s::vector) AS similitud
            FROM documentos
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k)
        )

    filas = cur.fetchall()
    cur.close()
    conn.close()

    # 3. Formatear resultados
    resultados = []
    for fila in filas:
        resultados.append({
            "id": fila[0],
            "titulo": fila[1],
            "contenido": fila[2],
            "categoria": fila[3],
            "autor": fila[4],
            "similitud": round(float(fila[5]), 4)
        })

    return resultados


if __name__ == "__main__":
    query = "¿Qué dice el contrato sobre el periodo de arrendamiento?"
    resultados = buscar_similares(query, top_k=3)

    print(f"Resultados para: '{query}'\n")
    for r in resultados:
        print(f"[{r['similitud']:.4f}] {r['titulo']} ({r['categoria']})")
        print(f"  {r['contenido'][:100]}...")
        print()
```

---

## 6. Índice HNSW para escalar

Sin índice, pgvector realiza una búsqueda exacta que recorre toda la tabla (O(n)). Para más de 100.000 vectores esto se vuelve lento. El índice **HNSW** (Hierarchical Navigable Small World) permite búsqueda aproximada con latencia constante:

```sql
-- Crear índice HNSW para distancia coseno
CREATE INDEX ON documentos
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

Parámetros clave:

| Parámetro | Valor por defecto | Efecto |
|---|---|---|
| `m` | 16 | Número de conexiones por nodo. Más alto = más preciso pero más memoria |
| `ef_construction` | 64 | Precisión durante la construcción. Más alto = mejor índice pero más lento de construir |

Ajustar la precisión en tiempo de consulta:

```sql
-- ef_search controla cuántos candidatos explorar en cada búsqueda
SET hnsw.ef_search = 100;   -- más lento pero más preciso (por defecto: 40)

-- La misma consulta se beneficia automáticamente del índice
SELECT titulo, 1 - (embedding <=> %s::vector) AS similitud
FROM documentos
ORDER BY embedding <=> %s::vector
LIMIT 5;
```

Desde Python:

```python
def configurar_precision_busqueda(cur, ef_search: int = 100):
    """Ajusta la precisión del índice HNSW para la sesión actual."""
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
```

Alternativa: **IVFFlat** (más rápido de construir, menor precisión que HNSW):

```sql
-- Requiere datos existentes para calcular los centroides
CREATE INDEX ON documentos
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);   -- sqrt(n_filas) es un buen punto de partida
```

---

## 7. Integrar pgvector en un pipeline RAG

Pipeline completo: PostgreSQL + pgvector para recuperar contexto, Claude para generar la respuesta.

```python
import os
import psycopg2
import anthropic
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vectordb",
    "user": "postgres",
    "password": "pass"
}


def recuperar_contexto(query: str, top_k: int = 5) -> list[dict]:
    """Recupera los documentos más relevantes para la consulta."""
    # Embedding de la consulta
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = respuesta.data[0].embedding

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            titulo,
            contenido,
            categoria,
            autor,
            1 - (embedding <=> %s::vector) AS similitud
        FROM documentos
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k)
    )

    filas = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "titulo": f[0],
            "contenido": f[1],
            "categoria": f[2],
            "autor": f[3],
            "similitud": float(f[4])
        }
        for f in filas
    ]


def construir_contexto_para_prompt(documentos: list[dict]) -> str:
    """Formatea los documentos recuperados como contexto para el prompt."""
    secciones = []
    for i, doc in enumerate(documentos, 1):
        secciones.append(
            f"[Documento {i}] {doc['titulo']} (autor: {doc['autor']}, "
            f"categoría: {doc['categoria']}, similitud: {doc['similitud']:.2f})\n"
            f"{doc['contenido']}"
        )
    return "\n\n---\n\n".join(secciones)


def chatbot_rag(pregunta: str) -> str:
    """
    Pipeline RAG completo:
    1. Recupera documentos relevantes desde pgvector
    2. Construye el prompt con contexto
    3. Genera la respuesta con Claude
    """
    print(f"Buscando documentos relevantes para: '{pregunta}'")
    documentos = recuperar_contexto(pregunta, top_k=5)

    if not documentos:
        return "No encontré documentos relevantes en la base de datos."

    print(f"Recuperados {len(documentos)} documentos (similitud máxima: {documentos[0]['similitud']:.2f})")

    contexto = construir_contexto_para_prompt(documentos)

    system_prompt = """Eres un asistente experto que responde preguntas basándose 
exclusivamente en los documentos proporcionados. 

Reglas:
- Cita siempre el documento fuente entre corchetes, por ejemplo: [Documento 1]
- Si la información no está en los documentos, dilo explícitamente
- Responde en español de forma clara y concisa"""

    prompt = f"""Documentos disponibles:

{contexto}

---

Pregunta del usuario: {pregunta}

Responde basándote únicamente en los documentos anteriores."""

    respuesta = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return respuesta.content[0].text


def main():
    preguntas = [
        "¿Cuánto tiempo dura el contrato de arrendamiento?",
        "¿Cómo protege la empresa los datos personales?",
        "¿Cuáles fueron los ingresos del tercer trimestre?",
    ]

    for pregunta in preguntas:
        print(f"\n{'='*60}")
        print(f"Pregunta: {pregunta}")
        print("-" * 60)
        respuesta = chatbot_rag(pregunta)
        print(f"Respuesta:\n{respuesta}")
        print()


if __name__ == "__main__":
    main()
```

---

## 8. Extensiones sugeridas

- **Búsqueda híbrida con pgvector**: combina la distancia coseno con búsqueda de texto completo (`tsvector` + `tsquery`) usando RRF (Reciprocal Rank Fusion) para mejorar la precisión.
- **Multitenancy**: añade una columna `tenant_id` y filtra por ella en cada consulta para aislar datos entre clientes.
- **Actualizaciones incrementales**: en lugar de re-embeber todos los documentos cuando cambia el modelo, mantén una columna `modelo_embedding` para gestionar migraciones gradual.
- **pgvector + LangChain / LlamaIndex**: ambas bibliotecas tienen integración nativa con pgvector (`PGVector` en LangChain, `PGVectorStore` en LlamaIndex).
- **Monitorización**: usa `EXPLAIN ANALYZE` para verificar que las consultas usan el índice HNSW y no hacen `Seq Scan`.

---

**Siguiente:** [02 — Bases de datos vectoriales en cloud: Pinecone y Weaviate](./02-pinecone-weaviate.md)
