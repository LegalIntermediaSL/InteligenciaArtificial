# 04 — Técnicas avanzadas de RAG

> **Bloque:** Bases de datos vectoriales · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Limitaciones del RAG básico](#1-limitaciones-del-rag-básico)
2. [HyDE — Hypothetical Document Embeddings](#2-hyde--hypothetical-document-embeddings)
3. [Chunking inteligente (Parent-Child chunks)](#3-chunking-inteligente-parent-child-chunks)
4. [Self-Query Retriever](#4-self-query-retriever)
5. [RAG multi-documento](#5-rag-multi-documento)
6. [Evaluación automática de RAG con ragas](#6-evaluación-automática-de-rag-con-ragas)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Limitaciones del RAG básico

Un pipeline RAG básico tiene tres problemas estructurales que limitan su calidad en producción:

**Problema 1 — Chunking ingenuo**

Dividir documentos en fragmentos de tamaño fijo (por ejemplo, cada 500 tokens) rompe el contexto semántico. Un párrafo sobre "las consecuencias del impago" puede quedar partido entre dos chunks, y ninguno de los dos contiene la información completa.

**Problema 2 — Queries cortas o ambiguas**

La query del usuario "¿qué dice el contrato?" genera un embedding pobre. No hay suficiente información semántica para recuperar los fragmentos correctos. La similitud coseno entre un embedding de 5 palabras y un embedding de un párrafo complejo es intrínsecamente baja.

**Problema 3 — Documentos irrelevantes en top-k**

Incluso con un buen modelo de embeddings, el top-k siempre devuelve algo. Si ningún documento responde realmente la pregunta, el modelo igual recibe contexto basura y puede alucinar o responder incorrectamente.

Este tutorial presenta técnicas específicas para cada uno de estos problemas.

---

## 2. HyDE — Hypothetical Document Embeddings

**Idea:** en lugar de buscar por el embedding de la query (corta, ambigua), primero le pedimos a Claude que genere una respuesta hipotética (larga, rica en vocabulario del dominio) y buscamos por el embedding de esa respuesta hipotética.

La intuición es que el embedding de "una respuesta ideal" es mucho más parecido al embedding de los documentos relevantes que el embedding de una pregunta corta.

```python
import os
import chromadb
import anthropic
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

chroma_client = chromadb.Client()
coleccion = chroma_client.get_or_create_collection(
    name="documentos_hyde",
    metadata={"hnsw:space": "cosine"}
)


def generar_embedding(texto: str) -> list[float]:
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding


def generar_respuesta_hipotetica(query: str) -> str:
    """
    Usa Claude para generar una respuesta hipotética a la query.
    Esta respuesta será más larga y semánticamente más rica que la query original,
    lo que mejora la calidad del embedding de búsqueda.
    """
    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system="""Eres un experto que genera respuestas hipotéticas detalladas.
Tu tarea es generar lo que podría ser la respuesta ideal a una pregunta,
como si existiera un documento que la respondiera perfectamente.
Escribe un párrafo denso en información, en español, sin decir explícitamente
que es una respuesta hipotética. Solo el contenido.""",
        messages=[
            {"role": "user", "content": f"Genera la respuesta ideal a esta pregunta:\n\n{query}"}
        ]
    )
    return respuesta.content[0].text


def hyde_search(query: str, n_resultados: int = 5) -> list[dict]:
    """
    Búsqueda HyDE:
    1. Genera respuesta hipotética con Claude
    2. Crea embedding de la respuesta hipotética (no de la query)
    3. Busca en la base de datos vectorial con ese embedding
    """
    print(f"Query original: '{query}'")

    # Paso 1: respuesta hipotética
    respuesta_hipotetica = generar_respuesta_hipotetica(query)
    print(f"\nRespuesta hipotética generada:\n{respuesta_hipotetica[:200]}...")

    # Paso 2: embedding de la respuesta hipotética
    embedding_hipotetico = generar_embedding(respuesta_hipotetica)

    # Paso 3: buscar por ese embedding
    resultados = coleccion.query(
        query_embeddings=[embedding_hipotetico],
        n_results=n_resultados,
        include=["documents", "metadatas", "distances"]
    )

    documentos = []
    for i in range(len(resultados["ids"][0])):
        documentos.append({
            "id":        resultados["ids"][0][i],
            "contenido": resultados["documents"][0][i],
            "titulo":    resultados["metadatas"][0][i].get("titulo", ""),
            "distancia": resultados["distances"][0][i]
        })

    return documentos


def comparar_busqueda_normal_vs_hyde(query: str):
    """Compara los resultados de búsqueda normal vs HyDE para la misma query."""
    # Búsqueda normal (embedding de la query directamente)
    query_embedding = generar_embedding(query)
    resultados_normal = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print("=== Búsqueda normal (embedding de la query) ===")
    for i, (doc_id, distancia, metadata) in enumerate(zip(
        resultados_normal["ids"][0],
        resultados_normal["distances"][0],
        resultados_normal["metadatas"][0]
    ), 1):
        print(f"  {i}. [{1-distancia:.3f}] {metadata.get('titulo', doc_id)}")

    print("\n=== Búsqueda HyDE (embedding de respuesta hipotética) ===")
    resultados_hyde = hyde_search(query, n_resultados=5)
    for i, doc in enumerate(resultados_hyde, 1):
        print(f"  {i}. [{1-doc['distancia']:.3f}] {doc['titulo']}")


# Poblar la colección con documentos de ejemplo
documentos_ejemplo = [
    ("d1", "Las penalizaciones por incumplimiento contractual incluyen indemnización por daños, resolución del contrato y posibles acciones judiciales.", {"titulo": "Consecuencias del incumplimiento de contrato"}),
    ("d2", "El arrendador puede iniciar desahucio tras dos mensualidades impagadas según la Ley de Arrendamientos Urbanos.", {"titulo": "Impago de renta: derechos del arrendador"}),
    ("d3", "Los contratos de arrendamiento para uso de vivienda tienen una duración mínima de 5 años si el arrendador es persona física.", {"titulo": "Duración de los contratos de arrendamiento"}),
    ("d4", "Los ingresos trimestrales crecieron un 15% impulsados por el segmento de servicios digitales.", {"titulo": "Resultados financieros Q3"}),
]

embeddings = [generar_embedding(doc[1]) for doc in documentos_ejemplo]
coleccion.add(
    ids=[d[0] for d in documentos_ejemplo],
    embeddings=embeddings,
    documents=[d[1] for d in documentos_ejemplo],
    metadatas=[d[2] for d in documentos_ejemplo]
)

if __name__ == "__main__":
    comparar_busqueda_normal_vs_hyde("¿qué pasa si no pago?")
```

---

## 3. Chunking inteligente (Parent-Child chunks)

**Idea:** indexamos chunks pequeños (alta precisión de recuperación) pero cuando encontramos un chunk relevante, devolvemos su documento padre completo (contexto completo para el LLM).

```python
import os
import uuid
import chromadb
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

chroma_client = chromadb.Client()

# Dos colecciones: una para chunks pequeños (búsqueda), otra para padres (contexto)
coleccion_chunks  = chroma_client.get_or_create_collection("chunks_pequenos")
coleccion_padres  = chroma_client.get_or_create_collection("documentos_padres")


def dividir_en_chunks(texto: str, tamaño_chunk: int = 200, solapamiento: int = 50) -> list[str]:
    """
    Divide un texto en chunks de tamaño fijo con solapamiento.
    El solapamiento evita perder contexto en los límites.
    """
    palabras = texto.split()
    chunks = []

    i = 0
    while i < len(palabras):
        chunk = " ".join(palabras[i:i + tamaño_chunk])
        chunks.append(chunk)
        i += tamaño_chunk - solapamiento   # avanzar con solapamiento

    return chunks


def indexar_con_parent_child(documentos: list[dict]) -> None:
    """
    Indexa documentos usando la estrategia Parent-Child:
    1. Guarda el documento completo como 'padre'
    2. Divide en chunks pequeños como 'hijos'
    3. Indexa solo los hijos (para búsqueda precisa)
    4. Cada hijo recuerda el ID de su padre
    """
    for doc in documentos:
        # Guardar el documento padre completo
        padre_id = doc["id"]
        coleccion_padres.add(
            ids=[padre_id],
            documents=[doc["contenido"]],
            metadatas=[{"titulo": doc["titulo"], "categoria": doc.get("categoria", "")}]
        )

        # Dividir en chunks pequeños
        chunks = dividir_en_chunks(doc["contenido"], tamaño_chunk=100, solapamiento=20)

        chunk_ids       = []
        chunk_textos    = []
        chunk_embeddings = []
        chunk_metadatas = []

        for j, chunk_texto in enumerate(chunks):
            chunk_id = f"{padre_id}-chunk-{j}"

            respuesta = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk_texto
            )
            embedding = respuesta.data[0].embedding

            chunk_ids.append(chunk_id)
            chunk_textos.append(chunk_texto)
            chunk_embeddings.append(embedding)
            chunk_metadatas.append({
                "padre_id": padre_id,          # referencia al documento completo
                "titulo":   doc["titulo"],
                "indice_chunk": j
            })

        coleccion_chunks.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=chunk_textos,
            metadatas=chunk_metadatas
        )

    print(f"{len(documentos)} documentos indexados con estrategia Parent-Child.")


def buscar_parent_child(query: str, top_k: int = 3) -> list[dict]:
    """
    Búsqueda Parent-Child:
    1. Busca chunks pequeños (alta precisión)
    2. Recupera el documento padre completo de cada chunk encontrado
    3. Deduplica padres (varios chunks pueden apuntar al mismo padre)
    """
    # Embedding de la query
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = respuesta.data[0].embedding

    # Buscar en chunks pequeños
    resultados_chunks = coleccion_chunks.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 3,   # recuperar más candidatos para tener variedad
        include=["metadatas", "distances"]
    )

    # Recopilar IDs únicos de documentos padre
    padres_ids_vistos = set()
    padres_ordenados = []

    for metadata, distancia in zip(
        resultados_chunks["metadatas"][0],
        resultados_chunks["distances"][0]
    ):
        padre_id = metadata["padre_id"]
        if padre_id not in padres_ids_vistos:
            padres_ids_vistos.add(padre_id)
            padres_ordenados.append({
                "padre_id": padre_id,
                "distancia_chunk": distancia,
                "titulo": metadata["titulo"]
            })

    # Limitar al top_k de padres únicos
    padres_ordenados = padres_ordenados[:top_k]

    # Recuperar el contenido completo de cada padre
    resultados_finales = []
    for entrada in padres_ordenados:
        padre_resultado = coleccion_padres.get(
            ids=[entrada["padre_id"]],
            include=["documents", "metadatas"]
        )

        if padre_resultado["documents"]:
            resultados_finales.append({
                "id":       entrada["padre_id"],
                "titulo":   entrada["titulo"],
                "contenido": padre_resultado["documents"][0],   # documento COMPLETO
                "similitud": round(1 - entrada["distancia_chunk"], 4)
            })

    return resultados_finales


# Documentos de prueba con contenido largo
docs_largos = [
    {
        "id": "contrato-001",
        "titulo": "Contrato de arrendamiento de vivienda",
        "contenido": """
El presente contrato de arrendamiento se celebra entre el arrendador, Don Carlos García López,
con DNI 12345678A, y el arrendatario, Doña María Pérez Sánchez, con DNI 87654321B.

El objeto del contrato es la cesión del uso del inmueble sito en Calle Mayor número 10, 
piso 3º B, de Madrid, con una superficie útil de 85 metros cuadrados.

La duración del contrato será de tres años, prorrogables por períodos anuales hasta un 
máximo de cinco años si el arrendador es persona física. La renta mensual pactada es de 
novecientos euros (900 €), pagaderos dentro de los cinco primeros días de cada mes.

En caso de impago de la renta, el arrendador queda facultado para iniciar el procedimiento 
de desahucio previsto en la Ley de Enjuiciamiento Civil tras dos mensualidades consecutivas 
sin abonar. Adicionalmente, el arrendatario deberá abonar los intereses de demora al tipo 
legal del dinero más dos puntos porcentuales sobre las cantidades adeudadas.

El arrendatario no podrá subarrendar el inmueble ni realizar obras sin el consentimiento 
expreso y por escrito del arrendador. Las obras de mejora correrán a cargo del arrendador, 
mientras que las reparaciones menores derivadas del uso ordinario serán responsabilidad del 
arrendatario hasta un importe máximo de ciento cincuenta euros (150 €) por incidencia.
        """.strip(),
        "categoria": "legal"
    },
]

indexar_con_parent_child(docs_largos)

if __name__ == "__main__":
    query = "¿Qué pasa si no pago la renta?"
    resultados = buscar_parent_child(query, top_k=2)

    for r in resultados:
        print(f"[{r['similitud']}] {r['titulo']}")
        print(f"Contenido completo ({len(r['contenido'])} caracteres):")
        print(r["contenido"][:300] + "...")
        print()
```

---

## 4. Self-Query Retriever

**Idea:** en lugar de que el usuario escriba filtros manualmente, Claude lee la query en lenguaje natural, extrae los filtros de metadata implícitos y construye una consulta estructurada.

```python
import os
import json
import chromadb
import anthropic
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

chroma_client = chromadb.Client()
coleccion = chroma_client.get_or_create_collection("documentos_self_query")


def extraer_filtros_con_claude(query: str) -> dict:
    """
    Usa Claude para extraer filtros de metadata de una query en lenguaje natural.
    Devuelve un dict con los filtros encontrados y la query reformulada.
    """
    schema_metadata = """
    Los documentos tienen los siguientes campos de metadata:
    - autor: string (nombre del autor del documento)
    - año: integer (año de publicación, entre 2020 y 2026)
    - categoria: string (valores posibles: "legal", "finanzas", "rrhh", "tecnologia")
    - departamento: string (nombre del departamento)
    """

    prompt = f"""Analiza esta query y extrae cualquier filtro de metadata implícito.

Schema de metadata disponible:
{schema_metadata}

Query del usuario: "{query}"

Responde SOLO con un JSON válido con esta estructura:
{{
  "query_limpia": "la query sin los filtros extraídos",
  "filtros": {{
    "autor": "valor o null",
    "año": número_o_null,
    "año_minimo": número_o_null,
    "año_maximo": número_o_null,
    "categoria": "valor o null",
    "departamento": "valor o null"
  }}
}}

Ejemplos:
- "documentos de García sobre contratos del 2024" → filtros: autor="García", año=2024, query_limpia="contratos"
- "informes financieros desde 2023" → filtros: categoria="finanzas", año_minimo=2023
- "¿qué dice la política de privacidad?" → filtros: todos null, query_limpia="política de privacidad"

Devuelve solo el JSON, sin explicaciones."""

    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    texto = respuesta.content[0].text.strip()

    # Limpiar posibles bloques de código markdown
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]

    return json.loads(texto)


def construir_where_chromadb(filtros: dict) -> dict | None:
    """Convierte el dict de filtros en sintaxis where de ChromaDB."""
    condiciones = []

    if filtros.get("autor"):
        condiciones.append({"autor": {"$eq": filtros["autor"]}})

    if filtros.get("año"):
        condiciones.append({"año": {"$eq": filtros["año"]}})

    if filtros.get("año_minimo"):
        condiciones.append({"año": {"$gte": filtros["año_minimo"]}})

    if filtros.get("año_maximo"):
        condiciones.append({"año": {"$lte": filtros["año_maximo"]}})

    if filtros.get("categoria"):
        condiciones.append({"categoria": {"$eq": filtros["categoria"]}})

    if filtros.get("departamento"):
        condiciones.append({"departamento": {"$eq": filtros["departamento"]}})

    if not condiciones:
        return None
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}


def self_query_search(query_original: str, top_k: int = 5) -> list[dict]:
    """
    Búsqueda Self-Query:
    1. Claude extrae filtros de la query en lenguaje natural
    2. Se construye la consulta estructurada para ChromaDB
    3. Búsqueda vectorial + filtros de metadata en una sola llamada
    """
    print(f"Query original: '{query_original}'")

    # Paso 1: extraer filtros con Claude
    extraido = extraer_filtros_con_claude(query_original)
    query_limpia = extraido["query_limpia"]
    filtros = extraido["filtros"]

    print(f"Query limpia:   '{query_limpia}'")
    print(f"Filtros extraídos: {json.dumps(filtros, ensure_ascii=False, indent=2)}")

    # Paso 2: construir filtro where para ChromaDB
    where = construir_where_chromadb(filtros)
    print(f"Filtro where: {where}\n")

    # Paso 3: embedding de la query limpia
    respuesta_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_limpia
    )
    query_embedding = respuesta_embedding.data[0].embedding

    # Paso 4: búsqueda con filtros
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    if where:
        kwargs["where"] = where

    resultados = coleccion.query(**kwargs)

    return [
        {
            "titulo":     resultados["metadatas"][0][i].get("titulo", ""),
            "contenido":  resultados["documents"][0][i],
            "autor":      resultados["metadatas"][0][i].get("autor", ""),
            "año":        resultados["metadatas"][0][i].get("año", ""),
            "categoria":  resultados["metadatas"][0][i].get("categoria", ""),
            "similitud":  round(1 - resultados["distances"][0][i], 4)
        }
        for i in range(len(resultados["ids"][0]))
    ]


# Datos de prueba
docs = [
    ("sq1", "Las cláusulas de arrendamiento requieren firma notarial en contratos superiores a 6 años.", {"titulo": "Contratos de larga duración", "autor": "García", "año": 2024, "categoria": "legal"}),
    ("sq2", "El balance financiero de 2024 muestra un crecimiento del 18% en servicios.", {"titulo": "Balance anual 2024", "autor": "Martínez", "año": 2024, "categoria": "finanzas"}),
    ("sq3", "Proceso de incorporación de nuevos empleados al equipo de tecnología.", {"titulo": "Onboarding tecnología", "autor": "López", "año": 2023, "categoria": "rrhh"}),
    ("sq4", "Informe de riesgos legales para el ejercicio 2023.", {"titulo": "Riesgos legales 2023", "autor": "García", "año": 2023, "categoria": "legal"}),
]

embeddings_sq = [openai_client.embeddings.create(model="text-embedding-3-small", input=d[1]).data[0].embedding for d in docs]
coleccion.add(
    ids=[d[0] for d in docs],
    embeddings=embeddings_sq,
    documents=[d[1] for d in docs],
    metadatas=[d[2] for d in docs]
)

if __name__ == "__main__":
    queries_prueba = [
        "documentos legales de García del 2024",
        "informes financieros desde 2023",
        "¿qué dice el manual de onboarding?"
    ]

    for q in queries_prueba:
        print("\n" + "=" * 60)
        resultados = self_query_search(q, top_k=3)
        for r in resultados:
            print(f"  [{r['similitud']}] {r['titulo']} ({r['autor']}, {r['año']})")
```

---

## 5. RAG multi-documento

Cuando la respuesta requiere combinar información de varios documentos, un solo prompt con todos los contextos puede sobrepasar la ventana de contexto o diluir la atención del modelo. El patrón **map-reduce** procesa cada documento por separado y luego combina:

```python
import os
import anthropic

claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def map_documento(query: str, documento: dict) -> str:
    """
    Fase MAP: extrae información relevante de un único documento.
    Si el documento no es relevante, devuelve cadena vacía.
    """
    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system="""Eres un extractor de información preciso.
Tu tarea: dado un documento y una pregunta, extrae SOLO la información del documento
que sea directamente relevante para responder la pregunta.
Si el documento no contiene información relevante, responde exactamente: [NO RELEVANTE]
No añadas explicaciones ni contexto adicional.""",
        messages=[
            {
                "role": "user",
                "content": f"Pregunta: {query}\n\nDocumento '{documento['titulo']}':\n{documento['contenido']}\n\nInformación relevante:"
            }
        ]
    )
    texto = respuesta.content[0].text.strip()
    return "" if texto == "[NO RELEVANTE]" else texto


def reduce_respuestas(query: str, extracciones: list[dict]) -> str:
    """
    Fase REDUCE: combina todas las extracciones en una respuesta coherente.
    Solo recibe las extracciones no vacías.
    """
    # Filtrar documentos sin información relevante
    extracciones_relevantes = [e for e in extracciones if e["extraccion"]]

    if not extracciones_relevantes:
        return "No encontré información suficiente en los documentos disponibles para responder esta pregunta."

    contexto_combinado = "\n\n".join([
        f"--- Fuente: {e['titulo']} ---\n{e['extraccion']}"
        for e in extracciones_relevantes
    ])

    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system="""Eres un asistente experto que sintetiza información de múltiples fuentes.
Combina la información proporcionada en una respuesta coherente y bien estructurada.
Cita las fuentes relevantes usando el nombre del documento entre corchetes.
Responde en español.""",
        messages=[
            {
                "role": "user",
                "content": f"Pregunta: {query}\n\nInformación extraída de los documentos:\n\n{contexto_combinado}\n\nResponde la pregunta integrando toda la información relevante:"
            }
        ]
    )
    return respuesta.content[0].text


def rag_multi_documento(query: str, documentos: list[dict]) -> str:
    """
    Pipeline RAG multi-documento con map-reduce.

    Flujo:
    1. MAP: para cada documento, extrae solo la información relevante (en paralelo conceptual)
    2. REDUCE: combina las extracciones en una respuesta final coherente
    """
    print(f"Procesando {len(documentos)} documentos con map-reduce...")

    # Fase MAP: procesar cada documento independientemente
    extracciones = []
    for i, doc in enumerate(documentos, 1):
        print(f"  [{i}/{len(documentos)}] Extrayendo de: {doc['titulo']}")
        extraccion = map_documento(query, doc)
        extracciones.append({
            "titulo":    doc["titulo"],
            "extraccion": extraccion,
            "relevante": bool(extraccion)
        })

    n_relevantes = sum(1 for e in extracciones if e["relevante"])
    print(f"\n{n_relevantes}/{len(documentos)} documentos con información relevante.")

    # Fase REDUCE: combinar en respuesta final
    print("Generando respuesta final...")
    return reduce_respuestas(query, extracciones)


# Documentos de prueba
documentos_empresa = [
    {
        "titulo": "Política salarial 2024",
        "contenido": "Los empleados con más de 2 años de antigüedad tienen derecho a una revisión salarial anual del IPC + 1%. Los bonus de desempeño se calculan trimestralmente."
    },
    {
        "titulo": "Plan de beneficios sociales",
        "contenido": "La empresa ofrece seguro médico privado, ticket restaurante de 11€/día, y 23 días de vacaciones al año más los festivos locales."
    },
    {
        "titulo": "Procedimiento de solicitud de vacaciones",
        "contenido": "Las vacaciones deben solicitarse con 15 días de antelación mínima a través del portal de RRHH. Se aprueban por el responsable directo."
    },
    {
        "titulo": "Contrato marco de proveedores",
        "contenido": "Los proveedores deben cumplir los plazos de entrega acordados. El incumplimiento reiterado puede dar lugar a penalizaciones del 2% del valor del contrato."
    },
    {
        "titulo": "Informe de resultados Q2",
        "contenido": "El segundo trimestre cerró con ingresos de 1.8M€, un margen bruto del 42% y una reducción del 8% en costes operativos respecto al Q1."
    },
]

if __name__ == "__main__":
    query = "¿Qué beneficios tienen los empleados y cómo piden vacaciones?"
    print(f"Query: {query}\n{'='*60}\n")
    respuesta = rag_multi_documento(query, documentos_empresa)
    print(f"\nRespuesta final:\n{respuesta}")
```

---

## 6. Evaluación automática de RAG con ragas

`ragas` es una biblioteca diseñada para evaluar pipelines RAG automáticamente usando LLMs como juez:

```bash
pip install ragas datasets openai
```

```python
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # ¿La respuesta está respaldada por el contexto?
    answer_relevancy,      # ¿La respuesta es relevante para la pregunta?
    context_recall,        # ¿El contexto recuperado contiene la información necesaria?
    context_precision,     # ¿Cuánto del contexto recuperado es realmente útil?
)
import anthropic
from openai import OpenAI

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def ejecutar_pipeline_rag(query: str, documentos: list[dict]) -> tuple[str, list[str]]:
    """
    Ejecuta el pipeline RAG y devuelve (respuesta, lista_de_contextos).
    En producción, aquí iría tu pipeline real (ChromaDB + reranking + Claude).
    """
    contextos = [doc["contenido"] for doc in documentos[:3]]
    contexto_unido = "\n\n".join([f"[{doc['titulo']}]\n{doc['contenido']}" for doc in documentos[:3]])

    respuesta = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"Contexto:\n{contexto_unido}\n\nPregunta: {query}\n\nResponde basándote solo en el contexto:"
            }
        ]
    )

    return respuesta.content[0].text, contextos


def construir_dataset_evaluacion(
    pares_qa: list[dict],
    documentos_disponibles: list[dict]
) -> Dataset:
    """
    Construye el dataset de evaluación ejecutando el pipeline RAG
    sobre cada par pregunta-respuesta de referencia.

    pares_qa: lista de dicts con keys 'pregunta' y 'respuesta_referencia'
    """
    preguntas         = []
    respuestas_ia     = []
    contextos_ia      = []
    respuestas_ref    = []

    for par in pares_qa:
        respuesta, contextos = ejecutar_pipeline_rag(par["pregunta"], documentos_disponibles)

        preguntas.append(par["pregunta"])
        respuestas_ia.append(respuesta)
        contextos_ia.append(contextos)
        respuestas_ref.append(par["respuesta_referencia"])

    return Dataset.from_dict({
        "question":   preguntas,
        "answer":     respuestas_ia,
        "contexts":   contextos_ia,
        "ground_truth": respuestas_ref
    })


# Dataset de prueba con preguntas y respuestas de referencia (creadas por humanos)
pares_qa = [
    {
        "pregunta": "¿Cuánto tiempo dura el contrato de arrendamiento?",
        "respuesta_referencia": "El contrato de arrendamiento tiene una duración de 3 años, prorrogables por períodos anuales hasta un máximo de 5 años."
    },
    {
        "pregunta": "¿Qué ocurre si el arrendatario no paga la renta?",
        "respuesta_referencia": "El arrendador puede iniciar un procedimiento de desahucio tras dos mensualidades impagadas, y el arrendatario deberá abonar intereses de demora."
    },
    {
        "pregunta": "¿Puede el arrendatario hacer obras en el inmueble?",
        "respuesta_referencia": "No puede hacer obras sin el consentimiento expreso y por escrito del arrendador."
    },
]

documentos_base = [
    {
        "titulo": "Contrato de arrendamiento",
        "contenido": """El contrato de arrendamiento tiene una duración de 3 años, prorrogables anualmente hasta 5 años.
La renta mensual es de 900€. El impago durante 2 meses consecutivos faculta al arrendador para el desahucio
e intereses de demora al tipo legal más 2 puntos. El arrendatario no puede subarrendar ni hacer obras
sin consentimiento escrito del arrendador."""
    },
    {
        "titulo": "Información general del inmueble",
        "contenido": "El inmueble está ubicado en Calle Mayor 10, 3ºB, Madrid, con 85m² útiles."
    },
    {
        "titulo": "Condiciones generales",
        "contenido": "Las reparaciones menores hasta 150€ por incidencia son responsabilidad del arrendatario."
    },
]


def evaluar_rag_completo():
    """Ejecuta la evaluación completa del pipeline RAG."""
    print("Construyendo dataset de evaluación...")
    dataset = construir_dataset_evaluacion(pares_qa, documentos_base)

    print(f"\nDataset construido: {len(dataset)} ejemplos")
    print("Ejecutando evaluación con ragas...\n")

    # ragas usa OpenAI por defecto para evaluar
    # Se puede configurar para usar otros modelos
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    resultado = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,        # ¿Respuesta respaldada por contexto? (0-1)
            answer_relevancy,    # ¿Respuesta relevante para la pregunta? (0-1)
            context_recall,      # ¿Contexto cubre lo necesario para responder? (0-1)
            context_precision,   # ¿Qué proporción del contexto es útil? (0-1)
        ]
    )

    print("=" * 50)
    print("RESULTADOS DE EVALUACIÓN RAG")
    print("=" * 50)
    print(f"Faithfulness (fidelidad al contexto): {resultado['faithfulness']:.3f}")
    print(f"Answer Relevancy (relevancia respuesta): {resultado['answer_relevancy']:.3f}")
    print(f"Context Recall (cobertura del contexto): {resultado['context_recall']:.3f}")
    print(f"Context Precision (precisión del contexto): {resultado['context_precision']:.3f}")
    print()

    # Interpretar resultados
    promedio = (
        resultado['faithfulness'] +
        resultado['answer_relevancy'] +
        resultado['context_recall'] +
        resultado['context_precision']
    ) / 4

    print(f"Score promedio: {promedio:.3f}")

    if promedio >= 0.8:
        print("Evaluación: EXCELENTE — el pipeline RAG funciona muy bien.")
    elif promedio >= 0.6:
        print("Evaluación: ACEPTABLE — hay margen de mejora.")
    else:
        print("Evaluación: MEJORABLE — considera añadir reranking o mejorar el chunking.")

    return resultado


if __name__ == "__main__":
    evaluar_rag_completo()
```

Interpretación de las métricas:

| Métrica | Qué mide | Valor bajo indica... |
|---|---|---|
| **Faithfulness** | ¿La respuesta está justificada por el contexto? | El modelo alucina o añade información inventada |
| **Answer Relevancy** | ¿La respuesta responde la pregunta? | Respuestas divagantes o fuera de tema |
| **Context Recall** | ¿El contexto recuperado cubre la respuesta de referencia? | El sistema de recuperación es deficiente |
| **Context Precision** | ¿Qué porcentaje del contexto es realmente útil? | Se recuperan demasiados documentos irrelevantes |

---

## 7. Extensiones sugeridas

- **Adaptive RAG**: mide la confianza de la respuesta y, si es baja, reformula la query automáticamente y vuelve a buscar antes de responder al usuario.
- **RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**: construye un árbol jerárquico de resúmenes sobre los documentos para responder preguntas que requieren comprensión global del corpus.
- **Corrective RAG (CRAG)**: añade un paso de verificación donde el LLM evalúa si los documentos recuperados son realmente relevantes antes de generar la respuesta. Si no lo son, reformula la búsqueda.
- **Graph RAG**: construye un grafo de conocimiento sobre los documentos (entidades y relaciones) y combina búsqueda en grafos con búsqueda vectorial. Microsoft publicó un paper y código abierto de referencia.
- **Streaming con RAG**: en lugar de esperar a que Claude genere toda la respuesta, usa `stream=True` en la API de Anthropic para mostrar la respuesta token a token mientras se genera, mejorando la experiencia del usuario.
- **Cache semántico**: guarda las respuestas a preguntas previas y, cuando llega una pregunta similar (similitud > 0.95), devuelve la respuesta cacheada sin llamar al LLM. Reduce coste y latencia drásticamente para preguntas repetidas.

---

**Fin del bloque**
