# Extracción de Grafos de Conocimiento con Claude

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/graphrag/03-extraccion-grafos-con-llm.ipynb)

Construir un grafo de conocimiento propio sobre tus datos privados, sin depender de pipelines externos, es posible combinando Claude con Pydantic y Neo4j. Este tutorial cubre el proceso completo: desde extraer entidades y relaciones de un texto hasta cargar el grafo en Neo4j y consultarlo con Cypher.

---

## Índice

1. [Por qué Claude es ideal para extraer grafos de conocimiento](#1-por-qué-claude-es-ideal-para-extraer-grafos-de-conocimiento)
2. [Esquema de salida con Pydantic](#2-esquema-de-salida-con-pydantic)
3. [Función extraer_grafo con tool_use de Claude](#3-función-extraer_grafo-con-tool_use-de-claude)
4. [Procesamiento de documentos largos](#4-procesamiento-de-documentos-largos)
5. [Deduplicación de entidades con embeddings](#5-deduplicación-de-entidades-con-embeddings)
6. [Guardar el grafo en Neo4j](#6-guardar-el-grafo-en-neo4j)
7. [Ejemplo completo: artículo de empresa a grafo consultable](#7-ejemplo-completo-artículo-de-empresa-a-grafo-consultable)
8. [Pipeline automático para un directorio de PDFs](#8-pipeline-automático-para-un-directorio-de-pdfs)

---

## 1. Por qué Claude es ideal para extraer grafos de conocimiento

Extraer entidades y relaciones de texto no estructurado es una tarea donde los LLMs con contexto largo tienen una ventaja enorme sobre los métodos basados en reglas o en NER clásico:

- **Contexto largo (200K tokens)**: puede procesar documentos completos sin trocearlos, capturando relaciones que se establecen en párrafos distantes.
- **Structured output / tool_use**: la función `tool_use` de Claude permite especificar un esquema JSON exacto de salida, eliminando el post-procesado frágil de texto libre.
- **Comprensión semántica**: identifica que "el CEO de Acme" y "Juan Pérez, director ejecutivo de Acme Corp." se refieren a la misma entidad, sin necesidad de reglas explícitas.
- **Multilingüe**: extrae igual de bien de documentos en español, inglés o mixtos.

---

## 2. Esquema de salida con Pydantic

Definir el esquema de salida con Pydantic antes de escribir cualquier llamada a la API hace el código más robusto y autodocumentado:

```python
from pydantic import BaseModel, Field
from typing import Literal


class Entidad(BaseModel):
    """Una entidad del dominio (persona, organización, tecnología, concepto...)."""
    id: str = Field(
        description="Identificador único en snake_case, ej: juan_perez, acme_corp"
    )
    nombre: str = Field(
        description="Nombre canónico de la entidad tal como aparece en el texto"
    )
    tipo: Literal["PERSONA", "ORGANIZACION", "TECNOLOGIA", "LUGAR", "CONCEPTO", "EVENTO"] = Field(
        description="Categoría de la entidad"
    )
    descripcion: str = Field(
        description="Breve descripción de la entidad basada en el texto (1-2 frases)"
    )
    menciones: list[str] = Field(
        default_factory=list,
        description="Otras formas en que se menciona esta entidad en el texto"
    )


class Relacion(BaseModel):
    """Una relación dirigida entre dos entidades."""
    origen_id: str = Field(description="ID de la entidad origen")
    destino_id: str = Field(description="ID de la entidad destino")
    tipo: str = Field(
        description="Tipo de relación en MAYUSCULAS_CON_GUION, ej: TRABAJA_EN, DESARROLLA, COMPITE_CON"
    )
    descripcion: str = Field(
        description="Descripción de la relación con contexto del texto"
    )
    peso: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confianza en la relación (0.0-1.0)"
    )


class GrafoConocimiento(BaseModel):
    """Grafo de conocimiento extraído de un fragmento de texto."""
    entidades: list[Entidad] = Field(default_factory=list)
    relaciones: list[Relacion] = Field(default_factory=list)
    resumen: str = Field(
        description="Resumen de una frase del contenido del fragmento"
    )

    def merge(self, otro: "GrafoConocimiento") -> "GrafoConocimiento":
        """Combina este grafo con otro, uniendo entidades y relaciones."""
        ids_existentes = {e.id for e in self.entidades}
        entidades_nuevas = [e for e in otro.entidades if e.id not in ids_existentes]

        tipos_existentes = {(r.origen_id, r.destino_id, r.tipo) for r in self.relaciones}
        relaciones_nuevas = [
            r for r in otro.relaciones
            if (r.origen_id, r.destino_id, r.tipo) not in tipos_existentes
        ]

        return GrafoConocimiento(
            entidades=self.entidades + entidades_nuevas,
            relaciones=self.relaciones + relaciones_nuevas,
            resumen=self.resumen  # Conservar el resumen del primero
        )
```

---

## 3. Función extraer_grafo con tool_use de Claude

```python
import json
import anthropic
from models import GrafoConocimiento  # el módulo del apartado anterior


client = anthropic.Anthropic()

HERRAMIENTA_GRAFO = {
    "name": "guardar_grafo_conocimiento",
    "description": (
        "Guarda el grafo de conocimiento extraído del texto con todas las "
        "entidades, relaciones y un resumen del contenido."
    ),
    "input_schema": GrafoConocimiento.model_json_schema()
}

SYSTEM_PROMPT = """Eres un experto en extracción de información y construcción de grafos de conocimiento.

Tu tarea es analizar el texto proporcionado y extraer:
1. ENTIDADES: personas, organizaciones, tecnologías, lugares, conceptos y eventos relevantes.
2. RELACIONES: conexiones significativas entre entidades con su tipo y descripción.

Reglas de extracción:
- Extrae SOLO entidades y relaciones que aparecen explícita o claramente implícitas en el texto.
- Normaliza los nombres: usa el nombre más completo como canónico.
- Los IDs deben ser únicos, descriptivos y en snake_case.
- Las relaciones deben ser dirigidas y tener un tipo claro en MAYUSCULAS_CON_GUION.
- Asigna peso 0.9-1.0 para relaciones explícitas, 0.6-0.8 para inferidas.
- Siempre llama a la herramienta guardar_grafo_conocimiento con los resultados."""


def extraer_grafo(texto: str, modelo: str = "claude-sonnet-4-6") -> GrafoConocimiento:
    """
    Extrae un grafo de conocimiento de un fragmento de texto usando Claude tool_use.

    Args:
        texto: El texto del que extraer entidades y relaciones.
        modelo: Modelo de Claude a usar.

    Returns:
        GrafoConocimiento con entidades y relaciones extraídas.
    """
    respuesta = client.messages.create(
        model=modelo,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[HERRAMIENTA_GRAFO],
        tool_choice={"type": "any"},  # Fuerza el uso de la herramienta
        messages=[
            {
                "role": "user",
                "content": f"Extrae el grafo de conocimiento del siguiente texto:\n\n{texto}"
            }
        ]
    )

    # Extraer el resultado de la llamada a la herramienta
    for bloque in respuesta.content:
        if bloque.type == "tool_use" and bloque.name == "guardar_grafo_conocimiento":
            return GrafoConocimiento.model_validate(bloque.input)

    # Si no se usó la herramienta, devolver grafo vacío
    return GrafoConocimiento(resumen="No se pudo extraer el grafo")


# Prueba rápida
texto_ejemplo = """
Acme Technologies, liderada por su CEO Juan Pérez, anunció hoy el lanzamiento de
su nueva plataforma de inteligencia artificial llamada NeuralCore. La plataforma,
desarrollada en colaboración con el Instituto de IA de Madrid, utiliza modelos
de lenguaje de gran tamaño y técnicas de GraphRAG para responder preguntas complejas
sobre corpus de documentos empresariales.

La directora técnica, María López, afirmó que NeuralCore puede procesar hasta
10.000 documentos en menos de una hora gracias a su arquitectura distribuida
basada en Apache Kafka y Neo4j. El producto compite directamente con la plataforma
Cognify de DataCorp, empresa fundada en 2019 en Barcelona.
"""

grafo = extraer_grafo(texto_ejemplo)
print(f"Entidades extraídas: {len(grafo.entidades)}")
for e in grafo.entidades:
    print(f"  [{e.tipo}] {e.nombre} (id: {e.id})")

print(f"\nRelaciones extraídas: {len(grafo.relaciones)}")
for r in grafo.relaciones:
    print(f"  {r.origen_id} --[{r.tipo}]--> {r.destino_id} (peso: {r.peso})")
```

---

## 4. Procesamiento de documentos largos

Para documentos que superan los 10.000 tokens, es mejor trocearlos y combinar los grafos parciales:

```python
import tiktoken
from models import GrafoConocimiento


def trocear_texto(
    texto: str,
    max_tokens: int = 6000,
    overlap_tokens: int = 200,
    modelo_encoding: str = "cl100k_base"
) -> list[str]:
    """
    Divide un texto largo en chunks con solapamiento para no perder contexto.

    Args:
        texto: Texto a dividir.
        max_tokens: Máximo de tokens por chunk.
        overlap_tokens: Tokens de solapamiento entre chunks consecutivos.
        modelo_encoding: Encoding del tokenizador (cl100k_base para Claude/GPT-4).

    Returns:
        Lista de strings, cada uno un chunk del texto original.
    """
    enc = tiktoken.get_encoding(modelo_encoding)
    tokens = enc.encode(texto)

    chunks = []
    inicio = 0
    while inicio < len(tokens):
        fin = min(inicio + max_tokens, len(tokens))
        chunk_tokens = tokens[inicio:fin]
        chunks.append(enc.decode(chunk_tokens))
        if fin == len(tokens):
            break
        inicio = fin - overlap_tokens  # Retroceder para el solapamiento

    return chunks


def extraer_grafo_documento_largo(
    texto: str,
    max_tokens_por_chunk: int = 6000
) -> GrafoConocimiento:
    """
    Extrae un grafo de conocimiento de un documento largo dividiéndolo en chunks
    y combinando los grafos parciales.

    Args:
        texto: Texto completo del documento.
        max_tokens_por_chunk: Tamaño máximo de cada chunk en tokens.

    Returns:
        GrafoConocimiento combinado de todos los chunks.
    """
    chunks = trocear_texto(texto, max_tokens=max_tokens_por_chunk)
    print(f"Documento dividido en {len(chunks)} chunks")

    grafo_total = GrafoConocimiento(
        entidades=[],
        relaciones=[],
        resumen="Grafo combinado de múltiples fragmentos"
    )

    for i, chunk in enumerate(chunks):
        print(f"  Procesando chunk {i + 1}/{len(chunks)}...")
        grafo_chunk = extraer_grafo(chunk)
        grafo_total = grafo_total.merge(grafo_chunk)
        print(f"    +{len(grafo_chunk.entidades)} entidades, +{len(grafo_chunk.relaciones)} relaciones")

    print(f"\nGrafo final: {len(grafo_total.entidades)} entidades, {len(grafo_total.relaciones)} relaciones")
    return grafo_total
```

---

## 5. Deduplicación de entidades con embeddings

El merge básico evita duplicados exactos por ID, pero "Acme Corp" y "Acme Technologies S.L." pueden ser la misma entidad con IDs distintos. La deduplicación semántica usa embeddings para detectarlo:

```python
import numpy as np
from openai import OpenAI
from models import Entidad, GrafoConocimiento

openai_client = OpenAI()


def calcular_similitud_coseno(v1: list[float], v2: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def generar_embedding_entidad(entidad: Entidad) -> list[float]:
    """Genera el embedding de una entidad combinando nombre, tipo y descripción."""
    texto = f"{entidad.nombre} ({entidad.tipo}): {entidad.descripcion}"
    respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return respuesta.data[0].embedding


def deduplicar_entidades(
    entidades: list[Entidad],
    umbral_similitud: float = 0.92
) -> tuple[list[Entidad], dict[str, str]]:
    """
    Detecta y fusiona entidades duplicadas usando similitud de embeddings.

    Args:
        entidades: Lista de entidades a deduplicar.
        umbral_similitud: Umbral de similitud coseno (0.92 = muy conservador).

    Returns:
        (lista de entidades deduplicadas, mapa id_duplicado -> id_canónico)
    """
    if not entidades:
        return [], {}

    print(f"Calculando embeddings para {len(entidades)} entidades...")
    embeddings = [generar_embedding_entidad(e) for e in entidades]

    # Union-Find para agrupar duplicados
    padre = {e.id: e.id for e in entidades}

    def encontrar_raiz(x):
        if padre[x] != x:
            padre[x] = encontrar_raiz(padre[x])
        return padre[x]

    def unir(x, y):
        padre[encontrar_raiz(x)] = encontrar_raiz(y)

    # Comparar todos los pares del mismo tipo
    for i, ei in enumerate(entidades):
        for j, ej in enumerate(entidades[i + 1:], start=i + 1):
            if ei.tipo != ej.tipo:
                continue  # Solo deduplicar entidades del mismo tipo
            similitud = calcular_similitud_coseno(embeddings[i], embeddings[j])
            if similitud >= umbral_similitud:
                print(f"  Duplicado detectado ({similitud:.3f}): '{ei.nombre}' == '{ej.nombre}'")
                unir(ei.id, ej.id)

    # Construir mapa de IDs y lista de entidades únicas
    grupos: dict[str, list[Entidad]] = {}
    for e in entidades:
        raiz = encontrar_raiz(e.id)
        grupos.setdefault(raiz, []).append(e)

    entidades_unicas = []
    mapa_ids: dict[str, str] = {}

    for raiz, grupo in grupos.items():
        # Elegir la entidad con nombre más largo como canónica
        canonica = max(grupo, key=lambda e: len(e.nombre))
        entidades_unicas.append(canonica)
        for e in grupo:
            mapa_ids[e.id] = canonica.id

    print(f"Deduplicación: {len(entidades)} -> {len(entidades_unicas)} entidades únicas")
    return entidades_unicas, mapa_ids


def deduplicar_grafo(grafo: GrafoConocimiento, umbral: float = 0.92) -> GrafoConocimiento:
    """Aplica deduplicación de entidades al grafo completo y actualiza las relaciones."""
    entidades_unicas, mapa_ids = deduplicar_entidades(grafo.entidades, umbral)

    # Actualizar IDs en las relaciones
    relaciones_actualizadas = []
    vistas = set()
    for r in grafo.relaciones:
        origen = mapa_ids.get(r.origen_id, r.origen_id)
        destino = mapa_ids.get(r.destino_id, r.destino_id)
        clave = (origen, destino, r.tipo)
        if origen != destino and clave not in vistas:
            r.origen_id = origen
            r.destino_id = destino
            relaciones_actualizadas.append(r)
            vistas.add(clave)

    return GrafoConocimiento(
        entidades=entidades_unicas,
        relaciones=relaciones_actualizadas,
        resumen=grafo.resumen
    )
```

---

## 6. Guardar el grafo en Neo4j

```python
from neo4j import GraphDatabase
from models import GrafoConocimiento


def guardar_grafo_en_neo4j(
    grafo: GrafoConocimiento,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password123",
    limpiar: bool = False
) -> None:
    """
    Carga el grafo de conocimiento en Neo4j.
    Usa MERGE para que sea idempotente (re-ejecutar no duplica datos).

    Args:
        grafo: El grafo a cargar.
        uri: URI de conexión a Neo4j.
        user: Usuario de Neo4j.
        password: Contraseña de Neo4j.
        limpiar: Si True, elimina todos los datos antes de cargar.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as s:
        if limpiar:
            s.run("MATCH (n) DETACH DELETE n")
            print("Base de datos limpiada.")

        # Insertar entidades con MERGE (idempotente por ID)
        for entidad in grafo.entidades:
            s.run(
                """
                MERGE (e:Entidad {id: $id})
                SET e.nombre = $nombre,
                    e.tipo = $tipo,
                    e.descripcion = $descripcion,
                    e.menciones = $menciones
                WITH e
                CALL apoc.create.addLabels(e, [$tipo]) YIELD node
                RETURN node
                """,
                {
                    "id": entidad.id,
                    "nombre": entidad.nombre,
                    "tipo": entidad.tipo,
                    "descripcion": entidad.descripcion,
                    "menciones": entidad.menciones
                }
            )

        # Insertar relaciones con MERGE
        for relacion in grafo.relaciones:
            s.run(
                """
                MATCH (origen:Entidad {id: $origen_id}), (destino:Entidad {id: $destino_id})
                MERGE (origen)-[r:RELACION {tipo: $tipo}]->(destino)
                SET r.descripcion = $descripcion,
                    r.peso = $peso,
                    r.tipo_etiqueta = $tipo
                """,
                {
                    "origen_id": relacion.origen_id,
                    "destino_id": relacion.destino_id,
                    "tipo": relacion.tipo,
                    "descripcion": relacion.descripcion,
                    "peso": relacion.peso
                }
            )

    driver.close()
    print(f"Grafo cargado: {len(grafo.entidades)} entidades, {len(grafo.relaciones)} relaciones.")
```

---

## 7. Ejemplo completo: artículo de empresa a grafo consultable

```python
from extractor import extraer_grafo, deduplicar_grafo
from neo4j_loader import guardar_grafo_en_neo4j
from neo4j import GraphDatabase

# Texto de ejemplo
articulo = """
TechVentures España, con sede en Madrid, anunció la adquisición de DataPulse,
una startup barcelonesa especializada en análisis de datos en tiempo real.
La operación, valorada en 45 millones de euros, fue asesorada por el banco
de inversión Meridian Capital.

Clara Ruiz, CEO de TechVentures, explicó que DataPulse potenciará la plataforma
de inteligencia empresarial que la compañía desarrolla desde 2022 con tecnología
de Apache Spark y Kubernetes. El equipo de DataPulse, liderado por su fundador
Andreu Vidal, se incorpora íntegramente a TechVentures.

Esta adquisición sitúa a TechVentures como competidor directo de Palantir y
de la división de analítica de IBM en el mercado ibérico.
"""

# 1. Extraer grafo
print("Extrayendo grafo de conocimiento...")
grafo = extraer_grafo(articulo)

# 2. Deduplicar entidades
grafo_limpio = deduplicar_grafo(grafo, umbral=0.90)

# 3. Guardar en Neo4j
guardar_grafo_en_neo4j(grafo_limpio, limpiar=True)

# 4. Consultar con Cypher
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
with driver.session() as s:
    # ¿Qué tecnologías usa TechVentures?
    tecnologias = s.run(
        """
        MATCH (org:ORGANIZACION {nombre: "TechVentures España"})-[r]->(t:TECNOLOGIA)
        RETURN t.nombre AS tecnologia, r.tipo_etiqueta AS relacion
        """
    )
    print("\nTecnologías de TechVentures:")
    for rec in tecnologias:
        print(f"  [{rec['relacion']}] {rec['tecnologia']}")

    # ¿Quién compite con quién?
    competidores = s.run(
        """
        MATCH (a)-[r:RELACION {tipo_etiqueta: "COMPITE_CON"}]->(b)
        RETURN a.nombre AS empresa, b.nombre AS competidor
        """
    )
    print("\nRelaciones de competencia:")
    for rec in competidores:
        print(f"  {rec['empresa']} <-> {rec['competidor']}")

driver.close()
```

---

## 8. Pipeline automático para un directorio de PDFs

```python
import os
from pathlib import Path
import fitz  # pymupdf


def pdf_a_texto(ruta_pdf: Path) -> str:
    """Extrae el texto completo de un PDF usando pymupdf."""
    doc = fitz.open(str(ruta_pdf))
    paginas = [pagina.get_text() for pagina in doc]
    doc.close()
    return "\n\n".join(paginas)


def procesar_directorio_pdfs(
    directorio: str | Path,
    umbral_deduplicacion: float = 0.92,
    max_tokens_chunk: int = 6000
) -> GrafoConocimiento:
    """
    Procesa todos los PDFs de un directorio y construye un grafo de conocimiento unificado.

    Args:
        directorio: Ruta al directorio con los PDFs.
        umbral_deduplicacion: Umbral de similitud para deduplicar entidades.
        max_tokens_chunk: Tamaño máximo de chunk para documentos largos.

    Returns:
        GrafoConocimiento combinado de todos los PDFs.
    """
    directorio = Path(directorio)
    pdfs = list(directorio.glob("**/*.pdf"))

    if not pdfs:
        raise ValueError(f"No se encontraron PDFs en {directorio}")

    print(f"Procesando {len(pdfs)} PDFs en {directorio}")

    grafo_total = GrafoConocimiento(
        entidades=[], relaciones=[], resumen="Corpus completo"
    )

    for i, pdf in enumerate(pdfs):
        print(f"\n[{i + 1}/{len(pdfs)}] {pdf.name}")
        try:
            texto = pdf_a_texto(pdf)
            if len(texto.strip()) < 100:
                print(f"  Saltando (texto insuficiente: {len(texto)} chars)")
                continue

            grafo_pdf = extraer_grafo_documento_largo(texto, max_tokens_chunk)
            grafo_total = grafo_total.merge(grafo_pdf)
            print(f"  Grafo acumulado: {len(grafo_total.entidades)} entidades")

        except Exception as e:
            print(f"  Error procesando {pdf.name}: {e}")
            continue

    # Deduplicar el grafo combinado
    print("\nDeduplicando grafo completo...")
    grafo_final = deduplicar_grafo(grafo_total, umbral=umbral_deduplicacion)

    # Guardar en Neo4j
    guardar_grafo_en_neo4j(grafo_final, limpiar=True)

    return grafo_final


# Uso
grafo = procesar_directorio_pdfs(
    directorio="./documentos",
    umbral_deduplicacion=0.92
)
print(f"\nGrafo final: {len(grafo.entidades)} entidades, {len(grafo.relaciones)} relaciones")
```

---

**Siguiente:** [04 — GraphRAG en producción: comparativa y casos empresariales](./04-graphrag-en-produccion.md)
