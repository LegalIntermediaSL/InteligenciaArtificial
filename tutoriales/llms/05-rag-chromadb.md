# 05 — RAG completo con ChromaDB

> **Bloque:** LLMs · **Nivel:** Intermedio-Avanzado · **Tiempo estimado:** 40 min

---

## Índice

1. [Qué es RAG y por qué ChromaDB](#1-qué-es-rag-y-por-qué-chromadb)
2. [Instalación y configuración](#2-instalación-y-configuración)
3. [Fase 1: Ingesta de documentos](#3-fase-1-ingesta-de-documentos)
4. [Fase 2: Recuperación](#4-fase-2-recuperación)
5. [Fase 3: Generación](#5-fase-3-generación)
6. [Pipeline RAG completo](#6-pipeline-rag-completo)
7. [Técnicas avanzadas](#7-técnicas-avanzadas)
8. [Evaluación del sistema RAG](#8-evaluación-del-sistema-rag)
9. [Resumen](#9-resumen)

---

## 1. Qué es RAG y por qué ChromaDB

**RAG (Retrieval-Augmented Generation)** combina un sistema de búsqueda semántica con un LLM: en lugar de que el modelo "memorice" tu información, la recupera en tiempo real desde una base de datos de vectores.

**ChromaDB** es una base de datos vectorial open source, embebida (no requiere servidor separado), ideal para prototipos y proyectos medianos.

```
TUS DOCUMENTOS
      ↓
  [Chunking]          Dividir en fragmentos
      ↓
  [Embedding]         Convertir a vectores numéricos
      ↓
  [ChromaDB]          Almacenar vectores + texto original
      
      ↕  (en tiempo real)
      
  PREGUNTA DEL USUARIO
      ↓
  [Embedding]         Vectorizar la pregunta
      ↓
  [Búsqueda]          Top-K fragmentos más similares
      ↓
  [Claude]            Responde basándose en los fragmentos
      ↓
  RESPUESTA FUNDAMENTADA
```

---

## 2. Instalación y configuración

```bash
pip install chromadb anthropic python-dotenv sentence-transformers
```

```python
import chromadb
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# Cliente de Anthropic
client_ai = anthropic.Anthropic()

# Cliente de ChromaDB (modo embebido, sin servidor)
client_db = chromadb.PersistentClient(path="./chroma_db")

print("ChromaDB versión:", chromadb.__version__)
```

---

## 3. Fase 1: Ingesta de documentos

### 3.1 Chunking: dividir documentos en fragmentos

```python
from pathlib import Path
import re

def limpiar_texto(texto: str) -> str:
    """Normaliza espacios y elimina caracteres problemáticos."""
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip()
    return texto

def dividir_en_chunks(texto: str,
                       tamano: int = 500,
                       solapamiento: int = 50) -> list[dict]:
    """
    Divide el texto en chunks con metadatos de posición.
    Retorna lista de dicts con 'texto' e 'indice'.
    """
    palabras = texto.split()
    chunks = []
    inicio = 0

    while inicio < len(palabras):
        fin = min(inicio + tamano, len(palabras))
        chunk_texto = " ".join(palabras[inicio:fin])
        chunks.append({
            "texto": chunk_texto,
            "indice": len(chunks),
            "palabras": fin - inicio
        })
        if fin == len(palabras):
            break
        inicio = fin - solapamiento

    return chunks

# Ejemplo
documento = Path("manual_empresa.txt").read_text(encoding="utf-8")
chunks = dividir_en_chunks(documento)
print(f"Documento: {len(documento.split())} palabras → {len(chunks)} chunks")
```

### 3.2 Embeddings: convertir texto a vectores

Usaremos el modelo `all-MiniLM-L6-v2` de Sentence Transformers, que funciona en local y en español.

```python
from sentence_transformers import SentenceTransformer

modelo_embedding = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def generar_embeddings(textos: list[str]) -> list[list[float]]:
    """Genera embeddings para una lista de textos."""
    return modelo_embedding.encode(textos, show_progress_bar=True).tolist()

# Alternativa: embeddings de OpenAI (mejor calidad, requiere API key)
def generar_embeddings_openai(textos: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client_openai = OpenAI()
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )
    return [item.embedding for item in response.data]
```

### 3.3 Indexar en ChromaDB

```python
from tqdm import tqdm

def crear_coleccion(nombre: str, recrear: bool = False) -> chromadb.Collection:
    """Crea o recupera una colección en ChromaDB."""
    if recrear:
        try:
            client_db.delete_collection(nombre)
        except Exception:
            pass
    return client_db.get_or_create_collection(
        name=nombre,
        metadata={"hnsw:space": "cosine"}  # Similitud coseno
    )

def indexar_documentos(coleccion: chromadb.Collection,
                        chunks: list[dict],
                        fuente: str = "documento") -> int:
    """
    Indexa una lista de chunks en ChromaDB.
    Procesa en lotes para eficiencia.
    """
    LOTE = 100
    total_indexados = 0

    for i in tqdm(range(0, len(chunks), LOTE), desc="Indexando"):
        lote = chunks[i:i + LOTE]
        textos = [c["texto"] for c in lote]
        embeddings = generar_embeddings(textos)

        coleccion.add(
            ids=[f"{fuente}_{i + j}" for j in range(len(lote))],
            embeddings=embeddings,
            documents=textos,
            metadatas=[{
                "fuente": fuente,
                "indice": c["indice"],
                "palabras": c["palabras"]
            } for c in lote]
        )
        total_indexados += len(lote)

    return total_indexados


# ── Indexar múltiples documentos ─────────────────────────────────────────────
def indexar_directorio(directorio: str, coleccion: chromadb.Collection):
    """Indexa todos los .txt de un directorio."""
    ruta = Path(directorio)
    ficheros = list(ruta.glob("*.txt"))
    print(f"Indexando {len(ficheros)} documentos...")

    for fichero in ficheros:
        texto = limpiar_texto(fichero.read_text(encoding="utf-8"))
        chunks = dividir_en_chunks(texto)
        n = indexar_documentos(coleccion, chunks, fuente=fichero.stem)
        print(f"  ✓ {fichero.name}: {n} chunks indexados")

    print(f"\nTotal en colección: {coleccion.count()} chunks")
```

---

## 4. Fase 2: Recuperación

```python
def buscar(coleccion: chromadb.Collection,
           pregunta: str,
           top_k: int = 5,
           filtro_fuente: str = None) -> list[dict]:
    """
    Busca los fragmentos más relevantes para una pregunta.

    top_k:          Número de resultados a devolver
    filtro_fuente:  Limitar la búsqueda a un documento concreto
    """
    # Vectorizar la pregunta
    embedding_pregunta = generar_embeddings([pregunta])[0]

    # Construir filtro opcional
    where = {"fuente": filtro_fuente} if filtro_fuente else None

    # Buscar en ChromaDB
    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    # Formatear resultados
    fragmentos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        fragmentos.append({
            "texto": doc,
            "fuente": meta["fuente"],
            "similitud": round(1 - dist, 4),  # Convertir distancia a similitud
            "indice": meta["indice"]
        })

    # Ordenar por similitud descendente
    return sorted(fragmentos, key=lambda x: x["similitud"], reverse=True)


# Ejemplo de búsqueda
coleccion = crear_coleccion("base_conocimiento")
# indexar_directorio("./documentos", coleccion)  # Solo si tienes documentos

resultados = buscar(coleccion, "¿Cuál es la política de vacaciones?", top_k=3)
for r in resultados:
    print(f"[{r['similitud']:.1%}] {r['fuente']}: {r['texto'][:100]}...")
```

---

## 5. Fase 3: Generación

```python
def generar_respuesta_rag(pregunta: str,
                           fragmentos: list[dict],
                           modelo: str = "claude-sonnet-4-6") -> dict:
    """
    Genera una respuesta usando los fragmentos recuperados como contexto.
    Devuelve la respuesta y las fuentes utilizadas.
    """
    if not fragmentos:
        return {
            "respuesta": "No encontré información relevante en la base de conocimiento.",
            "fuentes": [],
            "tiene_contexto": False
        }

    # Construir el contexto con los fragmentos
    contexto = "\n\n".join([
        f"[Fuente: {f['fuente']} | Relevancia: {f['similitud']:.0%}]\n{f['texto']}"
        for f in fragmentos
    ])

    prompt = f"""Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
- Si la respuesta está en el contexto, respóndela con precisión
- Si el contexto no contiene la respuesta, di exactamente: "No tengo información sobre esto en la base de conocimiento"
- Cita la fuente entre paréntesis cuando uses información específica
- No inventes ni extrapoles información que no esté en el contexto

Contexto:
<contexto>
{contexto}
</contexto>

Pregunta: {pregunta}"""

    respuesta = client_ai.messages.create(
        model=modelo,
        max_tokens=1024,
        temperature=0.0,  # Determinista para respuestas factuales
        messages=[{"role": "user", "content": prompt}]
    )

    fuentes_usadas = list({f["fuente"] for f in fragmentos})

    return {
        "respuesta": respuesta.content[0].text,
        "fuentes": fuentes_usadas,
        "fragmentos_usados": len(fragmentos),
        "tiene_contexto": True
    }
```

---

## 6. Pipeline RAG completo

```python
class SistemaRAG:
    """
    Sistema RAG completo: ingesta, búsqueda y generación.
    """

    def __init__(self, nombre_coleccion: str = "rag_default",
                 top_k: int = 5, umbral_similitud: float = 0.3):
        self.coleccion = crear_coleccion(nombre_coleccion)
        self.top_k = top_k
        self.umbral_similitud = umbral_similitud

    def indexar(self, texto: str, fuente: str = "documento"):
        """Indexa un texto en la base de conocimiento."""
        chunks = dividir_en_chunks(limpiar_texto(texto))
        n = indexar_documentos(self.coleccion, chunks, fuente)
        print(f"✓ {fuente}: {n} chunks indexados (total: {self.coleccion.count()})")

    def indexar_fichero(self, ruta: str):
        """Indexa un fichero de texto."""
        texto = Path(ruta).read_text(encoding="utf-8")
        self.indexar(texto, fuente=Path(ruta).stem)

    def preguntar(self, pregunta: str, verbose: bool = True) -> dict:
        """Responde una pregunta usando la base de conocimiento."""

        # 1. Recuperar fragmentos relevantes
        fragmentos = buscar(self.coleccion, pregunta, top_k=self.top_k)

        # Filtrar por umbral de similitud
        fragmentos_validos = [f for f in fragmentos if f["similitud"] >= self.umbral_similitud]

        if verbose:
            print(f"🔍 Fragmentos recuperados: {len(fragmentos)} "
                  f"(válidos ≥{self.umbral_similitud:.0%}: {len(fragmentos_validos)})")

        # 2. Generar respuesta
        resultado = generar_respuesta_rag(pregunta, fragmentos_validos)

        if verbose:
            print(f"📚 Fuentes: {', '.join(resultado['fuentes']) or 'ninguna'}")
            print(f"\n💬 Respuesta:\n{resultado['respuesta']}")

        return resultado

    def estadisticas(self):
        """Muestra estadísticas de la colección."""
        print(f"Colección: {self.coleccion.name}")
        print(f"Chunks indexados: {self.coleccion.count()}")


# ── Uso del sistema ───────────────────────────────────────────────────────────
rag = SistemaRAG(nombre_coleccion="mi_base_conocimiento")

# Indexar documentos
rag.indexar("""
La empresa TechCorp fue fundada en 2010 en Madrid.
Cuenta con 150 empleados y opera en 8 países europeos.
Su producto principal es TechCloud, una plataforma SaaS de gestión empresarial.
""", fuente="sobre_techcorp")

rag.indexar("""
Política de vacaciones: Los empleados tienen derecho a 23 días laborables de vacaciones anuales.
Las vacaciones se solicitan con al menos 15 días de antelación a través del portal de RRHH.
En agosto se garantizan al menos 10 días consecutivos.
""", fuente="politica_rrhh")

# Hacer preguntas
rag.preguntar("¿Cuántos días de vacaciones tengo?")
print("\n" + "─" * 60 + "\n")
rag.preguntar("¿En qué año fue fundada la empresa?")
print("\n" + "─" * 60 + "\n")
rag.preguntar("¿Cuál es el salario mínimo?")  # No está en los documentos
```

---

## 7. Técnicas avanzadas

### Búsqueda híbrida (semántica + keyword)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BuscadorHibrido:
    """Combina búsqueda semántica (vectores) con keyword (TF-IDF)."""

    def __init__(self, coleccion: chromadb.Collection, alpha: float = 0.7):
        """
        alpha: peso de la búsqueda semántica (1 - alpha = peso keyword)
        """
        self.coleccion = coleccion
        self.alpha = alpha
        self._construir_indice_tfidf()

    def _construir_indice_tfidf(self):
        todos = self.coleccion.get(include=["documents"])
        self.documentos = todos["documents"]
        self.ids = todos["ids"]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.matriz_tfidf = self.tfidf.fit_transform(self.documentos)

    def buscar(self, pregunta: str, top_k: int = 5) -> list[dict]:
        # Búsqueda semántica
        emb = generar_embeddings([pregunta])[0]
        res_semantico = self.coleccion.query(
            query_embeddings=[emb], n_results=top_k * 2,
            include=["documents", "distances", "ids"]
        )
        scores_semanticos = {
            id_: 1 - dist
            for id_, dist in zip(res_semantico["ids"][0], res_semantico["distances"][0])
        }

        # Búsqueda keyword (TF-IDF)
        vec_pregunta = self.tfidf.transform([pregunta])
        scores_tfidf_arr = cosine_similarity(vec_pregunta, self.matriz_tfidf)[0]
        scores_keyword = {self.ids[i]: float(s) for i, s in enumerate(scores_tfidf_arr)}

        # Combinar scores (Reciprocal Rank Fusion simplificado)
        todos_ids = set(scores_semanticos) | set(scores_keyword)
        scores_combinados = {
            id_: self.alpha * scores_semanticos.get(id_, 0)
                 + (1 - self.alpha) * scores_keyword.get(id_, 0)
            for id_ in todos_ids
        }

        top_ids = sorted(scores_combinados, key=scores_combinados.get, reverse=True)[:top_k]

        resultados = self.coleccion.get(
            ids=top_ids, include=["documents", "metadatas"]
        )
        return [
            {
                "texto": doc,
                "score": round(scores_combinados[id_], 4),
                "fuente": meta.get("fuente", "")
            }
            for doc, meta, id_ in zip(
                resultados["documents"], resultados["metadatas"], top_ids
            )
        ]
```

### Re-ranking

```python
def reranker(pregunta: str, fragmentos: list[dict], top_k: int = 3) -> list[dict]:
    """
    Re-ordena los fragmentos según su relevancia real usando Claude.
    Útil cuando la similitud vectorial no es suficiente.
    """
    fragmentos_str = "\n\n".join([
        f"[{i}] {f['texto'][:200]}"
        for i, f in enumerate(fragmentos)
    ])

    prompt = f"""Ordena estos fragmentos por relevancia para responder la pregunta.
Devuelve SOLO una lista JSON de índices ordenados de más a menos relevante.
Ejemplo: [2, 0, 3, 1]

Pregunta: {pregunta}

Fragmentos:
{fragmentos_str}"""

    r = client_ai.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        orden = json.loads(r.content[0].text)
        return [fragmentos[i] for i in orden[:top_k] if i < len(fragmentos)]
    except Exception:
        return fragmentos[:top_k]
```

---

## 8. Evaluación del sistema RAG

```python
def evaluar_rag(sistema: SistemaRAG, preguntas_con_respuesta: list[dict]) -> dict:
    """
    Evalúa el sistema RAG con un conjunto de preguntas y respuestas esperadas.

    preguntas_con_respuesta: lista de dicts con 'pregunta' y 'respuesta_esperada'
    """
    resultados = []

    for item in preguntas_con_respuesta:
        resultado = sistema.preguntar(item["pregunta"], verbose=False)

        # Evaluar con Claude como juez
        prompt_eval = f"""Evalúa si la respuesta generada es equivalente a la esperada.
Responde SOLO con JSON: {{"correcto": true/false, "score": 0.0-1.0, "razon": "..."}}

Pregunta: {item['pregunta']}
Respuesta esperada: {item['respuesta_esperada']}
Respuesta generada: {resultado['respuesta']}"""

        r = client_ai.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt_eval}]
        )
        evaluacion = json.loads(r.content[0].text)

        resultados.append({
            "pregunta": item["pregunta"],
            "correcto": evaluacion["correcto"],
            "score": evaluacion["score"],
            "razon": evaluacion["razon"]
        })

    accuracy = sum(r["correcto"] for r in resultados) / len(resultados)
    score_medio = sum(r["score"] for r in resultados) / len(resultados)

    return {
        "accuracy": round(accuracy, 3),
        "score_medio": round(score_medio, 3),
        "total_preguntas": len(resultados),
        "detalle": resultados
    }
```

---

## 9. Resumen

| Fase | Herramienta | Descripción |
|---|---|---|
| **Chunking** | Python nativo | Dividir documentos en fragmentos solapados |
| **Embedding** | Sentence Transformers / OpenAI | Convertir texto a vectores |
| **Indexación** | ChromaDB | Almacenar vectores y texto |
| **Recuperación** | ChromaDB query | Búsqueda por similitud semántica |
| **Re-ranking** | Claude | Reordenar por relevancia real |
| **Generación** | Claude | Responder anclado al contexto |

**Cuándo usar cada BD vectorial:**

| BD Vectorial | Mejor para |
|---|---|
| ChromaDB | Prototipos, proyectos medianos, sin servidor |
| Pinecone | Producción a escala, sin mantenimiento |
| pgvector | Si ya tienes PostgreSQL |
| Qdrant | Alto rendimiento, filtros complejos |
| FAISS | Búsqueda en memoria, datasets grandes |

---

**Anterior:** [04 — Agentes de IA](./04-agentes-ia.md) · **Siguiente:** [06 — Fine-tuning con LoRA](./06-finetuning-lora.md)
