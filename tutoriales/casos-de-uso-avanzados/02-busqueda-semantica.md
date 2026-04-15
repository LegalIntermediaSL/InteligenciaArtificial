# 02 — Búsqueda semántica con embeddings

> **Bloque:** Casos de uso avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Búsqueda semántica vs búsqueda clásica](#1-búsqueda-semántica-vs-búsqueda-clásica)
2. [Embeddings: representar texto como vectores](#2-embeddings-representar-texto-como-vectores)
3. [Indexar una base de conocimiento](#3-indexar-una-base-de-conocimiento)
4. [Búsqueda por similitud coseno](#4-búsqueda-por-similitud-coseno)
5. [RAG simple: chatbot con búsqueda semántica](#5-rag-simple-chatbot-con-búsqueda-semántica)
6. [Caso práctico: buscador de documentación interna](#6-caso-práctico-buscador-de-documentación-interna)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Búsqueda semántica vs búsqueda clásica

La búsqueda clásica compara palabras exactas. La búsqueda semántica compara el **significado** del texto.

| Característica | Búsqueda clásica (BM25/grep) | Búsqueda semántica (embeddings) |
|---|---|---|
| **Coincidencia** | Palabras exactas | Significado y contexto |
| **Query** | "error conexión base datos" | "no puedo conectarme a la DB" |
| **Sinónimos** | No los encuentra | Los entiende automáticamente |
| **Idioma** | Sensible al idioma exacto | Funciona entre idiomas (multilingüe) |
| **Velocidad** | Muy rápida | Más lenta (requiere vectores) |
| **Infraestructura** | Simple | Requiere almacenar vectores |
| **Caso de uso ideal** | Búsqueda de logs, código exacto | FAQs, documentación, soporte |

**Cuándo usar cada una:**
- Búsqueda exacta (IDs, errores, código): búsqueda clásica.
- Búsqueda por concepto o intención del usuario: búsqueda semántica.

---

## 2. Embeddings: representar texto como vectores

Un embedding convierte texto en un vector de números (por ejemplo, 1536 dimensiones). Textos similares producen vectores cercanos en el espacio vectorial.

```python
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def generar_embedding(texto: str, modelo: str = "text-embedding-3-small") -> list[float]:
    """
    Genera un embedding para un texto usando la API de OpenAI.

    Args:
        texto: El texto a convertir en vector.
        modelo: Modelo de embedding a usar.

    Returns:
        Lista de floats que representan el vector.
    """
    # Limpiar el texto
    texto = texto.replace("\n", " ").strip()

    respuesta = client.embeddings.create(
        input=texto,
        model=modelo,
    )

    return respuesta.data[0].embedding


def similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.

    Returns:
        Valor entre -1 y 1. Más alto = más similares.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Demostración conceptual
if __name__ == "__main__":
    frases = [
        "El perro corre por el parque",
        "El can galopa en el jardín",      # Muy similar
        "La gata duerme en el sofá",       # Relacionado (animales)
        "La bolsa de valores subió hoy",   # Sin relación
    ]

    print("Generando embeddings...")
    embeddings = [generar_embedding(f) for f in frases]

    print(f"\nDimensiones del vector: {len(embeddings[0])}\n")

    # Comparar la primera frase con el resto
    referencia = frases[0]
    ref_embedding = embeddings[0]

    print(f"Frase de referencia: '{referencia}'\n")
    for i in range(1, len(frases)):
        sim = similitud_coseno(ref_embedding, embeddings[i])
        print(f"  Similitud con '{frases[i]}': {sim:.4f}")
```

**Salida esperada:**
```
Similitud con 'El can galopa en el jardín': 0.8912
Similitud con 'La gata duerme en el sofá': 0.6543
Similitud con 'La bolsa de valores subió hoy': 0.2341
```

---

## 3. Indexar una base de conocimiento

Lee todos los ficheros `.md` de una carpeta, genera sus embeddings y los guarda en un fichero JSON local (sin necesidad de base de datos vectorial).

```python
import json
import hashlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def dividir_en_fragmentos(texto: str, max_chars: int = 800, solapamiento: int = 100) -> list[str]:
    """
    Divide un texto largo en fragmentos con solapamiento.

    Args:
        texto: Texto a dividir.
        max_chars: Tamaño máximo de cada fragmento.
        solapamiento: Caracteres de solapamiento entre fragmentos consecutivos.

    Returns:
        Lista de fragmentos de texto.
    """
    if len(texto) <= max_chars:
        return [texto]

    fragmentos = []
    inicio = 0

    while inicio < len(texto):
        fin = min(inicio + max_chars, len(texto))

        # Intentar cortar en un punto natural (párrafo o frase)
        if fin < len(texto):
            corte_parrafo = texto.rfind("\n\n", inicio, fin)
            corte_frase = texto.rfind(". ", inicio, fin)

            if corte_parrafo > inicio + max_chars // 2:
                fin = corte_parrafo + 2
            elif corte_frase > inicio + max_chars // 2:
                fin = corte_frase + 2

        fragmento = texto[inicio:fin].strip()
        if fragmento:
            fragmentos.append(fragmento)

        inicio = fin - solapamiento

    return fragmentos


def indexar_carpeta(
    carpeta: str,
    ruta_indice: str = "indice_semantico.json",
    extensiones: list[str] = None,
) -> dict:
    """
    Indexa todos los documentos de una carpeta generando sus embeddings.

    Args:
        carpeta: Ruta a la carpeta con documentos.
        ruta_indice: Fichero JSON donde guardar el índice.
        extensiones: Extensiones a procesar (por defecto .md y .txt).

    Returns:
        Diccionario con el índice construido.
    """
    if extensiones is None:
        extensiones = [".md", ".txt"]

    # Cargar índice existente para evitar reindexar lo que no cambió
    indice_existente = {}
    if Path(ruta_indice).exists():
        with open(ruta_indice, "r", encoding="utf-8") as f:
            datos = json.load(f)
            # Usar hash como clave para detectar cambios
            indice_existente = {item["hash"]: item for item in datos.get("documentos", [])}

    documentos = []
    carpeta_path = Path(carpeta)

    ficheros = [
        f for f in carpeta_path.rglob("*")
        if f.suffix in extensiones and f.is_file()
    ]

    print(f"Ficheros encontrados: {len(ficheros)}")

    for fichero in ficheros:
        texto_completo = fichero.read_text(encoding="utf-8")
        hash_fichero = hashlib.md5(texto_completo.encode()).hexdigest()

        fragmentos = dividir_en_fragmentos(texto_completo)

        for i, fragmento in enumerate(fragmentos):
            hash_fragmento = hashlib.md5(fragmento.encode()).hexdigest()

            # Reusar embedding si el fragmento no cambió
            if hash_fragmento in indice_existente:
                documentos.append(indice_existente[hash_fragmento])
                continue

            print(f"  Indexando: {fichero.name} (fragmento {i+1}/{len(fragmentos)})")

            embedding = client.embeddings.create(
                input=fragmento.replace("\n", " "),
                model="text-embedding-3-small",
            ).data[0].embedding

            documentos.append({
                "hash": hash_fragmento,
                "fichero": str(fichero.relative_to(carpeta_path)),
                "fragmento_num": i,
                "texto": fragmento,
                "embedding": embedding,
            })

    indice = {
        "modelo": "text-embedding-3-small",
        "total_fragmentos": len(documentos),
        "documentos": documentos,
    }

    with open(ruta_indice, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False)

    print(f"\nIndice guardado: {len(documentos)} fragmentos en '{ruta_indice}'")
    return indice


# Uso
if __name__ == "__main__":
    # Crear carpeta de prueba con documentos de ejemplo
    Path("docs_prueba").mkdir(exist_ok=True)

    Path("docs_prueba/python.md").write_text("""
# Python para principiantes
Python es un lenguaje de programación de alto nivel. Es muy popular para
análisis de datos, inteligencia artificial y desarrollo web.

## Variables
En Python no necesitas declarar el tipo de la variable:
x = 5
nombre = "Ana"
""", encoding="utf-8")

    Path("docs_prueba/ia.md").write_text("""
# Inteligencia Artificial
La IA es la simulación de la inteligencia humana por parte de máquinas.
Los modelos de lenguaje como GPT o Claude son ejemplos de IA generativa.

## Machine Learning
El aprendizaje automático permite a los modelos aprender de datos sin
ser programados explícitamente para cada tarea.
""", encoding="utf-8")

    indice = indexar_carpeta("docs_prueba", "mi_indice.json")
```

---

## 4. Búsqueda por similitud coseno

Con el índice construido, buscar los fragmentos más relevantes para una query.

```python
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def cargar_indice(ruta_indice: str) -> dict:
    """Carga el índice semántico desde disco."""
    with open(ruta_indice, "r", encoding="utf-8") as f:
        return json.load(f)


def buscar(
    query: str,
    indice: dict,
    k: int = 5,
    umbral_similitud: float = 0.3,
) -> list[dict]:
    """
    Busca los k fragmentos más relevantes para una query.

    Args:
        query: Pregunta o texto de búsqueda.
        indice: Índice cargado con cargar_indice().
        k: Número de resultados a devolver.
        umbral_similitud: Similitud mínima para incluir un resultado (0-1).

    Returns:
        Lista de resultados ordenados por similitud descendente.
    """
    # Generar embedding de la query
    embedding_query = client.embeddings.create(
        input=query.replace("\n", " "),
        model="text-embedding-3-small",
    ).data[0].embedding

    vec_query = np.array(embedding_query)

    resultados = []
    for doc in indice["documentos"]:
        vec_doc = np.array(doc["embedding"])
        similitud = float(np.dot(vec_query, vec_doc) / (
            np.linalg.norm(vec_query) * np.linalg.norm(vec_doc)
        ))

        if similitud >= umbral_similitud:
            resultados.append({
                "texto": doc["texto"],
                "fichero": doc["fichero"],
                "similitud": similitud,
                "fragmento_num": doc["fragmento_num"],
            })

    # Ordenar por similitud descendente y devolver top-k
    resultados.sort(key=lambda x: x["similitud"], reverse=True)
    return resultados[:k]


def imprimir_resultados(resultados: list[dict]):
    """Muestra los resultados de búsqueda de forma legible."""
    if not resultados:
        print("No se encontraron resultados relevantes.")
        return

    for i, res in enumerate(resultados, 1):
        print(f"\n[{i}] Similitud: {res['similitud']:.4f} — {res['fichero']}")
        print(f"    {res['texto'][:200]}...")


# Uso
if __name__ == "__main__":
    indice = cargar_indice("mi_indice.json")

    queries = [
        "¿Cómo declaro una variable en Python?",
        "¿Qué es el machine learning?",
        "¿Para qué sirve la inteligencia artificial?",
    ]

    for query in queries:
        print(f"\nBUSCANDO: {query}")
        print("-" * 50)
        resultados = buscar(query, indice, k=3)
        imprimir_resultados(resultados)
```

---

## 5. RAG simple: chatbot con búsqueda semántica

RAG (Retrieval-Augmented Generation) combina búsqueda semántica con generación de texto. Aquí, sin ChromaDB, solo con numpy y un fichero JSON.

```python
import json
import numpy as np
import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()


class ChatbotRAG:
    """Chatbot que responde usando búsqueda semántica sobre una base de conocimiento."""

    def __init__(self, ruta_indice: str, k_resultados: int = 3):
        """
        Args:
            ruta_indice: Ruta al fichero JSON del índice semántico.
            k_resultados: Número de fragmentos a recuperar por query.
        """
        with open(ruta_indice, "r", encoding="utf-8") as f:
            self.indice = json.load(f)

        self.k = k_resultados
        self.historial: list[dict] = []
        print(f"Base de conocimiento cargada: {self.indice['total_fragmentos']} fragmentos")

    def _buscar_contexto(self, query: str) -> str:
        """Busca los fragmentos más relevantes y los combina en un contexto."""
        embedding_query = openai_client.embeddings.create(
            input=query.replace("\n", " "),
            model="text-embedding-3-small",
        ).data[0].embedding

        vec_query = np.array(embedding_query)
        resultados = []

        for doc in self.indice["documentos"]:
            vec_doc = np.array(doc["embedding"])
            similitud = float(np.dot(vec_query, vec_doc) / (
                np.linalg.norm(vec_query) * np.linalg.norm(vec_doc)
            ))
            resultados.append((similitud, doc))

        resultados.sort(key=lambda x: x[0], reverse=True)
        top_k = resultados[:self.k]

        partes_contexto = []
        for similitud, doc in top_k:
            if similitud > 0.3:
                partes_contexto.append(
                    f"[Fuente: {doc['fichero']} | Relevancia: {similitud:.2f}]\n{doc['texto']}"
                )

        return "\n\n---\n\n".join(partes_contexto)

    def chat(self, pregunta: str) -> str:
        """
        Responde una pregunta usando el contexto recuperado de la base de conocimiento.

        Args:
            pregunta: Pregunta del usuario.

        Returns:
            Respuesta basada en el contexto encontrado.
        """
        contexto = self._buscar_contexto(pregunta)

        system_prompt = """Eres un asistente experto que responde preguntas basándose
EXCLUSIVAMENTE en el contexto proporcionado. 

Reglas:
- Si la respuesta está en el contexto, responde de forma clara y concisa.
- Si el contexto no contiene información suficiente, dilo explícitamente.
- Cita la fuente cuando sea relevante.
- Responde siempre en español."""

        mensaje_con_contexto = f"""Contexto recuperado de la base de conocimiento:

{contexto}

---

Pregunta del usuario: {pregunta}"""

        self.historial.append({"role": "user", "content": mensaje_con_contexto})

        respuesta = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=self.historial,
        )

        texto = respuesta.content[0].text
        self.historial.append({"role": "assistant", "content": texto})

        return texto


# Uso
if __name__ == "__main__":
    bot = ChatbotRAG("mi_indice.json", k_resultados=3)

    preguntas = [
        "¿Cómo se declaran variables en Python?",
        "¿Qué diferencia hay entre IA y machine learning?",
        "¿Para qué se usa Python?",
    ]

    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")
        print("-" * 50)
        respuesta = bot.chat(pregunta)
        print(f"Respuesta: {respuesta}\n")
```

---

## 6. Caso práctico: buscador de documentación interna

Aplicación Streamlit completa end-to-end para buscar en documentación interna.

```python
# buscador_docs.py
# Ejecutar con: streamlit run buscador_docs.py

import json
import numpy as np
import streamlit as st
import anthropic
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

# --- Funciones de utilidad ---

def generar_embedding(texto: str) -> list[float]:
    return openai_client.embeddings.create(
        input=texto.replace("\n", " "),
        model="text-embedding-3-small",
    ).data[0].embedding


def dividir_en_fragmentos(texto: str, max_chars: int = 600) -> list[str]:
    if len(texto) <= max_chars:
        return [texto]
    fragmentos, inicio = [], 0
    while inicio < len(texto):
        fin = min(inicio + max_chars, len(texto))
        if fin < len(texto):
            corte = texto.rfind("\n", inicio, fin)
            if corte > inicio + max_chars // 2:
                fin = corte + 1
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            fragmentos.append(fragmento)
        inicio = fin - 50
    return fragmentos


@st.cache_data(show_spinner="Indexando documentos...")
def construir_indice(carpeta: str) -> list[dict]:
    """Construye el índice semántico (cacheado por Streamlit)."""
    documentos = []
    for fichero in Path(carpeta).rglob("*.md"):
        texto = fichero.read_text(encoding="utf-8")
        for i, fragmento in enumerate(dividir_en_fragmentos(texto)):
            embedding = generar_embedding(fragmento)
            documentos.append({
                "fichero": fichero.name,
                "texto": fragmento,
                "embedding": embedding,
            })
    return documentos


def buscar_semantico(query: str, documentos: list[dict], k: int = 4) -> list[dict]:
    vec_query = np.array(generar_embedding(query))
    resultados = []
    for doc in documentos:
        vec_doc = np.array(doc["embedding"])
        sim = float(np.dot(vec_query, vec_doc) / (
            np.linalg.norm(vec_query) * np.linalg.norm(vec_doc)
        ))
        resultados.append({**doc, "similitud": sim})
    resultados.sort(key=lambda x: x["similitud"], reverse=True)
    return [r for r in resultados[:k] if r["similitud"] > 0.3]


def responder_con_ia(pregunta: str, contexto: str) -> str:
    respuesta = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="Responde en español usando solo el contexto. Si no encuentras la respuesta, indícalo.",
        messages=[{
            "role": "user",
            "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
        }],
    )
    return respuesta.content[0].text


# --- Interfaz Streamlit ---

st.set_page_config(page_title="Buscador de Documentación", page_icon="🔍", layout="wide")
st.title("Buscador de Documentación Interna")
st.caption("Búsqueda semántica potenciada por embeddings de OpenAI y respuestas de Claude")

with st.sidebar:
    st.header("Configuración")
    carpeta_docs = st.text_input("Carpeta de documentos", value="docs_prueba")
    k_resultados = st.slider("Fragmentos a recuperar", 1, 10, 4)
    modo = st.radio("Modo", ["Solo buscar", "Buscar y responder con IA"])

if not Path(carpeta_docs).exists():
    st.warning(f"La carpeta '{carpeta_docs}' no existe. Crea la carpeta e incluye ficheros .md.")
    st.stop()

documentos = construir_indice(carpeta_docs)
st.success(f"Base de conocimiento lista: {len(documentos)} fragmentos indexados")

query = st.text_input("Escribe tu pregunta o búsqueda:", placeholder="¿Cómo instalo el paquete X?")

if query:
    with st.spinner("Buscando..."):
        resultados = buscar_semantico(query, documentos, k=k_resultados)

    if not resultados:
        st.warning("No se encontraron resultados relevantes. Intenta reformular la búsqueda.")
    else:
        if modo == "Buscar y responder con IA":
            contexto = "\n\n".join(r["texto"] for r in resultados)
            with st.spinner("Generando respuesta con Claude..."):
                respuesta_ia = responder_con_ia(query, contexto)
            st.subheader("Respuesta")
            st.markdown(respuesta_ia)
            st.divider()

        st.subheader(f"Fragmentos relevantes ({len(resultados)} encontrados)")
        for i, res in enumerate(resultados, 1):
            with st.expander(f"[{i}] {res['fichero']} — Similitud: {res['similitud']:.3f}"):
                st.markdown(res["texto"])
```

```bash
# Para ejecutar:
pip install streamlit openai anthropic numpy python-dotenv
streamlit run buscador_docs.py
```

---

## 7. Extensiones sugeridas

| Extensión | Descripción | Tecnología |
|---|---|---|
| **Base de datos vectorial** | Escalar a millones de documentos | ChromaDB, Pinecone, Weaviate |
| **Reranking** | Mejorar la precisión con un segundo modelo | Cohere Rerank |
| **Embeddings multilingüe** | Buscar en varios idiomas | `multilingual-e5-large` |
| **Indexado incremental** | Añadir documentos sin reindexar todo | Detección de cambios por hash |
| **Metadata filtering** | Filtrar por fecha, autor, categoría antes de buscar | JSON + numpy filtering |
| **Evaluación de RAG** | Medir la calidad de las respuestas | RAGAS framework |

---

**Siguiente:** [03 — Structured output con Instructor](./03-structured-output-instructor.md)
