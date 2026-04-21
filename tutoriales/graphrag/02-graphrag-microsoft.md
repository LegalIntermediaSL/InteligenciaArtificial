# GraphRAG de Microsoft: Indexación y Búsqueda con Grafos de Conocimiento

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/graphrag/02-graphrag-microsoft.ipynb)

Microsoft GraphRAG es un pipeline open-source que convierte un corpus de documentos en un grafo de conocimiento estructurado y permite responder tanto preguntas específicas como preguntas globales sobre todo el corpus. Es el enfoque más potente conocido actualmente para preguntas que requieren síntesis de información distribuida en muchos documentos.

---

## Índice

1. [Qué es GraphRAG y por qué supera al RAG vectorial en preguntas globales](#1-qué-es-graphrag-y-por-qué-supera-al-rag-vectorial-en-preguntas-globales)
2. [Arquitectura: indexación vs búsqueda](#2-arquitectura-indexación-vs-búsqueda)
3. [Instalación](#3-instalación)
4. [Configuración con Claude como LLM](#4-configuración-con-claude-como-llm)
5. [Pipeline de indexación](#5-pipeline-de-indexación)
6. [Búsqueda global vs local](#6-búsqueda-global-vs-local)
7. [GraphRAGClient: uso programático desde Python](#7-graphragclient-uso-programático-desde-python)
8. [Coste de indexación: estimación tokens y precio](#8-coste-de-indexación-estimación-tokens-y-precio)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)

---

## 1. Qué es GraphRAG y por qué supera al RAG vectorial en preguntas globales

El RAG vectorial convencional funciona así: dada una pregunta, busca los K fragmentos más similares en un índice de embeddings y los pasa como contexto al LLM. Esto funciona bien para preguntas factuales localizadas ("¿Qué dice la cláusula 4 del contrato?"), pero falla en preguntas que requieren síntesis del corpus completo:

- "¿Cuáles son los temas principales en estos 300 informes?"
- "¿Qué tendencias se repiten a lo largo de todas las entrevistas?"
- "¿Cuál es la posición general de la empresa sobre la sostenibilidad?"

Para estas preguntas, ningún fragmento individual contiene la respuesta. GraphRAG resuelve esto con un paso de **pre-indexación**: antes de cualquier consulta, el pipeline extrae entidades, relaciones y comunidades temáticas usando un LLM, y construye un grafo de conocimiento. En tiempo de búsqueda, puede razonar sobre ese grafo en lugar de sobre fragmentos de texto.

---

## 2. Arquitectura: indexación vs búsqueda

### Fase de indexación (ejecuta una vez por corpus)

```
Documentos → Chunking → Extracción de entidades y relaciones (LLM)
           → Construcción del grafo → Detección de comunidades (algoritmo Leiden)
           → Resúmenes de comunidades (LLM) → Índices (Parquet + embeddings)
```

Cada paso es costoso en tokens pero se ejecuta una sola vez. El resultado es un directorio con archivos Parquet que contienen el grafo, las entidades, las relaciones y los resúmenes de comunidades a varios niveles de granularidad.

### Fase de búsqueda (ejecuta en cada consulta)

**Búsqueda global:** usa los resúmenes de comunidades para responder preguntas sobre el conjunto completo. Genera respuestas parciales para cada comunidad y las sintetiza. Costo: proporcional al número de comunidades.

**Búsqueda local:** usa el grafo para encontrar la vecindad de las entidades mencionadas en la pregunta. Más rápida y barata que la global, ideal para preguntas sobre entidades específicas.

---

## 3. Instalación

```bash
# Instalar GraphRAG (Python 3.10+)
pip install graphrag

# Verificar instalación
python -m graphrag --version
```

Crear la estructura de directorios para un proyecto:

```bash
# Crear directorio raíz del proyecto
mkdir ragtest
cd ragtest

# Crear directorio para los documentos de entrada
mkdir -p input

# Copiar documentos .txt al directorio input
cp /ruta/a/tus/documentos/*.txt input/

# Inicializar la configuración de GraphRAG
python -m graphrag init --root .
```

Esto genera:
- `settings.yaml` — configuración del pipeline
- `.env` — variables de entorno (claves API)
- `prompts/` — prompts personalizables para extracción

---

## 4. Configuración con Claude como LLM

El archivo `settings.yaml` controla todos los aspectos del pipeline. Por defecto apunta a OpenAI, pero puedes configurar Claude a través de la interfaz compatible con OpenAI:

```yaml
# settings.yaml
encoding_model: cl100k_base
skip_workflows: []

llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: claude-sonnet-4-6
  model_supports_json: true
  max_tokens: 4096
  temperature: 0
  # Usar la API de Anthropic con el endpoint compatible
  api_base: https://api.anthropic.com/v1
  # Nota: para usar Claude directamente configura GRAPHRAG_API_KEY=sk-ant-...

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    # Los embeddings sí necesitan OpenAI por ahora
    api_base: https://api.openai.com/v1

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: "input"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

entity_extraction:
  max_gleanings: 1

summarize_descriptions:
  max_length: 500

community_reports:
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

umap:
  enabled: false
```

Variables de entorno en `.env`:

```bash
# .env
GRAPHRAG_API_KEY=sk-ant-api03-...          # Tu clave de Anthropic
OPENAI_API_KEY=sk-...                       # Para embeddings (text-embedding-3-small)
```

---

## 5. Pipeline de indexación

```bash
# Desde el directorio raíz del proyecto
cd ragtest

# Ejecutar la indexación completa
python -m graphrag index --root .

# Con logging detallado para ver el progreso
python -m graphrag index --root . --verbose

# La indexación puede tardar varios minutos según el tamaño del corpus
# Progreso visible en logs/indexing-engine.log
```

Estructura de salida tras la indexación:

```
output/
├── artifacts/
│   ├── create_final_communities.parquet      # Comunidades detectadas
│   ├── create_final_community_reports.parquet # Resúmenes de comunidades
│   ├── create_final_entities.parquet          # Entidades extraídas
│   ├── create_final_relationships.parquet     # Relaciones extraídas
│   ├── create_final_nodes.parquet             # Nodos del grafo
│   ├── create_final_text_units.parquet        # Chunks de texto
│   └── create_final_documents.parquet         # Documentos originales
```

Inspeccionar los resultados con pandas:

```python
import pandas as pd

# Ver entidades extraídas
entidades = pd.read_parquet("output/artifacts/create_final_entities.parquet")
print(f"Total entidades: {len(entidades)}")
print(entidades[["title", "type", "description"]].head(10))

# Ver relaciones
relaciones = pd.read_parquet("output/artifacts/create_final_relationships.parquet")
print(f"Total relaciones: {len(relaciones)}")
print(relaciones[["source", "target", "description", "weight"]].head(10))

# Ver comunidades
comunidades = pd.read_parquet("output/artifacts/create_final_community_reports.parquet")
print(f"Total comunidades: {len(comunidades)}")
print(comunidades[["community", "title", "summary"]].head(5))
```

---

## 6. Búsqueda global vs local

### Búsqueda global — preguntas sobre el corpus completo

```bash
# Pregunta que requiere síntesis de todo el corpus
python -m graphrag query \
  --root ./ragtest \
  --method global \
  --query "¿Cuáles son los principales temas que se repiten en todos los documentos?"
```

La búsqueda global recorre los resúmenes de comunidades a nivel jerárquico y sintetiza una respuesta. Es la más potente pero también la más costosa (puede consumir miles de tokens por consulta).

### Búsqueda local — preguntas sobre entidades específicas

```bash
# Pregunta sobre una entidad conocida en el corpus
python -m graphrag query \
  --root ./ragtest \
  --method local \
  --query "¿Qué se dice sobre Ana García y su rol en los proyectos de IA?"
```

La búsqueda local navega el subgrafo alrededor de las entidades relevantes. Es más rápida, más barata y más precisa para preguntas concretas.

### Diferencias clave

| Aspecto | Búsqueda global | Búsqueda local |
|---|---|---|
| **Tipo de pregunta** | Síntesis del corpus completo | Preguntas sobre entidades/relaciones específicas |
| **Mecanismo** | Resúmenes de comunidades jerárquicas | Subgrafo de entidades relevantes |
| **Coste de tokens** | Alto (O(comunidades)) | Bajo-medio (O(vecindad)) |
| **Latencia** | Alta (5-30 segundos) | Baja (1-5 segundos) |
| **Cobertura** | Todo el corpus | Vecindad de las entidades mencionadas |
| **Ejemplo** | "¿Qué temas aparecen en el corpus?" | "¿Qué proyectos lidera María?" |

---

## 7. GraphRAGClient: uso programático desde Python

Envolver el CLI en una clase Python permite integrarlo en aplicaciones:

```python
import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ResultadoGraphRAG:
    """Resultado de una consulta a GraphRAG."""
    respuesta: str
    metodo: str
    query: str


class GraphRAGClient:
    """
    Cliente programático para el pipeline de GraphRAG de Microsoft.
    Envuelve el CLI para su uso desde código Python.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise ValueError(f"El directorio raíz no existe: {self.root}")

    def indexar(self, verbose: bool = False) -> None:
        """
        Ejecuta el pipeline de indexación completo.
        Puede tardar varios minutos según el tamaño del corpus.
        """
        cmd = ["python", "-m", "graphrag", "index", "--root", str(self.root)]
        if verbose:
            cmd.append("--verbose")

        print(f"Iniciando indexación en {self.root}...")
        resultado = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(self.root)
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                f"Error durante la indexación:\n{resultado.stderr}"
            )
        print("Indexación completada.")

    def consultar(
        self,
        query: str,
        metodo: str = "local",
    ) -> ResultadoGraphRAG:
        """
        Ejecuta una consulta sobre el grafo indexado.

        Args:
            query: La pregunta a responder.
            metodo: 'global' para preguntas sobre el corpus completo,
                    'local' para preguntas sobre entidades específicas.

        Returns:
            ResultadoGraphRAG con la respuesta y metadatos.
        """
        if metodo not in ("global", "local"):
            raise ValueError("metodo debe ser 'global' o 'local'")

        cmd = [
            "python", "-m", "graphrag", "query",
            "--root", str(self.root),
            "--method", metodo,
            "--query", query
        ]

        resultado = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.root)
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                f"Error durante la consulta:\n{resultado.stderr}"
            )

        return ResultadoGraphRAG(
            respuesta=resultado.stdout.strip(),
            metodo=metodo,
            query=query
        )

    def indexado(self) -> bool:
        """Verifica si el corpus ya ha sido indexado (existe el directorio output)."""
        artifacts = self.root / "output" / "artifacts"
        return artifacts.exists() and any(artifacts.glob("*.parquet"))

    def estadisticas(self) -> dict:
        """Devuelve estadísticas básicas del corpus indexado."""
        import pandas as pd

        artifacts = self.root / "output" / "artifacts"
        if not self.indexado():
            return {"error": "El corpus no ha sido indexado todavía"}

        stats = {}
        for fichero, clave in [
            ("create_final_entities.parquet", "entidades"),
            ("create_final_relationships.parquet", "relaciones"),
            ("create_final_communities.parquet", "comunidades"),
            ("create_final_documents.parquet", "documentos"),
        ]:
            ruta = artifacts / fichero
            if ruta.exists():
                df = pd.read_parquet(ruta)
                stats[clave] = len(df)

        return stats


# Ejemplo de uso
cliente = GraphRAGClient("./ragtest")

if not cliente.indexado():
    cliente.indexar()

print("Estadísticas del corpus:", cliente.estadisticas())

# Pregunta global
resultado_global = cliente.consultar(
    "¿Cuáles son los temas principales del corpus?",
    metodo="global"
)
print(f"\n[Global] {resultado_global.respuesta}")

# Pregunta local
resultado_local = cliente.consultar(
    "¿Qué tecnologías se mencionan más frecuentemente?",
    metodo="local"
)
print(f"\n[Local] {resultado_local.respuesta}")
```

---

## 8. Coste de indexación: estimación tokens y precio

El coste principal de GraphRAG está en la fase de indexación (no en la búsqueda). Estimación para un corpus típico de 100 documentos de 1-2 páginas cada uno:

| Paso del pipeline | Tokens aprox. | Coste con Claude Sonnet ($3/MTok input) |
|---|---|---|
| Extracción de entidades y relaciones | ~500K tokens | ~$1.50 |
| Resúmenes de entidades | ~200K tokens | ~$0.60 |
| Resúmenes de comunidades (nivel 0-2) | ~400K tokens | ~$1.20 |
| **Total indexación** | **~1.1M tokens** | **~$3.30** |

Para la búsqueda en tiempo de consulta:

| Tipo de consulta | Tokens por consulta | Coste aprox. |
|---|---|---|
| Búsqueda local | ~5K-20K tokens | ~$0.01-0.06 |
| Búsqueda global (corpus pequeño) | ~50K-200K tokens | ~$0.15-0.60 |

**Regla práctica:** la indexación de GraphRAG cuesta entre 5-20x más tokens que indexar el mismo corpus con RAG vectorial convencional, pero se ejecuta una sola vez y habilita capacidades de razonamiento global que no son posibles de otro modo.

Para reducir costes:
1. Usa `claude-haiku-4-5` para la indexación y `claude-sonnet-4-6` solo para las consultas.
2. Reduce `max_gleanings` a 0 en `settings.yaml` (elimina la extracción iterativa).
3. Usa chunks más grandes (`size: 2000`) para reducir el número de llamadas al LLM.

---

## 9. Limitaciones conocidas

GraphRAG es potente pero tiene restricciones importantes a tener en cuenta antes de adoptarlo en producción:

**Actualizaciones incrementales**: el pipeline no soporta añadir documentos nuevos sin reindexar todo el corpus. Si tu corpus cambia frecuentemente, los costes de re-indexación se acumulan. Alternativa: segmentar el corpus en "bloques históricos" que no cambian y reindexar solo el bloque más reciente.

**Idiomas**: el pipeline fue diseñado para inglés. En español funciona pero puede requerir personalizar los prompts en `prompts/` para mejorar la calidad de la extracción.

**Documentos de formato mixto**: solo acepta archivos `.txt` como entrada. PDFs y Word deben convertirse antes. Usa `pymupdf` o `python-docx` para la conversión.

**Privacidad**: durante la indexación, el texto de todos los documentos se envía al LLM configurado. Si los datos son confidenciales, configura un LLM local (Ollama + Llama 3) en lugar de la API de Anthropic/OpenAI.

**Escalabilidad**: corpus muy grandes (>10.000 documentos) pueden requerir ajustar la paralelización y los límites de rate del API en `settings.yaml`.

```yaml
# Ajustes de paralelización para corpus grandes
llm:
  max_retries: 10
  requests_per_minute: 50     # Ajusta según tu tier de API
  tokens_per_minute: 100000
  concurrent_requests: 5
```

---

**Siguiente:** [03 — Extracción de grafos de conocimiento con Claude](./03-extraccion-grafos-con-llm.md)
