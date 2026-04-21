# GraphRAG en Producción: Comparativa y Casos Empresariales

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/graphrag/04-graphrag-en-produccion.ipynb)

Elegir la arquitectura de RAG correcta para un caso empresarial concreto requiere entender las fortalezas y limitaciones de cada enfoque. Este tutorial cubre la comparativa técnica, el RAG híbrido que combina lo mejor de grafos y vectores, estrategias de actualización incremental y las métricas de producción que importan.

---

## Índice

1. [Comparativa técnica: GraphRAG vs VectorRAG vs RAG híbrido](#1-comparativa-técnica-graphrag-vs-vectorrag-vs-rag-híbrido)
2. [Cuándo elegir cada arquitectura](#2-cuándo-elegir-cada-arquitectura)
3. [RAG híbrido: Neo4j + ChromaDB en una misma consulta](#3-rag-híbrido-neo4j--chromadb-en-una-misma-consulta)
4. [Clase HybridRAG con enrutamiento inteligente](#4-clase-hybridrag-con-enrutamiento-inteligente)
5. [Casos empresariales](#5-casos-empresariales)
6. [Actualización incremental del grafo](#6-actualización-incremental-del-grafo)
7. [Monitorización en producción](#7-monitorización-en-producción)
8. [Checklist de producción](#8-checklist-de-producción)

---

## 1. Comparativa técnica: GraphRAG vs VectorRAG vs RAG híbrido

| Criterio | VectorRAG | GraphRAG (Microsoft) | Grafo propio (Neo4j) | RAG Híbrido |
|---|---|---|---|---|
| **Tipo de pregunta fuerte** | Factual/local | Global/síntesis | Relacional/multi-salto | Cualquier tipo |
| **Latencia por consulta** | Baja (0.5-2 s) | Alta (5-30 s global) | Media (1-5 s) | Media (1-8 s) |
| **Coste de indexación** | Bajo ($0.01-0.10 / 100 docs) | Alto ($3-20 / 100 docs) | Medio ($0.5-2 / 100 docs) | Medio-alto |
| **Coste por consulta** | Muy bajo ($0.001-0.01) | Alto global ($0.15-0.60) | Bajo ($0.01-0.05) | Bajo-medio |
| **Actualizaciones incrementales** | Trivial | Difícil (re-indexar) | Sencillo (MERGE) | Sencillo |
| **Precisión en preguntas globales** | Baja | Alta | Alta | Alta |
| **Precisión en preguntas factuales** | Alta | Media | Media | Alta |
| **Explicabilidad** | Baja (fragmentos) | Media (comunidades) | Alta (caminos del grafo) | Alta |
| **Complejidad operativa** | Baja | Media | Alta | Alta |
| **Privacidad de datos** | Alta (local posible) | Media (necesita LLM potente) | Alta (local posible) | Alta |

---

## 2. Cuándo elegir cada arquitectura

### VectorRAG — elige cuando:
- Las preguntas son mayoritariamente **factuales y localizadas** ("¿Qué dice la cláusula 7.3?")
- El corpus cambia frecuentemente y necesitas **actualizaciones inmediatas**
- El presupuesto de indexación es limitado
- El equipo tiene experiencia con bases de datos vectoriales pero no con grafos

### GraphRAG Microsoft — elige cuando:
- Las preguntas requieren **síntesis del corpus completo** ("¿Cuáles son los riesgos principales en todos nuestros contratos?")
- El corpus es **estático o cambia poco** (informes anuales, documentación técnica)
- Puedes asumir el coste de re-indexación cuando hay cambios significativos
- Necesitas **descubrimiento temático** automatizado

### Grafo propio con Neo4j — elige cuando:
- Necesitas **preguntas relacionales multi-salto** ("¿Qué proveedores de tu proveedor principal tienen exposición a ese país?")
- El **esquema del dominio está definido** (sabes qué entidades y relaciones importan)
- Quieres **actualizaciones en tiempo real** sin re-indexar
- La **explicabilidad** es un requisito (auditoría, compliance)

### RAG Híbrido — elige cuando:
- Los usuarios hacen **tipos mixtos de preguntas** y no sabes de antemano cuál predominará
- Quieres la **mejor respuesta posible** sin preocuparte por el tipo de pregunta
- Tienes recursos para mantener dos índices en sincronía

---

## 3. RAG híbrido: Neo4j + ChromaDB en una misma consulta

El RAG híbrido mantiene en paralelo un índice vectorial (ChromaDB) y un grafo de conocimiento (Neo4j). Cada consulta se resuelve con ambos y los resultados se fusionan:

```python
import os
import chromadb
from neo4j import GraphDatabase
import anthropic
from openai import OpenAI

# Clientes globales
chroma_client = chromadb.Client()
neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password123")
    )
)
claude_client = anthropic.Anthropic()
openai_client = OpenAI()


def buscar_vectorial(
    coleccion: chromadb.Collection,
    query: str,
    top_k: int = 5
) -> list[dict]:
    """Búsqueda semántica en ChromaDB."""
    embedding_respuesta = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = embedding_respuesta.data[0].embedding

    resultados = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    fragmentos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        fragmentos.append({
            "tipo": "vectorial",
            "contenido": doc,
            "metadata": meta,
            "similitud": round(1 - dist, 4)
        })

    return fragmentos


def buscar_grafo(entidades_mencionadas: list[str], profundidad: int = 2) -> list[dict]:
    """Búsqueda en Neo4j: recupera el subgrafo alrededor de las entidades mencionadas."""
    if not entidades_mencionadas:
        return []

    with neo4j_driver.session() as s:
        resultado = s.run(
            """
            UNWIND $nombres AS nombre
            MATCH (e:Entidad)
            WHERE toLower(e.nombre) CONTAINS toLower(nombre)
            WITH e LIMIT 5
            MATCH p = (e)-[*0..{profundidad}]-(vecino)
            RETURN
                e.nombre AS entidad_central,
                vecino.nombre AS entidad_vecina,
                vecino.tipo AS tipo_vecino,
                vecino.descripcion AS descripcion,
                [rel IN relationships(p) | rel.tipo_etiqueta] AS relaciones
            LIMIT 20
            """.replace("{profundidad}", str(profundidad)),
            {"nombres": entidades_mencionadas}
        )

        hechos = []
        for rec in resultado:
            hechos.append({
                "tipo": "grafo",
                "contenido": (
                    f"{rec['entidad_central']} -> {rec['relaciones']} -> "
                    f"{rec['entidad_vecina']} ({rec['tipo_vecino']}): {rec['descripcion']}"
                ),
                "metadata": {
                    "entidad_central": rec["entidad_central"],
                    "entidad_vecina": rec["entidad_vecina"]
                },
                "similitud": 1.0  # Los hechos del grafo siempre son relevantes si se encuentran
            })

    return hechos
```

---

## 4. Clase HybridRAG con enrutamiento inteligente

En lugar de ejecutar siempre ambos backends (costoso), Claude clasifica la pregunta y decide qué backend usar:

```python
from dataclasses import dataclass, field


@dataclass
class RespuestaHybridRAG:
    """Resultado de una consulta al sistema RAG híbrido."""
    respuesta: str
    backend_usado: str
    fragmentos_vectoriales: int
    hechos_grafo: int
    tokens_usados: int


class HybridRAG:
    """
    Sistema RAG híbrido que combina búsqueda vectorial (ChromaDB)
    y búsqueda en grafo de conocimiento (Neo4j).

    El enrutamiento lo decide Claude según el tipo de pregunta:
    - 'vectorial': preguntas factuales sobre fragmentos de texto
    - 'grafo': preguntas relacionales sobre entidades y conexiones
    - 'hibrido': preguntas que se benefician de ambos enfoques
    """

    PROMPT_CLASIFICACION = """Analiza la siguiente pregunta y clasifícala según el tipo de búsqueda más adecuado:

- "vectorial": la respuesta está en un fragmento concreto de texto (fechas, cifras, definiciones, citas)
- "grafo": la respuesta requiere navegar relaciones entre entidades (quién trabaja con quién, qué conecta A con B, cadenas de relaciones)
- "hibrido": la respuesta se beneficia de ambas fuentes (preguntas complejas que mezclan hechos concretos y relaciones)

Responde SOLO con una de estas tres palabras: vectorial, grafo, hibrido

Pregunta: {pregunta}"""

    def __init__(
        self,
        coleccion_chroma: chromadb.Collection,
        top_k_vectorial: int = 5,
        profundidad_grafo: int = 2
    ) -> None:
        self.coleccion = coleccion_chroma
        self.top_k = top_k_vectorial
        self.profundidad = profundidad_grafo

    def _clasificar_pregunta(self, pregunta: str) -> str:
        """Usa Claude para clasificar el tipo de pregunta."""
        respuesta = claude_client.messages.create(
            model="claude-haiku-4-5",  # Modelo rápido y barato para clasificación
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": self.PROMPT_CLASIFICACION.format(pregunta=pregunta)
            }]
        )
        tipo = respuesta.content[0].text.strip().lower()
        return tipo if tipo in ("vectorial", "grafo", "hibrido") else "hibrido"

    def _extraer_entidades_mencionadas(self, pregunta: str) -> list[str]:
        """Extrae entidades nombradas de la pregunta para la búsqueda en grafo."""
        respuesta = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Lista las entidades nombradas (personas, organizaciones, tecnologías, lugares) "
                    f"que aparecen en esta pregunta. Una por línea, sin explicaciones.\n\n"
                    f"Pregunta: {pregunta}"
                )
            }]
        )
        lineas = respuesta.content[0].text.strip().split("\n")
        return [l.strip("- •").strip() for l in lineas if l.strip()]

    def consultar(self, pregunta: str) -> RespuestaHybridRAG:
        """
        Responde a una pregunta usando el backend más adecuado.

        Args:
            pregunta: La pregunta del usuario.

        Returns:
            RespuestaHybridRAG con la respuesta y metadatos de la consulta.
        """
        # 1. Clasificar la pregunta
        tipo = self._clasificar_pregunta(pregunta)
        print(f"Tipo de pregunta detectado: {tipo}")

        fragmentos_vectoriales = []
        hechos_grafo = []

        # 2. Recuperar contexto según el tipo
        if tipo in ("vectorial", "hibrido"):
            fragmentos_vectoriales = buscar_vectorial(self.coleccion, pregunta, self.top_k)

        if tipo in ("grafo", "hibrido"):
            entidades = self._extraer_entidades_mencionadas(pregunta)
            hechos_grafo = buscar_grafo(entidades, self.profundidad)

        # 3. Construir el prompt con el contexto recuperado
        contexto_partes = []

        if fragmentos_vectoriales:
            contexto_partes.append("## Fragmentos de documentos relevantes\n")
            for i, f in enumerate(fragmentos_vectoriales, 1):
                fuente = f["metadata"].get("fuente", "documento")
                contexto_partes.append(
                    f"[Fragmento {i} — {fuente}, similitud: {f['similitud']}]\n{f['contenido']}"
                )

        if hechos_grafo:
            contexto_partes.append("\n## Hechos del grafo de conocimiento\n")
            for h in hechos_grafo:
                contexto_partes.append(f"- {h['contenido']}")

        contexto = "\n\n".join(contexto_partes) if contexto_partes else "No se encontró contexto relevante."

        prompt = f"""Responde la siguiente pregunta basándote en el contexto proporcionado.
Cita las fuentes cuando uses información específica.

{contexto}

---

Pregunta: {pregunta}"""

        # 4. Generar respuesta con Claude
        respuesta_llm = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="Eres un asistente experto que responde con precisión basándose en el contexto dado. Cita siempre las fuentes.",
            messages=[{"role": "user", "content": prompt}]
        )

        return RespuestaHybridRAG(
            respuesta=respuesta_llm.content[0].text,
            backend_usado=tipo,
            fragmentos_vectoriales=len(fragmentos_vectoriales),
            hechos_grafo=len(hechos_grafo),
            tokens_usados=respuesta_llm.usage.input_tokens + respuesta_llm.usage.output_tokens
        )


# Ejemplo de uso
coleccion = chroma_client.get_or_create_collection("documentos_empresa")
rag = HybridRAG(coleccion)

preguntas = [
    "¿Cuál es la cláusula de penalización del contrato con Proveedor X?",           # vectorial
    "¿Qué empresas están relacionadas con el contrato C-1047?",                      # grafo
    "¿Quién puede aprobar pagos superiores a 50.000€ y qué contratos gestiona?",    # híbrido
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    resultado = rag.consultar(pregunta)
    print(f"Backend: {resultado.backend_usado} | "
          f"Fragmentos: {resultado.fragmentos_vectoriales} | "
          f"Hechos: {resultado.hechos_grafo} | "
          f"Tokens: {resultado.tokens_usados}")
    print(f"Respuesta:\n{resultado.respuesta[:300]}...")
```

---

## 5. Casos empresariales

### Due diligence legal

**Problema:** un equipo de M&A necesita analizar 800 contratos de una empresa target en 48 horas para identificar riesgos, partes relacionadas y cláusulas problemáticas.

**Solución con GraphRAG:**
- Indexar todos los contratos con el pipeline de extracción (Tutorial 03)
- Extraer entidades: partes contratantes, jurisdicciones, tecnologías, fechas, importes
- Consultas globales: "¿Qué cláusulas de change of control aparecen y en qué contratos?"
- Consultas relacionales: "¿Qué partes tienen contratos cruzados (A proveedor de B, B proveedor de A)?"

**Resultado típico:** reducción de 3-4 semanas de revisión manual a 2-3 días, con cobertura del 100% de los documentos.

### Gestión del conocimiento corporativo

**Problema:** una empresa con 500 empleados pierde conocimiento crítico cuando un experto se va. Las wikis y documentos están desorganizados y nadie sabe quién sabe qué.

**Solución con Grafo propio:**
- Extraer grafo de conocimiento de emails, documentos técnicos y tickets de soporte
- Nodos: Empleado, Proyecto, Tecnología, Cliente, Problema_Resuelto
- Consultas: "¿Quién ha resuelto problemas similares a este en los últimos 2 años?" → shortestPath

### Análisis de redes de proveedores

**Problema:** un equipo de compliance necesita identificar exposición a un país sancionado a través de la cadena de suministro.

**Solución con Neo4j:**
- Modelar la cadena: Empresa → COMPRA_A → Proveedor → SOURCEA_EN → País
- Cypher para detectar exposición a N saltos:

```cypher
MATCH camino = (mi_empresa:Empresa {nombre: "MiEmpresa"})
              -[:COMPRA_A*1..4]->
              (proveedor:Empresa)
              -[:OPERA_EN]->
              (pais:País {sancionado: true})
RETURN camino, length(camino) AS saltos
ORDER BY saltos
LIMIT 20
```

---

## 6. Actualización incremental del grafo

El grafo propio en Neo4j permite actualizaciones sin re-indexar. La estrategia de versiones mantiene la trazabilidad:

```python
from datetime import datetime
from neo4j import GraphDatabase


def actualizar_entidad_neo4j(
    driver,
    entidad_id: str,
    nuevas_propiedades: dict,
    version: str | None = None
) -> None:
    """
    Actualiza una entidad en Neo4j preservando el historial de versiones.
    Usa MERGE para no duplicar y SET para actualizar propiedades.
    """
    version = version or datetime.utcnow().isoformat()

    with driver.session() as s:
        # Guardar snapshot de la versión anterior
        s.run(
            """
            MATCH (e:Entidad {id: $id})
            WITH e, properties(e) AS props_anteriores
            CREATE (hist:HistorialEntidad {
                entidad_id: $id,
                propiedades: toString(props_anteriores),
                version: e.version_actual,
                fecha_archivo: datetime()
            })
            """,
            {"id": entidad_id}
        )

        # Actualizar con las nuevas propiedades
        s.run(
            """
            MATCH (e:Entidad {id: $id})
            SET e += $propiedades,
                e.version_actual = $version,
                e.ultima_actualizacion = datetime()
            """,
            {
                "id": entidad_id,
                "propiedades": nuevas_propiedades,
                "version": version
            }
        )


def agregar_documentos_nuevos(
    driver,
    nuevos_textos: list[dict],
) -> None:
    """
    Añade documentos nuevos al grafo existente sin borrar nada.
    Usa MERGE para que las entidades existentes se actualicen, no se dupliquen.

    Args:
        nuevos_textos: Lista de {texto: str, fuente: str, fecha: str}
    """
    from extractor import extraer_grafo, deduplicar_grafo
    from neo4j_loader import guardar_grafo_en_neo4j

    for doc in nuevos_textos:
        print(f"Procesando: {doc.get('fuente', 'documento nuevo')}")
        grafo = extraer_grafo(doc["texto"])
        # limpiar=False para MERGE sobre el grafo existente
        guardar_grafo_en_neo4j(grafo, limpiar=False)

    print(f"Añadidos {len(nuevos_textos)} documentos al grafo.")
```

---

## 7. Monitorización en producción

```python
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from statistics import mean, median

logger = logging.getLogger("graphrag")


@dataclass
class MetricaConsulta:
    """Métricas de una consulta individual."""
    timestamp: float
    tipo_backend: str
    latencia_ms: float
    tokens_entrada: int
    tokens_salida: int
    coste_usd: float
    error: str | None = None


class MonitorGraphRAG:
    """
    Monitor de métricas para un sistema GraphRAG en producción.
    Mantiene un historial de las últimas N consultas en memoria.
    """

    # Precios aproximados Claude Sonnet (actualiza según tarifa actual)
    PRECIO_INPUT_POR_MTOKEN = 3.0
    PRECIO_OUTPUT_POR_MTOKEN = 15.0

    def __init__(self, ventana: int = 1000) -> None:
        self._historial: deque[MetricaConsulta] = deque(maxlen=ventana)

    def registrar(
        self,
        tipo_backend: str,
        latencia_ms: float,
        tokens_entrada: int,
        tokens_salida: int,
        error: str | None = None
    ) -> None:
        """Registra las métricas de una consulta."""
        coste = (
            tokens_entrada / 1_000_000 * self.PRECIO_INPUT_POR_MTOKEN +
            tokens_salida / 1_000_000 * self.PRECIO_OUTPUT_POR_MTOKEN
        )
        metrica = MetricaConsulta(
            timestamp=time.time(),
            tipo_backend=tipo_backend,
            latencia_ms=latencia_ms,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            coste_usd=coste,
            error=error
        )
        self._historial.append(metrica)

        if error:
            logger.error(f"Error en consulta [{tipo_backend}]: {error}")
        elif latencia_ms > 10_000:
            logger.warning(f"Consulta lenta [{tipo_backend}]: {latencia_ms:.0f}ms")

    def resumen(self) -> dict:
        """Devuelve un resumen de las métricas del periodo."""
        if not self._historial:
            return {"error": "Sin datos"}

        latencias = [m.latencia_ms for m in self._historial]
        costes = [m.coste_usd for m in self._historial]
        errores = [m for m in self._historial if m.error]

        return {
            "total_consultas": len(self._historial),
            "tasa_error_pct": round(len(errores) / len(self._historial) * 100, 2),
            "latencia_media_ms": round(mean(latencias), 0),
            "latencia_mediana_ms": round(median(latencias), 0),
            "latencia_p95_ms": round(sorted(latencias)[int(len(latencias) * 0.95)], 0),
            "coste_total_usd": round(sum(costes), 4),
            "coste_medio_por_consulta_usd": round(mean(costes), 5),
            "backends": {
                backend: len([m for m in self._historial if m.tipo_backend == backend])
                for backend in {"vectorial", "grafo", "hibrido"}
            }
        }


# Integración en HybridRAG (añadir al método consultar)
monitor = MonitorGraphRAG()

def consultar_con_monitoreo(rag: HybridRAG, pregunta: str) -> RespuestaHybridRAG:
    """Envuelve una consulta con registro de métricas."""
    inicio = time.time()
    error = None
    resultado = None

    try:
        resultado = rag.consultar(pregunta)
    except Exception as e:
        error = str(e)
        raise
    finally:
        latencia_ms = (time.time() - inicio) * 1000
        monitor.registrar(
            tipo_backend=resultado.backend_usado if resultado else "desconocido",
            latencia_ms=latencia_ms,
            tokens_entrada=resultado.tokens_usados if resultado else 0,
            tokens_salida=0,
            error=error
        )

    return resultado
```

---

## 8. Checklist de producción

Antes de poner un sistema GraphRAG en producción, verifica estos puntos:

### Arquitectura y datos
- [ ] El esquema del grafo refleja exactamente las entidades y relaciones del dominio
- [ ] Los documentos de entrada están normalizados (encoding UTF-8, sin caracteres corruptos)
- [ ] Hay un proceso de validación que detecta extracción fallida (grafos vacíos)
- [ ] Los IDs de entidades son estables y no cambian entre ejecuciones

### Rendimiento
- [ ] Las consultas Cypher más frecuentes tienen índices en las propiedades usadas en `WHERE`
- [ ] El índice vectorial de ChromaDB o pgvector tiene HNSW activado
- [ ] La clasificación de tipo de pregunta usa un modelo pequeño (Haiku, no Sonnet)
- [ ] Hay caché de respuestas para preguntas frecuentes idénticas

### Costes
- [ ] Hay un límite de gasto mensual configurado en el proveedor de API
- [ ] Las consultas globales de GraphRAG tienen un tope de tokens por consulta
- [ ] El monitor de métricas alerta cuando el coste por consulta supera un umbral

### Fiabilidad
- [ ] Hay reintentos con backoff exponencial en las llamadas a la API
- [ ] Los errores de extracción no abortan el pipeline completo (continue en el bucle)
- [ ] Hay pruebas de regresión con un conjunto de preguntas de referencia y respuestas esperadas
- [ ] El monitor registra la tasa de error y alerta si supera el 5%

### Seguridad y privacidad
- [ ] Los datos sensibles (DNIs, datos de salud) están enmascarados antes de enviarse al LLM
- [ ] Las claves de API están en variables de entorno, nunca en el código
- [ ] El acceso a Neo4j está protegido (no usa credenciales por defecto en producción)
- [ ] Hay un proceso de eliminación de datos GDPR que borra nodos específicos del grafo

### Mantenimiento
- [ ] Hay un proceso documentado para añadir documentos nuevos sin re-indexar todo
- [ ] Las versiones del modelo LLM están fijadas (`claude-sonnet-4-6`, no `claude-latest`)
- [ ] Hay un proceso de evaluación periódica de la calidad de las respuestas

---

**Anterior:** [03 — Extracción de grafos de conocimiento con Claude](./03-extraccion-grafos-con-llm.md)

**Volver al índice:** [Bloque 30 — GraphRAG y Bases de Conocimiento](./README.md)
