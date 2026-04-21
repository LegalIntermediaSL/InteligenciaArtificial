# Bases de Datos de Grafos y Neo4j para IA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/graphrag/01-grafos-y-neo4j.ipynb)

Neo4j es la base de datos de grafos más usada en producción y el almacén natural para grafos de conocimiento en sistemas de IA. Este tutorial cubre desde los fundamentos de Cypher hasta la integración con LLMs para construir sistemas RAG relacionales.

---

## Índice

1. [Por qué los grafos capturan relaciones mejor que tablas o vectores](#1-por-qué-los-grafos-capturan-relaciones-mejor-que-tablas-o-vectores)
2. [Neo4j: conceptos fundamentales](#2-neo4j-conceptos-fundamentales)
3. [Instalación con Docker](#3-instalación-con-docker)
4. [Cypher: el lenguaje de consulta de grafos](#4-cypher-el-lenguaje-de-consulta-de-grafos)
5. [Modelado de un grafo de conocimiento empresarial](#5-modelado-de-un-grafo-de-conocimiento-empresarial)
6. [Neo4jConnector en Python](#6-neo4jconnector-en-python)
7. [CRUD de nodos y aristas](#7-crud-de-nodos-y-aristas)
8. [Consultas de caminos](#8-consultas-de-caminos)
9. [Algoritmos de grafos con GDS](#9-algoritmos-de-grafos-con-gds)
10. [Cuándo usar Neo4j vs pgvector vs ChromaDB](#10-cuándo-usar-neo4j-vs-pgvector-vs-chromadb)

---

## 1. Por qué los grafos capturan relaciones mejor que tablas o vectores

Imagina que tienes este problema: "Dado el contrato C-1047, ¿qué empleados de tu empresa tienen experiencia con las tecnologías que ese contrato requiere, y cuáles de ellos ya colaboraron con ese cliente en proyectos anteriores?"

En una base de datos relacional responder esto requiere tres o cuatro JOINs, y la consulta crece en complejidad cada vez que añades un salto más en la cadena de relaciones. En un modelo vectorial es directamente imposible: los embeddings capturan similitud semántica pero no estructura relacional.

En un grafo, la consulta es directa porque la **relación es un ciudadano de primera clase**: tiene tipo, dirección y propiedades propias. Recorrer conexiones entre entidades es la operación fundamental del modelo, no un JOIN costoso.

Ventajas concretas de los grafos para IA:

- **Razonamiento multi-salto**: "¿Quién conoce a alguien que sepa Python en el equipo de infraestructura?" se resuelve con un patrón de dos saltos.
- **Detección de comunidades**: agrupar entidades relacionadas para identificar temas, clusters o áreas de riesgo.
- **Explicabilidad**: el camino entre dos nodos es la explicación visual de por qué están relacionados.
- **Contexto enriquecido para RAG**: en lugar de chunks de texto, puedes pasar a Claude el subgrafo relevante para una entidad.

---

## 2. Neo4j: conceptos fundamentales

| Concepto | Descripción | Ejemplo |
|---|---|---|
| **Nodo** | Entidad del dominio | `(:Empleado)`, `(:Proyecto)`, `(:Tecnología)` |
| **Etiqueta** | Tipo del nodo (puede tener varias) | `(:Empleado:Directivo)` |
| **Propiedad** | Atributo del nodo o arista | `{nombre: "Ana", años_experiencia: 5}` |
| **Arista (Relationship)** | Conexión entre dos nodos | `-[:TRABAJA_EN]->`, `-[:CONOCE_TECNOLOGIA]->` |
| **Dirección** | Las aristas tienen un sentido, pero se pueden recorrer en ambos | `(a)-[:DEPENDE_DE]->(b)` |
| **Grafo de propiedades** | El modelo completo: nodos + aristas + propiedades | El modelo nativo de Neo4j |

Diferencia clave respecto a RDF/OWL: Neo4j usa el **modelo de grafo de propiedades** (property graph), que es más pragmático que los grafos semánticos formales. No requiere ontologías estrictas y es mucho más fácil de cargar con datos del mundo real.

---

## 3. Instalación con Docker

```bash
# Neo4j Community Edition con el plugin Graph Data Science
docker run \
  --name neo4j-ia \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.* \
  -d neo4j:5.18-community

# Esperar a que arranque (~15 segundos) y verificar
docker logs neo4j-ia --tail 20
```

Interfaz web disponible en `http://localhost:7474`. Conéctate con usuario `neo4j` y contraseña `password123`.

Instalar el driver Python:

```bash
pip install neo4j
```

---

## 4. Cypher: el lenguaje de consulta de grafos

Cypher usa una sintaxis visual que representa los patrones del grafo con paréntesis (nodos) y flechas (aristas).

### Crear nodos

```cypher
// Crear nodos individuales
CREATE (:Empleado {id: 1, nombre: "Ana García", rol: "Ingeniería"})
CREATE (:Empleado {id: 2, nombre: "Luis Mora", rol: "Data Science"})
CREATE (:Tecnología {nombre: "Python", categoria: "lenguaje"})
CREATE (:Proyecto {id: "P-001", nombre: "Plataforma RAG", estado: "activo"})

// Crear nodo y devolver referencia
CREATE (p:Proyecto {id: "P-002", nombre: "Chatbot Legal", estado: "activo"})
RETURN p
```

### Crear aristas

```cypher
// Conectar empleado con tecnología
MATCH (e:Empleado {nombre: "Ana García"}), (t:Tecnología {nombre: "Python"})
CREATE (e)-[:CONOCE_TECNOLOGIA {nivel: "experto", años: 6}]->(t)

// Conectar empleado con proyecto
MATCH (e:Empleado {nombre: "Ana García"}), (p:Proyecto {id: "P-001"})
CREATE (e)-[:TRABAJA_EN {rol: "tech lead", desde: "2024-01"}]->(p)
```

### Consultas con MATCH y WHERE

```cypher
// Todos los empleados que conocen Python con nivel experto
MATCH (e:Empleado)-[:CONOCE_TECNOLOGIA {nivel: "experto"}]->(t:Tecnología {nombre: "Python"})
RETURN e.nombre, e.rol

// Proyectos activos con sus empleados
MATCH (e:Empleado)-[:TRABAJA_EN]->(p:Proyecto {estado: "activo"})
RETURN p.nombre AS proyecto, collect(e.nombre) AS equipo

// Empleados de Data Science que trabajan en proyectos con Python
MATCH (e:Empleado {rol: "Data Science"})-[:TRABAJA_EN]->(p:Proyecto)
WHERE EXISTS {
  MATCH (p)<-[:TRABAJA_EN]-(:Empleado)-[:CONOCE_TECNOLOGIA]->(:Tecnología {nombre: "Python"})
}
RETURN e.nombre, p.nombre
```

### WITH para encadenar pasos

```cypher
// Tecnologías más usadas en proyectos activos (pipeline en dos pasos)
MATCH (e:Empleado)-[:TRABAJA_EN]->(p:Proyecto {estado: "activo"})
WITH p, collect(e) AS equipo
MATCH (miembro:Empleado)-[:CONOCE_TECNOLOGIA]->(t:Tecnología)
WHERE miembro IN equipo
WITH t.nombre AS tecnologia, count(*) AS frecuencia
ORDER BY frecuencia DESC
RETURN tecnologia, frecuencia
LIMIT 10
```

---

## 5. Modelado de un grafo de conocimiento empresarial

Esquema para una empresa de tecnología con tres entidades principales:

```
(:Empleado {id, nombre, email, departamento, fecha_ingreso})
    -[:CONOCE_TECNOLOGIA {nivel: "junior|medio|senior|experto", años}]->
(:Tecnología {nombre, categoria, version_actual})

(:Empleado)
    -[:TRABAJA_EN {rol, desde, horas_semanales}]->
(:Proyecto {id, nombre, cliente, presupuesto, estado, fecha_inicio})

(:Proyecto)
    -[:USA_TECNOLOGIA]->
(:Tecnología)

(:Empleado)
    -[:REPORTA_A]->
(:Empleado)
```

Este esquema permite responder preguntas como:
- ¿Quién puede sustituir a un empleado en un proyecto dado?
- ¿Qué tecnologías usa cada cliente de forma indirecta (a través de proyectos)?
- ¿Cuál es la cadena de mando completa hasta el CEO?

---

## 6. Neo4jConnector en Python

```python
import os
from contextlib import contextmanager
from typing import Any

from neo4j import GraphDatabase, Driver, Session


class Neo4jConnector:
    """Conector reutilizable para Neo4j con gestión automática de sesiones."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password123")
        self._driver: Driver | None = None

    def connect(self) -> None:
        """Abre la conexión al servidor Neo4j."""
        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        self._driver.verify_connectivity()
        print(f"Conectado a Neo4j en {self.uri}")

    def close(self) -> None:
        """Cierra la conexión."""
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self):
        """Context manager que proporciona una sesión y la cierra automáticamente."""
        if not self._driver:
            self.connect()
        s = self._driver.session()
        try:
            yield s
        finally:
            s.close()

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Ejecuta una consulta Cypher y devuelve los resultados como lista de dicts."""
        with self.session() as s:
            result = s.run(cypher, params or {})
            return [dict(record) for record in result]

    def execute(self, cypher: str, params: dict | None = None) -> None:
        """Ejecuta una instrucción Cypher sin devolver resultados (CREATE, DELETE, etc.)."""
        with self.session() as s:
            s.run(cypher, params or {})

    def limpiar_base_de_datos(self) -> None:
        """Elimina todos los nodos y aristas. Usar solo en desarrollo."""
        self.execute("MATCH (n) DETACH DELETE n")
        print("Base de datos limpiada.")


# Uso básico
db = Neo4jConnector()
db.connect()
```

---

## 7. CRUD de nodos y aristas

```python
from neo4j_connector import Neo4jConnector  # módulo del apartado anterior

db = Neo4jConnector()


def crear_empleado(db: Neo4jConnector, datos: dict) -> dict:
    """Crea un nodo Empleado y lo devuelve."""
    resultado = db.query(
        """
        CREATE (e:Empleado {
            id: $id,
            nombre: $nombre,
            email: $email,
            departamento: $departamento
        })
        RETURN e
        """,
        datos
    )
    return resultado[0]["e"] if resultado else {}


def crear_tecnologia(db: Neo4jConnector, nombre: str, categoria: str) -> None:
    """Crea un nodo Tecnología (idempotente con MERGE)."""
    db.execute(
        "MERGE (:Tecnología {nombre: $nombre, categoria: $categoria})",
        {"nombre": nombre, "categoria": categoria}
    )


def asignar_habilidad(
    db: Neo4jConnector,
    empleado_id: int,
    tecnologia: str,
    nivel: str,
    años: int
) -> None:
    """Conecta un Empleado con una Tecnología creando la arista CONOCE_TECNOLOGIA."""
    db.execute(
        """
        MATCH (e:Empleado {id: $empleado_id}), (t:Tecnología {nombre: $tecnologia})
        MERGE (e)-[r:CONOCE_TECNOLOGIA]->(t)
        SET r.nivel = $nivel, r.años = $años
        """,
        {
            "empleado_id": empleado_id,
            "tecnologia": tecnologia,
            "nivel": nivel,
            "años": años
        }
    )


def buscar_expertos(db: Neo4jConnector, tecnologia: str) -> list[dict]:
    """Devuelve empleados con nivel 'experto' en una tecnología."""
    return db.query(
        """
        MATCH (e:Empleado)-[r:CONOCE_TECNOLOGIA {nivel: 'experto'}]->(t:Tecnología {nombre: $tecnologia})
        RETURN e.nombre AS nombre, e.departamento AS departamento, r.años AS años_experiencia
        ORDER BY r.años DESC
        """,
        {"tecnologia": tecnologia}
    )


def eliminar_empleado(db: Neo4jConnector, empleado_id: int) -> None:
    """Elimina un empleado y todas sus relaciones (DETACH DELETE)."""
    db.execute(
        "MATCH (e:Empleado {id: $id}) DETACH DELETE e",
        {"id": empleado_id}
    )


# Ejemplo de uso
db.connect()

crear_tecnologia(db, "Python", "lenguaje")
crear_tecnologia(db, "Neo4j", "base_de_datos")
crear_tecnologia(db, "LangChain", "framework")

crear_empleado(db, {"id": 1, "nombre": "Ana García", "email": "ana@empresa.com", "departamento": "Ingeniería"})
crear_empleado(db, {"id": 2, "nombre": "Luis Mora", "email": "luis@empresa.com", "departamento": "Data Science"})

asignar_habilidad(db, 1, "Python", "experto", 6)
asignar_habilidad(db, 1, "Neo4j", "senior", 3)
asignar_habilidad(db, 2, "Python", "experto", 4)
asignar_habilidad(db, 2, "LangChain", "medio", 1)

expertos_python = buscar_expertos(db, "Python")
for exp in expertos_python:
    print(f"{exp['nombre']} ({exp['departamento']}) — {exp['años_experiencia']} años")
```

---

## 8. Consultas de caminos

Los caminos son la característica que distingue a los grafos: la capacidad de encontrar la ruta más corta entre dos nodos arbitrarios.

```python
def camino_mas_corto(db: Neo4jConnector, nombre_origen: str, nombre_destino: str) -> list:
    """
    Encuentra el camino más corto (por número de saltos) entre dos empleados.
    Útil para descubrir conexiones ocultas en organigramas o redes de colaboración.
    """
    resultado = db.query(
        """
        MATCH (origen:Empleado {nombre: $origen}), (destino:Empleado {nombre: $destino}),
              p = shortestPath((origen)-[*]-(destino))
        RETURN [n IN nodes(p) | coalesce(n.nombre, n.nombre)] AS camino,
               length(p) AS saltos
        """,
        {"origen": nombre_origen, "destino": nombre_destino}
    )
    return resultado[0] if resultado else {}


def todos_los_caminos_cortos(
    db: Neo4jConnector,
    nombre_origen: str,
    nombre_destino: str,
    max_saltos: int = 5
) -> list[dict]:
    """Devuelve todos los caminos más cortos entre dos nodos (puede ser más de uno)."""
    return db.query(
        """
        MATCH (origen:Empleado {nombre: $origen}), (destino:Empleado {nombre: $destino}),
              p = allShortestPaths((origen)-[*..{max_saltos}]-(destino))
        RETURN [n IN nodes(p) | coalesce(n.nombre, labels(n)[0])] AS camino,
               length(p) AS saltos
        """,
        {"origen": nombre_origen, "destino": nombre_destino, "max_saltos": max_saltos}
    )


def empleados_a_distancia_n(
    db: Neo4jConnector,
    nombre: str,
    distancia: int = 2
) -> list[dict]:
    """
    Encuentra todos los empleados a exactamente N saltos de relación.
    Ejemplo: con distancia=2, encuentra 'amigos de amigos' en la red.
    """
    return db.query(
        f"""
        MATCH (origen:Empleado {{nombre: $nombre}})-[*{distancia}]-(destino:Empleado)
        WHERE origen <> destino
        RETURN DISTINCT destino.nombre AS nombre, destino.departamento AS departamento
        """,
        {"nombre": nombre}
    )
```

---

## 9. Algoritmos de grafos con GDS

El plugin **Graph Data Science (GDS)** de Neo4j incluye más de 65 algoritmos de grafos listos para usar. Los más relevantes para IA:

```python
def calcular_pagerank(db: Neo4jConnector, proyeccion: str = "empleados-proyecto") -> list[dict]:
    """
    PageRank: identifica los nodos más influyentes en la red.
    Un empleado con alto PageRank es central en muchas colaboraciones.
    """
    # 1. Crear proyección del grafo en memoria
    db.execute(
        """
        CALL gds.graph.project(
            $proyeccion,
            ['Empleado', 'Proyecto'],
            {TRABAJA_EN: {orientation: 'UNDIRECTED'}}
        )
        """,
        {"proyeccion": proyeccion}
    )

    # 2. Ejecutar PageRank
    resultados = db.query(
        """
        CALL gds.pageRank.stream($proyeccion)
        YIELD nodeId, score
        MATCH (n) WHERE id(n) = nodeId AND n:Empleado
        RETURN n.nombre AS nombre, score
        ORDER BY score DESC
        LIMIT 10
        """,
        {"proyeccion": proyeccion}
    )

    # 3. Eliminar proyección de memoria
    db.execute("CALL gds.graph.drop($proyeccion)", {"proyeccion": proyeccion})

    return resultados


def calcular_betweenness(db: Neo4jConnector) -> list[dict]:
    """
    Betweenness centrality: identifica empleados que actúan como puentes
    entre diferentes partes de la organización (brokers de información).
    """
    db.execute(
        """
        CALL gds.graph.project(
            'org-graph',
            ['Empleado'],
            {REPORTA_A: {orientation: 'UNDIRECTED'}}
        )
        """
    )
    resultados = db.query(
        """
        CALL gds.betweenness.stream('org-graph')
        YIELD nodeId, score
        MATCH (e:Empleado) WHERE id(e) = nodeId
        RETURN e.nombre AS nombre, round(score, 2) AS centralidad
        ORDER BY centralidad DESC
        LIMIT 10
        """
    )
    db.execute("CALL gds.graph.drop('org-graph')")
    return resultados
```

---

## 10. Cuándo usar Neo4j vs pgvector vs ChromaDB

| Criterio | Neo4j | pgvector | ChromaDB |
|---|---|---|---|
| **Tipo de dato primario** | Relaciones entre entidades | Embeddings + metadatos SQL | Embeddings + metadatos clave-valor |
| **Consulta característica** | "¿Qué conecta A con B?" | "¿Qué texto es similar a X?" | "¿Qué texto es similar a X?" |
| **Razonamiento multi-salto** | Nativo y eficiente | No soportado | No soportado |
| **Búsqueda semántica** | Con plugins (vector index) | Nativo | Nativo |
| **Joins y transacciones** | Limitados | ACID completo (PostgreSQL) | No |
| **Escala** | Millones de nodos/aristas | 10-50 M vectores con HNSW | Hasta ~1 M vectores en local |
| **Infraestructura** | Docker o Neo4j AuraDB | PostgreSQL existente | Librería Python embebida |
| **Coste en cloud** | Neo4j AuraDB (~$65/mes starter) | Incluido en PostgreSQL | Chroma Cloud o self-hosted |
| **Mejor para** | Knowledge graphs, grafos org, cadenas de suministro | RAG sobre datos estructurados mixtos | Prototipos y RAG simple |

**Regla práctica:**

- Si tus preguntas implican **relaciones entre entidades** con múltiples saltos → Neo4j.
- Si necesitas **búsqueda semántica combinada con filtros SQL exactos** → pgvector.
- Si quieres un **RAG rápido de levantar** sin infraestructura → ChromaDB.
- Para **producción con preguntas globales** sobre un corpus → GraphRAG + Neo4j.

---

**Siguiente:** [02 — GraphRAG de Microsoft: indexación y búsqueda](./02-graphrag-microsoft.md)
