# Bloque 30 — GraphRAG y Bases de Conocimiento

> **Bloque:** GraphRAG y bases de conocimiento · **Nivel:** Avanzado · **Tiempo estimado:** 6-8 horas

---

## Qué aprenderás en este bloque

El RAG vectorial convencional recupera fragmentos de texto similares a la consulta, pero tiene un talón de Aquiles: las **preguntas globales**. Si preguntas "¿Cuáles son los temas principales de este corpus de 500 documentos?" o "¿Qué relación existe entre el proveedor A y la empresa B?", la búsqueda por similitud coseno no puede responder porque ningún fragmento individual contiene esa información.

**GraphRAG** resuelve esto construyendo un **grafo de conocimiento** sobre tus documentos: extrae entidades, relaciones y comunidades temáticas, y usa esa estructura para responder preguntas que requieren razonamiento sobre el conjunto completo. La combinación de grafos + vectores + LLMs define el estado del arte actual en sistemas de recuperación de información empresarial.

Este bloque cubre tres tecnologías complementarias:

- **Neo4j**: base de datos de grafos de producción, el estándar para knowledge graphs empresariales.
- **GraphRAG (Microsoft)**: pipeline open-source de indexación y búsqueda basado en grafos de conocimiento.
- **Extracción con Claude**: usar LLMs para construir grafos de conocimiento desde texto no estructurado.

---

## Tutoriales del bloque

| # | Archivo | Tema | Tiempo estimado |
|---|---|---|---|
| 01 | [01-grafos-y-neo4j.md](./01-grafos-y-neo4j.md) | Bases de datos de grafos y Neo4j para IA | 90 min |
| 02 | [02-graphrag-microsoft.md](./02-graphrag-microsoft.md) | GraphRAG de Microsoft: indexación y búsqueda con grafos de conocimiento | 100 min |
| 03 | [03-extraccion-grafos-con-llm.md](./03-extraccion-grafos-con-llm.md) | Extracción de grafos de conocimiento con Claude | 90 min |
| 04 | [04-graphrag-en-produccion.md](./04-graphrag-en-produccion.md) | GraphRAG en producción: comparativa y casos empresariales | 80 min |

---

## Por qué GraphRAG supera al RAG vectorial en preguntas complejas

| Tipo de pregunta | RAG vectorial | GraphRAG |
|---|---|---|
| "¿Qué dice el contrato sobre el plazo de pago?" | Excelente | Bueno |
| "¿Cuáles son los temas recurrentes en 200 contratos?" | Pobre | Excelente |
| "¿Qué camino conecta al proveedor X con el cliente Y?" | Imposible | Excelente |
| "¿Quién mencionó esta tecnología en los últimos 6 meses?" | Regular | Excelente |
| "Resume los principales riesgos detectados en todo el corpus" | Pobre | Excelente |

La clave es que GraphRAG no busca fragmentos: **razona sobre comunidades de entidades relacionadas** que han sido extraídas y organizadas durante la indexación.

---

## Requisitos de instalación

Instala las dependencias principales del bloque:

```bash
pip install graphrag neo4j langchain-community anthropic openai
```

Dependencias complementarias según el tutorial:

```bash
# Para extracción de grafos con Pydantic y procesado de PDFs
pip install pydantic pymupdf tiktoken

# Para algoritmos de grafos sobre Neo4j (GDS)
pip install graphdatascience

# Para RAG híbrido (grafo + vectores)
pip install chromadb sentence-transformers
```

Variables de entorno necesarias:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."          # Usado por el pipeline de GraphRAG de Microsoft
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="tu-password"
```

---

## Infraestructura: Neo4j con Docker

El tutorial 01 usa Neo4j localmente. La forma más sencilla de levantarlo es con Docker Compose:

```bash
# Levantar Neo4j con la imagen oficial
docker run \
  --name neo4j-ia \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  -d neo4j:5.18-community

# Verificar que está levantado
docker ps | grep neo4j

# Abrir la interfaz web
open http://localhost:7474
```

Credenciales por defecto: usuario `neo4j`, contraseña `password123`.

---

## Cómo está organizado este bloque

El recorrido recomendado es lineal, pero cada tutorial puede leerse de forma independiente:

1. **Tutorial 01 — Neo4j**: aprende los fundamentos de los grafos de conocimiento y Cypher. Aunque no uses Neo4j en producción, los conceptos (nodo, arista, propiedad, comunidad) son esenciales para entender GraphRAG.

2. **Tutorial 02 — GraphRAG Microsoft**: el pipeline más potente para preguntas globales sobre un corpus. Requiere una clave de API de OpenAI o Claude configurada como LLM personalizado.

3. **Tutorial 03 — Extracción con Claude**: cuando quieres construir un grafo de conocimiento propio sobre tus datos privados, sin depender de un pipeline externo. Ideal para datos sensibles o esquemas de dominio específicos.

4. **Tutorial 04 — Producción**: cómo elegir la arquitectura correcta para cada caso, cómo combinar grafos y vectores, y qué monitorizar en producción.

---

**Siguiente:** [01 — Bases de datos de grafos y Neo4j para IA](./01-grafos-y-neo4j.md)
