# Comparativa de frameworks de agentes

PydanticAI, LangGraph, CrewAI y AutoGen tienen filosofías distintas.
Esta guía te ayuda a elegir el correcto para cada situación.

---

## Resumen rápido

| Framework | Lenguaje | Filosofía | Ideal para |
|---|---|---|---|
| **PydanticAI** | Python | Tipado estricto, testeable | Agentes de producción, APIs |
| **LangGraph** | Python | Grafos de estado, control total | Flujos complejos, human-in-the-loop |
| **CrewAI** | Python | Multi-agente declarativo, roles | Pipelines de contenido, automatización |
| **AutoGen** | Python/multi | Conversaciones multi-agente | Investigación, código autónomo |
| **Mastra** | TypeScript | Full-stack, workflows, memoria | Aplicaciones web, Next.js |
| **Langchain.js** | TypeScript | Chains y RAG | RAG en Node.js, cadenas LCEL |

---

## 1. PydanticAI vs LangGraph

```
                PydanticAI              LangGraph
Paradigma       Agente = función       Agente = grafo de estados
Tipado          Pydantic + mypy        TypedDict + anotaciones
Testing         TestModel (nativo)     Mocking manual
Control flujo   Implícito (LLM decide) Explícito (defines las aristas)
Human-in-loop   Manual                 interrupt_before/after nativo
Curva aprendizaje  Baja               Media-Alta
Mejor para      APIs, agentes simples  Flujos con lógica condicional compleja
```

### Cuándo usar PydanticAI
- Tu agente tiene una entrada clara y una salida tipada
- Necesitas tests automatizados del agente
- Integras con FastAPI o cualquier framework Python
- Quieres agnósticismo de proveedor (cambia de Claude a GPT sin tocar lógica)

### Cuándo usar LangGraph
- El flujo del agente tiene ramificaciones condicionales complejas
- Necesitas persistencia de estado entre invocaciones (`SqliteSaver`, `PostgresSaver`)
- Requieres human-in-the-loop con aprobación en pasos específicos
- Construyes un agente que puede "volver atrás" en el grafo

---

## 2. PydanticAI vs CrewAI

```python
# CrewAI — declarativo, roles y tareas
from crewai import Agent, Task, Crew

investigador = Agent(
    role='Investigador Legal',
    goal='Encontrar jurisprudencia relevante',
    backstory='Experto con 20 años en derecho mercantil',
    llm='anthropic/claude-haiku-4-5-20251001',
)

redactor = Agent(
    role='Redactor de Informes',
    goal='Sintetizar hallazgos en informes claros',
    backstory='Especialista en comunicación jurídica',
    llm='anthropic/claude-haiku-4-5-20251001',
)

tarea_investigar = Task(
    description='Investiga la jurisprudencia sobre esta cláusula: {clausula}',
    expected_output='Lista de 5 casos relevantes con resumen',
    agent=investigador,
)

tarea_redactar = Task(
    description='Redacta un informe ejecutivo con los hallazgos',
    expected_output='Informe de 2 páginas con recomendaciones',
    agent=redactor,
    context=[tarea_investigar],
)

crew = Crew(agents=[investigador, redactor], tasks=[tarea_investigar, tarea_redactar])
resultado = crew.kickoff(inputs={'clausula': 'Cláusula de arbitraje...'})
```

```python
# PydanticAI — mismo pipeline, más control y tipado
from pydantic_ai import Agent
from pydantic import BaseModel

class Jurisprudencia(BaseModel):
    casos: list[str]
    fuentes: list[str]

class Informe(BaseModel):
    resumen_ejecutivo: str
    hallazgos: list[str]
    recomendaciones: list[str]

agente_investigacion = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=Jurisprudencia,
    system_prompt='Investigas jurisprudencia legal con precisión.',
)

agente_redaccion = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=Informe,
    system_prompt='Redactas informes ejecutivos claros y accionables.',
)

@agente_redaccion.tool_plain
async def obtener_jurisprudencia(clausula: str) -> dict:
    resultado = await agente_investigacion.run(clausula)
    return resultado.data.model_dump()

async def pipeline_completo(clausula: str) -> Informe:
    resultado = await agente_redaccion.run(
        f'Investiga y redacta un informe sobre: {clausula}'
    )
    return resultado.data
```

**CrewAI** es más rápido para prototipado con múltiples agentes; **PydanticAI** tiene
mejor tipado, testing y control del output.

---

## 3. AutoGen — para investigación y código autónomo

AutoGen está orientado a conversaciones multi-agente donde los agentes se hablan entre sí:

```python
import autogen

config_list = [{'model': 'claude-haiku-4-5-20251001', 'api_type': 'anthropic'}]

asistente = autogen.AssistantAgent(
    name='Asistente',
    llm_config={'config_list': config_list},
)

ejecutor = autogen.UserProxyAgent(
    name='Ejecutor',
    human_input_mode='NEVER',
    code_execution_config={'work_dir': '/tmp/autogen', 'use_docker': False},
)

# El asistente genera código; el ejecutor lo ejecuta y devuelve el resultado
ejecutor.initiate_chat(
    asistente,
    message='Escribe un script Python que descargue los últimos 10 titulares de Hacker News y los muestre ordenados por puntuación.',
)
```

**AutoGen** brilla en tareas de código autónomo y búsqueda iterativa. No está diseñado para
producción con usuarios reales (el human_input_mode puede bloquear procesos).

---

## 4. Árbol de decisión

```
¿Necesitas ejecución de código autónoma?
  └─► SÍ → AutoGen
  └─► NO → ¿Cuántos agentes interactúan?
             └─► 1 agente principal → ¿TypeScript o Python?
                   └─► TypeScript → Mastra.ai
                   └─► Python → PydanticAI
             └─► Múltiples agentes → ¿Control de flujo complejo?
                   └─► SÍ (condicionales, loops, HITL) → LangGraph
                   └─► NO (roles declarativos) → CrewAI
```

---

## 5. Comparativa de rendimiento y coste

| Métrica | PydanticAI | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Overhead por llamada | ~0ms | ~2ms | ~5ms | ~10ms |
| Facilidad de testing | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| Madurez en producción | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Documentación | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Curva de aprendizaje | Baja | Media | Baja | Media |
| Observabilidad | Logfire nativo | LangSmith | Integración manual | Nativa |

---

## 6. Compatibilidad con Claude

| Framework | Claude | GPT-4 | Gemini | Mistral | Local (Ollama) |
|---|---|---|---|---|---|
| PydanticAI | ✅ Nativo | ✅ | ✅ | ✅ | ✅ |
| LangGraph | ✅ langchain-anthropic | ✅ | ✅ | ✅ | ✅ |
| CrewAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| AutoGen | ✅ (config) | ✅ | ⚠️ | ⚠️ | ✅ |
| Mastra | ✅ @ai-sdk/anthropic | ✅ | ✅ | ✅ | ✅ |

---

## Recomendación final

- **Empieza con PydanticAI** si construyes APIs o agentes de producción en Python.
- **Usa LangGraph** cuando el flujo de decisiones del agente sea complejo o necesites HITL.
- **Usa CrewAI** para pipelines de contenido multi-agente rápidos de prototipar.
- **Usa Mastra** si tu stack es TypeScript/Next.js.
- **Usa AutoGen** para investigación con generación y ejecución de código autónoma.

La combinación más común en producción: **PydanticAI para agentes unitarios** + **LangGraph para orquestar flujos multi-paso**.
