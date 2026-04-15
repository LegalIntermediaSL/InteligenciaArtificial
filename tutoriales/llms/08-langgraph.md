# 08 — LangGraph: Agentes con estado y grafos de flujo

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [¿Por qué LangGraph?](#1-por-qué-langgraph)
2. [Conceptos fundamentales](#2-conceptos-fundamentales)
3. [Tu primer grafo](#3-tu-primer-grafo)
4. [Edges condicionales y ciclos](#4-edges-condicionales-y-ciclos)
5. [Agente con herramientas](#5-agente-con-herramientas)
6. [Memoria y checkpointing](#6-memoria-y-checkpointing)
7. [Human-in-the-loop](#7-human-in-the-loop)
8. [Patrones multi-agente](#8-patrones-multi-agente)
9. [Streaming y observabilidad](#9-streaming-y-observabilidad)
10. [Cuándo usar LangGraph vs otras opciones](#10-cuándo-usar-langgraph-vs-otras-opciones)
11. [Resumen](#11-resumen)

---

## 1. ¿Por qué LangGraph?

Los agentes construidos con bucles simples (como el patrón ReAct visto en el tutorial 04) funcionan bien para tareas lineales, pero se quedan cortos en cuanto aparecen:

- **Flujos ramificados**: rutas distintas según el resultado de un paso
- **Ciclos complejos**: volver a un paso anterior con nueva información
- **Estado persistente**: recordar lo que pasó en pasos previos
- **Paralelismo**: ejecutar varias ramas a la vez
- **Intervención humana**: pausar y esperar aprobación antes de continuar

LangGraph resuelve todo esto modelando el agente como un **grafo dirigido**:

```
                    ┌──────────────────────────────────────────────┐
                    │              GRAFO DE AGENTE                  │
                    │                                              │
    Input ─────► [Nodo A] ─────► [Nodo B] ─────► [Nodo C] ──► Output
                    │                │                              │
                    │                └──► [Nodo D] ─────────────► ─┘
                    │                         ↑                    │
                    │                         │   (ciclo)          │
                    └─────────────────────────┘                    │
                    └──────────────────────────────────────────────┘
```

### Comparativa: bucle simple vs LangGraph

| Aspecto | Bucle while simple | LangGraph |
|---|---|---|
| Estructura | Lineal | Grafo dirigido (nodos + edges) |
| Ciclos | Manual y frágil | Nativo, con control explícito |
| Estado | Variables locales | Estado tipado compartido |
| Memoria entre sesiones | No | Sí (checkpointers) |
| Human-in-the-loop | Muy difícil | Primera clase |
| Paralelismo | No | Sí (fan-out / fan-in) |
| Observabilidad | `print()` | Tracing con LangSmith |
| Streaming | Manual | Nativo (`.stream()`) |

---

## 2. Conceptos fundamentales

### 2.1 Estado (`State`)

El **estado** es el objeto compartido que fluye por todos los nodos del grafo. Se define con `TypedDict` de Python.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class EstadoConversacion(TypedDict):
    messages: Annotated[list, add_messages]  # lista de mensajes; add_messages los acumula
    iteraciones: int
    tarea_completada: bool
```

`Annotated[list, add_messages]` es un reductor: en lugar de reemplazar la lista en cada paso, la extiende. LangGraph admite reductores propios para cualquier campo.

### 2.2 Nodos (`Nodes`)

Un nodo es una **función Python** que recibe el estado y devuelve un diccionario con las claves a actualizar:

```python
def mi_nodo(state: EstadoConversacion) -> dict:
    # Leer estado
    mensajes = state["messages"]
    # Hacer algo...
    return {"iteraciones": state["iteraciones"] + 1}
```

### 2.3 Edges (aristas)

Conectan nodos y determinan el flujo:

| Tipo | Función | Cuándo usar |
|---|---|---|
| Edge normal | `graph.add_edge(A, B)` | Siempre ir de A a B |
| Edge condicional | `graph.add_conditional_edges(A, fn)` | La función decide adónde ir |
| Entry point | `graph.set_entry_point(A)` | Primer nodo a ejecutar |
| `END` | `graph.add_edge(A, END)` | Finalizar el grafo |

### 2.4 El grafo

```python
from langgraph.graph import StateGraph, END

# 1. Construir
graph = StateGraph(EstadoConversacion)

# 2. Añadir nodos
graph.add_node("nodo_a", funcion_a)
graph.add_node("nodo_b", funcion_b)

# 3. Conectar
graph.set_entry_point("nodo_a")
graph.add_edge("nodo_a", "nodo_b")
graph.add_edge("nodo_b", END)

# 4. Compilar
app = graph.compile()

# 5. Ejecutar
resultado = app.invoke({"messages": [], "iteraciones": 0, "tarea_completada": False})
```

---

## 3. Tu primer grafo

Un grafo mínimo que procesa una pregunta con Claude:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Estado
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Modelo
llm = ChatAnthropic(model="claude-sonnet-4-6")

# Nodo: llamar al LLM
def llamar_llm(state: State) -> dict:
    respuesta = llm.invoke(state["messages"])
    return {"messages": [respuesta]}

# Grafo
grafo = StateGraph(State)
grafo.add_node("llm", llamar_llm)
grafo.set_entry_point("llm")
grafo.add_edge("llm", END)
app = grafo.compile()

# Ejecución
resultado = app.invoke({
    "messages": [HumanMessage(content="¿Qué es LangGraph en una frase?")]
})
print(resultado["messages"][-1].content)
```

Esto parece trivial porque solo tiene un nodo, pero establece la base para todo lo que viene.

---

## 4. Edges condicionales y ciclos

### 4.1 El patrón agéntico con ciclos

El patrón central de LangGraph para agentes es:

```
[llm] ──► ¿herramienta? ──sí──► [herramientas] ──► [llm] (ciclo)
                         └─no──► END
```

La **edge condicional** toma el estado y devuelve el nombre del próximo nodo:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatAnthropic(model="claude-sonnet-4-6")

def debe_usar_herramienta(state: State) -> Literal["herramientas", "__end__"]:
    """Decide si el último mensaje del LLM incluye llamadas a herramientas."""
    ultimo = state["messages"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "herramientas"
    return "__end__"

def nodo_llm(state: State) -> dict:
    respuesta = llm.invoke(state["messages"])
    return {"messages": [respuesta]}

grafo = StateGraph(State)
grafo.add_node("llm", nodo_llm)
grafo.set_entry_point("llm")
grafo.add_conditional_edges(
    "llm",
    debe_usar_herramienta,
    # Mapa: valor_devuelto → nombre_nodo
    {"herramientas": "herramientas", "__end__": END}
)
```

### 4.2 Múltiples rutas

```python
from typing import Literal

def enrutar_tarea(state: State) -> Literal["investigar", "redactar", "revisar", "__end__"]:
    """Decide el siguiente paso según el contenido del último mensaje."""
    ultimo = state["messages"][-1].content.lower()
    if "busca" in ultimo or "investiga" in ultimo:
        return "investigar"
    if "redacta" in ultimo or "escribe" in ultimo:
        return "redactar"
    if "revisa" in ultimo or "corrige" in ultimo:
        return "revisar"
    return "__end__"

grafo.add_conditional_edges(
    "coordinador",
    enrutar_tarea,
    {
        "investigar": "nodo_investigacion",
        "redactar": "nodo_redaccion",
        "revisar": "nodo_revision",
        "__end__": END
    }
)
```

---

## 5. Agente con herramientas

LangGraph incluye `ToolNode`, un nodo prebuilteado que ejecuta herramientas automáticamente:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import math

# ── Definir herramientas con @tool ─────────────────────────────────────────────

@tool
def calcular(expresion: str) -> float:
    """Evalúa una expresión matemática. Admite operadores estándar y funciones de math."""
    return eval(expresion, {"__builtins__": {}}, {"math": math})

@tool
def convertir_moneda(cantidad: float, de: str, a: str) -> str:
    """Convierte entre EUR, USD y GBP usando tasas fijas de ejemplo."""
    tasas = {"EUR": 1.0, "USD": 1.08, "GBP": 0.86}
    if de not in tasas or a not in tasas:
        return f"Moneda no soportada. Disponibles: {list(tasas.keys())}"
    resultado = cantidad / tasas[de] * tasas[a]
    return f"{cantidad} {de} = {resultado:.2f} {a}"

@tool
def buscar_capital(pais: str) -> str:
    """Devuelve la capital de un país (base de datos de ejemplo)."""
    capitales = {
        "España": "Madrid", "Francia": "París", "Alemania": "Berlín",
        "Italia": "Roma", "Portugal": "Lisboa", "Japón": "Tokio"
    }
    return capitales.get(pais, f"Capital de '{pais}' no encontrada")

# ── Estado y modelo ────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

herramientas = [calcular, convertir_moneda, buscar_capital]
llm = ChatAnthropic(model="claude-sonnet-4-6")
llm_con_tools = llm.bind_tools(herramientas)

# ── Nodos ──────────────────────────────────────────────────────────────────────

def nodo_llm(state: State) -> dict:
    return {"messages": [llm_con_tools.invoke(state["messages"])]}

nodo_herramientas = ToolNode(herramientas)

# ── Grafo ──────────────────────────────────────────────────────────────────────

grafo = StateGraph(State)
grafo.add_node("llm", nodo_llm)
grafo.add_node("herramientas", nodo_herramientas)

grafo.set_entry_point("llm")

# tools_condition: devuelve "tools" si hay tool_calls, END en caso contrario
grafo.add_conditional_edges("llm", tools_condition)
grafo.add_edge("herramientas", "llm")   # ← ciclo: después de ejecutar, vuelve al LLM

app = grafo.compile()

# ── Pruebas ────────────────────────────────────────────────────────────────────

preguntas = [
    "¿Cuánto es la raíz cúbica de 1728?",
    "Convierte 500 USD a EUR y dime cuál es la capital de Francia.",
    "Si tengo 1000 GBP y los convierto a USD, ¿cuánto obtengo? ¿Y si los invierto al 5% anual durante 3 años?",
]

for pregunta in preguntas:
    print(f"\n{'─'*60}")
    print(f"Pregunta: {pregunta}")
    resultado = app.invoke({"messages": [HumanMessage(content=pregunta)]})
    print(f"Respuesta: {resultado['messages'][-1].content}")
```

### 5.1 Visualizar el grafo

```python
# Requiere: pip install pygraphviz  (o pip install grandalf para una alternativa ligera)
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # Fallback: diagrama en texto Mermaid
    print(app.get_graph().draw_mermaid())
```

Salida Mermaid típica:

```
graph TD
    __start__ --> llm
    llm --> herramientas
    herramientas --> llm
    llm --> __end__
```

---

## 6. Memoria y checkpointing

Por defecto, cada llamada a `.invoke()` es una sesión nueva. Para persistir el estado entre llamadas, LangGraph usa **checkpointers**.

### 6.1 MemorySaver (en memoria)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app_con_memoria = grafo.compile(checkpointer=checkpointer)

# El thread_id identifica la conversación (como un session_id)
config = {"configurable": {"thread_id": "usuario_42"}}

# Turno 1
app_con_memoria.invoke(
    {"messages": [HumanMessage(content="Mi nombre es Ana.")]},
    config=config
)

# Turno 2 — el grafo recuerda el contexto anterior
resultado = app_con_memoria.invoke(
    {"messages": [HumanMessage(content="¿Cómo me llamo?")]},
    config=config
)
print(resultado["messages"][-1].content)
# → "Te llamas Ana."
```

### 6.2 SqliteSaver (persistencia real)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Persiste en disco entre reinicios del proceso
with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    app_persistente = grafo.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "proyecto_ia"}}
    resultado = app_persistente.invoke(
        {"messages": [HumanMessage(content="Hola")]},
        config=config
    )
```

### 6.3 Inspeccionar el estado guardado

```python
# Ver el estado actual de un thread
snapshot = app_con_memoria.get_state(config)
print(snapshot.values)          # El estado completo
print(snapshot.next)            # Próximo nodo a ejecutar (si está pausado)

# Ver el historial completo de pasos
for checkpoint in app_con_memoria.get_state_history(config):
    print(checkpoint.metadata["step"], checkpoint.values.keys())
```

---

## 7. Human-in-the-loop

El checkpointer permite **pausar el grafo** antes de acciones críticas y esperar aprobación humana.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    esperando_aprobacion: bool

llm = ChatAnthropic(model="claude-sonnet-4-6")

def planificar(state: State) -> dict:
    """El LLM genera un plan de acción."""
    respuesta = llm.invoke(state["messages"])
    return {"messages": [respuesta], "esperando_aprobacion": True}

def ejecutar(state: State) -> dict:
    """Ejecuta el plan (solo llega aquí si fue aprobado)."""
    return {"messages": [AIMessage(content="✅ Plan ejecutado correctamente.")],
            "esperando_aprobacion": False}

def revisar_aprobacion(state: State) -> str:
    if state.get("esperando_aprobacion"):
        return "esperando"
    return "ejecutar"

checkpointer = MemorySaver()
grafo = StateGraph(State)
grafo.add_node("planificar", planificar)
grafo.add_node("ejecutar", ejecutar)

grafo.set_entry_point("planificar")
grafo.add_conditional_edges(
    "planificar",
    revisar_aprobacion,
    {"esperando": END, "ejecutar": "ejecutar"}   # ← END pausa el grafo
)
grafo.add_edge("ejecutar", END)

app = grafo.compile(
    checkpointer=checkpointer,
    interrupt_before=["ejecutar"]  # ← pausa ANTES de ejecutar
)

config = {"configurable": {"thread_id": "aprobacion_001"}}

# Paso 1: generar plan (se pausa antes de ejecutar)
app.invoke(
    {"messages": [HumanMessage(content="Envía un email a todos los clientes con el informe de ventas.")],
     "esperando_aprobacion": False},
    config=config
)
snapshot = app.get_state(config)
print("Plan:", snapshot.values["messages"][-1].content)
print("Esperando aprobación — próximo nodo:", snapshot.next)

# Paso 2: el humano aprueba → continuar desde donde se pausó
aprobado = True  # en producción, esto vendría de una UI o CLI
if aprobado:
    app.invoke(None, config=config)  # None = reanudar sin nuevos inputs
    print("Ejecución completada.")
```

---

## 8. Patrones multi-agente

### 8.1 Supervisor + subagentes

```
          ┌──────────────┐
  Input ─►│  Supervisor  │◄────────────────────┐
          └──────┬───────┘                     │
                 │ decide                      │ resultado
          ┌──────┼──────────────┐              │
          ▼      ▼              ▼              │
    [Investigador] [Redactor] [Revisor] ───────┘
```

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

class Estado(TypedDict):
    messages: Annotated[list, add_messages]
    siguiente: str
    iteraciones: int

llm = ChatAnthropic(model="claude-sonnet-4-6")
AGENTES = ["investigador", "redactor", "revisor", "FINISH"]

def supervisor(state: Estado) -> dict:
    """Decide qué agente actúa a continuación."""
    prompt_supervisor = f"""Eres un supervisor coordinando un equipo de trabajo.
Agentes disponibles: {', '.join(AGENTES[:-1])}.
Basándote en la conversación, decide quién debe actuar ahora.
Responde SOLO con el nombre del agente o 'FINISH' si la tarea está completa."""

    respuesta = llm.invoke(
        [SystemMessage(content=prompt_supervisor)] + state["messages"]
    )
    siguiente = respuesta.content.strip()
    if siguiente not in AGENTES:
        siguiente = "FINISH"
    return {"siguiente": siguiente, "iteraciones": state["iteraciones"] + 1}

def crear_agente(nombre: str, descripcion: str):
    """Factory: crea un nodo-agente con un rol específico."""
    def nodo(state: Estado) -> dict:
        respuesta = llm.invoke(
            [SystemMessage(content=f"Eres el {nombre}. {descripcion}")] +
            state["messages"]
        )
        return {"messages": [respuesta]}
    nodo.__name__ = nombre
    return nodo

nodo_investigador = crear_agente(
    "investigador",
    "Recopila información relevante, identifica fuentes y datos clave."
)
nodo_redactor = crear_agente(
    "redactor",
    "Redacta contenido claro y bien estructurado basándote en la investigación."
)
nodo_revisor = crear_agente(
    "revisor",
    "Revisa el contenido, corrige errores y sugiere mejoras."
)

def enrutar_supervisor(state: Estado) -> str:
    siguiente = state.get("siguiente", "FINISH")
    if siguiente == "FINISH" or state["iteraciones"] > 6:
        return END
    return siguiente

grafo = StateGraph(Estado)
grafo.add_node("supervisor", supervisor)
grafo.add_node("investigador", nodo_investigador)
grafo.add_node("redactor", nodo_redactor)
grafo.add_node("revisor", nodo_revisor)

grafo.set_entry_point("supervisor")
grafo.add_conditional_edges(
    "supervisor",
    enrutar_supervisor,
    {"investigador": "investigador", "redactor": "redactor",
     "revisor": "revisor", END: END}
)
# Todos los subagentes vuelven al supervisor
for agente in ["investigador", "redactor", "revisor"]:
    grafo.add_edge(agente, "supervisor")

app_multi = grafo.compile()

resultado = app_multi.invoke({
    "messages": [HumanMessage(content="Escribe un breve artículo sobre los beneficios de los modelos de IA locales.")],
    "siguiente": "",
    "iteraciones": 0
})
print(resultado["messages"][-1].content)
```

### 8.2 Fan-out / Fan-in (paralelismo)

```python
# Ejecutar múltiples nodos en paralelo y combinar sus resultados
grafo.add_node("analisis_sentimiento", nodo_sentimiento)
grafo.add_node("extraccion_entidades", nodo_entidades)
grafo.add_node("resumen", nodo_resumen)
grafo.add_node("combinar", nodo_combinar)

grafo.set_entry_point("coordinador")
# Fan-out: el coordinador lanza 3 ramas en paralelo
grafo.add_edge("coordinador", "analisis_sentimiento")
grafo.add_edge("coordinador", "extraccion_entidades")
grafo.add_edge("coordinador", "resumen")
# Fan-in: todas convergen en "combinar"
grafo.add_edge("analisis_sentimiento", "combinar")
grafo.add_edge("extraccion_entidades", "combinar")
grafo.add_edge("resumen", "combinar")
grafo.add_edge("combinar", END)
```

---

## 9. Streaming y observabilidad

### 9.1 Streaming por eventos

```python
config = {"configurable": {"thread_id": "stream_demo"}}

# stream() devuelve eventos de cada nodo según se ejecutan
for evento in app.stream(
    {"messages": [HumanMessage(content="¿Cuánto es 2^32?")]},
    config=config
):
    for nodo, valor in evento.items():
        print(f"\n[{nodo}]")
        if "messages" in valor:
            ultimo = valor["messages"][-1]
            if hasattr(ultimo, "content"):
                print(f"  → {ultimo.content[:200]}")
            if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
                for tc in ultimo.tool_calls:
                    print(f"  🔧 {tc['name']}({tc['args']})")
```

### 9.2 Streaming de tokens

```python
# Para recibir el texto token a token (requiere astream_events)
async def stream_tokens():
    async for evento in app.astream_events(
        {"messages": [HumanMessage(content="Explica qué es un grafo en IA")]},
        config=config,
        version="v2"
    ):
        tipo = evento["event"]
        if tipo == "on_chat_model_stream":
            chunk = evento["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(stream_tokens())
```

---

## 10. Cuándo usar LangGraph vs otras opciones

```
                    ¿Necesitas control explícito del flujo?
                              │
                    ┌─────────┴─────────┐
                   SÍ                   NO
                    │                   │
           ¿Estado persistente?    Cadena LCEL simple
                    │
           ┌────────┴────────┐
          SÍ                 NO
           │                 │
     ¿Human-in-loop     LangGraph sin
     o multi-sesión?    checkpointer
           │
    LangGraph completo
    con checkpointer
```

| Escenario | Recomendación |
|---|---|
| Chatbot conversacional simple | LCEL + historial en memoria |
| Agente ReAct de un solo paso | LangGraph mínimo o bucle while |
| Flujo con rutas condicionales | LangGraph con edges condicionales |
| Agente con memoria entre sesiones | LangGraph + SqliteSaver o PostgresSaver |
| Aprobación humana antes de actuar | LangGraph + `interrupt_before` |
| Múltiples agentes coordinados | LangGraph multi-agente (Supervisor) |
| Paralelismo de subtareas | LangGraph con fan-out/fan-in |
| Producción con trazabilidad | LangGraph + LangSmith |

---

## 11. Resumen

LangGraph convierte el agente de un bucle frágil en un **sistema de flujo explícito y robusto**:

- **`StateGraph`** — el grafo comparte estado tipado entre todos los nodos
- **Nodos** — funciones Python que leen y actualizan el estado
- **Edges condicionales** — enrutamiento dinámico según el estado actual
- **`ToolNode`** — ejecuta herramientas automáticamente cerrando el ciclo ReAct
- **Checkpointers** — persisten el estado entre sesiones (MemorySaver, SqliteSaver)
- **`interrupt_before`** — pausa el grafo para aprobación humana
- **Multi-agente** — patrón Supervisor que coordina subagentes especializados
- **Streaming** — `.stream()` y `.astream_events()` para UIs reactivas

```
   Input
     │
     ▼
[Entry Node] ──► [Nodo A] ──►  ¿condición?  ──sí──► [Nodo B] ──┐
                                    │                            │
                                    └──no──► [Nodo C] ──────────┤
                                                                 │
                                                               [END]
```

### Recursos

- [Documentación oficial LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- Tutorial anterior: [07 — Tipos y Arquitecturas de Agentes](./07-tipos-agentes.md)
- Tutorial siguiente: [09 — Modelos locales con Ollama](./09-modelos-locales-ollama.md)
- Notebook interactivo: [08-langgraph.ipynb](../notebooks/llms/08-langgraph.ipynb)
