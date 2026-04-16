# 07 — LangGraph: Agentes con Estado y Memoria

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. [Por qué LangGraph](#1-por-qué-langgraph)
2. [Conceptos clave: StateGraph, nodos y aristas](#2-conceptos-clave-stategraph-nodos-y-aristas)
3. [Primer grafo: agente ReAct con herramientas](#3-primer-grafo-agente-react-con-herramientas)
4. [Checkpointing: memoria persistente entre sesiones](#4-checkpointing-memoria-persistente-entre-sesiones)
5. [Human-in-the-loop: interrupciones y reanudación](#5-human-in-the-loop-interrupciones-y-reanudación)
6. [Subgrafos y ejecución paralela](#6-subgrafos-y-ejecución-paralela)
7. [Streaming de tokens y eventos del grafo](#7-streaming-de-tokens-y-eventos-del-grafo)
8. [Patrones de producción: errores, reintentos y timeout](#8-patrones-de-producción-errores-reintentos-y-timeout)

---

## 1. Por qué LangGraph

### 1.1 El problema del bucle manual

Un agente ReAct implementado a mano suele ser un bucle `while` con condicionales:

```python
while not done:
    response = llm.invoke(messages)
    if needs_tool(response):
        result = run_tool(response)
        messages.append(result)
    else:
        done = True
```

Este enfoque se rompe en cuanto el flujo deja de ser lineal: pasos paralelos, aprobación humana intermedia, recuperación de errores, persistencia entre sesiones, o grafos donde el camino depende del estado. Todo eso requiere código ad hoc que rápidamente se vuelve inmantenible.

### 1.2 LangChain clásico vs LangGraph

| Característica | LangChain clásico (LCEL) | LangGraph |
|---|---|---|
| Modelo de ejecución | Cadenas lineales o ramas fijas | Grafo dirigido con ciclos |
| Estado | Sin estado nativo | TypedDict compartido entre nodos |
| Persistencia | Manual | Checkpointers integrados |
| Human-in-the-loop | No nativo | `interrupt_before` / `interrupt_after` |
| Ejecución paralela | Limitada (RunnableParallel) | Nodos paralelos nativos |
| Observabilidad | LangSmith | LangSmith + streaming de eventos |

LangGraph no reemplaza LCEL: los nodos del grafo pueden contener chains LCEL. Lo que añade es la capa de **orquestación stateful**.

### 1.3 Cuándo usar LangGraph

- El agente necesita memoria entre invocaciones
- El flujo tiene bifurcaciones condicionales complejas
- Se requiere intervención humana en puntos concretos
- El sistema ejecuta pasos en paralelo y luego los combina
- Se necesita reanudar una ejecución interrumpida desde el checkpoint

---

## 2. Conceptos clave: StateGraph, nodos y aristas

### 2.1 El State: TypedDict compartido

El estado es la única fuente de verdad que circula por el grafo. Todos los nodos lo leen y pueden modificarlo. Se define como un `TypedDict` con `Annotated` para controlar cómo se fusionan los valores cuando varios nodos escriben al mismo campo.

```python
# state_definition.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# add_messages es un reducer: en lugar de sobrescribir la lista de mensajes,
# los nuevos mensajes se añaden al final. Es el comportamiento estándar.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Campos adicionales sin reducer se sobrescriben con cada escritura
    iteracion: int
    resultado_final: str | None
```

### 2.2 Anatomía de un StateGraph

```
                ┌──────────────────────────────────────────┐
                │              StateGraph                  │
                │                                          │
   START ──────►│  nodo_a ──► nodo_b ──► nodo_c ──► END  │
                │               │                          │
                │               └──► nodo_d (condicional) │
                └──────────────────────────────────────────┘
```

- **Nodo**: función Python que recibe el estado y devuelve un dict parcial con los campos actualizados.
- **Arista**: conexión directa entre dos nodos. Siempre se recorre.
- **Arista condicional**: función que inspecciona el estado y devuelve el nombre del siguiente nodo.
- **START / END**: nodos especiales que marcan entrada y salida del grafo.

### 2.3 Construcción y compilación

```python
# graph_basics.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatAnthropic(model="claude-opus-4-5")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    contador: int


def nodo_saludo(state: State) -> dict:
    """Primer nodo: genera un saludo."""
    respuesta = llm.invoke(state["messages"])
    return {
        "messages": [respuesta],
        "contador": state.get("contador", 0) + 1,
    }


def nodo_despedida(state: State) -> dict:
    """Segundo nodo: añade una despedida."""
    ultimo_msg = state["messages"][-1].content
    msg_despedida = AIMessage(content=f"{ultimo_msg}\n\n[Fin de la respuesta — iteración {state['contador']}]")
    return {"messages": [msg_despedida]}


def debe_continuar(state: State) -> str:
    """Arista condicional: si el contador supera 3, terminar."""
    if state["contador"] >= 3:
        return "terminar"
    return "continuar"


# Construir el grafo
builder = StateGraph(State)
builder.add_node("saludo", nodo_saludo)
builder.add_node("despedida", nodo_despedida)

# Aristas
builder.add_edge(START, "saludo")

# Arista condicional desde "saludo"
builder.add_conditional_edges(
    "saludo",
    debe_continuar,
    {
        "continuar": "despedida",
        "terminar": END,
    },
)
builder.add_edge("despedida", END)

# Compilar: devuelve un objeto ejecutable (Runnable)
grafo = builder.compile()

# Ejecución
resultado = grafo.invoke({
    "messages": [HumanMessage(content="Hola, ¿cómo estás?")],
    "contador": 0,
})
print(resultado["messages"][-1].content)
```

---

## 3. Primer grafo: agente ReAct con herramientas

El patrón **ReAct** (Razonar + Actuar) es el más común en agentes: el LLM decide qué herramienta usar, la herramienta se ejecuta, el resultado vuelve al LLM, y el ciclo continúa hasta que el LLM emite una respuesta sin llamadas a herramientas.

LangGraph incluye `ToolNode`, que automatiza la ejecución de herramientas a partir de los `tool_calls` del mensaje del LLM.

```python
# react_agent.py
import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ── Definición de herramientas ───────────────────────────────────────────────

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática Python. Ejemplo: '2 ** 10 + 5'."""
    try:
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return str(resultado)
    except Exception as e:
        return f"Error: {e}"


@tool
def buscar_wikipedia(termino: str) -> str:
    """Devuelve un resumen simulado de Wikipedia sobre el término dado."""
    # En producción, usar la API real de Wikipedia o Tavily
    return (
        f"Información sobre '{termino}': "
        f"Este es un resultado simulado de Wikipedia. "
        f"En producción, conectar a la API de Wikipedia o un motor de búsqueda."
    )


herramientas = [calcular, buscar_wikipedia]


# ── Estado del agente ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Nodos ────────────────────────────────────────────────────────────────────

# Vincular herramientas al modelo para que pueda emitir tool_calls
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(herramientas)


def nodo_agente(state: AgentState) -> dict:
    """El LLM decide qué hacer: responder o usar una herramienta."""
    respuesta = llm.invoke(state["messages"])
    return {"messages": [respuesta]}


# ToolNode lee los tool_calls del último mensaje AIMessage,
# ejecuta las herramientas y devuelve ToolMessages con los resultados
nodo_herramientas = ToolNode(herramientas)


# ── Arista condicional ───────────────────────────────────────────────────────

def hay_tool_calls(state: AgentState) -> str:
    """Comprueba si el último mensaje del LLM incluye tool_calls."""
    ultimo = state["messages"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "usar_herramienta"
    return "terminar"


# ── Construcción del grafo ───────────────────────────────────────────────────

builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", nodo_herramientas)

builder.add_edge(START, "agente")
builder.add_conditional_edges(
    "agente",
    hay_tool_calls,
    {
        "usar_herramienta": "herramientas",
        "terminar": END,
    },
)
# Después de ejecutar herramientas, volvemos al agente (ciclo ReAct)
builder.add_edge("herramientas", "agente")

grafo = builder.compile()


# ── Ejecución ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pregunta = "¿Cuánto es 2 elevado a 32? Y también, ¿qué es la computación cuántica?"
    resultado = grafo.invoke({
        "messages": [HumanMessage(content=pregunta)]
    })
    # El último mensaje es la respuesta final del LLM
    print(resultado["messages"][-1].content)
```

### 3.1 Diagrama del flujo ReAct

```
START
  │
  ▼
┌───────┐   tool_calls    ┌────────────┐
│agente │ ──────────────► │herramientas│
│ (LLM) │ ◄───────────── │ (ToolNode) │
└───────┘  ToolMessages   └────────────┘
  │
  │ sin tool_calls
  ▼
 END
```

---

## 4. Checkpointing: memoria persistente entre sesiones

Por defecto, cada llamada a `grafo.invoke()` es independiente. Para que el agente recuerde conversaciones anteriores se usa un **checkpointer**: serializa el estado completo del grafo después de cada nodo y lo recupera al reanudar.

### 4.1 MemorySaver (en memoria, ideal para desarrollo)

```python
# checkpointing_memory.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def obtener_fecha_actual() -> str:
    """Devuelve la fecha y hora actuales."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


herramientas = [obtener_fecha_actual]
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(herramientas)


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def hay_tool_calls(state: AgentState) -> str:
    ultimo = state["messages"][-1]
    return "usar_herramienta" if getattr(ultimo, "tool_calls", None) else "terminar"


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", ToolNode(herramientas))
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", hay_tool_calls, {
    "usar_herramienta": "herramientas",
    "terminar": END,
})
builder.add_edge("herramientas", "agente")

# Compilar CON checkpointer
checkpointer = MemorySaver()
grafo = builder.compile(checkpointer=checkpointer)


# El thread_id identifica la sesión. Mismo thread_id = misma conversación.
config_sesion_1 = {"configurable": {"thread_id": "usuario_42"}}

# Primera interacción
r1 = grafo.invoke(
    {"messages": [HumanMessage(content="Hola, me llamo Carlos.")]},
    config=config_sesion_1,
)
print(r1["messages"][-1].content)

# Segunda interacción — el grafo recuerda que el usuario se llama Carlos
r2 = grafo.invoke(
    {"messages": [HumanMessage(content="¿Cómo me llamo?")]},
    config=config_sesion_1,
)
print(r2["messages"][-1].content)

# Sesión distinta — no sabe quién es el usuario
config_sesion_2 = {"configurable": {"thread_id": "usuario_99"}}
r3 = grafo.invoke(
    {"messages": [HumanMessage(content="¿Cómo me llamo?")]},
    config=config_sesion_2,
)
print(r3["messages"][-1].content)
```

### 4.2 SqliteSaver (persistente entre reinicios del proceso)

```python
# checkpointing_sqlite.py
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(model="claude-opus-4-5")


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_edge(START, "agente")
builder.add_edge("agente", END)

# SqliteSaver guarda checkpoints en un archivo SQLite.
# La conexión debe ser check_same_thread=False para uso en hilos.
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    grafo = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "sesion_persistente"}}

    # Primera ejecución
    r = grafo.invoke(
        {"messages": [HumanMessage(content="Recuerda: el proyecto se llama ATLAS.")]},
        config=config,
    )
    print(r["messages"][-1].content)

    # Al reiniciar el proceso y usar el mismo thread_id,
    # el historial completo se recupera de checkpoints.db
    r2 = grafo.invoke(
        {"messages": [HumanMessage(content="¿Cómo se llama el proyecto?")]},
        config=config,
    )
    print(r2["messages"][-1].content)

# Para producción usar PostgresSaver (paquete langgraph-checkpoint-postgres)
# from langgraph.checkpoint.postgres import PostgresSaver
```

### 4.3 Inspeccionar el estado guardado

```python
# inspeccionar_checkpoint.py
from langgraph.checkpoint.memory import MemorySaver

# Asumiendo que 'grafo' y 'checkpointer' ya están definidos (ver ejemplo anterior)

config = {"configurable": {"thread_id": "usuario_42"}}

# Estado actual del thread
snapshot = grafo.get_state(config)
print("Nodo actual:", snapshot.next)
print("Mensajes en el estado:", len(snapshot.values["messages"]))

# Historial de checkpoints (ordenados del más reciente al más antiguo)
for checkpoint in grafo.get_state_history(config):
    print(f"Checkpoint en nodo '{checkpoint.next}' — {len(checkpoint.values['messages'])} mensajes")
```

---

## 5. Human-in-the-loop: interrupciones y reanudación

LangGraph permite pausar la ejecución del grafo antes o después de un nodo para que un humano revise o apruebe la acción antes de continuar.

### 5.1 interrupt_before: aprobar antes de ejecutar herramientas

```python
# human_in_the_loop.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def eliminar_registro(id_registro: str) -> str:
    """Elimina un registro de la base de datos por su ID. Acción irreversible."""
    # En producción: lógica real de eliminación
    return f"Registro {id_registro} eliminado correctamente."


@tool
def consultar_registro(id_registro: str) -> str:
    """Consulta un registro de la base de datos por su ID."""
    return f"Registro {id_registro}: {{nombre: 'Test', estado: 'activo'}}"


herramientas = [eliminar_registro, consultar_registro]
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(herramientas)


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def hay_tool_calls(state: AgentState) -> str:
    ultimo = state["messages"][-1]
    return "usar_herramienta" if getattr(ultimo, "tool_calls", None) else "terminar"


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", ToolNode(herramientas))
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", hay_tool_calls, {
    "usar_herramienta": "herramientas",
    "terminar": END,
})
builder.add_edge("herramientas", "agente")

checkpointer = MemorySaver()

# interrupt_before: el grafo se detiene ANTES de ejecutar el nodo "herramientas"
grafo = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["herramientas"],
)

config = {"configurable": {"thread_id": "aprobacion_001"}}

# Primera ejecución: el agente decide usar una herramienta y se pausa
print("=== Primera ejecución ===")
estado = grafo.invoke(
    {"messages": [HumanMessage(content="Elimina el registro con ID=789")]},
    config=config,
)

# El grafo está pausado — podemos inspeccionar qué va a hacer
snapshot = grafo.get_state(config)
print("Grafo pausado. Próximo nodo:", snapshot.next)

ultimo_msg = snapshot.values["messages"][-1]
if hasattr(ultimo_msg, "tool_calls") and ultimo_msg.tool_calls:
    tool_call = ultimo_msg.tool_calls[0]
    print(f"El agente quiere llamar a: {tool_call['name']}")
    print(f"Con argumentos: {tool_call['args']}")

# Simular aprobación humana
aprobado = input("¿Aprobar esta acción? (s/n): ").strip().lower() == "s"

if aprobado:
    # Reanudar la ejecución desde el checkpoint
    # Pasar None como input reanuda sin añadir nuevos mensajes
    resultado = grafo.invoke(None, config=config)
    print("Resultado:", resultado["messages"][-1].content)
else:
    # Rechazar: modificar el estado para cancelar la acción
    grafo.update_state(
        config,
        {"messages": [AIMessage(content="Acción cancelada por el operador.")]},
        as_node="agente",
    )
    print("Acción rechazada.")
```

### 5.2 Command: reanudación con instrucciones adicionales

```python
# command_resume.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def enviar_email(destinatario: str, asunto: str, cuerpo: str) -> str:
    """Envía un email. Requiere aprobación humana antes de ejecutarse."""
    return f"Email enviado a {destinatario} con asunto '{asunto}'."


herramientas = [enviar_email]
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(herramientas)


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def hay_tool_calls(state: AgentState) -> str:
    ultimo = state["messages"][-1]
    return "usar_herramienta" if getattr(ultimo, "tool_calls", None) else "terminar"


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", ToolNode(herramientas))
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", hay_tool_calls, {
    "usar_herramienta": "herramientas",
    "terminar": END,
})
builder.add_edge("herramientas", "agente")

grafo = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["herramientas"],
)

config = {"configurable": {"thread_id": "email_flow"}}

# Ejecutar hasta la interrupción
grafo.invoke(
    {"messages": [HumanMessage(content="Envía un email a ana@empresa.com diciéndole que la reunión es mañana a las 10.")]},
    config=config,
)

snapshot = grafo.get_state(config)
tool_call = snapshot.values["messages"][-1].tool_calls[0]
print(f"Herramienta a ejecutar: {tool_call['name']}")
print(f"Argumentos: {tool_call['args']}")

# Opción 1: Reanudar normalmente
# grafo.invoke(None, config=config)

# Opción 2: Reanudar con un ToolMessage personalizado que sobreescribe el resultado
# Útil cuando el humano quiere modificar los argumentos sin rechazar la acción
resultado_manual = ToolMessage(
    tool_call_id=tool_call["id"],
    content="Email enviado a ana@empresa.com (asunto modificado por el operador: 'Reunión — CONFIRMADA para mañana 10:00').",
)

# update_state inyecta el ToolMessage directamente en el historial
grafo.update_state(config, {"messages": [resultado_manual]}, as_node="herramientas")

# Reanudar desde "agente" (después de haber inyectado el resultado de la herramienta)
resultado_final = grafo.invoke(None, config=config)
print(resultado_final["messages"][-1].content)
```

---

## 6. Subgrafos y ejecución paralela

### 6.1 Nodos paralelos con Send

LangGraph permite ramificar la ejecución en paralelo usando la primitiva `Send`. Cada `Send` lanza una ejecución independiente de un nodo con un estado distinto.

```python
# paralelismo.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send


llm = ChatAnthropic(model="claude-opus-4-5")


# Estado del grafo principal
class EstadoPrincipal(TypedDict):
    documento: str
    analisis: Annotated[list[str], lambda a, b: a + b]  # reducer: acumula resultados


# Estado de cada rama paralela
class EstadoRama(TypedDict):
    documento: str
    perspectiva: str


def nodo_analisis_perspectiva(state: EstadoRama) -> dict:
    """Analiza el documento desde una perspectiva específica."""
    respuesta = llm.invoke([
        SystemMessage(content=f"Analiza el siguiente texto exclusivamente desde una perspectiva {state['perspectiva']}. Sé conciso (máx. 100 palabras)."),
        HumanMessage(content=state["documento"]),
    ])
    return {"analisis": [f"[{state['perspectiva'].upper()}]: {respuesta.content}"]}


def distribuir_perspectivas(state: EstadoPrincipal):
    """Genera un Send por cada perspectiva de análisis, en paralelo."""
    perspectivas = ["legal", "financiera", "técnica", "riesgo"]
    return [
        Send("analisis_perspectiva", {
            "documento": state["documento"],
            "perspectiva": p,
        })
        for p in perspectivas
    ]


def sintetizar(state: EstadoPrincipal) -> dict:
    """Combina los análisis paralelos en una síntesis final."""
    todos = "\n\n".join(state["analisis"])
    respuesta = llm.invoke([
        SystemMessage(content="Sintetiza los siguientes análisis en un párrafo ejecutivo."),
        HumanMessage(content=todos),
    ])
    return {"analisis": [f"[SÍNTESIS]: {respuesta.content}"]}


builder = StateGraph(EstadoPrincipal)
builder.add_node("analisis_perspectiva", nodo_analisis_perspectiva)
builder.add_node("sintetizar", sintetizar)

# Desde START, distribuir en paralelo usando Send
builder.add_conditional_edges(START, distribuir_perspectivas, ["analisis_perspectiva"])
# Cuando todas las ramas terminan, confluyen en "sintetizar"
builder.add_edge("analisis_perspectiva", "sintetizar")
builder.add_edge("sintetizar", END)

grafo = builder.compile()

resultado = grafo.invoke({
    "documento": "El contrato establece un plazo de entrega de 30 días con penalización del 2% diario por retraso, limitada al 20% del valor total. El importe es de 500.000€.",
    "analisis": [],
})

for linea in resultado["analisis"]:
    print(linea)
    print()
```

### 6.2 Subgrafos

Un subgrafo es un grafo compilado que se usa como nodo dentro de otro grafo. Permite encapsular lógica compleja y reutilizarla.

```python
# subgrafos.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


llm = ChatAnthropic(model="claude-opus-4-5")


# ── Subgrafo: validación de documento ────────────────────────────────────────

class EstadoValidacion(TypedDict):
    texto: str
    errores: list[str]
    aprobado: bool


def verificar_longitud(state: EstadoValidacion) -> dict:
    errores = list(state.get("errores", []))
    if len(state["texto"]) < 50:
        errores.append("El texto es demasiado corto (mínimo 50 caracteres).")
    return {"errores": errores}


def verificar_contenido(state: EstadoValidacion) -> dict:
    errores = list(state.get("errores", []))
    palabras_prohibidas = ["urgente", "inmediato", "gratis"]
    encontradas = [p for p in palabras_prohibidas if p in state["texto"].lower()]
    if encontradas:
        errores.append(f"Palabras no permitidas: {encontradas}")
    return {"errores": errores}


def determinar_aprobacion(state: EstadoValidacion) -> dict:
    return {"aprobado": len(state.get("errores", [])) == 0}


builder_validacion = StateGraph(EstadoValidacion)
builder_validacion.add_node("longitud", verificar_longitud)
builder_validacion.add_node("contenido", verificar_contenido)
builder_validacion.add_node("aprobacion", determinar_aprobacion)
builder_validacion.add_edge(START, "longitud")
builder_validacion.add_edge("longitud", "contenido")
builder_validacion.add_edge("contenido", "aprobacion")
builder_validacion.add_edge("aprobacion", END)

subgrafo_validacion = builder_validacion.compile()


# ── Grafo principal ──────────────────────────────────────────────────────────

class EstadoPrincipal(TypedDict):
    messages: Annotated[list, add_messages]
    texto_a_procesar: str
    validacion_aprobada: bool
    errores_validacion: list[str]


def nodo_extraer_texto(state: EstadoPrincipal) -> dict:
    """El LLM extrae el texto a procesar del mensaje del usuario."""
    respuesta = llm.invoke([
        SystemMessage(content="Extrae y devuelve únicamente el texto que el usuario quiere procesar, sin explicaciones adicionales."),
        *state["messages"],
    ])
    return {
        "texto_a_procesar": respuesta.content,
        "messages": [respuesta],
    }


def nodo_validar(state: EstadoPrincipal) -> dict:
    """Invoca el subgrafo de validación."""
    resultado = subgrafo_validacion.invoke({
        "texto": state["texto_a_procesar"],
        "errores": [],
        "aprobado": False,
    })
    return {
        "validacion_aprobada": resultado["aprobado"],
        "errores_validacion": resultado["errores"],
    }


def nodo_procesar(state: EstadoPrincipal) -> dict:
    """Procesa el texto si pasó la validación."""
    respuesta = llm.invoke([
        SystemMessage(content="Resume el siguiente texto en una oración."),
        HumanMessage(content=state["texto_a_procesar"]),
    ])
    return {"messages": [respuesta]}


def nodo_rechazar(state: EstadoPrincipal) -> dict:
    from langchain_core.messages import AIMessage
    errores = "; ".join(state["errores_validacion"])
    return {"messages": [AIMessage(content=f"El texto no pasó la validación: {errores}")]}


def ruta_post_validacion(state: EstadoPrincipal) -> str:
    return "procesar" if state["validacion_aprobada"] else "rechazar"


builder = StateGraph(EstadoPrincipal)
builder.add_node("extraer", nodo_extraer_texto)
builder.add_node("validar", nodo_validar)
builder.add_node("procesar", nodo_procesar)
builder.add_node("rechazar", nodo_rechazar)

builder.add_edge(START, "extraer")
builder.add_edge("extraer", "validar")
builder.add_conditional_edges("validar", ruta_post_validacion, {
    "procesar": "procesar",
    "rechazar": "rechazar",
})
builder.add_edge("procesar", END)
builder.add_edge("rechazar", END)

grafo = builder.compile()

resultado = grafo.invoke({
    "messages": [HumanMessage(content="Procesa este texto: 'La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que requieren inteligencia humana.'")],
    "texto_a_procesar": "",
    "validacion_aprobada": False,
    "errores_validacion": [],
})
print(resultado["messages"][-1].content)
```

---

## 7. Streaming de tokens y eventos del grafo

LangGraph expone tres modos de streaming. Se pueden combinar.

| Modo | Qué devuelve |
|---|---|
| `"values"` | Estado completo del grafo después de cada nodo |
| `"updates"` | Solo el dict de cambios que cada nodo devolvió |
| `"messages"` | Tokens individuales del LLM a medida que se generan |

```python
# streaming.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática."""
    try:
        return str(eval(expresion, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


herramientas = [calcular]
llm = ChatAnthropic(model="claude-opus-4-5").bind_tools(herramientas)


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def hay_tool_calls(state: AgentState) -> str:
    ultimo = state["messages"][-1]
    return "usar_herramienta" if getattr(ultimo, "tool_calls", None) else "terminar"


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_node("herramientas", ToolNode(herramientas))
builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", hay_tool_calls, {
    "usar_herramienta": "herramientas",
    "terminar": END,
})
builder.add_edge("herramientas", "agente")

grafo = builder.compile()

entrada = {"messages": [HumanMessage(content="¿Cuánto es 123456789 * 987654321? Muéstrame el proceso.")]}


# ── Modo 1: stream de updates (por nodo) ────────────────────────────────────
print("=== UPDATES (por nodo) ===")
for chunk in grafo.stream(entrada, stream_mode="updates"):
    for nodo, cambios in chunk.items():
        print(f"[{nodo}] devolvió {len(cambios.get('messages', []))} mensaje(s)")


# ── Modo 2: stream de tokens (nivel de carácter) ─────────────────────────────
print("\n=== TOKENS (streaming en tiempo real) ===")
for chunk, metadata in grafo.stream(entrada, stream_mode="messages"):
    # chunk es un AIMessageChunk cuando viene del LLM
    from langchain_core.messages import AIMessageChunk
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
print()


# ── Modo 3: stream de eventos (más detallado) ────────────────────────────────
print("\n=== EVENTOS DEL GRAFO ===")
for evento in grafo.stream(entrada, stream_mode="values"):
    # Cada evento es el estado completo tras cada nodo
    ultimo = evento["messages"][-1]
    print(f"Tipo mensaje: {type(ultimo).__name__} | Contenido: {str(ultimo.content)[:80]}")


# ── Streaming asíncrono ──────────────────────────────────────────────────────
import asyncio

async def stream_asincrono():
    print("\n=== STREAMING ASÍNCRONO ===")
    async for chunk, metadata in grafo.astream(entrada, stream_mode="messages"):
        from langchain_core.messages import AIMessageChunk
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_asincrono())
```

---

## 8. Patrones de producción: errores, reintentos y timeout

### 8.1 Manejo de errores en nodos

```python
# error_handling.py
import time
import random
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intentos: int
    max_intentos: int
    error_actual: str | None


llm = ChatAnthropic(model="claude-opus-4-5")


@tool
def servicio_externo(consulta: str) -> str:
    """Simula un servicio externo que puede fallar."""
    if random.random() < 0.5:
        raise ConnectionError("Servicio no disponible temporalmente.")
    return f"Resultado del servicio para: {consulta}"


def nodo_agente_seguro(state: AgentState) -> dict:
    """Nodo del agente con manejo de errores explícito."""
    try:
        llm_con_herramientas = llm.bind_tools([servicio_externo])
        respuesta = llm_con_herramientas.invoke(state["messages"])
        return {
            "messages": [respuesta],
            "error_actual": None,
        }
    except Exception as e:
        # En lugar de propagar la excepción y romper el grafo,
        # registramos el error en el estado y dejamos que el grafo decida
        return {
            "messages": [AIMessage(content=f"Error al invocar el LLM: {e}")],
            "error_actual": str(e),
            "intentos": state.get("intentos", 0) + 1,
        }


def nodo_ejecutar_herramientas(state: AgentState) -> dict:
    """Ejecuta herramientas con reintentos y backoff exponencial."""
    ultimo = state["messages"][-1]
    if not (hasattr(ultimo, "tool_calls") and ultimo.tool_calls):
        return {}

    resultados = []
    for tool_call in ultimo.tool_calls:
        for intento in range(3):  # máximo 3 reintentos por herramienta
            try:
                if tool_call["name"] == "servicio_externo":
                    resultado = servicio_externo.invoke(tool_call["args"])
                    from langchain_core.messages import ToolMessage
                    resultados.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(resultado),
                    ))
                    break  # éxito, salir del bucle de reintentos
            except Exception as e:
                if intento < 2:
                    espera = 2 ** intento  # backoff: 1s, 2s, 4s
                    print(f"Reintento {intento + 1}/3 en {espera}s para {tool_call['name']}: {e}")
                    time.sleep(espera)
                else:
                    # Agotar reintentos: devolver error como ToolMessage
                    from langchain_core.messages import ToolMessage
                    resultados.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=f"Error tras 3 intentos: {e}",
                    ))

    return {
        "messages": resultados,
        "intentos": state.get("intentos", 0) + 1,
    }


def ruta_con_error(state: AgentState) -> str:
    """Gestiona el flujo considerando errores y límite de intentos."""
    ultimo = state["messages"][-1]
    intentos = state.get("intentos", 0)
    max_intentos = state.get("max_intentos", 5)

    if intentos >= max_intentos:
        return "agotar"

    if state.get("error_actual"):
        return "recuperar_error"

    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "herramientas"

    return "terminar"


def nodo_recuperar_error(state: AgentState) -> dict:
    """Intenta recuperarse de un error con una estrategia alternativa."""
    respuesta = llm.invoke([
        SystemMessage(content="Ha ocurrido un error. Intenta responder la pregunta sin usar herramientas externas."),
        *state["messages"],
    ])
    return {
        "messages": [respuesta],
        "error_actual": None,
    }


def nodo_agotar_intentos(state: AgentState) -> dict:
    """Se activa cuando se supera el máximo de intentos."""
    return {
        "messages": [AIMessage(content=(
            "No he podido completar la tarea después de varios intentos. "
            "Por favor, inténtelo más tarde o contacte con soporte."
        ))],
    }


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente_seguro)
builder.add_node("herramientas", nodo_ejecutar_herramientas)
builder.add_node("recuperar_error", nodo_recuperar_error)
builder.add_node("agotar", nodo_agotar_intentos)

builder.add_edge(START, "agente")
builder.add_conditional_edges("agente", ruta_con_error, {
    "herramientas": "herramientas",
    "recuperar_error": "recuperar_error",
    "agotar": "agotar",
    "terminar": END,
})
builder.add_edge("herramientas", "agente")
builder.add_edge("recuperar_error", "agente")
builder.add_edge("agotar", END)

grafo = builder.compile()
```

### 8.2 Timeout con asyncio

```python
# timeout.py
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatAnthropic(model="claude-opus-4-5")


def nodo_agente(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(AgentState)
builder.add_node("agente", nodo_agente)
builder.add_edge(START, "agente")
builder.add_edge("agente", END)
grafo = builder.compile()


async def invocar_con_timeout(entrada: dict, timeout_segundos: float = 30.0) -> dict:
    """Invoca el grafo con un timeout máximo. Lanza asyncio.TimeoutError si se supera."""
    try:
        resultado = await asyncio.wait_for(
            grafo.ainvoke(entrada),
            timeout=timeout_segundos,
        )
        return resultado
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"El grafo no completó la ejecución en {timeout_segundos}s. "
            f"Considera reducir la complejidad de la tarea o aumentar el timeout."
        )


async def main():
    try:
        resultado = await invocar_con_timeout(
            {"messages": [HumanMessage(content="Resume la historia de la inteligencia artificial.")]},
            timeout_segundos=30.0,
        )
        print(resultado["messages"][-1].content)
    except TimeoutError as e:
        print(f"Timeout: {e}")


asyncio.run(main())
```

### 8.3 Checklist de producción

```
Antes de desplegar un grafo LangGraph en producción:

Persistencia
  [ ] Usar SqliteSaver o PostgresSaver (nunca MemorySaver en prod)
  [ ] Definir política de retención de checkpoints
  [ ] Probar recuperación ante reinicios del proceso

Errores
  [ ] Todos los nodos con acceso a servicios externos tienen try/except
  [ ] El estado incluye contadores de intentos y campos de error
  [ ] El grafo tiene un nodo de fallback cuando se agotan los reintentos

Observabilidad
  [ ] LangSmith configurado (LANGCHAIN_TRACING_V2=true)
  [ ] Logs estructurados en cada nodo con el thread_id
  [ ] Métricas de latencia por nodo

Seguridad
  [ ] Validar entradas antes de pasarlas a herramientas con eval()
  [ ] Human-in-the-loop activo para acciones destructivas o irreversibles
  [ ] Límites de iteraciones para evitar bucles infinitos

Rendimiento
  [ ] Usar ainvoke/astream para cargas concurrentes
  [ ] Paralelizar nodos independientes con Send
  [ ] Limitar el historial de mensajes si la ventana de contexto es un cuello de botella
```

---

**Anterior:** [06 — A2A Protocol](./06-a2a-protocol.md) · **Siguiente tutorial:** [08 — Evaluación de Agentes](./08-evaluacion-agentes.md)
