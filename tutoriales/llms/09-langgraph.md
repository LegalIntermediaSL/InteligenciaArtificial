# 09 — LangGraph: Flujos de agentes con estado

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Por qué LangGraph](#1-por-qué-langgraph)
2. [Conceptos clave](#2-conceptos-clave)
3. [Instalación](#3-instalación)
4. [Primer grafo: chatbot con estado](#4-primer-grafo-chatbot-con-estado)
5. [Aristas condicionales](#5-aristas-condicionales)
6. [Persistencia entre sesiones](#6-persistencia-entre-sesiones)
7. [Caso práctico: agente de análisis con ciclo de revisión](#7-caso-práctico-agente-de-análisis-con-ciclo-de-revisión)
8. [Human-in-the-loop](#8-human-in-the-loop)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Por qué LangGraph

Los bucles agénticos simples tienen limitaciones estructurales que se vuelven críticas en producción.

**El bucle agéntico básico (sin LangGraph):**

```python
# Patrón típico "manual" — frágil y sin control
historial = []
while True:
    respuesta = llm(historial)
    if respuesta.stop_reason == "end_turn":
        break
    # ¿Qué pasa si el agente entra en un bucle infinito?
    # ¿Cómo pausar para aprobación humana?
    # ¿Cómo retomar la conversación mañana?
    # ¿Cómo ir a una rama diferente según el resultado?
    # → No hay respuesta elegante sin construirlo todo desde cero
```

**Problemas concretos:**

| Problema | Sin LangGraph | Con LangGraph |
|---------|--------------|--------------|
| Bucles infinitos | Control manual con `max_steps` | Límites declarativos por grafo |
| Flujos ramificados | Condicionales anidados difíciles de mantener | Aristas condicionales explícitas |
| Estado persistente | Serialización manual | `MemorySaver` / `SqliteSaver` |
| Aprobación humana | Interrumpir el proceso a mano | `interrupt_before` nativo |
| Observabilidad | Logging manual | Trazas estructuradas por nodo |
| Tests | Difícil aislar nodos | Cada nodo es una función testeable |

LangGraph representa el flujo del agente como un **grafo dirigido con estado compartido**, lo que convierte decisiones de control de flujo implícitas en estructura explícita.

---

## 2. Conceptos clave

**Grafo, nodos, aristas y estado:**

```
Estado compartido (TypedDict)
┌─────────────────────────────────────┐
│  mensajes: list                     │
│  siguiente: str                     │
│  iteraciones: int                   │
└─────────────────────────────────────┘
            │
    ┌───────▼───────┐
    │   Nodo A      │  ← función Python que lee/escribe el estado
    │  (procesar)   │
    └───────┬───────┘
            │ arista simple
    ┌───────▼───────┐
    │   Nodo B      │
    │  (clasificar) │
    └───────┬───────┘
            │ arista condicional
       ┌────┴────┐
       ▼         ▼
  [Nodo C]   [Nodo D]    ← la función de enrutamiento decide el camino
  (rama A)   (rama B)
       │         │
       └────┬────┘
            ▼
           END
```

**Glosario:**

- **Estado**: un `TypedDict` que todos los nodos comparten y pueden modificar.
- **Nodo**: función Python `(estado) → actualización_parcial_del_estado`.
- **Arista simple**: conexión directa de nodo A a nodo B.
- **Arista condicional**: una función decide a qué nodo ir basándose en el estado.
- **`START`**: punto de entrada del grafo.
- **`END`**: señal de terminación.
- **Checkpointer**: componente que persiste el estado entre ejecuciones.

---

## 3. Instalación

```bash
pip install langgraph langchain-anthropic
```

Verificar la instalación:

```python
import langgraph
import langchain_anthropic
print("LangGraph:", langgraph.__version__)
```

Variables de entorno necesarias:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 4. Primer grafo: chatbot con estado

Un chatbot básico donde el estado es el historial de mensajes.

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END


# --- 1. Definir el estado ---
# Annotated[list, operator.add] indica que las actualizaciones
# se AÑADEN a la lista en lugar de reemplazarla

class EstadoChatbot(TypedDict):
    mensajes: Annotated[list[BaseMessage], operator.add]


# --- 2. Inicializar el modelo ---

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=1024,
)


# --- 3. Definir los nodos ---
# Cada nodo recibe el estado completo y devuelve una actualización parcial

def nodo_chatbot(estado: EstadoChatbot) -> dict:
    """Nodo principal: envía los mensajes al LLM y devuelve la respuesta."""
    respuesta = llm.invoke(estado["mensajes"])
    return {"mensajes": [respuesta]}  # operator.add añade al historial


# --- 4. Construir el grafo ---

builder = StateGraph(EstadoChatbot)

builder.add_node("chatbot", nodo_chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

grafo = builder.compile()


# --- 5. Ejecutar ---

resultado = grafo.invoke({
    "mensajes": [HumanMessage(content="Hola, ¿qué es LangGraph en una frase?")]
})

print(resultado["mensajes"][-1].content)


# --- 6. Conversación multi-turno ---

historial = []

for turno in ["¿Qué es LangGraph?", "¿Y para qué sirven las aristas condicionales?"]:
    historial.append(HumanMessage(content=turno))
    resultado = grafo.invoke({"mensajes": historial})
    respuesta = resultado["mensajes"][-1]
    historial.append(respuesta)
    print(f"Usuario: {turno}")
    print(f"Asistente: {respuesta.content}\n")
```

---

## 5. Aristas condicionales

Las aristas condicionales permiten que el grafo tome decisiones de flujo basándose en el estado actual.

```python
from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import json


# --- Estado con campo de intención ---

class EstadoRouter(TypedDict):
    mensajes: Annotated[list[BaseMessage], operator.add]
    intencion: str          # "codigo", "redaccion", "matematicas", "general"
    respuesta_final: str


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)


# --- Nodo clasificador ---

def clasificar_intencion(estado: EstadoRouter) -> dict:
    """Determina qué tipo de ayuda necesita el usuario."""
    ultimo_mensaje = estado["mensajes"][-1].content

    r = llm.invoke([
        SystemMessage(content="""Clasifica la intención del usuario en UNA de estas categorías:
- codigo: preguntas sobre programación, debugging, algoritmos
- redaccion: escritura, corrección de texto, emails
- matematicas: cálculos, ecuaciones, estadística
- general: cualquier otra cosa

Responde SOLO con el JSON: {"intencion": "categoria"}"""),
        HumanMessage(content=ultimo_mensaje),
    ])

    try:
        datos = json.loads(r.content)
        intencion = datos.get("intencion", "general")
    except (json.JSONDecodeError, AttributeError):
        intencion = "general"

    print(f"[Router] Intención detectada: {intencion}")
    return {"intencion": intencion}


# --- Nodos especializados ---

def especialista_codigo(estado: EstadoRouter) -> dict:
    """Responde a preguntas de programación con ejemplos de código."""
    r = llm.invoke([
        SystemMessage(content="Eres un experto en programación. Incluye siempre ejemplos de código funcional."),
        *estado["mensajes"],
    ])
    return {"respuesta_final": r.content}


def especialista_redaccion(estado: EstadoRouter) -> dict:
    """Ayuda con escritura y corrección de texto."""
    r = llm.invoke([
        SystemMessage(content="Eres un experto en comunicación escrita. Prioriza claridad y concisión."),
        *estado["mensajes"],
    ])
    return {"respuesta_final": r.content}


def especialista_matematicas(estado: EstadoRouter) -> dict:
    """Resuelve problemas matemáticos paso a paso."""
    r = llm.invoke([
        SystemMessage(content="Eres un profesor de matemáticas. Explica cada paso del razonamiento."),
        *estado["mensajes"],
    ])
    return {"respuesta_final": r.content}


def asistente_general(estado: EstadoRouter) -> dict:
    """Respuesta de propósito general."""
    r = llm.invoke([
        SystemMessage(content="Eres un asistente útil y conciso."),
        *estado["mensajes"],
    ])
    return {"respuesta_final": r.content}


# --- Función de enrutamiento ---

def enrutar_segun_intencion(
    estado: EstadoRouter,
) -> Literal["especialista_codigo", "especialista_redaccion", "especialista_matematicas", "asistente_general"]:
    """Devuelve el nombre del nodo al que ir según la intención detectada."""
    rutas = {
        "codigo": "especialista_codigo",
        "redaccion": "especialista_redaccion",
        "matematicas": "especialista_matematicas",
        "general": "asistente_general",
    }
    return rutas.get(estado["intencion"], "asistente_general")


# --- Construir el grafo con aristas condicionales ---

builder = StateGraph(EstadoRouter)

builder.add_node("clasificar", clasificar_intencion)
builder.add_node("especialista_codigo", especialista_codigo)
builder.add_node("especialista_redaccion", especialista_redaccion)
builder.add_node("especialista_matematicas", especialista_matematicas)
builder.add_node("asistente_general", asistente_general)

builder.add_edge(START, "clasificar")

# Arista condicional: después de clasificar, enrutar al especialista correcto
builder.add_conditional_edges(
    "clasificar",
    enrutar_segun_intencion,
    {
        "especialista_codigo": "especialista_codigo",
        "especialista_redaccion": "especialista_redaccion",
        "especialista_matematicas": "especialista_matematicas",
        "asistente_general": "asistente_general",
    },
)

# Todos los especialistas terminan en END
for nodo in ["especialista_codigo", "especialista_redaccion", "especialista_matematicas", "asistente_general"]:
    builder.add_edge(nodo, END)

grafo_router = builder.compile()


# --- Probar el router ---

preguntas = [
    "¿Cómo implemento un árbol binario en Python?",
    "Corrígeme este email para que suene más profesional: 'necesito el informe ya'",
    "¿Cuál es la integral de x^2?",
    "¿Cuál es la capital de Francia?",
]

for pregunta in preguntas:
    resultado = grafo_router.invoke({
        "mensajes": [HumanMessage(content=pregunta)],
        "intencion": "",
        "respuesta_final": "",
    })
    print(f"Pregunta: {pregunta[:50]}...")
    print(f"Especialista: {resultado['intencion']}")
    print(f"Respuesta: {resultado['respuesta_final'][:100]}...\n")
```

---

## 6. Persistencia entre sesiones

`MemorySaver` guarda el estado del grafo en memoria. Al reanudar con el mismo `thread_id`, el grafo recuerda la conversación completa.

```python
from typing import Annotated
import operator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class EstadoConMemoria(TypedDict):
    mensajes: Annotated[list[BaseMessage], operator.add]


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)


def chatear(estado: EstadoConMemoria) -> dict:
    respuesta = llm.invoke([
        SystemMessage(content="Eres un asistente con memoria de conversación. Eres conciso."),
        *estado["mensajes"],
    ])
    return {"mensajes": [respuesta]}


# MemorySaver persiste el estado por thread_id
checkpointer = MemorySaver()

builder = StateGraph(EstadoConMemoria)
builder.add_node("chatear", chatear)
builder.add_edge(START, "chatear")
builder.add_edge("chatear", END)

# Compilar con el checkpointer
grafo_persistente = builder.compile(checkpointer=checkpointer)


# --- Sesión 1: primera conversación ---

config_usuario_1 = {"configurable": {"thread_id": "usuario_42"}}

print("=== Sesión 1 ===")

r1 = grafo_persistente.invoke(
    {"mensajes": [HumanMessage(content="Hola, me llamo Carlos y trabajo en fintech.")]},
    config=config_usuario_1,
)
print("Asistente:", r1["mensajes"][-1].content)

r2 = grafo_persistente.invoke(
    {"mensajes": [HumanMessage(content="¿Cuál es mi nombre?")]},
    config=config_usuario_1,
)
print("Asistente:", r2["mensajes"][-1].content)
# → Recuerda que se llama Carlos


# --- Sesión 2: reanudar la conversación (mismo thread_id) ---

print("\n=== Sesión 2 (días después) ===")

r3 = grafo_persistente.invoke(
    {"mensajes": [HumanMessage(content="¿En qué sector trabajo?")]},
    config=config_usuario_1,
)
print("Asistente:", r3["mensajes"][-1].content)
# → Recuerda fintech aunque fue otra "sesión"


# --- Usuario diferente (thread_id distinto, sin memoria compartida) ---

config_usuario_2 = {"configurable": {"thread_id": "usuario_99"}}

r4 = grafo_persistente.invoke(
    {"mensajes": [HumanMessage(content="¿Cuál es mi nombre?")]},
    config=config_usuario_2,
)
print("\n=== Usuario diferente ===")
print("Asistente:", r4["mensajes"][-1].content)
# → No sabe el nombre, es un hilo diferente


# --- Inspeccionar el estado guardado ---

estado_guardado = grafo_persistente.get_state(config_usuario_1)
print(f"\nMensajes guardados para usuario_42: {len(estado_guardado.values['mensajes'])}")


# --- Persistencia en disco con SqliteSaver ---
# Para producción, usar SqliteSaver en lugar de MemorySaver

# from langgraph.checkpoint.sqlite import SqliteSaver
# checkpointer_disco = SqliteSaver.from_conn_string("conversaciones.db")
# grafo_disco = builder.compile(checkpointer=checkpointer_disco)
```

---

## 7. Caso práctico: agente de análisis con ciclo de revisión

Un agente que analiza un texto, lo somete a revisión y repite hasta que la calidad es suficiente (máximo 3 iteraciones).

```
                    ┌─────────────────────────────────────┐
                    │          GRAFO DE ANÁLISIS          │
                    └─────────────────────────────────────┘

                         START
                           │
                    ┌──────▼──────┐
                    │  analizar   │  ← genera análisis del texto
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   revisar   │  ← evalúa la calidad (0-10)
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────────┐
                    │   ¿aprobado? (puntuacion≥7) │
                    └──────┬───────────┬──────────┘
                  SÍ       │           │  NO (y iter < 3)
                    ┌──────▼──────┐   │
                    │   formato   │   │  ← formatea la salida final
                    └──────┬──────┘   │
                           │          │
                          END    ┌────▼────┐
                                 │ analizar│  ← vuelve a analizar con feedback
                                 └─────────┘
```

```python
from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import json


# --- Estado ---

class EstadoAnalisis(TypedDict):
    texto_original: str
    analisis_actual: str
    puntuacion: float
    feedback: str
    iteraciones: int
    resultado_final: str


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048)
MAX_ITERACIONES = 3


# --- Nodos ---

def nodo_analizar(estado: EstadoAnalisis) -> dict:
    """Genera o mejora el análisis del texto."""
    iteracion = estado.get("iteraciones", 0) + 1
    print(f"[Analizar] Iteración {iteracion}/{MAX_ITERACIONES}")

    prompt_base = f"Analiza el siguiente texto en profundidad:\n\n{estado['texto_original']}"

    # En iteraciones posteriores, incorporar el feedback del revisor
    if estado.get("feedback") and iteracion > 1:
        prompt_base += f"\n\nAnálisis anterior:\n{estado['analisis_actual']}\n\nFeedback del revisor (incorpóralo):\n{estado['feedback']}"

    r = llm.invoke([
        SystemMessage(content="""Eres un analista experto. Tu análisis debe incluir:
1. Tema principal e ideas secundarias
2. Tono y estilo del autor
3. Puntos fuertes y débiles del argumento
4. Conclusión con tu valoración"""),
        HumanMessage(content=prompt_base),
    ])

    return {
        "analisis_actual": r.content,
        "iteraciones": iteracion,
    }


def nodo_revisar(estado: EstadoAnalisis) -> dict:
    """Evalúa la calidad del análisis con una puntuación del 0 al 10."""
    r = llm.invoke([
        SystemMessage(content="""Eres un revisor de calidad. Evalúa el análisis y devuelve SOLO JSON:
{
  "puntuacion": <número entre 0 y 10>,
  "feedback": "<instrucciones específicas de mejora si puntuacion < 7, o vacío si es suficiente>",
  "justificacion": "<una frase explicando la puntuación>"
}"""),
        HumanMessage(content=f"Texto original:\n{estado['texto_original']}\n\nAnálisis a revisar:\n{estado['analisis_actual']}"),
    ])

    try:
        datos = json.loads(r.content)
        puntuacion = float(datos.get("puntuacion", 5.0))
        feedback = datos.get("feedback", "")
        justificacion = datos.get("justificacion", "")
    except (json.JSONDecodeError, ValueError):
        puntuacion = 5.0
        feedback = "Mejorar profundidad y estructura."
        justificacion = "Error al parsear la evaluación."

    print(f"[Revisar] Puntuación: {puntuacion}/10 — {justificacion}")
    return {"puntuacion": puntuacion, "feedback": feedback}


def nodo_formato(estado: EstadoAnalisis) -> dict:
    """Formatea el análisis aprobado para la salida final."""
    resultado = f"""# Análisis aprobado (puntuación: {estado['puntuacion']}/10)

{estado['analisis_actual']}

---
*Generado en {estado['iteraciones']} iteración(es)*"""
    return {"resultado_final": resultado}


# --- Función de enrutamiento ---

def decidir_siguiente(
    estado: EstadoAnalisis,
) -> Literal["nodo_formato", "nodo_analizar"]:
    """Decide si aprobar el análisis o pedir una nueva iteración."""
    aprobado = estado["puntuacion"] >= 7.0
    limite_alcanzado = estado["iteraciones"] >= MAX_ITERACIONES

    if aprobado or limite_alcanzado:
        if limite_alcanzado and not aprobado:
            print(f"[Router] Límite de {MAX_ITERACIONES} iteraciones alcanzado. Finalizando con puntuación {estado['puntuacion']}/10.")
        else:
            print(f"[Router] Análisis aprobado con {estado['puntuacion']}/10.")
        return "nodo_formato"
    else:
        print(f"[Router] Puntuación insuficiente ({estado['puntuacion']}/10). Solicitando mejora.")
        return "nodo_analizar"


# --- Construir el grafo ---

builder = StateGraph(EstadoAnalisis)

builder.add_node("nodo_analizar", nodo_analizar)
builder.add_node("nodo_revisar", nodo_revisar)
builder.add_node("nodo_formato", nodo_formato)

builder.add_edge(START, "nodo_analizar")
builder.add_edge("nodo_analizar", "nodo_revisar")
builder.add_conditional_edges(
    "nodo_revisar",
    decidir_siguiente,
    {
        "nodo_formato": "nodo_formato",
        "nodo_analizar": "nodo_analizar",
    },
)
builder.add_edge("nodo_formato", END)

grafo_analisis = builder.compile()


# --- Ejecutar ---

texto_prueba = """
La inteligencia artificial generativa está transformando la economía global.
Las empresas que adopten estas tecnologías pronto tendrán ventajas competitivas significativas.
Sin embargo, la regulación aún está rezagada respecto a la velocidad de adopción,
lo que crea riesgos tanto para trabajadores como para consumidores.
"""

resultado = grafo_analisis.invoke({
    "texto_original": texto_prueba,
    "analisis_actual": "",
    "puntuacion": 0.0,
    "feedback": "",
    "iteraciones": 0,
    "resultado_final": "",
})

print("\n" + resultado["resultado_final"])
```

---

## 8. Human-in-the-loop

`interrupt_before` pausa el grafo justo antes de ejecutar un nodo crítico, permitiendo que un humano revise el estado y decida si continuar o modificar el plan.

```python
from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import json


class EstadoAprobacion(TypedDict):
    mensajes: Annotated[list[BaseMessage], operator.add]
    plan_accion: str          # Plan generado por la IA
    aprobado_por_humano: bool
    resultado_ejecucion: str


llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)


# --- Nodos ---

def generar_plan(estado: EstadoAprobacion) -> dict:
    """Genera un plan de acción basado en los mensajes del usuario."""
    r = llm.invoke([
        SystemMessage(content="""Genera un plan de acción concreto para la tarea del usuario.
El plan debe incluir pasos numerados y el impacto esperado.
Devuelve SOLO el plan, sin conversación."""),
        *estado["mensajes"],
    ])
    print(f"\n[Plan generado]\n{r.content}\n")
    return {"plan_accion": r.content}


def nodo_critico(estado: EstadoAprobacion) -> dict:
    """
    Nodo que realiza una acción de alto impacto.
    El grafo se pausará ANTES de ejecutar este nodo para aprobación humana.
    """
    print("[Ejecutando acción crítica...]")
    # En producción: enviar emails, modificar base de datos, llamar APIs externas, etc.
    resultado = f"Acción ejecutada exitosamente basándose en el plan:\n{estado['plan_accion']}"
    return {"resultado_ejecucion": resultado}


def finalizar(estado: EstadoAprobacion) -> dict:
    """Genera la respuesta final."""
    r = llm.invoke([
        SystemMessage(content="Resume el resultado de la operación de forma clara y concisa."),
        HumanMessage(content=f"Resultado de la ejecución:\n{estado['resultado_ejecucion']}"),
    ])
    return {"mensajes": [r]}


# --- Construir el grafo ---

checkpointer = MemorySaver()
builder = StateGraph(EstadoAprobacion)

builder.add_node("generar_plan", generar_plan)
builder.add_node("nodo_critico", nodo_critico)
builder.add_node("finalizar", finalizar)

builder.add_edge(START, "generar_plan")
builder.add_edge("generar_plan", "nodo_critico")
builder.add_edge("nodo_critico", "finalizar")
builder.add_edge("finalizar", END)

# interrupt_before pausa el grafo ANTES de ejecutar "nodo_critico"
grafo_hitl = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["nodo_critico"],
)

config = {"configurable": {"thread_id": "aprobacion_sesion_1"}}


# --- Flujo de trabajo con aprobación humana ---

print("=== FASE 1: Generar el plan ===")

# El grafo se ejecuta hasta el punto de interrupción (antes de "nodo_critico")
grafo_hitl.invoke(
    {
        "mensajes": [HumanMessage(content="Envía un email a todos los clientes anunciando el nuevo precio.")],
        "plan_accion": "",
        "aprobado_por_humano": False,
        "resultado_ejecucion": "",
    },
    config=config,
)

# En este punto el grafo está PAUSADO — inspeccionar el estado
estado_actual = grafo_hitl.get_state(config)
print(f"\nGrafo pausado. Nodo siguiente: {estado_actual.next}")
print(f"\nPlan generado:\n{estado_actual.values['plan_accion']}")


# --- Simular la decisión del humano ---

print("\n=== FASE 2: Decisión humana ===")
decision = input("¿Aprobar el plan? (s/n): ").strip().lower()

if decision == "s":
    print("\nPlan aprobado. Reanudando ejecución...")
    # Reanudar el grafo desde donde se pausó
    resultado_final = grafo_hitl.invoke(None, config=config)
    print("\nEjecución completada.")
    print(resultado_final["mensajes"][-1].content)
else:
    print("\nPlan rechazado. Modificando el estado antes de reanudar...")
    # Opción: actualizar el plan antes de continuar
    nuevo_plan = "Plan alternativo más conservador: enviar solo a clientes Premium."
    grafo_hitl.update_state(
        config,
        {"plan_accion": nuevo_plan},
    )
    # Reanudar con el plan modificado
    resultado_final = grafo_hitl.invoke(None, config=config)
    print("\nEjecución completada con plan modificado.")
    print(resultado_final["mensajes"][-1].content)
```

**Flujo visual con `interrupt_before`:**

```
invoke() → [generar_plan] → PAUSA ← aquí el humano inspecciona
                                  ↓
                           invoke(None) → [nodo_critico] → [finalizar] → END
```

---

## 9. Extensiones sugeridas

| Extensión | Descripción | Recurso |
|-----------|-------------|---------|
| **SqliteSaver** | Persistencia en disco en lugar de en memoria | `from langgraph.checkpoint.sqlite import SqliteSaver` |
| **Multi-agente con LangGraph** | Supervisor que delega en agentes especializados como subgrafos | Documentación oficial LangGraph |
| **Streaming de nodos** | Recibir tokens a medida que cada nodo produce output | `grafo.stream(input, stream_mode="values")` |
| **LangGraph Studio** | UI visual para inspeccionar grafos en tiempo real | [smith.langchain.com](https://smith.langchain.com) |
| **LangGraph Cloud** | Despliegue gestionado con API REST | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| **DSPy + LangGraph** | Usar módulos DSPy optimizados como nodos de un grafo | Ver tutorial 08-dspy |

---

**Anterior:** [08 — DSPy: Programar en lugar de escribir prompts](./08-dspy.md) · Fin del bloque LLMs
