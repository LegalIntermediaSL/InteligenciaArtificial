# 05 — AutoGen: Conversaciones Multi-Agente

> **Bloque:** Agentes avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 55 min

---

## Índice

1. [Qué es AutoGen y cuándo usarlo](#1-qué-es-autogen-y-cuándo-usarlo)
2. [Conceptos fundamentales](#2-conceptos-fundamentales)
3. [Conversación de dos agentes](#3-conversación-de-dos-agentes)
4. [GroupChat: múltiples agentes](#4-groupchat-múltiples-agentes)
5. [Agentes con herramientas y código](#5-agentes-con-herramientas-y-código)
6. [Patrones de orquestación](#6-patrones-de-orquestación)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Qué es AutoGen y cuándo usarlo

**AutoGen** (Microsoft Research) es un framework para construir sistemas donde múltiples agentes LLM colaboran mediante conversaciones estructuradas. A diferencia de LangChain o CrewAI, AutoGen pone el énfasis en la **conversación** como primitiva central.

### Cuándo usar AutoGen vs otras opciones

| Framework | Modelo de colaboración | Ideal para |
|---|---|---|
| **AutoGen** | Conversación entre agentes | Tareas que se benefician de debate, revisión y crítica entre agentes |
| **CrewAI** | Roles y tareas asignadas | Pipelines de trabajo con roles bien definidos |
| **LangGraph** | Grafo de estados | Flujos de trabajo complejos con lógica condicional y ciclos |
| **Código manual** | Custom | Control total, casos de uso muy específicos |

AutoGen destaca en: **revisión de código**, **debate de soluciones**, **investigación con múltiples perspectivas** y **generación iterativa de contenido**.

---

## 2. Conceptos fundamentales

```bash
pip install pyautogen
```

```python
# autogen_concepts.py
import autogen

# Configuración del LLM (compatible con OpenAI, Claude via proxy, etc.)
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": "sk-...",
        }
    ],
    "temperature": 0.3,
    "timeout": 120,
    "cache_seed": None  # None para deshabilitar cache en desarrollo
}

# Con Claude (via API compatible con OpenAI)
llm_config_claude = {
    "config_list": [
        {
            "model": "claude-opus-4-6",
            "api_key": "sk-ant-...",
            "base_url": "https://api.anthropic.com/v1",
            "api_type": "anthropic"
        }
    ]
}
```

---

## 3. Conversación de dos agentes

```python
# two_agents.py
import autogen

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "sk-..."}],
    "temperature": 0.3
}

# AssistantAgent: agente que usa LLM para responder
asistente = autogen.AssistantAgent(
    name="Asistente",
    llm_config=llm_config,
    system_message=(
        "Eres un experto en Python. Cuando escribas código, "
        "siempre incluye tests y manejo de errores. "
        "Cuando creas que la tarea está completa, di TERMINATE."
    )
)

# UserProxyAgent: representa al usuario o ejecuta código
usuario = autogen.UserProxyAgent(
    name="Usuario",
    human_input_mode="NEVER",      # "NEVER" para automatizar, "ALWAYS" para interactivo
    max_consecutive_auto_reply=10, # máximo de turnos automáticos
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    code_execution_config={
        "work_dir": "/tmp/autogen",
        "use_docker": False  # True en producción para aislamiento
    }
)

# Iniciar conversación
resultado = usuario.initiate_chat(
    asistente,
    message=(
        "Escribe una función Python que calcule el número de palabras únicas "
        "en un texto, ignorando mayúsculas y puntuación. "
        "Incluye tests con al menos 3 casos."
    )
)

# El agente genera código, el UserProxy lo ejecuta y devuelve el resultado
# Este ciclo continúa hasta que se alcanza TERMINATE o max_consecutive_auto_reply
```

---

## 4. GroupChat: múltiples agentes

```python
# groupchat.py
import autogen

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]}

# Agentes con roles especializados
arquitecto = autogen.AssistantAgent(
    name="Arquitecto",
    llm_config=llm_config,
    system_message=(
        "Eres un arquitecto de software senior. "
        "Propones la estructura y diseño de alto nivel. "
        "Siempre justificas tus decisiones de arquitectura."
    )
)

desarrollador = autogen.AssistantAgent(
    name="Desarrollador",
    llm_config=llm_config,
    system_message=(
        "Eres un desarrollador Python experto. "
        "Implementas el código siguiendo las directrices del Arquitecto. "
        "Escribes código limpio, con docstrings y tests."
    )
)

revisor = autogen.AssistantAgent(
    name="Revisor",
    llm_config=llm_config,
    system_message=(
        "Eres un revisor de código experimentado. "
        "Revisas el código del Desarrollador buscando: "
        "bugs, problemas de seguridad, mejoras de rendimiento y buenas prácticas. "
        "Sé constructivo pero estricto."
    )
)

gestor_proyecto = autogen.UserProxyAgent(
    name="GerenteProyecto",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=20,
    is_termination_msg=lambda msg: "APROBADO" in msg.get("content", ""),
    code_execution_config={"work_dir": "/tmp/autogen", "use_docker": False}
)

# GroupChat orquesta la conversación entre múltiples agentes
groupchat = autogen.GroupChat(
    agents=[gestor_proyecto, arquitecto, desarrollador, revisor],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"  # el LLM decide quién habla
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Iniciar proceso de desarrollo colaborativo
gestor_proyecto.initiate_chat(
    manager,
    message=(
        "Necesitamos implementar un sistema de caché LRU (Least Recently Used) "
        "en Python. Debe ser thread-safe y soportar TTL (expiración de entradas). "
        "El Arquitecto diseñará la solución, el Desarrollador la implementará "
        "y el Revisor la revisará. Di APROBADO cuando el código esté listo para producción."
    )
)
```

---

## 5. Agentes con herramientas y código

```python
# autogen_tools.py
import autogen
import httpx
import json
from typing import Annotated

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]}


# Definir herramientas como funciones Python decoradas
def buscar_en_wikipedia(
    query: Annotated[str, "Término de búsqueda en Wikipedia"]
) -> str:
    """Busca información en Wikipedia y devuelve un resumen."""
    try:
        r = httpx.get(
            "https://es.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("extract", "No encontrado")[:1000]
        return "No se encontró información"
    except Exception as e:
        return f"Error: {e}"


def calcular(
    expresion: Annotated[str, "Expresión matemática a evaluar, ej: '2 + 2 * 10'"]
) -> str:
    """Evalúa una expresión matemática de forma segura."""
    import ast
    try:
        # Solo permitir operaciones matemáticas seguras
        tree = ast.parse(expresion, mode='eval')
        result = eval(compile(tree, "<string>", "eval"))
        return str(result)
    except Exception as e:
        return f"Error en expresión: {e}"


# Registrar herramientas en los agentes
asistente_con_herramientas = autogen.AssistantAgent(
    name="AsistenteInvestigador",
    llm_config={
        **llm_config,
        "functions": [
            {
                "name": "buscar_en_wikipedia",
                "description": "Busca información en Wikipedia",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Término de búsqueda"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calcular",
                "description": "Evalúa expresiones matemáticas",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expresion": {"type": "string"}
                    },
                    "required": ["expresion"]
                }
            }
        ]
    },
    system_message="Eres un investigador. Usa las herramientas disponibles para responder."
)

proxy = autogen.UserProxyAgent(
    name="Proxy",
    human_input_mode="NEVER",
    function_map={
        "buscar_en_wikipedia": buscar_en_wikipedia,
        "calcular": calcular
    }
)

# AutoGen 0.2+ soporta decoradores para registro más limpio
# autogen.register_function(buscar_en_wikipedia, caller=asistente, executor=proxy, ...)
```

---

## 6. Patrones de orquestación

```python
# orchestration_patterns.py
import autogen

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]}


# Patrón 1: Secuencial (A → B → C)
def pipeline_secuencial(tarea: str):
    """Cada agente procesa la salida del anterior."""
    redactor = autogen.AssistantAgent("Redactor", llm_config=llm_config,
        system_message="Escribe un primer borrador del contenido solicitado.")
    editor = autogen.AssistantAgent("Editor", llm_config=llm_config,
        system_message="Mejora el borrador: claridad, concisión, estructura. Devuelve el texto mejorado.")
    corrector = autogen.AssistantAgent("Corrector", llm_config=llm_config,
        system_message="Corrige errores gramaticales y ortográficos. Devuelve el texto final. Di TERMINE al acabar.")

    proxy = autogen.UserProxyAgent("Proxy", human_input_mode="NEVER",
        is_termination_msg=lambda m: "TERMINE" in m.get("content", ""))

    # Paso 1: Redactar
    proxy.initiate_chat(redactor, message=tarea, max_turns=2)
    borrador = proxy.last_message()["content"]

    # Paso 2: Editar
    proxy.initiate_chat(editor, message=f"Mejora este texto:\n{borrador}", max_turns=2)
    editado = proxy.last_message()["content"]

    # Paso 3: Corregir
    proxy.initiate_chat(corrector, message=f"Corrige este texto:\n{editado}", max_turns=2)
    return proxy.last_message()["content"]


# Patrón 2: Debate y consenso
def debate_consenso(pregunta: str) -> str:
    """Dos agentes debaten y llegan a un consenso."""
    pro = autogen.AssistantAgent("Defensor", llm_config=llm_config,
        system_message="Defiendes el punto de vista afirmativo con argumentos sólidos.")
    contra = autogen.AssistantAgent("Crítico", llm_config=llm_config,
        system_message="Presentas contraargumentos y dudas sobre el punto afirmativo.")
    moderador = autogen.AssistantAgent("Moderador", llm_config=llm_config,
        system_message=(
            "Escuchas el debate y sintetizas una conclusión equilibrada. "
            "Cuando tengas suficiente información, sintetiza y di CONSENSO ALCANZADO."
        ))

    groupchat = autogen.GroupChat(
        agents=[pro, contra, moderador],
        messages=[],
        max_round=8,
        speaker_selection_method="round_robin"  # turnos rotativos
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    proxy = autogen.UserProxyAgent("Proxy", human_input_mode="NEVER",
        is_termination_msg=lambda m: "CONSENSO ALCANZADO" in m.get("content", ""))
    proxy.initiate_chat(manager, message=pregunta)

    return proxy.last_message()["content"]
```

---

## 7. Extensiones sugeridas

- **AutoGen Studio**: interfaz visual para construir y probar flujos multi-agente sin código
- **AutoGen 0.4**: nueva versión con AgentChat API más flexible y soporte asíncrono nativo
- **Persistent agents**: guardar el estado de la conversación en base de datos para reanudar sesiones
- **Integración con CrewAI**: usar agentes de CrewAI como herramientas dentro de AutoGen

---

**Anterior:** [04 — Memoria a largo plazo](./04-memoria-largo-plazo.md) · **Siguiente:** [06 — A2A Protocol](./06-a2a-protocol.md)
