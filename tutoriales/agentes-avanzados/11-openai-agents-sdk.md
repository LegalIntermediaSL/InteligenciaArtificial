# 11 — OpenAI Agents SDK: Handoffs, Guardrails y Swarm

> **Bloque:** Agentes Avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 65 min

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/agentes-avanzados/11-openai-agents-sdk.ipynb)

---

## Índice

1. [Qué es el OpenAI Agents SDK](#1-qué-es-el-openai-agents-sdk)
2. [Instalación y configuración](#2-instalación-y-configuración)
3. [Agent básico con herramientas](#3-agent-básico-con-herramientas)
4. [Handoffs: pasar el control entre agentes](#4-handoffs-pasar-el-control-entre-agentes)
5. [Guardrails de entrada y salida](#5-guardrails-de-entrada-y-salida)
6. [Sistema multi-agente: triage y especialistas](#6-sistema-multi-agente-triage-y-especialistas)
7. [Runner síncrono vs streaming](#7-runner-síncrono-vs-streaming)
8. [Comparativa con otras alternativas](#8-comparativa-con-otras-alternativas)
9. [Cuándo elegir OpenAI Agents SDK](#9-cuándo-elegir-openai-agents-sdk)
10. [Extensiones sugeridas](#10-extensiones-sugeridas)

---

## 1. Qué es el OpenAI Agents SDK

El **OpenAI Agents SDK** (anteriormente conocido como Swarm) es el framework oficial de OpenAI para construir sistemas multi-agente. Fue lanzado como librería de producción en 2025 tras las lecciones aprendidas con Swarm (el experimento educativo de 2024).

**Filosofía de diseño:** minimalista y composable. A diferencia de LangGraph (que modela flujos como grafos) o LangChain (que encadena componentes), el SDK de OpenAI apuesta por tres primitivas simples que se pueden combinar de forma infinita.

### Primitivas del SDK

| Primitiva | Descripción | Analogía |
|-----------|-------------|---------|
| `Agent` | LLM con instrucciones, herramientas y handoffs disponibles | Empleado con rol específico |
| `Runner` | Ejecutor del bucle agente → herramienta → agente | Orquestador |
| `handoff()` | Transferencia de control a otro agente | Derivación a otro departamento |
| `tool` | Función Python decorada que el agente puede llamar | Capacidad del empleado |
| `InputGuardrail` | Validación de la entrada del usuario | Control de acceso |
| `OutputGuardrail` | Validación de la respuesta antes de devolverla | Control de calidad |

**Bucle de ejecución:**

```
Usuario → Runner.run(agent, input)
  → Agent selecciona: herramienta / handoff / respuesta final
  → Si herramienta: ejecuta, añade resultado al contexto, repite
  → Si handoff: el agente destino toma el control, repite
  → Si respuesta final: Runner devuelve RunResult
```

---

## 2. Instalación y configuración

```python
# Instalación
# pip install openai-agents

import os
from agents import Agent, Runner, handoff, tool, RunResult
from agents.guardrails import InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

# Configurar API key
os.environ["OPENAI_API_KEY"] = "sk-..."  # O usar variable de entorno

# Verificar instalación
import agents
print(f"OpenAI Agents SDK versión: {agents.__version__}")
```

---

## 3. Agent básico con herramientas

Un `Agent` se define con un nombre, instrucciones (system prompt efectivo) y una lista de herramientas.

```python
import datetime
import json

# --- Definición de herramientas ---

@tool
def obtener_hora_actual(zona_horaria: str = "Europe/Madrid") -> str:
    """Devuelve la hora actual en la zona horaria especificada."""
    # En producción usarías pytz o zoneinfo
    ahora = datetime.datetime.now()
    return f"Hora actual en {zona_horaria}: {ahora.strftime('%H:%M:%S del %d/%m/%Y')}"


@tool
def calcular_precio_con_iva(precio_base: float, tipo_iva: float = 21.0) -> dict:
    """
    Calcula el precio final con IVA.
    
    Args:
        precio_base: Precio sin IVA en euros
        tipo_iva: Porcentaje de IVA (por defecto 21%)
    
    Returns:
        dict con precio base, IVA y precio total
    """
    iva_euros = precio_base * (tipo_iva / 100)
    total = precio_base + iva_euros
    return {
        "precio_base": round(precio_base, 2),
        "iva_porcentaje": tipo_iva,
        "iva_euros": round(iva_euros, 2),
        "precio_total": round(total, 2)
    }


@tool
def buscar_producto(nombre: str) -> list[dict]:
    """
    Busca productos en el catálogo por nombre.
    
    Args:
        nombre: Nombre o parte del nombre del producto
    
    Returns:
        Lista de productos encontrados con precio y stock
    """
    # Catálogo simulado
    catalogo = [
        {"id": "P001", "nombre": "Laptop Pro 15", "precio": 1299.99, "stock": 5},
        {"id": "P002", "nombre": "Laptop Air 13", "precio": 899.99, "stock": 12},
        {"id": "P003", "nombre": "Mouse Inalámbrico", "precio": 29.99, "stock": 50},
        {"id": "P004", "nombre": "Teclado Mecánico", "precio": 89.99, "stock": 8},
        {"id": "P005", "nombre": "Monitor 4K 27\"", "precio": 399.99, "stock": 3},
    ]
    
    nombre_lower = nombre.lower()
    return [p for p in catalogo if nombre_lower in p["nombre"].lower()]


# --- Definición del agente ---

agente_ventas_basico = Agent(
    name="AgenteVentas",
    instructions="""Eres un asistente de ventas amable y eficiente para una tienda de tecnología.
    
Tu función es:
- Ayudar a los clientes a encontrar productos que se ajusten a sus necesidades
- Proporcionar información clara sobre precios con y sin IVA
- Informar sobre disponibilidad de stock
- Responder en el mismo idioma que el cliente

Siempre usa las herramientas disponibles antes de inventar información sobre productos o precios.
Sé conciso pero completo en tus respuestas.""",
    tools=[buscar_producto, calcular_precio_con_iva, obtener_hora_actual],
    model="gpt-4o-mini"
)

# --- Ejecución ---
async def demo_agente_basico():
    resultado: RunResult = await Runner.run(
        agente_ventas_basico,
        "Hola, busco un laptop. ¿Qué tienes disponible y cuánto costaría con IVA?"
    )
    print(resultado.final_output)

# En Jupyter: await demo_agente_basico()
# En script: asyncio.run(demo_agente_basico())
```

---

## 4. Handoffs: pasar el control entre agentes

Un **handoff** es la transferencia de control de un agente a otro. El agente origen decide cuándo y a quién transferir según su lógica interna. El contexto de la conversación se preserva automáticamente.

```python
# --- Agentes especializados ---

agente_soporte_tecnico = Agent(
    name="SoporteTecnico",
    instructions="""Eres el especialista en soporte técnico.
    
Resuelves problemas técnicos: configuración, errores, compatibilidad, drivers.
Si el cliente tiene un problema de facturación o quiere comprar, transfiere a Ventas.
Pide siempre el modelo del producto y el sistema operativo antes de diagnosticar.""",
    tools=[],  # Añadiría herramientas de diagnóstico en producción
    model="gpt-4o-mini"
)

agente_facturacion = Agent(
    name="Facturacion",
    instructions="""Eres el especialista en facturación y pagos.
    
Gestionas: facturas, devoluciones, cambios de método de pago, estados de pedido.
Si el cliente tiene una duda técnica, transfiere a Soporte Técnico.
Siempre solicita el número de pedido o DNI antes de acceder a datos de cuenta.""",
    tools=[calcular_precio_con_iva],
    model="gpt-4o-mini"
)

# Agente de triage con handoffs a los especialistas
agente_triage = Agent(
    name="Triage",
    instructions="""Eres el primer punto de contacto de atención al cliente.
    
Tu única función es identificar el tipo de consulta y transferir al especialista correcto:
- Problemas técnicos, errores, configuración → Soporte Técnico
- Facturas, pagos, pedidos, devoluciones → Facturación
- Compras, catálogo, precios → Ventas

No intentes resolver los problemas tú mismo. Identifica y transfiere.
Saluda brevemente y confirma al cliente adónde lo estás derivando.""",
    handoffs=[
        handoff(agente_soporte_tecnico, tool_name_override="transferir_a_soporte"),
        handoff(agente_facturacion, tool_name_override="transferir_a_facturacion"),
        handoff(agente_ventas_basico, tool_name_override="transferir_a_ventas"),
    ],
    model="gpt-4o-mini"
)

# --- Rastrear el agente que manejó la conversación ---
async def demo_handoff():
    consultas = [
        "Mi laptop no arranca después de la última actualización de Windows",
        "Necesito la factura del pedido 12345 para presentarla en Hacienda",
        "¿Tenéis monitores 4K disponibles?"
    ]
    
    for consulta in consultas:
        print(f"\n{'='*60}")
        print(f"Cliente: {consulta}")
        
        resultado = await Runner.run(agente_triage, consulta)
        
        print(f"Agente que respondió: {resultado.last_agent.name}")
        print(f"Respuesta: {resultado.final_output}")
```

### Handoff con contexto adicional

Puedes pasar información adicional al agente destino al hacer el handoff:

```python
from agents import handoff

def construir_handoff_con_contexto(agente_destino: Agent, motivo: str):
    """Handoff que enriquece el contexto para el agente destino."""
    
    async def on_handoff(ctx):
        # Se ejecuta justo antes de la transferencia
        print(f"[Handoff] Transfiriendo a {agente_destino.name}: {motivo}")
    
    return handoff(
        agente_destino,
        on_handoff=on_handoff,
        input_filter=lambda msgs: msgs  # Puedes filtrar/modificar el historial
    )
```

---

## 5. Guardrails de entrada y salida

Los guardrails permiten **validar y filtrar** lo que entra al agente y lo que sale, sin modificar la lógica del agente mismo.

### 5.1 InputGuardrail: filtrar entradas inapropiadas

```python
from agents.guardrails import InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

class ClasificacionConsulta(BaseModel):
    """Resultado de clasificar si la consulta es apropiada."""
    es_apropiada: bool
    categoria: str  # "apropiada" | "off_topic" | "inapropiada" | "competencia"
    razon: str


agente_clasificador = Agent(
    name="ClasificadorConsultas",
    instructions="""Clasifica si la consulta del usuario es apropiada para un
    servicio de atención al cliente de una tienda de tecnología.
    
    - apropiada: relacionada con productos, pedidos, soporte técnico
    - off_topic: preguntas no relacionadas con la tienda (política, medicina, etc.)
    - inapropiada: lenguaje ofensivo, amenazas, spam
    - competencia: preguntas comparativas sobre competidores directos
    
    Responde en formato JSON con los campos: es_apropiada, categoria, razon""",
    output_type=ClasificacionConsulta,
    model="gpt-4o-mini"
)


async def guardrail_entrada(ctx, agent: Agent, input_data: str) -> GuardrailFunctionOutput:
    """
    Guardrail que evalúa si la entrada del usuario es apropiada.
    Se ejecuta antes de que el agente principal procese el mensaje.
    """
    resultado = await Runner.run(agente_clasificador, input_data)
    clasificacion: ClasificacionConsulta = resultado.final_output
    
    return GuardrailFunctionOutput(
        output_info=clasificacion,
        tripwire_triggered=not clasificacion.es_apropiada
        # Si tripwire=True, el Runner lanza GuardrailTripwireTriggered
    )


# Aplicar el guardrail al agente de triage
agente_con_guardrail = Agent(
    name="TriageProtegido",
    instructions=agente_triage.instructions,
    handoffs=agente_triage.handoffs,
    input_guardrails=[
        InputGuardrail(guardrail_function=guardrail_entrada)
    ],
    model="gpt-4o-mini"
)
```

### 5.2 OutputGuardrail: validar respuestas antes de devolverlas

```python
from agents.guardrails import OutputGuardrail

class ValidacionRespuesta(BaseModel):
    """Validación de que la respuesta cumple los estándares."""
    es_valida: bool
    contiene_pii: bool  # Información personal identificable
    tono_apropiado: bool
    problemas: list[str]


agente_validador_respuestas = Agent(
    name="ValidadorRespuestas",
    instructions="""Analiza si la respuesta de un agente de atención al cliente
    cumple estos criterios:
    1. No contiene información personal identificable (PII) no solicitada
    2. El tono es profesional y amable
    3. No hace promesas no autorizadas (descuentos no estándar, fechas imposibles)
    4. No menciona sistemas internos o datos confidenciales
    
    Responde en formato JSON con: es_valida, contiene_pii, tono_apropiado, problemas (lista)""",
    output_type=ValidacionRespuesta,
    model="gpt-4o-mini"
)


async def guardrail_salida(ctx, agent: Agent, output: str) -> GuardrailFunctionOutput:
    """Valida la respuesta generada antes de enviarla al usuario."""
    resultado = await Runner.run(agente_validador_respuestas, output)
    validacion: ValidacionRespuesta = resultado.final_output
    
    if not validacion.es_valida:
        print(f"[OutputGuardrail] Respuesta bloqueada. Problemas: {validacion.problemas}")
    
    return GuardrailFunctionOutput(
        output_info=validacion,
        tripwire_triggered=not validacion.es_valida
    )


# Agente con ambos guardrails
agente_completo = Agent(
    name="AgenteCompleto",
    instructions=agente_triage.instructions,
    handoffs=agente_triage.handoffs,
    input_guardrails=[InputGuardrail(guardrail_function=guardrail_entrada)],
    output_guardrails=[OutputGuardrail(guardrail_function=guardrail_salida)],
    model="gpt-4o-mini"
)
```

---

## 6. Sistema multi-agente: triage y especialistas

Aquí integramos todo en un sistema de atención al cliente con tres agentes especializados más un triage.

```python
from agents import Agent, Runner, handoff, tool
from agents.exceptions import GuardrailTripwireTriggered
import asyncio

# --- Herramientas por dominio ---

@tool
def consultar_estado_pedido(numero_pedido: str) -> dict:
    """Consulta el estado de un pedido por su número."""
    # Simulación de BD
    pedidos = {
        "12345": {"estado": "En tránsito", "fecha_entrega": "23/04/2026", "transportista": "SEUR"},
        "12346": {"estado": "Entregado", "fecha_entrega": "18/04/2026", "transportista": "DHL"},
        "12347": {"estado": "Procesando", "fecha_entrega": "25/04/2026", "transportista": "Pendiente"},
    }
    return pedidos.get(numero_pedido, {"error": "Pedido no encontrado"})


@tool
def iniciar_devolucion(numero_pedido: str, motivo: str) -> dict:
    """Inicia el proceso de devolución para un pedido."""
    return {
        "solicitud_id": f"DEV-{numero_pedido}-001",
        "estado": "Iniciada",
        "instrucciones": "Recibirás un email con la etiqueta de devolución en 24h",
        "plazo_reembolso": "5-7 días hábiles"
    }


@tool
def diagnosticar_problema(
    modelo_dispositivo: str,
    sistema_operativo: str,
    descripcion_problema: str
) -> dict:
    """Proporciona diagnóstico inicial para problemas técnicos."""
    return {
        "diagnostico": "Posible conflicto de drivers tras actualización",
        "pasos_iniciales": [
            "Reiniciar en modo seguro (F8 durante arranque)",
            "Ejecutar sfc /scannow en CMD como administrador",
            "Verificar actualizaciones pendientes en Device Manager"
        ],
        "escalacion_necesaria": False,
        "ticket_id": "TKT-2026-04210001"
    }


# --- Sistema multi-agente completo ---

especialista_ventas = Agent(
    name="EspecialistaVentas",
    instructions="""Especialista en ventas de tecnología. Ayudas a los clientes a:
    - Encontrar el producto adecuado según sus necesidades y presupuesto
    - Entender las diferencias entre productos
    - Completar su proceso de compra
    
    Siempre pregunta: ¿para qué usará el producto? y ¿cuál es su presupuesto?
    Ofrece siempre 2-3 opciones con pros y contras.""",
    tools=[buscar_producto, calcular_precio_con_iva],
    model="gpt-4o-mini"
)

especialista_pedidos = Agent(
    name="EspecialistaPedidos",
    instructions="""Especialista en gestión de pedidos y post-venta. Gestionas:
    - Consultas sobre estado de pedidos
    - Devoluciones y cambios
    - Reclamaciones de entrega
    
    SIEMPRE solicita el número de pedido antes de consultar nada.
    Sigue el protocolo: consultar estado → ofrecer soluciones → iniciar acción.""",
    tools=[consultar_estado_pedido, iniciar_devolucion, calcular_precio_con_iva],
    model="gpt-4o-mini"
)

especialista_tecnico = Agent(
    name="EspecialistaTecnico",
    instructions="""Especialista en soporte técnico. Resuelves:
    - Problemas de software y hardware
    - Configuración de dispositivos
    - Errores y mensajes de sistema
    
    Protocolo de diagnóstico:
    1. Modelo exacto del dispositivo
    2. Sistema operativo y versión
    3. Descripción detallada del problema
    4. Cuándo empezó y qué cambió""",
    tools=[diagnosticar_problema],
    model="gpt-4o-mini"
)

agente_triage_completo = Agent(
    name="Recepcionista",
    instructions="""Eres la recepcionista virtual de TechStore.
    
Saluda amablemente y redirige cada consulta al especialista correcto:
- Compras y catálogo → EspecialistaVentas
- Pedidos, entregas, devoluciones → EspecialistaPedidos  
- Problemas técnicos → EspecialistaTecnico

No resuelvas consultas tú mismo. Tu función es clasificar y derivar.
Al derivar, informa brevemente al cliente: "Te paso con [nombre], que te ayudará con [tema]".""",
    handoffs=[
        handoff(especialista_ventas),
        handoff(especialista_pedidos),
        handoff(especialista_tecnico),
    ],
    model="gpt-4o-mini"
)


# --- Función principal con gestión de guardrails ---

async def atender_cliente(consulta: str) -> str:
    """
    Punto de entrada del sistema multi-agente.
    Gestiona errores de guardrail y devuelve respuesta al cliente.
    """
    try:
        resultado = await Runner.run(agente_triage_completo, consulta)
        return resultado.final_output
    
    except GuardrailTripwireTriggered as e:
        return ("Lo sentimos, no podemos procesar esta solicitud. "
                "Si crees que es un error, contacta con nosotros en soporte@techstore.es")
    
    except Exception as e:
        return f"Error inesperado: {str(e)}"


# Prueba del sistema completo
async def demo_sistema_completo():
    casos = [
        "¿Tenéis laptops por menos de 1000€? Necesito una para diseño gráfico",
        "¿Dónde está mi pedido 12345?",
        "Mi Laptop Pro 15 no reconoce el WiFi desde la última actualización",
    ]
    
    for caso in casos:
        print(f"\nCliente: {caso}")
        respuesta = await atender_cliente(caso)
        print(f"Sistema: {respuesta[:300]}...")
```

---

## 7. Runner síncrono vs streaming

### 7.1 Ejecución estándar (síncrona/async)

```python
import asyncio

# Forma async (recomendada)
async def ejecutar_async():
    resultado = await Runner.run(agente_triage_completo, "¿Cuál es vuestro laptop más vendido?")
    print(f"Agente final: {resultado.last_agent.name}")
    print(f"Respuesta: {resultado.final_output}")
    
    # Acceder al historial completo de mensajes
    for item in resultado.new_items:
        print(f"  [{item.type}] {str(item)[:100]}")

# Forma síncrona (para scripts, no en async contexts)
def ejecutar_sync():
    resultado = Runner.run_sync(agente_triage_completo, "¿Cuál es vuestro laptop más vendido?")
    print(resultado.final_output)
```

### 7.2 Streaming con Runner.run_streamed()

El streaming es crucial para interfaces de usuario donde el usuario debe ver la respuesta mientras se genera.

```python
async def demo_streaming():
    """Muestra la respuesta token a token conforme se genera."""
    
    print("Respuesta en streaming:\n")
    
    result = Runner.run_streamed(
        agente_ventas_basico,
        "Explícame las diferencias entre todos vuestros laptops disponibles"
    )
    
    async for event in result.stream_events():
        # Tipos de eventos: raw_response_event, run_item_stream_event, agent_updated_stream_event
        if event.type == "raw_response_event":
            # Tokens de texto conforme llegan
            if hasattr(event.data, "delta") and hasattr(event.data.delta, "text"):
                print(event.data.delta.text, end="", flush=True)
        
        elif event.type == "agent_updated_stream_event":
            # Cuando hay un handoff
            print(f"\n[Handoff → {event.new_agent.name}]\n")
    
    print("\n\n--- Stream completado ---")
    print(f"Agente final: {result.current_agent.name}")
```

---

## 8. Comparativa con otras alternativas

| Característica | OpenAI Agents SDK | Claude Agent SDK | LangGraph |
|----------------|:-----------------:|:----------------:|:---------:|
| **Proveedor** | OpenAI | Anthropic | LangChain |
| **Modelos** | GPT-4o, GPT-4o-mini | Claude Sonnet/Haiku/Opus | Cualquiera |
| **Curva de aprendizaje** | Baja | Baja | Media-Alta |
| **Primitivas** | Agent, Runner, handoff, tool | Agent, tool, memory | Node, Edge, Graph |
| **Streaming** | Sí (nativo) | Sí (nativo) | Sí |
| **Handoffs** | Nativo y simple | Nativo | Via edges |
| **Guardrails** | Nativo | Via tools | Manual |
| **Visualización de grafo** | No | No | Sí |
| **Ciclos complejos** | Limitado | Limitado | Excelente |
| **Persistencia de estado** | Manual | Manual | LangGraph Platform |
| **Multi-proveedor** | No | No | Sí |
| **Producción lista** | Sí | Sí | Sí |
| **Licencia** | MIT | Anthropic | MIT |

**Cuándo cada uno brilla:**

- **OpenAI Agents SDK:** ya usas GPT-4o, quieres handoffs simples y guardrails sin complejidad
- **Claude Agent SDK:** quieres las capacidades de razonamiento de Claude, Extended Thinking en agentes
- **LangGraph:** flujos complejos con ciclos, múltiples proveedores, trazabilidad completa, estado persistente

---

## 9. Cuándo elegir OpenAI Agents SDK

**Casos donde es la mejor opción:**

✅ Ya tienes integración con la API de OpenAI y quieres añadir agentes con mínimo cambio  
✅ Necesitas handoffs simples entre agentes con comportamiento determinista  
✅ Tu equipo conoce Python y quiere una curva de aprendizaje mínima  
✅ Quieres guardrails integrados sin librería adicional  
✅ Prototype rápido que puede ir a producción sin reescritura

**Casos donde probablemente no es la mejor opción:**

❌ Necesitas flujos no lineales complejos (loops, condicionales anidados) → LangGraph  
❌ Quieres usar múltiples proveedores de LLM en el mismo sistema → LangGraph + LiteLLM  
❌ Tu caso requiere capacidades avanzadas de razonamiento → Claude Agent SDK con Extended Thinking  
❌ Necesitas visualización del grafo de ejecución para debug → LangGraph  
❌ Requieres persistencia de estado entre sesiones con soporte oficial → LangGraph Platform

---

## 10. Extensiones sugeridas

- **Context object:** usar el objeto `RunContext` para pasar datos compartidos entre agentes (usuario autenticado, preferencias, historial) sin contaminar los mensajes
- **Trazabilidad:** integrar con OpenAI Traces o con Langfuse para observabilidad completa
- **Paralelismo:** ejecutar múltiples `Runner.run()` en paralelo con `asyncio.gather()` para sub-tareas independientes
- **Agent como herramienta:** un agente puede exponer otro agente como herramienta, creando jerarquías más profundas
- **Dynamic instructions:** generar las instrucciones del agente dinámicamente según el contexto del usuario

---

*Siguiente artículo: [12 — Model Context Protocol (MCP): Herramientas Externas Estandarizadas](12-model-context-protocol.md)*
