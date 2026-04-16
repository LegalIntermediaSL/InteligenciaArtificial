# 12 — Function Calling Avanzado

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Qué es function calling y cómo funciona](#1-qué-es-function-calling-y-cómo-funciona)
2. [Definición de herramientas con esquemas JSON](#2-definición-de-herramientas-con-esquemas-json)
3. [Tool use en Anthropic (Claude)](#3-tool-use-en-anthropic-claude)
4. [Parallel tool use y tool chaining](#4-parallel-tool-use-y-tool-chaining)
5. [Herramientas con validación Pydantic](#5-herramientas-con-validación-pydantic)
6. [Patrones avanzados: agentes con herramientas](#6-patrones-avanzados-agentes-con-herramientas)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Qué es function calling y cómo funciona

**Function calling** (o tool use) es la capacidad de un LLM para decidir cuándo y cómo llamar a una función externa, recibir su resultado e integrarlo en la respuesta final.

El modelo no ejecuta código directamente — genera una instrucción estructurada que tu código ejecuta y devuelve como resultado.

```
Usuario: "¿Cuánto es el stock actual de producto SKU-123?"

LLM decide: necesito llamar a get_stock(sku="SKU-123")
→ Tu código ejecuta la consulta real a la BD
→ Tu código devuelve: {"stock": 47, "warehouse": "Madrid"}
→ LLM genera respuesta: "El producto SKU-123 tiene 47 unidades en el almacén de Madrid."
```

---

## 2. Definición de herramientas con esquemas JSON

Las herramientas se definen con JSON Schema. Un esquema claro y bien descrito mejora significativamente la precisión del modelo al elegir y llamar herramientas.

```python
# tool_definitions.py

HERRAMIENTA_BUSCAR_PRODUCTO = {
    "name": "buscar_producto",
    "description": (
        "Busca productos en el catálogo por nombre, SKU o categoría. "
        "Usa esta herramienta cuando el usuario pregunte por disponibilidad, "
        "precio o características de un producto."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Término de búsqueda: nombre del producto, SKU o descripción"
            },
            "categoria": {
                "type": "string",
                "enum": ["electronica", "ropa", "hogar", "deportes", "todos"],
                "description": "Filtrar por categoría. Usa 'todos' si no se especifica."
            },
            "max_resultados": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "Número máximo de resultados a devolver"
            }
        },
        "required": ["query"]
    }
}

HERRAMIENTA_CREAR_PEDIDO = {
    "name": "crear_pedido",
    "description": (
        "Crea un nuevo pedido para un cliente. "
        "IMPORTANTE: Confirmar siempre con el usuario antes de llamar esta herramienta."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "cliente_id": {"type": "string", "description": "ID del cliente"},
            "productos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "cantidad": {"type": "integer", "minimum": 1}
                    },
                    "required": ["sku", "cantidad"]
                },
                "description": "Lista de productos y cantidades"
            },
            "direccion_envio": {"type": "string"}
        },
        "required": ["cliente_id", "productos"]
    }
}

HERRAMIENTAS = [HERRAMIENTA_BUSCAR_PRODUCTO, HERRAMIENTA_CREAR_PEDIDO]
```

---

## 3. Tool use en Anthropic (Claude)

```python
# tool_use_anthropic.py
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()


# ── Implementaciones reales de las herramientas ────────────────────────────
def buscar_producto(query: str, categoria: str = "todos", max_resultados: int = 5) -> list[dict]:
    """Simula búsqueda en base de datos de productos."""
    productos_ficticios = [
        {"sku": "ELEC-001", "nombre": "Auriculares Bluetooth XZ200", "precio": 89.99, "stock": 23, "categoria": "electronica"},
        {"sku": "ELEC-002", "nombre": "Teclado Mecánico RGB Pro", "precio": 149.99, "stock": 8, "categoria": "electronica"},
        {"sku": "ROPA-001", "nombre": "Camiseta Running Dry-Fit", "precio": 29.99, "stock": 45, "categoria": "ropa"},
    ]
    resultados = [p for p in productos_ficticios if query.lower() in p["nombre"].lower()]
    if categoria != "todos":
        resultados = [p for p in resultados if p["categoria"] == categoria]
    return resultados[:max_resultados]


def crear_pedido(cliente_id: str, productos: list[dict], direccion_envio: str = "") -> dict:
    """Simula creación de pedido."""
    import uuid
    pedido_id = str(uuid.uuid4())[:8].upper()
    total = sum(p.get("precio", 0) * p.get("cantidad", 1) for p in productos)
    return {
        "pedido_id": pedido_id,
        "cliente_id": cliente_id,
        "estado": "confirmado",
        "total_estimado": total,
        "fecha_entrega_estimada": "3-5 días laborables"
    }


EJECUTORES_HERRAMIENTAS = {
    "buscar_producto": lambda args: buscar_producto(**args),
    "crear_pedido": lambda args: crear_pedido(**args),
}


# ── Bucle de tool use ───────────────────────────────────────────────────────
def agente_con_herramientas(pregunta: str, verbose: bool = True) -> str:
    """
    Bucle agentico con tool use:
    1. Enviar mensaje al LLM
    2. Si responde con tool_use, ejecutar la herramienta
    3. Devolver resultado al LLM
    4. Repetir hasta obtener respuesta final
    """
    messages = [{"role": "user", "content": pregunta}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=HERRAMIENTAS,
            messages=messages
        )

        if verbose:
            print(f"\nStop reason: {response.stop_reason}")

        # Respuesta final — sin más tool use
        if response.stop_reason == "end_turn":
            texto = next(b.text for b in response.content if hasattr(b, "text"))
            return texto

        # El modelo quiere usar una herramienta
        if response.stop_reason == "tool_use":
            # Añadir respuesta del asistente al historial
            messages.append({"role": "assistant", "content": response.content})

            # Ejecutar todas las herramientas solicitadas
            resultados_herramientas = []
            for bloque in response.content:
                if bloque.type == "tool_use":
                    nombre = bloque.name
                    args = bloque.input

                    if verbose:
                        print(f"  → Llamando {nombre}({json.dumps(args, ensure_ascii=False)})")

                    if nombre in EJECUTORES_HERRAMIENTAS:
                        resultado = EJECUTORES_HERRAMIENTAS[nombre](args)
                    else:
                        resultado = {"error": f"Herramienta {nombre} no implementada"}

                    if verbose:
                        print(f"  ← Resultado: {json.dumps(resultado, ensure_ascii=False)[:200]}")

                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": json.dumps(resultado, ensure_ascii=False)
                    })

            # Devolver resultados al modelo
            messages.append({"role": "user", "content": resultados_herramientas})


# Uso
respuesta = agente_con_herramientas("¿Tienes auriculares disponibles? Busca en electrónica.")
print(f"\nRespuesta final:\n{respuesta}")
```

---

## 4. Parallel tool use y tool chaining

```python
# parallel_tools.py

def agente_parallel_tools(pregunta: str) -> str:
    """
    Claude puede solicitar múltiples herramientas en paralelo en una sola respuesta.
    Esto es más eficiente que el encadenamiento secuencial.
    """
    messages = [{"role": "user", "content": pregunta}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            tools=HERRAMIENTAS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            # Todas las herramientas se ejecutan en paralelo (o secuencialmente aquí,
            # pero en producción usar asyncio.gather para true paralelismo)
            tool_results = []
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            print(f"  → Ejecutando {len(tool_calls)} herramientas en paralelo")

            for bloque in tool_calls:
                resultado = EJECUTORES_HERRAMIENTAS.get(bloque.name, lambda x: {"error": "no encontrado"})(bloque.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": json.dumps(resultado, ensure_ascii=False)
                })

            messages.append({"role": "user", "content": tool_results})


# Para paralelismo real con asyncio:
import asyncio

async def ejecutar_herramienta_async(nombre: str, args: dict) -> dict:
    """Wrapper async para ejecutar herramientas de forma paralela."""
    loop = asyncio.get_event_loop()
    ejecutor = EJECUTORES_HERRAMIENTAS.get(nombre, lambda x: {"error": "no encontrado"})
    return await loop.run_in_executor(None, ejecutor, args)
```

---

## 5. Herramientas con validación Pydantic

```python
# pydantic_tools.py
from pydantic import BaseModel, Field, validator
from typing import Optional
import inspect


class BuscarProductoArgs(BaseModel):
    query: str = Field(..., description="Término de búsqueda", min_length=1)
    categoria: str = Field("todos", description="Filtrar por categoría")
    max_resultados: int = Field(5, ge=1, le=20)

    @validator("categoria")
    def categoria_valida(cls, v):
        validas = {"electronica", "ropa", "hogar", "deportes", "todos"}
        if v not in validas:
            raise ValueError(f"Categoría debe ser una de: {validas}")
        return v


def pydantic_a_json_schema(modelo: type[BaseModel], descripcion: str) -> dict:
    """Convierte un modelo Pydantic a formato de herramienta de Anthropic."""
    schema = modelo.model_json_schema()
    return {
        "name": modelo.__name__.lower().replace("args", ""),
        "description": descripcion,
        "input_schema": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
    }


def ejecutar_con_validacion(nombre: str, args_raw: dict, modelo_args: type[BaseModel], fn) -> dict:
    """Valida los argumentos con Pydantic antes de ejecutar la herramienta."""
    try:
        args_validados = modelo_args(**args_raw)
        return fn(**args_validados.model_dump())
    except Exception as e:
        return {"error": str(e), "tipo": type(e).__name__}


# Uso
herramienta_validada = pydantic_a_json_schema(
    BuscarProductoArgs,
    "Busca productos en el catálogo. Usa esta herramienta para consultas de inventario."
)
print(json.dumps(herramienta_validada, ensure_ascii=False, indent=2))
```

---

## 6. Patrones avanzados: agentes con herramientas

```python
# agent_patterns.py

# Patrón: forzar el uso de una herramienta específica
def extraer_datos_estructurados(texto: str) -> dict:
    """Fuerza a Claude a usar una herramienta para estructurar la salida."""

    herramienta_extraccion = {
        "name": "guardar_datos_extraidos",
        "description": "Guarda los datos estructurados extraídos del texto.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string"},
                "fecha": {"type": "string", "description": "Formato YYYY-MM-DD"},
                "importe": {"type": "number"},
                "moneda": {"type": "string", "enum": ["EUR", "USD", "GBP"]},
                "concepto": {"type": "string"}
            },
            "required": ["nombre", "importe"]
        }
    }

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        tools=[herramienta_extraccion],
        tool_choice={"type": "tool", "name": "guardar_datos_extraidos"},  # forzar herramienta
        messages=[{
            "role": "user",
            "content": f"Extrae los datos de esta factura:\n{texto}"
        }]
    )

    # La respuesta siempre será un tool_use (forzado)
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input


# Ejemplo
factura = "Factura a nombre de Empresa ABC SL, fecha 15/03/2024, total 1.234,50€, concepto: Servicios consultoría"
datos = extraer_datos_estructurados(factura)
print(json.dumps(datos, ensure_ascii=False, indent=2))
```

---

## 7. Extensiones sugeridas

- **Computer use tools**: herramientas que controlan el navegador o el sistema operativo (ver Bloque 9)
- **MCP (Model Context Protocol)**: estándar para registrar herramientas de forma dinámica (ver Bloque 9)
- **Streaming con tool use**: procesar tool calls mientras llegan con `stream=True`
- **Historial de herramientas**: guardar todas las llamadas a herramientas para auditoría

---

**Anterior:** [11 — Tokenización](./11-tokenizacion.md) · **Siguiente bloque:** [Bloque 3 — APIs de IA](../apis/)
