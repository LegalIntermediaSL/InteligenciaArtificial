# 04 — Agentes de IA

> **Bloque:** LLMs · **Nivel:** Intermedio-Avanzado · **Tiempo estimado:** 35 min

---

## Índice

1. [¿Qué es un agente de IA?](#1-qué-es-un-agente-de-ia)
2. [El patrón ReAct](#2-el-patrón-react)
3. [Tool Use con Claude](#3-tool-use-con-claude)
4. [Agente con múltiples herramientas](#4-agente-con-múltiples-herramientas)
5. [Bucle agéntico completo](#5-bucle-agéntico-completo)
6. [Patrones de agentes avanzados](#6-patrones-de-agentes-avanzados)
7. [Seguridad y límites](#7-seguridad-y-límites)
8. [Resumen](#8-resumen)

---

## 1. ¿Qué es un agente de IA?

Un **agente de IA** es un sistema que usa un LLM como motor de razonamiento para planificar y ejecutar acciones de forma autónoma, con acceso a herramientas externas, para alcanzar un objetivo.

La diferencia con un chatbot simple:

| Chatbot | Agente |
|---|---|
| Recibe una pregunta → genera texto | Recibe un objetivo → planifica pasos → ejecuta acciones → obtiene resultado |
| Sin acceso a herramientas | Puede buscar en internet, ejecutar código, consultar bases de datos... |
| Una interacción | Múltiples iteraciones hasta completar la tarea |
| Salida: texto | Salida: resultado concreto (fichero, dato, acción realizada) |

### El ciclo agéntico

```
┌─────────────────────────────────────────────────────┐
│                      AGENTE                         │
│                                                     │
│  Objetivo                                           │
│     ↓                                               │
│  [LLM] Razona → ¿Qué herramienta necesito?         │
│     ↓                                               │
│  [Tool] Ejecuta la herramienta                      │
│     ↓                                               │
│  [LLM] Observa el resultado → ¿He terminado?        │
│     ↓                                               │
│  Si NO → vuelve a razonar                           │
│  Si SÍ → devuelve resultado final                   │
└─────────────────────────────────────────────────────┘
```

---

## 2. El patrón ReAct

**ReAct** (Reason + Act) es el patrón fundamental de los agentes. El modelo alterna entre:

- **Thought:** razonamiento interno sobre qué hacer
- **Action:** llamada a una herramienta
- **Observation:** resultado de la herramienta

```
Objetivo: "¿Cuántos empleados tiene Anthropic y cuándo fue fundada?"

Thought: Necesito buscar información sobre Anthropic.
Action: buscar_web("Anthropic empresa fundación empleados")
Observation: Anthropic fue fundada en 2021 por Dario Amodei...

Thought: Tengo la información de fundación. Necesito el número de empleados.
Action: buscar_web("Anthropic número de empleados 2024")
Observation: Anthropic tiene aproximadamente 500-800 empleados...

Thought: Ya tengo toda la información necesaria para responder.
Answer: Anthropic fue fundada en 2021 y tiene entre 500 y 800 empleados.
```

---

## 3. Tool Use con Claude

Claude implementa el uso de herramientas de forma nativa. El flujo tiene dos llamadas a la API:

### Paso 1: Claude decide qué herramienta usar

```python
import anthropic
import json

client = anthropic.Anthropic()

# Definir herramientas disponibles
tools = [
    {
        "name": "buscar_empresa",
        "description": "Busca información sobre una empresa en la base de datos",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre de la empresa"},
                "campo": {
                    "type": "string",
                    "enum": ["fundacion", "empleados", "sede", "productos"],
                    "description": "Campo de información a buscar"
                }
            },
            "required": ["nombre", "campo"]
        }
    },
    {
        "name": "calcular",
        "description": "Realiza cálculos matemáticos",
        "input_schema": {
            "type": "object",
            "properties": {
                "expresion": {"type": "string", "description": "Expresión matemática a evaluar"}
            },
            "required": ["expresion"]
        }
    }
]

# Primera llamada: Claude decide qué herramienta necesita
respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "¿En qué año fue fundada Anthropic?"
    }]
)

print(f"Stop reason: {respuesta.stop_reason}")
# → "tool_use"

for bloque in respuesta.content:
    if bloque.type == "tool_use":
        print(f"Herramienta: {bloque.name}")
        print(f"Argumentos: {bloque.input}")
```

### Paso 2: Ejecutar la herramienta y devolver el resultado

```python
# Implementaciones reales de las herramientas
def buscar_empresa(nombre: str, campo: str) -> dict:
    # En producción, esto consultaría una BD real o una API
    datos = {
        "Anthropic": {
            "fundacion": "2021",
            "empleados": "~700",
            "sede": "San Francisco, California",
            "productos": "Claude (Opus, Sonnet, Haiku)"
        }
    }
    empresa = datos.get(nombre, {})
    return {"empresa": nombre, "campo": campo, "valor": empresa.get(campo, "No encontrado")}

def calcular(expresion: str) -> dict:
    try:
        # NOTA: en producción, usa un parser seguro en lugar de eval()
        resultado = eval(expresion, {"__builtins__": {}}, {})
        return {"expresion": expresion, "resultado": resultado}
    except Exception as e:
        return {"error": str(e)}

# Mapa de herramientas disponibles
herramientas_disponibles = {
    "buscar_empresa": buscar_empresa,
    "calcular": calcular,
}

# Ejecutar la herramienta solicitada
tool_use = next(b for b in respuesta.content if b.type == "tool_use")
funcion = herramientas_disponibles[tool_use.name]
resultado = funcion(**tool_use.input)

print(f"Resultado de la herramienta: {resultado}")

# Segunda llamada: devolver resultado a Claude
respuesta_final = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "¿En qué año fue fundada Anthropic?"},
        {"role": "assistant", "content": respuesta.content},
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(resultado, ensure_ascii=False)
            }]
        }
    ]
)

print(f"\nRespuesta final: {respuesta_final.content[0].text}")
```

---

## 4. Agente con múltiples herramientas

Un agente real necesita un conjunto de herramientas variadas:

```python
import anthropic
import json
import os
import subprocess
from datetime import datetime

client = anthropic.Anthropic()

# ── Definición de herramientas ────────────────────────────────────────────────
TOOLS = [
    {
        "name": "leer_fichero",
        "description": "Lee el contenido de un fichero de texto",
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {"type": "string", "description": "Ruta del fichero"}
            },
            "required": ["ruta"]
        }
    },
    {
        "name": "escribir_fichero",
        "description": "Escribe contenido en un fichero",
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {"type": "string"},
                "contenido": {"type": "string"}
            },
            "required": ["ruta", "contenido"]
        }
    },
    {
        "name": "ejecutar_python",
        "description": "Ejecuta un fragmento de código Python y devuelve el output",
        "input_schema": {
            "type": "object",
            "properties": {
                "codigo": {"type": "string", "description": "Código Python a ejecutar"}
            },
            "required": ["codigo"]
        }
    },
    {
        "name": "obtener_fecha_hora",
        "description": "Devuelve la fecha y hora actuales",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# ── Implementaciones ──────────────────────────────────────────────────────────
def leer_fichero(ruta: str) -> dict:
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return {"ruta": ruta, "contenido": f.read()}
    except FileNotFoundError:
        return {"error": f"Fichero no encontrado: {ruta}"}

def escribir_fichero(ruta: str, contenido: str) -> dict:
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    return {"ruta": ruta, "bytes_escritos": len(contenido.encode())}

def ejecutar_python(codigo: str) -> dict:
    try:
        result = subprocess.run(
            ["python", "-c", codigo],
            capture_output=True, text=True, timeout=10
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": "Timeout: el código tardó más de 10 segundos"}

def obtener_fecha_hora() -> dict:
    return {"datetime": datetime.now().isoformat(), "timestamp": datetime.now().timestamp()}

EJECUTORES = {
    "leer_fichero": leer_fichero,
    "escribir_fichero": escribir_fichero,
    "ejecutar_python": ejecutar_python,
    "obtener_fecha_hora": lambda: obtener_fecha_hora(),
}
```

---

## 5. Bucle agéntico completo

```python
def ejecutar_agente(objetivo: str, max_pasos: int = 10, verbose: bool = True) -> str:
    """
    Ejecuta un agente que itera hasta completar el objetivo o alcanzar max_pasos.
    """
    mensajes = [{"role": "user", "content": objetivo}]
    paso = 0

    if verbose:
        print(f"🎯 Objetivo: {objetivo}")
        print("=" * 60)

    while paso < max_pasos:
        paso += 1
        if verbose:
            print(f"\n[Paso {paso}]")

        # Llamada al LLM
        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=TOOLS,
            system="""Eres un agente autónomo. Para completar el objetivo dado:
1. Razona sobre qué herramienta necesitas
2. Úsala
3. Analiza el resultado
4. Repite hasta tener la respuesta final
Cuando hayas completado el objetivo, responde directamente sin usar más herramientas.""",
            messages=mensajes
        )

        # Añadir respuesta del asistente al historial
        mensajes.append({"role": "assistant", "content": respuesta.content})

        # ¿Terminó? (sin más llamadas a herramientas)
        if respuesta.stop_reason == "end_turn":
            texto_final = next(
                (b.text for b in respuesta.content if hasattr(b, "text")), ""
            )
            if verbose:
                print(f"✅ Completado:\n{texto_final}")
            return texto_final

        # Procesar llamadas a herramientas
        resultados_tools = []
        for bloque in respuesta.content:
            if bloque.type != "tool_use":
                continue

            if verbose:
                print(f"  🔧 Herramienta: {bloque.name}({bloque.input})")

            ejecutor = EJECUTORES.get(bloque.name)
            if ejecutor:
                resultado = ejecutor(**bloque.input) if bloque.input else ejecutor()
            else:
                resultado = {"error": f"Herramienta desconocida: {bloque.name}"}

            if verbose:
                print(f"  📦 Resultado: {str(resultado)[:100]}...")

            resultados_tools.append({
                "type": "tool_result",
                "tool_use_id": bloque.id,
                "content": json.dumps(resultado, ensure_ascii=False)
            })

        mensajes.append({"role": "user", "content": resultados_tools})

    return "⚠️ Se alcanzó el límite de pasos sin completar el objetivo."


# Ejemplo de uso
resultado = ejecutar_agente(
    "¿Qué fecha y hora es ahora? Escribe un fichero llamado 'timestamp.txt' con esa información."
)
```

---

## 6. Patrones de agentes avanzados

### Agente con memoria

```python
from collections import deque

class AgenteConMemoria:
    """Agente que mantiene un resumen de las acciones previas."""

    def __init__(self, max_memoria: int = 5):
        self.memoria = deque(maxlen=max_memoria)
        self.client = anthropic.Anthropic()

    def recordar(self, accion: str, resultado: str):
        self.memoria.append(f"- {accion}: {resultado[:100]}")

    def contexto_memoria(self) -> str:
        if not self.memoria:
            return ""
        return "Acciones previas:\n" + "\n".join(self.memoria)

    def ejecutar(self, objetivo: str) -> str:
        system = f"""Eres un agente con memoria de sesión.
{self.contexto_memoria()}

Objetivo actual: {objetivo}"""

        # ... resto del bucle agéntico
        return "resultado"
```

### Agentes en paralelo (multi-agente)

```python
from concurrent.futures import ThreadPoolExecutor

def agente_especialista(subtarea: str, especialidad: str) -> str:
    """Agente especializado en un dominio concreto."""
    system = f"Eres un experto en {especialidad}. Resuelve la tarea con precisión."
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": subtarea}]
    )
    return r.content[0].text

def orquestador(tarea_compleja: str) -> str:
    """Descompone una tarea y la distribuye entre agentes especializados."""
    # 1. Descomponer la tarea
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content":
            f"Descompone esta tarea en 3 subtareas independientes en JSON: {tarea_compleja}\n"
            f"Formato: [{{\"subtarea\": \"...\", \"especialidad\": \"...\"}}]"
        }]
    )
    subtareas = json.loads(r.content[0].text)

    # 2. Ejecutar subtareas en paralelo
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuros = [
            executor.submit(agente_especialista, st["subtarea"], st["especialidad"])
            for st in subtareas
        ]
        resultados = [f.result() for f in futuros]

    # 3. Sintetizar resultados
    r_final = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content":
            f"Sintetiza estos resultados de agentes especializados en una respuesta coherente:\n\n" +
            "\n\n".join([f"[{st['especialidad']}]: {res}"
                         for st, res in zip(subtareas, resultados)])
        }]
    )
    return r_final.content[0].text
```

---

## 7. Seguridad y límites

Los agentes tienen acceso a herramientas potencialmente peligrosas. Buenas prácticas:

```python
import re

# 1. Validar entradas antes de ejecutar
RUTAS_PERMITIDAS = ["/tmp/agente/", "./workspace/"]

def leer_fichero_seguro(ruta: str) -> dict:
    """Solo permite leer ficheros en rutas autorizadas."""
    ruta_abs = os.path.abspath(ruta)
    if not any(ruta_abs.startswith(p) for p in RUTAS_PERMITIDAS):
        return {"error": f"Ruta no permitida: {ruta}. Solo se permite: {RUTAS_PERMITIDAS}"}
    return leer_fichero(ruta)

# 2. Sandboxing para ejecución de código
OPERACIONES_PROHIBIDAS = ["import os", "import subprocess", "open(", "__import__"]

def ejecutar_python_seguro(codigo: str) -> dict:
    """Rechaza código con operaciones peligrosas."""
    for op in OPERACIONES_PROHIBIDAS:
        if op in codigo:
            return {"error": f"Operación no permitida: {op}"}
    return ejecutar_python(codigo)

# 3. Límite de iteraciones y timeout
# 4. Logging de todas las acciones del agente
# 5. Aprobación humana para acciones irreversibles (human-in-the-loop)
```

**Principios de seguridad en agentes:**

| Principio | Descripción |
|---|---|
| **Mínimo privilegio** | Las herramientas solo pueden hacer lo estrictamente necesario |
| **Sandboxing** | Ejecutar en entornos aislados (Docker, VMs) |
| **Human-in-the-loop** | Pedir confirmación antes de acciones destructivas |
| **Auditoría** | Registrar todas las acciones del agente |
| **Límites explícitos** | max_pasos, timeout, límite de coste por sesión |

---

## 8. Resumen

| Concepto | Descripción |
|---|---|
| Agente | LLM + herramientas + bucle de ejecución autónomo |
| ReAct | Patrón Thought → Action → Observation |
| Tool Use | Mecanismo nativo de Claude para llamar funciones |
| Bucle agéntico | Iterar hasta completar el objetivo o alcanzar el límite |
| Multi-agente | Varios agentes especializados en paralelo |
| Human-in-the-loop | Confirmación humana en pasos críticos |

---

**Anterior:** [03 — Fine-tuning vs RAG](./03-finetuning-vs-rag.md) · **Siguiente:** [05 — RAG con ChromaDB](./05-rag-chromadb.md)
