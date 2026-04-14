# 07 — Tipos y Arquitecturas de Agentes de IA

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Taxonomía de agentes](#1-taxonomía-de-agentes)
2. [Por capacidad de razonamiento](#2-por-capacidad-de-razonamiento)
3. [Por arquitectura](#3-por-arquitectura)
4. [Patrones de planificación](#4-patrones-de-planificación)
5. [Sistemas de memoria](#5-sistemas-de-memoria)
6. [Tipos por dominio](#6-tipos-por-dominio)
7. [Frameworks de agentes](#7-frameworks-de-agentes)
8. [Evaluación de agentes](#8-evaluación-de-agentes)
9. [Guía de selección](#9-guía-de-selección)

---

## 1. Taxonomía de agentes

Los agentes de IA se pueden clasificar a lo largo de varios ejes ortogonales. No son mutuamente excluyentes: un agente real suele combinar características de varias categorías.

```
                      TAXONOMÍA DE AGENTES DE IA
                      ══════════════════════════

  Por razonamiento     Por arquitectura     Por planificación
  ─────────────────    ─────────────────    ─────────────────
  • Reactivo           • Agente único       • ReAct
  • Deliberativo       • Pipeline           • Plan & Execute
  • BDI                • Multi-agente       • ReWOO
  • Reflexivo          • Jerárquico         • Reflexion
  • De aprendizaje     • Swarm              • LATS

  Por memoria          Por dominio
  ─────────────────    ─────────────────
  • Sin estado         • Task agent
  • Memoria corta      • Research agent
  • Memoria larga      • Code agent
  • Episódica          • Data agent
  • Semántica          • Conversational
```

---

## 2. Por capacidad de razonamiento

### 2.1 Agentes reactivos

Responden **directamente** al estímulo sin mantener estado interno ni planificar.

```
Estímulo → [Regla fija] → Acción
```

**Características:**
- Sin memoria entre interacciones
- Respuesta inmediata y predecible
- Limitados a situaciones vistas en entrenamiento

**Cuándo usarlos:** tareas bien definidas y repetitivas (clasificación, extracción de datos simples, respuestas FAQ).

```python
import anthropic

client = anthropic.Anthropic()

def agente_reactivo(entrada: str) -> str:
    """Agente reactivo: responde al estímulo sin historial ni planificación."""
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="Clasifica el sentimiento del texto: POSITIVO, NEGATIVO o NEUTRO. Solo responde con la etiqueta.",
        messages=[{"role": "user", "content": entrada}]
    )
    return r.content[0].text.strip()

# Cada llamada es independiente — sin contexto previo
print(agente_reactivo("El servicio fue excelente, muy recomendable"))  # POSITIVO
print(agente_reactivo("Tardaron mucho y el producto llegó roto"))      # NEGATIVO
```

---

### 2.2 Agentes deliberativos

Mantienen un **modelo interno del mundo** y razonan sobre él antes de actuar.

```
Estímulo → [Modelo del mundo] → [Razonamiento] → Plan → Acción
```

**Características:**
- Mantienen estado sobre el entorno
- Planifican secuencias de acciones
- Pueden simular consecuencias antes de actuar

```python
class AgenteDeliberativo:
    """Agente que mantiene un modelo del mundo y planifica explícitamente."""

    def __init__(self):
        self.cliente = anthropic.Anthropic()
        self.modelo_mundo = {}  # Representación interna del estado

    def actualizar_estado(self, clave: str, valor):
        self.modelo_mundo[clave] = valor

    def deliberar(self, objetivo: str) -> str:
        """Genera un plan explícito antes de actuar."""
        estado_actual = "\n".join(
            f"  - {k}: {v}" for k, v in self.modelo_mundo.items()
        ) or "  (sin información previa)"

        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
Estado actual del mundo:
{estado_actual}

Objetivo: {objetivo}

1. Analiza el estado actual.
2. Genera un plan paso a paso para alcanzar el objetivo.
3. Identifica posibles obstáculos.
4. Ejecuta el plan.
"""}]
        )
        return r.content[0].text

# Ejemplo
agente = AgenteDeliberativo()
agente.actualizar_estado("presupuesto", "5.000€")
agente.actualizar_estado("tiempo_disponible", "2 semanas")
agente.actualizar_estado("equipo", "2 desarrolladores")

plan = agente.deliberar("Lanzar una MVP de chatbot de atención al cliente")
print(plan)
```

---

### 2.3 Agentes BDI (Beliefs-Desires-Intentions)

Modelo clásico de la IA simbólica adaptado a LLMs:

| Componente | Descripción | Ejemplo |
|-----------|-------------|---------|
| **Beliefs** (Creencias) | Lo que el agente sabe sobre el mundo | "El servidor está caído", "El usuario es un experto" |
| **Desires** (Deseos) | Los objetivos que quiere alcanzar | "Resolver el ticket", "Maximizar satisfacción del usuario" |
| **Intentions** (Intenciones) | El plan concreto que ha comprometido ejecutar | "Enviar un email → escalar al equipo → notificar al usuario" |

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class EstadoBDI:
    creencias: dict[str, Any] = field(default_factory=dict)
    deseos: list[str] = field(default_factory=list)
    intenciones: list[str] = field(default_factory=list)

class AgenteBDI:
    """Agente que razona explícitamente sobre creencias, deseos e intenciones."""

    def __init__(self):
        self.cliente = anthropic.Anthropic()
        self.estado = EstadoBDI()

    def percibir(self, observacion: str):
        """Actualiza las creencias a partir de una nueva observación."""
        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
Observación nueva: {observacion}

Creencias actuales: {self.estado.creencias}

Extrae los hechos relevantes de la observación en JSON:
{{"nuevas_creencias": {{"clave": "valor"}}, "creencias_a_eliminar": ["clave1"]}}
"""}]
        )
        import json
        try:
            cambios = json.loads(r.content[0].text)
            self.estado.creencias.update(cambios.get("nuevas_creencias", {}))
            for k in cambios.get("creencias_a_eliminar", []):
                self.estado.creencias.pop(k, None)
        except json.JSONDecodeError:
            pass

    def deliberar_sobre_deseos(self) -> list[str]:
        """Filtra deseos alcanzables dados las creencias actuales."""
        if not self.estado.deseos:
            return []

        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
Creencias: {self.estado.creencias}
Deseos pendientes: {self.estado.deseos}

¿Cuáles son alcanzables dado el estado actual? Devuelve solo la lista JSON de deseos seleccionados.
"""}]
        )
        import json
        try:
            return json.loads(r.content[0].text)
        except Exception:
            return self.estado.deseos[:1]

    def formar_intenciones(self, deseos_seleccionados: list[str]) -> list[str]:
        """Convierte deseos en un plan de acciones concretas."""
        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
Deseos a satisfacer: {deseos_seleccionados}
Recursos disponibles (creencias): {self.estado.creencias}

Genera un plan de acciones concretas y secuenciales. Devuelve una lista JSON de strings.
"""}]
        )
        import json
        try:
            return json.loads(r.content[0].text)
        except Exception:
            return ["Ejecutar plan general"]

    def ciclo_bdi(self, nueva_observacion: str) -> list[str]:
        """Ejecuta un ciclo completo: percibir → deliberar → intencionar."""
        self.percibir(nueva_observacion)
        deseos_viables = self.deliberar_sobre_deseos()
        self.estado.intenciones = self.formar_intenciones(deseos_viables)
        return self.estado.intenciones
```

---

### 2.4 Agente reflexivo (Self-Critique)

El agente genera una respuesta y luego la **critica y mejora** antes de entregarla.

```python
def agente_reflexivo(tarea: str, iteraciones: int = 2) -> str:
    """Agente que revisa y mejora su propia respuesta."""
    cliente = anthropic.Anthropic()

    # Generar respuesta inicial
    r = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": tarea}]
    )
    respuesta = r.content[0].text
    print(f"[v1] {respuesta[:100]}...\n")

    for i in range(iteraciones):
        # Autocrítica
        critica = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
Tarea original: {tarea}

Mi respuesta actual:
{respuesta}

Identifica en 3 puntos concretos qué mejorarías:
1. ¿Hay errores factuales?
2. ¿Falta claridad o profundidad?
3. ¿Se puede mejorar la estructura?
"""}]
        )

        # Mejorar basándose en la crítica
        mejorada = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
Tarea: {tarea}
Respuesta anterior: {respuesta}
Crítica: {critica.content[0].text}

Genera una versión mejorada que resuelva todos los problemas identificados.
"""}]
        )
        respuesta = mejorada.content[0].text
        print(f"[v{i+2}] {respuesta[:100]}...\n")

    return respuesta

# Uso
respuesta_final = agente_reflexivo(
    "Explica las ventajas y limitaciones de los agentes de IA en producción",
    iteraciones=2
)
```

---

## 3. Por arquitectura

### 3.1 Agente único

La arquitectura más simple: un solo LLM con acceso a herramientas.

```
Usuario → [Agente] → Herramientas → [Agente] → Respuesta
```

**Ventajas:** simplicidad, fácil debugging, bajo coste  
**Limitaciones:** un solo punto de fallo, contexto limitado, sin especialización

---

### 3.2 Pipeline secuencial

Los agentes se encadenan: la salida de uno es la entrada del siguiente.

```
Input → [Agente 1: Extracción] → [Agente 2: Análisis] → [Agente 3: Redacción] → Output
```

```python
def pipeline_agentes(documento: str) -> dict:
    """Pipeline de 3 agentes en secuencia."""
    cliente = anthropic.Anthropic()

    # Agente 1: Extracción de hechos
    r1 = cliente.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="Extrae los hechos clave del documento en formato JSON: {\"hechos\": [...]}",
        messages=[{"role": "user", "content": documento}]
    )

    # Agente 2: Análisis de riesgos (recibe la salida del agente 1)
    r2 = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="Analiza los hechos y evalúa riesgos. JSON: {\"riesgos\": [...], \"nivel\": \"alto|medio|bajo\"}",
        messages=[{"role": "user", "content": f"Hechos: {r1.content[0].text}"}]
    )

    # Agente 3: Redacción del informe final
    r3 = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="Redacta un informe ejecutivo claro y conciso.",
        messages=[{"role": "user", "content": f"Hechos: {r1.content[0].text}\nRiesgos: {r2.content[0].text}"}]
    )

    return {
        "hechos": r1.content[0].text,
        "analisis_riesgos": r2.content[0].text,
        "informe": r3.content[0].text
    }
```

---

### 3.3 Arquitectura jerárquica (Orchestrator-Subagents)

Un **orquestador** descompone el objetivo y delega en **subagentes especializados**.

```
                    [Orquestador]
                   /      |      \
          [Subagente 1] [Subagente 2] [Subagente 3]
          Investigación   Análisis     Redacción
```

```python
from concurrent.futures import ThreadPoolExecutor
import json

def subagente(tarea: str, rol: str, modelo: str = "claude-haiku-4-5-20251001") -> str:
    """Subagente especializado en un rol concreto."""
    cliente = anthropic.Anthropic()
    r = cliente.messages.create(
        model=modelo,
        max_tokens=1024,
        system=f"Eres un experto en {rol}. Completa la tarea con precisión y sé conciso.",
        messages=[{"role": "user", "content": tarea}]
    )
    return r.content[0].text

def orquestador(objetivo: str) -> str:
    """Orquestador que descompone y delega."""
    cliente = anthropic.Anthropic()

    # Paso 1: Descomponer el objetivo en subtareas
    r_plan = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"""
Descompón este objetivo en 3 subtareas independientes para agentes especializados.
Objetivo: {objetivo}
Responde SOLO con JSON: [{{"subtarea": "...", "especialidad": "...", "prioridad": 1}}]
"""}]
    )

    try:
        subtareas = json.loads(r_plan.content[0].text)
    except Exception:
        return "Error al descomponer el objetivo"

    print(f"Plan: {len(subtareas)} subtareas identificadas")
    for st in subtareas:
        print(f"  [{st.get('prioridad', '?')}] {st['especialidad']}: {st['subtarea'][:60]}...")

    # Paso 2: Ejecutar subtareas en paralelo
    with ThreadPoolExecutor(max_workers=len(subtareas)) as executor:
        futuros = {
            executor.submit(subagente, st["subtarea"], st["especialidad"]): st
            for st in subtareas
        }
        resultados = {}
        for futuro, st in futuros.items():
            resultados[st["especialidad"]] = futuro.result()

    # Paso 3: Sintetizar resultados
    contexto_resultados = "\n\n".join(
        f"=== {esp} ===\n{res}" for esp, res in resultados.items()
    )
    r_final = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"""
Objetivo original: {objetivo}

Resultados de los subagentes:
{contexto_resultados}

Sintetiza todos los resultados en una respuesta coherente y completa.
"""}]
    )
    return r_final.content[0].text


# Ejemplo
resultado = orquestador(
    "Analiza el impacto de los LLMs en el sector legal: oportunidades, riesgos y regulación"
)
print(resultado)
```

---

### 3.4 Arquitectura Swarm

Muchos agentes simples que colaboran sin un coordinador central. Cada agente toma decisiones locales y puede **traspasar el control** a otro agente.

```
[Agente Triage] ──traspasa──→ [Agente Ventas]
                └──traspasa──→ [Agente Soporte]
                               └──traspasa──→ [Agente Escalado]
```

```python
from typing import Callable

@dataclass
class AgenteSwarm:
    nombre: str
    instrucciones: str
    herramientas: list = field(default_factory=list)
    funciones_traspaso: dict = field(default_factory=dict)

def ejecutar_swarm(
    agente_inicial: AgenteSwarm,
    mensaje_usuario: str,
    agentes: dict[str, AgenteSwarm],
    max_turnos: int = 10
) -> str:
    """Bucle Swarm: los agentes pueden traspasarse el control."""
    cliente = anthropic.Anthropic()
    agente_actual = agente_inicial
    mensajes = [{"role": "user", "content": mensaje_usuario}]
    turno = 0

    while turno < max_turnos:
        turno += 1
        print(f"\n[Turno {turno}] Agente activo: {agente_actual.nombre}")

        # Preparar herramientas de traspaso como tool use
        tools_traspaso = [
            {
                "name": f"traspasar_a_{nombre}",
                "description": f"Traspasa la conversación al agente de {nombre}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "razon": {"type": "string", "description": "Razón del traspaso"}
                    },
                    "required": ["razon"]
                }
            }
            for nombre in agente_actual.funciones_traspaso.keys()
        ]

        r = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=agente_actual.instrucciones,
            tools=tools_traspaso if tools_traspaso else anthropic.NOT_GIVEN,
            messages=mensajes
        )

        mensajes.append({"role": "assistant", "content": r.content})

        if r.stop_reason == "end_turn":
            texto = next((b.text for b in r.content if hasattr(b, "text")), "")
            print(f"✅ Respuesta final de {agente_actual.nombre}")
            return texto

        # Procesar traspaso
        for bloque in r.content:
            if bloque.type != "tool_use":
                continue

            nombre_destino = bloque.name.replace("traspasar_a_", "")
            print(f"  → Traspasando a: {nombre_destino} | Razón: {bloque.input.get('razon', '')}")

            agente_actual = agentes.get(nombre_destino, agente_actual)

            mensajes.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": f"Traspaso a {nombre_destino} completado"
                }]
            })
            break

    return "Límite de turnos alcanzado"
```

---

## 4. Patrones de planificación

### 4.1 ReAct — ya visto en 04-agentes-ia.md

Thought → Action → Observation (interleaved).

### 4.2 Plan & Execute

El agente genera **primero el plan completo** y luego ejecuta cada paso.

```
Objetivo
  ↓
[PLANNER] → Plan: [paso1, paso2, paso3, ...]
  ↓
[EXECUTOR] → ejecuta paso1
  ↓
[EXECUTOR] → ejecuta paso2
  ↓
[RESPUESTA FINAL]
```

**Ventaja sobre ReAct:** el plan es revisable por un humano antes de ejecutar.

```python
def plan_and_execute(objetivo: str, herramientas: dict) -> str:
    cliente = anthropic.Anthropic()

    # FASE 1: Planificación
    r_plan = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""
Objetivo: {objetivo}
Herramientas disponibles: {list(herramientas.keys())}

Genera un plan detallado en JSON:
{{
  "pasos": [
    {{"paso": 1, "accion": "nombre_herramienta", "argumentos": {{}}, "descripcion": "..."}}
  ],
  "estimacion_pasos": N
}}
"""}]
    )

    import json
    plan = json.loads(r_plan.content[0].text)
    print(f"📋 Plan generado: {len(plan['pasos'])} pasos")
    for p in plan["pasos"]:
        print(f"  Paso {p['paso']}: {p['descripcion']}")

    # FASE 2: Ejecución
    resultados = []
    for paso in plan["pasos"]:
        herramienta = herramientas.get(paso["accion"])
        if herramienta:
            resultado = herramienta(**paso.get("argumentos", {}))
            resultados.append({"paso": paso["paso"], "resultado": resultado})
            print(f"  ✓ Paso {paso['paso']} completado")
        else:
            print(f"  ⚠ Herramienta no encontrada: {paso['accion']}")

    # FASE 3: Síntesis
    r_final = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""
Objetivo: {objetivo}
Resultados de ejecución: {json.dumps(resultados, ensure_ascii=False)}
Genera la respuesta final sintetizando todos los resultados.
"""}]
    )
    return r_final.content[0].text
```

---

### 4.3 Reflexion

El agente itera con un **crítico externo** que evalúa y solicita mejoras.

```
[Actor] genera respuesta
    ↓
[Crítico] evalúa y da feedback
    ↓
[Actor] mejora basándose en feedback
    ↓
(repite hasta que el crítico aprueba o se alcanza el límite)
```

```python
def patron_reflexion(tarea: str, max_iteraciones: int = 3) -> str:
    cliente = anthropic.Anthropic()
    historial_actor = [{"role": "user", "content": tarea}]

    for iteracion in range(max_iteraciones):
        # ACTOR: genera o mejora la respuesta
        r_actor = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system="Eres el Actor. Genera la mejor respuesta posible a la tarea.",
            messages=historial_actor
        )
        respuesta = r_actor.content[0].text
        print(f"\n[Iteración {iteracion+1}] Actor generó: {respuesta[:80]}...")

        # CRÍTICO: evalúa la respuesta
        r_critico = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system="""Eres el Crítico. Evalúa la respuesta en 3 dimensiones:
1. Corrección factual (0-10)
2. Completitud (0-10)
3. Claridad (0-10)

Si la puntuación media es ≥ 8, responde: APROBADO
Si no, responde: MEJORAR: [instrucciones específicas]""",
            messages=[
                {"role": "user", "content": f"Tarea: {tarea}\n\nRespuesta del Actor:\n{respuesta}"}
            ]
        )
        feedback = r_critico.content[0].text
        print(f"Crítico: {feedback[:100]}...")

        if "APROBADO" in feedback:
            print(f"✅ Aprobado en iteración {iteracion+1}")
            return respuesta

        # El actor recibe el feedback para la siguiente iteración
        historial_actor.append({"role": "assistant", "content": respuesta})
        historial_actor.append({"role": "user", "content": f"Feedback del revisor:\n{feedback}\n\nMejora tu respuesta."})

    print(f"⚠ Límite de iteraciones ({max_iteraciones}) alcanzado")
    return respuesta
```

---

## 5. Sistemas de memoria

Los agentes pueden tener distintos tipos de memoria, inspirados en la memoria humana:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIPOS DE MEMORIA EN AGENTES                  │
│                                                                 │
│  Memoria de trabajo    Memoria episódica    Memoria semántica   │
│  ─────────────────     ──────────────────   ─────────────────   │
│  El contexto actual    Eventos pasados      Hechos del mundo    │
│  (ventana del LLM)     (vector store)       (knowledge base)    │
│  Dura: 1 sesión        Dura: persistente    Dura: persistente   │
│                                                                 │
│  Memoria procedural    Memoria epistemológica                   │
│  ──────────────────    ──────────────────────                   │
│  Cómo hacer cosas      Lo que el agente sabe que no sabe        │
│  (herramientas)        (incertidumbre explícita)                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.1 Memoria de trabajo (In-context)

La más simple: la ventana de contexto del LLM.

```python
class AgenteMemoraCorta:
    """Memoria en el contexto del LLM — se pierde al cerrar la sesión."""

    def __init__(self, max_mensajes: int = 20):
        self.cliente = anthropic.Anthropic()
        self.historial = []
        self.max_mensajes = max_mensajes

    def chatear(self, mensaje: str) -> str:
        self.historial.append({"role": "user", "content": mensaje})

        # Truncar si excede el límite
        if len(self.historial) > self.max_mensajes:
            # Siempre conservar los primeros mensajes (contexto inicial)
            self.historial = self.historial[-self.max_mensajes:]

        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=self.historial
        )
        respuesta = r.content[0].text
        self.historial.append({"role": "assistant", "content": respuesta})
        return respuesta
```

### 5.2 Memoria episódica (Vector Store)

Almacena eventos pasados de forma persistente con búsqueda semántica.

```python
import json
from datetime import datetime
from pathlib import Path

class MemoriaEpisodica:
    """
    Memoria persistente basada en archivos JSONL.
    En producción, usar ChromaDB, Pinecone o similar.
    """

    def __init__(self, ruta: str = "memoria_agente.jsonl"):
        self.ruta = Path(ruta)
        self.cliente = anthropic.Anthropic()

    def guardar_episodio(self, descripcion: str, datos: dict):
        """Guarda un episodio con su embedding semántico."""
        episodio = {
            "timestamp": datetime.now().isoformat(),
            "descripcion": descripcion,
            "datos": datos
        }
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(episodio, ensure_ascii=False) + "\n")

    def recuperar_relevantes(self, consulta: str, top_k: int = 3) -> list[dict]:
        """Recupera los episodios más relevantes (versión simplificada)."""
        if not self.ruta.exists():
            return []

        episodios = []
        with open(self.ruta, encoding="utf-8") as f:
            for linea in f:
                try:
                    episodios.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue

        if not episodios:
            return []

        # Usar el LLM para seleccionar los más relevantes
        descripciones = [f"[{i}] {ep['descripcion']}" for i, ep in enumerate(episodios)]
        r = self.cliente.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""
Consulta actual: {consulta}
Episodios disponibles:
{chr(10).join(descripciones)}

Devuelve los índices de los {top_k} episodios más relevantes como JSON: [{{"indice": N}}]
"""}]
        )
        try:
            indices = json.loads(r.content[0].text)
            return [episodios[idx["indice"]] for idx in indices if idx["indice"] < len(episodios)]
        except Exception:
            return episodios[-top_k:]

    def usar_con_agente(self, tarea: str) -> str:
        """Agente que usa la memoria episódica para contextualizar respuestas."""
        episodios_relevantes = self.recuperar_relevantes(tarea)
        contexto_memoria = ""
        if episodios_relevantes:
            contexto_memoria = "Experiencias previas relevantes:\n" + "\n".join(
                f"- [{ep['timestamp'][:10]}] {ep['descripcion']}: {ep['datos']}"
                for ep in episodios_relevantes
            )

        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=f"Eres un asistente con memoria de experiencias previas.\n{contexto_memoria}",
            messages=[{"role": "user", "content": tarea}]
        )
        respuesta = r.content[0].text

        # Guardar esta interacción en la memoria
        self.guardar_episodio(f"Tarea: {tarea[:50]}...", {"respuesta": respuesta[:200]})
        return respuesta
```

### 5.3 Resumen de memoria comprimida

Para conversaciones largas, se puede comprimir el historial en un resumen:

```python
class AgenteMemoriaComprimida:
    """Agente que comprime el historial cuando supera cierto tamaño."""

    def __init__(self, max_tokens_historial: int = 2000):
        self.cliente = anthropic.Anthropic()
        self.historial = []
        self.resumen_acumulado = ""
        self.max_tokens = max_tokens_historial

    def _tokens_estimados(self) -> int:
        return sum(len(m["content"]) // 4 for m in self.historial)

    def _comprimir_historial(self):
        """Comprime el historial antiguo en un resumen."""
        historial_texto = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self.historial[:-4]  # Conservar los últimos 4 mensajes
        )
        r = self.cliente.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
Resume esta conversación en 200 palabras, conservando la información importante:
{historial_texto}
"""}]
        )
        nuevo_resumen = r.content[0].text
        self.resumen_acumulado = f"{self.resumen_acumulado}\n\n{nuevo_resumen}".strip()
        self.historial = self.historial[-4:]  # Conservar solo los últimos 4
        print(f"📦 Historial comprimido. Resumen acumulado: {len(self.resumen_acumulado)} chars")

    def chatear(self, mensaje: str) -> str:
        if self._tokens_estimados() > self.max_tokens:
            self._comprimir_historial()

        self.historial.append({"role": "user", "content": mensaje})

        system = "Eres un asistente con memoria de la conversación."
        if self.resumen_acumulado:
            system += f"\n\nContexto previo de la conversación:\n{self.resumen_acumulado}"

        r = self.cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=self.historial
        )
        respuesta = r.content[0].text
        self.historial.append({"role": "assistant", "content": respuesta})
        return respuesta
```

---

## 6. Tipos por dominio

| Tipo | Objetivo | Herramientas típicas | Modelo recomendado |
|------|----------|---------------------|-------------------|
| **Research agent** | Investigar y sintetizar información | Web search, read_url, summarize | Sonnet/Opus |
| **Code agent** | Escribir, ejecutar y depurar código | execute_code, read_file, write_file, run_tests | Sonnet |
| **Data agent** | Analizar y visualizar datos | SQL, pandas, plot, statistics | Sonnet |
| **Task agent** | Completar tareas específicas | Herramientas del dominio | Haiku/Sonnet |
| **Conversational agent** | Mantener conversación natural | Memoria, personalización | Haiku |
| **Monitoring agent** | Vigilar sistemas y alertar | métricas, logs, alertas | Haiku |
| **Planning agent** | Planificar y coordinar proyectos | calendario, tareas, dependencias | Opus/Sonnet |

### Agente de investigación (Research Agent)

```python
def research_agent(pregunta: str) -> dict:
    """Agente especializado en investigación que estructura sus hallazgos."""
    cliente = anthropic.Anthropic()

    r = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="""Eres un investigador experto. Para cada pregunta:
1. Identifica las subtemas que necesitas investigar
2. Razona sobre cada uno
3. Sintetiza una respuesta estructurada
4. Identifica incertidumbres y limitaciones

Responde siempre con la estructura:
- HALLAZGOS PRINCIPALES
- ANÁLISIS
- LIMITACIONES Y DUDAS
- CONCLUSIÓN""",
        messages=[{"role": "user", "content": pregunta}]
    )

    return {
        "pregunta": pregunta,
        "investigacion": r.content[0].text,
        "tokens_usados": r.usage.input_tokens + r.usage.output_tokens
    }
```

---

## 7. Frameworks de agentes

### Comparativa de frameworks principales

| Framework | Lenguaje | Enfoque | Mejor para |
|-----------|----------|---------|-----------|
| **LangGraph** | Python/JS | Grafos de estado | Flujos complejos con estado, multi-agente |
| **CrewAI** | Python | Equipos de agentes con roles | Colaboración estructurada |
| **AutoGen** | Python | Conversación multi-agente | Investigación, automatización |
| **Claude Agent SDK** | Python | Agentes nativos con Claude | Producción con Claude |
| **LangChain Agents** | Python/JS | Agentes con chains | Ecosistema LangChain existente |
| **Vercel AI SDK** | JS/TS | Tool use con maxSteps | Apps web/Next.js |

### LangGraph — Agentes con estado

LangGraph representa el flujo del agente como un **grafo dirigido con estado**:

```python
# pip install langgraph langchain-anthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

class EstadoAgente(TypedDict):
    mensajes: Annotated[list, operator.add]
    siguiente: str

@tool
def calcular(expresion: str) -> str:
    """Evalúa una expresión matemática."""
    import math
    try:
        return str(eval(expresion, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return f"Error: {e}"

@tool
def buscar_info(tema: str) -> str:
    """Busca información sobre un tema."""
    info = {
        "python": "Python es el lenguaje dominante en IA",
        "claude": "Claude es la IA de Anthropic"
    }
    return info.get(tema.lower(), f"No hay info sobre {tema}")

def crear_grafo_agente():
    modelo = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([calcular, buscar_info])
    herramientas = ToolNode([calcular, buscar_info])

    def razonar(estado: EstadoAgente):
        return {"mensajes": [modelo.invoke(estado["mensajes"])]}

    def debe_continuar(estado: EstadoAgente) -> str:
        ultimo = estado["mensajes"][-1]
        return "herramientas" if ultimo.tool_calls else END

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("razonar", razonar)
    grafo.add_node("herramientas", herramientas)
    grafo.set_entry_point("razonar")
    grafo.add_conditional_edges("razonar", debe_continuar)
    grafo.add_edge("herramientas", "razonar")

    return grafo.compile()

# Ejecutar
agente_grafo = crear_grafo_agente()
resultado = agente_grafo.invoke({
    "mensajes": [{"role": "user", "content": "¿Cuánto es 15 al cuadrado? ¿Y qué es Python?"}]
})
print(resultado["mensajes"][-1].content)
```

### CrewAI — Equipos de agentes

```python
# pip install crewai crewai-tools
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

investigador = Agent(
    role="Investigador Senior",
    goal="Investigar y sintetizar información sobre el tema asignado",
    backstory="Experto en análisis de información con 10 años de experiencia",
    verbose=True,
    allow_delegation=False,
    llm="claude-sonnet-4-6"
)

redactor = Agent(
    role="Redactor de Contenidos",
    goal="Redactar artículos claros y atractivos basados en la investigación",
    backstory="Periodista especializado en tecnología",
    verbose=True,
    allow_delegation=False,
    llm="claude-sonnet-4-6"
)

tarea_investigacion = Task(
    description="Investiga los últimos avances en agentes de IA en 2025",
    expected_output="Informe estructurado con hallazgos clave, tendencias y ejemplos",
    agent=investigador
)

tarea_redaccion = Task(
    description="Escribe un artículo de blog de 500 palabras sobre los hallazgos",
    expected_output="Artículo listo para publicar con título, introducción, desarrollo y conclusión",
    agent=redactor,
    context=[tarea_investigacion]  # Depende del output del investigador
)

equipo = Crew(
    agents=[investigador, redactor],
    tasks=[tarea_investigacion, tarea_redaccion],
    process=Process.sequential,
    verbose=True
)

resultado = equipo.kickoff()
print(resultado)
```

---

## 8. Evaluación de agentes

Métricas clave para evaluar el rendimiento de un agente:

```python
import time
from dataclasses import dataclass

@dataclass
class MetricasAgente:
    tarea: str
    tasa_exito: float        # % de tareas completadas correctamente
    pasos_promedio: float    # Promedio de pasos hasta completar
    tiempo_promedio: float   # Segundos promedio por tarea
    coste_promedio: float    # Tokens usados / coste por tarea
    tasa_error: float        # % de errores de herramienta
    necesito_humano: float   # % de veces que necesitó intervención humana

def evaluar_agente(agente_fn, casos_prueba: list[dict]) -> MetricasAgente:
    """
    Evalúa un agente sobre un conjunto de casos de prueba.

    Cada caso: {"tarea": str, "respuesta_esperada": str, "validador": callable}
    """
    resultados = []

    for caso in casos_prueba:
        inicio = time.time()
        try:
            respuesta = agente_fn(caso["tarea"])
            exito = caso["validador"](respuesta) if "validador" in caso else True
            error = False
        except Exception as e:
            respuesta = str(e)
            exito = False
            error = True

        resultados.append({
            "exito": exito,
            "tiempo": time.time() - inicio,
            "error": error
        })

    n = len(resultados)
    return MetricasAgente(
        tarea=f"{n} casos evaluados",
        tasa_exito=sum(r["exito"] for r in resultados) / n,
        pasos_promedio=0,  # Calcular desde logs del agente
        tiempo_promedio=sum(r["tiempo"] for r in resultados) / n,
        coste_promedio=0,  # Calcular desde usage de la API
        tasa_error=sum(r["error"] for r in resultados) / n,
        necesito_humano=0
    )


# Casos de prueba para un agente de clasificación
casos = [
    {
        "tarea": "Clasifica: 'La app se crashea al login'",
        "validador": lambda r: "BUG" in r.upper()
    },
    {
        "tarea": "Clasifica: 'Añade modo oscuro'",
        "validador": lambda r: "FEATURE" in r.upper()
    },
]
```

### Framework de evaluación End-to-End

| Dimensión | Métrica | Cómo medirla |
|-----------|---------|--------------|
| **Efectividad** | Tasa de éxito en la tarea | Validador automático o juez LLM |
| **Eficiencia** | Pasos hasta completar | Conteo en el bucle agéntico |
| **Coste** | Tokens totales | `usage.input_tokens + output_tokens` |
| **Robustez** | Tasa de error ante inputs inesperados | Tests adversariales |
| **Latencia** | Tiempo de respuesta total | `time.time()` antes/después |
| **Seguridad** | Intentos de salirse del sandbox | Monitoreo de herramientas usadas |

---

## 9. Guía de selección

```
¿Qué tipo de agente necesito?
│
├── ¿La tarea es simple y bien definida?
│   └── Agente REACTIVO (Haiku, sin estado)
│
├── ¿Necesita planificar múltiples pasos?
│   ├── ¿El plan debe ser aprobado antes de ejecutar?
│   │   └── PLAN & EXECUTE
│   └── ¿Puede explorar interactivamente?
│       └── REACT
│
├── ¿La calidad importa más que la velocidad?
│   └── REFLEXION (agente + crítico)
│
├── ¿Necesita memoria entre sesiones?
│   ├── Pocos datos → Resumen comprimido
│   └── Muchos datos → Vector store (ChromaDB)
│
└── ¿Son varias tareas especializadas?
    ├── ¿Tareas independientes? → Multi-agente PARALELO
    ├── ¿Tareas secuenciales? → PIPELINE
    └── ¿Coordinación dinámica? → JERÁRQUICO u ORQUESTADOR
```

---

**Anterior:** [06 — Fine-tuning con LoRA](./06-finetuning-lora.md) · **Notebook:** [07-tipos-agentes.ipynb](../notebooks/llms/07-tipos-agentes.ipynb)
