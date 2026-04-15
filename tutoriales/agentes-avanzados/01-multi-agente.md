# 01 — Sistemas Multi-Agente

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 75 min

---

## Índice

1. Por qué sistemas multi-agente
2. Arquitecturas multi-agente
3. Implementación manual con Claude
4. Multi-agente con CrewAI
5. Comunicación entre agentes
6. Cuándo usar multi-agente vs agente único
7. Extensiones sugeridas

---

## 1. Por qué sistemas multi-agente

Un agente único con acceso a herramientas es suficiente para muchas tareas. Pero hay casos donde este enfoque se queda corto:

**Limitaciones de un agente único:**

- **Ventana de contexto saturada.** Una tarea compleja acumula miles de tokens de historial. El agente pierde el hilo o toma decisiones peores porque el contexto relevante queda enterrado.
- **Falta de especialización.** Un agente generalista que investiga, redacta, revisa código y gestiona bases de datos tiende a ser mediocre en todo. Un experto en cada área hace mejor su parte.
- **Ejecución secuencial forzada.** Un agente único no puede investigar y redactar al mismo tiempo. Un sistema multi-agente puede paralelizar trabajo.
- **Fallo catastrófico.** Si el único agente comete un error temprano, todo el flujo se contamina. Con múltiples agentes y un revisor, los errores se detectan antes.
- **Escalabilidad.** Añadir capacidades a un agente único vuelve su prompt cada vez más frágil. Añadir un nuevo agente especializado es modular.

**Cuándo un sistema multi-agente aporta valor real:**

- Tareas que tienen fases claramente distintas (investigar → sintetizar → redactar → revisar)
- Flujos de trabajo que pueden paralelizarse
- Dominios que requieren expertos distintos (código, legal, diseño, datos)
- Sistemas que deben operar durante horas con contextos muy largos

---

## 2. Arquitecturas multi-agente

Existen tres patrones principales. La elección depende del nivel de coordinación que necesitas.

### Orquestador-Trabajadores

El patrón más común. Un agente central (orquestador) descompone la tarea y delega subtareas a agentes especializados (trabajadores). Los trabajadores no se conocen entre sí.

```
          ┌─────────────────┐
          │   ORQUESTADOR   │
          │  (planifica y   │
          │   coordina)     │
          └────────┬────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
 ┌──────────┐ ┌──────────┐ ┌──────────┐
 │Agente    │ │Agente    │ │Agente    │
 │Investiga-│ │Redactor  │ │Revisor   │
 │dor       │ │          │ │          │
 └──────────┘ └──────────┘ └──────────┘
```

**Ventajas:** fácil de razonar, el orquestador tiene control total, fácil de depurar.
**Desventajas:** cuello de botella en el orquestador, difícil paralelizar.

### Par a Par (Peer-to-Peer)

Los agentes se comunican directamente entre sí. Adecuado para deliberación, negociación o revisión cruzada.

```
 ┌──────────┐     ┌──────────┐
 │ Agente A │◄───►│ Agente B │
 └────┬─────┘     └─────┬────┘
      │                 │
      └────────┬─────────┘
               ▼
        ┌──────────┐
        │ Agente C │
        └──────────┘
```

**Ventajas:** descentralizado, robusto ante fallos de un nodo.
**Desventajas:** puede derivar en bucles o desacuerdos sin resolución.

### Jerárquica

Múltiples niveles de orquestación. Los orquestadores de alto nivel delegan a suborquestadores, que a su vez delegan a trabajadores.

```
          ┌───────────────┐
          │  DIRECTOR IA  │
          └───────┬───────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
 ┌─────────────┐    ┌─────────────┐
 │Orq. Research│    │Orq. Content │
 └──────┬──────┘    └──────┬──────┘
        │                  │
   ┌────┴────┐         ┌────┴────┐
   ▼         ▼         ▼         ▼
[Web]     [DB]      [Draft]   [Edit]
```

**Ventajas:** escala bien para proyectos grandes, modular.
**Desventajas:** latencia acumulada, complejidad de depuración.

---

## 3. Implementación manual con Claude

Construiremos un sistema con tres agentes especializados que colaboran para producir un informe: un investigador, un redactor y un revisor. El orquestador coordina el flujo usando `tool_use` de la API de Anthropic.

```python
import anthropic
import json
from typing import Any

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Agentes especializados
# Cada agente es una función que recibe una tarea y devuelve un resultado.
# ---------------------------------------------------------------------------

def agente_investigador(tema: str) -> str:
    """Investiga un tema y devuelve hechos clave estructurados."""
    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=1024,
        system=(
            "Eres un investigador experto. Tu tarea es recopilar información "
            "factual, precisa y bien estructurada sobre el tema indicado. "
            "Devuelve entre 5 y 8 puntos clave en formato de lista numerada. "
            "Sé conciso pero completo."
        ),
        messages=[
            {"role": "user", "content": f"Investiga el siguiente tema y dame los puntos clave:\n\n{tema}"}
        ],
    )
    return respuesta.content[0].text


def agente_redactor(tema: str, investigacion: str) -> str:
    """Redacta un artículo bien estructurado a partir de la investigación."""
    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=2048,
        system=(
            "Eres un redactor profesional especializado en divulgación técnica. "
            "Tu tarea es transformar información de investigación en un artículo "
            "claro, bien estructurado y accesible para un público no especialista. "
            "Usa un tono informativo pero cercano. Incluye introducción, desarrollo "
            "y conclusión."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Escribe un artículo sobre: {tema}\n\n"
                    f"Basándote en esta investigación:\n{investigacion}"
                ),
            }
        ],
    )
    return respuesta.content[0].text


def agente_revisor(articulo: str) -> dict[str, Any]:
    """Revisa el artículo y devuelve puntuación, problemas y versión corregida."""
    respuesta = cliente.messages.create(
        model=MODELO,
        max_tokens=2048,
        system=(
            "Eres un editor senior con criterio exigente. Tu tarea es revisar "
            "artículos y devolver un JSON con exactamente estas claves:\n"
            "- puntuacion: número del 1 al 10\n"
            "- problemas: lista de strings con los problemas encontrados\n"
            "- articulo_corregido: el artículo con las correcciones aplicadas\n"
            "Devuelve SOLO el JSON, sin texto adicional."
        ),
        messages=[
            {"role": "user", "content": f"Revisa y corrige este artículo:\n\n{articulo}"}
        ],
    )
    texto = respuesta.content[0].text.strip()
    # Limpiar posibles bloques de código markdown
    if texto.startswith("```"):
        texto = texto.split("```")[1]
        if texto.startswith("json"):
            texto = texto[4:]
    return json.loads(texto)


# ---------------------------------------------------------------------------
# Orquestador
# Define las herramientas disponibles y coordina el flujo completo.
# ---------------------------------------------------------------------------

HERRAMIENTAS = [
    {
        "name": "investigar",
        "description": "Usa al agente investigador para recopilar información sobre un tema.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tema": {
                    "type": "string",
                    "description": "El tema a investigar.",
                }
            },
            "required": ["tema"],
        },
    },
    {
        "name": "redactar",
        "description": "Usa al agente redactor para escribir un artículo sobre un tema a partir de investigación previa.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tema": {"type": "string", "description": "El tema del artículo."},
                "investigacion": {"type": "string", "description": "La investigación recopilada."},
            },
            "required": ["tema", "investigacion"],
        },
    },
    {
        "name": "revisar",
        "description": "Usa al agente revisor para revisar y mejorar un artículo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "articulo": {"type": "string", "description": "El artículo a revisar."}
            },
            "required": ["articulo"],
        },
    },
]


def ejecutar_herramienta(nombre: str, parametros: dict) -> str:
    """Despacha la llamada al agente correspondiente."""
    if nombre == "investigar":
        resultado = agente_investigador(parametros["tema"])
        return resultado
    elif nombre == "redactar":
        resultado = agente_redactor(parametros["tema"], parametros["investigacion"])
        return resultado
    elif nombre == "revisar":
        resultado = agente_revisor(parametros["articulo"])
        return json.dumps(resultado, ensure_ascii=False, indent=2)
    else:
        return f"Herramienta desconocida: {nombre}"


def orquestador(tarea: str) -> str:
    """
    Orquestador principal.
    Coordina los agentes hasta producir el resultado final.
    """
    print(f"\n{'='*60}")
    print(f"ORQUESTADOR: Iniciando tarea")
    print(f"{'='*60}")
    print(f"Tarea: {tarea}\n")

    historial = [{"role": "user", "content": tarea}]

    while True:
        respuesta = cliente.messages.create(
            model=MODELO,
            max_tokens=4096,
            system=(
                "Eres un orquestador de agentes de IA. Tu trabajo es coordinar "
                "a tres agentes especializados (investigador, redactor, revisor) "
                "para producir un artículo de alta calidad sobre el tema solicitado.\n\n"
                "Flujo recomendado:\n"
                "1. Llama a 'investigar' para recopilar información.\n"
                "2. Llama a 'redactar' con el tema y la investigación.\n"
                "3. Llama a 'revisar' con el artículo producido.\n"
                "4. Presenta el artículo final corregido al usuario.\n\n"
                "Cuando tengas el artículo revisado y corregido, preséntalo "
                "directamente sin más llamadas a herramientas."
            ),
            tools=HERRAMIENTAS,
            messages=historial,
        )

        # Añadir la respuesta del asistente al historial
        historial.append({"role": "assistant", "content": respuesta.content})

        # Si el modelo termina sin usar herramientas, es la respuesta final
        if respuesta.stop_reason == "end_turn":
            texto_final = next(
                (bloque.text for bloque in respuesta.content if hasattr(bloque, "text")),
                "",
            )
            print("\n" + "="*60)
            print("RESULTADO FINAL")
            print("="*60)
            print(texto_final)
            return texto_final

        # Procesar llamadas a herramientas
        resultados_herramientas = []
        for bloque in respuesta.content:
            if bloque.type == "tool_use":
                print(f"\n[Orquestador] Delegando a: {bloque.name}")
                print(f"  Parámetros: {json.dumps(bloque.input, ensure_ascii=False)[:120]}...")

                resultado = ejecutar_herramienta(bloque.name, bloque.input)

                print(f"  Resultado recibido ({len(resultado)} caracteres)")

                resultados_herramientas.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": resultado,
                })

        # Añadir resultados al historial y continuar el bucle
        historial.append({"role": "user", "content": resultados_herramientas})


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tarea = (
        "Produce un artículo de divulgación sobre el impacto de la inteligencia "
        "artificial generativa en el mercado laboral. Debe ser accesible para "
        "el público general y tener aproximadamente 400 palabras."
    )
    orquestador(tarea)
```

**Puntos clave del código:**

- Cada agente tiene un `system` prompt muy específico que define su rol.
- El orquestador usa `tool_use` para delegar, manteniendo un historial que incluye tanto las llamadas a herramientas como sus resultados.
- El bucle `while True` continúa hasta que el modelo devuelve `stop_reason == "end_turn"`, señal de que ha terminado de coordinar.

---

## 4. Multi-agente con CrewAI

CrewAI es un framework de alto nivel que abstrae la creación de agentes, tareas y flujos de trabajo. Reduce el código repetitivo considerablemente.

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool  # opcional: búsqueda web real
import os

# Si tienes una clave de Serper para búsqueda web real:
# os.environ["SERPER_API_KEY"] = "tu-clave"

# CrewAI usa la variable OPENAI_API_KEY por defecto, pero puede configurarse
# para usar Anthropic. La configuración más sencilla es via variable de entorno:
os.environ["OPENAI_API_KEY"] = "no-necesaria"  # placeholder si usas Anthropic

# ---------------------------------------------------------------------------
# Definición de agentes
# ---------------------------------------------------------------------------

analista_mercado = Agent(
    role="Analista de Mercado Senior",
    goal="Recopilar y analizar datos de mercado para identificar tendencias y oportunidades",
    backstory=(
        "Llevas 15 años analizando mercados tecnológicos. Tu especialidad es "
        "identificar señales tempranas de disrupciones y cuantificar su impacto "
        "potencial. Eres metódico, basas tus conclusiones en datos y evitas "
        "las predicciones sin fundamento."
    ),
    verbose=True,
    allow_delegation=False,
    # tools=[SerperDevTool()]  # descomentar para búsqueda web real
)

estratega = Agent(
    role="Estratega de Negocio",
    goal="Transformar análisis de mercado en recomendaciones estratégicas accionables",
    backstory=(
        "Consultor estratégico con experiencia en startups y empresas Fortune 500. "
        "Tu enfoque es pragmático: tomas datos complejos y los conviertes en "
        "planes de acción claros con prioridades definidas y métricas de éxito."
    ),
    verbose=True,
    allow_delegation=False,
)

redactor_informe = Agent(
    role="Redactor de Informes Ejecutivos",
    goal="Producir informes ejecutivos claros, concisos y visualmente bien estructurados",
    backstory=(
        "Especialista en comunicación corporativa con experiencia redactando informes "
        "para C-level. Dominas la estructura pirámide invertida: primero la conclusión, "
        "luego el sustento. Tus informes son accionables desde la primera página."
    ),
    verbose=True,
    allow_delegation=False,
)

# ---------------------------------------------------------------------------
# Definición de tareas
# ---------------------------------------------------------------------------

tarea_analisis = Task(
    description=(
        "Realiza un análisis del mercado de inteligencia artificial generativa "
        "en 2025. Cubre: tamaño del mercado, principales jugadores, segmentos "
        "de mayor crecimiento, barreras de entrada y amenazas regulatorias. "
        "Presenta los datos en formato estructurado con métricas concretas."
    ),
    expected_output=(
        "Un análisis de mercado estructurado con secciones claras, métricas "
        "cuantitativas donde sea posible, y una síntesis de los 5 hallazgos más relevantes."
    ),
    agent=analista_mercado,
)

tarea_estrategia = Task(
    description=(
        "Basándote en el análisis de mercado previo, desarrolla 3 recomendaciones "
        "estratégicas para una empresa mediana (50-200 empleados) del sector servicios "
        "que quiere incorporar IA generativa en sus operaciones. "
        "Incluye: descripción de la oportunidad, pasos de implementación, "
        "inversión estimada, riesgos y métricas de éxito."
    ),
    expected_output=(
        "Tres recomendaciones estratégicas detalladas con plan de implementación, "
        "estimación de costos y KPIs para medir el éxito."
    ),
    agent=estratega,
    context=[tarea_analisis],  # esta tarea depende de la anterior
)

tarea_informe = Task(
    description=(
        "Redacta un informe ejecutivo de 2 páginas que combine el análisis de mercado "
        "y las recomendaciones estratégicas. El informe debe poder ser leído en 5 minutos "
        "y dejar claro al lector qué debe hacer y por qué. "
        "Usa encabezados, puntos clave y un resumen ejecutivo al inicio."
    ),
    expected_output=(
        "Un informe ejecutivo bien estructurado con resumen ejecutivo, "
        "hallazgos clave, recomendaciones priorizadas y próximos pasos."
    ),
    agent=redactor_informe,
    context=[tarea_analisis, tarea_estrategia],
)

# ---------------------------------------------------------------------------
# Crew: el equipo completo
# ---------------------------------------------------------------------------

equipo = Crew(
    agents=[analista_mercado, estratega, redactor_informe],
    tasks=[tarea_analisis, tarea_estrategia, tarea_informe],
    process=Process.sequential,  # las tareas se ejecutan en orden
    verbose=True,
)

# ---------------------------------------------------------------------------
# Ejecutar el flujo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Iniciando análisis de mercado con CrewAI...\n")
    resultado = equipo.kickoff()
    print("\n" + "="*60)
    print("INFORME FINAL")
    print("="*60)
    print(resultado)
```

**Conceptos de CrewAI:**

| Concepto | Descripción |
|---|---|
| `Agent` | Un agente con rol, objetivo y contexto (backstory) |
| `Task` | Una tarea con descripción, resultado esperado y agente responsable |
| `Crew` | El equipo: conjunto de agentes y tareas |
| `Process.sequential` | Las tareas se ejecutan en orden, pasando contexto |
| `Process.hierarchical` | Un manager LLM decide el orden de ejecución |
| `context` | Permite que una tarea acceda a los resultados de tareas anteriores |

---

## 5. Comunicación entre agentes

Para sistemas donde los agentes necesitan comunicarse de forma asíncrona (sin esperar turno), usamos colas de mensajes y memoria compartida.

```python
import asyncio
import anthropic
import json
from dataclasses import dataclass, field
from typing import Optional

cliente = anthropic.Anthropic()
MODELO = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Estructuras de datos compartidas
# ---------------------------------------------------------------------------

@dataclass
class Mensaje:
    de: str
    para: str
    contenido: str
    tipo: str = "tarea"  # "tarea", "resultado", "consulta"


@dataclass
class MemoriaCompartida:
    """Pizarra compartida donde los agentes depositan y consultan información."""
    datos: dict = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def escribir(self, clave: str, valor: str) -> None:
        async with self.lock:
            self.datos[clave] = valor
            print(f"  [Memoria] '{clave}' actualizado")

    async def leer(self, clave: str) -> Optional[str]:
        async with self.lock:
            return self.datos.get(clave)

    async def listar_claves(self) -> list[str]:
        async with self.lock:
            return list(self.datos.keys())


# ---------------------------------------------------------------------------
# Agentes asíncronos
# ---------------------------------------------------------------------------

async def agente_investigador_async(
    cola_entrada: asyncio.Queue,
    cola_salida: asyncio.Queue,
    memoria: MemoriaCompartida,
) -> None:
    """Agente investigador: procesa tareas de investigación de la cola."""
    print("[Investigador] En espera de tareas...")

    while True:
        mensaje = await cola_entrada.get()

        if mensaje.contenido == "TERMINAR":
            print("[Investigador] Recibida señal de término.")
            cola_entrada.task_done()
            break

        print(f"[Investigador] Procesando: {mensaje.contenido[:60]}...")

        respuesta = await asyncio.to_thread(
            lambda: cliente.messages.create(
                model=MODELO,
                max_tokens=512,
                system="Eres un investigador. Resume los puntos clave del tema en 3-5 bullet points.",
                messages=[{"role": "user", "content": mensaje.contenido}],
            )
        )

        resultado = respuesta.content[0].text
        await memoria.escribir(f"investigacion_{mensaje.contenido[:30]}", resultado)

        # Notificar al siguiente agente
        await cola_salida.put(
            Mensaje(
                de="investigador",
                para="redactor",
                contenido=resultado,
                tipo="resultado",
            )
        )

        cola_entrada.task_done()


async def agente_redactor_async(
    cola_entrada: asyncio.Queue,
    cola_salida: asyncio.Queue,
    memoria: MemoriaCompartida,
) -> None:
    """Agente redactor: recibe investigación y produce texto."""
    print("[Redactor] En espera de investigaciones...")

    while True:
        mensaje = await cola_entrada.get()

        if mensaje.contenido == "TERMINAR":
            print("[Redactor] Recibida señal de término.")
            cola_entrada.task_done()
            break

        print(f"[Redactor] Redactando a partir de investigación...")

        respuesta = await asyncio.to_thread(
            lambda: cliente.messages.create(
                model=MODELO,
                max_tokens=800,
                system="Eres un redactor. Transforma los puntos clave en un párrafo fluido y bien escrito.",
                messages=[{"role": "user", "content": mensaje.contenido}],
            )
        )

        borrador = respuesta.content[0].text
        await memoria.escribir("borrador_actual", borrador)

        await cola_salida.put(
            Mensaje(
                de="redactor",
                para="coordinador",
                contenido=borrador,
                tipo="resultado",
            )
        )

        cola_entrada.task_done()


async def coordinador(temas: list[str]) -> dict:
    """Coordina el flujo de trabajo entre agentes usando colas."""
    memoria = MemoriaCompartida()

    # Colas de comunicación entre agentes
    cola_investigacion = asyncio.Queue()
    cola_redaccion = asyncio.Queue()
    cola_resultados = asyncio.Queue()

    # Lanzar agentes en segundo plano
    tarea_investigador = asyncio.create_task(
        agente_investigador_async(cola_investigacion, cola_redaccion, memoria)
    )
    tarea_redactor = asyncio.create_task(
        agente_redactor_async(cola_redaccion, cola_resultados, memoria)
    )

    # Enviar tareas al investigador
    for tema in temas:
        await cola_investigacion.put(
            Mensaje(de="coordinador", para="investigador", contenido=tema)
        )

    # Señales de terminación
    await cola_investigacion.put(
        Mensaje(de="coordinador", para="investigador", contenido="TERMINAR")
    )

    # Recoger resultados
    resultados = []
    for _ in temas:
        msg = await cola_resultados.get()
        resultados.append(msg.contenido)
        cola_resultados.task_done()

    # Señal de terminación al redactor (llega después de que procese todo)
    await cola_redaccion.put(
        Mensaje(de="coordinador", para="redactor", contenido="TERMINAR")
    )

    # Esperar a que los agentes terminen
    await asyncio.gather(tarea_investigador, tarea_redactor)

    return {
        "resultados": resultados,
        "memoria": await memoria.listar_claves(),
    }


if __name__ == "__main__":
    temas = [
        "Impacto de la IA en la educación universitaria",
        "Uso de modelos de lenguaje en el sector salud",
    ]

    resultado = asyncio.run(coordinador(temas))
    print("\nClaves en memoria compartida:", resultado["memoria"])
    print(f"\nTotal de documentos producidos: {len(resultado['resultados'])}")
```

---

## 6. Cuándo usar multi-agente vs agente único

| Criterio | Agente único | Sistema multi-agente |
|---|---|---|
| **Complejidad de la tarea** | Baja o media | Alta, con fases distintas |
| **Tiempo de desarrollo** | Horas | Días o semanas |
| **Costo por ejecución** | Bajo (1 LLM) | Alto (N LLMs) |
| **Latencia** | Baja | Mayor (coordinación) |
| **Especialización** | Generalista | Cada agente es experto |
| **Paralelización** | No | Sí |
| **Depuración** | Simple | Compleja |
| **Escalabilidad** | Limitada | Alta |
| **Tolerancia a fallos** | Baja | Alta con revisores |
| **Contexto necesario** | Cabe en una ventana | Distribuido entre agentes |

**Regla práctica:** empieza siempre con un agente único. Migra a multi-agente cuando el agente único falle consistentemente o cuando la tarea tenga fases que un experto humano haría por separado.

---

## 7. Extensiones sugeridas

- **Persistencia de estado:** guarda el historial de cada agente en disco para poder reanudar flujos largos ante fallos.
- **Monitoreo:** añade logging estructurado (JSON) para rastrear qué agente ejecutó qué y cuánto tiempo tardó.
- **Agente evaluador:** agrega un agente cuya única función sea puntuar los resultados de los demás y decidir si hay que rehacer una tarea.
- **Paralelización real:** usa `asyncio.gather` para que el investigador y el redactor trabajen en paralelo sobre distintas secciones.
- **Integración con herramientas reales:** equipa al agente investigador con búsqueda web (Serper, Brave Search) o acceso a bases de datos.
- **CrewAI con Process.hierarchical:** prueba el modo jerárquico donde un manager LLM decide dinámicamente qué agente asignar a cada subtarea.

---

**Siguiente:** [02 — Model Context Protocol](./02-model-context-protocol.md)
