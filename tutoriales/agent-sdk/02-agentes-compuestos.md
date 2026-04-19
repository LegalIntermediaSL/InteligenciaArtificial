# Agentes Compuestos: Orquestador y Subagentes

## Arquitectura multi-agente

En sistemas complejos, un único agente no siempre es suficiente. Los **agentes compuestos**
dividen el trabajo entre un orquestador que coordina y subagentes especializados:

```
Orquestador (planifica y delega)
    ├── Subagente Investigador (busca y resume información)
    ├── Subagente Analista (procesa datos y genera insights)
    └── Subagente Redactor (produce el output final)
```

**Ventajas:**
- Cada subagente tiene contexto más pequeño (más eficiente y preciso)
- Paralelización de tareas independientes
- Especialización de roles y herramientas
- Fácil de escalar añadiendo nuevos subagentes

## Orquestador básico

```python
import anthropic
from typing import Callable

client = anthropic.Anthropic()

def crear_subagente(
    rol: str,
    herramientas: list,
    ejecutor_herramientas: Callable,
    modelo: str = "claude-haiku-4-5-20251001"
) -> Callable:
    """Factory que crea un subagente con rol y herramientas específicas."""

    def subagente(tarea: str, contexto: str = "") -> str:
        system = f"""Eres un especialista en {rol}.
{f'Contexto adicional: {contexto}' if contexto else ''}
Completa la tarea asignada usando las herramientas disponibles."""

        mensajes = [{"role": "user", "content": tarea}]

        for _ in range(15):  # max iteraciones
            resp = client.messages.create(
                model=modelo,
                max_tokens=1500,
                system=system,
                tools=herramientas,
                messages=mensajes
            )
            mensajes.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                return next((b.text for b in resp.content if hasattr(b, "text")), "")

            resultados = []
            for bloque in resp.content:
                if bloque.type == "tool_use":
                    resultado = ejecutor_herramientas(bloque.name, bloque.input)
                    resultados.append({"type": "tool_result",
                                       "tool_use_id": bloque.id,
                                       "content": str(resultado)})
            mensajes.append({"role": "user", "content": resultados})

        return "Tarea completada (máx iteraciones)"

    return subagente
```

## Sistema de agentes para análisis de mercado

```python
# ---- Herramientas y ejecutores ----

HERRAMIENTAS_INVESTIGADOR = [
    {"name": "buscar_noticias", "description": "Busca noticias recientes sobre un tema",
     "input_schema": {"type": "object",
                      "properties": {"tema": {"type": "string"},
                                     "dias": {"type": "integer", "default": 7}},
                      "required": ["tema"]}},
    {"name": "buscar_datos_mercado", "description": "Obtiene datos de mercado y estadísticas",
     "input_schema": {"type": "object",
                      "properties": {"sector": {"type": "string"},
                                     "metrica": {"type": "string"}},
                      "required": ["sector"]}}
]

HERRAMIENTAS_ANALISTA = [
    {"name": "calcular_tendencia", "description": "Calcula tendencia de una serie de datos",
     "input_schema": {"type": "object",
                      "properties": {"datos": {"type": "array", "items": {"type": "number"}},
                                     "periodo": {"type": "string"}},
                      "required": ["datos"]}},
    {"name": "comparar_competidores",
     "description": "Compara métricas entre empresas competidoras",
     "input_schema": {"type": "object",
                      "properties": {"empresas": {"type": "array", "items": {"type": "string"}},
                                     "metrica": {"type": "string"}},
                      "required": ["empresas", "metrica"]}}
]

def ejecutor_investigador(nombre: str, params: dict) -> str:
    """Ejecuta herramientas de investigación (simuladas)."""
    if nombre == "buscar_noticias":
        return f"Noticias sobre '{params['tema']}': [Crecimiento del sector 15% anual, nueva regulación en 2025, inversión récord de 2.3B€]"
    elif nombre == "buscar_datos_mercado":
        return f"Datos de mercado {params['sector']}: TAM 45B€, CAGR 12%, 3 jugadores principales con 65% cuota"
    return "Datos no disponibles"

def ejecutor_analista(nombre: str, params: dict) -> str:
    """Ejecuta herramientas de análisis (simuladas)."""
    if nombre == "calcular_tendencia":
        datos = params["datos"]
        if len(datos) >= 2:
            crecimiento = (datos[-1] - datos[0]) / datos[0] * 100
            return f"Tendencia: {crecimiento:+.1f}% total, {crecimiento/len(datos):+.1f}% por período"
        return "Insuficientes datos"
    elif nombre == "comparar_competidores":
        return f"Comparativa {params['metrica']} entre {params['empresas']}: Empresa A lidera con 34% cuota"
    return "Análisis no disponible"

# ---- Crear subagentes ----
subagente_investigador = crear_subagente(
    "investigación de mercado y recopilación de datos",
    HERRAMIENTAS_INVESTIGADOR,
    ejecutor_investigador
)

subagente_analista = crear_subagente(
    "análisis de datos e interpretación de tendencias",
    HERRAMIENTAS_ANALISTA,
    ejecutor_analista,
    modelo="claude-haiku-4-5-20251001"
)

# ---- Orquestador principal ----
def orquestador_analisis_mercado(sector: str) -> dict:
    """Orquestador que coordina investigación y análisis en paralelo."""
    print(f"Analizando mercado: {sector}")

    # Fase 1: Investigación (podría ser en paralelo con concurrent.futures)
    print("\n[Subagente Investigador]")
    info_mercado = subagente_investigador(
        f"Recopila información sobre el mercado de {sector} en Europa: "
        f"tamaño de mercado, tendencias, noticias recientes y principales actores."
    )
    print(f"  → {info_mercado[:100]}...")

    # Fase 2: Análisis basado en la investigación
    print("\n[Subagente Analista]")
    analisis = subagente_analista(
        f"Analiza esta información de mercado y extrae insights clave: {info_mercado[:500]}",
        contexto=f"Sector: {sector}"
    )
    print(f"  → {analisis[:100]}...")

    # Fase 3: Síntesis por el orquestador
    print("\n[Orquestador] Sintetizando resultados...")
    resp_final = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""Sintetiza estos resultados en un informe ejecutivo para inversores.

Investigación: {info_mercado}
Análisis: {analisis}

El informe debe tener: Resumen ejecutivo, Oportunidad de mercado, Riesgos, Recomendación."""
        }]
    )

    return {
        "sector": sector,
        "investigacion": info_mercado,
        "analisis": analisis,
        "informe_ejecutivo": resp_final.content[0].text
    }

resultado = orquestador_analisis_mercado("inteligencia artificial B2B en Europa")
print("\n=== INFORME EJECUTIVO ===")
print(resultado["informe_ejecutivo"])
```

## Paso de contexto entre agentes

```python
class ContextoCompartido:
    """Almacén de contexto que los agentes pueden leer y escribir."""

    def __init__(self):
        self._datos = {}
        self._historial = []

    def guardar(self, clave: str, valor):
        self._datos[clave] = valor
        self._historial.append(f"[ESCRITURA] {clave}")

    def leer(self, clave: str, default=None):
        return self._datos.get(clave, default)

    def resumen(self) -> str:
        """Genera un resumen del contexto para pasar a otros agentes."""
        return "\n".join(
            f"- {k}: {str(v)[:100]}"
            for k, v in self._datos.items()
        )

# Uso en pipeline multi-agente
contexto = ContextoCompartido()

# Agente 1: recopila datos
datos_crudos = subagente_investigador("Busca los 5 principales competidores en SaaS de RRHH")
contexto.guardar("competidores_raw", datos_crudos)

# Agente 2: recibe contexto del anterior
analisis_comp = subagente_analista(
    "Analiza fortalezas y debilidades de estos competidores",
    contexto=f"Datos previos:\n{contexto.resumen()}"
)
contexto.guardar("analisis_competidores", analisis_comp)
```

## Delegación dinámica con herramienta especial

```python
# El orquestador puede delegar a subagentes como si fueran herramientas
HERRAMIENTA_DELEGACION = {
    "name": "delegar_tarea",
    "description": "Delega una subtarea a un agente especializado",
    "input_schema": {
        "type": "object",
        "properties": {
            "agente": {
                "type": "string",
                "enum": ["investigador", "analista", "redactor"],
                "description": "Agente especializado al que delegar"
            },
            "tarea": {"type": "string", "description": "Descripción detallada de la tarea"}
        },
        "required": ["agente", "tarea"]
    }
}

SUBAGENTES = {
    "investigador": subagente_investigador,
    "analista": subagente_analista,
}

def ejecutar_delegacion(nombre: str, params: dict) -> str:
    if nombre == "delegar_tarea":
        agente_nombre = params["agente"]
        if agente_nombre in SUBAGENTES:
            return SUBAGENTES[agente_nombre](params["tarea"])
        return f"Agente '{agente_nombre}' no disponible"
    return "Herramienta desconocida"
```

## Recursos

- [Notebook interactivo](../notebooks/agent-sdk/02-agentes-compuestos.ipynb)
- [Anthropic Cookbook — Multi-agent](https://github.com/anthropics/anthropic-cookbook/tree/main/multiagent)
