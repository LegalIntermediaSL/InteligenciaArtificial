# PydanticAI — Introducción: agentes tipados en Python

PydanticAI es el framework de agentes de los creadores de Pydantic. A diferencia de LangChain
o CrewAI, pone el foco en **tipado estricto**, **testabilidad** y una interfaz limpia que sigue
los patrones de diseño de Python moderno.

---

## Instalación

```bash
pip install pydantic-ai
pip install pydantic-ai[anthropic]  # añade soporte para Claude
```

---

## 1. El agente más simple

```python
from pydantic_ai import Agent

agente = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    system_prompt='Eres un asistente de análisis de contratos legales.',
)

resultado = agente.run_sync('¿Qué es una cláusula de responsabilidad limitada?')
print(resultado.data)
```

La diferencia clave con usar el SDK directamente: el agente gestiona el historial,
la inyección de dependencias y la validación de outputs de forma automática.

---

## 2. Tipado de outputs — structured output nativo

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class AnalisisContrato(BaseModel):
    riesgo: str          # 'alto', 'medio', 'bajo'
    clausulas_criticas: list[str]
    recomendacion: str
    requiere_abogado: bool

agente_analista = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=AnalisisContrato,
    system_prompt='Analiza contratos e identifica riesgos.',
)

resultado = agente_analista.run_sync('''
Cláusula 5: El proveedor no asume responsabilidad alguna por pérdidas de datos,
interrupciones del servicio o daños indirectos, con independencia de su causa.
La indemnización máxima queda limitada a 100€.
''')

analisis: AnalisisContrato = resultado.data
print(f'Riesgo: {analisis.riesgo}')
print(f'Requiere abogado: {analisis.requiere_abogado}')
print(f'Recomendación: {analisis.recomendacion}')
```

Si el modelo no devuelve un JSON válido que encaje con `AnalisisContrato`, PydanticAI
reintenta automáticamente (hasta `max_retries` veces) con el error de validación como contexto.

---

## 3. Herramientas (tools)

```python
from pydantic_ai import Agent, RunContext
from datetime import date, timedelta

agente_plazos = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    system_prompt='Calcula plazos legales y vencimientos de contratos.',
)

@agente_plazos.tool_plain
def calcular_fecha_preaviso(fecha_vencimiento: str, dias_preaviso: int) -> str:
    """Calcula la fecha límite para enviar el preaviso de no renovación."""
    vencimiento = date.fromisoformat(fecha_vencimiento)
    preaviso = vencimiento - timedelta(days=dias_preaviso)
    hoy = date.today()
    dias_restantes = (preaviso - hoy).days
    return (
        f'Preaviso debe enviarse antes del {preaviso.isoformat()}. '
        f'Quedan {max(0, dias_restantes)} días.'
    )

@agente_plazos.tool_plain
def dias_habiles_hasta(fecha_objetivo: str) -> int:
    """Cuenta los días hábiles (lunes-viernes) hasta una fecha."""
    objetivo = date.fromisoformat(fecha_objetivo)
    hoy = date.today()
    dias = 0
    actual = hoy
    while actual < objetivo:
        actual += timedelta(days=1)
        if actual.weekday() < 5:  # lunes=0, viernes=4
            dias += 1
    return dias

resultado = agente_plazos.run_sync(
    'El contrato con Acme Corp vence el 2026-09-01 y requiere 90 días de preaviso. '
    '¿Cuándo es el límite para el preaviso y cuántos días hábiles quedan?'
)
print(resultado.data)
```

---

## 4. Dependencias inyectadas

El patrón de dependencias de PydanticAI permite inyectar servicios externos
(base de datos, cliente HTTP, configuración) de forma testeable.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
import httpx

@dataclass
class Dependencias:
    cliente_http: httpx.AsyncClient
    api_key_interna: str
    modo_debug: bool = False

agente_enriquecido = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    deps_type=Dependencias,
    system_prompt='Enriqueces datos de empresas usando APIs externas.',
)

@agente_enriquecido.tool
async def buscar_empresa(ctx: RunContext[Dependencias], nombre: str) -> dict:
    """Busca información de una empresa en la API interna."""
    if ctx.deps.modo_debug:
        return {'nombre': nombre, 'cif': 'B12345678', 'estado': 'activa', 'debug': True}

    headers = {'Authorization': f'Bearer {ctx.deps.api_key_interna}'}
    resp = await ctx.deps.cliente_http.get(
        f'https://api.interna.com/empresas?q={nombre}',
        headers=headers
    )
    return resp.json()

# En producción
async def main_produccion():
    async with httpx.AsyncClient() as http:
        deps = Dependencias(
            cliente_http=http,
            api_key_interna='sk-prod-xxx',
        )
        resultado = await agente_enriquecido.run(
            'Busca información sobre la empresa Acme Corp', deps=deps
        )
        print(resultado.data)

# En tests (sin llamadas reales)
from pydantic_ai.models.test import TestModel

async def test_buscar_empresa():
    deps = Dependencias(
        cliente_http=httpx.AsyncClient(),
        api_key_interna='test-key',
        modo_debug=True,  # evita llamadas HTTP reales
    )
    with agente_enriquecido.override(model=TestModel()):
        resultado = await agente_enriquecido.run('Busca Acme Corp', deps=deps)
        # TestModel devuelve datos predecibles → test determinista
```

---

## 5. Streaming

```python
import asyncio
from pydantic_ai import Agent

agente = Agent(model='anthropic:claude-haiku-4-5-20251001')

async def chat_streaming():
    async with agente.run_stream('Explica qué es un agente de IA en 3 puntos') as stream:
        async for texto in stream.stream_text(delta=True):
            print(texto, end='', flush=True)
        print()

        # También disponible: uso de tokens
        usage = stream.usage()
        print(f'\nTokens: entrada={usage.request_tokens}, salida={usage.response_tokens}')

asyncio.run(chat_streaming())
```

---

## 6. Historial de mensajes

```python
from pydantic_ai import Agent

agente = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    system_prompt='Eres un asistente de contratos.',
)

# Primer turno
resultado1 = agente.run_sync('El contrato de Acme vence en septiembre de 2026.')
print(resultado1.data)

# Segundo turno — pasa el historial del primero
resultado2 = agente.run_sync(
    '¿Cuándo debería enviar el preaviso si son 90 días?',
    message_history=resultado1.new_messages(),
)
print(resultado2.data)

# Ver el historial completo
for msg in resultado2.all_messages():
    print(f'[{msg.kind}] {str(msg)[:80]}')
```

---

## Resumen de la API

| Método | Cuándo usar |
|---|---|
| `agent.run_sync()` | Scripts, Jupyter, código síncrono |
| `await agent.run()` | Código async (FastAPI, etc.) |
| `async with agent.run_stream()` | Streaming en tiempo real |
| `agent.override(model=TestModel())` | Tests sin coste de API |

---

→ Siguiente: [PydanticAI avanzado](02-pydanticai-avanzado.md)
