# PydanticAI avanzado — Testing, validación y multi-agente

Patrones de producción: tests sin coste de API, validadores personalizados,
agentes anidados y despliegue con FastAPI.

---

## 1. Testing con TestModel

El punto más diferencial de PydanticAI: **tests deterministas sin llamar a la API real**.

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic import BaseModel
from dataclasses import dataclass

class ClasificacionEmail(BaseModel):
    categoria: str   # 'ventas', 'soporte', 'spam', 'otro'
    prioridad: int   # 1-5
    requiere_respuesta: bool

@dataclass
class DepsCRM:
    usuario_id: int
    limite_prioridad: int = 3

agente_clasificador = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=ClasificacionEmail,
    deps_type=DepsCRM,
    system_prompt='Clasifica emails entrantes del CRM.',
)

# Test sin llamadas a la API
def test_clasificar_email_spam():
    deps = DepsCRM(usuario_id=42)

    # TestModel devuelve un resultado vacío/por defecto o uno que especifiques
    with agente_clasificador.override(model=TestModel()):
        resultado = agente_clasificador.run_sync(
            'Gana dinero fácil desde casa!!!',
            deps=deps,
        )
    # TestModel produce un ClasificacionEmail con valores por defecto
    assert isinstance(resultado.data, ClasificacionEmail)

# Test con respuesta específica
def test_clasificar_email_ventas():
    deps = DepsCRM(usuario_id=42)

    respuesta_esperada = ClasificacionEmail(
        categoria='ventas',
        prioridad=4,
        requiere_respuesta=True,
    )

    with agente_clasificador.override(
        model=TestModel(custom_result_text=respuesta_esperada.model_dump_json())
    ):
        resultado = agente_clasificador.run_sync(
            'Estoy interesado en vuestro plan enterprise.',
            deps=deps,
        )
    assert resultado.data.categoria == 'ventas'
    assert resultado.data.requiere_respuesta is True
```

---

## 2. Validadores personalizados en el output

```python
from pydantic_ai import Agent, ModelRetry
from pydantic import BaseModel, field_validator
from typing import Literal

class PresupuestoLegal(BaseModel):
    concepto: str
    horas_estimadas: float
    coste_hora: float
    total: float
    moneda: Literal['EUR', 'USD', 'GBP']

    @field_validator('total')
    @classmethod
    def total_consistente(cls, v: float, info) -> float:
        """Verifica que total == horas * coste_hora."""
        if 'horas_estimadas' in info.data and 'coste_hora' in info.data:
            esperado = info.data['horas_estimadas'] * info.data['coste_hora']
            if abs(v - esperado) > 0.01:
                raise ValueError(f'Total {v} no coincide con {esperado:.2f} (horas × coste)')
        return v

    @field_validator('horas_estimadas')
    @classmethod
    def horas_razonables(cls, v: float) -> float:
        if v <= 0 or v > 500:
            raise ValueError(f'Horas estimadas {v} fuera de rango (0-500)')
        return v

agente_presupuesto = Agent(
    model='anthropic:claude-sonnet-4-6',  # Sonnet para tareas que requieren consistencia numérica
    result_type=PresupuestoLegal,
    system_prompt='Genera presupuestos legales detallados y matemáticamente consistentes.',
    retries=3,  # reintenta si la validación falla
)

# Si el modelo devuelve un total incorrecto, PydanticAI le enviará el error
# y reintentará hasta max_retries veces
resultado = agente_presupuesto.run_sync(
    'Presupuesto para revisión de contrato de 15 páginas. Tarifa: 150€/hora.'
)
p = resultado.data
print(f'{p.concepto}: {p.horas_estimadas}h × {p.coste_hora}€ = {p.total}€')
```

---

## 3. Agentes anidados — orquestador + trabajadores

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class ResumenDocumento(BaseModel):
    titulo: str
    puntos_clave: list[str]
    conclusion: str

class InformeCompleto(BaseModel):
    resumen_ejecutivo: str
    analisis_por_seccion: list[str]
    recomendaciones: list[str]
    siguiente_paso: str

# Agente especializado en resúmenes
agente_resumen = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=ResumenDocumento,
    system_prompt='Eres especialista en resumir documentos legales.',
)

# Agente orquestador que puede usar el agente de resumen como herramienta
agente_orquestador = Agent(
    model='anthropic:claude-sonnet-4-6',
    result_type=InformeCompleto,
    system_prompt='Produces informes completos combinando análisis de múltiples secciones.',
)

@agente_orquestador.tool_plain
async def resumir_seccion(texto_seccion: str) -> dict:
    """Delega el resumen de una sección al agente especializado."""
    resultado = await agente_resumen.run(texto_seccion)
    return resultado.data.model_dump()

# El orquestador decide cuándo y cómo usar el agente de resumen
async def generar_informe(documento_completo: str) -> InformeCompleto:
    resultado = await agente_orquestador.run(
        f'Genera un informe completo de este documento:\n\n{documento_completo}'
    )
    return resultado.data
```

---

## 4. Reintentos con ModelRetry

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel

class DatosEmpresa(BaseModel):
    nombre: str
    cif: str
    activa: bool

agente_verificador = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=DatosEmpresa,
)

@agente_verificador.tool_plain
def verificar_cif_formato(cif: str) -> str:
    """Verifica que el CIF tenga formato español válido."""
    import re
    if re.match(r'^[A-Z]\d{7}[A-Z0-9]$', cif):
        return f'CIF {cif} tiene formato válido'
    # ModelRetry indica al agente que corrija el error y reintente
    raise ModelRetry(f'CIF "{cif}" inválido. Formato esperado: letra + 7 dígitos + letra/dígito (ej: B12345678)')

@agente_verificador.result_validator
async def validar_datos(ctx: RunContext[None], resultado: DatosEmpresa) -> DatosEmpresa:
    """Valida el resultado completo antes de devolverlo."""
    if not resultado.cif.startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')):
        raise ModelRetry(
            f'El CIF de una empresa debe empezar por A-H. Recibido: {resultado.cif}'
        )
    return resultado
```

---

## 5. Despliegue con FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel
from pydantic_ai import Agent
import httpx

# Modelos de la API
class SolicitudAnalisis(PydanticBaseModel):
    texto_contrato: str
    tipo: str = 'general'

class RespuestaAnalisis(PydanticBaseModel):
    riesgo: str
    clausulas_criticas: list[str]
    recomendacion: str

# Agente
from dataclasses import dataclass

@dataclass
class DepsApp:
    cliente_http: httpx.AsyncClient

agente_api = Agent(
    model='anthropic:claude-haiku-4-5-20251001',
    result_type=RespuestaAnalisis,
    deps_type=DepsApp,
    system_prompt='Analiza contratos e identifica riesgos legales.',
)

# FastAPI
app = FastAPI(title='API de Análisis de Contratos')
cliente_http_global = httpx.AsyncClient()

@app.post('/analizar', response_model=RespuestaAnalisis)
async def analizar_contrato(solicitud: SolicitudAnalisis):
    deps = DepsApp(cliente_http=cliente_http_global)
    try:
        resultado = await agente_api.run(
            f'Tipo: {solicitud.tipo}\n\nContrato:\n{solicitud.texto_contrato}',
            deps=deps,
        )
        return resultado.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    return {'status': 'ok'}

# Streaming endpoint
from fastapi.responses import StreamingResponse

@app.post('/analizar/stream')
async def analizar_stream(solicitud: SolicitudAnalisis):
    agente_texto = Agent(model='anthropic:claude-haiku-4-5-20251001')

    async def generar():
        async with agente_texto.run_stream(solicitud.texto_contrato) as stream:
            async for chunk in stream.stream_text(delta=True):
                yield chunk

    return StreamingResponse(generar(), media_type='text/plain')
```

---

## 6. Observabilidad con Logfire

PydanticAI integra con Logfire (de los mismos autores) para tracing automático:

```python
import logfire
from pydantic_ai import Agent

logfire.configure()  # usa LOGFIRE_TOKEN del entorno
logfire.instrument_pydantic_ai()  # tracing automático de todos los agentes

agente = Agent(model='anthropic:claude-haiku-4-5-20251001')

with logfire.span('analizar-contrato'):
    resultado = agente.run_sync('Analiza esta cláusula...')
    logfire.info('Análisis completado', tokens=resultado.usage().total_tokens)
```

---

→ Siguiente: [Mastra.ai — framework TypeScript para agentes](03-mastra-ai.md)
