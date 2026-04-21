# Análisis Automático de Contratos con IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/01-analisis-contratos.ipynb)

Los contratos son el núcleo de cualquier operación empresarial. Una cartera mediana puede contener cientos de contratos activos con fechas de vencimiento, penalizaciones y cláusulas de riesgo dispersas en miles de páginas. Claude permite automatizar la extracción estructurada, la evaluación de riesgo y la comparación de versiones a escala.

> **Aviso:** Los ejemplos son ilustrativos; no constituyen asesoramiento jurídico.

---

## 1. Tipos de análisis que cubre este artículo

| Tipo | Descripción | Caso de uso |
|------|-------------|-------------|
| **Extracción de cláusulas** | Identificar y estructurar las partes esenciales del contrato | Onboarding de contratos al sistema |
| **Scoring de riesgo** | Puntuar cada cláusula del 1 al 5 según su exposición | Priorizar revisión del equipo legal |
| **Diff semántico** | Comparar dos versiones e identificar cambios relevantes | Renegociaciones, renovaciones |
| **Pipeline de cartera** | Procesar 100+ contratos y generar resumen ejecutivo | Auditorías, M&A |
| **Alertas de vencimiento** | Detectar contratos próximos a vencer o con hitos relevantes | Gestión proactiva |

El modelo `claude-sonnet-4-6` ofrece el equilibrio óptimo entre capacidad de razonamiento legal y coste por token para este tipo de tareas.

---

## 2. Schema Pydantic para análisis de contrato

Definir un schema robusto es el primer paso: garantiza que la salida de Claude sea siempre estructurada y validable.

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import IntEnum

class NivelRiesgo(IntEnum):
    BAJO = 1
    MODERADO_BAJO = 2
    MODERADO = 3
    MODERADO_ALTO = 4
    ALTO = 5

class ClausulaContrato(BaseModel):
    tipo: str = Field(description="Tipo de cláusula: partes, objeto, plazo, precio, penalización, ley_aplicable, otra")
    contenido_resumido: str = Field(description="Resumen de la cláusula en 1-3 frases")
    numero_clausula: Optional[str] = Field(default=None, description="Referencia al número o título de la cláusula original")
    fecha_relevante: Optional[date] = Field(default=None, description="Fecha de inicio, vencimiento o hito si aplica")

class RiesgoDetectado(BaseModel):
    clausula_referencia: str = Field(description="Tipo o número de la cláusula analizada")
    nivel: NivelRiesgo = Field(description="Nivel de riesgo del 1 (bajo) al 5 (alto)")
    justificacion: str = Field(description="Explicación del riesgo en 1-2 frases")
    recomendacion: str = Field(description="Acción sugerida para mitigar el riesgo")

class AnalisisContrato(BaseModel):
    titulo_contrato: str
    partes: list[str] = Field(description="Lista de partes contratantes")
    objeto: str = Field(description="Objeto del contrato en 1-2 frases")
    fecha_inicio: Optional[date] = None
    fecha_vencimiento: Optional[date] = None
    valor_total: Optional[str] = Field(default=None, description="Importe total o rango si está especificado")
    clausulas: list[ClausulaContrato]
    riesgos: list[RiesgoDetectado]
    riesgo_global: NivelRiesgo
    resumen_ejecutivo: str = Field(description="Síntesis del contrato en 3-5 frases para dirección")
```

---

## 3. Extracción de cláusulas con tool_use

`tool_use` permite que Claude devuelva estructuras JSON validadas en lugar de texto libre. Definimos la herramienta a partir del schema Pydantic:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Convertir el schema Pydantic a la definición de herramienta de Claude
herramienta_analisis = {
    "name": "analizar_contrato",
    "description": "Extrae y estructura toda la información relevante de un contrato legal",
    "input_schema": AnalisisContrato.model_json_schema()
}

def analizar_contrato(texto_contrato: str) -> AnalisisContrato:
    """Analiza un contrato y devuelve un AnalisisContrato validado."""
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        tools=[herramienta_analisis],
        tool_choice={"type": "tool", "name": "analizar_contrato"},
        messages=[{
            "role": "user",
            "content": f"""Eres un abogado mercantilista experto. Analiza el siguiente contrato
            y extrae toda la información estructurada solicitada. Sé preciso en las fechas
            y conservador en el scoring de riesgo (escala 1-5).

            CONTRATO:
            {texto_contrato}"""
        }]
    )

    # Extraer el resultado de la llamada a la herramienta
    for bloque in respuesta.content:
        if bloque.type == "tool_use":
            return AnalisisContrato.model_validate(bloque.input)

    raise ValueError("Claude no devolvió una llamada a herramienta válida")
```

---

## 4. Scoring de riesgo por cláusula

El scoring evalúa cada cláusula individualmente con contexto sobre por qué es arriesgada:

```python
CRITERIOS_RIESGO = {
    "penalizacion": "Evalúa si la penalización es desproporcionada, automática o de difícil impugnación",
    "ley_aplicable": "Verifica si la jurisdicción es desfavorable o genera costes de litigio altos",
    "plazo": "Detecta renovaciones automáticas sin preaviso, plazos de preaviso muy cortos",
    "precio": "Identifica fórmulas de revisión unilateral, indexación desfavorable, pagos anticipados",
    "objeto": "Comprueba si el alcance es ambiguo o incluye obligaciones desproporcionadas",
}

def score_clausula(clausula: str, tipo: str, contexto_empresa: str = "") -> RiesgoDetectado:
    """Puntúa una cláusula específica del 1 al 5."""
    criterio = CRITERIOS_RIESGO.get(tipo, "Evalúa el riesgo general de esta cláusula")

    herramienta_riesgo = {
        "name": "puntuar_riesgo",
        "description": "Puntúa el riesgo de una cláusula contractual",
        "input_schema": RiesgoDetectado.model_json_schema()
    }

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[herramienta_riesgo],
        tool_choice={"type": "tool", "name": "puntuar_riesgo"},
        messages=[{
            "role": "user",
            "content": f"""Analiza el riesgo de esta cláusula contractual.

            Tipo: {tipo}
            Criterio de evaluación: {criterio}
            Contexto de la empresa: {contexto_empresa or 'Empresa general sin información adicional'}

            CLÁUSULA:
            {clausula}

            Asigna una puntuación de 1 (riesgo mínimo) a 5 (riesgo crítico)."""
        }]
    )

    for bloque in respuesta.content:
        if bloque.type == "tool_use":
            return RiesgoDetectado.model_validate(bloque.input)

    raise ValueError("No se pudo puntuar la cláusula")
```

---

## 5. Diff semántico entre dos versiones del contrato

A diferencia de un diff de texto, el diff semántico identifica si los cambios son materiales:

```python
from dataclasses import dataclass

@dataclass
class CambioSemantico:
    clausula: str
    version_anterior: str
    version_nueva: str
    es_material: bool
    impacto: str  # 'favorable', 'desfavorable', 'neutro'
    descripcion: str

def diff_semantico(contrato_v1: str, contrato_v2: str) -> list[CambioSemantico]:
    """
    Compara dos versiones del mismo contrato y devuelve los cambios materiales.
    Ignora cambios de formato, numeración o redacción sin impacto sustantivo.
    """
    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Compara estas dos versiones del mismo contrato. Identifica únicamente
            los cambios MATERIALES (que afecten a derechos, obligaciones, plazos o importes).
            Ignora cambios puramente formales o de redacción sin consecuencias sustantivas.

            Para cada cambio, indica si es favorable, desfavorable o neutro para la primera parte firmante.

            Devuelve tu análisis en JSON con esta estructura exacta:
            {{
              "cambios": [
                {{
                  "clausula": "nombre o número de la cláusula",
                  "version_anterior": "texto relevante de la versión 1",
                  "version_nueva": "texto relevante de la versión 2",
                  "es_material": true,
                  "impacto": "favorable|desfavorable|neutro",
                  "descripcion": "explicación del impacto en 1-2 frases"
                }}
              ]
            }}

            VERSIÓN 1:
            {contrato_v1}

            VERSIÓN 2:
            {contrato_v2}"""
        }]
    )

    resultado = json.loads(respuesta.content[0].text)
    return [CambioSemantico(**c) for c in resultado["cambios"]]
```

---

## 6. Pipeline para cartera de contratos

Procesar 100 contratos requiere gestión de concurrencia y consolidación de resultados:

```python
import asyncio
from pathlib import Path
from typing import Generator

async def analizar_contrato_async(
    ruta: Path,
    semaforo: asyncio.Semaphore
) -> tuple[str, AnalisisContrato | Exception]:
    """Analiza un contrato de forma asíncrona con control de concurrencia."""
    async with semaforo:
        try:
            texto = ruta.read_text(encoding="utf-8")
            # Truncar si supera el contexto (ajustar según tamaño real)
            if len(texto) > 180_000:
                texto = texto[:180_000] + "\n[... documento truncado ...]"
            resultado = analizar_contrato(texto)  # función síncrona del paso 3
            return (ruta.name, resultado)
        except Exception as e:
            return (ruta.name, e)

async def procesar_cartera(directorio: str, max_concurrentes: int = 5) -> dict:
    """Procesa todos los .txt del directorio y devuelve análisis + resumen ejecutivo."""
    rutas = list(Path(directorio).glob("*.txt"))
    semaforo = asyncio.Semaphore(max_concurrentes)

    tareas = [analizar_contrato_async(r, semaforo) for r in rutas]
    resultados_raw = await asyncio.gather(*tareas)

    exitosos = [(n, r) for n, r in resultados_raw if isinstance(r, AnalisisContrato)]
    fallidos = [(n, r) for n, r in resultados_raw if isinstance(r, Exception)]

    # Calcular estadísticas de cartera
    riesgos_globales = [r.riesgo_global for _, r in exitosos]
    contratos_alto_riesgo = [(n, r) for n, r in exitosos if r.riesgo_global >= 4]

    # Vencimientos próximos (próximos 90 días)
    from datetime import date, timedelta
    hoy = date.today()
    ventana = hoy + timedelta(days=90)
    proximos_vencimientos = [
        (n, r) for n, r in exitosos
        if r.fecha_vencimiento and hoy <= r.fecha_vencimiento <= ventana
    ]

    return {
        "total_procesados": len(exitosos),
        "total_fallidos": len(fallidos),
        "contratos_alto_riesgo": contratos_alto_riesgo,
        "proximos_vencimientos": proximos_vencimientos,
        "riesgo_medio_cartera": sum(riesgos_globales) / len(riesgos_globales) if riesgos_globales else 0,
        "detalle": dict(exitosos),
        "errores": dict(fallidos),
    }

# Uso:
# resultado = asyncio.run(procesar_cartera("/ruta/a/contratos/"))
```

---

## 7. Integración con sistema de alertas

Una vez procesada la cartera, podemos enviar alertas automáticas:

```python
from datetime import date, timedelta
import smtplib
from email.mime.text import MIMEText

def generar_alerta_vencimientos(
    cartera: dict,
    dias_anticipacion: int = 60,
    destinatarios: list[str] | None = None
) -> str:
    """Genera un informe de vencimientos próximos en formato Markdown."""
    hoy = date.today()
    limite = hoy + timedelta(days=dias_anticipacion)

    lineas = [
        f"# Alerta de Vencimientos — {hoy.strftime('%d/%m/%Y')}",
        f"\nContratos que vencen antes del **{limite.strftime('%d/%m/%Y')}**:\n",
    ]

    if not cartera["proximos_vencimientos"]:
        lineas.append("*No hay vencimientos próximos en este periodo.*")
    else:
        lineas.append("| Contrato | Vencimiento | Riesgo Global | Partes |")
        lineas.append("|----------|-------------|---------------|--------|")
        for nombre, analisis in cartera["proximos_vencimientos"]:
            partes = ", ".join(analisis.partes[:2])
            lineas.append(
                f"| {nombre} | {analisis.fecha_vencimiento} "
                f"| {analisis.riesgo_global}/5 | {partes} |"
            )

    lineas.extend([
        f"\n## Contratos de alto riesgo ({len(cartera['contratos_alto_riesgo'])} total)",
        "\nRequieren revisión prioritaria del equipo legal.",
    ])

    for nombre, analisis in cartera["contratos_alto_riesgo"]:
        lineas.append(f"\n### {nombre}")
        lineas.append(f"**Riesgo:** {analisis.riesgo_global}/5")
        lineas.append(f"**Resumen:** {analisis.resumen_ejecutivo}")
        if analisis.riesgos:
            lineas.append("\n**Riesgos principales:**")
            for riesgo in sorted(analisis.riesgos, key=lambda r: r.nivel, reverse=True)[:3]:
                lineas.append(f"- [{riesgo.nivel}/5] {riesgo.justificacion}")

    return "\n".join(lineas)

def enviar_alerta_email(
    contenido_markdown: str,
    destinatarios: list[str],
    smtp_host: str = "localhost",
    smtp_port: int = 25
) -> None:
    """Envía la alerta por email. Adapta a tu proveedor SMTP."""
    msg = MIMEText(contenido_markdown, "plain", "utf-8")
    msg["Subject"] = f"Alerta Legal — Vencimientos y Riesgos {date.today().strftime('%d/%m/%Y')}"
    msg["From"] = "legal-ia@tuempresa.com"
    msg["To"] = ", ".join(destinatarios)

    with smtplib.SMTP(smtp_host, smtp_port) as servidor:
        servidor.sendmail(msg["From"], destinatarios, msg.as_string())
```

---

## Resumen

En este artículo hemos construido:

1. Un **schema Pydantic completo** (`AnalisisContrato`) con tipos anidados para cláusulas y riesgos
2. Un extractor basado en **tool_use** que garantiza salida estructurada y validable
3. Un **scorer de riesgo** por cláusula con criterios específicos por tipo
4. Un **diff semántico** que filtra cambios materiales de los formales
5. Un **pipeline asíncrono** para carteras de 100+ contratos con control de concurrencia
6. Un **sistema de alertas** que notifica vencimientos y contratos de alto riesgo

El siguiente paso natural es combinar este análisis con un sistema de Q&A sobre la cartera completa, que cubrimos en el artículo de Due Diligence.

---

*Siguiente: [Due Diligence Asistida por IA →](02-due-diligence-ia.md)*
