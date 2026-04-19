# IA en Despachos Legales y Cumplimiento Normativo

## El contexto legal de la IA

La IA en entornos legales opera bajo tres tensiones simultáneas:

- **Precisión vs. velocidad**: los errores legales tienen consecuencias graves
- **Confidencialidad vs. utilidad**: los datos de clientes son sensibles
- **Automatización vs. responsabilidad**: el abogado sigue siendo el responsable final

```
CASOS DE USO LEGALES POR RIESGO
─────────────────────────────────
BAJO RIESGO (empezar aquí):
  ├── Resumen de documentos largos
  ├── Búsqueda en base de conocimiento jurídica
  ├── Borradores de comunicaciones rutinarias
  └── Extracción de datos de contratos

MEDIO RIESGO (con revisión humana):
  ├── Análisis de riesgos contractuales
  ├── Due diligence automatizada
  └── Generación de cláusulas estándar

ALTO RIESGO (solo apoyo, no sustitución):
  ├── Estrategia procesal
  ├── Asesoramiento jurídico vinculante
  └── Decisiones con impacto en derechos
```

## Análisis de contratos con IA

```python
import anthropic
import json

client = anthropic.Anthropic()

CLAUSULAS_CRITICAS = [
    "penalización por incumplimiento",
    "propiedad intelectual",
    "resolución de conflictos",
    "limitación de responsabilidad",
    "confidencialidad",
    "rescisión anticipada"
]

def analizar_contrato(texto_contrato: str, tipo_contrato: str = "prestación de servicios") -> dict:
    """Analiza un contrato e identifica riesgos y cláusulas clave."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system="""Eres un asistente de análisis legal. Tu función es identificar y resumir
cláusulas relevantes, señalar riesgos potenciales y marcar lagunas contractuales.
IMPORTANTE: Tu análisis es orientativo. Un abogado debe revisar antes de firmar.""",
        messages=[{
            "role": "user",
            "content": f"""Analiza este contrato de {tipo_contrato} e identifica:

1. Cláusulas críticas presentes y su contenido resumido
2. Riesgos potenciales para el cliente
3. Cláusulas que faltan o están incompletas
4. Puntuación de riesgo global (1-10)

CONTRATO:
{texto_contrato[:5000]}

Responde en JSON con esta estructura:
{{
  "clausulas_encontradas": {{"nombre_clausula": "resumen breve"}},
  "riesgos": [{{"riesgo": "descripción", "severidad": "alta/media/baja"}}],
  "clausulas_faltantes": ["clausula 1", "clausula 2"],
  "puntuacion_riesgo": 7,
  "recomendaciones": ["recomendacion 1", "recomendacion 2"]
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"error": "No se pudo parsear la respuesta", "texto_raw": texto}


def comparar_contratos(contrato_a: str, contrato_b: str) -> str:
    """Compara dos versiones de un contrato e identifica cambios relevantes."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Compara estas dos versiones de un contrato.
Identifica: (1) cambios favorables para nosotros, (2) cambios desfavorables, (3) cambios neutros.

VERSIÓN A (original):
{contrato_a[:2000]}

VERSIÓN B (revisada):
{contrato_b[:2000]}

Sé específico: cita el texto exacto que cambió."""
        }]
    )
    return resp.content[0].text
```

## Sistema de cumplimiento normativo (compliance)

```python
NORMATIVAS_EU = {
    "GDPR": "Reglamento General de Protección de Datos (UE 2016/679)",
    "AI_ACT": "Reglamento de IA de la UE (en vigor desde 2024)",
    "NIS2": "Directiva de seguridad de redes e información",
    "DORA": "Resiliencia operativa digital para el sector financiero"
}

def verificar_cumplimiento(descripcion_proceso: str, normativas: list[str]) -> dict:
    """Evalúa si un proceso cumple con las normativas indicadas."""

    normativas_texto = "\n".join(
        f"- {n}: {NORMATIVAS_EU.get(n, n)}" for n in normativas
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system="Eres un especialista en compliance europeo. Analiza procesos empresariales frente a normativa vigente.",
        messages=[{
            "role": "user",
            "content": f"""Evalúa este proceso frente a las normativas indicadas:

PROCESO:
{descripcion_proceso}

NORMATIVAS A VERIFICAR:
{normativas_texto}

Para cada normativa indica:
1. ¿Cumple en principio? (sí/no/parcialmente)
2. Requisitos específicos que aplican
3. Gaps identificados
4. Acciones correctivas prioritarias

Formato JSON:
{{
  "GDPR": {{
    "cumplimiento": "parcialmente",
    "requisitos_aplican": ["..."],
    "gaps": ["..."],
    "acciones": ["..."]
  }}
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"analisis_texto": texto}


def generar_clausula_gdpr(tipo_tratamiento: str, datos_tratados: list[str]) -> str:
    """Genera una cláusula de protección de datos conforme a GDPR."""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": f"""Genera una cláusula de protección de datos conforme a GDPR para:
- Tipo de tratamiento: {tipo_tratamiento}
- Datos tratados: {', '.join(datos_tratados)}

La cláusula debe incluir: base legal, finalidad, plazo de conservación, derechos del interesado.
Redacta en español jurídico claro. Sin lagunas legales evidentes."""
        }]
    )
    return resp.content[0].text
```

## Due diligence automatizada

```python
CHECKLIST_DD_MERCANTIL = [
    "Constitución y estatutos sociales",
    "Actas de juntas generales (últimos 3 años)",
    "Contratos con clientes principales (>10% facturación)",
    "Contratos laborales y convenios colectivos",
    "Litigios pendientes y contingencias",
    "Propiedad intelectual registrada",
    "Licencias y permisos de actividad",
    "Deuda financiera y garantías",
    "Obligaciones fiscales pendientes"
]

def clasificar_documento_dd(nombre_archivo: str, contenido: str) -> dict:
    """Clasifica un documento en la estructura de due diligence."""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Clasifica este documento en el contexto de una due diligence mercantil.

Archivo: {nombre_archivo}
Contenido (primeras líneas): {contenido[:500]}

Responde en JSON:
{{
  "categoria": "categoria de la checklist",
  "relevancia": "alta/media/baja",
  "hallazgos_clave": ["hallazgo 1"],
  "requiere_revision_abogado": true/false,
  "notas": "observaciones breves"
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"notas": texto}


def generar_informe_dd(hallazgos: list[dict]) -> str:
    """Genera el informe ejecutivo de due diligence."""

    hallazgos_texto = "\n".join(
        f"- [{h['categoria']}] {h.get('notas', '')} (relevancia: {h.get('relevancia', 'N/A')})"
        for h in hallazgos
        if h.get("relevancia") in ("alta", "media")
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": f"""Redacta el resumen ejecutivo de due diligence basándote en estos hallazgos:

{hallazgos_texto}

El resumen debe incluir:
1. Principales riesgos identificados (clasificados por severidad)
2. Contingencias que pueden afectar al precio o condiciones
3. Condiciones precedentes recomendadas antes del cierre
4. Valoración global del riesgo de la operación

Tono: formal y técnico, dirigido a directivos no abogados."""
        }]
    )
    return resp.content[0].text
```

## Investigación jurídica asistida

```python
def investigar_jurisprudencia(pregunta_legal: str, jurisdiccion: str = "España") -> str:
    """Asiste en la investigación jurídica con fuentes citadas."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=f"""Eres un investigador jurídico especializado en derecho de {jurisdiccion}.
IMPORTANTE: Indica siempre cuando citas jurisprudencia o doctrina que el abogado debe verificar
en las fuentes oficiales (CENDOJ, BOE, etc.) antes de usarla en argumentaciones.""",
        messages=[{
            "role": "user",
            "content": f"""Investiga la siguiente cuestión jurídica:

{pregunta_legal}

Incluye:
1. Marco normativo aplicable (artículos concretos)
2. Doctrina jurisprudencial relevante (con advertencia de verificación)
3. Posibles líneas argumentales
4. Bibliografía especializada recomendada"""
        }]
    )
    return resp.content[0].text
```

## Consideraciones éticas y de responsabilidad

```
PRINCIPIOS PARA IA LEGAL RESPONSABLE
──────────────────────────────────────
1. SUPERVISIÓN HUMANA OBLIGATORIA
   → El abogado revisa y firma toda pieza legal
   → La IA es herramienta, no sustituto del criterio jurídico

2. CONFIDENCIALIDAD DE DATOS
   → No enviar datos de clientes identificables a APIs externas sin consentimiento
   → Considerar despliegue on-premise para datos muy sensibles
   → Anonimizar o pseudonimizar cuando sea posible

3. TRAZABILIDAD
   → Documentar qué análisis se hicieron con IA
   → Guardar prompts y respuestas para auditoría
   → El expediente debe reflejar el trabajo humano de verificación

4. FORMACIÓN DEL EQUIPO
   → Los abogados deben entender las limitaciones del modelo
   → Política interna de uso aceptable de IA
   → Protocolo para detectar alucinaciones en citas legales
```

## Recursos

- [Notebook interactivo](../notebooks/ia-empresarial/02-ia-legal-compliance.ipynb)
- [Reglamento de IA de la UE — texto oficial](https://artificialintelligenceact.eu/es/)
- [CENDOJ — Jurisprudencia española](https://www.poderjudicial.es/cgpj/es/Temas/CENDOJ)
- [Guía AEPD sobre IA y GDPR](https://www.aepd.es/es/prensa-y-comunicacion/notas-de-prensa/la-aepd-publica-una-guia-sobre-ia-y-proteccion-de-datos)
