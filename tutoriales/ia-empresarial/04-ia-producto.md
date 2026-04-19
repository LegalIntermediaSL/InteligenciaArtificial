# IA en Producto: User Research, Priorización y A/B Testing

## IA como copiloto del Product Manager

El PM moderno con IA puede hacer el trabajo de síntesis en horas en lugar de semanas,
pero la estrategia y el juicio sobre qué construir siguen siendo humanos.

```
DÓNDE APORTA MÁS LA IA EN PRODUCTO
─────────────────────────────────────
Alto impacto:
  ├── Síntesis de entrevistas y feedback de usuarios
  ├── Análisis de datos cualitativos a escala
  ├── Generación de hipótesis de producto
  └── Redacción de specs y user stories

Impacto medio:
  ├── Priorización asistida (con marcos como RICE, ICE)
  ├── Análisis de competidores
  └── Análisis de reviews de app stores

Impacto bajo (aún requiere humano):
  ├── Estrategia de producto a largo plazo
  ├── Decisiones sobre trade-offs de negocio
  └── Visión de producto
```

## Síntesis de entrevistas de usuario

```python
import anthropic
import json

client = anthropic.Anthropic()

def sintetizar_entrevistas(transcripciones: list[str], pregunta_investigacion: str) -> dict:
    """Sintetiza múltiples entrevistas de usuario en insights accionables."""

    # Procesar en batch si hay muchas (aquí procesamos en un prompt)
    texto_entrevistas = "\n\n---ENTREVISTA---\n".join(
        f"Entrevista {i+1}:\n{t[:1500]}"
        for i, t in enumerate(transcripciones)
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system="""Eres un UX researcher experto en síntesis de investigación cualitativa.
Identificas patrones, necesidades no satisfechas y oportunidades de producto.""",
        messages=[{
            "role": "user",
            "content": f"""Sintetiza estas {len(transcripciones)} entrevistas de usuario.

PREGUNTA DE INVESTIGACIÓN: {pregunta_investigacion}

ENTREVISTAS:
{texto_entrevistas}

Devuelve JSON con:
{{
  "temas_principales": [
    {{
      "tema": "nombre del tema",
      "frecuencia": "cuántas personas lo mencionaron",
      "citas_representativas": ["cita textual 1", "cita textual 2"],
      "insight": "qué significa para el producto"
    }}
  ],
  "necesidades_no_satisfechas": ["necesidad 1", "necesidad 2"],
  "oportunidades_producto": ["oportunidad 1", "oportunidad 2"],
  "segmentos_usuarios": [
    {{"perfil": "descripción", "necesidades_especificas": ["..."]}}
  ],
  "hipotesis_a_validar": ["hipótesis 1", "hipótesis 2"]
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"sintesis": texto}


def generar_guion_entrevista(objetivo: str, perfil_usuario: str, duracion_min: int = 45) -> str:
    """Genera un guion de entrevista de usuario estructurado."""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Crea un guion de entrevista de usuario para:
- Objetivo: {objetivo}
- Perfil del entrevistado: {perfil_usuario}
- Duración: {duracion_min} minutos

El guion debe incluir:
1. Apertura y rapport (5 min)
2. Preguntas de contexto sobre el usuario (10 min)
3. Preguntas sobre el problema/comportamiento actual (15 min)
4. Exploración de soluciones actuales y pain points (10 min)
5. Cierre y próximos pasos (5 min)

Usa preguntas abiertas. Evita preguntas sugestivas."""
        }]
    )
    return resp.content[0].text
```

## Priorización de features con IA

```python
def puntuar_feature_rice(feature: dict) -> dict:
    """Calcula el score RICE de una feature con ayuda de IA para estimaciones."""

    # RICE = (Reach * Impact * Confidence) / Effort

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Evalúa esta feature con el marco RICE para priorización de producto:

Feature: {feature['nombre']}
Descripción: {feature.get('descripcion', '')}
Contexto adicional: {feature.get('contexto', '')}

Estima con razonamiento:
- Reach (usuarios afectados por trimestre): número
- Impact (escala 0.25/0.5/1/2/3): número
- Confidence (% certeza sobre estimaciones): porcentaje
- Effort (persona-semanas): número

Devuelve JSON:
{{
  "reach": 500,
  "impact": 2,
  "confidence": 70,
  "effort": 3,
  "rice_score": 0,
  "razonamiento": "breve explicación de las estimaciones"
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        resultado = json.loads(texto)
        r = resultado.get("reach", 0)
        i = resultado.get("impact", 0)
        c = resultado.get("confidence", 0) / 100
        e = resultado.get("effort", 1)
        resultado["rice_score"] = round(r * i * c / e, 1)
        return resultado
    except json.JSONDecodeError:
        return {"error": texto}


def priorizar_backlog(features: list[dict]) -> list[dict]:
    """Prioriza el backlog completo con RICE y genera resumen."""

    features_puntuadas = []
    for f in features:
        puntuacion = puntuar_feature_rice(f)
        f["rice"] = puntuacion
        features_puntuadas.append(f)

    # Ordenar por RICE score
    features_ordenadas = sorted(
        features_puntuadas,
        key=lambda x: x["rice"].get("rice_score", 0),
        reverse=True
    )

    return features_ordenadas
```

## Generación de specs y user stories

```python
def generar_user_story(
    problema_usuario: str,
    perfil_usuario: str,
    contexto_producto: str
) -> dict:
    """Genera user story completa con criterios de aceptación."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": f"""Genera una user story completa para este problema:

PROBLEMA: {problema_usuario}
PERFIL USUARIO: {perfil_usuario}
CONTEXTO DEL PRODUCTO: {contexto_producto}

Devuelve JSON con:
{{
  "titulo": "título conciso",
  "user_story": "Como [tipo de usuario], quiero [acción], para [beneficio]",
  "criterios_aceptacion": [
    "Dado [contexto], cuando [acción], entonces [resultado esperado]"
  ],
  "out_of_scope": ["qué NO incluye esta story"],
  "consideraciones_ux": ["punto de diseño 1", "punto de diseño 2"],
  "metricas_exito": ["métrica 1", "métrica 2"],
  "estimacion_complejidad": "XS/S/M/L/XL",
  "dependencias": ["dependencia técnica si aplica"]
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"user_story": texto}


def generar_prd_seccion(feature_nombre: str, user_stories: list[dict]) -> str:
    """Genera una sección de PRD (Product Requirements Document)."""

    stories_texto = "\n".join(
        f"- {s.get('user_story', s.get('titulo', ''))}"
        for s in user_stories
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Redacta la sección de PRD para la feature: {feature_nombre}

User stories a cubrir:
{stories_texto}

La sección debe incluir:
1. Objetivo y motivación (por qué)
2. Descripción funcional (qué)
3. Flujos principales de usuario
4. Requisitos no funcionales relevantes
5. Métricas de éxito y cómo medirlas
6. Riesgos y mitigaciones

Formato: prosa técnica clara, con bullet points para listas."""
        }]
    )
    return resp.content[0].text
```

## Análisis de reviews para insights de producto

```python
def analizar_reviews_app_store(reviews: list[dict], n_max: int = 50) -> dict:
    """Analiza reviews de app store para extraer insights de producto."""

    # Tomar muestra representativa
    reviews_texto = "\n".join(
        f"[{r.get('rating', '?')}★] {r.get('texto', '')[:200]}"
        for r in reviews[:n_max]
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Analiza estas {min(len(reviews), n_max)} reviews de app store:

{reviews_texto}

Extrae insights de producto en JSON:
{{
  "problemas_frecuentes": [
    {{"problema": "descripción", "menciones_estimadas": 5, "impacto_rating": "negativo"}}
  ],
  "funcionalidades_mas_valoradas": ["feature 1", "feature 2"],
  "solicitudes_mas_pedidas": ["feature solicitada 1"],
  "segmentos_detectados": [
    {{"segmento": "tipo de usuario", "necesidades": ["..."]}}
  ],
  "sentiment_general": "positivo/mixto/negativo",
  "acciones_producto_prioritarias": [
    {{"accion": "qué hacer", "impacto_esperado": "en qué métrica"}}
  ]
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"analisis": texto}
```

## Diseño de experimentos A/B

```python
def disenar_experimento_ab(hipotesis: str, metrica_objetivo: str, contexto: str) -> dict:
    """Diseña un experimento A/B estructurado para validar una hipótesis de producto."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Diseña un experimento A/B para validar esta hipótesis:

HIPÓTESIS: {hipotesis}
MÉTRICA OBJETIVO: {metrica_objetivo}
CONTEXTO: {contexto}

Devuelve JSON con el diseño del experimento:
{{
  "hipotesis_nula": "H0: ...",
  "hipotesis_alternativa": "H1: ...",
  "variante_control": "descripción del control",
  "variante_tratamiento": "descripción del tratamiento",
  "metrica_primaria": "nombre y cómo medirla",
  "metricas_guardianes": ["métrica que no debe empeorar"],
  "tamaño_muestra_estimado": 1000,
  "duracion_dias": 14,
  "criterio_exito": "qué resultado valida la hipótesis",
  "riesgos_del_experimento": ["riesgo 1"],
  "consideraciones_segmentacion": "a qué usuarios exponer"
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"diseno": texto}
```

## Recursos

- [Notebook interactivo](../notebooks/ia-empresarial/04-ia-producto.ipynb)
- [Lenny's Newsletter — Product with AI](https://www.lennysnewsletter.com)
- [Nielsen Norman Group — AI in UX research](https://www.nngroup.com/articles/ai-ux-research/)
- [Evan Miller — Calculadora de tamaño muestral A/B](https://www.evanmiller.org/ab-testing/sample-size.html)
