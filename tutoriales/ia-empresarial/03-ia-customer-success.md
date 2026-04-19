# IA en Customer Success: Retención y Predicción de Churn

## Por qué CS es el caso de uso más rentable de la IA

En SaaS B2B, retener un cliente cuesta 5-7x menos que adquirir uno nuevo.
Customer Success (CS) con IA permite escalar el trabajo de retención sin escalar el equipo.

```
IMPACTO TÍPICO DE IA EN CS
───────────────────────────
Detección temprana de churn:        -25% a -40% en tasa de cancelación
Tiempo de respuesta a tickets:      -60% con triaje automático
Cobertura de cuentas por CSM:       +3x con asistencia IA
NPS y CSAT:                         +10-15 puntos en 6 meses
```

## Predicción de churn con señales de comportamiento

```python
import anthropic
import json
from datetime import datetime, timedelta

client = anthropic.Anthropic()

def analizar_riesgo_cliente(datos_cliente: dict) -> dict:
    """Analiza el riesgo de churn de un cliente basado en sus señales de comportamiento."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system="""Eres un especialista en Customer Success con experiencia en predicción de churn SaaS.
Analiza señales de comportamiento y evalúa el riesgo de cancelación.""",
        messages=[{
            "role": "user",
            "content": f"""Evalúa el riesgo de churn de este cliente:

DATOS DEL CLIENTE:
{json.dumps(datos_cliente, ensure_ascii=False, indent=2)}

Devuelve JSON con:
{{
  "score_riesgo": 0-100,
  "nivel_riesgo": "crítico/alto/medio/bajo",
  "señales_preocupantes": ["señal 1", "señal 2"],
  "señales_positivas": ["señal 1"],
  "accion_recomendada": "descripción de la siguiente acción",
  "urgencia": "inmediata/esta_semana/este_mes",
  "argumentos_retencion": ["argumento 1", "argumento 2"]
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


# Señales típicas de churn en SaaS B2B
def construir_perfil_cliente(
    nombre: str,
    plan: str,
    mrr: float,
    logins_ultimo_mes: int,
    logins_mes_anterior: int,
    features_activas: int,
    features_total: int,
    tickets_abiertos: int,
    nps_score: int | None,
    dias_desde_ultima_sesion: int,
    renovacion_dias: int
) -> dict:
    """Construye el perfil de riesgo de un cliente."""
    return {
        "nombre": nombre,
        "plan": plan,
        "mrr_euros": mrr,
        "uso": {
            "logins_ultimo_mes": logins_ultimo_mes,
            "cambio_logins_pct": round((logins_ultimo_mes - logins_mes_anterior) / max(logins_mes_anterior, 1) * 100, 1),
            "adopcion_features_pct": round(features_activas / features_total * 100),
            "dias_desde_ultima_sesion": dias_desde_ultima_sesion
        },
        "soporte": {
            "tickets_abiertos": tickets_abiertos,
        },
        "satisfaccion": {
            "nps": nps_score
        },
        "contrato": {
            "dias_para_renovacion": renovacion_dias
        }
    }

# Ejemplo
cliente = construir_perfil_cliente(
    nombre="Acme Corp",
    plan="Professional",
    mrr=1200,
    logins_ultimo_mes=8,
    logins_mes_anterior=45,
    features_activas=3,
    features_total=12,
    tickets_abiertos=2,
    nps_score=5,
    dias_desde_ultima_sesion=12,
    renovacion_dias=45
)

riesgo = analizar_riesgo_cliente(cliente)
print(f"Score de riesgo: {riesgo['score_riesgo']}/100 ({riesgo['nivel_riesgo']})")
print(f"Urgencia: {riesgo['urgencia']}")
print(f"Acción: {riesgo['accion_recomendada']}")
```

## Generación de Business Reviews personalizadas

```python
def generar_qbr(datos_cuenta: dict, logros_trimestre: list[str], objetivos_cliente: list[str]) -> str:
    """Genera el borrador de una Quarterly Business Review personalizada."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Genera el guion de una Quarterly Business Review (QBR) para este cliente.

DATOS DE CUENTA:
{json.dumps(datos_cuenta, ensure_ascii=False, indent=2)}

LOGROS DEL TRIMESTRE:
{chr(10).join(f'- {l}' for l in logros_trimestre)}

OBJETIVOS DECLARADOS DEL CLIENTE:
{chr(10).join(f'- {o}' for o in objetivos_cliente)}

El QBR debe tener:
1. Apertura personalizada (conecta con los objetivos del cliente)
2. Resumen de uso y adopción (positivo primero)
3. ROI demostrado con datos concretos
4. Retos identificados y plan de acción
5. Roadmap de próximas funcionalidades relevantes para ellos
6. Siguiente paso concreto antes de cerrar la reunión

Tono: consultivo, orientado al negocio del cliente, no al producto."""
        }]
    )
    return resp.content[0].text
```

## Triaje y respuesta automática de tickets

```python
CATEGORIAS_TICKET = {
    "bug_critico": {"prioridad": 1, "sla_horas": 2},
    "bug_menor": {"prioridad": 2, "sla_horas": 24},
    "pregunta_funcionalidad": {"prioridad": 3, "sla_horas": 48},
    "solicitud_nueva_feature": {"prioridad": 4, "sla_horas": 72},
    "facturacion": {"prioridad": 2, "sla_horas": 24},
    "onboarding": {"prioridad": 2, "sla_horas": 4}
}

def clasificar_y_responder_ticket(
    asunto: str,
    mensaje: str,
    historial_cliente: str,
    base_conocimiento: str
) -> dict:
    """Clasifica un ticket, genera respuesta borrador y asigna prioridad."""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=900,
        system="""Eres un agente de Customer Success experto. Clasificas tickets,
generas respuestas empáticas y precisas, y escalas cuando es necesario.""",
        messages=[{
            "role": "user",
            "content": f"""Procesa este ticket de soporte:

ASUNTO: {asunto}
MENSAJE: {mensaje}

HISTORIAL DEL CLIENTE: {historial_cliente[:300]}
BASE DE CONOCIMIENTO RELEVANTE: {base_conocimiento[:500]}

Responde en JSON:
{{
  "categoria": "bug_critico/bug_menor/pregunta_funcionalidad/solicitud_nueva_feature/facturacion/onboarding",
  "sentimiento": "frustrado/neutro/positivo",
  "escalar_a_humano": true/false,
  "motivo_escalado": "motivo si aplica",
  "respuesta_borrador": "texto de respuesta al cliente",
  "acciones_internas": ["acción para el equipo"]
}}"""
        }]
    )

    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")

    try:
        resultado = json.loads(texto)
        categoria = resultado.get("categoria", "pregunta_funcionalidad")
        resultado["sla_horas"] = CATEGORIAS_TICKET.get(categoria, {}).get("sla_horas", 48)
        return resultado
    except json.JSONDecodeError:
        return {"respuesta_borrador": texto}
```

## Dashboard de salud de cartera

```python
def calcular_health_score(cliente: dict) -> int:
    """Calcula el health score de un cliente (0-100)."""

    pesos = {
        "uso": 0.35,        # Frecuencia y profundidad de uso
        "adopcion": 0.25,   # Porcentaje de features activas
        "satisfaccion": 0.20, # NPS o CSAT
        "soporte": 0.10,    # Tickets sin resolver
        "engagement": 0.10  # Asistencia a webinars, comunidad
    }

    scores = {
        "uso": min(100, cliente.get("logins_mes", 0) * 2),
        "adopcion": cliente.get("adopcion_pct", 0),
        "satisfaccion": max(0, (cliente.get("nps", 0) + 100) / 2),
        "soporte": max(0, 100 - cliente.get("tickets_abiertos", 0) * 20),
        "engagement": cliente.get("engagement_score", 50)
    }

    return round(sum(scores[k] * v for k, v in pesos.items()))


def segmentar_cartera(clientes: list[dict]) -> dict:
    """Segmenta la cartera de clientes por health score."""

    segmentos = {"champion": [], "saludable": [], "en_riesgo": [], "critico": []}

    for c in clientes:
        score = calcular_health_score(c)
        c["health_score"] = score
        if score >= 80:
            segmentos["champion"].append(c)
        elif score >= 60:
            segmentos["saludable"].append(c)
        elif score >= 40:
            segmentos["en_riesgo"].append(c)
        else:
            segmentos["critico"].append(c)

    return segmentos
```

## Métricas clave de CS con IA

```
MÉTRICAS A MONITORIZAR
──────────────────────
Retención:
  • Net Revenue Retention (NRR) — objetivo: >110%
  • Gross Revenue Retention (GRR) — objetivo: >90%
  • Churn rate mensual — objetivo: <2% en SaaS B2B

Eficiencia del equipo CS:
  • Cuentas por CSM (con IA: 150-200, sin IA: 50-80)
  • Tiempo hasta primera respuesta — objetivo: <2h
  • Tasa de resolución en primer contacto — objetivo: >70%

Calidad:
  • CSAT de tickets — objetivo: >4.2/5
  • NPS — objetivo: >40
  • Time to Value (TTV) para nuevos clientes
```

## Recursos

- [Notebook interactivo](../notebooks/ia-empresarial/03-ia-customer-success.ipynb)
- [Gainsight — The State of Customer Success](https://www.gainsight.com/resources/)
- [ChurnZero — Churn prediction best practices](https://churnzero.com/resources/)
