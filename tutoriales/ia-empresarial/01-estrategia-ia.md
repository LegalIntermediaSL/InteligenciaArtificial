# Estrategia de IA para Empresas

## Marco de evaluación e implementación

Antes de implementar IA, una empresa necesita responder tres preguntas:

1. **¿Dónde aporta más valor?** — Identificar procesos con alto volumen, reglas claras o datos abundantes
2. **¿Estamos listos?** — Evaluar datos disponibles, equipo técnico y cultura organizacional
3. **¿Cómo medimos el éxito?** — Definir KPIs antes de empezar, no después

```
EVALUACIÓN DE MADUREZ PARA IA
─────────────────────────────
Nivel 1: Datos fragmentados, procesos manuales, sin cultura data-driven
Nivel 2: Datos centralizados, primeros dashboards, equipo analítico básico
Nivel 3: ML en producción, casos de uso probados, ROI demostrado
Nivel 4: IA en el núcleo del producto, data flywheel, ventaja competitiva
```

## Identificar casos de uso de alto impacto

```python
import anthropic

client = anthropic.Anthropic()

def evaluar_caso_uso(descripcion: str, contexto_empresa: str) -> dict:
    """Evalúa un caso de uso de IA y devuelve análisis estructurado."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Evalúa este caso de uso de IA para una empresa:

CASO DE USO: {descripcion}
CONTEXTO EMPRESA: {contexto_empresa}

Responde en JSON con esta estructura:
{{
  "viabilidad": "alta/media/baja",
  "impacto_negocio": "alto/medio/bajo",
  "complejidad_tecnica": "alta/media/baja",
  "tiempo_implementacion": "semanas estimadas",
  "riesgos_principales": ["riesgo 1", "riesgo 2"],
  "kpis_sugeridos": ["kpi 1", "kpi 2"],
  "recomendacion": "texto breve"
}}"""
        }]
    )

    import json
    texto = resp.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")
    return json.loads(texto)

# Ejemplo de uso
casos = [
    "Automatizar la clasificación de emails de soporte al cliente",
    "Generar borradores de contratos personalizados",
    "Predecir qué clientes van a cancelar su suscripción"
]

contexto = "SaaS B2B, 50 empleados, 800 clientes, equipo de datos de 2 personas"

for caso in casos:
    analisis = evaluar_caso_uso(caso, contexto)
    print(f"\n{caso}")
    print(f"  Viabilidad: {analisis['viabilidad']} | Impacto: {analisis['impacto_negocio']}")
    print(f"  Tiempo: {analisis['tiempo_implementacion']}")
    print(f"  → {analisis['recomendacion']}")
```

## Matriz de priorización

```python
def priorizar_casos_uso(casos: list[dict]) -> list[dict]:
    """Ordena casos de uso por puntuación impacto/esfuerzo."""

    mapa_puntos = {"alto": 3, "alta": 3, "medio": 2, "media": 2, "bajo": 1, "baja": 1}

    for caso in casos:
        impacto = mapa_puntos.get(caso.get("impacto_negocio", "bajo"), 1)
        viabilidad = mapa_puntos.get(caso.get("viabilidad", "baja"), 1)
        complejidad_inv = 4 - mapa_puntos.get(caso.get("complejidad_tecnica", "alta"), 3)
        caso["puntuacion"] = impacto * viabilidad * complejidad_inv

    return sorted(casos, key=lambda x: x["puntuacion"], reverse=True)
```

## Hoja de ruta de implementación

Una implementación responsable sigue estas fases:

```
FASE 1 — PILOTO (semanas 1-6)
  ├── Seleccionar 1 caso de uso de bajo riesgo
  ├── Definir KPIs y baseline actual
  ├── Implementar MVP con Claude API
  └── Medir resultados con datos reales

FASE 2 — VALIDACIÓN (semanas 7-12)
  ├── Ampliar a más usuarios/volumen
  ├── Iterar basado en feedback
  ├── Documentar aprendizajes
  └── Calcular ROI real vs estimado

FASE 3 — ESCALA (mes 4+)
  ├── Desplegar en producción completa
  ├── Añadir nuevos casos de uso
  ├── Construir capacidades internas
  └── Medir impacto en negocio
```

## Calcular ROI de IA

```python
def calcular_roi_ia(
    horas_ahorradas_semana: float,
    coste_hora_empleado: float,
    coste_api_mensual: float,
    coste_implementacion: float,
    semanas_por_mes: float = 4.33
) -> dict:
    """Calcula el ROI de una implementación de IA."""

    ahorro_mensual = horas_ahorradas_semana * semanas_por_mes * coste_hora_empleado
    beneficio_neto_mensual = ahorro_mensual - coste_api_mensual
    meses_recuperacion = coste_implementacion / beneficio_neto_mensual if beneficio_neto_mensual > 0 else float("inf")
    roi_anual = ((beneficio_neto_mensual * 12) - coste_implementacion) / coste_implementacion * 100

    return {
        "ahorro_mensual": round(ahorro_mensual, 2),
        "coste_api_mensual": coste_api_mensual,
        "beneficio_neto_mensual": round(beneficio_neto_mensual, 2),
        "meses_para_recuperar": round(meses_recuperacion, 1),
        "roi_anual_pct": round(roi_anual, 1)
    }

# Ejemplo: automatización de soporte
roi = calcular_roi_ia(
    horas_ahorradas_semana=20,      # 20h/semana de trabajo de soporte
    coste_hora_empleado=35,          # 35€/h coste real (salario + ss)
    coste_api_mensual=400,           # uso de API Claude
    coste_implementacion=8000        # desarrollo + integración
)

for k, v in roi.items():
    print(f"  {k}: {v}")
```

## Gestión del cambio organizacional

La mayor barrera para la IA no es técnica, es humana.

```python
def generar_plan_comunicacion(departamento: str, caso_uso: str) -> str:
    """Genera un plan de comunicación para la adopción de IA."""

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""Crea un plan de comunicación para introducir IA en {departamento}.
El caso de uso es: {caso_uso}

El plan debe incluir:
1. Mensaje clave para el equipo (qué cambia, qué no cambia)
2. Cómo presentarlo para reducir miedo al reemplazo
3. Plan de formación básico
4. Cómo medir la adopción

Sé directo y práctico. Máximo 400 palabras."""
        }]
    )
    return resp.content[0].text

plan = generar_plan_comunicacion("Customer Success", "asistente IA para responder tickets")
print(plan)
```

## Checklist de estrategia de IA

```
Antes de empezar:
  ☐ Caso de uso identificado con impacto medible
  ☐ Datos disponibles y de calidad suficiente
  ☐ Patrocinador ejecutivo comprometido
  ☐ KPIs y baseline definidos
  ☐ Presupuesto aprobado (piloto + escala)

Durante el piloto:
  ☐ MVP funcional en < 6 semanas
  ☐ Equipo de negocio involucrado desde el día 1
  ☐ Proceso de feedback estructurado
  ☐ Monitorización de calidad activa

Para escalar:
  ☐ ROI del piloto documentado
  ☐ Plan de formación para usuarios
  ☐ Proceso de mejora continua definido
  ☐ Gobernanza y revisión humana donde aplique
```

## Recursos

- [Notebook interactivo](../notebooks/ia-empresarial/01-estrategia-ia.ipynb)
- [McKinsey Global Institute — The state of AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)
- [Anthropic — Prompting guide para empresas](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
