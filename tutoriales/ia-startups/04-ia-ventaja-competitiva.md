# IA como Ventaja Competitiva: Data Flywheel y Narrativa para Inversores

## Por qué la mayoría de startups de IA no tienen moat

Usar la API de Claude no es una ventaja competitiva. Tu competencia puede hacer lo mismo mañana.
El moat real de una startup de IA viene de tres fuentes:

```
FUENTES DE MOAT EN IA
──────────────────────
1. DATOS PROPIETARIOS
   → Datos que tu competencia no puede replicar
   → Feedback de usuarios que mejora el producto con el tiempo
   → Data flywheel: más usuarios → mejores datos → mejor producto

2. INTEGRACIÓN PROFUNDA
   → Embebido en el workflow del usuario (coste de cambio alto)
   → Conexiones con sistemas del cliente (ERP, CRM, etc.)
   → Historial y contexto acumulado del usuario

3. VELOCIDAD DE ITERACIÓN
   → Equipo pequeño que itera más rápido que grandes
   → Feedback loop corto: usuario → insight → mejora en días
   → Distribución enfocada en un nicho muy específico
```

## Data Flywheel: el activo más valioso de una startup de IA

```python
import anthropic
import json
from datetime import datetime
from pathlib import Path

client = anthropic.Anthropic()

class DataFlywheel:
    """
    Sistema para capturar feedback de usuarios y mejorar el producto.
    Cada interacción genera datos que hacen el producto mejor.
    """

    def __init__(self, directorio: str = "flywheel_data"):
        self.dir = Path(directorio)
        self.dir.mkdir(exist_ok=True)
        self.interacciones = []

    def registrar_interaccion(
        self,
        user_id: str,
        input_usuario: str,
        output_modelo: str,
        feedback: str | None = None,  # "positivo", "negativo", None
        score: int | None = None       # 1-5 si el usuario puntúa
    ):
        """Registra cada interacción con su feedback."""
        interaccion = {
            "id": f"{user_id}_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "input": input_usuario,
            "output": output_modelo,
            "feedback": feedback,
            "score": score
        }
        self.interacciones.append(interaccion)

        # Persistir en disco
        ruta = self.dir / f"{datetime.now().strftime('%Y%m')}_interacciones.jsonl"
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(interaccion, ensure_ascii=False) + "\n")

        return interaccion["id"]

    def analizar_patrones_fallo(self, n_recientes: int = 100) -> dict:
        """Identifica patrones en las interacciones negativas para mejorar el producto."""
        negativas = [i for i in self.interacciones[-n_recientes:]
                     if i.get("feedback") == "negativo" or (i.get("score") or 5) <= 2]

        if not negativas:
            return {"mensaje": "Sin suficientes interacciones negativas para analizar"}

        texto_negativas = "\n".join(
            f"Input: {i['input'][:100]} | Output: {i['output'][:100]}"
            for i in negativas[:10]
        )

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": f"""Analiza estas {len(negativas)} interacciones que recibieron feedback negativo
de usuarios de un producto de IA. Identifica patrones y sugiere mejoras al system prompt.

INTERACCIONES NEGATIVAS:
{texto_negativas}

Devuelve JSON:
{{
  "patrones_fallo": ["patrón 1", "patrón 2"],
  "hipotesis_causa": "por qué fallan",
  "mejoras_sugeridas_al_prompt": ["mejora 1", "mejora 2"],
  "casos_prueba_recomendados": ["caso para validar la mejora"]
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

    def metricas_calidad(self) -> dict:
        """Métricas de calidad del producto basadas en feedback real."""
        total = len(self.interacciones)
        if total == 0:
            return {"error": "Sin datos"}

        con_feedback = [i for i in self.interacciones if i.get("feedback")]
        positivos = sum(1 for i in con_feedback if i.get("feedback") == "positivo")
        con_score = [i for i in self.interacciones if i.get("score")]
        score_medio = sum(i["score"] for i in con_score) / len(con_score) if con_score else None

        return {
            "total_interacciones": total,
            "tasa_feedback_pct": round(len(con_feedback) / total * 100, 1),
            "tasa_satisfaccion_pct": round(positivos / max(len(con_feedback), 1) * 100, 1),
            "score_medio": round(score_medio, 2) if score_medio else "N/A",
            "interacciones_sin_feedback": total - len(con_feedback)
        }


# Demo del flywheel
flywheel = DataFlywheel()

# Simular interacciones con feedback variado
demos = [
    ("user_001", "Analiza esta cláusula de limitación de responsabilidad", "ALTO RIESGO: La cláusula excluye toda responsabilidad del proveedor...", "positivo", 5),
    ("user_002", "¿Qué significa NDA?", "Un NDA (Non-Disclosure Agreement) es un acuerdo de confidencialidad...", "positivo", 4),
    ("user_003", "Revisa mi contrato de trabajo", "No puedo procesar contratos de trabajo completos sin más contexto.", "negativo", 2),
    ("user_004", "¿Cuándo expira mi contrato?", "No tengo acceso a la fecha de tu contrato.", "negativo", 1),
    ("user_005", "Explica la cláusula de propiedad intelectual", "La cláusula de PI establece quién es dueño del trabajo creado...", "positivo", 5),
]

for uid, inp, out, fb, sc in demos:
    flywheel.registrar_interaccion(uid, inp, out, fb, sc)

print("Métricas del flywheel:")
for k, v in flywheel.metricas_calidad().items():
    print(f"  {k}: {v}")

print("\nAnálisis de fallos:")
patrones = flywheel.analizar_patrones_fallo()
for k, v in patrones.items():
    print(f"  {k}: {v}")
```

## Narrativa para inversores: cómo presentar tu IA

```python
def generar_narrativa_inversores(
    descripcion_producto: str,
    metricas_actuales: dict,
    diferenciadores: list[str]
) -> str:
    """Genera el pitch de IA para inversores basado en datos reales."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system="""Eres un asesor experto en fundraising para startups de IA.
Conoces qué buscan los inversores: moat defensible, data flywheel, unit economics y timing.""",
        messages=[{
            "role": "user",
            "content": f"""Redacta la sección de "IA y Ventaja Competitiva" para un pitch deck de inversores.

DESCRIPCIÓN DEL PRODUCTO:
{descripcion_producto}

MÉTRICAS ACTUALES:
{json.dumps(metricas_actuales, ensure_ascii=False, indent=2)}

DIFERENCIADORES CLAVE:
{chr(10).join(f'- {d}' for d in diferenciadores)}

La narrativa debe responder a las preguntas que haría un inversor de Series A:
1. ¿Por qué NO puede una empresa grande copiar esto en 6 meses?
2. ¿Cómo mejora el producto con más usuarios? (data flywheel)
3. ¿Cuál es el coste de cambio para el cliente?
4. ¿Qué pasa con los márgenes cuando escala?

Formato: 4 párrafos concisos, lenguaje de negocios, datos concretos.
Máximo 400 palabras."""
        }]
    )
    return resp.content[0].text


narrativa = generar_narrativa_inversores(
    descripcion_producto="SaaS de análisis automático de contratos para PYMEs españolas con IA. El usuario sube el contrato, recibe en 30 segundos un análisis de riesgos con recomendaciones concretas.",
    metricas_actuales={
        "usuarios_activos": 280,
        "contratos_analizados": 4200,
        "nps": 62,
        "churn_mensual_pct": 4.2,
        "mrr_eur": 8400,
        "cac_eur": 95,
        "ltv_eur": 420
    },
    diferenciadores=[
        "Base de datos propietaria de 4.200 contratos españoles con feedback de abogados",
        "Integración nativa con despachos legales (API bidireccional con los 3 principales ERP legales)",
        "Modelo fine-tuneado con jurisprudencia española del CENDOJ",
        "El producto mejora con cada contrato analizado (feedback loop cerrado)"
    ]
)

print("NARRATIVA PARA INVERSORES")
print("=" * 60)
print(narrativa)
```

## Análisis de competidores con IA

```python
def analizar_posicion_competitiva(
    descripcion_tuya: str,
    competidores: list[dict]
) -> dict:
    """Analiza tu posición competitiva y genera recomendaciones estratégicas."""

    competidores_texto = "\n".join(
        f"- {c['nombre']}: {c['descripcion']} | Precio: {c.get('precio', 'N/A')} | Debilidad: {c.get('debilidad', 'N/A')}"
        for c in competidores
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Analiza la posición competitiva de esta startup de IA:

MI PRODUCTO:
{descripcion_tuya}

COMPETIDORES:
{competidores_texto}

Devuelve JSON:
{{
  "ventajas_competitivas_reales": ["ventaja con evidencia"],
  "vulnerabilidades": ["donde eres más débil"],
  "oportunidades_de_diferenciacion": ["hueco de mercado no cubierto"],
  "riesgo_competitivo_principal": "descripción del mayor riesgo",
  "estrategia_recomendada": "en 2-3 frases, qué hacer ahora mismo",
  "metricas_a_defender": ["KPI que debes crecer para construir moat"]
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


analisis = analizar_posicion_competitiva(
    "Análisis automático de contratos para PYMEs con IA, especializado en derecho español",
    [
        {"nombre": "LegalZoom (adaptación EU)", "descripcion": "Plantillas de contratos online", "precio": "29€/mes", "debilidad": "No analiza contratos de terceros"},
        {"nombre": "Ironclad", "descripcion": "CLM enterprise para grandes empresas", "precio": ">500€/mes", "debilidad": "Demasiado caro y complejo para PYMEs"},
        {"nombre": "ChatGPT Plus", "descripcion": "IA generalista, no especializada en derecho español", "precio": "20€/mes", "debilidad": "Sin conocimiento de jurisprudencia española, sin integración"}
    ]
)

print("ANÁLISIS COMPETITIVO")
print("=" * 60)
for k, v in analisis.items():
    if isinstance(v, list):
        print(f"\n{k}:")
        for item in v:
            print(f"  • {item}")
    else:
        print(f"\n{k}: {v}")
```

## Checklist: ¿tienes un moat real?

```python
def evaluar_moat(respuestas: dict) -> dict:
    """Evalúa si tu startup tiene un moat defensible de IA."""

    preguntas_moat = {
        "datos_propietarios": "¿Tienes datos que tu competencia no puede replicar fácilmente?",
        "mejora_con_uso": "¿El producto mejora automáticamente con más usuarios?",
        "coste_de_cambio": "¿Cuánto tardaría un cliente en migrar a un competidor?",
        "integracion_profunda": "¿Estás integrado en el workflow diario del cliente?",
        "efecto_red": "¿El producto es mejor cuantos más usuarios tiene?",
        "marca_especializada": "¿Eres conocido como el líder en tu nicho específico?"
    }

    puntuacion = sum(1 for k, v in respuestas.items() if v)
    nivel_moat = "fuerte" if puntuacion >= 4 else "medio" if puntuacion >= 2 else "débil"

    debilidades = [preguntas_moat[k] for k, v in respuestas.items() if not v]

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Una startup de IA tiene moat {nivel_moat} ({puntuacion}/6 puntos).
Áreas sin moat: {debilidades}

¿Qué debe hacer en los próximos 90 días para fortalecer su posición?
Máximo 3 acciones concretas y priorizadas."""
        }]
    )

    return {
        "puntuacion": f"{puntuacion}/6",
        "nivel_moat": nivel_moat,
        "acciones_90_dias": resp.content[0].text
    }


evaluacion = evaluar_moat({
    "datos_propietarios": True,
    "mejora_con_uso": True,
    "coste_de_cambio": False,
    "integracion_profunda": False,
    "efecto_red": False,
    "marca_especializada": True
})

print("EVALUACIÓN DE MOAT")
print("=" * 40)
print(f"Puntuación: {evaluacion['puntuacion']} ({evaluacion['nivel_moat']})")
print(f"\nAcciones próximos 90 días:\n{evaluacion['acciones_90_dias']}")
```

## Recursos

- [Notebook interactivo](../notebooks/ia-startups/04-ia-ventaja-competitiva.ipynb)
- [a16z — The AI Startup Playbook](https://a16z.com/ai/)
- [Sequoia — Generative AI's Act Two](https://www.sequoiacap.com/article/generative-ai-act-two/)
- [Hamilton Helmer — 7 Powers](https://7powers.com)
