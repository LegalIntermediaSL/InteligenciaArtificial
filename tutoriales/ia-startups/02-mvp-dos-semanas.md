# MVP con IA en 2 Semanas

## Por qué 2 semanas y no 2 meses

Un MVP de IA tiene un solo objetivo: **demostrar que el usuario paga por el valor generado**.
Todo lo que no sirve para eso es deuda técnica anticipada.

```
SEMANA 1 — Validación técnica
  Día 1-2: Diseño de system prompt + pruebas manuales
  Día 3-4: Backend mínimo (FastAPI + Claude API)
  Día 5-7: Frontend básico con streaming

SEMANA 2 — Validación de mercado
  Día 8-9: Autenticación + primeros 10 usuarios beta
  Día 10-11: Feedback loop + iteraciones de prompt
  Día 12-14: Métricas, pricing y primera venta
```

## Día 1: Diseñar el system prompt antes de escribir código

El system prompt ES tu producto en fase MVP. Dedícale el primer día.

```python
import anthropic

client = anthropic.Anthropic()

def probar_prompt(system: str, casos_de_prueba: list[dict]) -> list[dict]:
    """Evalúa un system prompt con casos de prueba antes de integrarlo."""
    resultados = []

    for caso in casos_de_prueba:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": caso["input"]}]
        )
        output = resp.content[0].text

        resultado = {
            "input": caso["input"],
            "output": output,
            "esperado": caso.get("esperado", ""),
            "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            "coste_usd": (resp.usage.input_tokens + resp.usage.output_tokens) / 1_000_000 * 1.20
        }
        resultados.append(resultado)

    return resultados


# Ejemplo: startup de revisión de contratos
SYSTEM_V1 = """Eres un asistente especializado en análisis de contratos para PYMEs españolas.
Tu función es identificar cláusulas problemáticas, explicarlas en lenguaje claro
y sugerir alternativas más favorables para el cliente.

REGLAS:
- Explica cada cláusula problemática en una frase
- Indica el riesgo en escala: BAJO / MEDIO / ALTO
- Sugiere siempre una alternativa concreta
- Si no hay problemas, dilo explícitamente
- Nunca des asesoramiento jurídico vinculante"""

casos_prueba = [
    {
        "input": "Cláusula 5: En caso de incumplimiento, el proveedor no será responsable de ningún daño directo o indirecto, incluyendo pérdida de beneficios.",
        "esperado": "Identificar limitación de responsabilidad total como riesgo ALTO"
    },
    {
        "input": "Cláusula 3: El precio se revisará anualmente según el IPC publicado por el INE.",
        "esperado": "Cláusula razonable, riesgo BAJO"
    },
    {
        "input": "Cláusula 8: Cualquier disputa se resolverá en los tribunales de la ciudad de Nueva York bajo la ley del estado de Delaware.",
        "esperado": "Identificar jurisdicción extranjera como problema para empresa española"
    }
]

resultados = probar_prompt(SYSTEM_V1, casos_prueba)

print("EVALUACIÓN DEL SYSTEM PROMPT v1.0")
print("=" * 60)
for r in resultados:
    print(f"\nInput: {r['input'][:80]}...")
    print(f"Output: {r['output'][:200]}...")
    print(f"Tokens: {r['tokens']} | Coste: ${r['coste_usd']:.4f}")
```

## Iterar el prompt con versiones

```python
import json
from pathlib import Path
from datetime import datetime

class GestorDePrompts:
    """Gestiona versiones de system prompts como si fueran código."""

    def __init__(self, directorio: str = "prompts"):
        self.dir = Path(directorio)
        self.dir.mkdir(exist_ok=True)

    def guardar(self, nombre: str, contenido: str, version: str, notas: str = "") -> str:
        """Guarda una versión del prompt con metadata."""
        metadata = {
            "nombre": nombre,
            "version": version,
            "fecha": datetime.now().isoformat(),
            "notas": notas,
            "chars": len(contenido)
        }
        ruta_prompt = self.dir / f"{nombre}_v{version}.txt"
        ruta_meta = self.dir / f"{nombre}_v{version}.json"

        ruta_prompt.write_text(contenido, encoding="utf-8")
        ruta_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(ruta_prompt)

    def cargar(self, nombre: str, version: str) -> str:
        ruta = self.dir / f"{nombre}_v{version}.txt"
        return ruta.read_text(encoding="utf-8") if ruta.exists() else ""

    def listar_versiones(self, nombre: str) -> list:
        return sorted([
            json.loads(p.read_text())
            for p in self.dir.glob(f"{nombre}_v*.json")
        ], key=lambda x: x["version"])

    def comparar(self, nombre: str, v1: str, v2: str, caso_prueba: str) -> dict:
        """Compara el output de dos versiones del mismo prompt."""
        prompt_v1 = self.cargar(nombre, v1)
        prompt_v2 = self.cargar(nombre, v2)

        def llamar(system: str) -> tuple[str, int]:
            r = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": caso_prueba}]
            )
            return r.content[0].text, r.usage.input_tokens + r.usage.output_tokens

        out_v1, tok_v1 = llamar(prompt_v1)
        out_v2, tok_v2 = llamar(prompt_v2)

        return {
            "caso": caso_prueba[:100],
            f"output_v{v1}": out_v1[:300],
            f"tokens_v{v1}": tok_v1,
            f"output_v{v2}": out_v2[:300],
            f"tokens_v{v2}": tok_v2,
            "diferencia_tokens": tok_v2 - tok_v1
        }


gestor = GestorDePrompts("prompts_mvp")
gestor.guardar("analizador_contratos", SYSTEM_V1, "1.0", "Versión inicial con casos de prueba básicos")
print("Prompt guardado:", gestor.listar_versiones("analizador_contratos"))
```

## Pipeline completo: del PDF al análisis en 3 pasos

```python
def pipeline_analisis_contrato(texto_contrato: str, user_id: str) -> dict:
    """
    Pipeline completo de análisis de contrato para el MVP.
    Paso 1: Extraer cláusulas clave
    Paso 2: Analizar riesgos
    Paso 3: Generar resumen ejecutivo
    """

    # PASO 1: Extraer cláusulas estructuradas
    resp_extraccion = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Extrae las cláusulas principales de este contrato en JSON.
Formato: {{"clausulas": [{{"numero": 1, "titulo": "...", "texto": "..."}}]}}

CONTRATO:
{texto_contrato[:4000]}"""
        }]
    )

    texto = resp_extraccion.content[0].text
    if "```" in texto:
        texto = texto.split("```")[1].lstrip("json\n")
    try:
        clausulas = json.loads(texto).get("clausulas", [])
    except json.JSONDecodeError:
        clausulas = []

    # PASO 2: Analizar cada cláusula (usando el system prompt v1)
    riesgos = []
    for c in clausulas[:5]:  # limitar a 5 para controlar coste
        resp_riesgo = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_V1,
            messages=[{"role": "user", "content": f"Analiza esta cláusula: {c.get('texto', '')[:500]}"}]
        )
        riesgos.append({
            "clausula": c.get("titulo", f"Cláusula {c.get('numero', '?')}"),
            "analisis": resp_riesgo.content[0].text
        })

    # PASO 3: Resumen ejecutivo
    texto_riesgos = "\n".join(f"- {r['clausula']}: {r['analisis'][:100]}" for r in riesgos)
    resp_resumen = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Genera un resumen ejecutivo en 3 puntos para un empresario no abogado:
{texto_riesgos}

Formato: punto clave, recomendación y próximo paso."""
        }]
    )

    return {
        "user_id": user_id,
        "clausulas_analizadas": len(clausulas),
        "riesgos": riesgos,
        "resumen_ejecutivo": resp_resumen.content[0].text
    }
```

## Pricing: cuánto cobrar desde el día 1

```python
def calcular_precio_usuario(
    coste_tokens_por_usuario_mes: float,  # en USD
    margen_objetivo: float = 0.70,         # 70% de margen bruto
    factor_soporte: float = 1.30           # 30% overhead de soporte/infra
) -> dict:
    """Calcula el precio mínimo para un usuario según costes reales."""

    coste_total = coste_tokens_por_usuario_mes * factor_soporte
    precio_minimo = coste_total / (1 - margen_objetivo)

    # Redondear al tier de precio más cercano
    tiers = [9, 19, 29, 49, 79, 99, 149, 199]
    precio_mercado = next((t for t in tiers if t >= precio_minimo), tiers[-1])

    return {
        "coste_tokens_mes_usd": round(coste_tokens_por_usuario_mes, 2),
        "coste_total_con_overhead_usd": round(coste_total, 2),
        "precio_minimo_usd": round(precio_minimo, 2),
        "precio_recomendado_usd": precio_mercado,
        "margen_real_pct": round((1 - coste_total / precio_mercado) * 100, 1)
    }


# Ejemplo: usuario hace 50 análisis/mes, ~2000 tokens cada uno
coste_50_analisis = (50 * 2000) / 1_000_000 * 1.20  # Haiku pricing
precio = calcular_precio_usuario(coste_50_analisis)
print("Estructura de precios:")
for k, v in precio.items():
    print(f"  {k}: {v}")
```

## Recursos

- [Notebook interactivo](../notebooks/ia-startups/02-mvp-dos-semanas.ipynb)
- [Y Combinator — Startup advice](https://www.ycombinator.com/library)
- [Lenny's Newsletter — How to build an AI MVP](https://www.lennysnewsletter.com)
