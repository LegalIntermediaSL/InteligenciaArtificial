# Traducción Automática y Localización con IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/casos-de-uso/05-traduccion-y-localizacion.ipynb)

Los LLMs no solo traducen palabras: adaptan tono, formato y referencias culturales.
Este artículo muestra cuándo usarlos y cómo hacerlo de forma estructurada y evaluable.

---

## 1. LLM vs herramienta de traducción dedicada

```python
# Criterios para elegir entre Claude y DeepL/Google Translate

CRITERIOS = {
    "Usa DeepL o Google Translate si...": [
        "El texto es factual y no requiere contexto cultural",
        "Necesitas >100k caracteres/min (APIs de traducción son más rápidas)",
        "El presupuesto es ajustado (DeepL es ~10x más barato por palabra)",
        "El texto es HTML puro sin razonamiento semántico necesario",
    ],
    "Usa un LLM (Claude) si...": [
        "El texto tiene argot, humor o referencias culturales",
        "Necesitas adaptar el tono al mercado destino (formal/informal)",
        "El formato es complejo: MDX, Markdown con código, JSON estructurado",
        "Quieres preservar metáforas o juegos de palabras",
        "Necesitas localización completa (unidades, moneda, fechas)",
    ],
}

for categoria, puntos in CRITERIOS.items():
    print(f"\n{categoria}")
    for p in puntos:
        print(f"  • {p}")

# Regla práctica: si la oración necesita ser entendida para ser traducida, usa un LLM
```

---

## 2. Traducción con preservación de formato

```python
import anthropic
import json
import re

client = anthropic.Anthropic()

SYSTEM_TRADUCCION = """Eres un traductor profesional especializado en contenido técnico.
Reglas estrictas:
- Traduce SOLO el texto visible; no toques etiquetas HTML, atributos, URLs ni claves JSON
- Mantén el formato exacto (saltos de línea, indentación, puntuación Markdown)
- No añadas explicaciones ni comentarios fuera del texto traducido
- Responde ÚNICAMENTE con el contenido traducido, sin preámbulo"""


class TranslatorFormateado:
    """Traduce texto preservando HTML, Markdown y JSON."""

    def __init__(self):
        self._client = anthropic.Anthropic()

    def _traducir(self, texto: str, idioma_destino: str, contexto: str = "") -> str:
        instruccion = f"Traduce al {idioma_destino}"
        if contexto:
            instruccion += f" (contexto: {contexto})"
        instruccion += f":\n\n{texto}"

        respuesta = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM_TRADUCCION,
            messages=[{"role": "user", "content": instruccion}],
        )
        return respuesta.content[0].text.strip()

    def html(self, html: str, idioma: str) -> str:
        # Proteger atributos antes de enviar
        return self._traducir(html, idioma, contexto="HTML — no tocar etiquetas ni atributos")

    def markdown(self, md: str, idioma: str) -> str:
        return self._traducir(md, idioma, contexto="Markdown — mantener sintaxis exacta")

    def json_valores(self, data: dict, idioma: str, claves_traducir: list[str]) -> dict:
        """Traduce solo los valores de las claves indicadas."""
        resultado = dict(data)
        for clave in claves_traducir:
            if clave in data and isinstance(data[clave], str):
                resultado[clave] = self._traducir(data[clave], idioma)
        return resultado


# Demo
translator = TranslatorFormateado()

md_ejemplo = """## Getting Started

Install the package with `pip install mylib`.

> **Note**: Requires Python 3.10+

```python
from mylib import Client
client = Client(api_key="...")
```
"""

html_ejemplo = '<p class="intro">Welcome to our <strong>platform</strong>.</p>'

json_ejemplo = {
    "id": "prod-001",
    "name": "AI Assistant",
    "description": "A powerful tool for automating workflows.",
    "price": 49.99,
}

md_traducido   = translator.markdown(md_ejemplo, "español")
html_traducido = translator.html(html_ejemplo, "español")
json_traducido = translator.json_valores(json_ejemplo, "español", ["name", "description"])

print("=== Markdown traducido ===")
print(md_traducido)
print("\n=== HTML traducido ===")
print(html_traducido)
print("\n=== JSON traducido ===")
print(json.dumps(json_traducido, ensure_ascii=False, indent=2))
```

---

## 3. Detección automática de idioma

```python
import anthropic
import json

client = anthropic.Anthropic()

def detectar_idioma(texto: str) -> dict:
    """
    Detecta el idioma del texto sin llamar a APIs externas.
    Devuelve: {idioma, codigo_iso, confianza, razon}
    """
    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system="Responde únicamente con JSON válido, sin texto adicional.",
        messages=[{
            "role": "user",
            "content": (
                f"Detecta el idioma de este texto y responde con JSON:\n"
                f'{{"idioma": "...", "codigo_iso": "...", "confianza": 0.0-1.0, "razon": "..."}}\n\n'
                f'Texto: """{texto}"""'
            ),
        }],
    )

    try:
        return json.loads(respuesta.content[0].text)
    except json.JSONDecodeError:
        return {"idioma": "desconocido", "codigo_iso": "??", "confianza": 0.0, "razon": "parse error"}


# Prueba con varios idiomas
muestras = [
    "The transformer architecture revolutionized natural language processing.",
    "Le modèle de langage apprend à prédire le prochain token.",
    "Das Sprachmodell wurde auf Billionen von Token trainiert.",
    "大規模言語モデルは自然言語処理を変革しました。",
    "El fine-tuning adapta el modelo preentrenado a una tarea específica.",
]

for muestra in muestras:
    resultado = detectar_idioma(muestra)
    print(f"  [{resultado['codigo_iso']}] {resultado['idioma']} "
          f"(confianza: {resultado['confianza']:.0%}) — {muestra[:50]}...")
```

---

## 4. Localización vs traducción

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_LOCALIZACION = """Eres un especialista en localización de contenido.
La localización va más allá de la traducción:
- Adapta el tono al registro cultural del país destino (tuteo/ustedeo, formalidad)
- Convierte unidades (millas→km, °F→°C, lb→kg, USD→EUR con símbolo local)
- Adapta formatos de fecha y número al estándar local
- Sustituye referencias culturales por equivalentes locales cuando sea natural
Responde solo con el texto localizado."""


def localizar(texto: str, pais_destino: str, idioma: str) -> str:
    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_LOCALIZACION,
        messages=[{
            "role": "user",
            "content": (
                f"Localiza para {pais_destino} en {idioma}:\n\n{texto}"
            ),
        }],
    )
    return respuesta.content[0].text.strip()


# Texto de ejemplo con elementos culturales y unidades
texto_original_en = """
Our app helps you track your 5-mile morning runs.
Set reminders for the Super Bowl game or your Thanksgiving dinner.
Water temperature today: 68°F. Wind speed: 15 mph.
Price: $29.99/month. Sign up and get started today!
"""

paises = [
    ("España",    "español"),
    ("México",    "español"),
    ("Alemania",  "alemán"),
]

for pais, idioma in paises:
    localizado = localizar(texto_original_en, pais, idioma)
    print(f"\n=== {pais} ({idioma}) ===")
    print(localizado)
```

---

## 5. Pipeline multi-idioma con asyncio

```python
import asyncio
import time
import anthropic
from dataclasses import dataclass

@dataclass
class TraduccionResult:
    idioma:  str
    codigo:  str
    texto:   str
    tokens:  int
    ms:      float

IDIOMAS = [
    ("inglés",   "en"),
    ("francés",  "fr"),
    ("alemán",   "de"),
    ("italiano", "it"),
    ("portugués", "pt"),
]

SYSTEM_MULTI = ("Traduce el siguiente texto al idioma indicado. "
                "Responde únicamente con la traducción, sin comentarios.")


async def traducir_a(
    client: anthropic.AsyncAnthropic,
    semaforo: asyncio.Semaphore,
    texto: str,
    idioma: str,
    codigo: str,
) -> TraduccionResult:
    async with semaforo:
        inicio = time.monotonic()
        respuesta = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=SYSTEM_MULTI,
            messages=[{
                "role": "user",
                "content": f"Idioma destino: {idioma}\n\nTexto:\n{texto}",
            }],
        )
        ms = (time.monotonic() - inicio) * 1000
        return TraduccionResult(
            idioma  = idioma,
            codigo  = codigo,
            texto   = respuesta.content[0].text.strip(),
            tokens  = respuesta.usage.output_tokens,
            ms      = ms,
        )


async def pipeline_multiidioma(articulo: str) -> list[TraduccionResult]:
    """Traduce un artículo a 5 idiomas en paralelo."""
    semaforo = asyncio.Semaphore(5)   # todos a la vez (son solo 5)

    async with anthropic.AsyncAnthropic() as client:
        tareas = [
            traducir_a(client, semaforo, articulo, idioma, codigo)
            for idioma, codigo in IDIOMAS
        ]
        return await asyncio.gather(*tareas)


# Demo
articulo_es = """
La inteligencia artificial generativa está transformando la industria del software.
Los desarrolladores usan LLMs para automatizar tareas repetitivas, generar código
y mejorar la experiencia de sus usuarios. En 2024, el mercado de herramientas de IA
para desarrolladores creció un 340% respecto al año anterior.
"""

async def main() -> None:
    inicio_total = time.monotonic()
    resultados = await pipeline_multiidioma(articulo_es)
    total_ms = (time.monotonic() - inicio_total) * 1000

    print(f"Artículo traducido a {len(resultados)} idiomas en {total_ms:.0f}ms\n")
    for r in resultados:
        print(f"  [{r.codigo}] {r.idioma:<12} {r.tokens:>4} tokens  {r.ms:>6.0f}ms")
        print(f"       {r.texto[:80]}...")
        print()

asyncio.run(main())
```

---

## 6. Evaluación de calidad: BLEU y LLM-as-judge

```python
import anthropic
from collections import Counter
import math

# --- BLEU score simplificado (bigrama) ---

def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_1_2(hipotesis: str, referencia: str) -> dict[str, float]:
    """BLEU-1 y BLEU-2 sin smoothing (referencia única)."""
    h = hipotesis.lower().split()
    r = referencia.lower().split()

    scores = {}
    for n in (1, 2):
        h_ng = ngrams(h, n)
        r_ng = ngrams(r, n)
        # Recorte: no contar más ocurrencias de las que aparecen en referencia
        recortado = sum(min(cnt, r_ng[ng]) for ng, cnt in h_ng.items())
        total_h   = sum(h_ng.values())
        precision = recortado / total_h if total_h > 0 else 0.0
        scores[f"bleu_{n}"] = precision

    # Penalización por brevedad
    bp = 1.0 if len(h) >= len(r) else math.exp(1 - len(r) / max(len(h), 1))
    scores["bp"] = bp
    scores["bleu_combinado"] = bp * math.sqrt(scores["bleu_1"] * scores["bleu_2"] + 1e-12)
    return scores


# --- LLM-as-judge ---

client = anthropic.Anthropic()

def evaluar_traduccion_llm(
    original:    str,
    traduccion:  str,
    idioma:      str,
    referencia:  str | None = None,
) -> dict:
    contexto_ref = ""
    if referencia:
        contexto_ref = f"\nTraducción de referencia:\n{referencia}\n"

    prompt = (
        f"Evalúa esta traducción del español al {idioma}.\n"
        f"Original:\n{original}\n"
        f"Traducción a evaluar:\n{traduccion}\n"
        f"{contexto_ref}\n"
        "Responde con JSON: "
        '{"fidelidad": 1-5, "fluidez": 1-5, "errores": [...], "sugerencia": "..."}'
    )

    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="Responde únicamente con JSON válido.",
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    try:
        return json.loads(respuesta.content[0].text)
    except json.JSONDecodeError:
        return {"fidelidad": 0, "fluidez": 0, "errores": [], "sugerencia": "parse error"}


# Demo de evaluación
original   = "Los modelos de lenguaje grande aprenden de enormes corpus de texto."
traduccion = "The large language models learn from enormous corpus of text."
referencia = "Large language models learn from massive text corpora."

metricas_bleu = bleu_1_2(traduccion, referencia)
evaluacion_llm = evaluar_traduccion_llm(original, traduccion, "inglés", referencia)

print("=== Métricas automáticas (BLEU) ===")
for k, v in metricas_bleu.items():
    print(f"  {k}: {v:.4f}")

print("\n=== LLM-as-judge ===")
print(f"  Fidelidad: {evaluacion_llm.get('fidelidad')}/5")
print(f"  Fluidez:   {evaluacion_llm.get('fluidez')}/5")
print(f"  Errores:   {evaluacion_llm.get('errores')}")
print(f"  Sugerencia: {evaluacion_llm.get('sugerencia')}")
```

---

→ Anterior: [Extracción de PDFs](04-extraccion-pdfs.md) | → Siguiente: [Clasificación avanzada](../casos-de-uso-avanzados/01-clasificacion-jerarquica.md)
