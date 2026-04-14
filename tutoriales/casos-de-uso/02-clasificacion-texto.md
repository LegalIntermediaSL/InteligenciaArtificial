# 02 — Clasificación de texto

> **Bloque:** Casos de uso · **Nivel:** Práctico · **Tiempo estimado:** 35 min

---

## Índice

1. [Objetivo](#1-objetivo)
2. [Enfoque 1: Clasificación con LLM (zero-shot)](#2-enfoque-1-clasificación-con-llm-zero-shot)
3. [Enfoque 2: Clasificación con LLM (few-shot)](#3-enfoque-2-clasificación-con-llm-few-shot)
4. [Enfoque 3: Clasificación con ML clásico](#4-enfoque-3-clasificación-con-ml-clásico)
5. [Comparativa de enfoques](#5-comparativa-de-enfoques)
6. [Clasificación de múltiples documentos](#6-clasificación-de-múltiples-documentos)
7. [Cuándo usar cada enfoque](#7-cuándo-usar-cada-enfoque)

---

## 1. Objetivo

Clasificar textos en categorías predefinidas. Veremos tres enfoques distintos:

1. **LLM zero-shot** — sin ejemplos, el modelo usa su conocimiento general
2. **LLM few-shot** — con ejemplos en el prompt para mejorar precisión
3. **ML clásico (TF-IDF + Regresión logística)** — cuando tienes datos etiquetados

Caso de uso ejemplo: **análisis de sentimiento** de reseñas de productos (POSITIVO / NEGATIVO / NEUTRO).

---

## 2. Enfoque 1: Clasificación con LLM (zero-shot)

```python
import anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def clasificar_sentimiento_zerosshot(texto: str) -> dict:
    """
    Clasifica el sentimiento de un texto usando Claude sin ejemplos.
    Devuelve dict con 'etiqueta' y 'confianza'.
    """
    prompt = f"""Clasifica el sentimiento del siguiente texto en una de estas categorías:
- POSITIVO: opinión favorable, satisfacción, elogio
- NEGATIVO: opinión desfavorable, queja, crítica
- NEUTRO: descripción objetiva, sin valoración clara

Responde ÚNICAMENTE con un JSON válido con este formato exacto:
{{"etiqueta": "POSITIVO|NEGATIVO|NEUTRO", "confianza": 0.0-1.0, "razon": "breve explicación"}}

Texto a clasificar:
"{texto}"
"""

    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku es suficiente y más barato para clasificación
        max_tokens=150,
        temperature=0.0,  # Determinista para clasificación
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(respuesta.content[0].text)
    except json.JSONDecodeError:
        return {"etiqueta": "ERROR", "confianza": 0.0, "razon": "No se pudo parsear la respuesta"}


# Prueba
textos_ejemplo = [
    "El producto llegó en perfecto estado y antes de lo esperado. ¡Muy recomendable!",
    "Llevo tres semanas esperando el reembolso y nadie me responde. Pésimo servicio.",
    "El artículo tiene las dimensiones indicadas en la descripción.",
    "La calidad podría ser mejor para el precio que cobran.",
]

for texto in textos_ejemplo:
    resultado = clasificar_sentimiento_zerosshot(texto)
    print(f"Texto: {texto[:60]}...")
    print(f"  → {resultado['etiqueta']} (confianza: {resultado['confianza']:.0%})")
    print(f"  → Razón: {resultado['razon']}\n")
```

---

## 3. Enfoque 2: Clasificación con LLM (few-shot)

Para categorías específicas del negocio o cuando el zero-shot da resultados inconsistentes.

```python
EJEMPLOS_FEW_SHOT = [
    {
        "texto": "La app se cuelga cada vez que intento abrir el menú de configuración",
        "etiqueta": "BUG",
        "razon": "Problema técnico reproducible"
    },
    {
        "texto": "Sería genial poder exportar los datos en formato CSV",
        "etiqueta": "FEATURE_REQUEST",
        "razon": "Solicitud de nueva funcionalidad"
    },
    {
        "texto": "No entiendo cómo cambiar la contraseña desde el móvil",
        "etiqueta": "PREGUNTA",
        "razon": "El usuario pide ayuda, no reporta un fallo"
    },
    {
        "texto": "Llevo pagando la suscripción pero no tengo acceso premium",
        "etiqueta": "INCIDENCIA",
        "razon": "Problema de acceso vinculado a una transacción"
    },
]

CATEGORIAS = ["BUG", "FEATURE_REQUEST", "PREGUNTA", "INCIDENCIA", "OTRO"]


def clasificar_ticket_soporte(texto: str) -> dict:
    """Clasifica tickets de soporte con few-shot."""

    ejemplos_str = "\n".join([
        f'Ticket: "{ej["texto"]}"\n→ {{"etiqueta": "{ej["etiqueta"]}", "razon": "{ej["razon"]}"}}'
        for ej in EJEMPLOS_FEW_SHOT
    ])

    prompt = f"""Clasifica el siguiente ticket de soporte en una de estas categorías:
{', '.join(CATEGORIAS)}

Ejemplos:
{ejemplos_str}

Ahora clasifica este ticket:
"{texto}"

Responde SOLO con JSON: {{"etiqueta": "...", "confianza": 0.0-1.0, "razon": "..."}}"""

    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(respuesta.content[0].text)


# Prueba
tickets = [
    "El botón de pago no funciona en Safari",
    "¿Cuántos usuarios puedo añadir con la cuenta Pro?",
    "Me habéis cobrado dos veces este mes",
    "Necesito poder filtrar por fecha en los reportes",
]

for ticket in tickets:
    r = clasificar_ticket_soporte(ticket)
    print(f"'{ticket}'")
    print(f"  → {r['etiqueta']} ({r['confianza']:.0%}) — {r['razon']}\n")
```

---

## 4. Enfoque 3: Clasificación con ML clásico

Cuando tienes un dataset etiquetado y necesitas velocidad o bajo coste por predicción.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


def entrenar_clasificador_texto(df: pd.DataFrame,
                                 col_texto: str = "texto",
                                 col_etiqueta: str = "etiqueta") -> Pipeline:
    """Entrena un clasificador TF-IDF + Regresión Logística."""

    X = df[col_texto]
    y = df[col_etiqueta]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),        # Unigramas y bigramas
            min_df=2,                  # Ignorar términos que aparecen < 2 veces
            sublinear_tf=True,         # Suavizar frecuencias
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",  # Maneja clases desbalanceadas
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluación
    y_pred = pipeline.predict(X_test)
    print("=== Resultados en test ===")
    print(classification_report(y_test, y_pred))

    return pipeline


def clasificar_con_modelo(pipeline: Pipeline, textos: list[str]) -> list[dict]:
    """Clasifica una lista de textos con el modelo entrenado."""
    etiquetas = pipeline.predict(textos)
    probabilidades = pipeline.predict_proba(textos)
    clases = pipeline.classes_

    resultados = []
    for texto, etiqueta, probs in zip(textos, etiquetas, probabilidades):
        confianza = max(probs)
        resultados.append({
            "texto": texto,
            "etiqueta": etiqueta,
            "confianza": round(float(confianza), 3),
            "probabilidades": {c: round(float(p), 3) for c, p in zip(clases, probs)}
        })
    return resultados


# ─── Ejemplo de uso ──────────────────────────────────────────────────────────

# Simular dataset (en producción lo cargarías desde CSV)
datos = {
    "texto": [
        "Excelente producto, muy recomendable",
        "Llegó roto y el servicio no responde",
        "Producto estándar, hace lo que promete",
        "Increíble calidad para el precio",
        "Muy decepcionado con la compra",
        "Es lo que dice la descripción, ni más ni menos",
        "Superó todas mis expectativas, compraré más",
        "No vuelvo a comprar aquí, pésima experiencia",
        "Funciona correctamente",
        "Tardó demasiado en llegar y llegó dañado",
    ] * 20,  # Replicar para tener suficientes datos
    "etiqueta": [
        "POSITIVO", "NEGATIVO", "NEUTRO",
        "POSITIVO", "NEGATIVO", "NEUTRO",
        "POSITIVO", "NEGATIVO", "NEUTRO", "NEGATIVO"
    ] * 20
}
df = pd.DataFrame(datos)

# Entrenar
modelo = entrenar_clasificador_texto(df)

# Guardar modelo
joblib.dump(modelo, "modelo_sentimiento.joblib")

# Cargar y usar
modelo_cargado = joblib.load("modelo_sentimiento.joblib")
nuevos_textos = [
    "El envío fue rapidísimo y el producto perfecto",
    "Mala calidad, no lo recomiendo",
    "Cumple con lo indicado"
]
resultados = clasificar_con_modelo(modelo_cargado, nuevos_textos)
for r in resultados:
    print(f"{r['etiqueta']:8} ({r['confianza']:.0%}) — {r['texto']}")
```

---

## 5. Comparativa de enfoques

| Criterio | LLM Zero-shot | LLM Few-shot | ML clásico |
|---|---|---|---|
| **Datos de entrenamiento** | No necesita | Solo ejemplos en prompt | Sí (mínimo ~200/clase) |
| **Precisión en categorías estándar** | Alta | Muy alta | Alta (con suficientes datos) |
| **Precisión en categorías específicas** | Media | Alta | Muy alta |
| **Velocidad** | Lenta (~1s/texto) | Lenta (~1s/texto) | Muy rápida (<1ms/texto) |
| **Coste** | Medio | Medio | Muy bajo (solo CPU) |
| **Categorías nuevas** | Inmediato | Añadir ejemplo | Re-entrenar |
| **Explicabilidad** | Razonamiento en texto | Razonamiento en texto | Feature importance |
| **Producción a escala** | Costoso | Costoso | Ideal |

---

## 6. Clasificación de múltiples documentos

Para procesar grandes volúmenes de texto, procesa en lotes con control de tasa de peticiones:

```python
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def clasificar_lote(textos: list[str],
                    clasificador_fn,
                    workers: int = 5,
                    pausa_entre_lotes: float = 1.0) -> list[dict]:
    """
    Clasifica una lista de textos en paralelo con control de rate limit.
    """
    resultados = [None] * len(textos)
    lote_size = workers * 5  # Procesar en lotes

    for inicio in tqdm(range(0, len(textos), lote_size), desc="Clasificando"):
        lote = textos[inicio:inicio + lote_size]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futuros = {
                executor.submit(clasificador_fn, texto): i + inicio
                for i, texto in enumerate(lote)
            }
            for futuro in as_completed(futuros):
                idx = futuros[futuro]
                try:
                    resultados[idx] = futuro.result()
                except Exception as e:
                    resultados[idx] = {"etiqueta": "ERROR", "error": str(e)}

        time.sleep(pausa_entre_lotes)  # Respetar rate limits

    return resultados


# Uso
textos_a_clasificar = ["texto " + str(i) for i in range(100)]
resultados = clasificar_lote(
    textos_a_clasificar,
    clasificar_sentimiento_zerosshot,
    workers=3
)

# Guardar resultados
df_resultados = pd.DataFrame([
    {"texto": t, **r}
    for t, r in zip(textos_a_clasificar, resultados)
])
df_resultados.to_csv("resultados_clasificacion.csv", index=False)
```

---

## 7. Cuándo usar cada enfoque

```
¿Tienes un dataset etiquetado de al menos 200 ejemplos por categoría?
    │
    ├── SÍ → ML clásico (TF-IDF + LR o Random Forest)
    │         Ventaja: rápido, barato, escalable
    │
    └── NO
         │
         ├── ¿Las categorías son estándar (sentimiento, spam, idioma...)?
         │       └── SÍ → LLM zero-shot con Haiku (barato y suficiente)
         │
         └── ¿Las categorías son específicas de tu negocio?
                 └── SÍ → LLM few-shot (3-5 ejemplos por categoría en el prompt)

Combinación ganadora para producción a escala:
  → Usa LLM para etiquetar un dataset inicial (~500-1000 ejemplos)
  → Entrena un modelo ML clásico con esos datos
  → El modelo ML clasifica a alta velocidad y bajo coste
  → El LLM solo se usa para casos de baja confianza (fallback)
```

---

**Anterior:** [01 — Chatbot con Claude API](./01-chatbot-claude-api.md) · **Siguiente:** [03 — Resumen de documentos](./03-resumen-documentos.md)
