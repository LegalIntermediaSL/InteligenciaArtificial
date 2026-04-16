# 10 — Agentes Especializados por Dominio

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 70 min

---

## Índice

1. Por qué la especialización mejora los agentes
2. Agente de investigación (Research Agent)
3. Agente de análisis financiero (Financial Agent)
4. Agente de revisión de código (Code Review Agent)
5. Guía de selección: cuándo usar cada tipo
6. Patrones comunes de especialización

---

## 1. Por qué la especialización mejora los agentes

Un agente de propósito general tiene acceso a muchas herramientas y un system prompt amplio. Funciona para tareas variadas, pero su rendimiento en dominios específicos rara vez es óptimo. La especialización por dominio ataca este problema en tres frentes.

### 1.1 System prompt enfocado

Un system prompt genérico intenta cubrir todos los casos posibles. Como resultado, el modelo dedica tokens a instrucciones que no aplican a la tarea actual y la señal relevante queda diluida.

Un system prompt especializado describe exactamente:

- El rol del agente y su alcance
- El vocabulario y los estándares del dominio
- Las restricciones explícitas (qué no debe hacer nunca)
- El formato de salida esperado

Esto no es solo cuestión de estilo: hay evidencia empírica de que los modelos de lenguaje son más precisos cuando el contexto del sistema delimita claramente el espacio de respuestas válidas.

### 1.2 Herramientas acotadas al dominio

Un agente generalista puede tener decenas de herramientas registradas. Cuantas más herramientas hay disponibles, mayor es la probabilidad de que el modelo elija la herramienta incorrecta o se pierda en un espacio de decisión demasiado amplio.

Un agente especializado expone únicamente las herramientas que tienen sentido en su dominio:

```
Agente generalista (20 herramientas):
  buscar_web, leer_url, ejecutar_sql, leer_fichero, escribir_fichero,
  enviar_email, obtener_precio, calcular_metrica, ejecutar_linter,
  buscar_patron, listar_directorio, llamar_api, crear_calendario...

Agente de investigación (4 herramientas):
  buscar_web, leer_url, extraer_pdf, guardar_nota
```

Menos herramientas significa menos ambigüedad en la selección y prompts de herramientas más precisos.

### 1.3 Evaluación del dominio

Los agentes especializados pueden definir rúbricas de calidad específicas para su dominio. Un agente de revisión de código sabe que debe buscar vulnerabilidades de seguridad; un agente financiero sabe que debe incluir advertencias de riesgo; un agente de investigación sabe que debe verificar las fuentes.

Esta evaluación interna permite que el agente detecte sus propios errores antes de entregar el resultado.

### El coste de la especialización

La especialización tiene un coste: necesitas mantener múltiples agentes en lugar de uno. La compensación vale la pena cuando:

- El dominio tiene estándares propios (seguridad, finanzas, medicina, legal)
- Los errores en el dominio tienen consecuencias concretas
- El volumen de uso justifica el mantenimiento adicional

---

## 2. Agente de investigación (Research Agent)

El Research Agent toma una pregunta o tema y produce un informe estructurado con fuentes verificadas. Su pipeline tiene cuatro fases: búsqueda, lectura, síntesis y verificación.

### 2.1 Herramientas del dominio

| Herramienta | Propósito |
|---|---|
| `buscar_web` | Obtener URLs relevantes para una consulta |
| `leer_url` | Extraer el contenido limpio de una página web |
| `extraer_pdf` | Leer el texto de un documento PDF accesible por URL |
| `guardar_nota` | Almacenar fragmentos importantes para la síntesis final |

### 2.2 Pipeline de investigación

```
Consulta del usuario
       │
       ▼
┌─────────────────┐
│    BÚSQUEDA     │  → buscar_web() × 2-3 consultas distintas
│  Identifica     │
│  fuentes clave  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    LECTURA      │  → leer_url() / extraer_pdf() por cada fuente
│  Extrae         │
│  información    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    SÍNTESIS     │  → guardar_nota() con hallazgos clave
│  Consolida y    │
│  contrasta      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VERIFICACIÓN   │  → Contrasta afirmaciones entre fuentes
│  Detecta        │
│  contradicciones│
└────────┬────────┘
         │
         ▼
   Informe en Markdown
   con secciones y citas
```

### 2.3 Implementación completa

```python
# research_agent.py
# Agente de investigación especializado con Claude y herramientas de búsqueda.
# Dependencias: anthropic, requests, beautifulsoup4, pypdf2
# pip install anthropic requests beautifulsoup4 pypdf2

import json
import os
import re
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import anthropic
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Implementaciones de herramientas
# ---------------------------------------------------------------------------

def buscar_web(consulta: str, num_resultados: int = 5) -> list[dict]:
    """
    Busca en la web usando DuckDuckGo Instant Answer API (sin clave de API).
    Devuelve una lista de {titulo, url, snippet}.
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": consulta,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]

    resultados = []
    # RelatedTopics contiene los resultados más ricos
    for item in data.get("RelatedTopics", [])[:num_resultados]:
        if "FirstURL" in item and "Text" in item:
            resultados.append({
                "titulo": item.get("Text", "")[:80],
                "url": item["FirstURL"],
                "snippet": item.get("Text", ""),
            })

    # Si no hay resultados en RelatedTopics, usar Abstract
    if not resultados and data.get("AbstractURL"):
        resultados.append({
            "titulo": data.get("Heading", "Sin título"),
            "url": data["AbstractURL"],
            "snippet": data.get("AbstractText", ""),
        })

    return resultados if resultados else [{"mensaje": "Sin resultados para: " + consulta}]


def leer_url(url: str, max_chars: int = 4000) -> dict:
    """
    Descarga el HTML de una URL y extrae el texto limpio.
    Devuelve {url, titulo, contenido, num_palabras}.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"No se pudo acceder a {url}: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Eliminar elementos no relevantes
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    titulo = soup.title.string.strip() if soup.title else urlparse(url).netloc
    texto = " ".join(soup.get_text(separator=" ").split())

    # Truncar a max_chars para no saturar el contexto
    contenido = texto[:max_chars]
    if len(texto) > max_chars:
        contenido += "... [truncado]"

    return {
        "url": url,
        "titulo": titulo,
        "contenido": contenido,
        "num_palabras": len(texto.split()),
    }


def extraer_pdf(url: str, max_chars: int = 4000) -> dict:
    """
    Descarga un PDF desde una URL y extrae su texto.
    Devuelve {url, num_paginas, contenido}.
    """
    try:
        import PyPDF2
    except ImportError:
        return {"error": "Instala PyPDF2: pip install PyPDF2"}

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"No se pudo descargar el PDF: {e}"}

    try:
        reader = PyPDF2.PdfReader(BytesIO(resp.content))
        paginas = []
        for page in reader.pages:
            paginas.append(page.extract_text() or "")
        texto_completo = "\n".join(paginas)
        contenido = texto_completo[:max_chars]
        if len(texto_completo) > max_chars:
            contenido += "... [truncado]"
        return {
            "url": url,
            "num_paginas": len(reader.pages),
            "contenido": contenido,
        }
    except Exception as e:
        return {"error": f"Error al parsear el PDF: {e}"}


# Almacén en memoria para notas del agente durante la sesión
_notas: list[dict] = []

def guardar_nota(titulo: str, contenido: str, fuente: str = "") -> dict:
    """
    Guarda una nota con información relevante para la síntesis final.
    Devuelve confirmación con el número de notas acumuladas.
    """
    _notas.append({"titulo": titulo, "contenido": contenido, "fuente": fuente})
    return {"guardado": True, "total_notas": len(_notas), "titulo": titulo}


def obtener_notas() -> list[dict]:
    """Devuelve todas las notas guardadas durante la sesión."""
    return _notas


# ---------------------------------------------------------------------------
# Definición de herramientas para la API de Anthropic
# ---------------------------------------------------------------------------

HERRAMIENTAS_RESEARCH = [
    {
        "name": "buscar_web",
        "description": (
            "Busca en la web información sobre una consulta. "
            "Usa esta herramienta para encontrar fuentes relevantes sobre el tema de investigación. "
            "Devuelve una lista de URLs con título y snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Términos de búsqueda en lenguaje natural o keywords específicos.",
                },
                "num_resultados": {
                    "type": "integer",
                    "description": "Número de resultados a obtener (máximo 5).",
                    "default": 5,
                },
            },
            "required": ["consulta"],
        },
    },
    {
        "name": "leer_url",
        "description": (
            "Lee el contenido de una página web dado su URL. "
            "Usa esta herramienta después de buscar_web para acceder al contenido completo "
            "de las fuentes más relevantes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL completa de la página a leer (debe comenzar con http:// o https://).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "extraer_pdf",
        "description": (
            "Extrae el texto de un documento PDF accesible por URL. "
            "Usa esta herramienta cuando una fuente relevante sea un PDF (paper, informe, estudio)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL completa del archivo PDF.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "guardar_nota",
        "description": (
            "Guarda un fragmento de información importante para usarlo en la síntesis final. "
            "Usa esta herramienta para almacenar hallazgos clave mientras investigas, "
            "antes de redactar el informe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "titulo": {
                    "type": "string",
                    "description": "Título breve que identifica el hallazgo.",
                },
                "contenido": {
                    "type": "string",
                    "description": "Información relevante extraída de la fuente.",
                },
                "fuente": {
                    "type": "string",
                    "description": "URL o nombre de la fuente de donde proviene la información.",
                },
            },
            "required": ["titulo", "contenido"],
        },
    },
]

# ---------------------------------------------------------------------------
# Ejecución de herramientas
# ---------------------------------------------------------------------------

def ejecutar_herramienta_research(nombre: str, parametros: dict) -> Any:
    """Enruta las llamadas del agente a las funciones correspondientes."""
    if nombre == "buscar_web":
        return buscar_web(**parametros)
    elif nombre == "leer_url":
        return leer_url(**parametros)
    elif nombre == "extraer_pdf":
        return extraer_pdf(**parametros)
    elif nombre == "guardar_nota":
        return guardar_nota(**parametros)
    else:
        return {"error": f"Herramienta desconocida: {nombre}"}

# ---------------------------------------------------------------------------
# Bucle del agente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RESEARCH = """Eres un agente de investigación especializado. Tu misión es producir informes precisos, bien estructurados y con fuentes verificadas sobre cualquier tema que se te encargue.

Proceso de trabajo obligatorio:
1. Realiza al menos 2 búsquedas con consultas distintas para triangular la información.
2. Lee el contenido completo de las 3 fuentes más relevantes.
3. Guarda notas con los hallazgos más importantes antes de sintetizar.
4. Contrasta la información entre fuentes: si encuentras contradicciones, inclúyelas en el informe.
5. Redacta un informe en Markdown con las siguientes secciones:
   - ## Resumen ejecutivo (3-5 líneas)
   - ## Hallazgos principales (puntos clave con evidencia)
   - ## Análisis y contexto
   - ## Puntos en disputa o inciertos (si los hay)
   - ## Fuentes consultadas (lista numerada con URLs)

Restricciones:
- No inventes datos. Si no encuentras información fiable, indícalo explícitamente.
- No hagas afirmaciones sin citar la fuente.
- Si una fuente no es accesible, busca una alternativa.
- El informe final debe ser útil para alguien que no tenga tiempo de leer todas las fuentes."""


def research_agent(pregunta: str, max_iteraciones: int = 15) -> str:
    """
    Ejecuta el Research Agent para investigar una pregunta.

    Args:
        pregunta: Tema o pregunta de investigación.
        max_iteraciones: Límite de ciclos herramienta→modelo para evitar bucles.

    Returns:
        Informe en Markdown como string.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Limpiar notas de sesiones anteriores
    _notas.clear()

    mensajes = [{"role": "user", "content": pregunta}]
    iteracion = 0

    print(f"[Research Agent] Iniciando investigación: {pregunta[:60]}...")

    while iteracion < max_iteraciones:
        iteracion += 1

        respuesta = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT_RESEARCH,
            tools=HERRAMIENTAS_RESEARCH,
            messages=mensajes,
        )

        # Acumular la respuesta del asistente en el historial
        mensajes.append({"role": "assistant", "content": respuesta.content})

        # Si el modelo ha terminado, extraer el texto final
        if respuesta.stop_reason == "end_turn":
            for bloque in respuesta.content:
                if hasattr(bloque, "text"):
                    print(f"[Research Agent] Completado en {iteracion} iteraciones.")
                    return bloque.text
            break

        # Si el modelo quiere usar herramientas, ejecutarlas
        if respuesta.stop_reason == "tool_use":
            resultados_herramientas = []

            for bloque in respuesta.content:
                if bloque.type == "tool_use":
                    nombre = bloque.name
                    parametros = bloque.input
                    print(f"  → Herramienta: {nombre}({list(parametros.keys())})")

                    resultado = ejecutar_herramienta_research(nombre, parametros)

                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": json.dumps(resultado, ensure_ascii=False),
                    })

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return "Error: el agente superó el límite de iteraciones sin producir un informe."


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    informe = research_agent(
        "¿Cuál es el estado actual de la computación cuántica y cuándo se espera "
        "que supere a los ordenadores clásicos en tareas prácticas?"
    )
    print("\n" + "=" * 60)
    print(informe)
```

---

## 3. Agente de análisis financiero (Financial Agent)

El Financial Agent recibe una cartera de activos (símbolos de bolsa) y produce un informe de análisis en Markdown con métricas de riesgo y rendimiento.

**Advertencia importante:** este agente realiza exclusivamente análisis. No ejecuta operaciones, no envía órdenes a brokers y no mueve dinero. Toda decisión de inversión debe tomarla el usuario.

### 3.1 Herramientas del dominio

| Herramienta | Propósito |
|---|---|
| `obtener_precio_historico` | Descarga precios ajustados de cierre (yfinance) |
| `calcular_metricas` | Volatilidad anualizada, Sharpe ratio, drawdown máximo |
| `calcular_correlacion` | Matriz de correlación entre activos |
| `generar_resumen_activo` | Datos fundamentales: nombre, sector, capitalización |

### 3.2 Pipeline de análisis

```
Cartera de activos [símbolo, peso]
          │
          ▼
┌─────────────────────┐
│  DESCARGA DE DATOS  │  → obtener_precio_historico() por cada activo
│  Precios históricos │
│  últimos 12 meses   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ MÉTRICAS INVIDUAL   │  → calcular_metricas() por cada activo
│ Volatilidad, Sharpe │
│ Drawdown, retorno   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   CORRELACIONES     │  → calcular_correlacion() sobre toda la cartera
│   Diversificación   │
│   real vs. nominal  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  INFORME MARKDOWN   │  Resumen ejecutivo, tabla de métricas,
│  con advertencias   │  análisis de riesgo, advertencias legales
└─────────────────────┘
```

### 3.3 Implementación completa

```python
# financial_agent.py
# Agente de análisis financiero especializado con Claude y yfinance.
# Dependencias: anthropic, yfinance, numpy, pandas
# pip install anthropic yfinance numpy pandas

import json
import os
from datetime import datetime, timedelta
from typing import Any

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Implementaciones de herramientas
# ---------------------------------------------------------------------------

def obtener_precio_historico(simbolo: str, periodo_dias: int = 365) -> dict:
    """
    Descarga precios de cierre ajustados para el símbolo dado.
    Devuelve {simbolo, fechas, precios, retornos_diarios}.
    """
    fecha_fin = datetime.today()
    fecha_inicio = fecha_fin - timedelta(days=periodo_dias)

    try:
        ticker = yf.Ticker(simbolo)
        hist = ticker.history(start=fecha_inicio.strftime("%Y-%m-%d"),
                              end=fecha_fin.strftime("%Y-%m-%d"))

        if hist.empty:
            return {"error": f"No se encontraron datos para {simbolo}"}

        precios = hist["Close"].dropna()
        retornos = precios.pct_change().dropna()

        return {
            "simbolo": simbolo,
            "fecha_inicio": precios.index[0].strftime("%Y-%m-%d"),
            "fecha_fin": precios.index[-1].strftime("%Y-%m-%d"),
            "num_dias": len(precios),
            "precio_inicial": round(float(precios.iloc[0]), 4),
            "precio_final": round(float(precios.iloc[-1]), 4),
            "retorno_total_pct": round((float(precios.iloc[-1]) / float(precios.iloc[0]) - 1) * 100, 2),
            # Guardamos los retornos como lista para calcular métricas
            "_retornos_diarios": retornos.tolist(),
            "_precios": precios.tolist(),
        }
    except Exception as e:
        return {"error": f"Error al obtener datos de {simbolo}: {e}"}


def calcular_metricas(simbolo: str, retornos_diarios: list[float]) -> dict:
    """
    Calcula métricas de riesgo-rendimiento a partir de retornos diarios.
    Devuelve volatilidad anualizada, Sharpe ratio, drawdown máximo y más.
    """
    if not retornos_diarios or len(retornos_diarios) < 20:
        return {"error": "Se necesitan al menos 20 días de datos para calcular métricas."}

    r = np.array(retornos_diarios)

    # Tasa libre de riesgo diaria (asumimos 4.5% anual)
    tasa_libre_riesgo_diaria = 0.045 / 252

    # Métricas básicas
    retorno_medio_diario = float(np.mean(r))
    volatilidad_diaria = float(np.std(r, ddof=1))
    volatilidad_anualizada = volatilidad_diaria * np.sqrt(252)

    # Sharpe ratio anualizado
    exceso_retorno = retorno_medio_diario - tasa_libre_riesgo_diaria
    sharpe = (exceso_retorno / volatilidad_diaria) * np.sqrt(252) if volatilidad_diaria > 0 else 0.0

    # Drawdown máximo
    precios_norm = np.cumprod(1 + r)  # precio normalizado desde 1
    picos = np.maximum.accumulate(precios_norm)
    drawdowns = (precios_norm - picos) / picos
    max_drawdown = float(np.min(drawdowns))

    # Value at Risk al 95% (paramétrico)
    var_95 = float(np.percentile(r, 5))

    return {
        "simbolo": simbolo,
        "retorno_medio_diario_pct": round(retorno_medio_diario * 100, 4),
        "volatilidad_anualizada_pct": round(volatilidad_anualizada * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "var_95_diario_pct": round(var_95 * 100, 4),
        "dias_positivos_pct": round(float(np.sum(r > 0) / len(r)) * 100, 1),
    }


def calcular_correlacion(simbolos: list[str], retornos_por_simbolo: dict[str, list[float]]) -> dict:
    """
    Calcula la matriz de correlación entre los retornos de los activos.
    Devuelve la matriz como diccionario anidado y pares de correlación alta.
    """
    if len(simbolos) < 2:
        return {"mensaje": "Se necesitan al menos 2 activos para calcular correlaciones."}

    # Alinear las series (pueden tener distinta longitud)
    min_len = min(len(retornos_por_simbolo[s]) for s in simbolos if s in retornos_por_simbolo)
    df = pd.DataFrame(
        {s: retornos_por_simbolo[s][-min_len:] for s in simbolos if s in retornos_por_simbolo}
    )

    corr_matrix = df.corr()

    # Identificar pares con correlación alta (>0.7) o baja (<-0.3)
    pares_notables = []
    for i, s1 in enumerate(simbolos):
        for s2 in simbolos[i + 1:]:
            if s1 in corr_matrix.columns and s2 in corr_matrix.columns:
                c = float(corr_matrix.loc[s1, s2])
                if abs(c) > 0.7 or c < -0.3:
                    pares_notables.append({
                        "par": f"{s1}-{s2}",
                        "correlacion": round(c, 3),
                        "interpretacion": "alta correlación (baja diversificación)" if c > 0.7
                                          else "correlación negativa (buena cobertura)",
                    })

    # Convertir la matriz a diccionario serializable
    matriz = {
        s1: {s2: round(float(corr_matrix.loc[s1, s2]), 3)
             for s2 in corr_matrix.columns if s2 in corr_matrix.index}
        for s1 in corr_matrix.index
    }

    return {
        "matriz_correlacion": matriz,
        "pares_notables": pares_notables,
        "num_activos_analizados": len(df.columns),
    }


def generar_resumen_activo(simbolo: str) -> dict:
    """
    Obtiene información fundamental del activo: nombre, sector, capitalización.
    """
    try:
        ticker = yf.Ticker(simbolo)
        info = ticker.info

        return {
            "simbolo": simbolo,
            "nombre": info.get("longName", info.get("shortName", simbolo)),
            "sector": info.get("sector", "N/D"),
            "industria": info.get("industry", "N/D"),
            "pais": info.get("country", "N/D"),
            "moneda": info.get("currency", "N/D"),
            "capitalizacion_mercado": info.get("marketCap", None),
            "descripcion": (info.get("longBusinessSummary", "Sin descripción disponible.") or "")[:300],
        }
    except Exception as e:
        return {"simbolo": simbolo, "error": str(e)}


# ---------------------------------------------------------------------------
# Definición de herramientas para la API de Anthropic
# ---------------------------------------------------------------------------

HERRAMIENTAS_FINANCIAL = [
    {
        "name": "obtener_precio_historico",
        "description": (
            "Descarga los precios históricos de cierre ajustados de un activo financiero. "
            "Usa esta herramienta primero para cada activo de la cartera antes de calcular métricas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "simbolo": {
                    "type": "string",
                    "description": "Símbolo bursátil del activo (ej: AAPL, MSFT, BTC-USD, SPY).",
                },
                "periodo_dias": {
                    "type": "integer",
                    "description": "Número de días de histórico a descargar. Por defecto 365.",
                    "default": 365,
                },
            },
            "required": ["simbolo"],
        },
    },
    {
        "name": "calcular_metricas",
        "description": (
            "Calcula métricas de riesgo y rendimiento para un activo a partir de sus retornos diarios. "
            "Incluye volatilidad anualizada, Sharpe ratio, drawdown máximo y VaR al 95%."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "simbolo": {
                    "type": "string",
                    "description": "Símbolo bursátil del activo.",
                },
                "retornos_diarios": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Lista de retornos diarios como decimales (ej: 0.012 para un 1.2%).",
                },
            },
            "required": ["simbolo", "retornos_diarios"],
        },
    },
    {
        "name": "calcular_correlacion",
        "description": (
            "Calcula la matriz de correlación entre los retornos de múltiples activos. "
            "Úsala después de obtener los retornos de todos los activos de la cartera."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "simbolos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de símbolos bursátiles a comparar.",
                },
                "retornos_por_simbolo": {
                    "type": "object",
                    "description": "Diccionario {símbolo: [lista de retornos diarios]}.",
                },
            },
            "required": ["simbolos", "retornos_por_simbolo"],
        },
    },
    {
        "name": "generar_resumen_activo",
        "description": (
            "Obtiene información fundamental de un activo: nombre completo, sector, "
            "industria, país y capitalización de mercado."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "simbolo": {
                    "type": "string",
                    "description": "Símbolo bursátil del activo.",
                },
            },
            "required": ["simbolo"],
        },
    },
]

# ---------------------------------------------------------------------------
# Ejecución de herramientas
# ---------------------------------------------------------------------------

def ejecutar_herramienta_financial(nombre: str, parametros: dict) -> Any:
    """Enruta las llamadas del agente a las funciones correspondientes."""
    if nombre == "obtener_precio_historico":
        return obtener_precio_historico(**parametros)
    elif nombre == "calcular_metricas":
        return calcular_metricas(**parametros)
    elif nombre == "calcular_correlacion":
        return calcular_correlacion(**parametros)
    elif nombre == "generar_resumen_activo":
        return generar_resumen_activo(**parametros)
    else:
        return {"error": f"Herramienta desconocida: {nombre}"}

# ---------------------------------------------------------------------------
# Bucle del agente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_FINANCIAL = """Eres un agente especializado en análisis financiero cuantitativo. Tu función es analizar carteras de activos y producir informes objetivos de riesgo y rendimiento.

RESTRICCIÓN FUNDAMENTAL: Nunca recomiendes comprar, vender ni mantener ningún activo. Nunca ejecutes operaciones de ningún tipo. Tu único rol es el análisis descriptivo y la presentación de métricas.

Proceso de análisis obligatorio:
1. Para cada activo de la cartera: obtén el histórico de precios y calcula sus métricas individuales.
2. Obtén información fundamental de cada activo.
3. Calcula la matriz de correlación de la cartera completa.
4. Redacta un informe en Markdown con las siguientes secciones:

   ## Resumen de la cartera
   (Activos analizados, período de análisis, monedas)

   ## Métricas por activo
   (Tabla con: símbolo, retorno total, volatilidad anualizada, Sharpe ratio, drawdown máximo, VaR 95%)

   ## Análisis de correlaciones
   (Matriz simplificada, pares notables, comentario sobre diversificación)

   ## Perfil de riesgo de la cartera
   (Análisis cualitativo de los riesgos identificados en las métricas)

   ## Advertencia legal
   (Incluir siempre: este análisis es exclusivamente informativo y no constituye asesoramiento financiero)

Normas de formato:
- Usa tablas Markdown para métricas comparativas.
- Redondea métricas porcentuales a 2 decimales.
- Indica explícitamente si faltan datos de algún activo.
- Nunca omitas la sección de advertencia legal."""


def financial_agent(cartera: list[dict], max_iteraciones: int = 20) -> str:
    """
    Ejecuta el Financial Agent para analizar una cartera de activos.

    Args:
        cartera: Lista de dicts con claves 'simbolo' y opcionalmente 'peso'.
                 Ejemplo: [{"simbolo": "AAPL", "peso": 0.4}, {"simbolo": "MSFT", "peso": 0.6}]
        max_iteraciones: Límite de ciclos herramienta→modelo.

    Returns:
        Informe de análisis en Markdown como string.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    descripcion_cartera = json.dumps(cartera, ensure_ascii=False, indent=2)
    mensaje_usuario = (
        f"Analiza la siguiente cartera de activos y genera un informe completo de riesgo y rendimiento:\n\n"
        f"```json\n{descripcion_cartera}\n```"
    )

    mensajes = [{"role": "user", "content": mensaje_usuario}]
    iteracion = 0

    simbolos = [a["simbolo"] for a in cartera]
    print(f"[Financial Agent] Analizando cartera: {simbolos}")

    while iteracion < max_iteraciones:
        iteracion += 1

        respuesta = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT_FINANCIAL,
            tools=HERRAMIENTAS_FINANCIAL,
            messages=mensajes,
        )

        mensajes.append({"role": "assistant", "content": respuesta.content})

        if respuesta.stop_reason == "end_turn":
            for bloque in respuesta.content:
                if hasattr(bloque, "text"):
                    print(f"[Financial Agent] Completado en {iteracion} iteraciones.")
                    return bloque.text
            break

        if respuesta.stop_reason == "tool_use":
            resultados_herramientas = []

            for bloque in respuesta.content:
                if bloque.type == "tool_use":
                    nombre = bloque.name
                    parametros = bloque.input
                    print(f"  → Herramienta: {nombre}({list(parametros.keys())})")

                    resultado = ejecutar_herramienta_financial(nombre, parametros)

                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": json.dumps(resultado, ensure_ascii=False),
                    })

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return "Error: el agente superó el límite de iteraciones sin producir un informe."


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cartera_ejemplo = [
        {"simbolo": "AAPL", "peso": 0.30},
        {"simbolo": "MSFT", "peso": 0.25},
        {"simbolo": "SPY",  "peso": 0.25},
        {"simbolo": "BTC-USD", "peso": 0.20},
    ]

    informe = financial_agent(cartera_ejemplo)
    print("\n" + "=" * 60)
    print(informe)
```

---

## 4. Agente de revisión de código (Code Review Agent)

El Code Review Agent lee los ficheros modificados en un pull request o directorio, ejecuta herramientas de análisis estático y produce una revisión estructurada en Markdown lista para publicar como comentario de PR.

### 4.1 Herramientas del dominio

| Herramienta | Propósito |
|---|---|
| `leer_fichero` | Lee el contenido de un fichero de código fuente |
| `listar_directorio` | Lista los ficheros de un directorio con filtrado por extensión |
| `ejecutar_ruff` | Ejecuta el linter ruff y devuelve los problemas encontrados |
| `ejecutar_mypy` | Ejecuta mypy para análisis de tipos y devuelve los errores |
| `buscar_patron` | Busca patrones regex en un fichero (equivalente a grep) |

### 4.2 Rúbrica de revisión

El agente evalúa cada fichero según cuatro categorías:

```
SEGURIDAD
  - Inyección SQL, XSS, SSRF, path traversal
  - Credenciales o secretos hardcodeados
  - Deserialización insegura
  - Manejo incorrecto de excepciones (except: pass, bare except)

RENDIMIENTO
  - Consultas N+1 en bucles
  - Concatenación de strings en bucles
  - Importaciones innecesarias dentro de funciones
  - Objetos creados en cada llamada que podrían ser constantes

LEGIBILIDAD
  - Funciones con más de 40 líneas sin justificación
  - Nombres de variables de un solo carácter fuera de comprensiones
  - Comentarios que describen el "qué" en lugar del "por qué"
  - Complejidad ciclomática excesiva

TESTS
  - Cobertura de casos límite (None, lista vacía, valores negativos)
  - Tests que verifican el estado en lugar del comportamiento
  - Fixtures que podrían ser más granulares
```

### 4.3 Implementación completa

```python
# code_review_agent.py
# Agente de revisión de código especializado con Claude, ruff y mypy.
# Dependencias: anthropic, ruff, mypy
# pip install anthropic ruff mypy

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Implementaciones de herramientas
# ---------------------------------------------------------------------------

def leer_fichero(ruta: str, max_lineas: int = 300) -> dict:
    """
    Lee el contenido de un fichero de código fuente.
    Devuelve {ruta, contenido, num_lineas, truncado}.
    """
    try:
        path = Path(ruta)
        if not path.exists():
            return {"error": f"El fichero no existe: {ruta}"}
        if not path.is_file():
            return {"error": f"La ruta no es un fichero: {ruta}"}

        contenido = path.read_text(encoding="utf-8", errors="replace")
        lineas = contenido.splitlines()
        truncado = len(lineas) > max_lineas

        return {
            "ruta": str(path.resolve()),
            "extension": path.suffix,
            "num_lineas": len(lineas),
            "contenido": "\n".join(lineas[:max_lineas]),
            "truncado": truncado,
        }
    except Exception as e:
        return {"error": f"Error al leer {ruta}: {e}"}


def listar_directorio(directorio: str, extensiones: list[str] | None = None,
                       recursivo: bool = True) -> dict:
    """
    Lista los ficheros de un directorio, con filtrado opcional por extensión.
    Devuelve {directorio, ficheros: [{ruta, tamaño_kb, extension}]}.
    """
    try:
        base = Path(directorio)
        if not base.exists():
            return {"error": f"El directorio no existe: {directorio}"}

        patron = "**/*" if recursivo else "*"
        ficheros = []

        for path in sorted(base.glob(patron)):
            if not path.is_file():
                continue
            # Excluir directorios habituales que no son código fuente
            partes = path.parts
            if any(p in partes for p in {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}):
                continue
            if extensiones and path.suffix not in extensiones:
                continue

            ficheros.append({
                "ruta": str(path.relative_to(base)),
                "ruta_absoluta": str(path.resolve()),
                "extension": path.suffix,
                "tamaño_kb": round(path.stat().st_size / 1024, 2),
                "num_lineas": sum(1 for _ in path.open(encoding="utf-8", errors="replace")),
            })

        return {
            "directorio": str(base.resolve()),
            "num_ficheros": len(ficheros),
            "ficheros": ficheros,
        }
    except Exception as e:
        return {"error": f"Error al listar {directorio}: {e}"}


def ejecutar_ruff(ruta: str) -> dict:
    """
    Ejecuta el linter ruff sobre un fichero o directorio.
    Devuelve {ruta, num_problemas, problemas: [{linea, columna, codigo, mensaje}]}.
    """
    try:
        resultado = subprocess.run(
            ["ruff", "check", "--output-format=json", ruta],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # ruff devuelve JSON en stdout incluso con errores de lint
        if resultado.stdout.strip():
            try:
                diagnosticos = json.loads(resultado.stdout)
            except json.JSONDecodeError:
                return {"error": f"Salida de ruff no parseable: {resultado.stdout[:200]}"}
        else:
            diagnosticos = []

        problemas = [
            {
                "fichero": d.get("filename", ruta),
                "linea": d.get("location", {}).get("row", 0),
                "columna": d.get("location", {}).get("column", 0),
                "codigo": d.get("code", "?"),
                "mensaje": d.get("message", ""),
                "categoria": _categoria_ruff(d.get("code", "")),
            }
            for d in diagnosticos
        ]

        return {
            "ruta": ruta,
            "num_problemas": len(problemas),
            "problemas": problemas,
        }
    except FileNotFoundError:
        return {"error": "ruff no está instalado. Ejecuta: pip install ruff"}
    except subprocess.TimeoutExpired:
        return {"error": "ruff tardó demasiado (>30s)"}
    except Exception as e:
        return {"error": f"Error al ejecutar ruff: {e}"}


def _categoria_ruff(codigo: str) -> str:
    """Clasifica el código de ruff en una categoría legible."""
    prefijo = codigo[:1].upper() if codigo else ""
    mapa = {
        "E": "estilo", "W": "advertencia", "F": "error",
        "S": "seguridad", "B": "bug", "C": "complejidad",
        "N": "nomenclatura", "I": "imports", "UP": "modernización",
    }
    return mapa.get(prefijo, "otro")


def ejecutar_mypy(ruta: str) -> dict:
    """
    Ejecuta mypy sobre un fichero para análisis de tipos estáticos.
    Devuelve {ruta, num_errores, errores: [{linea, severidad, mensaje}]}.
    """
    try:
        resultado = subprocess.run(
            ["mypy", "--ignore-missing-imports", "--no-error-summary", ruta],
            capture_output=True,
            text=True,
            timeout=60,
        )

        errores = []
        patron = re.compile(r"^(.+):(\d+):\s*(error|warning|note):\s*(.+)$")

        for linea in resultado.stdout.splitlines():
            m = patron.match(linea)
            if m:
                errores.append({
                    "fichero": m.group(1),
                    "linea": int(m.group(2)),
                    "severidad": m.group(3),
                    "mensaje": m.group(4),
                })

        return {
            "ruta": ruta,
            "num_errores": len([e for e in errores if e["severidad"] == "error"]),
            "num_advertencias": len([e for e in errores if e["severidad"] == "warning"]),
            "errores": errores,
        }
    except FileNotFoundError:
        return {"error": "mypy no está instalado. Ejecuta: pip install mypy"}
    except subprocess.TimeoutExpired:
        return {"error": "mypy tardó demasiado (>60s)"}
    except Exception as e:
        return {"error": f"Error al ejecutar mypy: {e}"}


def buscar_patron(ruta: str, patron: str, contexto_lineas: int = 2) -> dict:
    """
    Busca un patrón regex en un fichero y devuelve las líneas coincidentes con contexto.
    Devuelve {ruta, patron, num_coincidencias, coincidencias: [{linea_num, linea, contexto}]}.
    """
    try:
        path = Path(ruta)
        if not path.exists():
            return {"error": f"El fichero no existe: {ruta}"}

        lineas = path.read_text(encoding="utf-8", errors="replace").splitlines()
        regex = re.compile(patron)
        coincidencias = []

        for i, linea in enumerate(lineas):
            if regex.search(linea):
                inicio = max(0, i - contexto_lineas)
                fin = min(len(lineas), i + contexto_lineas + 1)
                contexto = [
                    {"num": j + 1, "texto": lineas[j], "es_coincidencia": j == i}
                    for j in range(inicio, fin)
                ]
                coincidencias.append({
                    "linea_num": i + 1,
                    "linea": linea,
                    "contexto": contexto,
                })

        return {
            "ruta": ruta,
            "patron": patron,
            "num_coincidencias": len(coincidencias),
            "coincidencias": coincidencias,
        }
    except re.error as e:
        return {"error": f"Patrón regex inválido: {e}"}
    except Exception as e:
        return {"error": f"Error al buscar en {ruta}: {e}"}


# ---------------------------------------------------------------------------
# Definición de herramientas para la API de Anthropic
# ---------------------------------------------------------------------------

HERRAMIENTAS_CODE_REVIEW = [
    {
        "name": "leer_fichero",
        "description": (
            "Lee el contenido de un fichero de código fuente. "
            "Usa esta herramienta para examinar el código antes de emitir observaciones."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {
                    "type": "string",
                    "description": "Ruta absoluta o relativa al fichero.",
                },
                "max_lineas": {
                    "type": "integer",
                    "description": "Número máximo de líneas a leer. Por defecto 300.",
                    "default": 300,
                },
            },
            "required": ["ruta"],
        },
    },
    {
        "name": "listar_directorio",
        "description": (
            "Lista los ficheros de código en un directorio. "
            "Usa esta herramienta al inicio para entender el alcance de la revisión."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directorio": {
                    "type": "string",
                    "description": "Ruta al directorio a listar.",
                },
                "extensiones": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filtrar por extensiones (ej: [\".py\", \".ts\"]). Null para todas.",
                },
                "recursivo": {
                    "type": "boolean",
                    "description": "Si es true, busca en subdirectorios. Por defecto true.",
                    "default": True,
                },
            },
            "required": ["directorio"],
        },
    },
    {
        "name": "ejecutar_ruff",
        "description": (
            "Ejecuta el linter ruff sobre un fichero Python. "
            "Detecta errores de estilo, bugs comunes, problemas de seguridad y código muerto."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {
                    "type": "string",
                    "description": "Ruta al fichero o directorio Python a analizar.",
                },
            },
            "required": ["ruta"],
        },
    },
    {
        "name": "ejecutar_mypy",
        "description": (
            "Ejecuta mypy para verificar los tipos estáticos en un fichero Python. "
            "Detecta errores de tipos, atributos inexistentes y retornos incorrectos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {
                    "type": "string",
                    "description": "Ruta al fichero Python a verificar.",
                },
            },
            "required": ["ruta"],
        },
    },
    {
        "name": "buscar_patron",
        "description": (
            "Busca un patrón regex en un fichero. "
            "Usa esta herramienta para encontrar patrones de seguridad peligrosos, "
            "antipatrones conocidos o usos específicos de APIs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ruta": {
                    "type": "string",
                    "description": "Ruta al fichero donde buscar.",
                },
                "patron": {
                    "type": "string",
                    "description": "Expresión regular a buscar (sintaxis Python re).",
                },
                "contexto_lineas": {
                    "type": "integer",
                    "description": "Líneas de contexto a mostrar antes y después de cada coincidencia.",
                    "default": 2,
                },
            },
            "required": ["ruta", "patron"],
        },
    },
]

# ---------------------------------------------------------------------------
# Ejecución de herramientas
# ---------------------------------------------------------------------------

def ejecutar_herramienta_review(nombre: str, parametros: dict) -> Any:
    """Enruta las llamadas del agente a las funciones correspondientes."""
    if nombre == "leer_fichero":
        return leer_fichero(**parametros)
    elif nombre == "listar_directorio":
        return listar_directorio(**parametros)
    elif nombre == "ejecutar_ruff":
        return ejecutar_ruff(**parametros)
    elif nombre == "ejecutar_mypy":
        return ejecutar_mypy(**parametros)
    elif nombre == "buscar_patron":
        return buscar_patron(**parametros)
    else:
        return {"error": f"Herramienta desconocida: {nombre}"}

# ---------------------------------------------------------------------------
# Bucle del agente
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CODE_REVIEW = """Eres un agente especializado en revisión de código para entornos de producción. Tu objetivo es producir revisiones precisas, accionables y bien estructuradas.

Proceso de revisión obligatorio:
1. Lista los ficheros del directorio o revisa directamente los ficheros indicados.
2. Para cada fichero Python: ejecuta ruff y mypy antes de leer el código manualmente.
3. Lee el contenido del código para análisis manual.
4. Busca activamente patrones peligrosos de seguridad:
   - exec(, eval(, __import__(, pickle.loads(
   - password, secret, token, api_key en variables hardcodeadas
   - except: sin tipo de excepción (bare except)
   - subprocess con shell=True y entrada no sanitizada
5. Redacta la revisión en Markdown con la siguiente estructura:

   ## Resumen de la revisión
   Tabla: fichero | ruff | mypy | observaciones manuales | valoración (OK / Atención / Bloqueante)

   ## Problemas bloqueantes
   (Cualquier problema que impida el merge: seguridad crítica, errores de runtime)

   ## Observaciones de atención
   (Problemas importantes pero no bloqueantes: rendimiento, legibilidad compleja, tests faltantes)

   ## Sugerencias menores
   (Mejoras opcionales: estilo, nombres, comentarios)

   ## Puntos positivos
   (Menciona explícitamente lo que está bien hecho)

Normas de valoración:
- Bloqueante: vulnerabilidad de seguridad explotable, error de runtime garantizado, credenciales expuestas.
- Atención: código sin tests de casos límite, funciones >50 líneas sin justificación, dependencias sin versión fijada.
- Sugerencia: nombres poco descriptivos, comentarios redundantes, imports sin usar (si ruff no los detectó).

Formato de cada observación:
**[CATEGORÍA] Título breve**
Fichero: `ruta/al/fichero.py`, línea X
Descripción del problema.
```python
# código problemático
```
Sugerencia de corrección."""


def code_review_agent(objetivo: str, max_iteraciones: int = 25) -> str:
    """
    Ejecuta el Code Review Agent sobre un directorio o lista de ficheros.

    Args:
        objetivo: Ruta a un directorio o fichero a revisar. Puede incluir
                  instrucciones adicionales (ej: "Revisa /ruta/al/proyecto,
                  prestando especial atención a la autenticación").
        max_iteraciones: Límite de ciclos herramienta→modelo.

    Returns:
        Revisión completa en Markdown como string.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    mensajes = [{"role": "user", "content": objetivo}]
    iteracion = 0

    print(f"[Code Review Agent] Iniciando revisión: {objetivo[:60]}...")

    while iteracion < max_iteraciones:
        iteracion += 1

        respuesta = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=6000,
            system=SYSTEM_PROMPT_CODE_REVIEW,
            tools=HERRAMIENTAS_CODE_REVIEW,
            messages=mensajes,
        )

        mensajes.append({"role": "assistant", "content": respuesta.content})

        if respuesta.stop_reason == "end_turn":
            for bloque in respuesta.content:
                if hasattr(bloque, "text"):
                    print(f"[Code Review Agent] Completado en {iteracion} iteraciones.")
                    return bloque.text
            break

        if respuesta.stop_reason == "tool_use":
            resultados_herramientas = []

            for bloque in respuesta.content:
                if bloque.type == "tool_use":
                    nombre = bloque.name
                    parametros = bloque.input
                    print(f"  → Herramienta: {nombre}({list(parametros.keys())})")

                    resultado = ejecutar_herramienta_review(nombre, parametros)

                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": json.dumps(resultado, ensure_ascii=False),
                    })

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return "Error: el agente superó el límite de iteraciones sin producir una revisión."


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Revisar un directorio completo
    revision = code_review_agent(
        "Revisa el directorio /ruta/al/proyecto/src prestando especial atención "
        "a la seguridad en el manejo de entradas de usuario y a los tipos."
    )
    print("\n" + "=" * 60)
    print(revision)
```

---

## 5. Guía de selección: cuándo usar cada tipo de agente especializado

La siguiente tabla resume los criterios de decisión para elegir el tipo de agente adecuado.

| Criterio | Research Agent | Financial Agent | Code Review Agent |
|---|---|---|---|
| **Entrada principal** | Pregunta o tema en lenguaje natural | Lista de símbolos de activos | Ruta a código fuente |
| **Salida principal** | Informe con fuentes | Informe de métricas de riesgo | Revisión estructurada de PR |
| **Fuente de verdad** | Web, PDFs, URLs externas | APIs de datos de mercado | Sistema de ficheros local |
| **Herramientas críticas** | Búsqueda web, lectura de URLs | yfinance, numpy/pandas | ruff, mypy, regex |
| **Restricción más importante** | No inventar datos sin fuente | No ejecutar operaciones | No modificar código, solo revisar |
| **Latencia típica** | 30–120 s (depende de fuentes) | 20–60 s (descarga de datos) | 15–90 s (depende del tamaño) |
| **Coste de error** | Información incorrecta publicada | Análisis equivocado usado para decidir inversiones | Merge de código con vulnerabilidades |
| **Cuando es la elección correcta** | Investigación de mercado, due diligence, resúmenes de literatura | Reporting periódico, análisis de riesgo de cartera, backtest conceptual | CI/CD pre-merge, auditorías de código, onboarding a base de código nueva |
| **Cuando NO es la elección correcta** | Datos en tiempo real que requieren subscripciones de pago | Automatización de trades (nunca) | Generar código nuevo (usar un agente de generación) |

### Combinaciones habituales

En sistemas reales, estos agentes se combinan:

- **Research + Financial:** el Research Agent investiga el contexto macroeconómico de un sector; el Financial Agent analiza las métricas de los activos de ese sector.
- **Research + Code Review:** el Research Agent busca CVEs y vulnerabilidades conocidas en las dependencias; el Code Review Agent busca esos patrones en el código.
- **Orquestador general → Agente especializado:** un agente de propósito general detecta que la subtarea requiere análisis financiero y delega en el Financial Agent (patrón del Tutorial 01).

---

## 6. Patrones comunes de especialización

Estos tres patrones aparecen en los tres agentes implementados en este tutorial. Aplicarlos de forma consistente mejora la mantenibilidad y reduce los errores.

### 6.1 System prompt especializado

Un system prompt efectivo para un agente especializado tiene cinco componentes:

```
1. IDENTIDAD:   "Eres un agente especializado en X."
2. PROCESO:     Pasos obligatorios numerados que el agente debe seguir.
3. FORMATO:     Estructura exacta del output (secciones, tablas, listas).
4. RESTRICCIONES: Lo que el agente nunca debe hacer (crítico para dominios sensibles).
5. ESTÁNDARES:  Criterios de calidad específicos del dominio.
```

El orden importa. Las restricciones deben estar antes del formato para que el modelo las priorice. Un sistema prompt que empieza con formato y acaba con restricciones produce agentes que priorizan la forma sobre la seguridad.

### 6.2 Herramientas acotadas al dominio

El principio es el de mínimo privilegio aplicado a herramientas:

- Exponer solo las herramientas necesarias para el dominio.
- Cada herramienta debe tener una descripción que indique cuándo usarla y cuándo no.
- Los parámetros deben tener valores por defecto sensatos para el dominio.
- Las herramientas deben validar sus entradas y devolver errores descriptivos en lugar de lanzar excepciones.

```python
# Patrón recomendado para herramientas de agentes especializados
def herramienta_dominio(parametro_obligatorio: str, parametro_opcional: int = 10) -> dict:
    """
    Hace X específico del dominio.
    Devuelve {campo1, campo2} o {error: str} si falla.
    """
    # 1. Validar entradas
    if not parametro_obligatorio:
        return {"error": "parametro_obligatorio no puede estar vacío"}

    # 2. Ejecutar la lógica
    try:
        resultado = _logica_interna(parametro_obligatorio, parametro_opcional)
        return {"campo1": resultado.a, "campo2": resultado.b}
    except Exception as e:
        # 3. Devolver errores descriptivos, nunca lanzar
        return {"error": f"Descripción del fallo: {e}"}
```

### 6.3 Memoria episódica por dominio

Los tres agentes de este tutorial son sin estado: cada llamada empieza desde cero. Para casos de uso con múltiples sesiones, conviene añadir memoria episódica específica al dominio.

El patrón es simple: serializar y deserializar el historial relevante en JSON antes y después de cada sesión.

```python
# episodic_memory.py
# Memoria episódica genérica para agentes especializados.
# Cada dominio define qué considera un "episodio" relevante.

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class MemoriaEpisodica:
    """
    Almacena y recupera episodios relevantes para un agente especializado.
    Un episodio es cualquier unidad de información que el agente quiere recordar
    entre sesiones: un informe completado, una cartera analizada, una revisión entregada.
    """

    def __init__(self, ruta_almacen: str, dominio: str):
        """
        Args:
            ruta_almacen: Directorio donde se guardan los ficheros de memoria.
            dominio: Identificador del dominio (ej: "research", "financial", "code_review").
        """
        self.ruta = Path(ruta_almacen)
        self.ruta.mkdir(parents=True, exist_ok=True)
        self.dominio = dominio
        self.fichero = self.ruta / f"memoria_{dominio}.json"
        self._episodios: list[dict] = self._cargar()

    def _cargar(self) -> list[dict]:
        if self.fichero.exists():
            try:
                return json.loads(self.fichero.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _guardar(self) -> None:
        self.fichero.write_text(
            json.dumps(self._episodios, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def registrar(self, descripcion: str, datos: dict[str, Any], etiquetas: list[str] | None = None) -> None:
        """
        Registra un nuevo episodio en la memoria.

        Args:
            descripcion: Descripción breve del episodio (búsqueda, cartera analizada, PR revisado).
            datos: Datos estructurados del episodio (resultado, contexto, métricas clave).
            etiquetas: Lista de etiquetas para facilitar la recuperación posterior.
        """
        episodio = {
            "id": len(self._episodios) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "dominio": self.dominio,
            "descripcion": descripcion,
            "datos": datos,
            "etiquetas": etiquetas or [],
        }
        self._episodios.append(episodio)
        self._guardar()

    def buscar(self, texto: str | None = None, etiqueta: str | None = None,
               ultimo_n: int | None = None) -> list[dict]:
        """
        Recupera episodios de la memoria.

        Args:
            texto: Filtra episodios cuya descripción contenga este texto.
            etiqueta: Filtra episodios que tengan esta etiqueta.
            ultimo_n: Devuelve solo los N episodios más recientes.
        """
        resultados = self._episodios.copy()

        if texto:
            resultados = [e for e in resultados if texto.lower() in e["descripcion"].lower()]
        if etiqueta:
            resultados = [e for e in resultados if etiqueta in e.get("etiquetas", [])]
        if ultimo_n:
            resultados = resultados[-ultimo_n:]

        return resultados

    def contexto_para_agente(self, ultimo_n: int = 5) -> str:
        """
        Formatea los últimos N episodios como contexto para incluir en el system prompt.
        """
        episodios_recientes = self.buscar(ultimo_n=ultimo_n)
        if not episodios_recientes:
            return "Sin episodios previos en memoria."

        lineas = [f"## Memoria episódica ({self.dominio}) — últimos {len(episodios_recientes)} episodios\n"]
        for ep in episodios_recientes:
            lineas.append(f"- [{ep['timestamp'][:10]}] {ep['descripcion']}")
            if ep.get("etiquetas"):
                lineas.append(f"  Etiquetas: {', '.join(ep['etiquetas'])}")

        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Ejemplo de integración con el Research Agent
# ---------------------------------------------------------------------------

def research_agent_con_memoria(pregunta: str, ruta_memoria: str = "/tmp/agente_memoria") -> str:
    """
    Versión del Research Agent que recuerda los temas investigados previamente.
    Inyecta el contexto de la memoria en el system prompt.
    """
    import anthropic  # reutilizamos el cliente del módulo principal

    memoria = MemoriaEpisodica(ruta_memoria, "research")

    # Inyectar contexto de memoria en el system prompt
    contexto_previo = memoria.contexto_para_agente(ultimo_n=3)
    system_con_memoria = (
        SYSTEM_PROMPT_RESEARCH  # importado del módulo research_agent
        + f"\n\n---\n{contexto_previo}"
    )

    # Ejecutar el agente normalmente (omitido aquí por brevedad)
    # informe = _ejecutar_bucle_agente(system_con_memoria, pregunta)
    informe = f"[Informe para: {pregunta}]"  # placeholder

    # Registrar el episodio al finalizar
    memoria.registrar(
        descripcion=f"Investigación: {pregunta[:80]}",
        datos={"pregunta": pregunta, "longitud_informe": len(informe)},
        etiquetas=["investigacion", "completado"],
    )

    return informe
```

---

**Volver al índice del bloque:** [Bloque 9 — Agentes Avanzados](./README.md)
