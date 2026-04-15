# 04 — Análisis de datos con IA

> **Bloque:** Casos de uso avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [IA como analista de datos](#1-ia-como-analista-de-datos)
2. [Describir un DataFrame con Claude](#2-describir-un-dataframe-con-claude)
3. [Agente analista con herramientas ejecutables](#3-agente-analista-con-herramientas-ejecutables)
4. [Generación de visualizaciones](#4-generación-de-visualizaciones)
5. [Detección de anomalías con IA](#5-detección-de-anomalías-con-ia)
6. [Caso práctico: analista de ventas](#6-caso-práctico-analista-de-ventas)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. IA como analista de datos

Los LLMs no reemplazan a los analistas, pero aceleran enormemente tareas repetitivas.

| Tarea | ¿Puede la IA? | Limitación |
|---|---|---|
| Describir columnas y distribuciones | Sí | Necesita estadísticas como texto, no los datos crudos |
| Detectar outliers obvios | Sí | Con datos pequeños; no escala a millones de filas |
| Generar código de visualización | Sí, muy bien | Hay que revisar y ejecutar el código generado |
| Interpretar correlaciones | Sí | Puede confundir causalidad con correlación |
| Responder preguntas ad hoc | Sí (con tool use) | Limitado por el tamaño del contexto |
| Modelado estadístico avanzado | Parcialmente | Genera el código, pero no lo valida |

**Requisitos:**

```bash
pip install anthropic pandas matplotlib seaborn python-dotenv
```

---

## 2. Describir un DataFrame con Claude

Cargar un CSV, calcular estadísticas básicas con pandas y enviarlas a Claude para interpretación.

```python
import anthropic
import pandas as pd
import json
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def resumir_dataframe(df: pd.DataFrame) -> dict:
    """
    Genera un resumen estadístico de un DataFrame serializable como JSON.

    Args:
        df: DataFrame de pandas a resumir.

    Returns:
        Diccionario con metadatos, estadísticas y ejemplos.
    """
    resumen = {
        "num_filas": len(df),
        "num_columnas": len(df.columns),
        "columnas": {},
        "primeras_filas": df.head(3).to_dict(orient="records"),
    }

    for col in df.columns:
        info_col = {
            "tipo": str(df[col].dtype),
            "nulos": int(df[col].isna().sum()),
            "unicos": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            info_col.update({
                "min": round(float(stats["min"]), 2),
                "max": round(float(stats["max"]), 2),
                "media": round(float(stats["mean"]), 2),
                "mediana": round(float(df[col].median()), 2),
                "desv_std": round(float(stats["std"]), 2),
            })
        else:
            info_col["valores_frecuentes"] = (
                df[col].value_counts().head(5).to_dict()
            )

        resumen["columnas"][col] = info_col

    return resumen


def interpretar_datos(df: pd.DataFrame, pregunta: str = None) -> str:
    """
    Envía el resumen del DataFrame a Claude y solicita interpretación.

    Args:
        df: DataFrame a analizar.
        pregunta: Pregunta específica sobre los datos (opcional).

    Returns:
        Análisis en texto de Claude.
    """
    resumen = resumir_dataframe(df)
    resumen_json = json.dumps(resumen, ensure_ascii=False, indent=2)

    if pregunta:
        instruccion = f"Responde específicamente a esta pregunta: {pregunta}"
    else:
        instruccion = "Proporciona un análisis exploratorio completo."

    prompt = f"""Eres un analista de datos experto. Analiza el siguiente resumen estadístico de un dataset.

{instruccion}

Resumen del dataset:
```json
{resumen_json}
```

En tu análisis incluye:
1. Descripción general del dataset (qué tipo de datos parece contener)
2. Calidad de los datos (nulos, tipos incorrectos, anomalías)
3. Hallazgos principales y patrones observables
4. Recomendaciones para análisis adicional

Responde en español de forma clara y estructurada."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    return respuesta.content[0].text


# Uso
if __name__ == "__main__":
    # Crear un CSV de ejemplo
    csv_datos = """fecha,producto,categoria,ventas,precio_unitario,unidades,region
2026-01-15,Laptop Pro,Electrónica,12500,1250,10,Norte
2026-01-15,Auriculares BT,Electrónica,800,80,10,Sur
2026-01-16,Silla Ergonómica,Mobiliario,3600,450,8,Centro
2026-01-16,Laptop Pro,Electrónica,2500,1250,2,Sur
2026-01-17,Teclado Mecánico,Electrónica,450,150,3,Norte
2026-01-17,Monitor 4K,Electrónica,5600,700,8,Centro
2026-01-18,Silla Ergonómica,Mobiliario,900,450,2,Norte
2026-01-18,Auriculares BT,Electrónica,1600,80,20,Centro
2026-01-19,Webcam HD,Electrónica,320,80,4,Sur
2026-01-19,Laptop Pro,Electrónica,6250,1250,5,Norte
2026-01-20,Ratón Inalámbrico,Electrónica,200,40,5,Centro
2026-01-20,Monitor 4K,Electrónica,,,3,Sur
"""

    df = pd.read_csv(StringIO(csv_datos))
    df["fecha"] = pd.to_datetime(df["fecha"])

    print("=== ANÁLISIS EXPLORATORIO ===\n")
    analisis = interpretar_datos(df)
    print(analisis)

    print("\n\n=== PREGUNTA ESPECÍFICA ===\n")
    respuesta = interpretar_datos(
        df,
        pregunta="¿Qué producto y región generan más ventas? ¿Hay datos de calidad deficiente?"
    )
    print(respuesta)
```

---

## 3. Agente analista con herramientas ejecutables

Un agente con herramientas Python que Claude puede invocar para responder preguntas sobre los datos.

```python
import anthropic
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")  # Sin interfaz gráfica
import matplotlib.pyplot as plt
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# Dataset global del agente
_df: pd.DataFrame = None


def describe_dataframe() -> str:
    """Devuelve estadísticas descriptivas del DataFrame."""
    if _df is None:
        return "Error: no hay datos cargados."
    info = {
        "filas": len(_df),
        "columnas": list(_df.columns),
        "tipos": _df.dtypes.astype(str).to_dict(),
        "nulos": _df.isna().sum().to_dict(),
        "estadisticas": json.loads(_df.describe().to_json()),
    }
    return json.dumps(info, ensure_ascii=False)


def filter_rows(columna: str, operador: str, valor: str) -> str:
    """
    Filtra filas del DataFrame.
    Operadores: '>', '<', '>=', '<=', '==', '!='
    """
    if _df is None:
        return "Error: no hay datos cargados."
    try:
        if columna not in _df.columns:
            return f"Error: la columna '{columna}' no existe."

        # Intentar convertir el valor al tipo de la columna
        if pd.api.types.is_numeric_dtype(_df[columna]):
            val = float(valor)
        else:
            val = valor

        ops = {">": "__gt__", "<": "__lt__", ">=": "__ge__",
               "<=": "__le__", "==": "__eq__", "!=": "__ne__"}
        if operador not in ops:
            return f"Operador '{operador}' no válido. Usa: {', '.join(ops.keys())}"

        filtrado = _df[getattr(_df[columna], ops[operador])(val)]
        resultado = {
            "filas_resultado": len(filtrado),
            "muestra": filtrado.head(5).to_dict(orient="records"),
        }
        return json.dumps(resultado, ensure_ascii=False, default=str)
    except Exception as e:
        return f"Error al filtrar: {e}"


def plot_histogram(columna: str, bins: int = 10, ruta_salida: str = "histograma.png") -> str:
    """Genera un histograma de una columna numérica."""
    if _df is None:
        return "Error: no hay datos cargados."
    if columna not in _df.columns:
        return f"Error: la columna '{columna}' no existe."
    if not pd.api.types.is_numeric_dtype(_df[columna]):
        return f"Error: '{columna}' no es numérica."

    fig, ax = plt.subplots(figsize=(8, 5))
    _df[columna].dropna().hist(bins=bins, ax=ax, edgecolor="black", color="steelblue")
    ax.set_title(f"Histograma de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=100)
    plt.close()
    return f"Histograma guardado en: {ruta_salida}"


def correlacion(col_a: str, col_b: str) -> str:
    """Calcula la correlación de Pearson entre dos columnas numéricas."""
    if _df is None:
        return "Error: no hay datos cargados."
    for col in [col_a, col_b]:
        if col not in _df.columns:
            return f"Error: la columna '{col}' no existe."
        if not pd.api.types.is_numeric_dtype(_df[col]):
            return f"Error: '{col}' no es numérica."

    corr = _df[[col_a, col_b]].dropna().corr().loc[col_a, col_b]
    interpretacion = (
        "muy fuerte" if abs(corr) > 0.8 else
        "fuerte" if abs(corr) > 0.6 else
        "moderada" if abs(corr) > 0.4 else
        "débil"
    )
    direccion = "positiva" if corr > 0 else "negativa"
    return json.dumps({
        "correlacion_pearson": round(float(corr), 4),
        "interpretacion": f"Correlación {interpretacion} {direccion}",
    })


# Definición de herramientas para la API de Anthropic
HERRAMIENTAS = [
    {
        "name": "describe_dataframe",
        "description": "Obtiene estadísticas descriptivas completas del DataFrame: tipos, nulos, medias, etc.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "filter_rows",
        "description": "Filtra filas del DataFrame según una condición.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columna": {"type": "string", "description": "Nombre de la columna"},
                "operador": {"type": "string", "description": ">, <, >=, <=, ==, !="},
                "valor": {"type": "string", "description": "Valor para comparar"},
            },
            "required": ["columna", "operador", "valor"],
        },
    },
    {
        "name": "plot_histogram",
        "description": "Genera y guarda un histograma de una columna numérica.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columna": {"type": "string"},
                "bins": {"type": "integer", "description": "Número de barras (default 10)"},
                "ruta_salida": {"type": "string", "description": "Nombre del fichero PNG"},
            },
            "required": ["columna"],
        },
    },
    {
        "name": "correlacion",
        "description": "Calcula la correlación entre dos columnas numéricas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "col_a": {"type": "string"},
                "col_b": {"type": "string"},
            },
            "required": ["col_a", "col_b"],
        },
    },
]

MAPA_HERRAMIENTAS = {
    "describe_dataframe": describe_dataframe,
    "filter_rows": filter_rows,
    "plot_histogram": plot_histogram,
    "correlacion": correlacion,
}


def ejecutar_herramienta(nombre: str, inputs: dict) -> str:
    """Ejecuta una herramienta y devuelve el resultado como string."""
    if nombre not in MAPA_HERRAMIENTAS:
        return f"Herramienta '{nombre}' no encontrada."
    return MAPA_HERRAMIENTAS[nombre](**inputs)


def agente_analista(pregunta: str, df: pd.DataFrame, max_iteraciones: int = 10) -> str:
    """
    Agente analista que usa herramientas Python para responder preguntas sobre datos.

    Args:
        pregunta: Pregunta en lenguaje natural sobre el dataset.
        df: DataFrame a analizar.
        max_iteraciones: Límite de rondas de tool_use.

    Returns:
        Respuesta final del agente.
    """
    global _df
    _df = df

    mensajes = [{"role": "user", "content": pregunta}]

    system = """Eres un analista de datos experto. Tienes acceso a herramientas Python
para analizar un DataFrame. Usa las herramientas que necesites para responder
la pregunta con datos concretos. Responde siempre en español."""

    for _ in range(max_iteraciones):
        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            tools=HERRAMIENTAS,
            messages=mensajes,
        )

        mensajes.append({"role": "assistant", "content": respuesta.content})

        if respuesta.stop_reason == "end_turn":
            # Extraer texto final
            for bloque in respuesta.content:
                if hasattr(bloque, "text"):
                    return bloque.text
            return "El agente no generó respuesta textual."

        if respuesta.stop_reason == "tool_use":
            resultados_herramientas = []
            for bloque in respuesta.content:
                if bloque.type == "tool_use":
                    resultado = ejecutar_herramienta(bloque.name, bloque.input)
                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": resultado,
                    })

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return "El agente alcanzó el límite de iteraciones sin dar respuesta."


# Uso
if __name__ == "__main__":
    csv_datos = """producto,ventas,precio,unidades,region
Laptop Pro,12500,1250,10,Norte
Auriculares BT,800,80,10,Sur
Silla Ergonómica,3600,450,8,Centro
Laptop Pro,2500,1250,2,Sur
Teclado Mecánico,450,150,3,Norte
Monitor 4K,5600,700,8,Centro
Silla Ergonómica,900,450,2,Norte
Auriculares BT,1600,80,20,Centro
Webcam HD,320,80,4,Sur
Laptop Pro,6250,1250,5,Norte
"""

    df = pd.read_csv(StringIO(csv_datos))

    preguntas = [
        "¿Cuál es el producto con más ventas totales? Dame un resumen del dataset.",
        "¿Hay correlación entre el precio y las unidades vendidas?",
    ]

    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")
        print("-" * 60)
        respuesta = agente_analista(pregunta, df)
        print(respuesta)
```

---

## 4. Generación de visualizaciones

Claude genera código matplotlib/seaborn en lenguaje natural y se ejecuta con `exec()`.

```python
import anthropic
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def generar_codigo_visualizacion(descripcion_grafica: str, info_dataset: str) -> str:
    """
    Genera código Python de matplotlib/seaborn para una visualización descrita en lenguaje natural.

    Args:
        descripcion_grafica: Descripción de qué visualización se quiere.
        info_dataset: Descripción de las columnas del dataset disponible.

    Returns:
        Código Python ejecutable como string.
    """
    prompt = f"""Genera código Python usando matplotlib y/o seaborn para crear la siguiente visualización.

Dataset disponible (variable `df`):
{info_dataset}

Visualización solicitada:
{descripcion_grafica}

Requisitos del código:
- La variable `df` ya está cargada como DataFrame de pandas
- Usa matplotlib y/o seaborn
- Guarda la figura con: `plt.savefig('output_grafica.png', dpi=150, bbox_inches='tight')`
- Añade `plt.close()` al final
- El código debe ser autocontenido (sin input() ni show())
- Usa español para los títulos y etiquetas

Devuelve ÚNICAMENTE el código Python, sin explicaciones."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    codigo = respuesta.content[0].text.strip()

    # Limpiar bloques de código markdown
    if "```python" in codigo:
        codigo = codigo.split("```python")[1].split("```")[0].strip()
    elif "```" in codigo:
        codigo = codigo.split("```")[1].split("```")[0].strip()

    return codigo


def ejecutar_visualizacion(df: pd.DataFrame, descripcion: str) -> str:
    """
    Genera y ejecuta código de visualización.

    Args:
        df: DataFrame con los datos.
        descripcion: Descripción de la gráfica en lenguaje natural.

    Returns:
        Ruta al fichero PNG generado.
    """
    # Preparar descripción del dataset
    info = f"Columnas: {list(df.columns)}\n"
    for col in df.columns:
        tipo = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            info += f"  - {col} ({tipo}): min={df[col].min():.1f}, max={df[col].max():.1f}\n"
        else:
            info += f"  - {col} ({tipo}): {df[col].nunique()} valores únicos\n"

    codigo = generar_codigo_visualizacion(descripcion, info)
    print(f"Codigo generado:\n{codigo}\n")

    # Ejecutar en un namespace controlado
    namespace = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "np": np,
    }

    exec(codigo, namespace)
    return "output_grafica.png"


# Uso
if __name__ == "__main__":
    csv_datos = """mes,producto,ventas,region
Enero,Laptop,15000,Norte
Enero,Monitor,8000,Sur
Enero,Teclado,2000,Centro
Febrero,Laptop,18000,Norte
Febrero,Monitor,9500,Sur
Febrero,Teclado,2500,Centro
Marzo,Laptop,22000,Norte
Marzo,Monitor,11000,Sur
Marzo,Teclado,3000,Centro
"""

    df = pd.read_csv(StringIO(csv_datos))

    visualizaciones = [
        "Gráfico de barras apiladas mostrando las ventas por mes y producto con colores distintos",
        "Gráfico de líneas con la evolución de ventas por mes para cada producto",
    ]

    for descripcion in visualizaciones:
        print(f"Generando: {descripcion}")
        ruta = ejecutar_visualizacion(df, descripcion)
        print(f"Guardado en: {ruta}\n")
```

---

## 5. Detección de anomalías con IA

Enviar series de datos a Claude para que identifique outliers y explique por qué.

```python
import anthropic
import pandas as pd
import json
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def detectar_anomalias(
    serie: pd.Series,
    nombre_serie: str,
    contexto: str = "",
) -> dict:
    """
    Pide a Claude que identifique anomalías en una serie de datos.

    Args:
        serie: Serie de pandas con los valores a analizar.
        nombre_serie: Nombre descriptivo de la serie.
        contexto: Información adicional sobre qué representan los datos.

    Returns:
        Diccionario con anomalías detectadas y explicaciones.
    """
    # Estadísticas básicas para enriquecer el contexto
    stats = {
        "n": len(serie),
        "media": round(float(serie.mean()), 2),
        "mediana": round(float(serie.median()), 2),
        "desv_std": round(float(serie.std()), 2),
        "min": round(float(serie.min()), 2),
        "max": round(float(serie.max()), 2),
        "q25": round(float(serie.quantile(0.25)), 2),
        "q75": round(float(serie.quantile(0.75)), 2),
    }

    valores_con_indice = [
        {"indice": str(idx), "valor": float(val) if pd.notna(val) else None}
        for idx, val in serie.items()
    ]

    prompt = f"""Eres un experto en análisis de series temporales y detección de anomalías.

Serie: {nombre_serie}
{f"Contexto: {contexto}" if contexto else ""}

Estadísticas:
{json.dumps(stats, indent=2)}

Valores completos:
{json.dumps(valores_con_indice, indent=2)}

Analiza los datos y devuelve ÚNICAMENTE un JSON con esta estructura:
{{
  "anomalias": [
    {{
      "indice": "<valor del índice>",
      "valor": <número>,
      "tipo": "outlier_alto|outlier_bajo|cambio_brusco|patron_inusual",
      "severidad": "alta|media|baja",
      "explicacion": "<por qué es anómalo>",
      "posible_causa": "<hipótesis de la causa>"
    }}
  ],
  "patrones_globales": "<observaciones generales sobre la serie>",
  "recomendaciones": ["<acción 1>", "<acción 2>"]
}}"""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    texto = respuesta.content[0].text.strip()
    if "```json" in texto:
        texto = texto.split("```json")[1].split("```")[0].strip()
    elif "```" in texto:
        texto = texto.split("```")[1].split("```")[0].strip()

    return json.loads(texto)


# Uso
if __name__ == "__main__":
    # Serie de ventas diarias con algunas anomalías intencionadas
    fechas = pd.date_range("2026-01-01", periods=30, freq="D")
    ventas = [
        1200, 1150, 1300, 1250, 1180, 300,   # Caída brusca el día 6
        1220, 1280, 1190, 1240, 1210, 1300,
        1260, 1190, 1220, 5800,              # Pico extremo el día 16
        1240, 1200, 1280, 1150, 1220, 1190,
        1260, 1300, 1220, 1250, 1180, 1240,
        1200, 1190,
    ]

    serie = pd.Series(ventas, index=fechas, name="ventas_diarias")

    resultado = detectar_anomalias(
        serie,
        "Ventas diarias en EUR",
        contexto="Tienda online de electrónica. Los fines de semana suele haber más ventas."
    )

    print("=== DETECCIÓN DE ANOMALÍAS ===\n")
    print(f"Anomalías detectadas: {len(resultado['anomalias'])}\n")

    for anomalia in resultado["anomalias"]:
        print(f"[{anomalia['severidad'].upper()}] {anomalia['indice']} — "
              f"Valor: {anomalia['valor']}")
        print(f"  Tipo: {anomalia['tipo']}")
        print(f"  Por qué: {anomalia['explicacion']}")
        print(f"  Causa posible: {anomalia['posible_causa']}\n")

    print(f"Patrones globales:\n{resultado['patrones_globales']}\n")
    print("Recomendaciones:")
    for rec in resultado["recomendaciones"]:
        print(f"  - {rec}")
```

---

## 6. Caso práctico: analista de ventas

CSV de ventas mensuales con un agente que responde preguntas, genera gráficas y detecta tendencias.

```python
import anthropic
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

# Dataset de ventas mensuales
CSV_VENTAS = """mes,año,producto,categoria,ventas_eur,unidades,region,comercial
Enero,2025,Laptop Pro,Electrónica,45000,36,Norte,García
Enero,2025,Monitor 4K,Electrónica,28000,40,Norte,García
Enero,2025,Laptop Pro,Electrónica,38000,30,Sur,Martínez
Enero,2025,Auriculares BT,Electrónica,12000,150,Centro,López
Febrero,2025,Laptop Pro,Electrónica,52000,41,Norte,García
Febrero,2025,Monitor 4K,Electrónica,31500,45,Sur,Martínez
Febrero,2025,Auriculares BT,Electrónica,9600,120,Norte,García
Febrero,2025,Teclado Mec.,Accesorios,5400,36,Centro,López
Marzo,2025,Laptop Pro,Electrónica,61000,48,Norte,García
Marzo,2025,Monitor 4K,Electrónica,35000,50,Sur,Martínez
Marzo,2025,Auriculares BT,Electrónica,15200,190,Centro,López
Marzo,2025,Laptop Pro,Electrónica,42000,33,Centro,López
Abril,2025,Laptop Pro,Electrónica,48000,38,Norte,García
Abril,2025,Monitor 4K,Electrónica,29400,42,Norte,García
Abril,2025,Auriculares BT,Electrónica,11200,140,Sur,Martínez
Abril,2025,Teclado Mec.,Accesorios,7200,48,Centro,López
Mayo,2025,Laptop Pro,Electrónica,67000,53,Norte,García
Mayo,2025,Monitor 4K,Electrónica,42000,60,Sur,Martínez
Mayo,2025,Auriculares BT,Electrónica,17600,220,Centro,López
Mayo,2025,Webcam HD,Accesorios,4800,60,Norte,García
Junio,2025,Laptop Pro,Electrónica,72000,57,Norte,García
Junio,2025,Monitor 4K,Electrónica,45500,65,Sur,Martínez
Junio,2025,Auriculares BT,Electrónica,19200,240,Centro,López
Junio,2025,Teclado Mec.,Accesorios,8100,54,Norte,García
"""


class AnalistaVentas:
    """Agente analista especializado en datos de ventas."""

    def __init__(self, csv_datos: str):
        self.df = pd.read_csv(StringIO(csv_datos))
        self.df["periodo"] = self.df["mes"] + " " + self.df["año"].astype(str)
        self._df_global = self.df.copy()
        Path("graficas").mkdir(exist_ok=True)

    def _describe_dataframe(self) -> str:
        resumen = {
            "filas": len(self.df),
            "columnas": list(self.df.columns),
            "ventas_totales_eur": float(self.df["ventas_eur"].sum()),
            "periodo": f"{self.df['mes'].iloc[0]} {self.df['año'].iloc[0]} — "
                       f"{self.df['mes'].iloc[-1]} {self.df['año'].iloc[-1]}",
            "productos": list(self.df["producto"].unique()),
            "regiones": list(self.df["region"].unique()),
            "ventas_por_producto": self.df.groupby("producto")["ventas_eur"].sum().to_dict(),
            "ventas_por_region": self.df.groupby("region")["ventas_eur"].sum().to_dict(),
            "ventas_por_mes": self.df.groupby("mes")["ventas_eur"].sum().to_dict(),
        }
        return json.dumps(resumen, ensure_ascii=False, default=float)

    def _top_productos(self, n: int = 5) -> str:
        top = (
            self.df.groupby("producto")["ventas_eur"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return json.dumps(top.to_dict(), ensure_ascii=False, default=float)

    def _tendencia_mensual(self, producto: str = None) -> str:
        if producto:
            filtrado = self.df[self.df["producto"] == producto]
        else:
            filtrado = self.df
        tendencia = filtrado.groupby("mes")["ventas_eur"].sum().to_dict()
        return json.dumps(tendencia, ensure_ascii=False, default=float)

    def _grafica_ventas_mensuales(self, ruta: str = "graficas/ventas_mensuales.png") -> str:
        orden_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio"]
        ventas_mes = (
            self.df.groupby("mes")["ventas_eur"]
            .sum()
            .reindex([m for m in orden_meses if m in self.df["mes"].unique()])
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ventas_mes.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title("Ventas totales mensuales (EUR)")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Ventas (EUR)")
        ax.tick_params(axis="x", rotation=45)

        for i, v in enumerate(ventas_mes):
            ax.text(i, v + 1000, f"{v:,.0f}", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        plt.close()
        return f"Gráfica guardada en: {ruta}"

    def _grafica_por_producto(self, ruta: str = "graficas/ventas_producto.png") -> str:
        ventas_prod = self.df.groupby("producto")["ventas_eur"].sum().sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        colores = sns.color_palette("husl", len(ventas_prod))
        ventas_prod.plot(kind="barh", ax=ax, color=colores)
        ax.set_title("Ventas totales por producto (EUR)")
        ax.set_xlabel("Ventas (EUR)")
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        plt.close()
        return f"Gráfica guardada en: {ruta}"

    HERRAMIENTAS_VENTAS = [
        {
            "name": "describe_dataframe",
            "description": "Resumen completo del dataset de ventas con totales por producto, región y mes.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "top_productos",
            "description": "Productos con más ventas totales.",
            "input_schema": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Cuántos top productos (default 5)"}},
                "required": [],
            },
        },
        {
            "name": "tendencia_mensual",
            "description": "Evolución de ventas mes a mes. Puede filtrarse por producto.",
            "input_schema": {
                "type": "object",
                "properties": {"producto": {"type": "string", "description": "Nombre del producto (opcional)"}},
                "required": [],
            },
        },
        {
            "name": "grafica_ventas_mensuales",
            "description": "Genera una gráfica de barras con ventas mensuales totales.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "grafica_por_producto",
            "description": "Genera una gráfica de barras horizontales con ventas por producto.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ]

    def _ejecutar_herramienta(self, nombre: str, inputs: dict) -> str:
        mapa = {
            "describe_dataframe": lambda: self._describe_dataframe(),
            "top_productos": lambda: self._top_productos(**inputs),
            "tendencia_mensual": lambda: self._tendencia_mensual(**inputs),
            "grafica_ventas_mensuales": lambda: self._grafica_ventas_mensuales(),
            "grafica_por_producto": lambda: self._grafica_por_producto(),
        }
        if nombre not in mapa:
            return f"Herramienta '{nombre}' no encontrada."
        return mapa[nombre]()

    def preguntar(self, pregunta: str, max_iter: int = 8) -> str:
        """Responde una pregunta sobre los datos de ventas."""
        mensajes = [{"role": "user", "content": pregunta}]
        system = (
            "Eres un analista de ventas experto. Usa las herramientas disponibles "
            "para responder con datos concretos. Responde siempre en español y "
            "con un análisis claro y accionable."
        )

        for _ in range(max_iter):
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system,
                tools=self.HERRAMIENTAS_VENTAS,
                messages=mensajes,
            )

            mensajes.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "end_turn":
                for bloque in resp.content:
                    if hasattr(bloque, "text"):
                        return bloque.text
                return "Sin respuesta textual."

            if resp.stop_reason == "tool_use":
                resultados = []
                for bloque in resp.content:
                    if bloque.type == "tool_use":
                        resultado = self._ejecutar_herramienta(bloque.name, bloque.input)
                        resultados.append({
                            "type": "tool_result",
                            "tool_use_id": bloque.id,
                            "content": resultado,
                        })
                mensajes.append({"role": "user", "content": resultados})

        return "Límite de iteraciones alcanzado."


# Ejecución del caso práctico
if __name__ == "__main__":
    analista = AnalistaVentas(CSV_VENTAS)

    preguntas = [
        "¿Cuál es el producto estrella y cómo han evolucionado sus ventas mes a mes?",
        "Genera una gráfica de ventas mensuales y dime si hay tendencia creciente.",
        "¿Qué región tiene mejor rendimiento? ¿Hay algún producto que esté bajando?",
    ]

    for pregunta in preguntas:
        print(f"\n{'='*60}")
        print(f"Pregunta: {pregunta}")
        print(f"{'='*60}")
        respuesta = analista.preguntar(pregunta)
        print(respuesta)
```

---

## 7. Extensiones sugeridas

| Extensión | Descripción | Tecnología |
|---|---|---|
| **Code interpreter** | Ejecutar código generado de forma segura | Docker sandbox |
| **Consultas en lenguaje natural a SQL** | Traducir preguntas a SQL automáticamente | Text-to-SQL con Claude |
| **Dashboard interactivo** | Interfaz web para el agente analista | Streamlit |
| **Exportar informes** | Generar PDF con análisis y gráficas | `reportlab`, `fpdf2` |
| **Alertas automáticas** | Detectar anomalías en tiempo real | Cron job + Claude API |
| **Análisis multidataset** | Comparar y combinar varios CSVs | Agente con múltiples DataFrames |

---

**Fin del bloque**
