# Proyecto 4: Asistente de análisis de datos

## Qué vamos a construir

Un asistente conversacional que analiza datasets CSV/Excel usando tool use de Claude.
El usuario describe en lenguaje natural lo que quiere ver, y el asistente genera
las consultas Pandas, las ejecuta de forma segura y devuelve los resultados con
interpretación en lenguaje natural.

```
ARQUITECTURA
────────────────────────────────────────────────────
Usuario (lenguaje natural) → FastAPI
                    ↓
             Claude + Tool Use
             ┌────────────────┐
             │ herramientas:  │
             │ - cargar_csv   │
             │ - consulta_sql │
             │ - describir_df │
             │ - calcular_kpi │
             │ - detectar_anomalias │
             └────────────────┘
                    ↓
             Ejecución segura (sandbox)
                    ↓
             Interpretación con Claude
                    ↓
             Respuesta + gráfico (opcional)
```

**Bloques que combina:** 5 (casos de uso), 9 (agentes), 10 (tool use), 16 (MLOps).

---

## Herramientas disponibles para Claude

```python
import anthropic
import pandas as pd
import json
import io
import re
from datetime import datetime

client = anthropic.Anthropic()

# Dataset cargado en memoria (en producción: por sesión/usuario)
DATASETS: dict = {}

HERRAMIENTAS = [
    {
        "name": "cargar_dataset",
        "description": "Carga un CSV desde texto o ruta y lo registra con un nombre",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre para identificar el dataset"},
                "csv_texto": {"type": "string", "description": "Contenido CSV como texto"}
            },
            "required": ["nombre", "csv_texto"]
        }
    },
    {
        "name": "describir_dataset",
        "description": "Muestra estadísticas básicas del dataset: shape, columnas, tipos, nulos, describe()",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del dataset a describir"}
            },
            "required": ["nombre"]
        }
    },
    {
        "name": "ejecutar_pandas",
        "description": "Ejecuta código Pandas sobre el dataset y devuelve el resultado. Usa 'df' para referirte al dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string", "description": "Nombre del dataset"},
                "codigo": {"type": "string", "description": "Código Pandas a ejecutar. Resultado en variable 'resultado'."}
            },
            "required": ["nombre", "codigo"]
        }
    },
    {
        "name": "calcular_kpis",
        "description": "Calcula KPIs comunes: suma, media, mediana, % de cambio, top-N, distribución",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string"},
                "columna": {"type": "string", "description": "Columna numérica a analizar"},
                "kpis": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["suma", "media", "mediana", "max", "min", "std", "top10", "distribucion"]},
                    "description": "Lista de KPIs a calcular"
                },
                "agrupar_por": {"type": "string", "description": "Columna por la que agrupar (opcional)"}
            },
            "required": ["nombre", "columna", "kpis"]
        }
    },
    {
        "name": "detectar_anomalias",
        "description": "Detecta valores atípicos (outliers) en una columna numérica usando IQR",
        "input_schema": {
            "type": "object",
            "properties": {
                "nombre": {"type": "string"},
                "columna": {"type": "string"}
            },
            "required": ["nombre", "columna"]
        }
    }
]
```

---

## Ejecución segura de código Pandas

```python
def ejecutar_codigo_seguro(df: pd.DataFrame, codigo: str) -> dict:
    """
    Ejecuta código Pandas en un entorno restringido.
    Bloquea operaciones peligrosas: filesystem, network, subprocess.
    """
    # Bloquear operaciones peligrosas
    PATRONES_PROHIBIDOS = [
        r'\bopen\b', r'\bos\b', r'\bsubprocess\b', r'\beval\b',
        r'\bexec\b', r'\b__import__\b', r'\brequests\b', r'\burllib\b'
    ]
    for patron in PATRONES_PROHIBIDOS:
        if re.search(patron, codigo):
            return {"error": f"Operación no permitida detectada: {patron}"}
    
    # Entorno de ejecución limitado
    entorno_seguro = {
        "df": df.copy(),
        "pd": pd,
        "resultado": None,
        "__builtins__": {
            "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "sum": sum, "min": min, "max": max, "round": round,
            "print": print, "sorted": sorted, "enumerate": enumerate
        }
    }
    
    try:
        exec(codigo, entorno_seguro)
        resultado = entorno_seguro.get("resultado")
        
        # Convertir resultado a formato serializable
        if isinstance(resultado, pd.DataFrame):
            return {"tipo": "dataframe", "datos": resultado.head(20).to_dict(orient="records"), "filas_total": len(resultado)}
        elif isinstance(resultado, pd.Series):
            return {"tipo": "serie", "datos": resultado.to_dict()}
        elif resultado is not None:
            return {"tipo": "valor", "datos": resultado}
        else:
            return {"tipo": "ejecutado", "mensaje": "Código ejecutado sin valor de retorno"}
    except Exception as e:
        return {"error": str(e)}


def ejecutar_herramienta(nombre: str, params: dict) -> str:
    """Dispatcher de herramientas para el bucle agéntico."""
    
    if nombre == "cargar_dataset":
        try:
            df = pd.read_csv(io.StringIO(params["csv_texto"]))
            DATASETS[params["nombre"]] = df
            return json.dumps({
                "ok": True,
                "filas": len(df),
                "columnas": list(df.columns),
                "tipos": df.dtypes.astype(str).to_dict()
            })
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    elif nombre == "describir_dataset":
        nombre_ds = params["nombre"]
        if nombre_ds not in DATASETS:
            return json.dumps({"error": f"Dataset '{nombre_ds}' no encontrado"})
        df = DATASETS[nombre_ds]
        return json.dumps({
            "shape": df.shape,
            "columnas": list(df.columns),
            "tipos": df.dtypes.astype(str).to_dict(),
            "nulos": df.isnull().sum().to_dict(),
            "estadisticas": df.describe().to_dict()
        }, default=str)
    
    elif nombre == "ejecutar_pandas":
        nombre_ds = params["nombre"]
        if nombre_ds not in DATASETS:
            return json.dumps({"error": f"Dataset '{nombre_ds}' no encontrado"})
        resultado = ejecutar_codigo_seguro(DATASETS[nombre_ds], params["codigo"])
        return json.dumps(resultado, default=str)
    
    elif nombre == "calcular_kpis":
        nombre_ds = params["nombre"]
        if nombre_ds not in DATASETS:
            return json.dumps({"error": f"Dataset '{nombre_ds}' no encontrado"})
        df = DATASETS[nombre_ds]
        col = params["columna"]
        agrupar = params.get("agrupar_por")
        
        if col not in df.columns:
            return json.dumps({"error": f"Columna '{col}' no existe"})
        
        serie = df.groupby(agrupar)[col] if agrupar and agrupar in df.columns else df[col]
        
        resultado = {}
        for kpi in params.get("kpis", []):
            if kpi == "suma": resultado["suma"] = float(serie.sum()) if not agrupar else serie.sum().to_dict()
            elif kpi == "media": resultado["media"] = float(serie.mean()) if not agrupar else serie.mean().to_dict()
            elif kpi == "mediana": resultado["mediana"] = float(serie.median()) if not agrupar else serie.median().to_dict()
            elif kpi == "max": resultado["max"] = float(serie.max()) if not agrupar else serie.max().to_dict()
            elif kpi == "min": resultado["min"] = float(serie.min()) if not agrupar else serie.min().to_dict()
            elif kpi == "std": resultado["std"] = float(serie.std()) if not agrupar else serie.std().to_dict()
            elif kpi == "top10": resultado["top10"] = df.nlargest(10, col)[col].to_dict() if not agrupar else {}
            elif kpi == "distribucion": resultado["distribucion"] = df[col].value_counts().head(10).to_dict()
        
        return json.dumps(resultado, default=str)
    
    elif nombre == "detectar_anomalias":
        nombre_ds = params["nombre"]
        if nombre_ds not in DATASETS:
            return json.dumps({"error": f"Dataset '{nombre_ds}' no encontrado"})
        df = DATASETS[nombre_ds]
        col = params["columna"]
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        
        return json.dumps({
            "total_outliers": len(outliers),
            "pct_outliers": round(len(outliers) / len(df) * 100, 2),
            "limite_inferior": round(float(Q1 - 1.5 * IQR), 4),
            "limite_superior": round(float(Q3 + 1.5 * IQR), 4),
            "ejemplos": outliers[col].head(5).tolist()
        })
    
    return json.dumps({"error": f"Herramienta '{nombre}' no implementada"})
```

---

## Bucle agéntico (agente.py)

```python
def analizar_con_agente(pregunta: str, max_iteraciones: int = 6) -> dict:
    """
    Bucle agéntico: Claude razona, llama herramientas, interpreta resultados.
    Equivalente al patrón ReAct del Bloque 9 pero con tool_use nativo de Anthropic.
    """
    mensajes = [{"role": "user", "content": pregunta}]
    iteraciones = 0
    herramientas_usadas = []
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system="""Eres un analista de datos experto. El usuario te pide análisis en lenguaje natural.
Usa las herramientas disponibles para obtener los datos y luego interpreta los resultados
de forma clara y accionable. Siempre:
1. Carga el dataset si es necesario
2. Explora su estructura antes de analizar
3. Presenta resultados con contexto e interpretación""",
            tools=HERRAMIENTAS,
            messages=mensajes
        )
        
        # Añadir respuesta del asistente al historial
        mensajes.append({"role": "assistant", "content": response.content})
        
        # ¿Terminó?
        if response.stop_reason == "end_turn":
            texto_final = next((b.text for b in response.content if hasattr(b, "text")), "")
            return {
                "respuesta": texto_final,
                "iteraciones": iteraciones,
                "herramientas_usadas": herramientas_usadas
            }
        
        # Procesar llamadas a herramientas
        if response.stop_reason == "tool_use":
            resultados_herramientas = []
            
            for bloque in response.content:
                if bloque.type == "tool_use":
                    herramientas_usadas.append(bloque.name)
                    print(f"  → {bloque.name}({list(bloque.input.keys())})")
                    
                    resultado = ejecutar_herramienta(bloque.name, bloque.input)
                    resultados_herramientas.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": resultado
                    })
            
            mensajes.append({"role": "user", "content": resultados_herramientas})
    
    return {
        "respuesta": "Se alcanzó el límite de iteraciones.",
        "iteraciones": iteraciones,
        "herramientas_usadas": herramientas_usadas
    }
```

---

## API principal

```python
from fastapi import FastAPI
from pydantic import BaseModel
from agente import analizar_con_agente, DATASETS
import pandas as pd
import io

app = FastAPI(title="Asistente de Análisis de Datos", version="1.0")

class Pregunta(BaseModel):
    pregunta: str

class CargaCSV(BaseModel):
    nombre: str
    csv_contenido: str

@app.post("/cargar-csv")
async def cargar_csv(datos: CargaCSV):
    df = pd.read_csv(io.StringIO(datos.csv_contenido))
    DATASETS[datos.nombre] = df
    return {"ok": True, "filas": len(df), "columnas": list(df.columns)}

@app.post("/analizar")
async def analizar(entrada: Pregunta):
    resultado = analizar_con_agente(entrada.pregunta)
    return resultado

@app.get("/datasets")
async def listar_datasets():
    return {
        nombre: {"filas": len(df), "columnas": list(df.columns)}
        for nombre, df in DATASETS.items()
    }
```

---

## Ejemplo de sesión

```bash
# 1. Cargar dataset de ventas
curl -X POST http://localhost:8000/cargar-csv \
  -H "Content-Type: application/json" \
  -d '{
    "nombre": "ventas",
    "csv_contenido": "mes,producto,importe,region\nEnero,A,1200,Norte\nEnero,B,800,Sur\nFebrero,A,1500,Norte\nFebrero,B,600,Sur\nMarzo,A,900,Norte"
  }'

# 2. Análisis en lenguaje natural
curl -X POST http://localhost:8000/analizar \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Qué producto vende más y en qué región? ¿Hay alguna tendencia preocupante?"}'

# Respuesta esperada de Claude:
# "El producto A domina las ventas, especialmente en la región Norte donde concentra
# el 75% de los ingresos. El punto preocupante es que el producto A mostró una caída
# del 40% en marzo respecto a febrero (de 1.500€ a 900€), lo que podría indicar
# saturación o competencia. Recomiendo investigar las causas de esta caída..."
```

---

## Recursos

- [Notebook interactivo](../notebooks/proyectos-integradores/04-asistente-datos.ipynb)
- [Tool Use — Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Pandas — documentación oficial](https://pandas.pydata.org/docs/)
