# Agente Analista de Datos: SQL + Gráficas + Respuestas en Lenguaje Natural

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexfazio/InteligenciaArtificial/blob/main/tutoriales/notebooks/text-to-sql/03-agente-analista-sql.ipynb)

Un agente analista va más allá del Text-to-SQL: no solo consulta datos, sino que decide qué analizar, genera visualizaciones y sintetiza insights en lenguaje natural, igual que un analista de datos humano.

---

## 1. Arquitectura del agente

El agente tiene tres herramientas especializadas:

```
Usuario: "¿Cómo han evolucionado las ventas este año?"
         ↓
    Agente (Claude)
    ├── ejecutar_sql(query)      → DataFrame con datos
    ├── generar_grafica(df, tipo) → Imagen PNG en base64
    └── describir_datos(df)      → Estadísticas descriptivas
         ↓
    Respuesta final: insights + gráfica
```

El bucle agéntico funciona así:
1. Claude recibe la pregunta y las herramientas disponibles
2. Decide qué herramienta usar y con qué parámetros
3. El sistema ejecuta la herramienta y devuelve el resultado
4. Claude decide si necesita más herramientas o puede responder
5. Máximo 5 pasos para evitar loops infinitos

---

## 2. Definición de herramientas con schemas

```python
import anthropic
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Sin GUI para entornos servidor
import matplotlib.pyplot as plt
import io
import base64
import json

client = anthropic.Anthropic()

# Schemas de las herramientas para Claude
HERRAMIENTAS = [
    {
        "name": "ejecutar_sql",
        "description": "Ejecuta una consulta SQL en la base de datos y devuelve el resultado como tabla. Usar para obtener datos específicos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Consulta SQL SELECT válida para SQLite"
                },
                "descripcion": {
                    "type": "string",
                    "description": "Descripción breve de qué datos está obteniendo esta consulta"
                }
            },
            "required": ["sql", "descripcion"]
        }
    },
    {
        "name": "generar_grafica",
        "description": "Genera una gráfica (barras, línea o dispersión) a partir de datos. Usar para visualizar tendencias o comparaciones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "datos": {
                    "type": "array",
                    "description": "Lista de diccionarios con los datos a graficar",
                    "items": {"type": "object"}
                },
                "tipo": {
                    "type": "string",
                    "enum": ["barras", "linea", "dispersión", "torta"],
                    "description": "Tipo de gráfica"
                },
                "columna_x": {
                    "type": "string",
                    "description": "Columna para el eje X"
                },
                "columna_y": {
                    "type": "string",
                    "description": "Columna para el eje Y"
                },
                "titulo": {
                    "type": "string",
                    "description": "Título de la gráfica"
                }
            },
            "required": ["datos", "tipo", "columna_x", "columna_y", "titulo"]
        }
    },
    {
        "name": "describir_datos",
        "description": "Calcula estadísticas descriptivas de un conjunto de datos: media, mediana, min, max, distribución. Usar para analizar la forma de los datos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "datos": {
                    "type": "array",
                    "description": "Lista de diccionarios con los datos a describir",
                    "items": {"type": "object"}
                },
                "columna": {
                    "type": "string",
                    "description": "Nombre de la columna numérica a analizar"
                }
            },
            "required": ["datos", "columna"]
        }
    }
]
```

---

## 3. Bucle agéntico

```python
def ejecutar_herramienta(nombre: str, params: dict, conn: sqlite3.Connection) -> str:
    """Dispatcher de herramientas."""
    if nombre == "ejecutar_sql":
        return _tool_ejecutar_sql(params["sql"], conn)
    elif nombre == "generar_grafica":
        return _tool_generar_grafica(params)
    elif nombre == "describir_datos":
        return _tool_describir_datos(params["datos"], params["columna"])
    else:
        return f"Error: herramienta '{nombre}' no reconocida"

def agente_analista(pregunta: str, esquema: str, conn: sqlite3.Connection) -> dict:
    """
    Agente analista con bucle de herramientas, máximo 5 pasos.
    
    Returns:
        dict con: respuesta, grafica_b64, pasos_ejecutados, herramientas_usadas
    """
    mensajes = [{"role": "user", "content": pregunta}]
    pasos = 0
    max_pasos = 5
    grafica_b64 = None
    herramientas_usadas = []
    
    system = f"""Eres un analista de datos experto. Tienes acceso a una base de datos con el siguiente esquema:

{esquema}

Para responder preguntas sobre los datos:
1. Usa ejecutar_sql para obtener los datos necesarios
2. Usa describir_datos si necesitas estadísticas
3. Usa generar_grafica si la visualización ayuda a entender los datos
4. Sintetiza los hallazgos en una respuesta clara en español

Sé eficiente: usa solo las herramientas necesarias."""
    
    while pasos < max_pasos:
        respuesta = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=HERRAMIENTAS,
            messages=mensajes
        )
        
        pasos += 1
        
        # Agregar respuesta del asistente al historial
        mensajes.append({"role": "assistant", "content": respuesta.content})
        
        # Verificar si terminó
        if respuesta.stop_reason == "end_turn":
            texto_final = next(
                (b.text for b in respuesta.content if hasattr(b, "text")),
                "Análisis completado."
            )
            return {
                "respuesta": texto_final,
                "grafica_b64": grafica_b64,
                "pasos_ejecutados": pasos,
                "herramientas_usadas": herramientas_usadas
            }
        
        # Procesar tool_use
        if respuesta.stop_reason == "tool_use":
            resultados_tools = []
            
            for bloque in respuesta.content:
                if bloque.type != "tool_use":
                    continue
                
                nombre_tool = bloque.name
                herramientas_usadas.append(nombre_tool)
                
                resultado = ejecutar_herramienta(nombre_tool, bloque.input, conn)
                
                # Capturar gráfica si se generó
                if nombre_tool == "generar_grafica" and resultado.startswith("data:image"):
                    grafica_b64 = resultado.split(",", 1)[1]
                    resultado = "Gráfica generada exitosamente."
                
                resultados_tools.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": resultado
                })
            
            mensajes.append({"role": "user", "content": resultados_tools})
    
    return {
        "respuesta": "Se alcanzó el límite de pasos sin completar el análisis.",
        "grafica_b64": grafica_b64,
        "pasos_ejecutados": pasos,
        "herramientas_usadas": herramientas_usadas
    }
```

---

## 4. Generación de gráficas con matplotlib

```python
def _tool_ejecutar_sql(sql: str, conn: sqlite3.Connection) -> str:
    """Ejecuta SQL y devuelve resultado como JSON."""
    try:
        # Validación básica de seguridad
        sql_upper = sql.upper().strip()
        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return "Error: solo se permiten consultas SELECT"
        
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "La consulta no devolvió resultados."
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Error SQL: {str(e)}"

def _tool_generar_grafica(params: dict) -> str:
    """Genera una gráfica y la devuelve como data URL en base64."""
    try:
        datos = params["datos"]
        tipo = params["tipo"]
        col_x = params["columna_x"]
        col_y = params["columna_y"]
        titulo = params["titulo"]
        
        if not datos:
            return "Error: no hay datos para graficar"
        
        df = pd.DataFrame(datos)
        
        if col_x not in df.columns or col_y not in df.columns:
            cols_disponibles = ", ".join(df.columns.tolist())
            return f"Error: columnas disponibles son: {cols_disponibles}"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if tipo == "barras":
            ax.bar(df[col_x].astype(str), df[col_y], color="#4A90D9", edgecolor="white")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            plt.xticks(rotation=45, ha="right")
        elif tipo == "linea":
            ax.plot(df[col_x].astype(str), df[col_y], marker="o", color="#4A90D9", linewidth=2)
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            plt.xticks(rotation=45, ha="right")
        elif tipo == "torta":
            ax.pie(df[col_y], labels=df[col_x], autopct="%1.1f%%")
        elif tipo == "dispersión":
            ax.scatter(df[col_x], df[col_y], color="#4A90D9", alpha=0.6)
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
        
        ax.set_title(titulo, fontsize=14, pad=15)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        
        return f"data:image/png;base64,{img_b64}"
    
    except Exception as e:
        plt.close("all")
        return f"Error generando gráfica: {str(e)}"

def _tool_describir_datos(datos: list, columna: str) -> str:
    """Calcula estadísticas descriptivas de una columna."""
    try:
        df = pd.DataFrame(datos)
        if columna not in df.columns:
            return f"Columna '{columna}' no encontrada. Disponibles: {list(df.columns)}"
        
        serie = pd.to_numeric(df[columna], errors="coerce").dropna()
        if serie.empty:
            return f"La columna '{columna}' no tiene valores numéricos."
        
        stats = {
            "columna": columna,
            "count": int(serie.count()),
            "mean": round(float(serie.mean()), 2),
            "median": round(float(serie.median()), 2),
            "std": round(float(serie.std()), 2),
            "min": round(float(serie.min()), 2),
            "max": round(float(serie.max()), 2),
            "q25": round(float(serie.quantile(0.25)), 2),
            "q75": round(float(serie.quantile(0.75)), 2),
        }
        return json.dumps(stats, ensure_ascii=False)
    except Exception as e:
        return f"Error calculando estadísticas: {str(e)}"
```

---

## 5. Respuesta final con insights

El agente sintetiza automáticamente los hallazgos. Para preguntas como "¿Cómo van las ventas?", el agente típicamente:

1. Ejecuta SQL para obtener ventas por período
2. Llama a `describir_datos` para obtener estadísticas
3. Genera una gráfica de línea con la tendencia
4. Redacta una respuesta con los números clave y la tendencia

```python
def mostrar_resultado_agente(resultado: dict):
    """Muestra el resultado del agente de forma estructurada."""
    print("=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)
    print(f"Pasos ejecutados: {resultado['pasos_ejecutados']}")
    print(f"Herramientas usadas: {', '.join(resultado['herramientas_usadas'])}")
    print()
    print("RESPUESTA:")
    print(resultado["respuesta"])
    
    if resultado["grafica_b64"]:
        # En Jupyter, mostrar inline
        from IPython.display import Image, display
        img_bytes = base64.b64decode(resultado["grafica_b64"])
        display(Image(data=img_bytes))
        
        # O guardar a disco
        with open("grafica_analisis.png", "wb") as f:
            f.write(img_bytes)
        print("\n[Gráfica guardada en grafica_analisis.png]")
```

---

## 6. Demo con base de datos de ventas

```python
def crear_bd_ventas(conn: sqlite3.Connection):
    """Crea una BD de ventas con 5 tablas para la demo."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS regiones (
            id INTEGER PRIMARY KEY, nombre TEXT, pais TEXT
        );
        CREATE TABLE IF NOT EXISTS vendedores (
            id INTEGER PRIMARY KEY, nombre TEXT, region_id INTEGER,
            FOREIGN KEY (region_id) REFERENCES regiones(id)
        );
        CREATE TABLE IF NOT EXISTS categorias (
            id INTEGER PRIMARY KEY, nombre TEXT, margen_objetivo REAL
        );
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY, nombre TEXT, categoria_id INTEGER,
            precio REAL, costo REAL,
            FOREIGN KEY (categoria_id) REFERENCES categorias(id)
        );
        CREATE TABLE IF NOT EXISTS ventas (
            id INTEGER PRIMARY KEY, vendedor_id INTEGER, producto_id INTEGER,
            fecha TEXT, cantidad INTEGER, precio_final REAL,
            FOREIGN KEY (vendedor_id) REFERENCES vendedores(id),
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        );
        
        INSERT OR IGNORE INTO regiones VALUES (1,'Norte','España'),(2,'Sur','España'),(3,'Centro','España');
        INSERT OR IGNORE INTO categorias VALUES (1,'Electrónica',0.30),(2,'Periféricos',0.45),(3,'Audio',0.40);
        INSERT OR IGNORE INTO vendedores VALUES (1,'Ana G.',1),(2,'Luis M.',2),(3,'Sara P.',3),(4,'Carlos R.',1);
        INSERT OR IGNORE INTO productos VALUES
            (1,'Laptop Pro',1,1299.99,900.00),(2,'Monitor 4K',1,449.99,310.00),
            (3,'Teclado',2,89.99,45.00),(4,'Mouse',2,39.99,18.00),(5,'Auriculares',3,129.99,75.00);
        INSERT OR IGNORE INTO ventas VALUES
            (1,1,1,'2024-01-15',2,2599.98),(2,2,3,'2024-01-20',5,449.95),
            (3,3,5,'2024-02-01',3,389.97),(4,1,2,'2024-02-14',1,449.99),
            (5,4,4,'2024-03-05',10,399.90),(6,2,1,'2024-03-18',1,1299.99),
            (7,3,3,'2024-04-02',8,719.92),(8,1,5,'2024-04-20',2,259.98),
            (9,4,2,'2024-05-10',2,899.98),(10,2,4,'2024-05-25',15,599.85);
    """)
    conn.commit()

# Demo
conn = sqlite3.connect(":memory:")
crear_bd_ventas(conn)
esquema = extraer_esquema_con_relaciones(conn)

preguntas_demo = [
    "¿Cuál es el vendedor con más ventas totales? Muéstrame una gráfica comparativa.",
    "¿Cómo ha evolucionado el total de ventas mes a mes?",
    "¿Qué categoría de productos tiene mayor margen de ganancia real?",
]

for pregunta in preguntas_demo:
    print(f"\nPREGUNTA: {pregunta}")
    resultado = agente_analista(pregunta, esquema, conn)
    mostrar_resultado_agente(resultado)
```

---

## Resumen

El agente analista combina tres capacidades:

1. **SQL**: obtiene datos precisos de la base de datos
2. **Visualización**: genera gráficas automáticamente cuando mejoran la comprensión
3. **Síntesis**: redacta insights en lenguaje natural con los números clave

El bucle agéntico con máximo 5 pasos garantiza que el agente no entre en loops y siempre devuelva una respuesta. La arquitectura de herramientas es extensible: puedes agregar `exportar_excel`, `enviar_email` o `guardar_en_dashboard`.

**Siguiente**: [Text-to-SQL en Producción →](04-text-to-sql-produccion.md)
