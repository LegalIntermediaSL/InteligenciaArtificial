# Text-to-SQL Avanzado: Corrección Automática y Esquemas Complejos

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexfazio/InteligenciaArtificial/blob/main/tutoriales/notebooks/text-to-sql/02-text-to-sql-avanzado.ipynb)

El Text-to-SQL básico falla en ~30% de las consultas complejas. Con self-correction loops, schema-linking y manejo de fechas relativas, llevamos la precisión al 85%+ en esquemas reales.

---

## 1. Self-correction loop

El error más costoso en Text-to-SQL no es generar SQL incorrecto —es no detectarlo. El self-correction loop resuelve esto: si el SQL falla al ejecutarse, Claude recibe el error y tiene hasta 3 intentos de corregirlo.

```
Intento 1: SQL generado → ERROR
    ↓ Claude recibe: pregunta + SQL fallido + mensaje de error
Intento 2: SQL corregido → ERROR
    ↓ Claude recibe: historial completo + nuevo error
Intento 3: SQL corregido → ÉXITO o rendición
```

### Por qué funciona

Claude no generó el SQL incorrecto porque "no sabe SQL" —generó algo plausible pero con un detalle erróneo (nombre de columna, función no disponible en el dialecto, etc.). El mensaje de error de SQLite/PostgreSQL es exactamente la información que necesita para corregirlo.

```python
import anthropic
import sqlite3
import pandas as pd
from typing import Optional

client = anthropic.Anthropic()

def generar_sql_con_correccion(
    pregunta: str,
    esquema: str,
    conn: sqlite3.Connection,
    max_intentos: int = 3
) -> dict:
    """
    Genera SQL con self-correction: hasta max_intentos de corregir errores.
    """
    historial = []
    ultimo_error = None
    
    for intento in range(1, max_intentos + 1):
        # Construir prompt con historial de errores
        if ultimo_error:
            contenido = f"""La consulta anterior falló:
SQL: {historial[-1]['sql']}
Error: {ultimo_error}

Por favor corrige el SQL para la pregunta: {pregunta}"""
        else:
            contenido = pregunta
        
        respuesta = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=f"""Eres un experto en SQL para SQLite. Genera consultas SQL correctas.

Esquema:
{esquema}

Responde SOLO con la consulta SQL, sin markdown.""",
            messages=[
                *[msg for par in historial for msg in [
                    {"role": "user", "content": par["pregunta"]},
                    {"role": "assistant", "content": par["sql"]}
                ]],
                {"role": "user", "content": contenido}
            ]
        )
        
        sql = respuesta.content[0].text.strip()
        if sql.startswith("```"):
            sql = "\n".join(sql.split("\n")[1:-1]).strip()
        
        # Intentar ejecutar
        try:
            df = pd.read_sql_query(sql, conn)
            return {
                "sql": sql,
                "dataframe": df,
                "intentos": intento,
                "error": None
            }
        except Exception as e:
            ultimo_error = str(e)
            historial.append({"pregunta": contenido, "sql": sql})
    
    return {
        "sql": sql,
        "dataframe": None,
        "intentos": max_intentos,
        "error": ultimo_error
    }
```

---

## 2. Schema-linking

En bases de datos con 20+ tablas, pasar el esquema completo al prompt es ineficiente y confuso. Schema-linking identifica qué tablas y columnas son relevantes para cada pregunta antes de generar el SQL.

```python
def schema_linking(pregunta: str, esquema_completo: dict) -> str:
    """
    Identifica las tablas y columnas relevantes para una pregunta.
    
    Args:
        pregunta: Pregunta en lenguaje natural
        esquema_completo: Dict {tabla: [columnas]}
    
    Returns:
        Esquema filtrado con solo las tablas relevantes
    """
    # Serializar esquema para el prompt
    esquema_texto = "\n".join(
        f"{tabla}: {', '.join(cols)}"
        for tabla, cols in esquema_completo.items()
    )
    
    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="""Eres un experto en bases de datos. Dada una pregunta y un esquema completo,
identifica qué tablas y columnas son necesarias para responder la pregunta.
Responde SOLO con los nombres de las tablas necesarias, separados por coma.""",
        messages=[{
            "role": "user",
            "content": f"""Esquema completo:
{esquema_texto}

Pregunta: {pregunta}

¿Qué tablas son necesarias?"""
        }]
    )
    
    tablas_necesarias = [
        t.strip()
        for t in respuesta.content[0].text.split(",")
    ]
    
    # Filtrar el esquema
    esquema_filtrado = {
        tabla: cols
        for tabla, cols in esquema_completo.items()
        if tabla in tablas_necesarias
    }
    
    # Serializar esquema filtrado
    return "\n".join(
        f"{tabla}({', '.join(cols)})"
        for tabla, cols in esquema_filtrado.items()
    )
```

### Cuándo usar schema-linking

| Condición | Recomendación |
|-----------|--------------|
| Menos de 10 tablas | No necesario, pasar todo el esquema |
| 10-30 tablas | Recomendado para reducir tokens |
| Más de 30 tablas | Imprescindible, también considerar embeddings para búsqueda semántica |

---

## 3. Multi-tabla: JOINs desde claves foráneas

La información sobre relaciones entre tablas es crítica para generar JOINs correctos. Al incluir las FK en el esquema, Claude puede inferir cómo unir las tablas.

```python
def extraer_esquema_con_relaciones(conn: sqlite3.Connection) -> str:
    """Extrae esquema incluyendo foreign keys para guiar los JOINs."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablas = [row[0] for row in cursor.fetchall()]
    
    partes = []
    for tabla in tablas:
        cursor.execute(f"PRAGMA table_info({tabla})")
        columnas = {col[1]: col[2] for col in cursor.fetchall()}
        
        cursor.execute(f"PRAGMA foreign_key_list({tabla})")
        fks = cursor.fetchall()
        
        # Formato: tabla(col TIPO, col_fk TIPO → tabla_ref.col_ref)
        cols_desc = []
        for nombre, tipo in columnas.items():
            fk_info = next((fk for fk in fks if fk[3] == nombre), None)
            if fk_info:
                cols_desc.append(f"{nombre} {tipo} → {fk_info[2]}.{fk_info[4]}")
            else:
                cols_desc.append(f"{nombre} {tipo}")
        
        partes.append(f"{tabla}({', '.join(cols_desc)})")
    
    return "\n".join(partes)
```

### Ejemplo de esquema con relaciones

```
clientes(id INTEGER, nombre TEXT, email TEXT, ciudad TEXT)
productos(id INTEGER, nombre TEXT, categoria TEXT, precio REAL)
pedidos(id INTEGER, cliente_id INTEGER → clientes.id, fecha TEXT, total REAL)
detalle_pedidos(id INTEGER, pedido_id INTEGER → pedidos.id, producto_id INTEGER → productos.id, cantidad INTEGER)
```

Con este formato, Claude puede generar automáticamente:

```sql
-- "¿Qué productos compró Ana García?"
SELECT DISTINCT p.nombre
FROM productos p
JOIN detalle_pedidos dp ON p.id = dp.producto_id
JOIN pedidos pe ON dp.pedido_id = pe.id
JOIN clientes c ON pe.cliente_id = c.id
WHERE c.nombre = 'Ana García';
```

---

## 4. Fechas relativas

"El mes pasado", "esta semana", "últimos 30 días" son expresiones que los usuarios usan naturalmente pero que el SQL no entiende. La solución es inyectar la fecha actual en el prompt.

```python
from datetime import date

def agregar_contexto_temporal(esquema: str) -> str:
    """Agrega contexto de fecha actual al esquema para consultas temporales."""
    hoy = date.today().isoformat()
    
    contexto_temporal = f"""
Fecha actual: {hoy} (usar en WHERE para filtros temporales)
Ejemplos de conversión:
- "hoy" → DATE('{hoy}')
- "este mes" → strftime('%Y-%m', fecha) = strftime('%Y-%m', '{hoy}')
- "el mes pasado" → strftime('%Y-%m', fecha) = strftime('%Y-%m', DATE('{hoy}', '-1 month'))
- "últimos 30 días" → fecha >= DATE('{hoy}', '-30 days')
- "este año" → strftime('%Y', fecha) = strftime('%Y', '{hoy}')
"""
    return esquema + "\n" + contexto_temporal
```

### Prueba de fechas relativas

```python
preguntas_temporales = [
    "¿Cuántos pedidos se hicieron el mes pasado?",
    "¿Cuáles son las ventas de los últimos 30 días?",
    "¿Qué clientes se registraron este año?",
    "¿Cuál es el promedio diario de ventas esta semana?",
]

for pregunta in preguntas_temporales:
    print(f"\nPregunta: {pregunta}")
    # El esquema incluye el contexto temporal
    # ...
```

---

## 5. Clase `AdvancedTextToSQL`

```python
class AdvancedTextToSQL:
    """Text-to-SQL avanzado con self-correction, schema-linking y fechas relativas."""
    
    def __init__(self, conn: sqlite3.Connection, max_intentos: int = 3):
        self.client = anthropic.Anthropic()
        self.conn = conn
        self.max_intentos = max_intentos
        self.esquema_completo = self._cargar_esquema_completo()
    
    def _cargar_esquema_completo(self) -> dict:
        """Carga el esquema en formato {tabla: [columnas]}."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas = [row[0] for row in cursor.fetchall()]
        
        esquema = {}
        for tabla in tablas:
            cursor.execute(f"PRAGMA table_info({tabla})")
            esquema[tabla] = [col[1] for col in cursor.fetchall()]
        return esquema
    
    def _construir_esquema_para_query(self, pregunta: str) -> str:
        """Aplica schema-linking y agrega contexto temporal."""
        # Solo usar schema-linking si hay muchas tablas
        if len(self.esquema_completo) > 8:
            esquema = schema_linking(pregunta, self.esquema_completo)
        else:
            esquema = extraer_esquema_con_relaciones(self.conn)
        
        return agregar_contexto_temporal(esquema)
    
    def query(self, pregunta: str) -> dict:
        """Ejecuta una pregunta con self-correction integrado."""
        esquema = self._construir_esquema_para_query(pregunta)
        
        resultado = generar_sql_con_correccion(
            pregunta=pregunta,
            esquema=esquema,
            conn=self.conn,
            max_intentos=self.max_intentos
        )
        
        # Añadir respuesta en lenguaje natural si fue exitoso
        if resultado["dataframe"] is not None and not resultado["dataframe"].empty:
            resultado["respuesta"] = self._interpretar(
                pregunta,
                resultado["sql"],
                resultado["dataframe"]
            )
        elif resultado["dataframe"] is not None:
            resultado["respuesta"] = "La consulta no devolvió resultados."
        
        return resultado
    
    def _interpretar(self, pregunta: str, sql: str, df: pd.DataFrame) -> str:
        """Convierte resultado tabular a lenguaje natural."""
        respuesta = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"Pregunta: {pregunta}\nResultado:\n{df.to_string(index=False)}\n\nResponde en una oración clara en español."
            }]
        )
        return respuesta.content[0].text.strip()
```

---

## 6. Evaluación de precisión

Hay dos métricas estándar para evaluar sistemas Text-to-SQL:

### Execution Accuracy (EA)

Compara si el resultado de ejecutar el SQL generado es igual al resultado esperado. Es la métrica más útil para casos de uso reales.

```python
def execution_accuracy(sistema, dataset_evaluacion: list[dict]) -> float:
    """
    Calcula la Execution Accuracy del sistema.
    
    dataset_evaluacion: lista de {pregunta, sql_esperado, resultado_esperado}
    """
    correctos = 0
    
    for caso in dataset_evaluacion:
        resultado = sistema.query(caso["pregunta"])
        
        if resultado["dataframe"] is not None:
            # Comparar resultados (normalizar para comparación)
            df_generado = resultado["dataframe"].sort_values(
                by=list(resultado["dataframe"].columns)
            ).reset_index(drop=True)
            
            df_esperado = pd.read_sql_query(
                caso["sql_esperado"], sistema.conn
            ).sort_values(
                by=list(df_generado.columns) if list(df_generado.columns) else []
            ).reset_index(drop=True)
            
            if df_generado.equals(df_esperado):
                correctos += 1
    
    return correctos / len(dataset_evaluacion)
```

### Exact Match (EM)

Compara el SQL generado con el SQL esperado. Es más estricta y menos útil —hay muchas formas equivalentes de escribir el mismo SQL.

```python
def exact_match(sql_generado: str, sql_esperado: str) -> bool:
    """Compara SQL normalizado (sin espacios extra, lowercase)."""
    def normalizar(sql: str) -> str:
        return " ".join(sql.lower().split())
    
    return normalizar(sql_generado) == normalizar(sql_esperado)
```

### Comparación de métricas

| Métrica | Ventaja | Desventaja |
|---------|---------|------------|
| Execution Accuracy | Mide resultado real | Requiere ejecutar SQL |
| Exact Match | Rápida de calcular | Demasiado estricta |
| **Recomendación** | **Usar EA como métrica principal** | — |

Estado del arte en benchmarks públicos (Spider, BIRD):
- Modelos GPT-4/Claude Opus: ~85% EA en esquemas vistos
- Modelos más pequeños (Haiku): ~70% EA con few-shot

---

## Resumen

Añadimos tres técnicas que llevan el Text-to-SQL básico al siguiente nivel:

1. **Self-correction**: hasta 3 intentos de corrección con el mensaje de error real
2. **Schema-linking**: solo pasar las tablas relevantes al prompt (esencial con 10+ tablas)
3. **Fechas relativas**: inyectar `CURRENT_DATE` y patrones de conversión en el prompt

La clase `AdvancedTextToSQL` integra todo de forma transparente.

**Siguiente**: [Agente Analista de Datos →](03-agente-analista-sql.md)
