# Text-to-SQL: Traducir Lenguaje Natural a Consultas SQL

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexfazio/InteligenciaArtificial/blob/main/tutoriales/notebooks/text-to-sql/01-text-to-sql-basico.ipynb)

Los usuarios necesitan información de bases de datos, pero no saben SQL. Claude puede cerrar esa brecha traduciendo preguntas en lenguaje natural a consultas SQL correctas y seguras.

---

## 1. El problema: datos valiosos, acceso limitado

Las bases de datos relacionales almacenan el 80% de los datos empresariales: ventas, clientes, inventario, operaciones. El problema es el acceso: solo los analistas e ingenieros de datos saben SQL. El resto de la organización —gerentes, vendedores, operadores— no puede consultar esa información directamente.

Las soluciones tradicionales son costosas:
- Contratar analistas de datos para responder cada pregunta
- Construir dashboards predefinidos que no cubren todas las necesidades
- Entrenar a todos en SQL (costoso, poco realista)

**Text-to-SQL** es la solución: un sistema que recibe una pregunta en español, genera el SQL correspondiente, lo ejecuta y devuelve la respuesta en lenguaje natural.

---

## 2. Arquitectura básica

El flujo tiene cinco pasos:

```
Pregunta (NL) → Claude → SQL → Ejecutar en BD → Respuesta (NL)
```

1. **Entrada**: el usuario escribe una pregunta en español
2. **Generación**: Claude recibe el esquema de la BD y la pregunta, genera SQL
3. **Validación**: el sistema verifica que el SQL es seguro antes de ejecutar
4. **Ejecución**: el SQL se ejecuta contra la base de datos real
5. **Respuesta**: Claude convierte el resultado tabular en lenguaje natural

```python
import anthropic
import sqlite3
import pandas as pd

client = anthropic.Anthropic()

def texto_a_sql(pregunta: str, esquema: str) -> str:
    """Genera SQL a partir de una pregunta en lenguaje natural."""
    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=f"""Eres un experto en SQL. Convierte preguntas en lenguaje natural a SQL.
        
Esquema de la base de datos:
{esquema}

Reglas:
- Responde SOLO con la consulta SQL, sin explicaciones
- Usa nombres de columna exactos del esquema
- Limita resultados con LIMIT 100 por defecto""",
        messages=[{"role": "user", "content": pregunta}]
    )
    return respuesta.content[0].text.strip()
```

---

## 3. Inyección de esquema

El esquema es la información que Claude necesita para generar SQL correcto. Hay que pasarlo de forma compacta para no desperdiciar tokens.

### Formato compacto de esquema

En lugar de pasar el DDL completo, usamos un formato resumido:

```python
def extraer_esquema_compacto(conn: sqlite3.Connection) -> str:
    """Extrae el esquema de SQLite en formato compacto."""
    cursor = conn.cursor()
    
    # Obtener todas las tablas
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tablas = [row[0] for row in cursor.fetchall()]
    
    esquema_partes = []
    for tabla in tablas:
        # Obtener información de columnas
        cursor.execute(f"PRAGMA table_info({tabla})")
        columnas = cursor.fetchall()
        
        # Formato: tabla(col1 TIPO, col2 TIPO, ...)
        cols = ", ".join(f"{col[1]} {col[2]}" for col in columnas)
        esquema_partes.append(f"{tabla}({cols})")
    
    return "\n".join(esquema_partes)
```

### Ejemplo de esquema compacto

```
clientes(id INTEGER, nombre TEXT, email TEXT, ciudad TEXT, fecha_registro TEXT)
productos(id INTEGER, nombre TEXT, categoria TEXT, precio REAL, stock INTEGER)
pedidos(id INTEGER, cliente_id INTEGER, fecha TEXT, total REAL, estado TEXT)
detalle_pedidos(id INTEGER, pedido_id INTEGER, producto_id INTEGER, cantidad INTEGER, precio_unitario REAL)
```

Este formato ocupa ~4 líneas en lugar de 40+ líneas de DDL completo, ahorrando tokens y manteniendo toda la información relevante.

---

## 4. Clase `TextToSQL`

```python
import anthropic
import sqlite3
import pandas as pd
from typing import Optional

class TextToSQL:
    """Sistema básico de Text-to-SQL con SQLite."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.client = anthropic.Anthropic()
        self.conn = sqlite3.connect(db_path)
        self.esquema = self._cargar_esquema()
        self._crear_datos_demo()
    
    def _crear_datos_demo(self):
        """Crea tablas y datos de demostración."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS clientes (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                email TEXT,
                ciudad TEXT,
                fecha_registro TEXT
            );
            CREATE TABLE IF NOT EXISTS productos (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                categoria TEXT,
                precio REAL,
                stock INTEGER
            );
            CREATE TABLE IF NOT EXISTS pedidos (
                id INTEGER PRIMARY KEY,
                cliente_id INTEGER,
                fecha TEXT,
                total REAL,
                estado TEXT,
                FOREIGN KEY (cliente_id) REFERENCES clientes(id)
            );
            INSERT OR IGNORE INTO clientes VALUES
                (1,'Ana García','ana@mail.com','Madrid','2024-01-15'),
                (2,'Luis Martínez','luis@mail.com','Barcelona','2024-02-20'),
                (3,'María López','maria@mail.com','Valencia','2024-03-10'),
                (4,'Carlos Ruiz','carlos@mail.com','Madrid','2024-04-05');
            INSERT OR IGNORE INTO productos VALUES
                (1,'Laptop Pro','Electrónica',1299.99,15),
                (2,'Monitor 4K','Electrónica',449.99,30),
                (3,'Teclado Mecánico','Periféricos',89.99,50),
                (4,'Mouse Inalámbrico','Periféricos',39.99,75),
                (5,'Auriculares BT','Audio',129.99,20);
            INSERT OR IGNORE INTO pedidos VALUES
                (1,1,'2024-05-01',1299.99,'completado'),
                (2,2,'2024-05-03',539.98,'completado'),
                (3,1,'2024-05-10',89.99,'completado'),
                (4,3,'2024-05-15',169.98,'pendiente'),
                (5,4,'2024-05-20',449.99,'enviado');
        """)
        self.conn.commit()
        self.esquema = self._cargar_esquema()
    
    def _cargar_esquema(self) -> str:
        """Extrae el esquema en formato compacto."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tablas = [row[0] for row in cursor.fetchall()]
        
        if not tablas:
            return "(sin tablas)"
        
        partes = []
        for tabla in tablas:
            cursor.execute(f"PRAGMA table_info({tabla})")
            columnas = cursor.fetchall()
            cols = ", ".join(f"{c[1]} {c[2]}" for c in columnas)
            partes.append(f"{tabla}({cols})")
        return "\n".join(partes)
    
    def _generar_sql(self, pregunta: str) -> str:
        """Llama a Claude para generar SQL."""
        few_shot = """
Ejemplos:
P: ¿Cuántos clientes hay en total?
SQL: SELECT COUNT(*) AS total_clientes FROM clientes;

P: ¿Cuáles son los 3 productos más caros?
SQL: SELECT nombre, precio FROM productos ORDER BY precio DESC LIMIT 3;

P: ¿Cuál es el total de ventas completadas?
SQL: SELECT SUM(total) AS ventas_completadas FROM pedidos WHERE estado = 'completado';
"""
        respuesta = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=f"""Eres un experto en SQL. Convierte preguntas a SQL válido para SQLite.

Esquema:
{self.esquema}

{few_shot}

Responde SOLO con la consulta SQL, sin markdown, sin explicaciones.""",
            messages=[{"role": "user", "content": pregunta}]
        )
        sql = respuesta.content[0].text.strip()
        # Limpiar posibles bloques markdown
        if sql.startswith("```"):
            sql = "\n".join(sql.split("\n")[1:-1])
        return sql.strip()
    
    def _interpretar_resultado(self, pregunta: str, sql: str, df: pd.DataFrame) -> str:
        """Convierte el resultado tabular en lenguaje natural."""
        resultado_str = df.to_string(index=False) if not df.empty else "(sin resultados)"
        
        respuesta = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""Pregunta: {pregunta}
SQL ejecutado: {sql}
Resultado: {resultado_str}

Responde la pregunta en una oración clara y directa en español."""
            }]
        )
        return respuesta.content[0].text.strip()
    
    def query(self, pregunta: str) -> dict:
        """
        Traduce una pregunta a SQL, la ejecuta y devuelve la respuesta.
        
        Returns:
            dict con claves: sql, dataframe, respuesta, error
        """
        resultado = {"sql": None, "dataframe": None, "respuesta": None, "error": None}
        
        try:
            # Generar SQL
            sql = self._generar_sql(pregunta)
            resultado["sql"] = sql
            
            # Validar antes de ejecutar
            es_valido, razon = validar_sql(sql)
            if not es_valido:
                resultado["error"] = f"SQL rechazado por seguridad: {razon}"
                return resultado
            
            # Ejecutar
            df = pd.read_sql_query(sql, self.conn)
            resultado["dataframe"] = df
            
            # Interpretar
            resultado["respuesta"] = self._interpretar_resultado(pregunta, sql, df)
            
        except Exception as e:
            resultado["error"] = str(e)
        
        return resultado
```

---

## 5. Validación de SQL

Antes de ejecutar cualquier SQL generado por IA, debemos validar que no sea destructivo.

```python
import re

OPERACIONES_PELIGROSAS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bTRUNCATE\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"--",          # Comentarios SQL (posible inyección)
    r"/\*",         # Comentarios de bloque
]

def validar_sql(sql: str) -> tuple[bool, str]:
    """
    Valida que el SQL sea solo lectura y seguro.
    
    Returns:
        (True, "") si es válido
        (False, motivo) si es rechazado
    """
    sql_upper = sql.upper()
    
    for patron in OPERACIONES_PELIGROSAS:
        if re.search(patron, sql_upper):
            operacion = patron.replace(r"\b", "").replace("\\b", "")
            return False, f"Operación no permitida detectada: {operacion}"
    
    # Debe empezar con SELECT
    sql_limpio = sql.strip().lstrip("(")
    if not sql_limpio.upper().startswith("SELECT") and \
       not sql_limpio.upper().startswith("WITH"):
        return False, "Solo se permiten consultas SELECT o WITH"
    
    return True, ""
```

### Prueba de la validación

```python
casos = [
    ("SELECT * FROM clientes", True),
    ("DROP TABLE clientes", False),
    ("SELECT * FROM clientes; DELETE FROM pedidos", False),
    ("UPDATE productos SET precio=0", False),
    ("WITH cte AS (SELECT id FROM clientes) SELECT * FROM cte", True),
]

for sql, esperado in casos:
    valido, razon = validar_sql(sql)
    estado = "OK" if valido == esperado else "FALLO"
    print(f"[{estado}] {'VALIDO' if valido else 'RECHAZADO'}: {sql[:50]}")
    if razon:
        print(f"       Razón: {razon}")
```

---

## 6. Few-shot para mejorar precisión

Los few-shot examples en el system prompt mejoran significativamente la calidad del SQL generado. Incluir 3-5 ejemplos representativos reduce errores de alias, funciones y agrupaciones.

```python
FEW_SHOT_EJEMPLOS = """
Ejemplos de preguntas y SQL esperado:

P: ¿Cuántos clientes hay registrados?
SQL: SELECT COUNT(*) AS total FROM clientes;

P: ¿Cuáles son los productos más vendidos por categoría?
SQL: SELECT categoria, COUNT(*) AS total_productos, AVG(precio) AS precio_promedio
     FROM productos GROUP BY categoria ORDER BY total_productos DESC;

P: ¿Qué clientes han hecho más de un pedido?
SQL: SELECT c.nombre, COUNT(p.id) AS total_pedidos
     FROM clientes c JOIN pedidos p ON c.id = p.cliente_id
     GROUP BY c.id, c.nombre HAVING COUNT(p.id) > 1;
"""
```

Los ejemplos deben ser:
- **Representativos**: cubrir los patrones más comunes de tu BD
- **Variados**: incluir COUNT, SUM, GROUP BY, JOIN, HAVING
- **Correctos**: verificados manualmente antes de incluirlos

---

## 7. Limitaciones conocidas

| Limitación | Descripción | Solución parcial |
|-----------|-------------|-----------------|
| JOINs complejos | Con 4+ tablas la precisión cae | Usar schema-linking (Artículo 02) |
| Aliases ambiguos | Columnas con nombre genérico como `id` en múltiples tablas | Agregar comentarios al esquema |
| Fechas relativas | "el mes pasado", "esta semana" | Inyectar `CURRENT_DATE` (Artículo 02) |
| Agregaciones anidadas | `AVG(COUNT(*))` requiere subqueries complejos | Few-shot específicos |
| Dialectos SQL | SQLite ≠ PostgreSQL ≠ MySQL | Especificar el dialecto en el prompt |

### Demo completa

```python
# Crear el sistema
sistema = TextToSQL()

preguntas = [
    "¿Cuántos clientes hay en total?",
    "¿Cuáles son los 3 productos más caros?",
    "¿Cuál es el total de ventas por estado de pedido?",
    "¿Qué ciudad tiene más clientes registrados?",
]

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}")
    resultado = sistema.query(pregunta)
    
    if resultado["error"]:
        print(f"Error: {resultado['error']}")
    else:
        print(f"SQL: {resultado['sql']}")
        print(f"Respuesta: {resultado['respuesta']}")
```

---

## Resumen

En este artículo construimos un sistema Text-to-SQL básico que:

1. Inyecta el esquema de la BD en el contexto de Claude de forma compacta
2. Genera SQL usando few-shot examples para mayor precisión
3. Valida el SQL antes de ejecutarlo (no DROP, DELETE, UPDATE)
4. Ejecuta la consulta y convierte el resultado a lenguaje natural

En el siguiente artículo añadimos **self-correction**: cuando el SQL falla, Claude recibe el error y lo corrige automáticamente, manejando también JOINs complejos y fechas relativas.

**Siguiente**: [Text-to-SQL Avanzado →](02-text-to-sql-avanzado.md)
