# Bloque 31: Text-to-SQL e IA para Datos Estructurados

Aprende a construir sistemas que traducen preguntas en lenguaje natural a consultas SQL, desde lo básico hasta arquitecturas de producción con agentes analistas de datos.

## Artículos

| # | Artículo | Descripción | Tiempo |
|---|----------|-------------|--------|
| 01 | [Text-to-SQL Básico](01-text-to-sql-basico.md) | Traducir preguntas a SQL con Claude, inyección de esquema y validación | 60-90 min |
| 02 | [Text-to-SQL Avanzado](02-text-to-sql-avanzado.md) | Self-correction, schema-linking, JOINs y fechas relativas | 90-120 min |
| 03 | [Agente Analista SQL](03-agente-analista-sql.md) | Agente con herramientas: SQL + gráficas + lenguaje natural | 90-120 min |
| 04 | [Text-to-SQL en Producción](04-text-to-sql-produccion.md) | Seguridad, sandboxing, caché y casos empresariales | 60-90 min |

**Tiempo total estimado: 5-7 horas**

## Requisitos

```bash
pip install anthropic sqlalchemy pandas duckdb matplotlib
```

## Requisitos previos

- Python 3.10+
- Conocimientos básicos de SQL (SELECT, WHERE, JOIN)
- API key de Anthropic configurada: `export ANTHROPIC_API_KEY="sk-ant-..."`

## Estructura del bloque

```
text-to-sql/
├── README.md                      ← Este archivo
├── 01-text-to-sql-basico.md
├── 02-text-to-sql-avanzado.md
├── 03-agente-analista-sql.md
└── 04-text-to-sql-produccion.md

notebooks/text-to-sql/
├── 01-text-to-sql-basico.ipynb
├── 02-text-to-sql-avanzado.ipynb
├── 03-agente-analista-sql.ipynb
└── 04-text-to-sql-produccion.ipynb
```

## ¿Por qué Text-to-SQL?

El 80% de los datos empresariales vive en bases de datos relacionales, pero menos del 20% de los empleados saben SQL. Text-to-SQL cierra esa brecha: cualquier persona puede hacer preguntas en español y obtener respuestas basadas en datos reales, sin escribir una sola línea de código.

Con Claude como motor de lenguaje, podemos construir sistemas que:
- Comprenden el esquema de la base de datos
- Generan SQL correcto y seguro
- Corrigen errores automáticamente
- Explican los resultados en lenguaje natural
