# MCP Servers personalizados para Claude Code

## Qué es el Model Context Protocol (MCP)

MCP es un protocolo abierto que permite a Claude conectarse con herramientas y fuentes
de datos externas de forma estandarizada. Cada servidor MCP expone herramientas,
recursos y prompts que Claude puede usar durante la conversación.

```
Claude Code
    ↓ MCP protocol (JSON-RPC sobre stdio/HTTP)
MCP Server
    ↓
Tu API interna / Base de datos / Servicio externo
```

**Cuándo crear un MCP server propio:**
- Tienes una API interna que quieres que Claude pueda consultar
- Necesitas acceso a datos que no están en el filesystem (BD, CRM, ERP)
- Quieres encapsular operaciones complejas como herramientas reutilizables
- Necesitas conectar Claude Code con herramientas de tu empresa

---

## MCP Server mínimo en Python

```python
# mcp_server.py
# pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import json

app = Server("mi-servidor-mcp")

@app.list_tools()
async def listar_herramientas() -> list[Tool]:
    """Claude pregunta qué herramientas están disponibles."""
    return [
        Tool(
            name="buscar_cliente",
            description="Busca un cliente en la base de datos por nombre o email",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nombre o email del cliente"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="obtener_metricas",
            description="Obtiene métricas del dashboard de la semana actual",
            inputSchema={
                "type": "object",
                "properties": {
                    "periodo": {
                        "type": "string",
                        "enum": ["hoy", "semana", "mes"],
                        "description": "Periodo de tiempo"
                    }
                },
                "required": ["periodo"]
            }
        )
    ]

@app.call_tool()
async def ejecutar_herramienta(name: str, arguments: dict) -> list[TextContent]:
    """Claude llama a esta función cuando quiere usar una herramienta."""
    
    if name == "buscar_cliente":
        query = arguments["query"]
        # En producción: consultar base de datos real
        clientes = [
            {"id": "c001", "nombre": "ACME Corp", "email": "admin@acme.com", "plan": "Pro", "mrr": 299},
            {"id": "c002", "nombre": "StartupXYZ", "email": "cto@startupxyz.io", "plan": "Starter", "mrr": 29},
        ]
        resultados = [c for c in clientes if query.lower() in c["nombre"].lower() or query.lower() in c["email"].lower()]
        return [TextContent(type="text", text=json.dumps(resultados, ensure_ascii=False))]
    
    elif name == "obtener_metricas":
        periodo = arguments["periodo"]
        # En producción: consultar PostgreSQL, BigQuery, etc.
        metricas = {
            "hoy": {"nuevos_usuarios": 12, "mrr_nuevo": 870, "churn": 0},
            "semana": {"nuevos_usuarios": 67, "mrr_nuevo": 4230, "churn": 2},
            "mes": {"nuevos_usuarios": 280, "mrr_nuevo": 18400, "churn": 8}
        }
        return [TextContent(type="text", text=json.dumps(metricas[periodo], ensure_ascii=False))]
    
    return [TextContent(type="text", text=json.dumps({"error": f"Herramienta '{name}' desconocida"}))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Conectar el servidor a Claude Code

### Opción 1: Configuración global (~/.claude/settings.json)

```json
{
  "mcpServers": {
    "mi-empresa": {
      "command": "python",
      "args": ["/ruta/absoluta/mcp_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/mydb",
        "API_KEY": "sk-..."
      }
    }
  }
}
```

### Opción 2: Configuración por proyecto (.claude/settings.json)

```json
{
  "mcpServers": {
    "crm-interno": {
      "command": "python",
      "args": ["./tools/mcp_crm.py"]
    },
    "base-datos": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_CONNECTION_STRING": "postgresql://localhost/mydb"
      }
    }
  }
}
```

### Verificar que el servidor está conectado

```bash
# En el REPL de Claude Code
/mcp

# Output esperado:
# ✅ mi-empresa (2 tools, 0 resources)
#    - buscar_cliente
#    - obtener_metricas
```

---

## MCP Server con recursos (Resources)

Los recursos son datos que Claude puede leer bajo demanda, similar a archivos.

```python
from mcp.types import Resource

@app.list_resources()
async def listar_recursos() -> list[Resource]:
    return [
        Resource(
            uri="empresa://config/general",
            name="Configuración de la empresa",
            description="Variables de configuración y constantes del sistema",
            mimeType="application/json"
        ),
        Resource(
            uri="empresa://docs/api",
            name="Documentación de la API interna",
            description="Endpoints disponibles y sus parámetros",
            mimeType="text/markdown"
        )
    ]

@app.read_resource()
async def leer_recurso(uri: str) -> str:
    if uri == "empresa://config/general":
        config = {
            "nombre_empresa": "Mi SaaS SA",
            "moneda": "EUR",
            "zona_horaria": "Europe/Madrid",
            "planes": ["starter", "pro", "business"],
            "limite_api_pro": 1000
        }
        return json.dumps(config, ensure_ascii=False, indent=2)
    
    elif uri == "empresa://docs/api":
        return """# API Interna — v2.1

## Autenticación
Todas las rutas requieren `Authorization: Bearer <token>`

## Endpoints principales
- GET /api/clientes — lista paginada
- POST /api/clientes — crear cliente
- GET /api/metricas?periodo=semana — métricas del dashboard
- POST /api/facturas — generar factura
"""
    return "{}"
```

---

## MCP Server con prompts reutilizables

Los prompts son plantillas que Claude puede usar directamente.

```python
from mcp.types import Prompt, PromptMessage, PromptArgument

@app.list_prompts()
async def listar_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="analizar_cliente",
            description="Analiza el estado de salud de un cliente y sugiere acciones",
            arguments=[
                PromptArgument(name="cliente_id", description="ID del cliente", required=True)
            ]
        )
    ]

@app.get_prompt()
async def obtener_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "analizar_cliente":
        cliente_id = arguments["cliente_id"]
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Analiza el cliente {cliente_id} usando la herramienta buscar_cliente.
Luego evalúa:
1. ¿Está en riesgo de churn?
2. ¿Hay oportunidad de upsell?
3. ¿Qué acción debería tomar el equipo de Customer Success esta semana?

Sé concreto y accionable."""
                )
            )
        ]
    return []
```

---

## Servidor MCP sobre HTTP (para equipos)

En lugar de stdio, puedes exponer el servidor como API REST para que todo el equipo lo comparta:

```python
# mcp_server_http.py
# pip install mcp fastapi uvicorn

from mcp.server.fastapi import create_mcp_app
# ... (mismo código del servidor) ...

# Crear la app FastAPI con el servidor MCP
mcp_app = create_mcp_app(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(mcp_app, host="0.0.0.0", port=8080)
```

```json
// settings.json del equipo (apunta al servidor compartido)
{
  "mcpServers": {
    "empresa-shared": {
      "url": "http://mcp.empresa.internal:8080/mcp"
    }
  }
}
```

---

## Servidores MCP de la comunidad (listos para usar)

```bash
# Base de datos PostgreSQL
npx -y @modelcontextprotocol/server-postgres

# Filesystem extendido
npx -y @modelcontextprotocol/server-filesystem /ruta/al/proyecto

# GitHub (PRs, issues, repos)
npx -y @modelcontextprotocol/server-github

# Slack
npx -y @modelcontextprotocol/server-slack

# Notion
npx -y @modelcontextprotocol/server-notion

# Brave Search (búsqueda web)
npx -y @modelcontextprotocol/server-brave-search
```

Configuración para GitHub:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."
      }
    }
  }
}
```

---

## Caso práctico: MCP para acceso a base de datos

```python
# mcp_database.py — servidor MCP que expone consultas seguras a PostgreSQL
import asyncpg
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio
import os

app = Server("database-mcp")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost/mydb")

@app.list_tools()
async def tools() -> list[Tool]:
    return [
        Tool(
            name="consultar_ventas",
            description="Consulta las ventas agrupadas por periodo y/o producto",
            inputSchema={
                "type": "object",
                "properties": {
                    "desde": {"type": "string", "description": "Fecha inicio YYYY-MM-DD"},
                    "hasta": {"type": "string", "description": "Fecha fin YYYY-MM-DD"},
                    "agrupar_por": {"type": "string", "enum": ["dia", "semana", "mes", "producto", "region"]}
                },
                "required": ["desde", "hasta"]
            }
        ),
        Tool(
            name="top_clientes",
            description="Devuelve los N clientes con mayor MRR",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Número de clientes (default 10)", "default": 10}
                }
            }
        )
    ]

@app.call_tool()
async def call(name: str, arguments: dict) -> list[TextContent]:
    conn = await asyncpg.connect(DB_URL)
    try:
        if name == "consultar_ventas":
            desde = arguments["desde"]
            hasta = arguments["hasta"]
            agrupar = arguments.get("agrupar_por", "mes")
            
            # Consultas parametrizadas — nunca interpolación de strings
            if agrupar == "mes":
                rows = await conn.fetch("""
                    SELECT DATE_TRUNC('month', fecha) as periodo,
                           SUM(importe) as total,
                           COUNT(*) as num_ventas
                    FROM ventas
                    WHERE fecha BETWEEN $1 AND $2
                    GROUP BY 1 ORDER BY 1
                """, desde, hasta)
            elif agrupar == "producto":
                rows = await conn.fetch("""
                    SELECT producto, SUM(importe) as total, COUNT(*) as num_ventas
                    FROM ventas
                    WHERE fecha BETWEEN $1 AND $2
                    GROUP BY producto ORDER BY total DESC
                """, desde, hasta)
            else:
                rows = await conn.fetch("""
                    SELECT fecha, SUM(importe) as total FROM ventas
                    WHERE fecha BETWEEN $1 AND $2 GROUP BY fecha ORDER BY fecha
                """, desde, hasta)
            
            datos = [dict(row) for row in rows]
            return [TextContent(type="text", text=json.dumps(datos, default=str))]
        
        elif name == "top_clientes":
            n = arguments.get("n", 10)
            rows = await conn.fetch(
                "SELECT nombre, email, mrr FROM clientes ORDER BY mrr DESC LIMIT $1", n
            )
            return [TextContent(type="text", text=json.dumps([dict(r) for r in rows], default=str))]
    
    finally:
        await conn.close()
    
    return [TextContent(type="text", text="{}")]

async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

asyncio.run(main())
```

Con este servidor conectado, puedes preguntar en Claude Code:
```
> ¿Cuáles fueron las ventas mensuales del Q1 2026?
> Muéstrame los 5 clientes con mayor MRR y analiza si están en riesgo
```

## Recursos

- [Notebook interactivo](../notebooks/claude-code/02-mcp-servers.ipynb)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Directorio de servidores MCP](https://github.com/modelcontextprotocol/servers)
- [Especificación MCP](https://modelcontextprotocol.io/docs)
