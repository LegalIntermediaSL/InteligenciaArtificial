# 10 — MCP: Model Context Protocol y conexiones con herramientas externas

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [¿Qué es MCP?](#1-qué-es-mcp)
2. [Arquitectura del protocolo](#2-arquitectura-del-protocolo)
3. [Primitivas: Tools, Resources y Prompts](#3-primitivas-tools-resources-y-prompts)
4. [Crear un servidor MCP en Python](#4-crear-un-servidor-mcp-en-python)
5. [Conectar Claude Desktop a un servidor MCP](#5-conectar-claude-desktop-a-un-servidor-mcp)
6. [Servidores MCP oficiales relevantes](#6-servidores-mcp-oficiales-relevantes)
7. [MCP + NotebookLM y herramientas de productividad](#7-mcp--notebooklm-y-herramientas-de-productividad)
8. [Integrar MCP en aplicaciones propias](#8-integrar-mcp-en-aplicaciones-propias)
9. [MCP vs Tool Use directo — cuándo usar cada uno](#9-mcp-vs-tool-use-directo--cuándo-usar-cada-uno)
10. [Seguridad y buenas prácticas](#10-seguridad-y-buenas-prácticas)
11. [Resumen](#11-resumen)

---

## 1. ¿Qué es MCP?

**Model Context Protocol (MCP)** es un protocolo abierto creado por Anthropic en noviembre de 2024 para estandarizar cómo los modelos de lenguaje se conectan con herramientas externas, fuentes de datos y servicios.

Antes de MCP, cada integración era ad hoc: cada aplicación de IA implementaba sus propias funciones de herramientas, su propia gestión de contexto, su propia serialización. MCP define un **estándar universal** que funciona con cualquier LLM y cualquier herramienta.

### El problema que resuelve

```
ANTES — Integraciones dispersas (N × M combinaciones):

  Claude ─── función_buscar_en_github()
  Claude ─── función_leer_ficheros()
  GPT-4  ─── función_buscar_en_github()   ← código duplicado
  GPT-4  ─── función_leer_ficheros()      ← código duplicado
  Gemini ─── función_buscar_en_github()   ← código duplicado

DESPUÉS — Protocolo estándar (N + M):

  Claude  ──┐
  GPT-4   ──┼──► MCP ◄──┬── Servidor GitHub
  Gemini  ──┘            ├── Servidor Filesystem
                         └── Servidor Postgres
```

**Una sola implementación del servidor MCP sirve a todos los clientes que soporten el protocolo.**

### Adopción

MCP ya es compatible con:
- **Claude Desktop** (cliente de referencia)
- **Claude Code** (terminal)
- **Cursor** (IDE)
- **Zed** (editor)
- **Windsurf** (IDE)
- **Continue** (extensión VS Code)
- Múltiples frameworks: LangChain, LlamaIndex, AutoGen

---

## 2. Arquitectura del protocolo

### 2.1 Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLIENTE MCP                               │
│  (Claude Desktop, tu aplicación, Claude Code...)                │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   LLM Host   │    │   MCP Client │    │  Tu lógica   │     │
│   │  (Claude,    │◄──►│  (protocolo) │◄──►│  de negocio  │     │
│   │   GPT, etc.) │    │              │    │              │     │
│   └──────────────┘    └──────┬───────┘    └──────────────┘     │
└──────────────────────────────┼──────────────────────────────────┘
                               │ JSON-RPC 2.0
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌──────────────┐
    │  Servidor MCP   │ │ Servidor MCP│ │ Servidor MCP │
    │  (Filesystem)   │ │  (GitHub)   │ │  (Postgres)  │
    └─────────────────┘ └─────────────┘ └──────────────┘
```

### 2.2 Transporte

MCP soporta dos mecanismos de transporte:

| Transporte | Cuándo usar |
|---|---|
| **stdio** | Servidores locales (proceso hijo). El cliente lanza el servidor como subproceso y se comunica por stdin/stdout. El más común para herramientas locales. |
| **HTTP + SSE** | Servidores remotos. El servidor expone un endpoint HTTP. Permite servidores compartidos, multi-cliente, en la nube. |

### 2.3 El protocolo JSON-RPC

MCP usa JSON-RPC 2.0. La comunicación básica:

```json
// Cliente → Servidor: solicitar lista de herramientas
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

// Servidor → Cliente: lista de herramientas disponibles
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "tools": [
      {
        "name": "buscar_documentos",
        "description": "Busca en la base de conocimiento",
        "inputSchema": {
          "type": "object",
          "properties": {"consulta": {"type": "string"}},
          "required": ["consulta"]
        }
      }
    ]
  }
}

// Cliente → Servidor: llamar a una herramienta
{
  "jsonrpc": "2.0", "id": 2,
  "method": "tools/call",
  "params": {"name": "buscar_documentos", "arguments": {"consulta": "LangGraph"}}
}
```

---

## 3. Primitivas: Tools, Resources y Prompts

MCP define tres tipos de capacidades que un servidor puede exponer:

### 3.1 Tools (herramientas)

Son funciones que el LLM puede invocar para **ejecutar acciones** o **obtener datos dinámicos**. Equivalen a las funciones de tool use de Claude, pero expuestas de forma estándar.

```
Tool: buscar_web(consulta) → resultados de búsqueda
Tool: ejecutar_sql(query) → resultados de base de datos
Tool: crear_issue(título, descripción) → GitHub issue creado
Tool: leer_archivo(ruta) → contenido del fichero
```

### 3.2 Resources (recursos)

Son **datos estáticos o semi-estáticos** que el servidor puede proporcionar al LLM como contexto. A diferencia de las herramientas, no ejecutan acciones: solo exponen datos.

```
Resource: file://proyecto/README.md → contenido del fichero
Resource: db://clientes/schema → esquema de la base de datos
Resource: config://app/settings.json → configuración actual
```

Los recursos tienen URIs y pueden ser leídos por el cliente para incluirlos en el contexto del LLM.

### 3.3 Prompts

Son **plantillas de prompts reutilizables** que el servidor ofrece al cliente. Permiten que las herramientas vengan con instrucciones optimizadas sobre cómo usarlas.

```
Prompt: "analizar_codigo" → instrucción completa para revisar código
Prompt: "resumir_documento" → instrucción optimizada para resumir
Prompt: "depurar_error" → flujo para diagnóstico de errores
```

---

## 4. Crear un servidor MCP en Python

### 4.1 Instalación

```bash
pip install mcp
```

### 4.2 Servidor mínimo

```python
# servidor_mcp.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
import json
import math

app = Server("mi-servidor-ia")

# ── Registrar herramientas ─────────────────────────────────────────────────────

@app.list_tools()
async def listar_herramientas() -> list[Tool]:
    """Devuelve la lista de herramientas que ofrece este servidor."""
    return [
        Tool(
            name="calcular",
            description="Evalúa una expresión matemática",
            inputSchema={
                "type": "object",
                "properties": {
                    "expresion": {
                        "type": "string",
                        "description": "Expresión matemática (ej: math.sqrt(16), 2**10)"
                    }
                },
                "required": ["expresion"]
            }
        ),
        Tool(
            name="buscar_en_base_conocimiento",
            description="Busca información en la base de conocimiento interna",
            inputSchema={
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Término o pregunta a buscar"
                    }
                },
                "required": ["consulta"]
            }
        ),
    ]

BASE_CONOCIMIENTO = {
    "mcp": "MCP (Model Context Protocol) es el protocolo estándar de Anthropic para conectar LLMs con herramientas externas.",
    "langgraph": "LangGraph es un framework para construir agentes con grafos de estado, memoria y human-in-the-loop.",
    "ollama": "Ollama permite ejecutar LLMs open-source en local de forma sencilla, sin enviar datos a servidores externos.",
    "rag": "RAG combina recuperación de documentos con generación de texto para respuestas más precisas y actualizadas.",
}

@app.call_tool()
async def ejecutar_herramienta(name: str, arguments: dict) -> CallToolResult:
    """Ejecuta la herramienta solicitada."""
    if name == "calcular":
        try:
            resultado = eval(arguments["expresion"], {"__builtins__": {}}, {"math": math})
            return CallToolResult(content=[TextContent(type="text", text=str(resultado))])
        except Exception as e:
            return CallToolResult(content=[TextContent(type="text", text=f"Error: {e}")])

    elif name == "buscar_en_base_conocimiento":
        consulta = arguments["consulta"].lower()
        for clave, valor in BASE_CONOCIMIENTO.items():
            if clave in consulta:
                return CallToolResult(content=[TextContent(type="text", text=valor)])
        return CallToolResult(content=[TextContent(type="text",
            text=f"No se encontró información sobre: {arguments['consulta']}")])

    return CallToolResult(content=[TextContent(type="text", text=f"Herramienta '{name}' no encontrada")])

# ── Iniciar servidor ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(app))
```

### 4.3 Servidor con recursos

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, ResourceContents, TextResourceContents
import os

app = Server("servidor-ficheros")

@app.list_resources()
async def listar_recursos() -> list[Resource]:
    """Expone ficheros de configuración como recursos."""
    recursos = []
    for nombre in os.listdir('.'):
        if nombre.endswith(('.md', '.json', '.txt')):
            recursos.append(Resource(
                uri=f"file://{os.path.abspath(nombre)}",
                name=nombre,
                description=f"Fichero: {nombre}",
                mimeType="text/plain"
            ))
    return recursos

@app.read_resource()
async def leer_recurso(uri: str) -> ResourceContents:
    """Lee el contenido de un fichero por su URI."""
    ruta = uri.replace("file://", "")
    with open(ruta, 'r', encoding='utf-8') as f:
        contenido = f.read()
    return ResourceContents(
        uri=uri,
        contents=[TextResourceContents(uri=uri, text=contenido, mimeType="text/plain")]
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(app))
```

---

## 5. Conectar Claude Desktop a un servidor MCP

Claude Desktop es la forma más directa de conectar servidores MCP sin escribir código cliente.

### 5.1 Configuración

Edita el fichero de configuración de Claude Desktop:

```
# macOS
~/Library/Application Support/Claude/claude_desktop_config.json

# Windows
%APPDATA%\Claude\claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "mi-servidor": {
      "command": "python",
      "args": ["/ruta/absoluta/a/servidor_mcp.py"],
      "env": {
        "PYTHONPATH": "/ruta/a/tu/venv/lib/python3.11/site-packages"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/usuario/Documentos"]
    },
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

Una vez configurado, reinicia Claude Desktop. Verás el icono de herramientas (🔧) en el chat indicando que los servidores están activos.

### 5.2 Usar las herramientas desde el chat

Con los servidores activos, simplemente habla con Claude de forma natural:

```
Usuario: "Busca en mi servidor de conocimiento qué es MCP y calcula 2^16"
Claude: [Llama automáticamente a buscar_en_base_conocimiento("MCP")]
        [Llama a calcular("2**16")]
        "MCP es el protocolo estándar... y 2^16 = 65.536"
```

---

## 6. Servidores MCP oficiales relevantes

Anthropic y la comunidad mantienen un repositorio de servidores MCP listos para usar:

### 6.1 Servidores oficiales de Anthropic

| Servidor | Comando | Qué hace |
|---|---|---|
| **Filesystem** | `@modelcontextprotocol/server-filesystem` | Lee y escribe ficheros en directorios autorizados |
| **GitHub** | `@modelcontextprotocol/server-github` | Issues, PRs, repos, búsqueda de código |
| **GitLab** | `@modelcontextprotocol/server-gitlab` | Similar a GitHub pero para GitLab |
| **Postgres** | `@modelcontextprotocol/server-postgres` | Consultas SQL a bases de datos PostgreSQL |
| **SQLite** | `@modelcontextprotocol/server-sqlite` | Bases de datos SQLite locales |
| **Brave Search** | `@modelcontextprotocol/server-brave-search` | Búsqueda web con Brave API |
| **Google Maps** | `@modelcontextprotocol/server-google-maps` | Geocoding, rutas, lugares |
| **Fetch** | `@modelcontextprotocol/server-fetch` | Descargar y leer páginas web |
| **Memory** | `@modelcontextprotocol/server-memory` | Memoria persistente entre sesiones |
| **Puppeteer** | `@modelcontextprotocol/server-puppeteer` | Control de navegador Chrome |

### 6.2 Servidores de la comunidad

| Servidor | Qué hace |
|---|---|
| **Slack** | Enviar mensajes, buscar canales, leer conversaciones |
| **Notion** | Leer y escribir páginas de Notion |
| **Linear** | Gestionar issues y proyectos de Linear |
| **Jira** | Issues y proyectos de Atlassian Jira |
| **Figma** | Leer diseños y componentes de Figma |
| **Obsidian** | Leer y escribir notas del vault de Obsidian |
| **Docker** | Gestionar contenedores y servicios |
| **Kubernetes** | Consultar y gestionar pods y deployments |
| **AWS** | Interactuar con servicios de Amazon Web Services |

---

## 7. MCP + NotebookLM y herramientas de productividad

### 7.1 MCP y NotebookLM

**NotebookLM** (Google) es una herramienta de estudio e investigación basada en IA que procesa documentos propios. **No tiene integración directa con MCP** en la actualidad (2026), ya que NotebookLM es un producto cerrado de Google con su propia interfaz.

Sin embargo, hay flujos de trabajo complementarios:

```
Flujo típico de investigación con LLMs:

NotebookLM (análisis profundo de PDFs):
  1. Sube tus PDFs, artículos, notas
  2. Genera resúmenes, podcast de audio, preguntas clave
  3. Exporta el resumen como texto

Claude + MCP (acción y síntesis):
  1. Lee el texto exportado de NotebookLM (filesystem MCP)
  2. Combina con búsquedas en tu base de conocimiento
  3. Genera código, borradores, comparativas
  4. Guarda los resultados (filesystem MCP)
```

### 7.2 Flujo de trabajo real: investigación asistida

```json
// claude_desktop_config.json para un flujo de investigación
{
  "mcpServers": {
    "documentos": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/Users/usuario/Documentos/Investigacion"]
    },
    "web": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {"BRAVE_API_KEY": "..."}
    },
    "memoria": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

Con esta configuración, Claude puede:
- Leer los PDFs exportados de NotebookLM en tu carpeta de investigación
- Buscar información actualizada en la web
- Guardar notas y síntesis con memoria persistente entre sesiones

### 7.3 MCP + otras herramientas de productividad

**Notion + MCP:**
```json
{
  "notion": {
    "command": "npx",
    "args": ["-y", "@notionhq/mcp"],
    "env": {"NOTION_API_KEY": "secret_..."}
  }
}
```

Prompt ejemplo:
```
"Lee la página 'Proyecto IA' de Notion, busca en la web las últimas noticias sobre
LangGraph y añade un resumen de 3 párrafos en la sección 'Actualizaciones'."
```

**Obsidian + MCP:**
```json
{
  "obsidian": {
    "command": "npx",
    "args": ["-y", "mcp-obsidian", "/Users/usuario/ObsidianVault"]
  }
}
```

Prompt ejemplo:
```
"Revisa mis notas sobre 'RAG' en Obsidian, compáralas con los últimos papers
y crea una nueva nota de síntesis con los conceptos clave."
```

---

## 8. Integrar MCP en aplicaciones propias

### 8.1 Cliente MCP en Python

```python
import anthropic
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def usar_servidor_mcp(pregunta: str):
    """Conecta a un servidor MCP y usa Claude para responder."""

    # Parámetros del servidor (levantamos servidor_mcp.py como subproceso)
    server_params = StdioServerParameters(
        command="python",
        args=["servidor_mcp.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Inicializar sesión
            await session.initialize()

            # Obtener herramientas disponibles
            tools_response = await session.list_tools()
            tools = tools_response.tools
            print(f"Herramientas disponibles: {[t.name for t in tools]}")

            # Convertir al formato de Anthropic
            herramientas_anthropic = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                }
                for t in tools
            ]

            # Llamar a Claude con las herramientas del servidor MCP
            client = anthropic.Anthropic()
            mensajes = [{"role": "user", "content": pregunta}]

            while True:
                respuesta = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    tools=herramientas_anthropic,
                    messages=mensajes
                )
                mensajes.append({"role": "assistant", "content": respuesta.content})

                if respuesta.stop_reason == "end_turn":
                    texto = next((b.text for b in respuesta.content if hasattr(b, 'text')), '')
                    return texto

                # Ejecutar herramientas via MCP
                resultados_tools = []
                for bloque in respuesta.content:
                    if bloque.type != "tool_use":
                        continue
                    print(f"  Ejecutando: {bloque.name}({bloque.input})")
                    resultado = await session.call_tool(bloque.name, bloque.input)
                    texto_resultado = resultado.content[0].text if resultado.content else ""
                    resultados_tools.append({
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": texto_resultado
                    })
                mensajes.append({"role": "user", "content": resultados_tools})

# Uso
respuesta = asyncio.run(usar_servidor_mcp(
    "Busca información sobre MCP en la base de conocimiento y calcula 2^20"
))
print(f"\nRespuesta: {respuesta}")
```

### 8.2 Servidor MCP con autenticación

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
import os

app = Server("servidor-seguro")

# Token de autenticación desde variable de entorno
API_TOKEN = os.environ.get("MCP_API_TOKEN")

def verificar_token(token: str | None) -> bool:
    return token == API_TOKEN

@app.call_tool()
async def ejecutar_herramienta(name: str, arguments: dict) -> CallToolResult:
    # Verificar autenticación antes de ejecutar
    token = arguments.pop("_auth_token", None)
    if API_TOKEN and not verificar_token(token):
        return CallToolResult(
            content=[TextContent(type="text", text="Error: Token de autenticación inválido")],
            isError=True
        )
    # ... lógica de la herramienta
```

### 8.3 Servidor MCP como microservicio HTTP

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount

app_mcp = Server("servidor-http")

# ... definir herramientas ...

# Servir como HTTP con SSE
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await app_mcp.run(streams[0], streams[1], app_mcp.create_initialization_options())

starlette_app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Mount("/messages", app=sse.handle_post_message),
])

# uvicorn servidor_http:starlette_app --port 8000
```

---

## 9. MCP vs Tool Use directo — cuándo usar cada uno

| Criterio | Tool Use directo | MCP |
|---|---|---|
| **Complejidad de setup** | Mínima (solo código Python) | Requiere servidor y configuración |
| **Portabilidad** | Ligada a tu aplicación | Cualquier cliente MCP la reutiliza |
| **Reutilización** | Solo en tu código | Claude Desktop, Cursor, todos los clientes |
| **Múltiples LLMs** | Hay que adaptar a cada API | Un servidor MCP sirve a todos |
| **Herramientas locales** | Sí, directamente | Sí, via stdio |
| **Herramientas remotas** | Sí, via HTTP | Sí, via HTTP + SSE |
| **Autenticación** | Implementas tú | Framework lo facilita |
| **Ecosistema** | El tuyo | Creciente comunidad de servidores |

**Usa Tool Use directo cuando:**
- Construyes una aplicación propia con un solo cliente
- Las herramientas son específicas de tu dominio
- Quieres el menor overhead posible
- Estás prototipando

**Usa MCP cuando:**
- Quieres que las herramientas funcionen en Claude Desktop, Cursor, etc.
- Vas a reutilizar las herramientas en múltiples proyectos
- Construyes un producto que otros van a integrar
- Necesitas compatibilidad con múltiples LLMs

---

## 10. Seguridad y buenas prácticas

### 10.1 Principio de mínimo privilegio

```json
// INCORRECTO: acceso total al sistema de ficheros
{
  "filesystem": {
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/"]
  }
}

// CORRECTO: solo las carpetas necesarias
{
  "filesystem": {
    "args": ["-y", "@modelcontextprotocol/server-filesystem",
             "/Users/usuario/Proyectos/mi-proyecto"]
  }
}
```

### 10.2 Validar entradas siempre

```python
@app.call_tool()
async def ejecutar_sql(name: str, arguments: dict) -> CallToolResult:
    query = arguments.get("query", "")

    # Nunca ejecutar directamente el input del LLM
    # Validar que solo son SELECT (no DROP, DELETE, etc.)
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return CallToolResult(
            content=[TextContent(type="text", text="Solo se permiten consultas SELECT")],
            isError=True
        )

    # Usar parámetros preparados, nunca f-strings
    # ...
```

### 10.3 Tokens y credenciales

```bash
# INCORRECTO: credenciales en claude_desktop_config.json directamente
"env": {"DATABASE_URL": "postgresql://user:password@host/db"}

# CORRECTO: referenciar variables de entorno del sistema
# En claude_desktop_config.json:
"env": {"DATABASE_URL": "${DATABASE_URL}"}

# Y en tu .env o variables del sistema:
# DATABASE_URL=postgresql://user:password@host/db
```

### 10.4 Limitar operaciones destructivas

```python
OPERACIONES_PERMITIDAS = {"leer", "buscar", "listar"}
OPERACIONES_PROTEGIDAS = {"escribir", "borrar", "ejecutar"}

async def ejecutar_herramienta(name: str, arguments: dict) -> CallToolResult:
    operacion = arguments.get("operacion", "")
    if operacion in OPERACIONES_PROTEGIDAS:
        # Loguear y requerir confirmación explícita
        confirmado = arguments.get("confirmar", False)
        if not confirmado:
            return CallToolResult(content=[TextContent(
                type="text",
                text=f"La operación '{operacion}' requiere confirmación explícita. "
                     f"Incluye 'confirmar: true' para proceder."
            )])
```

---

## 11. Resumen

MCP estandariza la conexión entre LLMs y el mundo exterior con un protocolo abierto y reutilizable:

```
Tu herramienta         MCP Server          MCP Client          LLM
─────────────          ──────────          ──────────          ───
Base de datos    ──►  list_tools()   ──►  descubrir    ──►  Claude
API externa      ──►  call_tool()    ──►  invocar      ──►  GPT-4
Ficheros         ──►  list_resources()──►  leer        ──►  Gemini
Notas/Notion     ──►  get_prompt()   ──►  usar         ──►  cualquier LLM
```

**Las tres primitivas:**
- **Tools** — acciones que el LLM puede ejecutar
- **Resources** — datos que el LLM puede leer como contexto
- **Prompts** — plantillas optimizadas para casos de uso concretos

**Flujos prácticos con MCP:**
- Claude Desktop + filesystem MCP → gestión de proyectos con IA
- Claude Desktop + Notion MCP → asistente de productividad personal
- Claude Desktop + GitHub MCP → revisión y gestión de código
- Aplicación propia + servidor MCP personalizado → producto de IA reutilizable

### Recursos

- [Documentación oficial MCP](https://modelcontextprotocol.io)
- [Repositorio de servidores MCP](https://github.com/modelcontextprotocol/servers)
- [SDK Python de MCP](https://github.com/modelcontextprotocol/python-sdk)
- [Registro de servidores MCP de la comunidad](https://mcp.so)
- Tutorial anterior: [09 — Modelos locales con Ollama](./09-modelos-locales-ollama.md)
- Notebook interactivo: [10-mcp.ipynb](../notebooks/llms/10-mcp.ipynb)
