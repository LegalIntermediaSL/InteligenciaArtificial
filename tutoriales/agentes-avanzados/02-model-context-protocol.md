# 02 — Model Context Protocol (MCP)

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. Qué es MCP y por qué importa
2. Arquitectura MCP
3. Instalar y configurar MCP
4. Crear un servidor MCP simple en Python
5. Crear un servidor MCP con recursos
6. Conectar Claude Desktop a un servidor MCP
7. Casos de uso reales de MCP
8. Extensiones sugeridas

---

## 1. Qué es MCP y por qué importa

Antes de MCP, conectar un LLM a una herramienta externa (una base de datos, una API, el sistema de archivos) requería escribir integraciones personalizadas para cada par LLM-herramienta. El resultado: N LLMs × M herramientas = N×M integraciones distintas que mantener.

**El problema concreto:**

- Claude sabe usar herramientas definidas en `tools=[]`, pero esas herramientas viven en tu código de aplicación.
- Si quieres que Claude acceda a una base de datos PostgreSQL, debes escribir el conector, manejar la autenticación, serializar los resultados, etc.
- Si otro LLM quiere acceder a la misma base de datos, hay que repetir todo el trabajo.

**La solución de MCP:**

El Model Context Protocol es un protocolo abierto (especificado por Anthropic, adoptado por la industria) que estandariza cómo los LLMs se comunican con fuentes de datos y herramientas externas.

Funciona como un "USB-C para LLMs": un estándar único que cualquier cliente (Claude, Cursor, IDEs) puede usar para conectarse a cualquier servidor (Postgres, GitHub, Slack, tu filesystem).

**Ventajas:**

- Escribe el servidor MCP una vez; cualquier cliente compatible puede usarlo.
- Separación clara entre la lógica del LLM y la lógica de acceso a datos.
- Servidores reutilizables y publicables como paquetes.
- Anthropic mantiene un repositorio de servidores MCP de referencia.

---

## 2. Arquitectura MCP

```
┌─────────────────────────────────────────────────────────────┐
│                      APLICACIÓN HOST                        │
│  (Claude Desktop, Cursor, tu propio código Python)          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   CLIENTE MCP                        │  │
│  │  - Gestiona conexiones con servidores                │  │
│  │  - Traduce capacidades del servidor a tools/prompts  │  │
│  │  - Envía resultados al LLM                           │  │
│  └──────────────────┬───────────────────────────────────┘  │
└─────────────────────│───────────────────────────────────────┘
                      │ Protocolo MCP (JSON-RPC 2.0)
           ┌──────────┴──────────┐
           │                     │
  ┌────────▼────────┐   ┌────────▼────────┐
  │  SERVIDOR MCP A  │   │  SERVIDOR MCP B  │
  │  (filesystem)    │   │  (base de datos) │
  │                  │   │                  │
  │  Expone:         │   │  Expone:         │
  │  • Herramientas  │   │  • Herramientas  │
  │  • Recursos      │   │  • Recursos      │
  │  • Prompts       │   │  • Prompts       │
  └──────────────────┘   └──────────────────┘
```

**Los tres tipos de capacidades que expone un servidor MCP:**

| Capacidad | Descripción | Ejemplo |
|---|---|---|
| **Tools** (Herramientas) | Funciones que el LLM puede invocar | `buscar_archivo`, `ejecutar_query` |
| **Resources** (Recursos) | Datos que el LLM puede leer | contenido de un archivo, filas de una tabla |
| **Prompts** | Plantillas de prompt reutilizables | "Analiza este log de error" |

**Transporte:** MCP soporta dos mecanismos de transporte:
- **stdio**: el cliente lanza el servidor como subproceso y se comunica por stdin/stdout. Ideal para desarrollo local.
- **SSE (Server-Sent Events)**: el servidor es un proceso HTTP independiente. Ideal para despliegue remoto.

---

## 3. Instalar y configurar MCP

```bash
pip install mcp
```

El paquete `mcp` incluye tanto el SDK para construir servidores como las utilidades de cliente.

Verifica la instalación:

```python
import mcp
print(mcp.__version__)
```

Para desarrollo, también es útil:

```bash
pip install mcp[cli]        # incluye herramientas de línea de comandos
pip install anthropic        # para el cliente LLM
```

Estructura de un proyecto con servidor MCP:

```
mi_servidor_mcp/
├── servidor.py          # el servidor MCP
├── cliente_prueba.py    # para probar el servidor localmente
└── requirements.txt
```

---

## 4. Crear un servidor MCP simple en Python

Construiremos un servidor MCP con una herramienta que busca texto en archivos del sistema local. Este es un caso de uso práctico muy común: dar a Claude acceso al contenido de tus documentos sin copiar y pegar.

```python
# servidor.py
import asyncio
import os
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ---------------------------------------------------------------------------
# Inicialización del servidor
# ---------------------------------------------------------------------------

servidor = Server("buscador-archivos")

# Directorio base en el que el servidor puede buscar
# Ajusta esta ruta a tu directorio de documentos
DIRECTORIO_BASE = Path.home() / "Documents"


# ---------------------------------------------------------------------------
# Herramienta 1: buscar_en_archivos
# ---------------------------------------------------------------------------

@servidor.list_tools()
async def listar_herramientas() -> list[types.Tool]:
    """Declara las herramientas disponibles en este servidor."""
    return [
        types.Tool(
            name="buscar_en_archivos",
            description=(
                "Busca un texto en archivos de texto dentro del directorio de documentos. "
                "Devuelve los archivos que contienen el texto y las líneas donde aparece."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "texto": {
                        "type": "string",
                        "description": "El texto a buscar (no distingue mayúsculas).",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Filtrar por extensión de archivo (ej: '.txt', '.md'). Opcional.",
                        "default": "",
                    },
                    "max_resultados": {
                        "type": "integer",
                        "description": "Número máximo de resultados a devolver.",
                        "default": 10,
                    },
                },
                "required": ["texto"],
            },
        ),
        types.Tool(
            name="leer_archivo",
            description="Lee el contenido completo de un archivo de texto.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ruta": {
                        "type": "string",
                        "description": "Ruta absoluta o relativa al directorio de documentos.",
                    }
                },
                "required": ["ruta"],
            },
        ),
        types.Tool(
            name="listar_archivos",
            description="Lista los archivos en el directorio de documentos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subdirectorio": {
                        "type": "string",
                        "description": "Subdirectorio a listar (relativo al directorio base). Vacío para el raíz.",
                        "default": "",
                    },
                    "extension": {
                        "type": "string",
                        "description": "Filtrar por extensión. Ej: '.md'",
                        "default": "",
                    },
                },
                "required": [],
            },
        ),
    ]


@servidor.call_tool()
async def ejecutar_herramienta(
    nombre: str, argumentos: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Ejecuta la herramienta solicitada y devuelve el resultado."""

    if nombre == "buscar_en_archivos":
        return await _buscar_en_archivos(
            texto=argumentos["texto"],
            extension=argumentos.get("extension", ""),
            max_resultados=argumentos.get("max_resultados", 10),
        )

    elif nombre == "leer_archivo":
        return await _leer_archivo(argumentos["ruta"])

    elif nombre == "listar_archivos":
        return await _listar_archivos(
            subdirectorio=argumentos.get("subdirectorio", ""),
            extension=argumentos.get("extension", ""),
        )

    else:
        raise ValueError(f"Herramienta desconocida: {nombre}")


# ---------------------------------------------------------------------------
# Implementaciones de las herramientas
# ---------------------------------------------------------------------------

async def _buscar_en_archivos(
    texto: str, extension: str, max_resultados: int
) -> list[types.TextContent]:
    """Busca recursivamente en el directorio base."""
    resultados = []
    texto_lower = texto.lower()

    # Extensiones de texto que podemos leer de forma segura
    extensiones_texto = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv", ".html", ".css"}

    for ruta in DIRECTORIO_BASE.rglob("*"):
        if len(resultados) >= max_resultados:
            break

        # Filtros
        if not ruta.is_file():
            continue
        if extension and ruta.suffix.lower() != extension.lower():
            continue
        if not extension and ruta.suffix.lower() not in extensiones_texto:
            continue

        try:
            contenido = ruta.read_text(encoding="utf-8", errors="ignore")
            lineas_encontradas = []

            for num, linea in enumerate(contenido.splitlines(), 1):
                if texto_lower in linea.lower():
                    lineas_encontradas.append(f"  Línea {num}: {linea.strip()[:100]}")

            if lineas_encontradas:
                ruta_relativa = ruta.relative_to(DIRECTORIO_BASE)
                resultados.append(f"📄 {ruta_relativa}\n" + "\n".join(lineas_encontradas))

        except (PermissionError, OSError):
            continue

    if not resultados:
        respuesta = f"No se encontró '{texto}' en ningún archivo."
    else:
        respuesta = f"Se encontró '{texto}' en {len(resultados)} archivo(s):\n\n" + "\n\n".join(resultados)

    return [types.TextContent(type="text", text=respuesta)]


async def _leer_archivo(ruta: str) -> list[types.TextContent]:
    """Lee un archivo y devuelve su contenido."""
    # Seguridad: solo permitir acceso dentro del directorio base
    ruta_obj = Path(ruta)
    if not ruta_obj.is_absolute():
        ruta_obj = DIRECTORIO_BASE / ruta_obj

    try:
        ruta_obj = ruta_obj.resolve()
        if not str(ruta_obj).startswith(str(DIRECTORIO_BASE.resolve())):
            return [types.TextContent(type="text", text="Error: acceso denegado fuera del directorio permitido.")]

        contenido = ruta_obj.read_text(encoding="utf-8", errors="ignore")
        return [types.TextContent(type="text", text=f"Contenido de {ruta_obj.name}:\n\n{contenido}")]

    except FileNotFoundError:
        return [types.TextContent(type="text", text=f"Archivo no encontrado: {ruta}")]
    except PermissionError:
        return [types.TextContent(type="text", text=f"Sin permiso para leer: {ruta}")]


async def _listar_archivos(subdirectorio: str, extension: str) -> list[types.TextContent]:
    """Lista archivos en el directorio base o un subdirectorio."""
    directorio = DIRECTORIO_BASE / subdirectorio if subdirectorio else DIRECTORIO_BASE

    if not directorio.exists():
        return [types.TextContent(type="text", text=f"Directorio no encontrado: {directorio}")]

    archivos = []
    for entrada in sorted(directorio.iterdir()):
        if extension and entrada.is_file() and entrada.suffix.lower() != extension.lower():
            continue
        tipo = "📁" if entrada.is_dir() else "📄"
        archivos.append(f"{tipo} {entrada.name}")

    if not archivos:
        texto = "El directorio está vacío."
    else:
        texto = f"Contenido de {directorio}:\n\n" + "\n".join(archivos)

    return [types.TextContent(type="text", text=texto)]


# ---------------------------------------------------------------------------
# Punto de entrada: lanzar el servidor via stdio
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (flujo_lectura, flujo_escritura):
        await servidor.run(
            flujo_lectura,
            flujo_escritura,
            servidor.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

Para probar el servidor sin Claude Desktop, puedes usar el cliente MCP de Python:

```python
# cliente_prueba.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def probar_servidor():
    parametros = StdioServerParameters(
        command="python",
        args=["servidor.py"],
    )

    async with stdio_client(parametros) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            # Inicializar la conexión
            await sesion.initialize()

            # Listar herramientas disponibles
            herramientas = await sesion.list_tools()
            print("Herramientas disponibles:")
            for h in herramientas.tools:
                print(f"  - {h.name}: {h.description}")

            # Probar la herramienta de búsqueda
            print("\nProbando búsqueda...")
            resultado = await sesion.call_tool(
                "buscar_en_archivos",
                {"texto": "python", "extension": ".md", "max_resultados": 3},
            )
            print(resultado.content[0].text)


if __name__ == "__main__":
    asyncio.run(probar_servidor())
```

---

## 5. Crear un servidor MCP con recursos

Además de herramientas (funciones), MCP permite exponer **recursos**: datos que el LLM puede leer directamente, como el contenido de archivos o filas de una base de datos.

```python
# servidor_con_recursos.py
import asyncio
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

servidor = Server("gestor-notas")

# Simularemos una pequeña base de notas en JSON
ARCHIVO_NOTAS = Path("notas.json")


def cargar_notas() -> dict:
    if ARCHIVO_NOTAS.exists():
        return json.loads(ARCHIVO_NOTAS.read_text(encoding="utf-8"))
    # Datos de ejemplo
    return {
        "reunion-2025-01-15": {
            "titulo": "Reunión de planificación Q1",
            "fecha": "2025-01-15",
            "contenido": "Objetivos: lanzar MVP en marzo, contratar 2 desarrolladores.",
        },
        "ideas-producto": {
            "titulo": "Ideas para el producto",
            "fecha": "2025-01-20",
            "contenido": "Añadir modo oscuro, integración con Slack, API pública.",
        },
    }


def guardar_notas(notas: dict) -> None:
    ARCHIVO_NOTAS.write_text(json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Recursos: datos que el LLM puede leer
# ---------------------------------------------------------------------------

@servidor.list_resources()
async def listar_recursos() -> list[types.Resource]:
    """Expone cada nota como un recurso independiente."""
    notas = cargar_notas()
    recursos = []

    for clave, nota in notas.items():
        recursos.append(
            types.Resource(
                uri=f"nota://{clave}",
                name=nota["titulo"],
                description=f"Nota del {nota['fecha']}",
                mimeType="text/plain",
            )
        )

    # También exponemos el índice completo como recurso
    recursos.append(
        types.Resource(
            uri="nota://indice",
            name="Índice de todas las notas",
            description="Lista completa de notas disponibles en formato JSON",
            mimeType="application/json",
        )
    )

    return recursos


@servidor.read_resource()
async def leer_recurso(uri: types.AnyUrl) -> str:
    """Lee el contenido de un recurso por su URI."""
    uri_str = str(uri)
    notas = cargar_notas()

    if uri_str == "nota://indice":
        return json.dumps(
            {k: {"titulo": v["titulo"], "fecha": v["fecha"]} for k, v in notas.items()},
            ensure_ascii=False,
            indent=2,
        )

    # Extraer la clave de la URI (nota://clave → clave)
    clave = uri_str.replace("nota://", "")

    if clave not in notas:
        raise ValueError(f"Nota no encontrada: {clave}")

    nota = notas[clave]
    return f"# {nota['titulo']}\nFecha: {nota['fecha']}\n\n{nota['contenido']}"


# ---------------------------------------------------------------------------
# Herramientas: funciones que el LLM puede invocar
# ---------------------------------------------------------------------------

@servidor.list_tools()
async def listar_herramientas() -> list[types.Tool]:
    return [
        types.Tool(
            name="crear_nota",
            description="Crea una nueva nota con título y contenido.",
            inputSchema={
                "type": "object",
                "properties": {
                    "clave": {"type": "string", "description": "Identificador único (sin espacios, usar guiones)."},
                    "titulo": {"type": "string", "description": "Título de la nota."},
                    "contenido": {"type": "string", "description": "Contenido de la nota."},
                },
                "required": ["clave", "titulo", "contenido"],
            },
        ),
        types.Tool(
            name="eliminar_nota",
            description="Elimina una nota por su clave.",
            inputSchema={
                "type": "object",
                "properties": {
                    "clave": {"type": "string", "description": "Clave de la nota a eliminar."}
                },
                "required": ["clave"],
            },
        ),
    ]


@servidor.call_tool()
async def ejecutar_herramienta(nombre: str, argumentos: dict) -> list[types.TextContent]:
    notas = cargar_notas()

    if nombre == "crear_nota":
        from datetime import date
        clave = argumentos["clave"]
        if clave in notas:
            return [types.TextContent(type="text", text=f"Error: ya existe una nota con la clave '{clave}'.")]

        notas[clave] = {
            "titulo": argumentos["titulo"],
            "fecha": date.today().isoformat(),
            "contenido": argumentos["contenido"],
        }
        guardar_notas(notas)
        return [types.TextContent(type="text", text=f"Nota '{argumentos['titulo']}' creada con éxito.")]

    elif nombre == "eliminar_nota":
        clave = argumentos["clave"]
        if clave not in notas:
            return [types.TextContent(type="text", text=f"No existe ninguna nota con la clave '{clave}'.")]
        del notas[clave]
        guardar_notas(notas)
        return [types.TextContent(type="text", text=f"Nota '{clave}' eliminada.")]

    raise ValueError(f"Herramienta desconocida: {nombre}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (lectura, escritura):
        await servidor.run(lectura, escritura, servidor.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Conectar Claude Desktop a un servidor MCP

Claude Desktop puede conectarse a servidores MCP locales mediante su archivo de configuración.

**Ubicación del archivo de configuración:**

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Formato del archivo `claude_desktop_config.json`:**

```json
{
  "mcpServers": {
    "buscador-archivos": {
      "command": "python",
      "args": [
        "/ruta/absoluta/a/tu/servidor.py"
      ],
      "env": {
        "PYTHONPATH": "/ruta/absoluta/a/tu/proyecto"
      }
    },
    "gestor-notas": {
      "command": "python",
      "args": [
        "/ruta/absoluta/a/tu/servidor_con_recursos.py"
      ]
    }
  }
}
```

**Pasos para conectar:**

1. Crea o edita el archivo `claude_desktop_config.json` con las rutas absolutas a tus servidores.
2. Reinicia Claude Desktop completamente (Cmd+Q en macOS, no solo cerrar la ventana).
3. En la interfaz de Claude Desktop, verás un ícono de herramientas (🔧) que indica los servidores conectados.
4. Ahora puedes pedirle a Claude cosas como: *"Busca en mis documentos todo lo que mencione 'presupuesto'"*.

**Solución de problemas:**

```bash
# Verificar que el servidor funciona de forma independiente
python servidor.py

# Ver los logs de MCP en macOS
tail -f ~/Library/Logs/Claude/mcp*.log
```

**Nota importante:** usa siempre rutas absolutas en `args`. Las rutas relativas fallan porque Claude Desktop no tiene un directorio de trabajo definido.

---

## 7. Casos de uso reales de MCP

### Bases de datos

```python
# Fragmento: servidor MCP para SQLite
import sqlite3
from mcp.server import Server
from mcp import types

servidor = Server("sqlite-mcp")
RUTA_DB = "mi_base_de_datos.db"

@servidor.list_tools()
async def herramientas() -> list[types.Tool]:
    return [
        types.Tool(
            name="ejecutar_query",
            description="Ejecuta una consulta SQL de solo lectura en la base de datos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "Consulta SQL SELECT."}
                },
                "required": ["sql"],
            },
        )
    ]

@servidor.call_tool()
async def ejecutar(nombre: str, args: dict) -> list[types.TextContent]:
    if nombre == "ejecutar_query":
        sql = args["sql"].strip()
        # Seguridad: solo SELECT
        if not sql.upper().startswith("SELECT"):
            return [types.TextContent(type="text", text="Error: solo se permiten consultas SELECT.")]

        con = sqlite3.connect(RUTA_DB)
        con.row_factory = sqlite3.Row
        cursor = con.execute(sql)
        filas = [dict(fila) for fila in cursor.fetchall()]
        con.close()

        import json
        return [types.TextContent(type="text", text=json.dumps(filas, ensure_ascii=False, indent=2))]

    raise ValueError(f"Herramienta desconocida: {nombre}")
```

### Casos de uso más allá de archivos y bases de datos

| Caso | Herramientas/recursos que expone |
|---|---|
| **GitHub MCP** | Repositorios, issues, PRs, commits, ramas |
| **Slack MCP** | Canales, mensajes, usuarios, enviar mensajes |
| **Google Drive MCP** | Archivos, carpetas, búsqueda, lectura de documentos |
| **API REST genérica** | Endpoints como herramientas, respuestas como recursos |
| **Calendario** | Eventos, disponibilidad, crear citas |
| **Monitoring** | Métricas de sistema, logs, alertas |
| **CRM** | Contactos, oportunidades, historial de interacciones |

El repositorio oficial de servidores MCP de referencia está en: `github.com/modelcontextprotocol/servers`

---

## 8. Extensiones sugeridas

- **Autenticación en servidores MCP:** implementa OAuth 2.0 o API keys en el servidor para acceso seguro a datos sensibles.
- **Servidor MCP con SSE:** convierte tu servidor stdio en un servidor HTTP para despliegue en la nube.
- **MCP con múltiples transportes:** soporta stdio para desarrollo y SSE para producción en el mismo servidor.
- **Tests de servidores MCP:** escribe tests unitarios con `pytest` para cada herramienta y recurso.
- **Servidor MCP para PostgreSQL:** conecta a una base de datos real con `asyncpg` y expón las tablas como recursos.
- **Servidor MCP para correo:** conecta a IMAP para que Claude pueda leer y organizar emails.

---

**Siguiente:** [03 — Computer Use](./03-computer-use.md)
