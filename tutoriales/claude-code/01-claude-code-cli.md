# Claude Code CLI: guía completa

## Instalación y primeros pasos

```bash
# Instalar globalmente
npm install -g @anthropic-ai/claude-code

# Autenticarse (abre el navegador la primera vez)
claude

# Lanzar en el directorio del proyecto
cd mi-proyecto
claude
```

Claude Code detecta automáticamente el lenguaje, el framework y la estructura del proyecto
leyendo los archivos existentes antes de responder.

## Modos de uso

### Modo interactivo (REPL)

```bash
claude
# Abre el REPL interactivo
# > Escribe tu petición en lenguaje natural
```

### Modo una sola pregunta

```bash
# Responde y sale
claude "¿Qué hace la función processPayment en src/billing.ts?"

# Con pipe
cat error.log | claude "¿Qué causó este error?"

# Leer archivo específico
claude "Revisa este componente" < src/components/Header.tsx
```

### Modo headless (para CI/CD)

```bash
# Sin interactividad — ideal para scripts y pipelines
claude --print "Genera el changelog desde el último tag"
claude -p "Revisa la seguridad de este PR" --no-color
```

---

## Slash commands esenciales

| Comando | Función |
|---------|---------|
| `/help` | Lista todos los comandos disponibles |
| `/clear` | Limpia el contexto de la conversación actual |
| `/compact` | Comprime el historial para liberar contexto |
| `/cost` | Muestra el coste acumulado de la sesión |
| `/model` | Cambia el modelo (Sonnet / Opus / Haiku) |
| `/init` | Genera un CLAUDE.md inicial para el proyecto |
| `/memory` | Gestiona la memoria persistente del proyecto |
| `/review` | Inicia una revisión del PR actual |
| `/pr_comments` | Muestra y responde comentarios del PR |
| `/mcp` | Gestiona los servidores MCP conectados |
| `/doctor` | Diagnóstico de la instalación y configuración |
| `/vim` | Activa keybindings de Vim en el REPL |

---

## CLAUDE.md: instrucciones persistentes

El archivo `CLAUDE.md` en la raíz del proyecto se carga automáticamente en cada sesión.
Es el equivalente a un contrato entre tú y Claude Code sobre cómo trabajar en este repo.

```markdown
# CLAUDE.md — Mi Proyecto SaaS

## Stack
- Backend: FastAPI (Python 3.11)
- Frontend: Next.js 15 (TypeScript)
- BD: PostgreSQL con SQLAlchemy ORM
- Tests: pytest (backend), Vitest (frontend)
- Linting: Ruff (Python), ESLint + Prettier (TS)

## Convenciones de código
- Python: snake_case para funciones y variables
- TypeScript: camelCase para variables, PascalCase para componentes
- Commits: Conventional Commits (feat/fix/docs/refactor/test)
- Nunca usar `any` en TypeScript — usar tipos explícitos

## Comandos clave
- Tests backend: `pytest tests/ -v`
- Tests frontend: `npm run test`
- Linting: `ruff check . && ruff format .`
- Dev server: `uvicorn main:app --reload`

## Reglas importantes
- No modificar archivos en `migrations/` manualmente
- Las API keys van siempre en `.env`, nunca hardcodeadas
- Cada nueva feature necesita tests antes del merge
- Los endpoints nuevos requieren docstring con ejemplo de uso

## Contexto del proyecto
SaaS de gestión de contratos legales para despachos de abogados en España.
El equipo es de 3 personas. Prioridad: velocidad de desarrollo sin sacrificar seguridad.
```

### CLAUDE.md en subdirectorios

Puedes tener CLAUDE.md específicos por carpeta:

```
proyecto/
├── CLAUDE.md              ← instrucciones globales
├── backend/
│   └── CLAUDE.md          ← reglas específicas de Python/FastAPI
└── frontend/
    └── CLAUDE.md          ← reglas específicas de Next.js
```

---

## Gestión de memoria

Claude Code tiene tres niveles de memoria:

```bash
# Ver memoria actual
/memory

# Añadir un hecho a la memoria del proyecto
/memory add "El cliente principal es Despacho García — muy sensibles a la privacidad"

# La memoria persiste en ~/.claude/projects/<hash-del-proyecto>/memory.md
```

```markdown
# Tipos de memoria en Claude Code

## Memoria de proyecto (CLAUDE.md)
Instrucciones técnicas sobre el codebase. Commitear con el repo.

## Memoria de usuario (~/.claude/memory.md)
Preferencias personales que aplican a todos los proyectos.
Ejemplo: "Prefiero respuestas concisas. No expliques código obvio."

## Memoria de sesión (en contexto)
Dura mientras el REPL está abierto. /clear la borra.
```

---

## Configuración avanzada (~/.claude/settings.json)

```json
{
  "model": "claude-sonnet-4-6",
  "theme": "dark",
  "autoCompact": true,
  "compactThreshold": 0.85,
  "permissions": {
    "allow": [
      "Bash(git:*)",
      "Bash(npm:*)",
      "Bash(pytest:*)",
      "Read(**)",
      "Edit(**)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Bash(curl:*)",
      "Bash(wget:*)"
    ]
  },
  "env": {
    "ANTHROPIC_API_KEY": "sk-ant-..."
  }
}
```

### Configuración por proyecto (.claude/settings.json)

```json
{
  "model": "claude-opus-4-7",
  "permissions": {
    "allow": [
      "Bash(docker:*)",
      "Bash(make:*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "echo 'Ejecutando: $CLAUDE_TOOL_INPUT'"}]
      }
    ]
  }
}
```

---

## Flujos de trabajo comunes

### Explorar un codebase nuevo

```
> Eres nuevo en este proyecto. Explícame la arquitectura general,
  los puntos de entrada principales y las dependencias clave.

> ¿Dónde se gestiona la autenticación? ¿Qué middleware existe?

> ¿Qué convenciones de naming usa este equipo?
```

### Implementar una feature

```
> Necesito añadir un endpoint POST /api/invoices que:
  - Reciba un PDF adjunto
  - Lo analice con Claude para extraer los datos
  - Los guarde en la tabla invoices de PostgreSQL
  - Devuelva el JSON con los datos extraídos
  Sigue las convenciones del resto de endpoints en src/api/

> Ahora escribe los tests de integración para este endpoint.
  Usa fixtures de pytest y una base de datos de test en memoria.
```

### Revisar y mejorar código existente

```
> Revisa src/services/billing.py y dime:
  1. ¿Hay problemas de seguridad potenciales?
  2. ¿Qué casos borde no están manejados?
  3. ¿El código es testeable tal como está?

> Refactoriza la función calculate_invoice() para que sea más
  testeable. Extrae la lógica de negocio del I/O.
```

### Debug de errores

```bash
# Pegar el error directamente
> TypeError: Cannot read property 'id' of undefined
  at processPayment (src/billing.ts:45)
  Contexto: se produce cuando el usuario cancela el pago a mitad

> Busca en el código dónde se llama a processPayment y
  traza el flujo completo hasta el origen del error.
```

### Escribir documentación

```
> Genera la documentación OpenAPI (docstrings) para todos los
  endpoints en src/api/ que no tengan documentación.
  Incluye ejemplos de request y response.

> Actualiza el README.md con las instrucciones de instalación
  actuales. El README está desactualizado desde la migración a Docker.
```

---

## Atajos de teclado en el REPL

| Atajo | Función |
|-------|---------|
| `Ctrl+C` | Interrumpir la respuesta en curso |
| `Ctrl+L` | Limpiar pantalla (mantiene contexto) |
| `↑ / ↓` | Navegar por el historial de mensajes |
| `Escape` | Cancelar edición actual |
| `Tab` | Autocompletar rutas de archivo |

---

## Integración con IDEs

### VS Code

```bash
# Instalar extensión oficial
code --install-extension anthropic.claude-code

# O desde el REPL de Claude Code
claude  # y luego usar /terminal-setup
```

### JetBrains (IntelliJ, PyCharm, WebStorm)

Disponible como plugin en el Marketplace de JetBrains.
Buscar "Claude Code" en Preferences → Plugins.

---

## Buenas prácticas

```
✓ Sé específico en tus peticiones — incluye el archivo, la función y el contexto
✓ Usa /compact cuando la sesión lleva mucho rato (ahorra tokens)
✓ Commitea el CLAUDE.md con el repo — es documentación del proyecto
✓ Usa /cost para monitorizar el gasto en sesiones largas
✓ Para tareas grandes, divide en pasos: explorar → planificar → implementar
✓ Revisa siempre los cambios antes de aceptarlos (Claude Code muestra diffs)

✗ No uses Claude Code como motor de búsqueda general (usa /clear y enfócate)
✗ No omitas el contexto: "arregla el bug" es peor que "arregla el bug en checkout"
✗ No confíes ciegamente en código generado para endpoints de pago o auth
```

## Recursos

- [Notebook interactivo](../notebooks/claude-code/01-claude-code-cli.ipynb)
- [Claude Code — documentación oficial](https://docs.anthropic.com/en/docs/claude-code)
- [CLAUDE.md — referencia completa](https://docs.anthropic.com/en/docs/claude-code/memory)
