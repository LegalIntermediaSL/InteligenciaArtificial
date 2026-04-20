# Bloque 23 — Claude Code y desarrollo asistido por IA

Claude Code es el CLI oficial de Anthropic que convierte tu terminal en un agente
de desarrollo con acceso completo al codebase, git, tests y herramientas del sistema.

## Tutoriales

| # | Título | Descripción |
|---|--------|-------------|
| 01 | [Claude Code CLI](01-claude-code-cli.md) | Instalación, comandos, CLAUDE.md y configuración |
| 02 | [MCP Servers personalizados](02-mcp-servers.md) | Crear servidores MCP propios e integrarlos con Claude Code |
| 03 | [Hooks y automatización](03-hooks-automatizacion.md) | Hooks de ciclo de vida, validaciones y CI/CD |
| 04 | [Flujos de desarrollo con IA](04-flujos-desarrollo-ia.md) | Cursor vs Copilot vs Claude Code, workflows completos |

## ¿Por qué Claude Code?

```
EDITOR TRADICIONAL        CLAUDE CODE
──────────────────        ──────────────────────────────────
Escribir código    →      Describir qué construir
Buscar en docs     →      "¿Cómo funciona X en este repo?"
Revisar PR         →      /review — análisis automático
Refactorizar       →      "Refactoriza el módulo de auth"
Escribir tests     →      "Añade tests para todos los casos borde"
```

## Instalación

```bash
npm install -g @anthropic-ai/claude-code

# O con la última versión
npm install -g @anthropic-ai/claude-code@latest

# Verificar instalación
claude --version

# Primera vez: autenticarse
claude
```

## Requisitos

- Node.js 18+
- API key de Anthropic (o suscripción Claude Pro/Max)
- Git (para los comandos de control de versiones)
