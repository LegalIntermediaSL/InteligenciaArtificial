# Hooks y automatización en Claude Code

## Qué son los hooks

Los hooks son comandos shell que se ejecutan automáticamente en respuesta a eventos
del ciclo de vida de Claude Code: antes de usar una herramienta, después, al recibir
una notificación, etc.

```
EVENTO                    HOOK                    USO TÍPICO
──────────────────────────────────────────────────────────────
Claude va a editar        PreToolUse(Edit)     →  validar / formatear
Claude editó              PostToolUse(Edit)    →  ejecutar linter
Claude va a ejecutar bash PreToolUse(Bash)     →  auditar comandos
Respuesta lista           Notification        →  alertar al usuario
Sesión termina            Stop                →  commit automático
```

---

## Configuración de hooks

Los hooks se configuran en `.claude/settings.json` del proyecto o en `~/.claude/settings.json`.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Editando: $CLAUDE_TOOL_INPUT_FILE_PATH'"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "command",
            "command": "ruff check $CLAUDE_TOOL_INPUT_FILE_PATH --fix --quiet 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

### Variables de entorno disponibles en hooks

| Variable | Disponible en | Contenido |
|----------|---------------|-----------|
| `CLAUDE_TOOL_NAME` | Pre/Post ToolUse | Nombre de la herramienta (`Edit`, `Bash`, `Read`...) |
| `CLAUDE_TOOL_INPUT` | PreToolUse | JSON con los parámetros de la herramienta |
| `CLAUDE_TOOL_INPUT_FILE_PATH` | PreToolUse(Edit/Read) | Ruta del archivo afectado |
| `CLAUDE_TOOL_INPUT_COMMAND` | PreToolUse(Bash) | Comando bash que Claude va a ejecutar |
| `CLAUDE_TOOL_OUTPUT` | PostToolUse | Resultado de la herramienta |
| `CLAUDE_NOTIFICATION_MESSAGE` | Notification | Texto de la notificación |

---

## Casos de uso prácticos

### 1. Linting automático tras cada edición

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [{
          "type": "command",
          "command": "bash -c 'FILE=$CLAUDE_TOOL_INPUT_FILE_PATH; if [[ $FILE == *.py ]]; then ruff check $FILE --fix -q 2>/dev/null; elif [[ $FILE == *.ts || $FILE == *.tsx ]]; then npx eslint $FILE --fix -q 2>/dev/null; fi'"
        }]
      }
    ]
  }
}
```

### 2. Auditoría de comandos bash

```bash
#!/bin/bash
# scripts/audit_bash.sh — registra todos los comandos que Claude ejecuta

COMANDO=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command',''))")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$HOME/.claude/audit.log"

echo "$TIMESTAMP | $COMANDO" >> "$LOG_FILE"

# Bloquear comandos peligrosos
if echo "$COMANDO" | grep -qE "rm -rf|DROP TABLE|--force"; then
  echo "BLOQUEADO: Comando potencialmente destructivo detectado" >&2
  exit 1
fi

exit 0
```

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "bash scripts/audit_bash.sh"
        }]
      }
    ]
  }
}
```

### 3. Notificaciones de sistema al terminar tareas largas

```bash
#!/bin/bash
# scripts/notify.sh — notificación macOS/Linux cuando Claude termina

MESSAGE=$(echo "$CLAUDE_NOTIFICATION_MESSAGE" | head -c 100)

# macOS
if command -v osascript &>/dev/null; then
  osascript -e "display notification \"$MESSAGE\" with title \"Claude Code\""
fi

# Linux (notify-send)
if command -v notify-send &>/dev/null; then
  notify-send "Claude Code" "$MESSAGE"
fi
```

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "command",
          "command": "bash scripts/notify.sh"
        }]
      }
    ]
  }
}
```

### 4. Formateo automático antes de editar

```bash
#!/bin/bash
# Formatea el archivo con el formateador adecuado ANTES de que Claude lo edite
# Garantiza que Claude trabaja sobre código ya formateado

FILE=$(echo "$CLAUDE_TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))")

case "$FILE" in
  *.py)   python3 -m black "$FILE" --quiet 2>/dev/null ;;
  *.ts|*.tsx) npx prettier --write "$FILE" --log-level silent 2>/dev/null ;;
  *.json) python3 -m json.tool "$FILE" > /tmp/fmt.json && mv /tmp/fmt.json "$FILE" 2>/dev/null ;;
esac
```

### 5. Commit automático al terminar una sesión (Stop hook)

```bash
#!/bin/bash
# scripts/auto_commit.sh — propone un commit al cerrar Claude Code

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Solo si hay cambios
if git diff --quiet && git diff --staged --quiet; then
  exit 0
fi

echo "Claude Code detectó cambios sin commitear."
echo "¿Quieres crear un commit automático? (y/N)"
read -t 10 -n 1 respuesta

if [[ "$respuesta" == "y" || "$respuesta" == "Y" ]]; then
  git add -A
  MSG=$(claude --print "Genera un mensaje de commit en español para estos cambios: $(git diff --staged --stat)" 2>/dev/null)
  git commit -m "${MSG:-feat: cambios de Claude Code}"
  echo "✅ Commit creado: $MSG"
fi
```

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "command",
          "command": "bash scripts/auto_commit.sh"
        }]
      }
    ]
  }
}
```

---

## Hook de validación de tests

Este hook ejecuta los tests después de cada edición y bloquea a Claude si fallan:

```bash
#!/bin/bash
# scripts/run_tests_on_edit.sh

FILE="$CLAUDE_TOOL_INPUT_FILE_PATH"

# Solo ejecutar tests si se editó un archivo Python de producción
if [[ "$FILE" != *.py ]] || [[ "$FILE" == *test* ]] || [[ "$FILE" == *migration* ]]; then
  exit 0
fi

echo "Ejecutando tests relacionados con $FILE..."

# Descubrir tests relacionados con el módulo editado
MODULE=$(basename "$FILE" .py)
TEST_FILE="tests/test_${MODULE}.py"

if [ -f "$TEST_FILE" ]; then
  if ! pytest "$TEST_FILE" -q --tb=short 2>&1; then
    echo "❌ Tests fallando en $TEST_FILE. Claude no puede continuar hasta que pasen." >&2
    exit 1
  fi
  echo "✅ Tests pasando."
fi
```

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [{
          "type": "command",
          "command": "bash scripts/run_tests_on_edit.sh"
        }]
      }
    ]
  }
}
```

---

## Integración con CI/CD

### Usar Claude Code en GitHub Actions

```yaml
# .github/workflows/claude-review.yml
name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      - name: Run Claude Code Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # Obtener diff del PR
          git diff origin/${{ github.base_ref }}...HEAD > pr_diff.txt
          
          # Revisar con Claude Code en modo headless
          claude --print "
          Revisa este PR diff y proporciona:
          1. Resumen de los cambios (2-3 frases)
          2. Posibles problemas de seguridad
          3. Casos borde no manejados
          4. Calidad del código (1-10)
          
          DIFF:
          $(cat pr_diff.txt | head -500)
          " > review.md
          
          cat review.md

      - name: Post Review as Comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🤖 Revisión automática de Claude Code\n\n${review}`
            });
```

### Generación automática de changelog

```bash
#!/bin/bash
# scripts/generate_changelog.sh

ULTIMO_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~20")
COMMITS=$(git log $ULTIMO_TAG..HEAD --pretty=format:"%h %s" --no-merges)

if [ -z "$COMMITS" ]; then
  echo "Sin commits nuevos desde $ULTIMO_TAG"
  exit 0
fi

claude --print "
Genera una sección de CHANGELOG.md en formato Keep a Changelog para estos commits.
Agrupa por tipo: Added, Changed, Fixed, Security. Usa bullet points.
Fecha: $(date +%Y-%m-%d)

COMMITS:
$COMMITS
" >> CHANGELOG.md

echo "✅ CHANGELOG.md actualizado"
```

---

## Configuración completa de producción

```json
{
  "model": "claude-sonnet-4-6",
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "bash scripts/audit_bash.sh"}]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {"type": "command", "command": "bash scripts/lint_on_edit.sh"},
          {"type": "command", "command": "bash scripts/run_tests_on_edit.sh"}
        ]
      }
    ],
    "Notification": [
      {
        "matcher": ".*",
        "hooks": [{"type": "command", "command": "bash scripts/notify.sh"}]
      }
    ],
    "Stop": [
      {
        "matcher": ".*",
        "hooks": [{"type": "command", "command": "bash scripts/auto_commit.sh"}]
      }
    ]
  },
  "permissions": {
    "allow": ["Bash(git:*)", "Bash(pytest:*)", "Bash(ruff:*)", "Bash(npm run:*)"],
    "deny": ["Bash(rm -rf:*)", "Bash(git push --force:*)"]
  }
}
```

## Recursos

- [Notebook interactivo](../notebooks/claude-code/03-hooks-automatizacion.ipynb)
- [Claude Code Hooks — documentación](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Claude Code Settings — referencia](https://docs.anthropic.com/en/docs/claude-code/settings)
