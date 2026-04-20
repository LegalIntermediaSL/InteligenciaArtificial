# Flujos de desarrollo con IA: Claude Code, Cursor y Copilot

## Comparativa de herramientas

| | Claude Code | Cursor | GitHub Copilot |
|---|---|---|---|
| **Interfaz** | Terminal (REPL) | IDE propio (fork VS Code) | Plugin en cualquier IDE |
| **Modelo base** | Claude (Anthropic) | Claude / GPT-4 / Gemini | GPT-4o (OpenAI) |
| **Contexto de repo** | ✅ Completo (lee todo) | ✅ Indexación semántica | ⚠️ Ventana limitada |
| **Ejecución de código** | ✅ Bash nativo | ⚠️ Terminal integrado | ❌ Solo sugerencias |
| **MCP / Plugins** | ✅ MCP nativo | ⚠️ MCP experimental | ❌ No |
| **Hooks / CI** | ✅ Hooks de ciclo de vida | ❌ | ❌ |
| **Precio (2026)** | $20/mes (Pro) o API | $20/mes | $10-19/mes |
| **Mejor para** | Tareas complejas, agénticas | Edición rápida en IDE | Autocompletado inline |

**Regla general:**
- **Claude Code**: tareas de múltiples pasos, refactorizaciones grandes, exploración de repos nuevos
- **Cursor**: edición rápida dentro de un archivo, tab completion IA
- **Copilot**: autocompletado inline mientras escribes

---

## Flujo 1: Feature completa de principio a fin

Cómo implementar una feature nueva usando Claude Code en cada fase:

### Fase 1: Entender el contexto

```
> Necesito implementar un sistema de notificaciones por email.
  Antes de escribir código, explícame:
  - ¿Existe algún sistema de emails ya en el proyecto?
  - ¿Qué librerías de email se usan?
  - ¿Dónde deberían ir los nuevos archivos según la arquitectura actual?
```

### Fase 2: Diseñar antes de implementar

```
> Propón el diseño técnico para el sistema de notificaciones:
  - Modelo de datos (qué campos necesita la tabla notifications)
  - Interfaz de la clase NotificationService
  - Cómo se integra con el flujo de registro de usuario existente
  
  No escribas código todavía. Primero quiero revisar el diseño.
```

### Fase 3: Implementar por capas

```
> Perfecto, el diseño está aprobado. Ahora implementa en este orden:
  1. Primero el modelo de datos (migration + modelo SQLAlchemy)
  2. Luego el servicio NotificationService
  3. Finalmente el endpoint POST /api/notifications

  Implementa uno a uno y espera mi confirmación antes del siguiente.
```

### Fase 4: Tests

```
> Ahora escribe los tests para NotificationService:
  - Test de envío exitoso
  - Test cuando el email es inválido
  - Test cuando el servidor SMTP falla
  - Test de rate limiting (máx 3 emails/minuto por usuario)
  
  Usa mocks para el SMTP. Sigue el estilo de los tests existentes en tests/
```

### Fase 5: Documentación y PR

```
> Genera el mensaje de commit para estos cambios.
  Luego actualiza el README con la sección de notificaciones.
  Finalmente, /review para que hagas la revisión final del PR.
```

---

## Flujo 2: Debug de producción

```bash
# Situación: error en producción, logs del servidor
tail -f /var/log/app/error.log | claude "Analiza estos errores en tiempo real y dime cuándo aparece un patrón preocupante"
```

```
# Flujo de debug estructurado:

> Tenemos este error en producción desde las 14:32 UTC:
  [pegar stack trace]
  
  El error empezó justo después de desplegar la versión 2.4.1.
  Busca en el historial de git qué cambió entre 2.4.0 y 2.4.1
  que podría causar esto.

> Encontrado el cambio en billing.py línea 234. ¿Puedes reproducir
  el problema con un test? Luego escribe el fix mínimo necesario.

> El fix parece correcto. Escribe también un test de regresión para
  que este bug no vuelva a pasar desapercibido.
```

---

## Flujo 3: Refactorización segura

Refactorizar sin romper nada requiere disciplina. Claude Code lo gestiona así:

```
# Paso 1: Análisis sin tocar código
> Analiza el módulo src/legacy/payment_processor.py.
  Lista todos los problemas que tiene sin modificar nada todavía:
  - Código duplicado
  - Funciones con más de una responsabilidad
  - Dependencias ocultas
  - Casos sin testear

# Paso 2: Plan de refactorización
> Propón un plan de refactorización en fases pequeñas y seguras.
  Cada fase debe poder commitarse de forma independiente
  y no romper los tests existentes.

# Paso 3: Ejecución fase a fase
> Ejecuta la Fase 1 del plan: extraer la clase PaymentValidator.
  Después de cada cambio, ejecuta pytest tests/test_payment.py
  para verificar que nada se rompe.

# Paso 4: Tests de cobertura
> Muéstrame la cobertura de tests actual para payment_processor.py
  y añade tests para las funciones que tienen menos del 80%.
```

---

## Flujo 4: Code review como revisor senior

```
# Revisar tu propio código antes de hacer push
> Actúa como un senior engineer revisando mi código.
  Sé crítico. Busca en los archivos modificados desde el último commit:
  1. Problemas de seguridad (inyección SQL, XSS, credenciales expuestas)
  2. Problemas de rendimiento (N+1 queries, loops ineficientes)
  3. Manejo de errores incompleto
  4. Código que no sigue las convenciones del proyecto

> Para cada problema encontrado, muéstrame el código problemático
  y una solución concreta.
```

---

## Flujo 5: Onboarding en un proyecto nuevo

Cuando entras en un proyecto que no conoces:

```
# Sesión 1: mapa del territorio
> Soy nuevo en este proyecto. Dame un tour de la arquitectura:
  - Qué hace el proyecto y para quién
  - Estructura de carpetas y propósito de cada una
  - Stack tecnológico y versiones
  - Cómo correr el proyecto en local
  - Cómo ejecutar los tests

# Sesión 2: entender el dominio
> Explícame el flujo completo de un pedido en este e-commerce,
  desde que el usuario hace clic en "Comprar" hasta que recibe el email
  de confirmación. Traza el código real, archivo por archivo.

# Sesión 3: primera contribución guiada
> Quiero hacer mi primera contribución. Hay un bug reportado en el issue #234.
  Ayúdame a entender el bug, encontrar el código responsable y escribir el fix.
```

---

## Flujo 6: Generación de documentación técnica

```
# Documentar una API existente
> Genera la documentación OpenAPI completa para todos los endpoints
  en src/api/. Para cada endpoint incluye:
  - Descripción en español
  - Parámetros con tipos y ejemplos
  - Posibles respuestas (200, 400, 401, 404, 500)
  - Ejemplo de curl

# ADRs (Architecture Decision Records)
> Crea un ADR para la decisión de usar ChromaDB como base de datos
  vectorial en lugar de pgvector. Incluye: contexto, opciones consideradas,
  decisión tomada, consecuencias (positivas y negativas).

# Documentación de onboarding
> Actualiza el CONTRIBUTING.md con el flujo de trabajo actual del equipo:
  - Cómo crear una feature branch
  - Cómo ejecutar los tests antes de hacer PR
  - Convenciones de commits
  - Proceso de revisión y merge
```

---

## Patrones y anti-patrones

### ✅ Patrones que funcionan

```
CONTEXTO PRIMERO
"En el contexto del módulo de facturación, donde usamos el patrón Repository..."

OBJETIVOS, NO IMPLEMENTACIONES
"Necesito que los usuarios puedan filtrar facturas por fecha" (no "escribe un endpoint GET /api/facturas?desde=...")

PASOS PEQUEÑOS Y VERIFICABLES  
"Implementa solo el modelo de datos. Luego esperaré los tests antes de continuar."

FEEDBACK EXPLÍCITO
"Esto no me convence porque viola el principio de single responsibility. Inténtalo de otra forma."
```

### ❌ Anti-patrones comunes

```
DEMASIADO VAGO
"Arregla los bugs" → Claude no sabe qué bugs ni dónde

SIN CONTEXTO DE DOMINIO
"Añade validación" → ¿Qué validación? ¿Dónde? ¿Para qué?

TAREAS GIGANTES EN UN SOLO PASO
"Reescribe todo el módulo de auth" → mejor dividir en 10 commits

ACEPTAR SIN REVISAR
No commitear código de Claude sin leer el diff completo
```

---

## Configuración de CLAUDE.md para proyectos de equipo

```markdown
# CLAUDE.md

## Para el equipo
Este archivo es la "constitución" del proyecto para Claude Code.
Actualizarlo cuando cambien las convenciones del equipo.
Commitear con el mismo proceso que cualquier otro cambio.

## Comandos esenciales
- Test unitarios: `pytest tests/unit -v`
- Test integración: `pytest tests/integration -v --db-url=$TEST_DB`
- Linting: `ruff check . && mypy src/`
- Dev local: `docker compose up -d && uvicorn main:app --reload`
- Migrations: `alembic upgrade head`

## Arquitectura
- Patrón: Clean Architecture (domain → application → infrastructure → api)
- Nunca importar desde infrastructure en domain
- Los servicios van en application/, los repositorios en infrastructure/

## Reglas de negocio críticas
- Un cliente solo puede tener un plan activo a la vez
- Las facturas no se pueden borrar, solo anular (campo `anulada_en`)
- Los precios siempre en céntimos (int) para evitar decimales flotantes

## Seguridad
- Toda entrada de usuario se valida con Pydantic antes de llegar al servicio
- Los logs no deben contener PII (emails, nombres, IPs de clientes)
- Usar parámetros en todas las queries SQL — nunca f-strings

## Proceso de PR
1. Tests pasando (CI verde)
2. Cobertura > 80% en código nuevo
3. Sin warnings de mypy
4. Review de al menos 1 compañero
```

## Recursos

- [Notebook interactivo](../notebooks/claude-code/04-flujos-desarrollo-ia.ipynb)
- [Claude Code — documentación oficial](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor — documentación](https://docs.cursor.com)
- [GitHub Copilot — documentación](https://docs.github.com/en/copilot)
