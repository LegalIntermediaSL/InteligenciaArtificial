# Bloque 28 — Computer Use: Claude como operador de ordenador

Claude puede ver la pantalla, mover el ratón, escribir en teclado y ejecutar comandos. Este bloque explica cómo usar la Computer Use API para automatizar tareas visuales, construir agentes RPA inteligentes y hacerlo de forma segura.

## Contenido

| # | Artículo | Descripción |
|---|----------|-------------|
| 01 | [Introducción a Computer Use](01-computer-use-intro.md) | API, herramientas, bucle de control, primer agente |
| 02 | [Automatización web](02-automatizacion-web.md) | Scraping visual, formularios, navegación por SPAs |
| 03 | [Casos empresariales](03-casos-empresariales.md) | RPA inteligente: ERPs, emails, CRM, testing visual |
| 04 | [Seguridad y límites](04-seguridad-y-limites.md) | Sandboxing, mínimo privilegio, human-in-the-loop |

## Prerrequisitos

- Python 3.10+
- Docker instalado (para el entorno de ejecución)
- `pip install anthropic pillow`
- Conocimientos de agentes (Bloques 9 y 19)

## Tiempo estimado

~3 horas

## Entorno de ejecución

Computer Use requiere un entorno con display virtual. El repositorio incluye un `docker-compose.yml` de referencia con xvfb + VNC + noVNC.

## Notebooks interactivos

Cada artículo tiene su notebook en [`tutoriales/notebooks/computer-use-operator/`](../notebooks/computer-use-operator/).
