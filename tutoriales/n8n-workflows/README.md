# Bloque 22 — Workflows con n8n y Claude

Automatización de procesos de negocio combinando n8n como orquestador
visual con Claude como motor de razonamiento e inteligencia.

## Tutoriales

| # | Título | Descripción |
|---|--------|-------------|
| 01 | [Introducción a n8n](01-introduccion-n8n.md) | Instalación, conceptos clave y primer workflow con IA |
| 02 | [Workflows de documentos](02-workflows-documentos.md) | Procesamiento de facturas, contratos y PDFs con Claude |
| 03 | [Workflows de negocio](03-workflows-negocio.md) | CRM, email, Slack bot y reportes automáticos |
| 04 | [Workflows avanzados](04-workflows-avanzados.md) | Sub-workflows, webhooks, errores y despliegue en producción |

## ¿Para quién es este bloque?

- **Operaciones y RevOps** que quieren automatizar procesos sin código complejo
- **Fundadores** que necesitan automatizar flujos antes de contratar equipo
- **Desarrolladores** que quieren combinar código Python con n8n como orquestador
- **Equipos de producto** que integran IA en flujos existentes (CRM, email, Slack)

## Requisitos

```bash
# Docker (recomendado para desarrollo local)
docker run -it --rm \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# O instalar globalmente
npm install n8n -g
n8n start
```

Acceder en: `http://localhost:5678`
