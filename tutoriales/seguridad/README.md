# Bloque 12 — Seguridad en Sistemas de IA

> **Bloque:** Seguridad en IA · **Nivel:** Avanzado · **Tiempo estimado:** 4 h (bloque completo)

---

## Introducción

La seguridad en sistemas de Inteligencia Artificial no es un añadido opcional: es una responsabilidad fundamental cuando desplegamos modelos de lenguaje en producción. Este bloque cubre las amenazas reales que enfrentan los sistemas basados en LLMs y, sobre todo, cómo **defenderse** de ellas de forma práctica.

> **Nota importante sobre el enfoque de este bloque:**
> Todo el contenido es **educativo y defensivo**. Cuando se muestran ejemplos de ataques (prompt injection, jailbreaking, exfiltración de datos), el objetivo es que el lector comprenda exactamente qué debe prevenir y cómo detectarlo. No se proporciona ninguna guía para explotar sistemas ajenos. La ética profesional en IA exige construir sistemas seguros y responsables.

---

## Requisitos previos

- Conocimientos básicos de Python y la API de Anthropic (Bloques 1–4)
- Haber completado el Bloque 11 (Producción) es recomendable
- Cuenta activa en Anthropic con clave API

## Instalación de dependencias

```bash
pip install anthropic presidio-analyzer presidio-anonymizer fastapi uvicorn pandas
```

Para Presidio con soporte de español:

```bash
python -m spacy download es_core_news_md
```

---

## Tutoriales del bloque

| # | Tutorial | Tema principal | Tiempo |
|---|----------|---------------|--------|
| 01 | [Prompt Injection](./01-prompt-injection.md) | Ataques de inyección y defensa con guardrails | 60 min |
| 02 | [Jailbreaking y Guardrails](./02-jailbreaking-guardrails.md) | Técnicas de evasión y sistemas de moderación | 60 min |
| 03 | [Datos Sensibles y PII](./03-datos-sensibles-pii.md) | Detección y anonimización de información personal | 60 min |
| 04 | [Auditoría y Trazabilidad](./04-auditoria-seguridad.md) | Logging seguro, anomalías y dashboards | 60 min |

---

## Mapa conceptual del bloque

```
Amenazas externas                  Protección de datos              Trazabilidad
┌─────────────────┐               ┌──────────────────┐            ┌─────────────┐
│ Prompt injection│               │  Detección PII   │            │ Audit logs  │
│ Jailbreaking    │──► Guardrails─│  Anonimización   │──► Sistema─│ Anomalías   │
│ Indirect inject │               │  Logs seguros    │   seguro   │ Alertas     │
└─────────────────┘               └──────────────────┘            └─────────────┘
```

---

**Siguiente:** [01 — Prompt Injection](./01-prompt-injection.md)
