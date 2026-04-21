# Bloque 32: IA en el Sector Legal y Cumplimiento Normativo

> **Aviso legal:** Los ejemplos de este bloque son ilustrativos y tienen finalidad exclusivamente educativa. No constituyen asesoramiento jurídico. Para cualquier decisión legal, consulta siempre con un abogado colegiado.

La inteligencia artificial está transformando la práctica jurídica: desde el análisis masivo de contratos hasta la monitorización automática de cambios normativos. Este bloque cubre las aplicaciones más relevantes de Claude en el ámbito legal, con código funcional y patrones reutilizables en producción.

---

## Artículos del bloque

| # | Artículo | Descripción | Notebook |
|---|----------|-------------|---------|
| 01 | [Análisis Automático de Contratos](01-analisis-contratos.md) | Extracción de cláusulas, scoring de riesgo y diff semántico entre versiones | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/01-analisis-contratos.ipynb) |
| 02 | [Due Diligence Asistida por IA](02-due-diligence-ia.md) | Procesamiento de datarooms, detección de red flags y Q&A con citaciones | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/02-due-diligence-ia.ipynb) |
| 03 | [Cumplimiento Normativo con IA](03-cumplimiento-normativo.md) | Monitorización de GDPR, EU AI Act y LSD; alertas automáticas de cambios | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/03-cumplimiento-normativo.ipynb) |
| 04 | [Redacción de Documentos Legales](04-redaccion-documentos-legales.md) | NDAs, contratos de servicios, políticas de privacidad con revisión humana | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/ia-legal/04-redaccion-documentos-legales.ipynb) |

---

## ¿Qué aprenderás en este bloque?

- Diseñar pipelines de análisis legal con **schemas Pydantic** y **tool use** estructurado
- Procesar carteras de decenas o centenares de documentos de forma automatizada
- Implementar **RAG con citaciones** (Citations API) para Q&A sobre documentos legales
- Construir sistemas de **monitorización normativa** que detectan cambios relevantes en GDPR, EU AI Act y Ley de Servicios Digitales
- Generar y revisar documentos legales con flujos **human-in-the-loop**
- Exportar resultados en formatos utilizables por equipos jurídicos (Markdown estructurado, tablas de semáforos, diffs)

---

## Requisitos

### Instalación de dependencias

```bash
pip install anthropic "anthropic[bedrock]" pydantic pymupdf
```

### Variables de entorno

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Versión de Python

Los ejemplos requieren **Python 3.10 o superior** y usan `claude-sonnet-4-6` como modelo base. Los casos legales requieren mayor capacidad de razonamiento que Haiku.

---

## Estructura del bloque

```
tutoriales/
├── ia-legal/
│   ├── README.md                          ← Este archivo
│   ├── 01-analisis-contratos.md
│   ├── 02-due-diligence-ia.md
│   ├── 03-cumplimiento-normativo.md
│   └── 04-redaccion-documentos-legales.md
└── notebooks/
    └── ia-legal/
        ├── 01-analisis-contratos.ipynb
        ├── 02-due-diligence-ia.ipynb
        ├── 03-cumplimiento-normativo.ipynb
        └── 04-redaccion-documentos-legales.ipynb
```

---

## Consideraciones éticas y de responsabilidad

1. **Revisión humana obligatoria:** Toda salida de la IA debe ser revisada por un profesional del derecho antes de cualquier uso real.
2. **Confidencialidad:** Los documentos legales contienen información sensible. Evalúa si usar la API en la nube cumple con tus obligaciones de confidencialidad y protección de datos.
3. **Jurisdicción:** Las plantillas y ejemplos están orientados al derecho español y europeo. Adapta siempre a la jurisdicción aplicable.
4. **Alucinaciones:** Los LLMs pueden generar texto plausible pero incorrecto. Verifica referencias normativas, citas legales y articulado concreto.

---

*Bloque 32 — Repositorio de Inteligencia Artificial en Español*
