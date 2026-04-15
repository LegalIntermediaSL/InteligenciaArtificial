# Bloque 8 — Multimodalidad: Visión, Imágenes y Voz

> **Bloque:** Multimodalidad · **Nivel:** Práctico · **Tiempo estimado:** 3–4 horas en total

Este bloque cubre las capacidades multimodales de los modelos de IA modernos: cómo procesar imágenes, generar imágenes y trabajar con audio y voz. Al terminar podrás construir pipelines que combinan texto, imágenes y voz en una sola aplicación.

---

## Requisitos previos

Antes de empezar instala las dependencias necesarias:

```bash
pip install anthropic openai pillow requests openai-whisper pydub pyaudio diffusers transformers torch
```

También necesitarás:
- Una clave de API de Anthropic (`ANTHROPIC_API_KEY`)
- Una clave de API de OpenAI (`OPENAI_API_KEY`)
- Python 3.9 o superior

---

## Tutoriales del bloque

### [01 — Visión artificial con LLMs](./01-vision-llms.md)

Aprende a enviar imágenes a modelos de lenguaje y obtener análisis detallados. Cubrimos análisis de imágenes locales y remotas, extracción de datos estructurados desde facturas, comparación de imágenes y transcripción de documentos escaneados.

**Conceptos clave:** modelos multimodales, codificación base64, vision API, extracción estructurada con Pydantic.

---

### [02 — Generación de imágenes con IA](./02-generacion-imagenes.md)

Genera imágenes a partir de texto con DALL-E 3 y Stable Diffusion. Aprende prompt engineering para imágenes, edición con inpainting y cómo integrar generación de imágenes en pipelines con LLMs.

**Conceptos clave:** DALL-E 3, Stable Diffusion, prompt engineering visual, inpainting, pipelines LLM + generación.

---

### [03 — Voz e IA](./03-voz-ia.md)

Transcribe audio con Whisper, sintetiza voz con la API de OpenAI y construye un pipeline completo de voz a voz. Caso práctico: asistente de voz para resumir reuniones.

**Conceptos clave:** STT (Speech-to-Text), TTS (Text-to-Speech), Whisper, pipelines de voz, transcripción de reuniones.

---

## Estructura del bloque

```
tutoriales/multimodalidad/
├── README.md               ← Este archivo (índice del bloque)
├── 01-vision-llms.md       ← Análisis de imágenes con LLMs
├── 02-generacion-imagenes.md ← Generación de imágenes con IA
└── 03-voz-ia.md            ← Voz: transcripción, síntesis y pipelines
```
