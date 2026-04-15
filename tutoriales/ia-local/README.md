# 11 — Bloque: IA Local

> **Bloque:** IA local · **Nivel:** Intermedio/Avanzado · **Tiempo estimado:** 20 min

---

## Índice

1. [01 — Ejecutar modelos locales con Ollama](./01-ollama.md)
2. [02 — Inferencia local con Hugging Face Transformers](./02-transformers-local.md)
3. [03 — IA en el navegador con Transformers.js](./03-transformers-js-navegador.md)
4. [04 — Local vs cloud: cuándo usar cada opción](./04-comparativa-local-cloud.md)

---

## Sobre este bloque

La IA local te permite ejecutar modelos de lenguaje directamente en tu hardware, sin enviar datos a servidores externos. En este bloque aprenderás tres enfoques complementarios: Ollama para despliegue rápido de modelos open-source, Hugging Face Transformers para control total sobre el pipeline de inferencia, y Transformers.js para llevar la IA directamente al navegador sin ningún backend.

---

## Tutoriales del bloque

| # | Tutorial | Herramienta principal | Nivel |
|---|----------|-----------------------|-------|
| 01 | [Ejecutar modelos locales con Ollama](./01-ollama.md) | Ollama + Python SDK | Intermedio |
| 02 | [Inferencia local con Hugging Face Transformers](./02-transformers-local.md) | transformers + torch | Intermedio |
| 03 | [IA en el navegador con Transformers.js](./03-transformers-js-navegador.md) | Transformers.js + Web Workers | Avanzado |
| 04 | [Local vs cloud: cuándo usar cada opción](./04-comparativa-local-cloud.md) | Múltiples SDKs | Avanzado |

---

## Instalación de dependencias

```bash
# Dependencias Python para los tutoriales 01 y 02
pip install ollama transformers torch sentence-transformers

# Dependencias adicionales para cuantización y RAG
pip install bitsandbytes chromadb openai

# Dependencias Node.js para el tutorial 03
npm install @huggingface/transformers
```

---

## Requisitos de hardware

Cada enfoque tiene requisitos de hardware diferentes. Usa esta tabla para decidir qué opción se adapta mejor a tu equipo:

| Opción | RAM mínima | GPU recomendada | CPU (sin GPU) | Notas |
|--------|-----------|-----------------|---------------|-------|
| **Ollama — modelos 3B** (phi3-mini, llama3.2:3b) | 8 GB RAM | Cualquier GPU con 4 GB VRAM | Funciona, lento | Ideal para portátiles |
| **Ollama — modelos 7B** (mistral, llama3.2:7b) | 16 GB RAM | GPU con 8 GB VRAM | Muy lento | Buena relación calidad/coste |
| **Ollama — modelos 13B+** (llama3:70b, qwen2:72b) | 32 GB RAM | GPU con 16 GB+ VRAM | No recomendado | Requiere hardware dedicado |
| **Transformers — modelos 4-bit** (phi-3-mini cuantizado) | 8 GB RAM | GPU con 4 GB VRAM (CUDA/MPS) | Funciona | bitsandbytes solo en CUDA |
| **Transformers — modelos completos** (phi-3-mini FP16) | 16 GB RAM | GPU con 8 GB VRAM | Lento pero funcional | |
| **Transformers.js** | 4 GB RAM | No necesaria (WebGL opcional) | Cualquier CPU moderna | Corre íntegramente en el navegador |
| **Embeddings** (sentence-transformers, all-MiniLM) | 4 GB RAM | No necesaria | Cualquier CPU | Modelos pequeños y rápidos |

### Detectar tu hardware disponible

```python
import torch

print("CUDA disponible:", torch.cuda.is_available())
print("MPS disponible (Apple Silicon):", torch.backends.mps.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
```

---

## Cuándo usar cada opción

- **Ollama**: despliegue rápido, experimentación, aplicaciones sin requisito de control fino del modelo.
- **Transformers**: cuando necesitas acceso al modelo en Python, fine-tuning, cuantización personalizada o pipelines de ML complejos.
- **Transformers.js**: cuando el procesamiento debe ocurrir en el cliente (navegador o Node.js) sin ningún backend.
- **Cloud API**: cuando necesitas máxima calidad, no tienes restricciones de privacidad y prefieres no gestionar infraestructura.

---

**Siguiente:** [01 — Ejecutar modelos locales con Ollama](./01-ollama.md)
