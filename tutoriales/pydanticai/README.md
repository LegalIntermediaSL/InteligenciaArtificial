# Bloque 24 — PydanticAI y frameworks modernos de agentes

Frameworks de nueva generación para construir agentes con tipado estricto, testabilidad y producción en mente.

---

## Contenido

| Artículo | Descripción | Nivel |
|---|---|---|
| [01 — PydanticAI: introducción](01-pydanticai-intro.md) | Agentes tipados, dependencias, herramientas y streaming | Intermedio |
| [02 — PydanticAI avanzado](02-pydanticai-avanzado.md) | Testing, validación, multi-agente y despliegue | Avanzado |
| [03 — Mastra.ai](03-mastra-ai.md) | Framework TypeScript para agentes: workflows, memoria y tools | Avanzado |
| [04 — Comparativa de frameworks](04-comparativa-frameworks.md) | PydanticAI vs LangGraph vs CrewAI vs AutoGen | Referencia |

---

## ¿Por qué PydanticAI?

- **Tipado estricto**: los agentes están definidos con tipos Python. El IDE ayuda, mypy valida.
- **Testabilidad**: modo `TestModel` para tests sin coste de API.
- **Agnóstico de proveedor**: Claude, GPT-4o, Gemini, Ollama con la misma interfaz.
- **Streaming nativo**: streaming de texto y datos estructurados.
- **Dependencias inyectadas**: patrón de inversión de dependencias para producción.

---

## Requisitos

```bash
pip install pydantic-ai
pip install pydantic-ai[anthropic]  # con soporte Claude
pip install pydantic-ai[openai]     # con soporte OpenAI
```

---

## Tiempo estimado

4-6 horas para completar los 4 artículos + notebooks.
