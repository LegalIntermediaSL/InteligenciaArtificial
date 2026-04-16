# Bloque 9 — Agentes Avanzados

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 4–6 horas

---

Este bloque cubre las técnicas más avanzadas del ecosistema de agentes de IA: desde orquestar múltiples agentes especializados hasta controlar interfaces gráficas, pasando por el protocolo estándar de Anthropic para conectar LLMs con herramientas externas y por la implementación de memoria persistente entre sesiones.

## Requisitos previos

- Haber completado el Bloque 8 (Agentes con herramientas)
- Python 3.10 o superior
- Una clave de API de Anthropic válida

## Instalación de dependencias

```bash
pip install anthropic crewai mcp chromadb sentence-transformers pyautogui playwright
playwright install chromium
```

---

## Tutoriales de este bloque

### [01 — Sistemas Multi-Agente](./01-multi-agente.md)

Aprende a diseñar y construir sistemas donde varios agentes especializados colaboran para resolver tareas que un único agente no puede manejar eficientemente. Cubre arquitecturas orquestador-trabajadores, implementación manual con la API de Anthropic, y uso del framework CrewAI.

**Lo que aprenderás:** arquitecturas multi-agente, delegación de tareas, comunicación asíncrona entre agentes, y cuándo tiene sentido este enfoque.

---

### [02 — Model Context Protocol (MCP)](./02-model-context-protocol.md)

Domina el protocolo abierto de Anthropic que permite conectar cualquier LLM con cualquier fuente de datos o herramienta de forma estandarizada. Aprende a construir servidores MCP propios y a conectarlos con Claude Desktop.

**Lo que aprenderás:** arquitectura cliente-servidor de MCP, creación de herramientas y recursos con el SDK de Python, configuración de Claude Desktop, y casos de uso reales.

---

### [03 — Computer Use](./03-computer-use.md)

Explora la capacidad de Claude para ver y controlar interfaces gráficas: hacer capturas de pantalla, hacer clic en elementos, escribir texto y desplazarse por páginas. Incluye dos casos prácticos: automatización de formularios y scraping visual.

**Lo que aprenderás:** el ciclo de acción de Computer Use, implementación del bucle de control, automatización de tareas en el navegador, y consideraciones de seguridad.

---

### [04 — Memoria a Largo Plazo](./04-memoria-largo-plazo.md)

Resuelve el problema fundamental de la falta de memoria persistente en los LLMs. Implementa desde cero un sistema completo que combina memoria episódica en JSON, memoria semántica con embeddings vectoriales en ChromaDB, y compresión automática de conversaciones largas.

**Lo que aprenderás:** tipos de memoria, almacenamiento con ChromaDB, recuperación por similitud semántica, resumen automático de contexto, y una clase `MemoryManager` lista para producción.

---

### [05 — AutoGen: Conversaciones Multi-Agente](./05-autogen.md)

Framework de Microsoft Research para construir sistemas donde múltiples agentes LLM colaboran mediante conversaciones estructuradas. Cubre el patrón asistente-proxy, GroupChat con múltiples roles y orquestación avanzada.

**Lo que aprenderás:** AssistantAgent, UserProxyAgent, GroupChat, patrones de orquestación (secuencial, debate, consenso) y herramientas integradas.

---

### [06 — A2A Protocol: Agent-to-Agent](./06-a2a-protocol.md)

Protocolo abierto propuesto por Google en 2025 para la comunicación estandarizada entre agentes de IA de diferentes sistemas. Implementa un servidor y cliente A2A con FastAPI y orquestación multi-agente.

**Lo que aprenderás:** Agent Card, Task lifecycle, servidor A2A con streaming SSE, cliente A2A y patrones de orquestación multi-proveedor.

---

**Siguiente bloque:** [Bloque 10 — Casos de uso avanzados](../casos-de-uso-avanzados/README.md)
