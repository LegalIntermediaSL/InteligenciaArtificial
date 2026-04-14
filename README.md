# InteligenciaArtificial

> Tutoriales, sistemas y modelos de Inteligencia Artificial — por [LegalIntermediaSL](https://github.com/LegalIntermediaSL)

---

## Objetivo

Este repositorio es un recurso de aprendizaje y referencia sobre Inteligencia Artificial, con especial enfoque en su aplicación práctica en entornos profesionales y empresariales.

El proyecto nace con la vocación de ser una guía progresiva: desde los conceptos más básicos hasta la construcción de sistemas reales con IA. Está dirigido tanto a perfiles técnicos que quieren iniciarse en IA como a equipos que buscan integrar estas tecnologías en sus procesos.

---

## ¿Qué encontrarás aquí?

| Bloque | Descripción |
|---|---|
| **Fundamentos** | Conceptos clave de IA, ML y DL explicados de forma clara |
| **LLMs** | Cómo funcionan los modelos de lenguaje, prompt engineering, RAG, fine-tuning |
| **APIs de IA** | Integración con Anthropic (Claude), OpenAI y otros proveedores |
| **Python para IA** | Librerías esenciales, notebooks y pipelines de datos |
| **Casos de uso** | Proyectos prácticos aplicados: chatbots, clasificación, resumen, extracción |

---

## Estructura del repositorio

```
InteligenciaArtificial/
├── tutoriales/
│   ├── fundamentos/          # Conceptos base de IA
│   ├── llms/                 # Modelos de lenguaje (LLMs)
│   ├── apis/                 # Uso de APIs de IA (Claude, OpenAI, etc.)
│   ├── python-para-ia/       # Python aplicado a IA
│   └── casos-de-uso/         # Proyectos y ejemplos prácticos
├── README.md                 # Este archivo
├── CHANGELOG.md              # Historial de cambios del proyecto
├── BITACORA.md               # Diario de decisiones y aprendizajes
└── TODO.md                   # Tareas y hoja de ruta pendiente
```

Cada carpeta dentro de `tutoriales/` incluye su propio `README.md` con descripción, requisitos y orden de lectura recomendado.

---

## Hoja de ruta

### En desarrollo
- [ ] Fundamentos de IA: conceptos clave y tipos de sistemas
- [ ] ¿Qué es un LLM? Cómo funcionan los modelos de lenguaje
- [ ] Primeros pasos con la API de Anthropic (Claude)

### Planificado
- [ ] Prompt engineering: técnicas y buenas prácticas
- [ ] Fine-tuning vs RAG: cuándo usar cada uno
- [ ] Chatbot con Claude API (proyecto completo)
- [ ] Extracción de información de PDFs con IA
- [ ] Introducción a Python para IA

> Consulta [TODO.md](./TODO.md) para el detalle completo de tareas pendientes.

---

## Requisitos generales

Cada tutorial especifica sus propias dependencias, pero en general el proyecto trabaja con:

- **Python** 3.10 o superior
- **Jupyter Notebooks** (recomendado: JupyterLab o VS Code con extensión)
- **Librerías** indicadas en el `requirements.txt` de cada carpeta

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/LegalIntermediaSL/InteligenciaArtificial.git
cd InteligenciaArtificial

# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Instalar dependencias de un tutorial concreto
pip install -r tutoriales/apis/requirements.txt
```

---

## Cómo contribuir

Este repositorio es de uso interno y aprendizaje, pero las sugerencias son bienvenidas:

1. Abre un [issue](https://github.com/LegalIntermediaSL/InteligenciaArtificial/issues) describiendo la mejora o error
2. Haz un fork del repositorio
3. Crea una rama con tu aportación (`git checkout -b mejora/nombre`)
4. Abre un Pull Request

---

## Recursos externos recomendados

- [Documentación oficial de Anthropic](https://docs.anthropic.com)
- [OpenAI Cookbook](https://cookbook.openai.com)
- [fast.ai — Practical Deep Learning](https://course.fast.ai)
- [Hugging Face Learn](https://huggingface.co/learn)
- [Papers With Code](https://paperswithcode.com)

---

## Licencia

Distribuido bajo licencia **MIT** — ver [LICENSE](./LICENSE) para más detalles.

---

*Mantenido por [LegalIntermediaSL](https://github.com/LegalIntermediaSL) · Última actualización: abril 2026*
