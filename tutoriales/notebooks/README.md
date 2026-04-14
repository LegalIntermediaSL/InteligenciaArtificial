# Cuadernos Jupyter Notebook

Esta carpeta contiene los cuadernos interactivos del proyecto, organizados por bloque temático.

Los notebooks son el complemento práctico de los tutoriales en Markdown: permiten ejecutar código, ver resultados en tiempo real y experimentar directamente con los conceptos explicados.

## Estructura

```
notebooks/
├── fundamentos/          # Notebooks del bloque Fundamentos
├── llms/                 # Notebooks del bloque LLMs
├── apis/                 # Notebooks del bloque APIs de IA
├── python-para-ia/       # Notebooks del bloque Python para IA
└── casos-de-uso/         # Notebooks de proyectos prácticos
```

## Cómo usar los notebooks

### Opción 1 — JupyterLab (recomendado)

```bash
pip install jupyterlab
jupyter lab
```

### Opción 2 — VS Code

Instala la extensión **Jupyter** de Microsoft. Abre cualquier `.ipynb` directamente.

### Opción 3 — Google Colab

Sube el `.ipynb` a [colab.research.google.com](https://colab.research.google.com) o ábrelo directamente desde GitHub.

## Referencia de Markdown para Jupyter

Las celdas de texto de los notebooks usan sintaxis Markdown. Consulta esta cheatsheet para usarla con soltura:

- [Markdown for Jupyter Notebooks — IBM Watson Studio Cheatsheet](https://www.ibm.com/docs/en/watson-studio-local/1.2.3?topic=notebooks-markdown-jupyter-cheatsheet)

## Convenciones

- Los notebooks se nombran igual que su tutorial de referencia: `01-que-es-la-ia.ipynb`
- Cada notebook incluye una celda inicial con los requisitos e instrucciones de instalación
- Se recomienda ejecutar las celdas en orden secuencial

---

> Los notebooks se irán añadiendo a medida que se desarrolle cada bloque. Consulta [TODO.md](../../TODO.md) para ver el estado de cada uno.
