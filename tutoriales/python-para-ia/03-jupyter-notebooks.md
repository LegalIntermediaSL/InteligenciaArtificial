# 03 — Jupyter Notebooks

> **Bloque:** Python para IA · **Nivel:** Introductorio · **Tiempo estimado:** 20 min

---

## Índice

1. [¿Qué es Jupyter?](#1-qué-es-jupyter)
2. [Instalación y arranque](#2-instalación-y-arranque)
3. [Interfaz y tipos de celdas](#3-interfaz-y-tipos-de-celdas)
4. [Atajos de teclado esenciales](#4-atajos-de-teclado-esenciales)
5. [Magics de Jupyter](#5-magics-de-jupyter)
6. [Buenas prácticas](#6-buenas-prácticas)
7. [Extensiones útiles](#7-extensiones-útiles)
8. [Alternativas a Jupyter](#8-alternativas-a-jupyter)

---

## 1. ¿Qué es Jupyter?

**Jupyter Notebook** es un entorno de programación interactivo basado en web. Combina en un mismo documento:

- **Código** ejecutable (Python, R, Julia...)
- **Resultados** inmediatos (texto, tablas, gráficos)
- **Texto explicativo** en Markdown
- **Ecuaciones** en LaTeX

Es el formato estándar para exploración de datos, experimentación con modelos y presentación de resultados en IA.

```
Ventajas:
✅ Feedback inmediato — ejecutas una celda y ves el resultado al momento
✅ Narrativa integrada — código + explicación en el mismo sitio
✅ Exploración — puedes re-ejecutar celdas individuales
✅ Visualizaciones inline — los gráficos aparecen en el documento
✅ Compartible — .ipynb es un JSON que se puede compartir y renderizar en GitHub

Desventajas:
⚠️ Difícil de versionar con git (el JSON incluye outputs)
⚠️ Estado oculto — el orden de ejecución puede generar bugs sutiles
⚠️ No apto para producción — los notebooks no son módulos importables
```

---

## 2. Instalación y arranque

### JupyterLab (recomendado)

JupyterLab es la versión moderna, con interfaz de pestañas y más funcionalidades.

```bash
pip install jupyterlab

# Arrancar
jupyter lab
```

Se abre automáticamente en `http://localhost:8888`

### Jupyter Notebook (clásico)

```bash
pip install notebook

jupyter notebook
```

### Con conda

```bash
conda install -c conda-forge jupyterlab
jupyter lab
```

### Kernel de Python

El **kernel** es el proceso Python que ejecuta el código. Al abrir un notebook, se conecta a un kernel. Si el kernel muere o se reinicia, se pierde el estado (variables, imports, etc.).

```bash
# Instalar el kernel del entorno actual
pip install ipykernel
python -m ipykernel install --user --name mi-entorno --display-name "Python (mi-entorno)"
```

---

## 3. Interfaz y tipos de celdas

### Tipos de celdas

**Celda de código:**
```python
# Esta celda se ejecuta con Shift+Enter
import numpy as np
x = np.linspace(0, 2*np.pi, 100)
print(f"Array de {len(x)} puntos entre 0 y 2π")
```

**Celda Markdown:**
```markdown
## Análisis exploratorio

En esta sección analizamos la distribución de los datos.

- Primero cargamos el dataset
- Luego visualizamos las distribuciones
- Finalmente identificamos outliers
```

**Celda Raw:**
Texto sin procesar, útil para documentación que no quieres que Jupyter interprete.

### Modos de edición

| Modo | Color borde | Descripción |
|---|---|---|
| **Comando** (azul) | Azul | Navegar entre celdas, ejecutar comandos |
| **Edición** (verde) | Verde | Escribir dentro de la celda |

- `Esc` → Modo comando
- `Enter` → Modo edición

---

## 4. Atajos de teclado esenciales

### En modo comando (`Esc`)

| Atajo | Acción |
|---|---|
| `Shift + Enter` | Ejecutar celda y pasar a la siguiente |
| `Ctrl + Enter` | Ejecutar celda (sin moverse) |
| `Alt + Enter` | Ejecutar celda e insertar nueva debajo |
| `A` | Insertar celda **arriba** |
| `B` | Insertar celda **abajo** |
| `D D` | Eliminar celda |
| `Z` | Deshacer eliminación |
| `M` | Convertir a Markdown |
| `Y` | Convertir a código |
| `L` | Mostrar/ocultar números de línea |
| `Shift + M` | Combinar celdas seleccionadas |
| `Ctrl + Shift + -` | Dividir celda en el cursor |
| `00` | Reiniciar kernel |
| `H` | Ver todos los atajos |

### En modo edición (`Enter`)

| Atajo | Acción |
|---|---|
| `Tab` | Autocompletar |
| `Shift + Tab` | Mostrar docstring/ayuda |
| `Ctrl + /` | Comentar/descomentar línea |
| `Ctrl + Z` | Deshacer |
| `Ctrl + D` | Eliminar línea |

---

## 5. Magics de Jupyter

Los **magics** son comandos especiales que empiezan por `%` (línea) o `%%` (celda completa).

### Magics de línea (`%`)

```python
# Medir tiempo de ejecución
%time sum(range(1_000_000))

# Medir tiempo promedio (múltiples ejecuciones)
%timeit np.random.randn(1000, 1000)

# Ver variables en memoria
%who
%whos  # Con tipos y valores

# Historial de comandos
%history

# Ejecutar un script externo
%run mi_script.py

# Cargar contenido de fichero en una celda
%load utils.py

# Variables de entorno
%env ANTHROPIC_API_KEY=sk-ant-...

# Instalar paquetes sin salir del notebook
%pip install tqdm

# Ver el código fuente de una función
%pdef np.array
```

### Magics de celda (`%%`)

```python
%%time
# Medir el tiempo de toda la celda
resultado = []
for i in range(1_000_000):
    resultado.append(i ** 2)
```

```python
%%writefile utils.py
# Escribe el contenido de esta celda en un fichero
def saludar(nombre):
    return f"Hola, {nombre}!"
```

```python
%%bash
# Ejecutar comandos de shell
echo "Hola desde bash"
ls -la
pip list | grep numpy
```

```python
%%html
<!-- Renderizar HTML directamente -->
<h2 style="color: blue">Título en azul</h2>
<p>Párrafo con <strong>negrita</strong></p>
```

### Matplotlib inline

```python
# Al inicio del notebook, para que los gráficos aparezcan inline
%matplotlib inline

# O para gráficos interactivos
%matplotlib widget
```

---

## 6. Buenas prácticas

### Estructura recomendada de un notebook

```
1. Título y descripción (Markdown)
2. Imports y configuración
3. Carga de datos
4. Exploración y limpieza
5. Análisis / experimentos
6. Resultados y conclusiones
```

### Convenciones

```python
# ─── Celda 1: Imports (siempre al principio) ───────────────────────────────
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─── Celda 2: Configuración ────────────────────────────────────────────────
MODELO = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TEMPERATURA = 0.7
RUTA_DATOS = Path("datos/")

client = anthropic.Anthropic()
```

### Evitar el estado oculto

El mayor problema de los notebooks es ejecutar celdas fuera de orden. Solución:

```python
# Al terminar, siempre verifica que el notebook se ejecuta limpio de arriba a abajo:
# Kernel → Restart & Run All
```

```python
# Indica explícitamente las dependencias al principio de cada sección:
# Esta sección requiere que se hayan ejecutado las celdas 1-3
```

### Gestión de outputs en Git

Los notebooks guardan los outputs en el JSON, lo que ensucia los diffs. Opciones:

```bash
# Opción 1: limpiar outputs antes de commitear
jupyter nbconvert --clear-output --inplace mi_notebook.ipynb

# Opción 2: usar nbstripout (limpia automáticamente al hacer commit)
pip install nbstripout
nbstripout --install  # Configura el hook de git
```

### Comentar el código

```python
# ── Llamada a la API ──────────────────────────────────────────────────────
# Enviamos el prompt con temperatura baja porque queremos respuestas factuales
respuesta = client.messages.create(
    model=MODELO,
    max_tokens=MAX_TOKENS,
    temperature=0.0,  # Determinista para reproducibilidad
    messages=[{"role": "user", "content": prompt}]
)

# Extraemos solo el texto de la respuesta
texto = respuesta.content[0].text
```

---

## 7. Extensiones útiles

### Para JupyterLab

```bash
# Tabla de contenidos automática
pip install jupyterlab-toc

# Variables explorer (ver variables como tabla)
pip install lckr-jupyterlab-variableinspector

# Formateo automático de código
pip install jupyterlab-code-formatter black isort

# Gráficos interactivos
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### nbformat — manipular notebooks desde Python

```python
import nbformat

# Leer un notebook
with open("mi_notebook.ipynb", "r") as f:
    nb = nbformat.read(f, as_version=4)

# Ver las celdas
for celda in nb.cells:
    print(f"Tipo: {celda.cell_type}")
    print(f"Código: {celda.source[:100]}")
    print("---")
```

---

## 8. Alternativas a Jupyter

| Herramienta | Descripción | Cuándo usarla |
|---|---|---|
| **VS Code + extensión Jupyter** | Notebooks dentro de VS Code | Si ya usas VS Code; mejor integración con git |
| **Google Colab** | Notebooks en la nube, gratis | Cuando necesitas GPU gratuita o compartir fácilmente |
| **Kaggle Notebooks** | Similar a Colab, con datasets integrados | Competiciones y datasets de Kaggle |
| **Deepnote** | Colaborativo en tiempo real | Trabajo en equipo |
| **Marimo** | Alternativa reactiva a Jupyter | Notebooks reproducibles sin estado oculto |

### Google Colab: notas importantes

```python
# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Instalar paquetes
!pip install anthropic

# Usar secrets (equivalente a .env)
from google.colab import userdata
api_key = userdata.get('ANTHROPIC_API_KEY')

# Verificar si hay GPU disponible
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

**Anterior:** [02 — Librerías esenciales](./02-librerias-esenciales.md) · **Siguiente bloque:** [Casos de uso](../casos-de-uso/)
