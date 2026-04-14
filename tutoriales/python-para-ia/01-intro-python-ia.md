# 01 — Introducción a Python para IA

> **Bloque:** Python para IA · **Nivel:** Introductorio · **Tiempo estimado:** 30 min

---

## Índice

1. [¿Por qué Python?](#1-por-qué-python)
2. [Instalación y entornos virtuales](#2-instalación-y-entornos-virtuales)
3. [Tipos de datos esenciales](#3-tipos-de-datos-esenciales)
4. [Estructuras de control](#4-estructuras-de-control)
5. [Funciones y lambdas](#5-funciones-y-lambdas)
6. [Comprehensions](#6-comprehensions)
7. [Ficheros y JSON](#7-ficheros-y-json)
8. [Patrones habituales en IA](#8-patrones-habituales-en-ia)
9. [Recursos para profundizar](#9-recursos-para-profundizar)

---

## 1. ¿Por qué Python?

Python se ha convertido en el lenguaje estándar de la IA por varias razones:

| Razón | Detalle |
|---|---|
| **Ecosistema** | NumPy, Pandas, scikit-learn, TensorFlow, PyTorch, Hugging Face... todo está en Python |
| **Legibilidad** | Sintaxis clara que permite centrarse en los algoritmos, no en el lenguaje |
| **Velocidad de prototipado** | De idea a código funcional en minutos |
| **Comunidad** | Millones de ejemplos, tutoriales y respuestas en Stack Overflow |
| **Interoperabilidad** | Se integra fácilmente con C/C++ para partes críticas de rendimiento |
| **Jupyter** | Los notebooks permiten experimentar de forma interactiva |

---

## 2. Instalación y entornos virtuales

### Instalar Python

Descarga desde [python.org](https://python.org) la versión 3.10 o superior.

Verifica la instalación:
```bash
python --version   # Python 3.11.x
pip --version
```

### Entornos virtuales

Un entorno virtual aísla las dependencias de cada proyecto. **Siempre** trabaja con entornos virtuales.

**Con venv (estándar):**
```bash
# Crear entorno
python -m venv .venv

# Activar
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Desactivar
deactivate
```

**Con conda (recomendado para IA por la gestión de dependencias nativas):**
```bash
# Crear entorno
conda create -n mi-proyecto python=3.11

# Activar
conda activate mi-proyecto

# Desactivar
conda deactivate
```

### Gestión de dependencias

```bash
# Instalar paquetes
pip install numpy pandas scikit-learn

# Guardar dependencias del proyecto
pip freeze > requirements.txt

# Instalar desde requirements.txt
pip install -r requirements.txt
```

---

## 3. Tipos de datos esenciales

### Números

```python
entero = 42
flotante = 3.14159
complejo = 2 + 3j

# Operaciones
print(10 / 3)    # 3.3333... (división real)
print(10 // 3)   # 3         (división entera)
print(10 % 3)    # 1         (módulo)
print(2 ** 10)   # 1024      (potencia)
```

### Cadenas de texto

```python
texto = "Inteligencia Artificial"

# Métodos útiles
print(texto.lower())          # "inteligencia artificial"
print(texto.upper())          # "INTELIGENCIA ARTIFICIAL"
print(texto.split())          # ["Inteligencia", "Artificial"]
print(texto.replace("I", "i")) # "inteligencia artificial"
print(len(texto))             # 23

# F-strings (la forma moderna de formatear)
nombre = "Claude"
version = 4.6
print(f"Modelo: {nombre} {version}")  # "Modelo: Claude 4.6"

# Strings multilínea
prompt = """
Eres un asistente útil.
Responde siempre en español.
"""
```

### Listas

```python
modelos = ["claude", "gpt-4", "gemini", "llama"]

# Acceso
print(modelos[0])    # "claude"
print(modelos[-1])   # "llama" (último)
print(modelos[1:3])  # ["gpt-4", "gemini"] (slicing)

# Modificar
modelos.append("mistral")
modelos.insert(0, "grok")
modelos.remove("gpt-4")
modelos.sort()

# Iterar
for modelo in modelos:
    print(modelo)
```

### Diccionarios

Muy usados en IA para representar mensajes, configuraciones y respuestas de API.

```python
mensaje = {
    "role": "user",
    "content": "¿Qué es un LLM?"
}

# Acceso
print(mensaje["role"])           # "user"
print(mensaje.get("id", None))   # None (con valor por defecto)

# Modificar
mensaje["timestamp"] = "2026-04-14"

# Iterar
for clave, valor in mensaje.items():
    print(f"{clave}: {valor}")

# Dict anidado
config = {
    "model": "claude-sonnet-4-6",
    "params": {
        "temperature": 0.7,
        "max_tokens": 1024
    }
}
print(config["params"]["temperature"])  # 0.7
```

### Conjuntos y tuplas

```python
# Tupla: inmutable, para datos que no deben cambiar
dimensiones = (1920, 1080)
coordenadas = (40.4168, -3.7038)  # Madrid

# Conjunto: sin duplicados, para operaciones de conjuntos
idiomas_modelo_a = {"es", "en", "fr", "de"}
idiomas_modelo_b = {"es", "en", "zh", "ja"}
print(idiomas_modelo_a & idiomas_modelo_b)  # {"es", "en"} — intersección
print(idiomas_modelo_a | idiomas_modelo_b)  # unión
```

---

## 4. Estructuras de control

```python
# Condicionales
score = 0.87
if score >= 0.9:
    etiqueta = "excelente"
elif score >= 0.7:
    etiqueta = "bueno"
else:
    etiqueta = "mejorable"

# Condicional en una línea (ternario)
estado = "aprobado" if score >= 0.6 else "suspenso"

# Bucle for con enumerate
textos = ["hola", "mundo", "IA"]
for i, texto in enumerate(textos):
    print(f"{i}: {texto}")

# Bucle for con zip (iterar dos listas a la vez)
nombres = ["Alice", "Bob", "Carol"]
scores = [0.95, 0.72, 0.88]
for nombre, score in zip(nombres, scores):
    print(f"{nombre}: {score:.1%}")

# While con break
intentos = 0
while intentos < 3:
    respuesta = input("¿Contraseña? ")
    if respuesta == "1234":
        break
    intentos += 1

# Manejo de excepciones
try:
    resultado = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
finally:
    print("Esto siempre se ejecuta")
```

---

## 5. Funciones y lambdas

```python
# Función básica
def calcular_coste(tokens_entrada: int, tokens_salida: int,
                   precio_entrada: float = 3.0,
                   precio_salida: float = 15.0) -> float:
    """
    Calcula el coste de una llamada a Claude Sonnet en USD.
    Precios en USD por millón de tokens.
    """
    return (tokens_entrada * precio_entrada + tokens_salida * precio_salida) / 1_000_000

# Llamada con argumentos posicionales
print(calcular_coste(500, 200))  # Usa precios por defecto

# Llamada con argumentos nombrados
print(calcular_coste(tokens_entrada=1000, tokens_salida=500, precio_entrada=15))

# Función que devuelve múltiples valores
def procesar_respuesta(texto: str) -> tuple[str, int]:
    limpio = texto.strip().lower()
    return limpio, len(limpio.split())

texto_limpio, num_palabras = procesar_respuesta("  Hola Mundo  ")

# Lambda: funciones anónimas para operaciones simples
ordenar_por_score = lambda x: x["score"]
resultados = [{"nombre": "A", "score": 0.8}, {"nombre": "B", "score": 0.95}]
resultados.sort(key=ordenar_por_score, reverse=True)

# Funciones de orden superior
numeros = [1, 2, 3, 4, 5]
cuadrados = list(map(lambda x: x**2, numeros))   # [1, 4, 9, 16, 25]
pares = list(filter(lambda x: x % 2 == 0, numeros))  # [2, 4]
```

---

## 6. Comprehensions

Las comprehensions son una forma pythónica y eficiente de construir listas, dicts y sets.

```python
# List comprehension
textos = ["hola", "mundo", "inteligencia", "artificial"]
longitudes = [len(t) for t in textos]               # [4, 5, 12, 10]
largos = [t for t in textos if len(t) > 5]          # ["inteligencia", "artificial"]
mayusculas = [t.upper() for t in textos]

# Dict comprehension
precios = {"claude-haiku": 0.25, "claude-sonnet": 3.0, "claude-opus": 15.0}
caros = {k: v for k, v in precios.items() if v > 1.0}

# Comprehension anidada
matriz = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# Set comprehension
tokens_unicos = {token for frase in textos for token in frase.split()}
```

---

## 7. Ficheros y JSON

```python
import json
from pathlib import Path

# Leer y escribir texto
ruta = Path("datos/conversacion.txt")
ruta.parent.mkdir(parents=True, exist_ok=True)

ruta.write_text("Hola mundo", encoding="utf-8")
contenido = ruta.read_text(encoding="utf-8")

# JSON: el formato estándar para APIs de IA
datos = {
    "modelo": "claude-sonnet-4-6",
    "mensajes": [
        {"role": "user", "content": "¿Qué es la IA?"}
    ],
    "temperatura": 0.7
}

# Serializar a JSON
json_str = json.dumps(datos, ensure_ascii=False, indent=2)
print(json_str)

# Guardar en fichero
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(datos, f, ensure_ascii=False, indent=2)

# Cargar desde fichero
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# También con Path
config = json.loads(Path("config.json").read_text(encoding="utf-8"))
```

---

## 8. Patrones habituales en IA

### Variables de entorno para claves API

```python
import os
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()  # Carga variables desde .env

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY no está configurada")
```

Fichero `.env` (nunca lo subas a Git):
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Logging en lugar de print

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Iniciando llamada a la API")
logger.warning("Límite de tokens cercano")
logger.error("Error en la llamada: %s", str(e))
```

### Dataclasses para estructuras de datos

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ConfiguracionLLM:
    modelo: str = "claude-sonnet-4-6"
    temperatura: float = 0.7
    max_tokens: int = 1024
    system_prompt: Optional[str] = None
    historial: list = field(default_factory=list)

config = ConfiguracionLLM(temperatura=0.3, max_tokens=2048)
print(config.modelo)  # "claude-sonnet-4-6"
```

---

## 9. Recursos para profundizar

| Recurso | Nivel | Enlace |
|---|---|---|
| Python oficial (docs) | Todos | docs.python.org |
| Real Python | Principiante-Intermedio | realpython.com |
| Automate the Boring Stuff | Principiante | automatetheboringstuff.com |
| Python Cookbook | Intermedio-Avanzado | O'Reilly |
| Fast.ai (Python para DL) | Intermedio | course.fast.ai |

---

**Siguiente:** [02 — Librerías esenciales](./02-librerias-esenciales.md)
