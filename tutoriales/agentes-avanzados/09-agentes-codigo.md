# 09 — Agentes de Código

> **Bloque:** 9 · **Nivel:** Avanzado · **Tiempo estimado:** 60 min

---

## Índice

1. Qué es un code agent y casos de uso
2. Ejecución de código con exec() y eval()
3. Sandboxing seguro con E2B
4. Agente analista de datos
5. Agente de debugging automatizado
6. Agente generador de tests
7. Consideraciones de seguridad
8. Patrones de producción

---

## 1. Qué es un code agent y casos de uso

Un code agent es un agente de IA que, además de razonar en lenguaje natural, puede **generar código ejecutable, ejecutarlo en un entorno controlado y usar los resultados como retroalimentación** para continuar razonando o tomar decisiones.

El ciclo de trabajo es:

```
Tarea del usuario
      │
      ▼
┌─────────────────────────────┐
│  LLM (Claude)               │
│  Razona sobre la tarea      │
│  Genera código              │
└────────────┬────────────────┘
             │  código generado
             ▼
┌─────────────────────────────┐
│  Entorno de ejecución       │
│  (local, sandbox, contenedor│
│   o E2B)                    │
└────────────┬────────────────┘
             │  stdout / stderr / archivos
             ▼
┌─────────────────────────────┐
│  LLM (Claude)               │
│  Interpreta resultados      │
│  Decide: ¿seguir? ¿corregir?│
└────────────┬────────────────┘
             │  respuesta final
             ▼
       Usuario
```

**Casos de uso principales:**

- **Análisis de datos.** El agente recibe un CSV, genera código pandas/matplotlib, ejecuta el análisis y devuelve tablas e imágenes sin que el usuario escriba una línea de código.
- **Debugging automatizado.** El agente recibe código roto, analiza el traceback, propone un fix y lo verifica ejecutando los tests.
- **Generación de tests.** Dado código Python, el agente genera una suite pytest completa, la ejecuta y corrige los tests que fallan.
- **Refactoring asistido.** El agente reescribe código para mejorar su legibilidad o rendimiento, ejecuta la suite de tests existente y confirma que nada se rompe.

La diferencia con un chatbot de código es la **retroalimentación real**: el agente no solo sugiere código, lo ejecuta y razona sobre el resultado.

---

## 2. Ejecución de código con exec() y eval()

Python ofrece dos primitivas para ejecutar código en tiempo de ejecución: `exec()` para sentencias y `eval()` para expresiones. Son el punto de partida más simple antes de introducir sandboxes.

### 2.1 exec() con captura de stdout y stderr

```python
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr


def ejecutar_codigo(codigo: str, contexto: dict | None = None) -> dict:
    """
    Ejecuta código Python arbitrario y captura su salida.

    Args:
        codigo:   Cadena con el código Python a ejecutar.
        contexto: Diccionario que actúa como espacio de nombres (globals).
                  Permite inyectar variables previas y recuperar resultados.

    Returns:
        Diccionario con claves:
          - stdout: salida estándar capturada
          - stderr: salida de error capturada
          - error:  mensaje de excepción si la ejecución falló
          - exito:  bool indicando si terminó sin excepción
    """
    if contexto is None:
        contexto = {}

    buffer_out = io.StringIO()
    buffer_err = io.StringIO()

    try:
        with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
            exec(codigo, contexto)  # noqa: S102
        return {
            "stdout": buffer_out.getvalue(),
            "stderr": buffer_err.getvalue(),
            "error": None,
            "exito": True,
        }
    except Exception:
        return {
            "stdout": buffer_out.getvalue(),
            "stderr": buffer_err.getvalue(),
            "error": traceback.format_exc(),
            "exito": False,
        }


# Ejemplo de uso
if __name__ == "__main__":
    codigo_ejemplo = """
import math

valores = [1, 4, 9, 16, 25]
raices = [math.sqrt(v) for v in valores]
print("Raíces cuadradas:", raices)
print("Suma:", sum(raices))
"""
    resultado = ejecutar_codigo(codigo_ejemplo)
    print("STDOUT:", resultado["stdout"])
    print("STDERR:", resultado["stderr"])
    print("EXITO:", resultado["exito"])
```

### 2.2 eval() para expresiones simples

```python
def evaluar_expresion(expresion: str, variables: dict | None = None) -> dict:
    """
    Evalúa una expresión Python y devuelve su valor.

    Útil cuando el LLM necesita calcular un valor puntual, no ejecutar
    un bloque completo de código.

    Args:
        expresion: Expresión Python (no sentencias, solo expresiones).
        variables: Diccionario de variables disponibles en el contexto.

    Returns:
        Diccionario con:
          - valor: resultado de la evaluación
          - error: mensaje de excepción si falló
          - exito: bool
    """
    if variables is None:
        variables = {}

    try:
        valor = eval(expresion, {"__builtins__": {}}, variables)  # noqa: S307
        return {"valor": valor, "error": None, "exito": True}
    except Exception as exc:
        return {"valor": None, "error": str(exc), "exito": False}


# Ejemplo de uso
if __name__ == "__main__":
    resultado = evaluar_expresion(
        "suma / len(lista)",
        variables={"suma": 45, "lista": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    )
    print("Valor:", resultado["valor"])  # 5.0
```

### 2.3 Riesgos de exec() y eval()

El uso de `exec()` y `eval()` en producción sin aislamiento es peligroso:

```
Riesgo                        Ejemplo de código malicioso
─────────────────────────────────────────────────────────
Acceso al sistema de archivos  open("/etc/passwd").read()
Ejecución de comandos          import os; os.system("rm -rf /")
Exfiltración de datos          import socket; ...
Agotamiento de recursos        while True: pass
Modificación del proceso       import sys; sys.exit(1)
```

Para entornos de producción con código generado por IA, usa siempre un sandbox aislado como E2B (sección 3) o contenedores Docker.

---

## 3. Sandboxing seguro con E2B

[E2B](https://e2b.dev) proporciona sandboxes de código aislados en la nube. Cada sandbox es un microVM efímero con Python preinstalado. El código se ejecuta en infraestructura separada de tu servidor, eliminando el riesgo de daño al sistema anfitrión.

### 3.1 Instalación y configuración

```bash
pip install e2b-code-interpreter anthropic
```

Necesitas una API key de E2B (obtenida en [e2b.dev](https://e2b.dev)):

```bash
export E2B_API_KEY="tu_api_key_aqui"
export ANTHROPIC_API_KEY="tu_api_key_aqui"
```

### 3.2 Uso básico del sandbox E2B

```python
import os
from e2b_code_interpreter import Sandbox


def ejecutar_en_sandbox(codigo: str, paquetes: list[str] | None = None) -> dict:
    """
    Ejecuta código Python en un sandbox E2B aislado.

    Args:
        codigo:    Código Python a ejecutar.
        paquetes:  Lista de paquetes pip a instalar antes de ejecutar.

    Returns:
        Diccionario con stdout, stderr, resultados y error si aplica.
    """
    with Sandbox() as sandbox:
        # Instalar paquetes adicionales si se solicitan
        if paquetes:
            paquetes_str = " ".join(paquetes)
            resultado_pip = sandbox.commands.run(f"pip install {paquetes_str} -q")
            if resultado_pip.exit_code != 0:
                return {
                    "exito": False,
                    "error": f"Error instalando paquetes: {resultado_pip.stderr}",
                    "stdout": "",
                    "stderr": resultado_pip.stderr,
                    "resultados": [],
                }

        # Ejecutar el código
        ejecucion = sandbox.run_code(codigo)

        return {
            "exito": ejecucion.error is None,
            "error": str(ejecucion.error) if ejecucion.error else None,
            "stdout": "\n".join(
                r.text for r in ejecucion.results if hasattr(r, "text") and r.text
            ),
            "stderr": ejecucion.logs.stderr if ejecucion.logs else "",
            "logs_stdout": ejecucion.logs.stdout if ejecucion.logs else "",
            "resultados": ejecucion.results,
        }


# Ejemplo de uso
if __name__ == "__main__":
    codigo = """
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "ventas": np.random.randint(100, 1000, 30),
    "mes": range(1, 31),
})
print(df.describe())
print(f"\\nMáximo de ventas: {df['ventas'].max()}")
"""

    resultado = ejecutar_en_sandbox(codigo, paquetes=["pandas", "numpy"])
    print("EXITO:", resultado["exito"])
    print("STDOUT:", resultado["logs_stdout"])
```

### 3.3 Trabajo con ficheros en el sandbox

```python
import os
from e2b_code_interpreter import Sandbox


def ejecutar_con_fichero(ruta_csv_local: str, codigo_analisis: str) -> dict:
    """
    Sube un fichero CSV al sandbox, ejecuta código de análisis y
    descarga los resultados generados.

    Args:
        ruta_csv_local:  Ruta local al fichero CSV.
        codigo_analisis: Código Python que utiliza el fichero como
                         '/home/user/datos.csv'.

    Returns:
        Diccionario con stdout, stderr y rutas de ficheros generados.
    """
    with Sandbox() as sandbox:
        # Subir el fichero CSV al sandbox
        with open(ruta_csv_local, "rb") as f:
            sandbox.files.write("/home/user/datos.csv", f)

        # Ejecutar el análisis
        ejecucion = sandbox.run_code(codigo_analisis)

        resultado = {
            "exito": ejecucion.error is None,
            "error": str(ejecucion.error) if ejecucion.error else None,
            "logs": ejecucion.logs.stdout if ejecucion.logs else "",
            "imagenes": [],
        }

        # Descargar imágenes generadas si existen
        for r in ejecucion.results:
            if hasattr(r, "png") and r.png:
                resultado["imagenes"].append(r.png)  # bytes base64

        return resultado
```

---

## 4. Agente analista de datos

Este agente recibe un fichero CSV, genera código pandas/matplotlib para analizarlo, ejecuta ese código en E2B y devuelve un informe con los resultados y las visualizaciones.

```python
import os
import base64
import anthropic
from e2b_code_interpreter import Sandbox


MODELO = "claude-sonnet-4-6"
cliente = anthropic.Anthropic()


def analizar_csv_con_agente(ruta_csv: str, pregunta: str) -> dict:
    """
    Agente que analiza un CSV respondiendo una pregunta del usuario.

    El agente opera en un bucle:
      1. Recibe la pregunta y una muestra de los datos.
      2. Genera código pandas/matplotlib para responder.
      3. Ejecuta el código en un sandbox E2B.
      4. Si hay error, lo corrige y vuelve a intentar.
      5. Cuando tiene resultados, genera la respuesta final.

    Args:
        ruta_csv: Ruta al fichero CSV local.
        pregunta: Pregunta en lenguaje natural sobre los datos.

    Returns:
        Diccionario con respuesta textual e imágenes (bytes PNG en base64).
    """
    # Leer una muestra del CSV para incluirla en el contexto inicial
    with open(ruta_csv) as f:
        primeras_lineas = "".join(f.readline() for _ in range(6))

    # Herramienta que el agente puede llamar para ejecutar código
    herramientas = [
        {
            "name": "ejecutar_python",
            "description": (
                "Ejecuta código Python en un sandbox seguro. "
                "El CSV está disponible en '/home/user/datos.csv'. "
                "Usa print() para mostrar resultados. "
                "Usa matplotlib y guarda figuras con plt.savefig() "
                "o muéstralas con plt.show() para capturarlas."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo": {
                        "type": "string",
                        "description": "Código Python completo a ejecutar.",
                    }
                },
                "required": ["codigo"],
            },
        }
    ]

    mensajes = [
        {
            "role": "user",
            "content": (
                f"Tengo un fichero CSV con los siguientes datos (primeras filas):\n\n"
                f"```\n{primeras_lineas}\n```\n\n"
                f"Pregunta: {pregunta}\n\n"
                f"Usa la herramienta ejecutar_python para analizar los datos. "
                f"El fichero completo está en '/home/user/datos.csv'."
            ),
        }
    ]

    imagenes_capturadas: list[str] = []
    max_iteraciones = 5

    with Sandbox() as sandbox:
        # Subir el CSV al sandbox una sola vez
        with open(ruta_csv, "rb") as f:
            sandbox.files.write("/home/user/datos.csv", f)

        for _ in range(max_iteraciones):
            respuesta = cliente.messages.create(
                model=MODELO,
                max_tokens=4096,
                tools=herramientas,
                messages=mensajes,
            )

            # Si el agente terminó de razonar, devolver la respuesta
            if respuesta.stop_reason == "end_turn":
                texto_final = " ".join(
                    bloque.text
                    for bloque in respuesta.content
                    if hasattr(bloque, "text")
                )
                return {"respuesta": texto_final, "imagenes": imagenes_capturadas}

            # Procesar llamadas a herramientas
            if respuesta.stop_reason == "tool_use":
                # Añadir la respuesta del asistente al historial
                mensajes.append({"role": "assistant", "content": respuesta.content})

                resultados_herramientas = []
                for bloque in respuesta.content:
                    if bloque.type != "tool_use":
                        continue

                    codigo = bloque.input.get("codigo", "")
                    ejecucion = sandbox.run_code(codigo)

                    # Capturar imágenes si el código generó visualizaciones
                    for r in ejecucion.results:
                        if hasattr(r, "png") and r.png:
                            imagenes_capturadas.append(r.png)

                    # Construir el resultado de la herramienta
                    if ejecucion.error:
                        contenido_resultado = (
                            f"ERROR:\n{ejecucion.error}\n\n"
                            f"STDERR:\n{ejecucion.logs.stderr if ejecucion.logs else ''}"
                        )
                    else:
                        logs = ejecucion.logs.stdout if ejecucion.logs else ""
                        contenido_resultado = logs if logs else "(sin salida)"
                        if imagenes_capturadas:
                            contenido_resultado += (
                                f"\n\n[{len(imagenes_capturadas)} imagen(es) capturada(s)]"
                            )

                    resultados_herramientas.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": bloque.id,
                            "content": contenido_resultado,
                        }
                    )

                mensajes.append({"role": "user", "content": resultados_herramientas})

    return {
        "respuesta": "El agente alcanzó el límite de iteraciones.",
        "imagenes": imagenes_capturadas,
    }


def guardar_imagen(datos_base64: str, ruta: str) -> None:
    """Guarda una imagen PNG desde datos base64 en disco."""
    with open(ruta, "wb") as f:
        f.write(base64.b64decode(datos_base64))


# Punto de entrada de ejemplo
if __name__ == "__main__":
    resultado = analizar_csv_con_agente(
        ruta_csv="ventas.csv",
        pregunta=(
            "¿Cuáles son los tres productos con mayor volumen de ventas? "
            "Muestra un gráfico de barras con los 10 primeros."
        ),
    )

    print("=== RESPUESTA DEL AGENTE ===")
    print(resultado["respuesta"])

    for i, imagen in enumerate(resultado["imagenes"]):
        ruta_img = f"grafico_{i + 1}.png"
        guardar_imagen(imagen, ruta_img)
        print(f"Imagen guardada: {ruta_img}")
```

---

## 5. Agente de debugging automatizado

Este agente recibe código Python con un error, analiza el traceback, propone un fix y lo verifica ejecutando pytest.

```python
import os
import anthropic
from e2b_code_interpreter import Sandbox


MODELO = "claude-sonnet-4-6"
cliente = anthropic.Anthropic()


def depurar_codigo(codigo_roto: str, tests_existentes: str | None = None) -> dict:
    """
    Agente de debugging que corrige código Python con errores.

    Flujo:
      1. Ejecuta el código roto para obtener el traceback real.
      2. Claude analiza el error y propone un fix.
      3. Ejecuta el código corregido.
      4. Si hay tests, los ejecuta con pytest para verificar.
      5. Itera hasta corregir o agotar intentos.

    Args:
        codigo_roto:      Código Python que contiene uno o más errores.
        tests_existentes: Suite de tests pytest para verificar el fix.

    Returns:
        Diccionario con:
          - codigo_corregido: el código final (corregido o no)
          - explicacion:      qué se cambió y por qué
          - tests_pasaron:    bool, None si no había tests
          - historial:        lista de intentos realizados
    """
    herramientas = [
        {
            "name": "ejecutar_codigo",
            "description": "Ejecuta código Python y devuelve stdout, stderr y traceback.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo": {"type": "string", "description": "Código Python a ejecutar."},
                    "nombre_fichero": {
                        "type": "string",
                        "description": "Nombre con el que guardar el fichero (p.ej. 'modulo.py').",
                        "default": "codigo.py",
                    },
                },
                "required": ["codigo"],
            },
        },
        {
            "name": "ejecutar_pytest",
            "description": "Ejecuta pytest sobre un fichero de tests y devuelve el resultado.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo_tests": {
                        "type": "string",
                        "description": "Contenido del fichero de tests pytest.",
                    },
                    "nombre_modulo": {
                        "type": "string",
                        "description": "Nombre del módulo que importan los tests (sin .py).",
                    },
                },
                "required": ["codigo_tests", "nombre_modulo"],
            },
        },
        {
            "name": "proponer_codigo_corregido",
            "description": (
                "Registra la versión final del código corregido y la explicación del fix. "
                "Llama a esta herramienta cuando estés seguro de que el código es correcto."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo": {"type": "string", "description": "Código Python corregido."},
                    "explicacion": {
                        "type": "string",
                        "description": "Explicación técnica de qué estaba mal y cómo se corrigió.",
                    },
                },
                "required": ["codigo", "explicacion"],
            },
        },
    ]

    contexto_tests = ""
    if tests_existentes:
        contexto_tests = f"\n\nTests disponibles para verificar el fix:\n```python\n{tests_existentes}\n```"

    mensajes = [
        {
            "role": "user",
            "content": (
                f"El siguiente código Python tiene uno o más errores. "
                f"Analiza el problema, corrígelo y verifica que funciona correctamente.\n\n"
                f"```python\n{codigo_roto}\n```"
                f"{contexto_tests}\n\n"
                f"Empieza ejecutando el código para ver el error real."
            ),
        }
    ]

    codigo_final = codigo_roto
    explicacion_final = ""
    tests_pasaron: bool | None = None
    historial: list[dict] = []

    with Sandbox() as sandbox:
        for iteracion in range(6):
            respuesta = cliente.messages.create(
                model=MODELO,
                max_tokens=4096,
                tools=herramientas,
                messages=mensajes,
            )

            if respuesta.stop_reason == "end_turn":
                break

            mensajes.append({"role": "assistant", "content": respuesta.content})
            resultados_herramientas = []

            for bloque in respuesta.content:
                if bloque.type != "tool_use":
                    continue

                entrada = bloque.input
                resultado_str = ""

                if bloque.name == "ejecutar_codigo":
                    codigo = entrada["codigo"]
                    nombre = entrada.get("nombre_fichero", "codigo.py")
                    sandbox.files.write(f"/home/user/{nombre}", codigo.encode())
                    ejecucion = sandbox.run_code(codigo)
                    logs = ejecucion.logs.stdout if ejecucion.logs else ""
                    errores = ejecucion.logs.stderr if ejecucion.logs else ""
                    error_exc = str(ejecucion.error) if ejecucion.error else ""
                    resultado_str = (
                        f"STDOUT:\n{logs}\n"
                        f"STDERR:\n{errores}\n"
                        f"EXCEPCION:\n{error_exc}"
                    )
                    historial.append(
                        {"iteracion": iteracion, "tipo": "ejecucion", "resultado": resultado_str}
                    )

                elif bloque.name == "ejecutar_pytest":
                    codigo_tests = entrada["codigo_tests"]
                    nombre_modulo = entrada.get("nombre_modulo", "codigo")
                    sandbox.files.write("/home/user/test_modulo.py", codigo_tests.encode())
                    cmd = sandbox.commands.run(
                        "cd /home/user && pip install pytest -q && pytest test_modulo.py -v 2>&1"
                    )
                    resultado_str = cmd.stdout or cmd.stderr or ""
                    tests_pasaron = cmd.exit_code == 0
                    historial.append(
                        {"iteracion": iteracion, "tipo": "pytest", "resultado": resultado_str}
                    )

                elif bloque.name == "proponer_codigo_corregido":
                    codigo_final = entrada["codigo"]
                    explicacion_final = entrada["explicacion"]
                    resultado_str = "Código corregido registrado correctamente."
                    historial.append(
                        {"iteracion": iteracion, "tipo": "fix_final", "codigo": codigo_final}
                    )

                resultados_herramientas.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": resultado_str,
                    }
                )

            mensajes.append({"role": "user", "content": resultados_herramientas})

    return {
        "codigo_corregido": codigo_final,
        "explicacion": explicacion_final,
        "tests_pasaron": tests_pasaron,
        "historial": historial,
    }


# Ejemplo de uso
if __name__ == "__main__":
    codigo_con_error = """
def calcular_promedio(numeros):
    total = sum(numeros)
    return total / len(numeros)  # Falla si la lista está vacía

def procesar_datos(datos):
    grupos = {}
    for item in datos:
        clave = item["categoria"]
        if clave not in grupos:
            grupos[clave] = []
        grupos[clave].append(item["valor"])

    return {
        clave: calcular_promedio(valores)
        for clave, valores in grupos.items()
    }

# Error: la categoría "C" tiene lista vacía
resultado = procesar_datos([
    {"categoria": "A", "valor": 10},
    {"categoria": "B", "valor": 20},
    {"categoria": "A", "valor": 30},
    {"categoria": "C", "valor": None},  # None rompe sum()
])
print(resultado)
"""

    resultado = depurar_codigo(codigo_con_error)
    print("=== CÓDIGO CORREGIDO ===")
    print(resultado["codigo_corregido"])
    print("\n=== EXPLICACIÓN ===")
    print(resultado["explicacion"])
    print(f"\nTests pasaron: {resultado['tests_pasaron']}")
    print(f"Iteraciones: {len(resultado['historial'])}")
```

---

## 6. Agente generador de tests

Este agente recibe código Python y genera una suite pytest completa, la ejecuta en E2B y corrige los tests que fallen.

```python
import os
import anthropic
from e2b_code_interpreter import Sandbox


MODELO = "claude-sonnet-4-6"
cliente = anthropic.Anthropic()


def generar_tests(codigo_fuente: str, nombre_modulo: str = "modulo") -> dict:
    """
    Genera tests pytest para código Python dado.

    El agente:
      1. Analiza el código fuente para entender su comportamiento.
      2. Genera una suite pytest con casos normales, edge cases y
         casos de error esperados.
      3. Ejecuta los tests en E2B.
      4. Corrige los tests que fallen (distinguiendo entre bug en
         el test o bug en el código fuente).
      5. Devuelve los tests finales y el informe de ejecución.

    Args:
        codigo_fuente: Código Python para el que generar tests.
        nombre_modulo: Nombre del módulo (sin .py) para los imports.

    Returns:
        Diccionario con:
          - tests: código pytest final
          - informe: salida de pytest
          - tests_pasaron: bool
          - cobertura_estimada: descripción de qué se cubre
    """
    herramientas = [
        {
            "name": "ejecutar_tests",
            "description": (
                "Ejecuta una suite pytest junto con el código fuente "
                "en un sandbox aislado y devuelve el resultado."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo_tests": {
                        "type": "string",
                        "description": "Contenido completo del fichero de tests pytest.",
                    }
                },
                "required": ["codigo_tests"],
            },
        },
        {
            "name": "finalizar_tests",
            "description": (
                "Registra la versión final de los tests y el resumen de cobertura. "
                "Llama solo cuando todos los tests pasen."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo_tests": {
                        "type": "string",
                        "description": "Código pytest final y correcto.",
                    },
                    "cobertura_estimada": {
                        "type": "string",
                        "description": "Descripción de qué funcionalidades cubre la suite.",
                    },
                },
                "required": ["codigo_tests", "cobertura_estimada"],
            },
        },
    ]

    mensajes = [
        {
            "role": "user",
            "content": (
                f"Genera una suite de tests pytest completa para el siguiente código Python.\n\n"
                f"Nombre del módulo: `{nombre_modulo}`\n\n"
                f"```python\n{codigo_fuente}\n```\n\n"
                f"Requisitos para la suite:\n"
                f"- Cubre casos normales, edge cases y errores esperados.\n"
                f"- Usa pytest con fixtures cuando sea apropiado.\n"
                f"- Incluye docstrings descriptivos en cada test.\n"
                f"- Los imports deben referenciar el módulo `{nombre_modulo}`.\n\n"
                f"Ejecuta los tests para verificar que todos pasan antes de finalizar."
            ),
        }
    ]

    tests_finales = ""
    informe_final = ""
    cobertura_final = ""
    tests_pasaron = False

    with Sandbox() as sandbox:
        # Escribir el código fuente en el sandbox permanentemente
        sandbox.files.write(f"/home/user/{nombre_modulo}.py", codigo_fuente.encode())
        # Instalar pytest
        sandbox.commands.run("pip install pytest -q")

        for _ in range(5):
            respuesta = cliente.messages.create(
                model=MODELO,
                max_tokens=4096,
                tools=herramientas,
                messages=mensajes,
            )

            if respuesta.stop_reason == "end_turn":
                break

            mensajes.append({"role": "assistant", "content": respuesta.content})
            resultados_herramientas = []

            for bloque in respuesta.content:
                if bloque.type != "tool_use":
                    continue

                entrada = bloque.input
                resultado_str = ""

                if bloque.name == "ejecutar_tests":
                    codigo_tests = entrada["codigo_tests"]
                    sandbox.files.write("/home/user/test_suite.py", codigo_tests.encode())
                    cmd = sandbox.commands.run(
                        "cd /home/user && pytest test_suite.py -v --tb=short 2>&1"
                    )
                    resultado_str = cmd.stdout or cmd.stderr or "(sin salida)"
                    informe_final = resultado_str
                    tests_pasaron = cmd.exit_code == 0

                elif bloque.name == "finalizar_tests":
                    tests_finales = entrada["codigo_tests"]
                    cobertura_final = entrada.get("cobertura_estimada", "")
                    resultado_str = "Tests finalizados y registrados."

                resultados_herramientas.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": bloque.id,
                        "content": resultado_str,
                    }
                )

            mensajes.append({"role": "user", "content": resultados_herramientas})

            # Si los tests pasaron y están registrados, terminar
            if tests_pasaron and tests_finales:
                break

    return {
        "tests": tests_finales,
        "informe": informe_final,
        "tests_pasaron": tests_pasaron,
        "cobertura_estimada": cobertura_final,
    }


# Ejemplo de uso
if __name__ == "__main__":
    codigo_a_testear = """
from dataclasses import dataclass
from typing import Optional


@dataclass
class Producto:
    nombre: str
    precio: float
    stock: int

    def aplicar_descuento(self, porcentaje: float) -> float:
        \"\"\"Aplica un descuento y devuelve el precio resultante.\"\"\"
        if not 0 <= porcentaje <= 100:
            raise ValueError(f"El porcentaje debe estar entre 0 y 100, got {porcentaje}")
        return self.precio * (1 - porcentaje / 100)

    def hay_stock(self, cantidad: int = 1) -> bool:
        \"\"\"Comprueba si hay suficiente stock para la cantidad solicitada.\"\"\"
        return self.stock >= cantidad

    def vender(self, cantidad: int) -> None:
        \"\"\"Reduce el stock. Lanza ValueError si no hay suficiente.\"\"\"
        if not self.hay_stock(cantidad):
            raise ValueError(
                f"Stock insuficiente: {self.stock} disponibles, {cantidad} solicitados"
            )
        self.stock -= cantidad


class Carrito:
    def __init__(self) -> None:
        self.items: list[tuple[Producto, int]] = []

    def agregar(self, producto: Producto, cantidad: int = 1) -> None:
        if cantidad <= 0:
            raise ValueError("La cantidad debe ser positiva")
        self.items.append((producto, cantidad))

    def total(self) -> float:
        return sum(p.precio * c for p, c in self.items)

    def total_con_descuento(self, porcentaje: float) -> float:
        return sum(p.aplicar_descuento(porcentaje) * c for p, c in self.items)

    def __len__(self) -> int:
        return sum(c for _, c in self.items)
"""

    resultado = generar_tests(codigo_a_testear, nombre_modulo="tienda")

    print("=== TESTS GENERADOS ===")
    print(resultado["tests"])
    print("\n=== INFORME PYTEST ===")
    print(resultado["informe"])
    print(f"\nTests pasaron: {resultado['tests_pasaron']}")
    print(f"Cobertura: {resultado['cobertura_estimada']}")
```

---

## 7. Consideraciones de seguridad

### 7.1 Código que nunca debes ejecutar sin sandbox

Algunas categorías de código son inherentemente peligrosas en entornos locales:

```
Categoría               Patrones a detectar
──────────────────────────────────────────────────────────────
Acceso al sistema       os.system(), subprocess, shutil.rmtree
Operaciones de red      socket, urllib, requests, httpx
Ficheros del sistema    open() con rutas absolutas (/etc, /var)
Imports peligrosos      importlib, ctypes, cffi, pickle
Introspección profunda  __import__, globals(), locals()
Salida del proceso      sys.exit(), os._exit(), os.kill()
Bucles infinitos        while True sin break ni timeout
Consumo de memoria      [x]*10**9, bytearray(10**12)
```

### 7.2 Validación de código antes de ejecutar

```python
import ast
import re


# Módulos que nunca deben importarse en ejecución local
MODULOS_PROHIBIDOS = {
    "os", "subprocess", "sys", "shutil", "socket",
    "urllib", "requests", "httpx", "importlib",
    "ctypes", "cffi", "pickle", "shelve", "pty",
}

# Funciones peligrosas que no deben aparecer en el código
FUNCIONES_PROHIBIDAS = {
    "exec", "eval", "compile", "open", "__import__",
    "globals", "locals", "vars", "dir",
}


def validar_codigo(codigo: str) -> tuple[bool, list[str]]:
    """
    Valida estáticamente un bloque de código Python antes de ejecutarlo.

    Analiza el AST para detectar imports y llamadas peligrosas sin
    ejecutar el código. No es infalible (un atacante sofisticado puede
    evadir el análisis), pero reduce el riesgo en escenarios de bajo
    riesgo o como capa adicional con E2B.

    Args:
        codigo: Código Python a validar.

    Returns:
        Tupla (es_seguro, lista_de_problemas_detectados).
    """
    problemas: list[str] = []

    # 1. Verificar que el código es Python válido antes de parsear
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as exc:
        return False, [f"Error de sintaxis: {exc}"]

    # 2. Recorrer el AST buscando patrones peligrosos
    for nodo in ast.walk(arbol):
        # Detectar imports prohibidos
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                modulo_base = alias.name.split(".")[0]
                if modulo_base in MODULOS_PROHIBIDOS:
                    problemas.append(f"Import prohibido: {alias.name}")

        elif isinstance(nodo, ast.ImportFrom):
            modulo_base = (nodo.module or "").split(".")[0]
            if modulo_base in MODULOS_PROHIBIDOS:
                problemas.append(f"Import prohibido: from {nodo.module}")

        # Detectar llamadas a funciones prohibidas
        elif isinstance(nodo, ast.Call):
            if isinstance(nodo.func, ast.Name):
                if nodo.func.id in FUNCIONES_PROHIBIDAS:
                    problemas.append(f"Función prohibida: {nodo.func.id}()")

    # 3. Detectar patrones con regex como capa adicional
    patrones_regex = [
        (r"__[a-z]+__\s*\(", "Llamada a dunder method directa"),
        (r"getattr\s*\(", "Uso de getattr (puede usarse para evadir análisis)"),
        (r"setattr\s*\(", "Uso de setattr"),
    ]
    for patron, descripcion in patrones_regex:
        if re.search(patron, codigo):
            problemas.append(descripcion)

    return len(problemas) == 0, problemas


# Ejemplo de uso
if __name__ == "__main__":
    codigo_sospechoso = """
import os
resultado = os.system("ls -la /etc")
print(resultado)
"""
    es_seguro, problemas = validar_codigo(codigo_sospechoso)
    print(f"Seguro: {es_seguro}")
    print(f"Problemas: {problemas}")
    # Seguro: False
    # Problemas: ['Import prohibido: os']

    codigo_limpio = """
import math
valores = [1, 4, 9, 16]
print([math.sqrt(v) for v in valores])
"""
    es_seguro, problemas = validar_codigo(codigo_limpio)
    print(f"Seguro: {es_seguro}")
    # Seguro: True
```

### 7.3 Timeouts de ejecución

```python
import signal
import contextlib
from typing import Generator


class TimeoutError(Exception):
    pass


@contextlib.contextmanager
def limite_tiempo(segundos: int) -> Generator[None, None, None]:
    """
    Context manager que lanza TimeoutError si el bloque tarda
    más de `segundos` en completarse. Solo funciona en Unix.

    Uso:
        with limite_tiempo(10):
            resultado = ejecutar_codigo_largo()
    """
    def handler(signum, frame):
        raise TimeoutError(f"Ejecución superó el límite de {segundos} segundos")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(segundos)
    try:
        yield
    finally:
        signal.alarm(0)  # Cancelar la alarma si termina antes


def ejecutar_con_timeout(codigo: str, timeout_segundos: int = 30) -> dict:
    """
    Ejecuta código con un timeout estricto.

    En E2B, el timeout se configura directamente en el sandbox
    y es más fiable. Esta función sirve para ejecución local.
    """
    import io
    import traceback
    from contextlib import redirect_stdout, redirect_stderr

    buffer_out = io.StringIO()
    buffer_err = io.StringIO()

    try:
        with limite_tiempo(timeout_segundos):
            with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
                exec(codigo, {})  # noqa: S102
        return {"exito": True, "stdout": buffer_out.getvalue(), "error": None}
    except TimeoutError as exc:
        return {"exito": False, "stdout": buffer_out.getvalue(), "error": str(exc)}
    except Exception:
        return {
            "exito": False,
            "stdout": buffer_out.getvalue(),
            "error": traceback.format_exc(),
        }
```

---

## 8. Patrones de producción

### 8.1 Reintentos con backoff exponencial

```python
import time
import anthropic
from e2b_code_interpreter import Sandbox


def ejecutar_con_reintentos(
    codigo: str,
    max_intentos: int = 3,
    timeout_sandbox: int = 30,
) -> dict:
    """
    Ejecuta código en E2B con reintentos automáticos ante fallos
    transitorios (timeouts de red, errores de inicialización del sandbox).

    No reintenta ante errores de ejecución del código (SyntaxError,
    RuntimeError), solo ante fallos de infraestructura.

    Args:
        codigo:          Código a ejecutar.
        max_intentos:    Número máximo de intentos.
        timeout_sandbox: Tiempo máximo de ejecución en segundos.

    Returns:
        Resultado de la ejecución o diccionario de error tras agotar intentos.
    """
    ultimo_error = None

    for intento in range(1, max_intentos + 1):
        try:
            with Sandbox(timeout=timeout_sandbox) as sandbox:
                ejecucion = sandbox.run_code(codigo)
                return {
                    "exito": ejecucion.error is None,
                    "error": str(ejecucion.error) if ejecucion.error else None,
                    "stdout": ejecucion.logs.stdout if ejecucion.logs else "",
                    "stderr": ejecucion.logs.stderr if ejecucion.logs else "",
                    "intento": intento,
                }
        except Exception as exc:
            ultimo_error = exc
            if intento < max_intentos:
                espera = 2 ** (intento - 1)  # 1s, 2s, 4s...
                time.sleep(espera)

    return {
        "exito": False,
        "error": f"Agotados {max_intentos} intentos. Último error: {ultimo_error}",
        "stdout": "",
        "stderr": "",
        "intento": max_intentos,
    }
```

### 8.2 Límites de recursos y cuotas

```python
import anthropic
from e2b_code_interpreter import Sandbox


MODELO = "claude-sonnet-4-6"
cliente = anthropic.Anthropic()


class LimitesAgente:
    """
    Envuelve un agente de código con límites de recursos
    para uso en producción.

    Atributos configurables:
      - max_iteraciones:      Evita bucles infinitos de razonamiento.
      - max_tokens_por_turno: Controla el coste por llamada a la API.
      - timeout_ejecucion:    Mata ejecuciones que no terminan.
      - max_ejecuciones:      Limita el número de sandboxes creados.
    """

    def __init__(
        self,
        max_iteraciones: int = 10,
        max_tokens_por_turno: int = 2048,
        timeout_ejecucion: int = 30,
        max_ejecuciones: int = 5,
    ) -> None:
        self.max_iteraciones = max_iteraciones
        self.max_tokens_por_turno = max_tokens_por_turno
        self.timeout_ejecucion = timeout_ejecucion
        self.max_ejecuciones = max_ejecuciones

    def ejecutar_agente(self, tarea: str, herramientas: list[dict]) -> dict:
        """
        Ejecuta un agente con los límites configurados.

        Registra métricas de uso para monitorización externa.
        """
        mensajes = [{"role": "user", "content": tarea}]
        metricas = {
            "iteraciones": 0,
            "tokens_entrada_total": 0,
            "tokens_salida_total": 0,
            "ejecuciones_sandbox": 0,
            "errores_ejecucion": 0,
        }

        with Sandbox(timeout=self.timeout_ejecucion) as sandbox:
            for _ in range(self.max_iteraciones):
                metricas["iteraciones"] += 1

                respuesta = cliente.messages.create(
                    model=MODELO,
                    max_tokens=self.max_tokens_por_turno,
                    tools=herramientas,
                    messages=mensajes,
                )

                # Registrar uso de tokens
                metricas["tokens_entrada_total"] += respuesta.usage.input_tokens
                metricas["tokens_salida_total"] += respuesta.usage.output_tokens

                if respuesta.stop_reason == "end_turn":
                    texto = " ".join(
                        b.text for b in respuesta.content if hasattr(b, "text")
                    )
                    return {
                        "exito": True,
                        "respuesta": texto,
                        "metricas": metricas,
                    }

                if respuesta.stop_reason == "tool_use":
                    if metricas["ejecuciones_sandbox"] >= self.max_ejecuciones:
                        return {
                            "exito": False,
                            "respuesta": "Límite de ejecuciones de sandbox alcanzado.",
                            "metricas": metricas,
                        }

                    mensajes.append({"role": "assistant", "content": respuesta.content})
                    resultados = []

                    for bloque in respuesta.content:
                        if bloque.type != "tool_use":
                            continue

                        metricas["ejecuciones_sandbox"] += 1
                        codigo = bloque.input.get("codigo", "")
                        ejecucion = sandbox.run_code(codigo)

                        if ejecucion.error:
                            metricas["errores_ejecucion"] += 1

                        logs = ejecucion.logs.stdout if ejecucion.logs else ""
                        error = str(ejecucion.error) if ejecucion.error else ""
                        contenido = logs or error or "(sin salida)"

                        resultados.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": bloque.id,
                                "content": contenido,
                            }
                        )

                    mensajes.append({"role": "user", "content": resultados})

        return {
            "exito": False,
            "respuesta": f"Agente alcanzó el límite de {self.max_iteraciones} iteraciones.",
            "metricas": metricas,
        }


# Ejemplo de uso
if __name__ == "__main__":
    agente = LimitesAgente(
        max_iteraciones=8,
        max_tokens_por_turno=2048,
        timeout_ejecucion=20,
        max_ejecuciones=4,
    )

    herramientas_basicas = [
        {
            "name": "ejecutar_codigo",
            "description": "Ejecuta código Python en un sandbox.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "codigo": {"type": "string"}
                },
                "required": ["codigo"],
            },
        }
    ]

    resultado = agente.ejecutar_agente(
        tarea="Calcula los primeros 20 números de Fibonacci e imprime su suma.",
        herramientas=herramientas_basicas,
    )

    print("Respuesta:", resultado["respuesta"])
    print("Métricas:", resultado["metricas"])
```

### 8.3 Resumen de decisiones de diseño

```
Decisión                  Opción A (local)        Opción B (E2B)
────────────────────────────────────────────────────────────────
Aislamiento               Ninguno sin sandbox     Completo (microVM)
Instalación de paquetes   pip local               pip en sandbox efímero
Latencia                  Muy baja (<1ms)         Baja (200-500ms init)
Coste                     Solo servidor           API E2B + servidor
Multiproceso              Posible                 Soporte nativo
Estado entre ejecuciones  Contexto compartido     Sandbox persistente
Ficheros                  Sistema anfitrión       Sistema aislado
Uso recomendado           Código confiable        Código generado por IA
```

**Regla práctica:** en cualquier sistema donde el código lo genera un LLM a partir de entrada de usuario, usa siempre un sandbox aislado (E2B o equivalente). El análisis estático de la sección 7.2 es una capa complementaria, no un sustituto del aislamiento.

---

**Siguiente tutorial:** [10 — Agentes Especializados](./10-agentes-especializados.md)
