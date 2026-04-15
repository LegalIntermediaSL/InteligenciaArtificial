# 01 — Generación y revisión de código con IA

> **Bloque:** Casos de uso avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Casos de uso de IA para código](#1-casos-de-uso-de-ia-para-código)
2. [Revisor de código con Claude](#2-revisor-de-código-con-claude)
3. [Generador de tests unitarios](#3-generador-de-tests-unitarios)
4. [Documentador automático](#4-documentador-automático)
5. [Explicador de código](#5-explicador-de-código)
6. [Pipeline completo](#6-pipeline-completo)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Casos de uso de IA para código

Los LLMs han demostrado ser especialmente útiles en tareas relacionadas con el código. Las cuatro categorías principales son:

| Tarea | Descripción | Beneficio principal |
|---|---|---|
| **Generación** | Escribir código a partir de una especificación | Velocidad de prototipado |
| **Revisión** | Detectar bugs, antipatrones y problemas de seguridad | Calidad y consistencia |
| **Tests** | Generar casos de prueba para funciones existentes | Cobertura de tests |
| **Documentación** | Añadir docstrings y comentarios al código | Mantenibilidad |

En este tutorial construimos herramientas para cada categoría y las combinamos en un pipeline completo.

**Requisitos:**

```bash
pip install anthropic pydantic python-dotenv
```

---

## 2. Revisor de código con Claude

El revisor lee un fichero `.py`, lo envía a Claude con una rúbrica estructurada y recibe un JSON con los issues detectados.

```python
import anthropic
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


class Issue(BaseModel):
    tipo: Literal["bug", "seguridad", "rendimiento", "estilo", "mantenibilidad"]
    linea: int | None
    descripcion: str
    severidad: Literal["critica", "alta", "media", "baja"]
    sugerencia: str


class RevisionCodigo(BaseModel):
    fichero: str
    lenguaje: str
    issues: list[Issue]
    puntuacion_global: int  # 0-100
    resumen: str


def revisar_codigo(ruta_fichero: str) -> RevisionCodigo:
    """Revisa un fichero Python y devuelve un informe estructurado."""
    client = anthropic.Anthropic()

    codigo = Path(ruta_fichero).read_text(encoding="utf-8")
    nombre = Path(ruta_fichero).name

    prompt = f"""Revisa el siguiente código Python y devuelve un análisis en formato JSON.

Fichero: {nombre}

```python
{codigo}
```

Devuelve ÚNICAMENTE un objeto JSON con esta estructura exacta:
{{
  "fichero": "{nombre}",
  "lenguaje": "Python",
  "issues": [
    {{
      "tipo": "bug|seguridad|rendimiento|estilo|mantenibilidad",
      "linea": <número de línea o null>,
      "descripcion": "<descripción clara del problema>",
      "severidad": "critica|alta|media|baja",
      "sugerencia": "<cómo corregirlo>"
    }}
  ],
  "puntuacion_global": <0-100>,
  "resumen": "<resumen ejecutivo de 2-3 frases>"
}}

Rúbrica de revisión:
- Bugs: errores lógicos, excepciones no manejadas, condiciones de carrera
- Seguridad: inyección, datos expuestos, validación de entrada
- Rendimiento: complejidad innecesaria, operaciones costosas en bucles
- Estilo: PEP 8, nombres descriptivos, consistencia
- Mantenibilidad: funciones largas, código duplicado, acoplamiento excesivo"""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    texto = respuesta.content[0].text.strip()

    # Extraer JSON si viene envuelto en bloques de código
    if "```json" in texto:
        texto = texto.split("```json")[1].split("```")[0].strip()
    elif "```" in texto:
        texto = texto.split("```")[1].split("```")[0].strip()

    datos = json.loads(texto)
    return RevisionCodigo(**datos)


def imprimir_revision(revision: RevisionCodigo):
    """Formatea e imprime el informe de revisión."""
    print(f"\n{'='*60}")
    print(f"REVISIÓN DE CÓDIGO: {revision.fichero}")
    print(f"Puntuación: {revision.puntuacion_global}/100")
    print(f"{'='*60}")
    print(f"\nResumen: {revision.resumen}\n")

    if not revision.issues:
        print("No se detectaron issues.")
        return

    # Ordenar por severidad
    orden = {"critica": 0, "alta": 1, "media": 2, "baja": 3}
    issues_ordenados = sorted(revision.issues, key=lambda x: orden[x.severidad])

    for i, issue in enumerate(issues_ordenados, 1):
        linea_info = f"Línea {issue.linea}" if issue.linea else "General"
        print(f"[{i}] [{issue.severidad.upper()}] {issue.tipo} — {linea_info}")
        print(f"    Problema:   {issue.descripcion}")
        print(f"    Sugerencia: {issue.sugerencia}\n")


# Uso
if __name__ == "__main__":
    # Crear un fichero de prueba con problemas intencionados
    codigo_con_problemas = '''
import os
import subprocess

password = "admin123"  # hardcoded password

def procesar_usuarios(lista):
    resultado = []
    for i in range(len(lista)):
        user = lista[i]
        # SQL sin parametrizar
        query = "SELECT * FROM users WHERE name = '" + user + "'"
        resultado.append(query)
    return resultado

def dividir(a, b):
    return a / b  # sin manejo de ZeroDivisionError

def ejecutar_comando(cmd):
    os.system(cmd)  # inyección de comandos
'''

    Path("codigo_prueba.py").write_text(codigo_con_problemas, encoding="utf-8")

    revision = revisar_codigo("codigo_prueba.py")
    imprimir_revision(revision)
```

---

## 3. Generador de tests unitarios

Dado el código de una función Python, genera automáticamente tests con `pytest`.

```python
import anthropic
import ast
import inspect
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class TestGenerator:
    """Genera tests unitarios con pytest a partir de código Python."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def extraer_funciones(self, codigo: str) -> list[str]:
        """Extrae los nombres de todas las funciones de un fragmento de código."""
        try:
            tree = ast.parse(codigo)
            return [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
        except SyntaxError:
            return []

    def generar_tests(self, codigo_fuente: str, nombre_modulo: str = "modulo") -> str:
        """Genera tests pytest para el código dado."""
        funciones = self.extraer_funciones(codigo_fuente)
        lista_funciones = ", ".join(funciones) if funciones else "todas las funciones"

        prompt = f"""Genera tests pytest completos y funcionales para el siguiente código Python.

Módulo: {nombre_modulo}
Funciones detectadas: {lista_funciones}

```python
{codigo_fuente}
```

Requisitos para los tests:
1. Importar el módulo correctamente: `from {nombre_modulo} import *`
2. Una clase de test por función: `class TestNombreFuncion`
3. Al menos 3 casos por función: caso normal, caso borde, caso de error
4. Usar `pytest.raises` para excepciones esperadas
5. Nombres de tests descriptivos: `test_suma_numeros_positivos`
6. Docstring en cada test explicando qué verifica
7. No uses mocks salvo que sean estrictamente necesarios

Devuelve ÚNICAMENTE el código Python de los tests, sin explicaciones adicionales."""

        respuesta = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        texto = respuesta.content[0].text.strip()

        # Limpiar bloques de código markdown si los hay
        if "```python" in texto:
            texto = texto.split("```python")[1].split("```")[0].strip()
        elif "```" in texto:
            texto = texto.split("```")[1].split("```")[0].strip()

        return texto

    def guardar_tests(self, tests: str, ruta_salida: str):
        """Guarda los tests generados en un fichero."""
        Path(ruta_salida).write_text(tests, encoding="utf-8")
        print(f"Tests guardados en: {ruta_salida}")


# Uso
if __name__ == "__main__":
    codigo_a_testear = '''
def suma(a: float, b: float) -> float:
    """Suma dos números."""
    return a + b


def dividir(a: float, b: float) -> float:
    """Divide a entre b. Lanza ValueError si b es cero."""
    if b == 0:
        raise ValueError("No se puede dividir entre cero")
    return a / b


def es_palindromo(texto: str) -> bool:
    """Comprueba si una cadena es palíndroma (ignora mayúsculas y espacios)."""
    limpio = texto.lower().replace(" ", "")
    return limpio == limpio[::-1]


def factorial(n: int) -> int:
    """Calcula el factorial de n. Lanza ValueError si n es negativo."""
    if n < 0:
        raise ValueError("El factorial no está definido para números negativos")
    if n == 0:
        return 1
    return n * factorial(n - 1)
'''

    generator = TestGenerator()
    tests = generator.generar_tests(codigo_a_testear, "matematicas")
    print(tests)
    generator.guardar_tests(tests, "test_matematicas.py")
```

---

## 4. Documentador automático

Detecta funciones sin docstring usando `ast` y las documenta automáticamente con Claude.

```python
import anthropic
import ast
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ParseFunctionDoc:
    """Extrae y documenta funciones Python sin docstrings."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def encontrar_sin_docstring(self, codigo: str) -> list[dict]:
        """
        Encuentra funciones que no tienen docstring.

        Returns:
            Lista de dicts con 'nombre', 'linea_inicio', 'linea_fin', 'codigo'
        """
        tree = ast.parse(codigo)
        lineas = codigo.splitlines()
        funciones_sin_doc = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tiene_docstring = (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )

                if not tiene_docstring:
                    inicio = node.lineno - 1
                    fin = node.end_lineno
                    codigo_funcion = "\n".join(lineas[inicio:fin])

                    funciones_sin_doc.append({
                        "nombre": node.name,
                        "linea_inicio": node.lineno,
                        "linea_fin": node.end_lineno,
                        "codigo": codigo_funcion,
                    })

        return funciones_sin_doc

    def generar_docstring(self, codigo_funcion: str) -> str:
        """Genera un docstring para una función dada."""
        prompt = f"""Genera un docstring en español para la siguiente función Python.

```python
{codigo_funcion}
```

Formato requerido (Google style):
\"\"\"Descripción breve en una línea.

Args:
    param1 (tipo): Descripción.
    param2 (tipo): Descripción.

Returns:
    tipo: Descripción de lo que devuelve.

Raises:
    TipoError: Cuándo se lanza.
\"\"\"

Devuelve ÚNICAMENTE el docstring (con las triples comillas), sin código adicional."""

        respuesta = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        return respuesta.content[0].text.strip()

    def documentar_fichero(self, ruta_entrada: str, ruta_salida: str = None):
        """
        Procesa un fichero Python y añade docstrings a las funciones que no los tienen.
        """
        codigo = Path(ruta_entrada).read_text(encoding="utf-8")
        funciones = self.encontrar_sin_docstring(codigo)

        if not funciones:
            print("Todas las funciones ya tienen docstring.")
            return codigo

        print(f"Funciones sin docstring: {[f['nombre'] for f in funciones]}")
        lineas = codigo.splitlines()

        # Procesar en orden inverso para no desplazar índices
        for funcion in reversed(funciones):
            print(f"  Documentando: {funcion['nombre']}...")
            docstring = self.generar_docstring(funcion["codigo"])

            # Detectar indentación de la función
            linea_def = lineas[funcion["linea_inicio"] - 1]
            indentacion = len(linea_def) - len(linea_def.lstrip())
            indent_body = " " * (indentacion + 4)

            # Formatear el docstring con la indentación correcta
            lineas_doc = docstring.strip().splitlines()
            docstring_indentado = "\n".join(
                indent_body + l if i > 0 else indent_body + l
                for i, l in enumerate(lineas_doc)
            )

            # Insertar después de la línea def
            linea_insercion = funcion["linea_inicio"]  # índice 1-based
            lineas.insert(linea_insercion, docstring_indentado)

        codigo_documentado = "\n".join(lineas)

        if ruta_salida:
            Path(ruta_salida).write_text(codigo_documentado, encoding="utf-8")
            print(f"\nFichero documentado guardado en: {ruta_salida}")

        return codigo_documentado


# Uso
if __name__ == "__main__":
    codigo_sin_docs = '''
def calcular_imc(peso_kg: float, altura_m: float) -> float:
    if altura_m <= 0:
        raise ValueError("La altura debe ser positiva")
    return peso_kg / (altura_m ** 2)


def clasificar_imc(imc: float) -> str:
    if imc < 18.5:
        return "Bajo peso"
    elif imc < 25:
        return "Normopeso"
    elif imc < 30:
        return "Sobrepeso"
    else:
        return "Obesidad"


def procesar_paciente(nombre: str, peso: float, altura: float) -> dict:
    imc = calcular_imc(peso, altura)
    categoria = clasificar_imc(imc)
    return {"nombre": nombre, "imc": round(imc, 2), "categoria": categoria}
'''

    Path("sin_docs.py").write_text(codigo_sin_docs, encoding="utf-8")

    doc = ParseFunctionDoc()
    resultado = doc.documentar_fichero("sin_docs.py", "con_docs.py")
    print("\n--- Código documentado ---")
    print(resultado)
```

---

## 5. Explicador de código

Recibe un fragmento difícil y devuelve una explicación línea a línea.

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()


def explicar_codigo(fragmento: str, nivel: str = "intermedio") -> str:
    """
    Explica un fragmento de código línea a línea.

    Args:
        fragmento: El código a explicar.
        nivel: 'principiante', 'intermedio' o 'avanzado'.

    Returns:
        Explicación detallada del código.
    """
    client = anthropic.Anthropic()

    niveles = {
        "principiante": "sin conocimientos previos de Python",
        "intermedio": "con conocimientos básicos de Python",
        "avanzado": "con experiencia en Python y CS",
    }

    descripcion_nivel = niveles.get(nivel, niveles["intermedio"])

    prompt = f"""Explica el siguiente código Python de forma clara para alguien {descripcion_nivel}.

```python
{fragmento}
```

Formato de respuesta:
1. **Propósito general**: Qué hace este código en una frase.
2. **Explicación línea a línea**: Explica cada línea o bloque significativo.
3. **Conceptos clave**: Lista los patrones o conceptos de Python utilizados.
4. **Ejemplo de uso**: Muestra cómo se llamaría este código y qué devolvería.

Responde en español."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    return respuesta.content[0].text


# Uso
if __name__ == "__main__":
    codigo_complejo = '''
from functools import reduce
from typing import Callable, TypeVar

T = TypeVar("T")

def pipeline(*funciones: Callable) -> Callable:
    return lambda x: reduce(lambda v, f: f(v), funciones, x)

limpiar = pipeline(
    str.strip,
    str.lower,
    lambda s: s.replace("-", " "),
    lambda s: " ".join(s.split()),
)

resultado = limpiar("  Hola--Mundo   ")
'''

    print("=== EXPLICACIÓN DE CÓDIGO ===\n")
    explicacion = explicar_codigo(codigo_complejo, nivel="intermedio")
    print(explicacion)
```

---

## 6. Pipeline completo

Un solo script que encadena todas las herramientas: revisar → documentar → generar tests.

```python
import anthropic
import json
import ast
from pathlib import Path
from dotenv import load_dotenv

# Importar clases definidas en las secciones anteriores
# (en un proyecto real estarían en módulos separados)

load_dotenv()


def pipeline_codigo(ruta_fichero: str, directorio_salida: str = "output"):
    """
    Ejecuta el pipeline completo de mejora de código:
    1. Revisión de calidad
    2. Documentación automática
    3. Generación de tests

    Args:
        ruta_fichero: Ruta al fichero Python a procesar.
        directorio_salida: Carpeta donde guardar los resultados.
    """
    Path(directorio_salida).mkdir(exist_ok=True)
    nombre = Path(ruta_fichero).stem

    print(f"\n{'='*60}")
    print(f"PIPELINE DE MEJORA DE CÓDIGO: {ruta_fichero}")
    print(f"{'='*60}\n")

    client = anthropic.Anthropic()
    codigo = Path(ruta_fichero).read_text(encoding="utf-8")

    # --- PASO 1: Revisión ---
    print("PASO 1/3 — Revisando código...")
    revision_prompt = f"""Revisa este código Python y devuelve JSON con esta estructura:
{{"puntuacion": 0-100, "issues_criticos": ["..."], "issues_menores": ["..."], "resumen": "..."}}

```python
{codigo}
```

Solo JSON, sin texto adicional."""

    resp_revision = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": revision_prompt}],
    )

    texto_revision = resp_revision.content[0].text.strip()
    if "```" in texto_revision:
        texto_revision = texto_revision.split("```")[1].split("```")[0]
        if texto_revision.startswith("json"):
            texto_revision = texto_revision[4:]

    revision = json.loads(texto_revision.strip())
    print(f"  Puntuacion: {revision['puntuacion']}/100")
    print(f"  Issues criticos: {len(revision.get('issues_criticos', []))}")
    print(f"  Issues menores: {len(revision.get('issues_menores', []))}")

    # Guardar informe de revisión
    ruta_revision = Path(directorio_salida) / f"{nombre}_revision.json"
    ruta_revision.write_text(
        json.dumps(revision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- PASO 2: Documentación ---
    print("\nPASO 2/3 — Documentando funciones sin docstring...")

    doc = ParseFunctionDoc()
    funciones_sin_doc = doc.encontrar_sin_docstring(codigo)

    if funciones_sin_doc:
        print(f"  Funciones a documentar: {[f['nombre'] for f in funciones_sin_doc]}")
        codigo_documentado = doc.documentar_fichero(ruta_fichero)
        ruta_doc = Path(directorio_salida) / f"{nombre}_documentado.py"
        ruta_doc.write_text(codigo_documentado, encoding="utf-8")
        print(f"  Guardado en: {ruta_doc}")
    else:
        print("  Todas las funciones ya tienen docstring.")
        codigo_documentado = codigo

    # --- PASO 3: Tests ---
    print("\nPASO 3/3 — Generando tests unitarios...")

    gen = TestGenerator()
    tests = gen.generar_tests(codigo_documentado, nombre)
    ruta_tests = Path(directorio_salida) / f"test_{nombre}.py"
    gen.guardar_tests(tests, str(ruta_tests))

    # --- Resumen final ---
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"  Revision:      {ruta_revision}")
    print(f"  Documentado:   {Path(directorio_salida) / f'{nombre}_documentado.py'}")
    print(f"  Tests:         {ruta_tests}")
    print(f"\nResumen: {revision['resumen']}")


# Clases necesarias (incluidas aquí para que el script sea autocontenido)
class ParseFunctionDoc:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def encontrar_sin_docstring(self, codigo: str) -> list[dict]:
        tree = ast.parse(codigo)
        lineas = codigo.splitlines()
        funciones_sin_doc = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tiene_docstring = (
                    isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                if not tiene_docstring:
                    inicio = node.lineno - 1
                    fin = node.end_lineno
                    funciones_sin_doc.append({
                        "nombre": node.name,
                        "linea_inicio": node.lineno,
                        "linea_fin": node.end_lineno,
                        "codigo": "\n".join(lineas[inicio:fin]),
                    })
        return funciones_sin_doc

    def documentar_fichero(self, ruta_entrada: str) -> str:
        codigo = Path(ruta_entrada).read_text(encoding="utf-8")
        funciones = self.encontrar_sin_docstring(codigo)
        lineas = codigo.splitlines()
        for funcion in reversed(funciones):
            docstring = self._generar_docstring(funcion["codigo"])
            linea_def = lineas[funcion["linea_inicio"] - 1]
            indentacion = len(linea_def) - len(linea_def.lstrip())
            indent_body = " " * (indentacion + 4)
            lineas_doc = docstring.strip().splitlines()
            docstring_indentado = "\n".join(indent_body + l for l in lineas_doc)
            lineas.insert(funcion["linea_inicio"], docstring_indentado)
        return "\n".join(lineas)

    def _generar_docstring(self, codigo_funcion: str) -> str:
        respuesta = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"Genera un docstring en español (Google style) para:\n```python\n{codigo_funcion}\n```\nDevuelve solo el docstring con triples comillas."
            }],
        )
        return respuesta.content[0].text.strip()


class TestGenerator:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def generar_tests(self, codigo: str, nombre_modulo: str) -> str:
        respuesta = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": f"Genera tests pytest completos para el siguiente código. Importa con `from {nombre_modulo} import *`. Devuelve solo el código Python.\n\n```python\n{codigo}\n```"
            }],
        )
        texto = respuesta.content[0].text.strip()
        if "```python" in texto:
            texto = texto.split("```python")[1].split("```")[0].strip()
        return texto

    def guardar_tests(self, tests: str, ruta: str):
        Path(ruta).write_text(tests, encoding="utf-8")
        print(f"  Guardado en: {ruta}")


# Ejemplo de ejecución
if __name__ == "__main__":
    codigo_ejemplo = '''
def calcular_descuento(precio: float, porcentaje: float) -> float:
    return precio * (1 - porcentaje / 100)

def aplicar_iva(precio: float, tasa_iva: float = 21.0) -> float:
    return precio * (1 + tasa_iva / 100)

def precio_final(precio_base: float, descuento: float = 0, iva: float = 21.0) -> float:
    con_descuento = calcular_descuento(precio_base, descuento)
    return aplicar_iva(con_descuento, iva)
'''

    Path("tienda.py").write_text(codigo_ejemplo, encoding="utf-8")
    pipeline_codigo("tienda.py", directorio_salida="output_tienda")
```

---

## 7. Extensiones sugeridas

| Extensión | Descripción | Tecnología |
|---|---|---|
| **Revisión en CI/CD** | Integrar el revisor como paso de GitHub Actions | `actions/python-versions` |
| **Soporte multi-lenguaje** | Extender a JavaScript, TypeScript, Go | Detectar extensión del fichero |
| **Comparación antes/después** | Mostrar diff de código documentado | `difflib.unified_diff` |
| **Interfaz web** | Subir fichero y ver informe en navegador | `streamlit` |
| **Métricas de calidad** | Combinar con `pylint` o `flake8` | `subprocess` + IA |
| **Cache de revisiones** | Evitar revisar ficheros no modificados | Hash SHA256 + JSON local |

---

**Siguiente:** [02 — Búsqueda semántica con embeddings](./02-busqueda-semantica.md)
