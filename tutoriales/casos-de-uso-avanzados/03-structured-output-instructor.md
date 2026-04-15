# 03 — Structured output robusto con Instructor y Pydantic

> **Bloque:** Casos de uso avanzados · **Nivel:** Avanzado · **Tiempo estimado:** 40 min

---

## Índice

1. [El problema de json.loads directo](#1-el-problema-de-jsonloads-directo)
2. [Introducción a Instructor](#2-introducción-a-instructor)
3. [Extracción básica](#3-extracción-básica)
4. [Modelos anidados](#4-modelos-anidados)
5. [Validación personalizada con field_validator](#5-validación-personalizada-con-field_validator)
6. [Extracción desde texto no estructurado](#6-extracción-desde-texto-no-estructurado)
7. [Manejo de errores y reintentos](#7-manejo-de-errores-y-reintentos)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. El problema de json.loads directo

Cuando se pide a un LLM que devuelva JSON y luego se hace `json.loads()`, los fallos en producción son frecuentes:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Ejemplo de lo que puede salir mal
prompt_naive = """Extrae el nombre y la edad del siguiente texto y devuelve JSON.
Texto: "María tiene 28 años y trabaja como ingeniera."
"""

respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": prompt_naive}],
)

texto_respuesta = respuesta.content[0].text
print(f"Respuesta raw:\n{texto_respuesta}\n")

# Casos que hacen fallar json.loads():
outputs_problematicos = [
    # Caso 1: El modelo añade explicación antes del JSON
    'Aquí está el JSON que pediste:\n```json\n{"nombre": "María", "edad": 28}\n```',

    # Caso 2: Usa comillas simples en lugar de dobles
    "{'nombre': 'María', 'edad': 28}",

    # Caso 3: Añade una coma al final (JSON inválido)
    '{"nombre": "María", "edad": 28,}',

    # Caso 4: Campo con tipo incorrecto
    '{"nombre": "María", "edad": "veintiocho"}',

    # Caso 5: Campos adicionales no esperados o campos que faltan
    '{"nombre": "María"}',
]

print("=== Fallos con json.loads() directo ===\n")
for i, output in enumerate(outputs_problematicos, 1):
    try:
        datos = json.loads(output)
        print(f"[{i}] OK: {datos}")
    except json.JSONDecodeError as e:
        print(f"[{i}] ERROR JSONDecodeError: {e}")
    except Exception as e:
        print(f"[{i}] ERROR: {type(e).__name__}: {e}")
```

**Resultado típico:** 4 de 5 fallan silenciosamente o con excepciones no controladas. En producción, esto provoca errores intermitentes difíciles de reproducir.

---

## 2. Introducción a Instructor

Instructor parchea el cliente de Anthropic u OpenAI para que devuelva directamente modelos Pydantic validados, con reintentos automáticos.

```bash
pip install instructor pydantic anthropic openai
```

```python
import instructor
import anthropic
import openai

# Parchear cliente Anthropic
client_anthropic = instructor.from_anthropic(anthropic.Anthropic())

# Parchear cliente OpenAI
client_openai = instructor.from_openai(openai.OpenAI())

# A partir de aquí, ambos clientes aceptan response_model=
# y devuelven instancias Pydantic validadas, no texto crudo.

print("Instructor configurado correctamente.")
print(f"Versión instructor: {instructor.__version__}")
```

**Cómo funciona internamente:**
1. Instructor convierte el modelo Pydantic en un JSON Schema.
2. Lo envía al LLM como tool definition o system prompt.
3. El LLM devuelve JSON que coincide con el esquema.
4. Instructor valida con Pydantic. Si falla, reintenta automáticamente.

---

## 3. Extracción básica

El caso más sencillo: extraer campos tipados de texto libre.

```python
import instructor
import anthropic
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_anthropic(anthropic.Anthropic())


class Persona(BaseModel):
    nombre: str
    edad: int
    profesion: str
    ciudad: str | None = None  # Opcional


def extraer_persona(texto: str) -> Persona:
    """Extrae información de una persona a partir de texto libre."""
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"Extrae la información de la persona del siguiente texto:\n\n{texto}",
            }
        ],
        response_model=Persona,
    )


# Pruebas
textos = [
    "María García tiene 28 años y es ingeniera de software en Barcelona.",
    "Juan, 45 años, trabaja como médico.",
    "La doctora Ana López (32) lleva 5 años como pediatra en Madrid.",
]

for texto in textos:
    persona = extraer_persona(texto)
    print(f"Texto: {texto}")
    print(f"  Nombre:    {persona.nombre}")
    print(f"  Edad:      {persona.edad}")
    print(f"  Profesion: {persona.profesion}")
    print(f"  Ciudad:    {persona.ciudad or 'No mencionada'}")
    print()
```

**Ventajas frente a json.loads:**
- Tipos garantizados (`edad` siempre es `int`).
- Campos opcionales con valores por defecto.
- Si el LLM devuelve algo inválido, Instructor reintenta automáticamente.

---

## 4. Modelos anidados

Extracción de estructuras complejas con listas y objetos anidados.

```python
import instructor
import anthropic
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_anthropic(anthropic.Anthropic())


class Empleado(BaseModel):
    nombre: str
    rol: str
    departamento: str
    skills: list[str]
    anios_experiencia: int


class Empresa(BaseModel):
    nombre: str
    sector: str
    pais: str
    num_empleados_total: int | None = None
    empleados: list[Empleado]


def extraer_empresa(descripcion: str) -> Empresa:
    """Extrae la estructura completa de una empresa desde texto."""
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""Extrae toda la información estructurada de la empresa
                y sus empleados del siguiente texto:\n\n{descripcion}""",
            }
        ],
        response_model=Empresa,
    )


# Texto de prueba con múltiples empleados
descripcion_empresa = """
TechSolutions S.L. es una empresa española del sector fintech fundada en 2018.
Tienen aproximadamente 150 empleados en total.

El equipo técnico está formado por:
- Carlos Ruiz, CTO con 12 años de experiencia, experto en arquitectura cloud,
  Python y Kubernetes. Trabaja en el departamento de Ingeniería.
- Laura Martínez, desarrolladora senior de 7 años de experiencia especializada
  en React, TypeScript y GraphQL, del equipo de Frontend.
- Andrés Moreno, data scientist con 4 años, domina Python, R, TensorFlow y
  SQL. Pertenece al departamento de Datos.
- Sofía Chen, DevOps engineer con 6 años de experiencia en AWS, Docker y
  Terraform. Trabaja en Infraestructura.
"""

empresa = extraer_empresa(descripcion_empresa)

print(f"Empresa: {empresa.nombre}")
print(f"Sector: {empresa.sector} | País: {empresa.pais}")
print(f"Total empleados: {empresa.num_empleados_total or 'N/A'}")
print(f"\nEmpleados extraídos: {len(empresa.empleados)}")

for emp in empresa.empleados:
    print(f"\n  {emp.nombre} — {emp.rol} ({emp.departamento})")
    print(f"    Experiencia: {emp.anios_experiencia} años")
    print(f"    Skills: {', '.join(emp.skills)}")
```

---

## 5. Validación personalizada con field_validator

Añadir lógica de validación propia que se ejecuta antes de devolver el resultado.

```python
import instructor
import anthropic
from pydantic import BaseModel, field_validator, model_validator
import re
from typing import Self
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_anthropic(anthropic.Anthropic())


class RegistroUsuario(BaseModel):
    nombre: str
    email: str
    edad: int
    telefono: str | None = None
    rol: str = "usuario"

    @field_validator("email")
    @classmethod
    def validar_email(cls, v: str) -> str:
        """Valida que el email tiene formato correcto."""
        patron = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(patron, v):
            raise ValueError(f"Email inválido: '{v}'. Debe tener formato usuario@dominio.ext")
        return v.lower()

    @field_validator("edad")
    @classmethod
    def validar_edad(cls, v: int) -> int:
        """Valida que la edad es un valor razonable."""
        if v < 0:
            raise ValueError(f"La edad no puede ser negativa: {v}")
        if v > 130:
            raise ValueError(f"La edad {v} no es realista")
        return v

    @field_validator("telefono")
    @classmethod
    def validar_telefono(cls, v: str | None) -> str | None:
        """Normaliza el formato del teléfono si está presente."""
        if v is None:
            return v
        # Eliminar caracteres no numéricos excepto el +
        limpio = re.sub(r"[^\d+]", "", v)
        if len(limpio) < 9:
            raise ValueError(f"Teléfono demasiado corto: '{v}'")
        return limpio

    @field_validator("nombre")
    @classmethod
    def validar_nombre(cls, v: str) -> str:
        """Normaliza el nombre a Title Case."""
        if len(v.strip()) < 2:
            raise ValueError("El nombre es demasiado corto")
        return v.strip().title()

    @model_validator(mode="after")
    def validar_consistencia(self) -> Self:
        """Validaciones que dependen de múltiples campos."""
        if self.rol == "admin" and self.edad < 18:
            raise ValueError("Los administradores deben ser mayores de edad")
        return self


def extraer_usuario(texto: str) -> RegistroUsuario:
    """Extrae y valida un registro de usuario desde texto libre."""
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"Extrae los datos del usuario del siguiente texto:\n\n{texto}",
            }
        ],
        response_model=RegistroUsuario,
        max_retries=3,  # Instructor reintentará si la validación falla
    )


# Pruebas con diferentes entradas
casos = [
    "El usuario Juan García, de 32 años, se registra con el email juan.garcia@empresa.com y teléfono 612-345-678.",
    "María López (26 años), email: maria@test.es, rol admin, tel: +34 699 111 222.",
    "Pedro Sánchez, 45, pedro.sanchez@correo.org",
]

for caso in casos:
    try:
        usuario = extraer_usuario(caso)
        print(f"OK: {usuario.nombre} | {usuario.email} | {usuario.edad} años | tel: {usuario.telefono or 'N/A'}")
    except Exception as e:
        print(f"ERROR: {e}")
```

---

## 6. Extracción desde texto no estructurado

Procesar noticias y extraer múltiples campos de forma robusta.

```python
import instructor
import anthropic
from pydantic import BaseModel, field_validator
from datetime import date
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

client = instructor.from_anthropic(anthropic.Anthropic())


class Entidad(BaseModel):
    texto: str
    tipo: Literal["persona", "organizacion", "lugar", "producto", "evento"]


class AnalisisNoticia(BaseModel):
    titular: str
    fecha_publicacion: str | None  # ISO 8601 o None si no se menciona
    entidades: list[Entidad]
    sentimiento: Literal["positivo", "negativo", "neutro", "mixto"]
    puntuacion_sentimiento: float  # -1.0 a 1.0
    temas_principales: list[str]
    resumen: str
    fiabilidad_estimada: Literal["alta", "media", "baja"]

    @field_validator("puntuacion_sentimiento")
    @classmethod
    def validar_puntuacion(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"La puntuación de sentimiento debe estar entre -1.0 y 1.0, got {v}")
        return round(v, 2)

    @field_validator("temas_principales")
    @classmethod
    def validar_temas(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Debe haber al menos un tema principal")
        return [t.lower().strip() for t in v]


def analizar_noticia(texto_noticia: str) -> AnalisisNoticia:
    """Analiza una noticia y extrae todos sus metadatos."""
    return client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Analiza la siguiente noticia y extrae toda la información estructurada.
Para la puntuacion_sentimiento usa: 1.0 = muy positivo, 0 = neutro, -1.0 = muy negativo.
Para fiabilidad_estimada, evalúa el tono: sensacionalista=baja, equilibrado=alta.

Noticia:
{texto_noticia}""",
            }
        ],
        response_model=AnalisisNoticia,
        max_retries=3,
    )


# Prueba con una noticia de ejemplo
noticia = """
El Banco Central Europeo anunció ayer en Frankfurt una bajada de tipos de interés
de 25 puntos básicos, situándolos en el 3,25%. La presidenta Christine Lagarde
explicó que la medida busca estimular la economía ante la desaceleración observada
en Alemania, Francia e Italia durante el último trimestre.

Los mercados reaccionaron positivamente: el IBEX 35 subió un 1,8% y el DAX alemán
avanzó un 2,1%. Sin embargo, los economistas del FMI advierten que la medida podría
incrementar la inflación en países del sur de Europa como España y Portugal.

La decisión fue adoptada por unanimidad en la reunión del Consejo de Gobierno
celebrada el 14 de abril de 2026.
"""

analisis = analizar_noticia(noticia)

print(f"Titular:    {analisis.titular}")
print(f"Fecha:      {analisis.fecha_publicacion or 'No especificada'}")
print(f"Sentimiento: {analisis.sentimiento} ({analisis.puntuacion_sentimiento:+.2f})")
print(f"Fiabilidad: {analisis.fiabilidad_estimada}")
print(f"Temas:      {', '.join(analisis.temas_principales)}")
print(f"\nResumen:\n{analisis.resumen}")
print(f"\nEntidades detectadas ({len(analisis.entidades)}):")
for ent in analisis.entidades:
    print(f"  [{ent.tipo:15s}] {ent.texto}")
```

---

## 7. Manejo de errores y reintentos

Instructor gestiona automáticamente los reintentos cuando la validación Pydantic falla. Aquí se muestra cómo funciona y cómo personalizar el comportamiento.

```python
import instructor
import anthropic
from pydantic import BaseModel, field_validator
from instructor import Instructor
from instructor.exceptions import InstructorRetryException
from dotenv import load_dotenv
import logging

load_dotenv()

# Activar logging para ver los reintentos en acción
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("instructor")


class ProductoValidado(BaseModel):
    nombre: str
    precio: float
    moneda: str
    en_stock: bool
    categoria: str

    @field_validator("precio")
    @classmethod
    def precio_positivo(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"El precio debe ser positivo, se recibió: {v}")
        return round(v, 2)

    @field_validator("moneda")
    @classmethod
    def moneda_valida(cls, v: str) -> str:
        codigos_validos = {"EUR", "USD", "GBP", "JPY", "CHF"}
        v_upper = v.upper()
        if v_upper not in codigos_validos:
            raise ValueError(
                f"Moneda '{v}' no reconocida. Usa uno de: {', '.join(codigos_validos)}"
            )
        return v_upper


def extraer_producto_con_reintentos(
    descripcion: str,
    max_reintentos: int = 3,
) -> ProductoValidado | None:
    """
    Extrae un producto con reintentos automáticos.

    Cuando Pydantic lanza un ValueError en un validator, Instructor:
    1. Captura el error.
    2. Lo incluye en el siguiente mensaje al LLM como feedback.
    3. Reintenta hasta max_reintentos veces.

    Args:
        descripcion: Texto con la descripción del producto.
        max_reintentos: Número máximo de reintentos.

    Returns:
        ProductoValidado o None si todos los reintentos fallan.
    """
    client = instructor.from_anthropic(anthropic.Anthropic())

    try:
        producto = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": f"Extrae la información del producto:\n\n{descripcion}",
                }
            ],
            response_model=ProductoValidado,
            max_retries=max_reintentos,
        )
        return producto

    except InstructorRetryException as e:
        print(f"Fallaron todos los reintentos ({max_reintentos}).")
        print(f"Último error: {e}")
        return None


# Casos de prueba
descripciones = [
    # Caso normal
    "Laptop Dell XPS 15, precio: 1.299,99 EUR, en stock, categoría: Electrónica",

    # Caso con moneda ambigua (el modelo puede escribir 'euros' → validator corrige)
    "Auriculares Sony WH-1000XM5, 349 dólares, disponible, accesorios",

    # Caso con precio en formato texto (instructor reintentará si el LLM lo pone como string)
    "Libro 'Python Avanzado', coste: veinticinco euros, sí está disponible, libros",
]

print("=== Extracción con reintentos automáticos ===\n")
for desc in descripciones:
    print(f"Entrada: {desc}")
    resultado = extraer_producto_con_reintentos(desc)
    if resultado:
        print(f"  OK: {resultado.nombre} — {resultado.precio} {resultado.moneda} — "
              f"{'En stock' if resultado.en_stock else 'Sin stock'}")
    else:
        print("  FALLIDO: no se pudo extraer el producto")
    print()
```

**Flujo de reintentos de Instructor:**

```
1ª llamada → LLM devuelve JSON → Pydantic valida → ✓ OK
                                                  → ✗ ValueError
                                                       ↓
2ª llamada → [system: "El valor X falló por Y. Corrige."] → ...
                                                  → ✓ OK
                                                  → ✗ → 3ª llamada → ...
```

---

## 8. Extensiones sugeridas

| Extensión | Descripción | Tecnología |
|---|---|---|
| **Extracción en batch** | Procesar cientos de documentos en paralelo | `asyncio` + `instructor` async |
| **Modo streaming** | Recibir el objeto Pydantic parcialmente mientras llega | `client.messages.create_partial()` |
| **Cache de extracciones** | Evitar llamadas repetidas con el mismo input | Hash SHA256 + SQLite |
| **Esquemas dinámicos** | Crear modelos Pydantic en tiempo de ejecución | `create_model()` de Pydantic |
| **OpenAI Structured Outputs** | Alternativa nativa para GPT-4o | `response_format={"type": "json_schema"}` |
| **Métricas de calidad** | Medir tasa de éxito y número de reintentos | Decorador con contador |

---

**Siguiente:** [04 — Análisis de datos con IA](./04-analisis-datos-ia.md)
