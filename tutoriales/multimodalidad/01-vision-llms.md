# 01 — Visión artificial con LLMs

> **Bloque:** Multimodalidad · **Nivel:** Práctico · **Tiempo estimado:** 60 min

---

## Índice

1. [Qué son los modelos multimodales y cuándo usarlos](#1-qué-son-los-modelos-multimodales-y-cuándo-usarlos)
2. [Analizar una imagen local con Claude](#2-analizar-una-imagen-local-con-claude)
3. [Analizar una imagen desde URL](#3-analizar-una-imagen-desde-url)
4. [Caso práctico — extractor de datos de facturas](#4-caso-práctico--extractor-de-datos-de-facturas)
5. [Caso práctico — comparar dos imágenes](#5-caso-práctico--comparar-dos-imágenes)
6. [Caso práctico — transcribir capturas de pantalla y documentos escaneados](#6-caso-práctico--transcribir-capturas-de-pantalla-y-documentos-escaneados)
7. [Límites y consideraciones](#7-límites-y-consideraciones)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Qué son los modelos multimodales y cuándo usarlos

Un modelo **multimodal** es aquel que puede procesar más de un tipo de dato de forma nativa. Hasta hace poco, los modelos de lenguaje solo aceptaban texto. Hoy, modelos como `claude-sonnet-4-6` de Anthropic o `gpt-4o` de OpenAI pueden recibir imágenes, audio o vídeo junto con texto y generar respuestas que integran toda esa información.

### ¿Cuándo tiene sentido usar visión?

| Caso de uso | Descripción |
|---|---|
| Análisis de documentos | Facturas, contratos, formularios escaneados |
| Control de calidad | Detectar defectos en productos en fotografías |
| Accesibilidad | Describir imágenes para personas con discapacidad visual |
| Extracción de datos | Convertir tablas en imágenes a datos estructurados |
| Moderación de contenido | Revisar si una imagen cumple políticas de uso |
| Comparación visual | Detectar diferencias entre dos versiones de un diseño |

### Cómo funcionan internamente

Los modelos multimodales pasan las imágenes por un **encoder visual** (similar a CLIP) que convierte los píxeles en vectores. Esos vectores se combinan con los embeddings de texto y se procesan juntos en el transformer. El resultado es un modelo que "entiende" el contenido visual de la misma forma en que comprende el texto.

---

## 2. Analizar una imagen local con Claude

Para enviar una imagen local a Claude hay que codificarla en **base64** y especificar su tipo MIME. La API de Anthropic acepta los formatos `image/jpeg`, `image/png`, `image/gif` e `image/webp`.

```python
import anthropic
import base64
from pathlib import Path

def analizar_imagen_local(ruta_imagen: str, pregunta: str = "Describe esta imagen en detalle.") -> str:
    """
    Envía una imagen local a Claude y devuelve el análisis.

    Args:
        ruta_imagen: Ruta al archivo de imagen en el disco.
        pregunta: Instrucción o pregunta sobre la imagen.

    Returns:
        Respuesta textual de Claude.
    """
    # Leer la imagen y codificarla en base64
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    with open(ruta, "rb") as f:
        datos_imagen = base64.standard_b64encode(f.read()).decode("utf-8")

    # Detectar el tipo MIME según la extensión
    extension = ruta.suffix.lower()
    tipos_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    tipo_mime = tipos_mime.get(extension, "image/jpeg")

    # Crear el cliente de Anthropic
    cliente = anthropic.Anthropic()  # Usa ANTHROPIC_API_KEY del entorno

    # Construir el mensaje con la imagen en base64
    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_mime,
                            "data": datos_imagen,
                        },
                    },
                    {
                        "type": "text",
                        "text": pregunta,
                    },
                ],
            }
        ],
    )

    return mensaje.content[0].text


# --- Uso ---
if __name__ == "__main__":
    # Ajusta la ruta a una imagen que tengas en tu disco
    resultado = analizar_imagen_local(
        ruta_imagen="foto.jpg",
        pregunta="Describe esta imagen. ¿Qué elementos aparecen? ¿Cuál es el contexto?",
    )
    print(resultado)
```

### Puntos clave

- `base64.standard_b64encode` convierte los bytes de la imagen a una cadena ASCII segura para JSON.
- El campo `source.type` debe ser `"base64"` cuando enviamos datos locales.
- `media_type` debe coincidir con el formato real del archivo, no solo la extensión.

---

## 3. Analizar una imagen desde URL

Si la imagen está disponible públicamente en Internet puedes indicarle a Claude que la descargue directamente usando `source.type = "url"`. Esto evita tener que descargar y codificar la imagen tú mismo.

```python
import anthropic

def analizar_imagen_url(url: str, pregunta: str = "Describe esta imagen en detalle.") -> str:
    """
    Envía una URL de imagen a Claude y devuelve el análisis.

    Args:
        url: URL pública de la imagen.
        pregunta: Instrucción o pregunta sobre la imagen.

    Returns:
        Respuesta textual de Claude.
    """
    cliente = anthropic.Anthropic()

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    },
                    {
                        "type": "text",
                        "text": pregunta,
                    },
                ],
            }
        ],
    )

    return mensaje.content[0].text


# --- Uso ---
if __name__ == "__main__":
    url_prueba = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    resultado = analizar_imagen_url(
        url=url_prueba,
        pregunta="¿Qué muestra esta imagen? Sé preciso.",
    )
    print(resultado)
```

### Cuándo usar URL vs base64

| Situación | Recomendación |
|---|---|
| Imagen en un servidor público | URL (más rápido, sin transferencia de datos) |
| Imagen local o privada | base64 |
| Imagen ya descargada en memoria | base64 (desde bytes) |
| Imagen detrás de autenticación | base64 (descarga tú mismo con credenciales) |

---

## 4. Caso práctico — extractor de datos de facturas

Un caso de uso muy común en empresas es extraer datos de facturas escaneadas o en PDF. Con visión podemos convertir una imagen de factura en un JSON estructurado sin OCR tradicional.

```python
import anthropic
import base64
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


# Modelo Pydantic que define la estructura esperada
class DatosFactura(BaseModel):
    numero_factura: Optional[str] = Field(None, description="Número o ID de la factura")
    fecha_emision: Optional[str] = Field(None, description="Fecha de emisión (YYYY-MM-DD)")
    fecha_vencimiento: Optional[str] = Field(None, description="Fecha de vencimiento si aparece")
    proveedor_nombre: Optional[str] = Field(None, description="Nombre del proveedor o emisor")
    proveedor_nif: Optional[str] = Field(None, description="NIF/CIF/RFC del proveedor")
    cliente_nombre: Optional[str] = Field(None, description="Nombre del cliente o receptor")
    importe_base: Optional[float] = Field(None, description="Base imponible sin impuestos")
    impuestos: Optional[float] = Field(None, description="Importe total de impuestos (IVA, etc.)")
    importe_total: Optional[float] = Field(None, description="Total final a pagar")
    moneda: Optional[str] = Field(None, description="Código de moneda, ej: EUR, USD, MXN")
    concepto: Optional[str] = Field(None, description="Descripción breve del servicio o producto")


def extraer_datos_factura(ruta_imagen: str) -> DatosFactura:
    """
    Extrae datos estructurados de una imagen de factura usando Claude.

    Args:
        ruta_imagen: Ruta al archivo de imagen de la factura.

    Returns:
        Objeto DatosFactura con los campos extraídos.
    """
    # Cargar y codificar la imagen
    ruta = Path(ruta_imagen)
    with open(ruta, "rb") as f:
        datos_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    extension = ruta.suffix.lower()
    tipos_mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    tipo_mime = tipos_mime.get(extension, "image/jpeg")

    cliente = anthropic.Anthropic()

    # Prompt estructurado para extracción de datos
    prompt_sistema = """Eres un sistema de extracción de datos de facturas.
Tu tarea es analizar la imagen de una factura y extraer la información en formato JSON.
Responde ÚNICAMENTE con un objeto JSON válido que siga exactamente este esquema:

{
  "numero_factura": "string o null",
  "fecha_emision": "YYYY-MM-DD o null",
  "fecha_vencimiento": "YYYY-MM-DD o null",
  "proveedor_nombre": "string o null",
  "proveedor_nif": "string o null",
  "cliente_nombre": "string o null",
  "importe_base": número o null,
  "impuestos": número o null,
  "importe_total": número o null,
  "moneda": "EUR/USD/MXN/etc o null",
  "concepto": "string o null"
}

Si un campo no está visible en la factura, usa null. No incluyas texto fuera del JSON."""

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=prompt_sistema,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_mime,
                            "data": datos_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extrae todos los datos de esta factura.",
                    },
                ],
            }
        ],
    )

    # Parsear la respuesta JSON
    texto_respuesta = mensaje.content[0].text.strip()

    # Limpiar posibles bloques de código markdown
    if texto_respuesta.startswith("```"):
        lineas = texto_respuesta.split("\n")
        texto_respuesta = "\n".join(lineas[1:-1])

    datos_dict = json.loads(texto_respuesta)
    return DatosFactura(**datos_dict)


# --- Uso ---
if __name__ == "__main__":
    # Reemplaza con la ruta a una imagen de factura real
    factura = extraer_datos_factura("factura.jpg")

    print("=== DATOS EXTRAÍDOS DE LA FACTURA ===")
    print(f"Número:      {factura.numero_factura}")
    print(f"Fecha:       {factura.fecha_emision}")
    print(f"Proveedor:   {factura.proveedor_nombre} ({factura.proveedor_nif})")
    print(f"Cliente:     {factura.cliente_nombre}")
    print(f"Base:        {factura.importe_base} {factura.moneda}")
    print(f"Impuestos:   {factura.impuestos} {factura.moneda}")
    print(f"Total:       {factura.importe_total} {factura.moneda}")
    print(f"Concepto:    {factura.concepto}")

    # También disponible como diccionario
    print("\nJSON completo:")
    print(factura.model_dump_json(indent=2))
```

---

## 5. Caso práctico — comparar dos imágenes

Claude puede recibir múltiples imágenes en el mismo mensaje. Esto permite comparar versiones de un diseño, detectar diferencias visuales o analizar relaciones entre imágenes.

```python
import anthropic
import base64
from pathlib import Path


def cargar_imagen_base64(ruta: str) -> tuple[str, str]:
    """Carga una imagen y devuelve (datos_base64, tipo_mime)."""
    ruta = Path(ruta)
    with open(ruta, "rb") as f:
        datos = base64.standard_b64encode(f.read()).decode("utf-8")
    tipos = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    tipo = tipos.get(ruta.suffix.lower(), "image/jpeg")
    return datos, tipo


def comparar_imagenes(
    ruta_imagen_a: str,
    ruta_imagen_b: str,
    instruccion: str = "Compara estas dos imágenes en detalle. ¿Qué diferencias y similitudes encuentras?",
) -> str:
    """
    Envía dos imágenes a Claude y obtiene un análisis comparativo.

    Args:
        ruta_imagen_a: Ruta a la primera imagen.
        ruta_imagen_b: Ruta a la segunda imagen.
        instruccion: Pregunta o instrucción de comparación.

    Returns:
        Análisis comparativo de Claude.
    """
    datos_a, tipo_a = cargar_imagen_base64(ruta_imagen_a)
    datos_b, tipo_b = cargar_imagen_base64(ruta_imagen_b)

    cliente = anthropic.Anthropic()

    # Enviamos ambas imágenes en el mismo mensaje con etiquetas para identificarlas
    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Imagen A:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_a,
                            "data": datos_a,
                        },
                    },
                    {"type": "text", "text": "Imagen B:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_b,
                            "data": datos_b,
                        },
                    },
                    {"type": "text", "text": instruccion},
                ],
            }
        ],
    )

    return mensaje.content[0].text


# --- Uso: comparar dos versiones de un diseño ---
if __name__ == "__main__":
    analisis = comparar_imagenes(
        ruta_imagen_a="diseno_v1.png",
        ruta_imagen_b="diseno_v2.png",
        instruccion=(
            "Eres un revisor de diseño gráfico. "
            "Compara la Imagen A (versión 1) con la Imagen B (versión 2). "
            "Lista todos los cambios detectados: tipografía, colores, layout, elementos nuevos o eliminados."
        ),
    )

    print("=== ANÁLISIS COMPARATIVO ===")
    print(analisis)
```

---

## 6. Caso práctico — transcribir capturas de pantalla y documentos escaneados

Los LLMs con visión pueden leer texto de imágenes con mucha más flexibilidad que el OCR clásico: entienden el contexto, corrigen errores de escritura y pueden extraer solo la información relevante.

```python
import anthropic
import base64
from pathlib import Path


def transcribir_documento(ruta_imagen: str, modo: str = "completo") -> str:
    """
    Transcribe el texto visible en una imagen o captura de pantalla.

    Args:
        ruta_imagen: Ruta a la imagen del documento.
        modo: "completo" para transcripción literal,
              "limpio" para texto estructurado sin ruido,
              "resumen" para un resumen del contenido.

    Returns:
        Texto transcrito o resumido.
    """
    ruta = Path(ruta_imagen)
    with open(ruta, "rb") as f:
        datos_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    tipos = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    tipo_mime = tipos.get(ruta.suffix.lower(), "image/jpeg")

    prompts = {
        "completo": (
            "Transcribe todo el texto visible en esta imagen exactamente como aparece, "
            "manteniendo la estructura (párrafos, listas, tablas). "
            "No omitas ningún texto aunque parezca irrelevante."
        ),
        "limpio": (
            "Extrae y estructura el texto de este documento. "
            "Ignora cabeceras, pies de página repetitivos y elementos decorativos. "
            "Presenta el contenido principal de forma legible con jerarquía clara."
        ),
        "resumen": (
            "Lee el texto de esta imagen y proporciona: "
            "1) Un resumen de 2-3 oraciones del contenido principal. "
            "2) Los puntos o datos clave en formato de lista. "
            "3) El tipo de documento que parece ser."
        ),
    }

    instruccion = prompts.get(modo, prompts["completo"])

    cliente = anthropic.Anthropic()

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_mime,
                            "data": datos_b64,
                        },
                    },
                    {"type": "text", "text": instruccion},
                ],
            }
        ],
    )

    return mensaje.content[0].text


def transcribir_captura_codigo(ruta_imagen: str) -> str:
    """
    Caso especial: transcribe una captura de pantalla de código fuente.
    Útil para convertir imágenes de código a texto editable.
    """
    ruta = Path(ruta_imagen)
    with open(ruta, "rb") as f:
        datos_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    tipos = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    tipo_mime = tipos.get(ruta.suffix.lower(), "image/png")

    cliente = anthropic.Anthropic()

    mensaje = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": tipo_mime,
                            "data": datos_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Esta imagen contiene código fuente. "
                            "Transcríbelo exactamente, respetando la indentación y los espacios. "
                            "Identifica el lenguaje de programación y devuelve el código "
                            "dentro de un bloque de código markdown con el lenguaje correcto."
                        ),
                    },
                ],
            }
        ],
    )

    return mensaje.content[0].text


# --- Uso ---
if __name__ == "__main__":
    # Transcripción completa de un documento escaneado
    texto = transcribir_documento("contrato_escaneado.png", modo="limpio")
    print("=== TRANSCRIPCIÓN ===")
    print(texto)

    print("\n" + "=" * 50 + "\n")

    # Código fuente desde una captura de pantalla
    codigo = transcribir_captura_codigo("screenshot_codigo.png")
    print("=== CÓDIGO TRANSCRITO ===")
    print(codigo)
```

---

## 7. Límites y consideraciones

### Tamaño y formatos

| Parámetro | Límite |
|---|---|
| Formatos soportados | JPEG, PNG, GIF, WebP |
| Tamaño máximo por imagen | 20 MB (recomendado < 5 MB) |
| Imágenes por mensaje | Hasta 20 (varía según modelo) |
| Resolución efectiva | Claude redimensiona internamente imágenes muy grandes |

### Rendimiento y coste

- Las imágenes consumen tokens adicionales. Una imagen de 1024×1024 px puede equivaler a ~1.600 tokens de entrada.
- Para imágenes grandes, Claude usa un modo de "alta resolución" que las divide en tiles y los procesa por separado.
- Si solo necesitas texto de una imagen, un tamaño de 800–1200 px de ancho suele ser suficiente.

### Privacidad y datos sensibles

- **No envíes imágenes con datos personales innecesarios** (rostros, DNI, datos médicos) a APIs externas si no tienes base legal para ello.
- Para entornos regulados (salud, finanzas, GDPR) considera usar modelos desplegados en tu infraestructura o con acuerdos de procesamiento de datos adecuados.
- Las imágenes enviadas a la API de Anthropic siguen su política de privacidad; no se usan para reentrenar modelos por defecto.

### Qué no puede hacer visión

- No puede leer texto en imágenes muy pequeñas o borrosas con fiabilidad del 100%.
- No identifica personas por su rostro (por política de uso).
- No analiza vídeos directamente (solo fotogramas individuales).
- Las coordenadas exactas de elementos en pantalla pueden ser imprecisas.

---

## 8. Extensiones sugeridas

- **Pipeline de procesamiento en lote**: procesa carpetas completas de documentos escaneados en paralelo con `asyncio` y la API asíncrona de Anthropic.
- **Integración con bases de datos**: guarda los datos extraídos de facturas directamente en PostgreSQL o SQLite con SQLAlchemy.
- **API web para OCR**: crea un endpoint FastAPI que reciba imágenes por POST y devuelva los datos estructurados.
- **Validación con múltiples pasadas**: envía la misma factura dos veces y cruza los resultados para detectar errores de extracción.
- **Análisis de accesibilidad**: genera automáticamente texto alternativo (alt text) para imágenes en sitios web.

---

**Siguiente:** [02 — Generación de imágenes con IA](./02-generacion-imagenes.md)
