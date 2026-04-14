# 04 — Extracción de información de PDFs con IA

> **Bloque:** Casos de uso · **Nivel:** Práctico · **Tiempo estimado:** 35 min

---

## Índice

1. [Objetivo](#1-objetivo)
2. [Requisitos](#2-requisitos)
3. [Leer texto de un PDF](#3-leer-texto-de-un-pdf)
4. [Extraer datos estructurados](#4-extraer-datos-estructurados)
5. [Procesar facturas](#5-procesar-facturas)
6. [Procesar contratos](#6-procesar-contratos)
7. [Pipeline de múltiples PDFs](#7-pipeline-de-múltiples-pdfs)
8. [PDFs con imágenes (visión)](#8-pdfs-con-imágenes-visión)
9. [Buenas prácticas](#9-buenas-prácticas)

---

## 1. Objetivo

Construir un pipeline que lea PDFs y extraiga información estructurada usando Claude: datos de facturas, cláusulas de contratos, métricas de informes, etc.

Casos de uso típicos:
- **Facturas:** extraer proveedor, importe, fecha, líneas de detalle
- **Contratos:** extraer partes, fechas, cláusulas clave, obligaciones
- **Informes:** extraer KPIs, conclusiones, datos tabulares
- **CVs:** extraer experiencia, formación, habilidades

---

## 2. Requisitos

```bash
pip install pypdf anthropic python-dotenv
# Opcional para PDFs con imágenes:
pip install pymupdf pillow
```

---

## 3. Leer texto de un PDF

```python
import anthropic
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def leer_pdf_texto(ruta_pdf: str) -> dict:
    """
    Extrae el texto de un PDF usando pypdf.
    Devuelve dict con texto por página y metadata.
    """
    import pypdf

    ruta = Path(ruta_pdf)
    if not ruta.exists():
        raise FileNotFoundError(f"PDF no encontrado: {ruta_pdf}")

    with open(ruta, "rb") as f:
        reader = pypdf.PdfReader(f)

        metadata = {
            "titulo": reader.metadata.title if reader.metadata else None,
            "autor": reader.metadata.author if reader.metadata else None,
            "num_paginas": len(reader.pages),
        }

        paginas = []
        for i, pagina in enumerate(reader.pages):
            texto = pagina.extract_text() or ""
            paginas.append({
                "numero": i + 1,
                "texto": texto.strip(),
                "num_caracteres": len(texto)
            })

    texto_completo = "\n\n".join([
        f"--- Página {p['numero']} ---\n{p['texto']}"
        for p in paginas if p["texto"]
    ])

    return {
        "metadata": metadata,
        "paginas": paginas,
        "texto_completo": texto_completo,
        "num_palabras": len(texto_completo.split())
    }


# Uso básico
pdf = leer_pdf_texto("documento.pdf")
print(f"Páginas: {pdf['metadata']['num_paginas']}")
print(f"Palabras: {pdf['num_palabras']:,}")
print(pdf["paginas"][0]["texto"][:500])
```

---

## 4. Extraer datos estructurados

```python
def extraer_con_schema(texto: str, schema: dict, instrucciones: str = "") -> dict:
    """
    Extrae datos estructurados de un texto según un schema JSON.

    texto:        Texto del documento
    schema:       Diccionario con los campos a extraer y su descripción
    instrucciones: Instrucciones específicas para el tipo de documento
    """

    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    prompt = f"""Extrae información del siguiente documento según el schema indicado.

{instrucciones}

Schema de extracción (devuelve SOLO este JSON, con los valores reales del documento):
{schema_str}

Reglas:
- Usa null para campos no encontrados en el documento
- Para listas, devuelve [] si no hay elementos
- Mantén los valores exactamente como aparecen en el documento (no traduzcas ni interpretes)
- Para fechas, usa formato ISO 8601 (YYYY-MM-DD) cuando sea posible

Documento:
<documento>
{texto[:15000]}  
</documento>

Responde ÚNICAMENTE con el JSON."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    texto_respuesta = respuesta.content[0].text.strip()

    # Limpiar posibles backticks de markdown
    if texto_respuesta.startswith("```"):
        texto_respuesta = texto_respuesta.split("```")[1]
        if texto_respuesta.startswith("json"):
            texto_respuesta = texto_respuesta[4:]

    try:
        return json.loads(texto_respuesta)
    except json.JSONDecodeError as e:
        return {"_error": f"JSON inválido: {e}", "_raw": texto_respuesta}
```

---

## 5. Procesar facturas

```python
SCHEMA_FACTURA = {
    "numero_factura": "string — número o código de la factura",
    "fecha_emision": "string — fecha en formato YYYY-MM-DD",
    "fecha_vencimiento": "string — fecha de vencimiento si aparece",
    "emisor": {
        "nombre": "string",
        "nif_cif": "string",
        "direccion": "string",
        "email": "string o null",
        "telefono": "string o null"
    },
    "receptor": {
        "nombre": "string",
        "nif_cif": "string",
        "direccion": "string"
    },
    "lineas": [
        {
            "descripcion": "string",
            "cantidad": "number",
            "precio_unitario": "number",
            "importe": "number"
        }
    ],
    "subtotal": "number — base imponible",
    "iva_porcentaje": "number — porcentaje de IVA",
    "iva_importe": "number — importe del IVA",
    "total": "number — importe total a pagar",
    "moneda": "string — EUR, USD, etc.",
    "metodo_pago": "string o null",
    "numero_cuenta_bancaria": "string o null",
    "notas": "string o null"
}

INSTRUCCIONES_FACTURA = """Estás procesando una factura comercial.
- Extrae todos los importes como números (sin símbolos de moneda)
- Si hay múltiples tipos de IVA, usa el más representativo
- Para el número de factura, incluye el formato completo (ej: FAC-2026-0042)"""


def procesar_factura(ruta_pdf: str) -> dict:
    """Extrae todos los datos de una factura en PDF."""
    pdf = leer_pdf_texto(ruta_pdf)
    datos = extraer_con_schema(
        pdf["texto_completo"],
        SCHEMA_FACTURA,
        INSTRUCCIONES_FACTURA
    )
    datos["_metadata_pdf"] = pdf["metadata"]
    return datos


# Uso
factura = procesar_factura("factura_proveedor.pdf")
print(json.dumps(factura, ensure_ascii=False, indent=2))

# Procesar múltiples facturas
import os
directorio = Path("facturas/")
resultados = []
for pdf_file in directorio.glob("*.pdf"):
    print(f"Procesando: {pdf_file.name}")
    datos = procesar_factura(str(pdf_file))
    datos["_fichero"] = pdf_file.name
    resultados.append(datos)

# Exportar a CSV
import pandas as pd
df = pd.json_normalize(resultados)
df.to_csv("facturas_procesadas.csv", index=False)
print(f"✓ {len(resultados)} facturas procesadas")
```

---

## 6. Procesar contratos

```python
SCHEMA_CONTRATO = {
    "tipo_contrato": "string — compraventa, arrendamiento, servicios, NDA, etc.",
    "fecha_firma": "string — YYYY-MM-DD o null",
    "fecha_inicio": "string — YYYY-MM-DD o null",
    "fecha_fin": "string — YYYY-MM-DD o null (null si indefinido)",
    "duracion": "string — descripción de la duración si no hay fechas exactas",
    "partes": [
        {
            "rol": "string — comprador/vendedor/arrendador/arrendatario/cliente/proveedor",
            "nombre": "string",
            "nif_cif": "string o null",
            "representante": "string o null"
        }
    ],
    "objeto": "string — descripción del objeto del contrato",
    "precio": {
        "importe": "number o null",
        "moneda": "string",
        "periodicidad": "string — único/mensual/anual/etc. o null",
        "forma_pago": "string o null"
    },
    "clausulas_clave": [
        {
            "titulo": "string — nombre o número de la cláusula",
            "resumen": "string — resumen en 1-2 frases"
        }
    ],
    "obligaciones_parte_a": ["lista de obligaciones principales"],
    "obligaciones_parte_b": ["lista de obligaciones principales"],
    "penalizaciones": "string — descripción de penalizaciones o null",
    "ley_aplicable": "string — jurisdicción y ley aplicable",
    "alertas": ["string — aspectos inusuales o de alto riesgo a revisar por un abogado"]
}

INSTRUCCIONES_CONTRATO = """Estás analizando un contrato legal.
- Sé preciso con las fechas y los importes
- Para 'clausulas_clave', incluye máximo 10 cláusulas de mayor relevancia
- Para 'alertas', señala condiciones inusuales, penalizaciones excesivas o ambigüedades
- IMPORTANTE: Esta extracción es solo informativa. Siempre recomienda revisión legal profesional"""


def analizar_contrato(ruta_pdf: str) -> dict:
    """Analiza un contrato en PDF y extrae los datos clave."""
    pdf = leer_pdf_texto(ruta_pdf)

    # Los contratos pueden ser muy largos, usar solo las primeras páginas si es necesario
    texto = pdf["texto_completo"]
    if pdf["num_palabras"] > 10000:
        print(f"Contrato largo ({pdf['num_palabras']:,} palabras), procesando primeras páginas...")

    datos = extraer_con_schema(texto, SCHEMA_CONTRATO, INSTRUCCIONES_CONTRATO)

    # Añadir advertencia legal
    datos["_advertencia"] = (
        "Esta extracción es automática e informativa. "
        "Consulte con un abogado antes de tomar decisiones basadas en este análisis."
    )

    return datos
```

---

## 7. Pipeline de múltiples PDFs

```python
from tqdm import tqdm
import time


def procesar_directorio(
    directorio: str,
    tipo: str = "generico",
    schema: dict = None,
    instrucciones: str = "",
    pausa: float = 0.5
) -> list[dict]:
    """
    Procesa todos los PDFs de un directorio.

    tipo: factura | contrato | generico
    """
    ruta_dir = Path(directorio)
    pdfs = list(ruta_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No se encontraron PDFs en {directorio}")
        return []

    print(f"Procesando {len(pdfs)} PDFs...")
    resultados = []
    errores = []

    for pdf_path in tqdm(pdfs, desc="Procesando PDFs"):
        try:
            if tipo == "factura":
                datos = procesar_factura(str(pdf_path))
            elif tipo == "contrato":
                datos = analizar_contrato(str(pdf_path))
            else:
                pdf = leer_pdf_texto(str(pdf_path))
                datos = extraer_con_schema(
                    pdf["texto_completo"],
                    schema or {},
                    instrucciones
                )

            datos["_fichero"] = pdf_path.name
            datos["_procesado_ok"] = True
            resultados.append(datos)

        except Exception as e:
            errores.append({"fichero": pdf_path.name, "error": str(e)})
            print(f"\nError en {pdf_path.name}: {e}")

        time.sleep(pausa)  # Respetar rate limits

    print(f"\n✓ {len(resultados)} procesados correctamente")
    if errores:
        print(f"✗ {len(errores)} con errores")

    return resultados


def guardar_resultados(resultados: list[dict], nombre_base: str = "resultado"):
    """Guarda los resultados en JSON y CSV."""
    # JSON completo
    with open(f"{nombre_base}.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # CSV (aplanado)
    df = pd.json_normalize(resultados, sep="_")
    df.to_csv(f"{nombre_base}.csv", index=False)

    print(f"Guardado en {nombre_base}.json y {nombre_base}.csv")


# Ejemplo completo
facturas = procesar_directorio("mis_facturas/", tipo="factura")
guardar_resultados(facturas, "facturas_2026")
```

---

## 8. PDFs con imágenes (visión)

Cuando el PDF contiene texto como imagen (escaneado) o gráficos, hay que usar visión.

```python
import base64
import fitz  # pymupdf


def pdf_a_imagenes(ruta_pdf: str, dpi: int = 150) -> list[bytes]:
    """Convierte cada página de un PDF en una imagen PNG."""
    doc = fitz.open(ruta_pdf)
    imagenes = []

    for pagina in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # Factor de escala
        pix = pagina.get_pixmap(matrix=mat)
        imagenes.append(pix.tobytes("png"))

    doc.close()
    return imagenes


def extraer_con_vision(ruta_pdf: str, instrucciones: str = "") -> str:
    """
    Extrae texto e información de un PDF escaneado usando visión de Claude.
    Útil para PDFs sin texto seleccionable.
    """
    imagenes = pdf_a_imagenes(ruta_pdf)
    print(f"PDF convertido a {len(imagenes)} imágenes")

    respuestas = []

    for i, img_bytes in enumerate(imagenes):
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        contenido = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64
                }
            },
            {
                "type": "text",
                "text": f"Página {i+1} de {len(imagenes)}. "
                        f"{instrucciones or 'Extrae todo el texto visible y cualquier dato relevante.'}"
            }
        ]

        respuesta = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": contenido}]
        )
        respuestas.append(respuesta.content[0].text)

    return "\n\n".join(respuestas)


# Usar para PDFs escaneados
texto_escaneado = extraer_con_vision(
    "factura_escaneada.pdf",
    instrucciones="Extrae todos los datos de esta factura: número, fecha, importes, partes."
)
print(texto_escaneado)
```

---

## 9. Buenas prácticas

**Validar los datos extraídos:**
```python
def validar_factura(datos: dict) -> list[str]:
    """Valida que los datos extraídos de una factura son coherentes."""
    problemas = []

    if datos.get("total") is None:
        problemas.append("No se encontró el importe total")

    if datos.get("subtotal") and datos.get("iva_importe") and datos.get("total"):
        total_calculado = datos["subtotal"] + datos["iva_importe"]
        if abs(total_calculado - datos["total"]) > 0.02:
            problemas.append(
                f"Total inconsistente: {datos['subtotal']} + {datos['iva_importe']} "
                f"= {total_calculado:.2f} ≠ {datos['total']}"
            )

    if not datos.get("numero_factura"):
        problemas.append("No se encontró el número de factura")

    return problemas
```

**Caché de resultados (evitar reprocesar):**
```python
import hashlib

def calcular_hash_pdf(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def procesar_con_cache(ruta_pdf: str, cache_dir: str = ".cache") -> dict:
    hash_pdf = calcular_hash_pdf(ruta_pdf)
    ruta_cache = Path(cache_dir) / f"{hash_pdf}.json"

    if ruta_cache.exists():
        print(f"Usando caché para {Path(ruta_pdf).name}")
        return json.loads(ruta_cache.read_text(encoding="utf-8"))

    datos = procesar_factura(ruta_pdf)
    ruta_cache.parent.mkdir(exist_ok=True)
    ruta_cache.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return datos
```

---

**Anterior:** [03 — Resumen de documentos](./03-resumen-documentos.md) · **Fin del bloque Casos de uso**

> Siguiente paso recomendado: explora los [Notebooks interactivos](../notebooks/) para practicar estos casos de uso con código ejecutable.
