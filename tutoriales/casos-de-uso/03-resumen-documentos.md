# 03 — Resumen automático de documentos

> **Bloque:** Casos de uso · **Nivel:** Práctico · **Tiempo estimado:** 30 min

---

## Índice

1. [Objetivo](#1-objetivo)
2. [Resumen de texto corto](#2-resumen-de-texto-corto)
3. [El problema del texto largo](#3-el-problema-del-texto-largo)
4. [Estrategia Map-Reduce](#4-estrategia-map-reduce)
5. [Estrategia Refine](#5-estrategia-refine)
6. [Resumen con formato estructurado](#6-resumen-con-formato-estructurado)
7. [Pipeline completo para ficheros](#7-pipeline-completo-para-ficheros)
8. [Cuándo usar cada estrategia](#8-cuándo-usar-cada-estrategia)

---

## 1. Objetivo

Construir un sistema que resuma automáticamente documentos de cualquier longitud usando la API de Claude. Se cubren:

- Resumen de textos cortos (caben en el contexto)
- Estrategia **Map-Reduce** para documentos muy largos
- Estrategia **Refine** para mayor coherencia
- Generación de resúmenes estructurados (ejecutivos, bullets, etc.)

---

## 2. Resumen de texto corto

Para textos que caben en el contexto (hasta ~150.000 palabras con Claude):

```python
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def resumir(texto: str,
            estilo: str = "ejecutivo",
            longitud: str = "media",
            idioma: str = "español") -> str:
    """
    Resume un texto con el estilo y longitud indicados.

    estilo:   ejecutivo | bullets | tecnico | divulgativo
    longitud: corta (3-5 frases) | media (1-2 párrafos) | larga (3-5 párrafos)
    """

    instrucciones_estilo = {
        "ejecutivo": "Resume de forma ejecutiva: idea principal, puntos clave y conclusión.",
        "bullets":   "Resume en bullets (•) concisos, máximo 7 puntos.",
        "tecnico":   "Resume manteniendo la terminología técnica y los datos numéricos.",
        "divulgativo": "Resume en lenguaje sencillo apto para público no especializado.",
    }

    instrucciones_longitud = {
        "corta":  "El resumen debe ser de 3-5 frases.",
        "media":  "El resumen debe ocupar 1-2 párrafos.",
        "larga":  "El resumen puede ocupar hasta 5 párrafos.",
    }

    prompt = f"""Resume el siguiente texto.

Estilo: {instrucciones_estilo.get(estilo, instrucciones_estilo['ejecutivo'])}
Longitud: {instrucciones_longitud.get(longitud, instrucciones_longitud['media'])}
Idioma de respuesta: {idioma}

Texto:
<documento>
{texto}
</documento>

Escribe únicamente el resumen, sin introducción ni comentarios adicionales."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return respuesta.content[0].text


# Ejemplo
texto_corto = """
La inteligencia artificial generativa ha transformado radicalmente el panorama tecnológico
en los últimos años. Modelos como GPT-4 de OpenAI y Claude de Anthropic han demostrado
capacidades sorprendentes en generación de texto, código y análisis de información compleja.
Las empresas de todos los sectores están integrando estas tecnologías para automatizar tareas,
mejorar la atención al cliente y acelerar la innovación. Sin embargo, este avance también
plantea importantes preguntas éticas sobre la privacidad, los sesgos y el impacto en el
empleo. Los expertos coinciden en que la clave está en una adopción responsable y regulada
que maximice los beneficios mientras minimiza los riesgos.
"""

print("=== Resumen ejecutivo ===")
print(resumir(texto_corto, estilo="ejecutivo", longitud="corta"))

print("\n=== Resumen en bullets ===")
print(resumir(texto_corto, estilo="bullets", longitud="media"))
```

---

## 3. El problema del texto largo

Los LLMs tienen una **ventana de contexto finita**. Claude soporta hasta 200.000 tokens (~150.000 palabras), pero:

- Los documentos muy largos superan ese límite
- Incluir todo el texto en cada llamada es muy caro
- La calidad puede degradarse en los extremos del contexto (efecto "lost in the middle")

**Solución:** dividir el documento en fragmentos (chunks) y aplicar estrategias de agregación.

```python
def dividir_en_chunks(texto: str,
                      tamano_chunk: int = 3000,
                      solapamiento: int = 200) -> list[str]:
    """
    Divide el texto en fragmentos con solapamiento para no perder contexto
    entre chunks adyacentes.

    tamano_chunk: número aproximado de palabras por chunk
    solapamiento: palabras que se comparten entre chunks consecutivos
    """
    palabras = texto.split()
    chunks = []
    inicio = 0

    while inicio < len(palabras):
        fin = min(inicio + tamano_chunk, len(palabras))
        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk)

        if fin == len(palabras):
            break

        inicio = fin - solapamiento  # Retroceder para el solapamiento

    return chunks


# Ejemplo
texto_largo = "palabra " * 10000  # Simulación de documento largo
chunks = dividir_en_chunks(texto_largo, tamano_chunk=2000, solapamiento=100)
print(f"Documento de {len(texto_largo.split())} palabras → {len(chunks)} chunks")
```

---

## 4. Estrategia Map-Reduce

La más eficiente. Resumen cada chunk por separado (Map), luego combina los resúmenes (Reduce).

```python
from tqdm import tqdm


def resumir_chunk(chunk: str, numero: int, total: int) -> str:
    """Resume un fragmento del documento."""
    prompt = f"""Estás resumiendo un fragmento ({numero}/{total}) de un documento más largo.
Resume los puntos más importantes de ESTE fragmento. Sé conciso (máximo 3-4 frases).

Fragmento:
<fragmento>
{chunk}
</fragmento>"""

    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku para los chunks (más barato)
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta.content[0].text


def combinar_resumenes(resumenes: list[str], contexto: str = "") -> str:
    """Combina los resúmenes parciales en uno final cohesivo."""
    resumenes_str = "\n\n".join([
        f"Fragmento {i+1}:\n{r}"
        for i, r in enumerate(resumenes)
    ])

    prompt = f"""A continuación tienes los resúmenes de los fragmentos de un documento.
{"Contexto del documento: " + contexto if contexto else ""}

Crea un resumen final cohesivo y bien estructurado que integre todos los puntos importantes.
El resumen final debe tener entre 2 y 4 párrafos.

Resúmenes parciales:
{resumenes_str}"""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",  # Sonnet para la síntesis final
        max_tokens=1024,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta.content[0].text


def resumir_documento_map_reduce(texto: str,
                                  tamano_chunk: int = 3000,
                                  contexto: str = "") -> dict:
    """
    Resume un documento largo usando estrategia Map-Reduce.
    Devuelve el resumen final y metadata del proceso.
    """
    # 1. Dividir
    chunks = dividir_en_chunks(texto, tamano_chunk=tamano_chunk)
    print(f"Documento dividido en {len(chunks)} fragmentos")

    # 2. Map: resumir cada chunk
    resumenes_parciales = []
    for i, chunk in enumerate(tqdm(chunks, desc="Resumiendo fragmentos")):
        resumen = resumir_chunk(chunk, i + 1, len(chunks))
        resumenes_parciales.append(resumen)

    # 3. Reduce: combinar
    print("Combinando resúmenes...")
    resumen_final = combinar_resumenes(resumenes_parciales, contexto)

    return {
        "resumen": resumen_final,
        "num_chunks": len(chunks),
        "resumenes_parciales": resumenes_parciales
    }
```

---

## 5. Estrategia Refine

Más lenta pero con mejor coherencia narrativa. Cada chunk actualiza/refina el resumen anterior.

```python
def resumir_documento_refine(texto: str, tamano_chunk: int = 3000) -> str:
    """
    Resume un documento largo usando estrategia Refine.
    Más coherente que Map-Reduce pero más lenta.
    """
    chunks = dividir_en_chunks(texto, tamano_chunk=tamano_chunk)
    print(f"Procesando {len(chunks)} fragmentos con estrategia Refine...")

    # Resumen inicial del primer chunk
    resumen_actual = resumir_chunk(chunks[0], 1, len(chunks))

    # Refinar con cada chunk siguiente
    for i, chunk in enumerate(tqdm(chunks[1:], desc="Refinando"), start=2):
        prompt = f"""Tienes un resumen parcial de un documento y un nuevo fragmento del mismo.
Actualiza el resumen incorporando la información relevante del nuevo fragmento.
El resumen actualizado debe ser cohesivo y no exceder 3-4 párrafos.

Resumen actual:
<resumen>
{resumen_actual}
</resumen>

Nuevo fragmento ({i}/{len(chunks)}):
<fragmento>
{chunk}
</fragmento>

Escribe únicamente el resumen actualizado."""

        respuesta = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        resumen_actual = respuesta.content[0].text

    return resumen_actual
```

---

## 6. Resumen con formato estructurado

Para informes ejecutivos, actas de reunión, artículos técnicos, etc.

```python
import json


def resumir_estructurado(texto: str, tipo_documento: str = "informe") -> dict:
    """
    Genera un resumen estructurado según el tipo de documento.
    Devuelve un dict con los campos relevantes.
    """

    esquemas = {
        "informe": {
            "descripcion": "Informe técnico o de negocio",
            "campos": ["titulo_sugerido", "resumen_ejecutivo", "puntos_clave",
                      "conclusiones", "proximos_pasos", "palabras_clave"]
        },
        "acta": {
            "descripcion": "Acta de reunión",
            "campos": ["fecha", "participantes", "temas_tratados",
                      "decisiones", "tareas_asignadas", "proxima_reunion"]
        },
        "articulo": {
            "descripcion": "Artículo académico o de divulgación",
            "campos": ["titulo", "resumen", "metodologia", "resultados",
                      "conclusiones", "limitaciones"]
        }
    }

    esquema = esquemas.get(tipo_documento, esquemas["informe"])

    campos_json = {campo: "..." for campo in esquema["campos"]}
    campos_str = json.dumps(campos_json, ensure_ascii=False, indent=2)

    prompt = f"""Analiza el siguiente {esquema['descripcion']} y extrae la información
estructurada según el formato JSON indicado.

Formato de respuesta (JSON válido):
{campos_str}

Notas:
- "puntos_clave", "temas_tratados", "tareas_asignadas" deben ser listas de strings
- "palabras_clave" debe ser una lista de 5-8 términos relevantes
- Usa null si un campo no aplica o no está en el texto

Documento:
<documento>
{texto}
</documento>

Responde ÚNICAMENTE con el JSON, sin texto adicional."""

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(respuesta.content[0].text)
    except json.JSONDecodeError:
        return {"error": "No se pudo parsear la respuesta", "raw": respuesta.content[0].text}
```

---

## 7. Pipeline completo para ficheros

```python
from pathlib import Path


def resumir_fichero(ruta_fichero: str,
                    estrategia: str = "auto",
                    formato: str = "texto") -> dict:
    """
    Lee un fichero de texto y lo resume.

    estrategia: auto | map_reduce | refine | directo
    formato:    texto | estructurado
    """
    ruta = Path(ruta_fichero)

    # Leer el fichero
    if ruta.suffix == ".txt":
        texto = ruta.read_text(encoding="utf-8")
    elif ruta.suffix == ".md":
        texto = ruta.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Formato no soportado: {ruta.suffix}. Usa .txt o .md")

    num_palabras = len(texto.split())
    print(f"Documento: {ruta.name} ({num_palabras:,} palabras)")

    # Elegir estrategia automáticamente
    if estrategia == "auto":
        if num_palabras < 3000:
            estrategia = "directo"
        elif num_palabras < 30000:
            estrategia = "map_reduce"
        else:
            estrategia = "map_reduce"  # Refine es demasiado lento para >30K palabras

    print(f"Estrategia: {estrategia}")

    # Resumir
    if estrategia == "directo":
        resumen = resumir(texto)
        resultado = {"resumen": resumen, "estrategia": "directo"}
    elif estrategia == "map_reduce":
        resultado = resumir_documento_map_reduce(texto)
        resultado["estrategia"] = "map_reduce"
    elif estrategia == "refine":
        resumen = resumir_documento_refine(texto)
        resultado = {"resumen": resumen, "estrategia": "refine"}

    if formato == "estructurado":
        resultado["estructurado"] = resumir_estructurado(resultado["resumen"])

    return resultado


# Uso
resultado = resumir_fichero("informe_anual.txt", estrategia="auto", formato="estructurado")
print(resultado["resumen"])
print(json.dumps(resultado.get("estructurado", {}), ensure_ascii=False, indent=2))
```

---

## 8. Cuándo usar cada estrategia

| Situación | Estrategia recomendada |
|---|---|
| Texto < 3.000 palabras | Directo (una sola llamada) |
| Texto 3.000–50.000 palabras | Map-Reduce (eficiente y suficiente) |
| Texto > 50.000 palabras | Map-Reduce con chunks más grandes |
| Necesitas alta coherencia narrativa | Refine (más lento, mejor resultado) |
| Resumen de múltiples documentos | Map-Reduce sobre todos, luego Reduce global |
| Necesitas campos estructurados | Resumen directo + post-proceso estructurado |

---

**Anterior:** [02 — Clasificación de texto](./02-clasificacion-texto.md) · **Siguiente:** [04 — Extracción de PDFs](./04-extraccion-pdfs.md)
