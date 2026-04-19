# Citations API: Respuestas con Citas Verificables

## ¿Qué son las Citations?

La funcionalidad de Citations de Claude permite generar respuestas donde cada
afirmación está **vinculada a su fuente exacta** en los documentos proporcionados.
Claude indica el índice de caracteres o el número de página donde encontró la información.

**Beneficios:**
- Respuestas verificables y auditables
- Reduce alucinaciones al forzar grounding en el texto
- Ideal para aplicaciones legales, médicas o de investigación
- Compatible con RAG para atribución automática de fuentes

## Cómo funciona

```
Usuario → "¿Cuál es la política de devoluciones?"
Claude → Lee los documentos proporcionados
       → Identifica los fragmentos relevantes
       → Genera respuesta con referencias [doc:0, chars 145-230]
       → El cliente puede mostrar la cita destacada
```

## Sintaxis básica

```python
import anthropic

client = anthropic.Anthropic()

# Documentos de referencia
documentos = [
    {
        "type": "document",
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": """Política de devoluciones (versión 4.1, enero 2025):
Los productos pueden devolverse en un plazo de 30 días desde la fecha de compra.
El artículo debe estar en su embalaje original y sin uso.
Los gastos de envío de devolución corren a cargo del cliente salvo defecto de fábrica.
Para iniciar una devolución, contacte con soporte@empresa.com o llame al 900 123 456."""
        },
        "title": "Política de Devoluciones",
        "citations": {"enabled": True}    # Activar citas para este documento
    }
]

respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": documentos + [{
            "type": "text",
            "text": "¿Cuántos días tengo para devolver un producto y quién paga el envío?"
        }]
    }]
)

# Procesar respuesta con citas
for bloque in respuesta.content:
    if bloque.type == "text":
        print("RESPUESTA:", bloque.text)
    elif bloque.type == "citations":
        print("\nCITAS:")
        for cita in bloque.citations:
            print(f"  [{cita.document_index}] chars {cita.start_char_index}-{cita.end_char_index}:")
            print(f"  '{cita.cited_text}'")
```

## Múltiples documentos con citas

```python
documentos_empresa = [
    {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain",
                   "data": "Contrato de servicio v2.3...\nArtículo 5: El servicio incluye soporte 24/7 por email..."},
        "title": "Contrato de Servicio",
        "citations": {"enabled": True}
    },
    {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain",
                   "data": "FAQ actualizado diciembre 2024...\nP: ¿Tiene soporte telefónico? R: No, solo email y chat..."},
        "title": "FAQ",
        "citations": {"enabled": True}
    },
    {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain",
                   "data": "Precios 2025: Plan Básico 29€/mes, Plan Pro 79€/mes, Enterprise desde 299€/mes..."},
        "title": "Precios",
        "citations": {"enabled": True}
    }
]

respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=800,
    messages=[{
        "role": "user",
        "content": documentos_empresa + [{
            "type": "text",
            "text": "¿Qué incluye el soporte y cuáles son los precios?"
        }]
    }]
)

# Mostrar respuesta con fuentes
def mostrar_con_citas(respuesta):
    for bloque in respuesta.content:
        if bloque.type == "text":
            print(bloque.text)
        elif bloque.type == "citations":
            print(f"\n📎 Fuentes:")
            for c in bloque.citations:
                print(f"   [{c.document_index}: {c.document_title}] → \"{c.cited_text[:80]}...\"")

mostrar_con_citas(respuesta)
```

## Citations con PDF (Files API)

```python
# Subir PDF y usar citations
with open("manual_tecnico.pdf", "rb") as f:
    archivo = client.beta.files.upload(
        file=("manual_tecnico.pdf", f, "application/pdf")
    )

respuesta = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=600,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {"type": "file", "file_id": archivo.id},
                "title": "Manual Técnico",
                "citations": {"enabled": True}
            },
            {
                "type": "text",
                "text": "¿Cuáles son los requisitos mínimos del sistema?"
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)

# Para PDFs: las citas incluyen page_number en lugar de char_index
for bloque in respuesta.content:
    if bloque.type == "citations":
        for cita in bloque.citations:
            pagina = getattr(cita, "page_number", "N/A")
            print(f"Página {pagina}: '{cita.cited_text[:100]}'")
```

## RAG con Citations: pipeline completo

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Indexar documentos en ChromaDB
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
db = chromadb.EphemeralClient()
coleccion = db.create_collection("base_conocimiento")

fragmentos = [
    {"id": "doc1_p1", "texto": "El proceso de onboarding dura 5 días hábiles...", "fuente": "Manual Onboarding"},
    {"id": "doc1_p2", "texto": "Los accesos se crean automáticamente al firmar el contrato...", "fuente": "Manual Onboarding"},
    {"id": "doc2_p1", "texto": "Las facturas se emiten el día 1 de cada mes...", "fuente": "Política Facturación"},
]

embeddings = embedder.encode([f["texto"] for f in fragmentos]).tolist()
coleccion.add(
    ids=[f["id"] for f in fragmentos],
    embeddings=embeddings,
    documents=[f["texto"] for f in fragmentos],
    metadatas=[{"fuente": f["fuente"]} for f in fragmentos]
)

def rag_con_citations(pregunta: str, top_k: int = 3) -> dict:
    """RAG que devuelve respuesta con citas verificables."""
    # Recuperar documentos relevantes
    embedding_pregunta = embedder.encode([pregunta]).tolist()
    resultados = coleccion.query(query_embeddings=embedding_pregunta, n_results=top_k)

    # Construir documentos con citations habilitadas
    docs_claude = []
    for texto, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
        docs_claude.append({
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": texto},
            "title": meta["fuente"],
            "citations": {"enabled": True}
        })

    # Llamar a Claude
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": docs_claude + [{"type": "text", "text": pregunta}]
        }]
    )

    return {"respuesta": resp, "documentos_usados": docs_claude}

resultado = rag_con_citations("¿Cuándo recibiré mi primera factura?")
mostrar_con_citas(resultado["respuesta"])
```

## Estructura de una cita

```python
# Estructura de un objeto Citation:
{
    "type": "char_location",           # o "page_location" para PDFs
    "document_index": 0,               # índice del documento (0-based)
    "document_title": "Manual",        # título si se proporcionó
    "start_char_index": 145,           # inicio del fragmento citado
    "end_char_index": 230,             # fin del fragmento citado
    "cited_text": "El producto puede...", # texto exacto citado
}
```

## Mostrar citas en UI (ejemplo HTML)

```python
def generar_html_con_citas(respuesta, documentos_originales: list) -> str:
    """Genera HTML con la respuesta y citas interactivas."""
    html = "<div class='respuesta'>"
    for bloque in respuesta.content:
        if bloque.type == "text":
            html += f"<p>{bloque.text}</p>"
        elif bloque.type == "citations":
            for cita in bloque.citations:
                doc_titulo = cita.document_title or f"Documento {cita.document_index}"
                html += f"""<span class='cita' title='{cita.cited_text}'
                    data-doc='{doc_titulo}' data-start='{cita.start_char_index}'>
                    <sup>[{cita.document_index + 1}]</sup></span>"""
    html += "</div><div class='fuentes'><h4>Fuentes:</h4><ol>"
    vistos = set()
    for bloque in respuesta.content:
        if bloque.type == "citations":
            for cita in bloque.citations:
                key = f"{cita.document_index}"
                if key not in vistos:
                    vistos.add(key)
                    titulo = cita.document_title or f"Documento {cita.document_index}"
                    html += f"<li>{titulo}</li>"
    html += "</ol></div>"
    return html
```

## Cuándo usar Citations

| Caso | ¿Citations útil? | Alternativa si no |
|------|-----------------|-------------------|
| Atención al cliente con políticas | ✅ Sí | Prompt sin citations |
| Legal / compliance | ✅ Sí | Búsqueda manual |
| Investigación académica | ✅ Sí | — |
| Chatbot creativo | ❌ No | Respuesta libre |
| Extracción JSON | ❌ No | Structured output |
| Resumen de un solo doc | ⚠️ Opcional | Prompt simple |

## Recursos

- [Documentación Citations](https://docs.anthropic.com/en/docs/build-with-claude/citations)
- [Notebook interactivo](../notebooks/apis-avanzadas/04-citations-api.ipynb)
