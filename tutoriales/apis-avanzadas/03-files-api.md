# Files API: Gestión de Documentos en la API

## ¿Qué es la Files API?

La Files API permite **subir documentos una vez y referenciarlos múltiples veces**
en diferentes llamadas a la API. En lugar de enviar el contenido del documento en
cada request (consumiendo tokens de entrada), subes el archivo y usas su `file_id`.

**Ventajas:**
- Reducción de latencia (no hay que transmitir el documento en cada llamada)
- Reducción de costes (los tokens de entrada del documento se cobran solo una vez si se combina con Prompt Caching)
- Reutilización entre usuarios o sesiones
- Soporte para documentos grandes (PDFs, CSVs, código fuente)

## Formatos soportados

| Tipo | Extensiones | Límite |
|------|------------|--------|
| Documentos | PDF, TXT, MD, HTML, XML | 32 MB |
| Hojas de cálculo | CSV | 32 MB |
| Código | py, js, ts, java, cpp, etc. | 32 MB |
| Imágenes | PNG, JPG, GIF, WebP | 20 MB |

Límite total por organización: configurado en el panel de Anthropic.

## Subir un archivo

```python
import anthropic

client = anthropic.Anthropic()

# Subir un PDF
with open("informe_anual.pdf", "rb") as f:
    archivo = client.beta.files.upload(
        file=("informe_anual.pdf", f, "application/pdf")
    )

print(f"ID del archivo: {archivo.id}")
print(f"Nombre: {archivo.filename}")
print(f"Tamaño: {archivo.size} bytes")
print(f"Creado: {archivo.created_at}")
```

## Usar el archivo en una llamada

```python
# Referenciar el archivo por su ID
respuesta = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": archivo.id      # Usar file_id en lugar de base64
                }
            },
            {
                "type": "text",
                "text": "Resume los puntos principales de este informe en 5 bullets"
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)

print(respuesta.content[0].text)
```

## Subir texto plano y CSV

```python
import io

# Texto plano
contenido_txt = "Política de privacidad versión 3.2\n\nArtículo 1..."
archivo_txt = client.beta.files.upload(
    file=("politica.txt", io.BytesIO(contenido_txt.encode()), "text/plain")
)

# CSV
import pandas as pd
df = pd.DataFrame({"ventas": [100, 200], "mes": ["Enero", "Febrero"]})
csv_bytes = df.to_csv(index=False).encode()
archivo_csv = client.beta.files.upload(
    file=("ventas.csv", io.BytesIO(csv_bytes), "text/csv")
)
print(f"CSV subido: {archivo_csv.id}")
```

## Imágenes con Files API

```python
# Subir imagen
with open("diagrama.png", "rb") as f:
    archivo_img = client.beta.files.upload(
        file=("diagrama.png", f, "image/png")
    )

# Usar en llamada multimodal
respuesta = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": archivo_img.id
                }
            },
            {
                "type": "text",
                "text": "Describe este diagrama técnicamente"
            }
        ]
    }],
    betas=["files-api-2025-04-14"]
)
```

## Listar y gestionar archivos

```python
# Listar todos los archivos
archivos = client.beta.files.list()
for archivo in archivos.data:
    tamaño_mb = archivo.size / 1024 / 1024
    print(f"{archivo.id} | {archivo.filename} | {tamaño_mb:.2f} MB | {archivo.created_at}")

# Obtener metadata de un archivo
info = client.beta.files.retrieve_metadata(archivo.id)
print(info)

# Eliminar un archivo (libera espacio)
client.beta.files.delete(archivo.id)
print("Archivo eliminado")
```

## Patrón: base de conocimiento compartida

```python
# ---- Setup (una vez) ----
class BaseConocimiento:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.archivos = {}

    def cargar_documento(self, ruta: str, nombre: str) -> str:
        """Sube un documento y guarda su file_id."""
        with open(ruta, "rb") as f:
            mime = "application/pdf" if ruta.endswith(".pdf") else "text/plain"
            archivo = self.client.beta.files.upload(
                file=(nombre, f, mime)
            )
        self.archivos[nombre] = archivo.id
        print(f"✓ {nombre} → {archivo.id}")
        return archivo.id

    def preguntar(self, pregunta: str, documentos: list[str]) -> str:
        """Responde basándose en los documentos especificados."""
        contenido = []
        for nombre_doc in documentos:
            if nombre_doc in self.archivos:
                contenido.append({
                    "type": "document",
                    "source": {"type": "file", "file_id": self.archivos[nombre_doc]}
                })
        contenido.append({"type": "text", "text": pregunta})

        resp = self.client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": contenido}],
            betas=["files-api-2025-04-14"]
        )
        return resp.content[0].text

# ---- Uso ----
kb = BaseConocimiento()
kb.cargar_documento("manual_producto.pdf", "manual")
kb.cargar_documento("faq.txt", "faq")

# Múltiples usuarios pueden hacer preguntas sin re-subir los documentos
respuesta = kb.preguntar("¿Cómo se configura el modo avanzado?", ["manual", "faq"])
print(respuesta)
```

## Combinar con Prompt Caching

Para máxima eficiencia, combina Files API + Prompt Caching:

```python
respuesta = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": "Eres un experto en el siguiente documento:"
        },
        {
            "type": "document",
            "source": {"type": "file", "file_id": archivo.id},
            "cache_control": {"type": "ephemeral"}   # Caché del documento
        }
    ],
    messages=[{"role": "user", "content": "¿Qué dice el artículo 5?"}],
    betas=["files-api-2025-04-14", "prompt-caching-2024-07-31"]
)
```

## Comparativa: base64 vs Files API

| Aspecto | Base64 en cada request | Files API |
|---------|----------------------|-----------|
| Latencia primera vez | Alta (transmitir doc) | Media (subir) |
| Latencia siguientes | Alta (re-transmitir) | Baja (solo file_id) |
| Coste tokens | Por request | Una vez (si + caching) |
| Tamaño máx. request | Limitado | 32 MB por archivo |
| Persistencia | No | Sí (hasta 30 días) |
| Compartir entre usuarios | No nativo | Sí por file_id |

## Recursos

- [Documentación Files API](https://docs.anthropic.com/en/docs/build-with-claude/files)
- [Notebook interactivo](../notebooks/apis-avanzadas/03-files-api.ipynb)
