# Batch API: Procesamiento Masivo Asíncrono

## ¿Qué es la Batch API?

La Batch API de Anthropic permite enviar **miles de solicitudes en un único lote**
que se procesan de forma asíncrona. A cambio de no necesitar los resultados en tiempo
real, obtienes hasta un **50% de descuento** en el coste por token.

**Ideal para:**
- Clasificación masiva de textos (miles de emails, reseñas, tickets)
- Generación de resúmenes para un catálogo completo de productos
- Evaluación automatizada de respuestas LLM
- Extracción de datos estructurados de documentos en masa
- Re-procesamiento de datasets históricos

## Flujo de trabajo

```
1. Crear batch con lista de requests → obtener batch_id
2. Esperar (polling o webhook) hasta que status = "ended"
3. Descargar resultados con download_url
4. Procesar resultados exitosos y reintentar fallos
```

El tiempo de procesamiento es de **hasta 24 horas**, pero en práctica suele
completarse en minutos a pocas horas para lotes medianos.

## Crear un batch

```python
import anthropic
import json

client = anthropic.Anthropic()

# Preparar las solicitudes del batch
requests = []
textos = [
    "Me encanta este producto, superó todas mis expectativas",
    "Muy decepcionante, llegó roto y el servicio es pésimo",
    "Cumple su función, nada especial",
    # ... hasta miles de textos
]

for i, texto in enumerate(textos):
    requests.append({
        "custom_id": f"clasificacion_{i:04d}",   # ID único por request
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": f"Clasifica el sentimiento: '{texto}'. Responde solo: positivo/negativo/neutro"
            }]
        }
    })

# Enviar el batch
batch = client.messages.batches.create(requests=requests)
print(f"Batch creado: {batch.id}")
print(f"Estado inicial: {batch.processing_status}")
print(f"Requests: {batch.request_counts}")
```

## Formato de request con todas las opciones

```python
request_completo = {
    "custom_id": "mi_request_001",
    "params": {
        "model": "claude-haiku-4-5-20251001",    # Cualquier modelo disponible
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Tu prompt aquí"}
        ],
        "system": "Eres un asistente experto en...",  # Opcional
        "temperature": 0.3,                            # Opcional
        "top_p": 0.9,                                  # Opcional
        "stop_sequences": ["\n\n"],                    # Opcional
        "metadata": {"user_id": "u_123"}               # Opcional
    }
}
```

## Monitorizar el estado del batch

```python
import time

def esperar_batch(client, batch_id: str, intervalo: int = 30) -> object:
    """Espera hasta que el batch termine con polling periódico."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired

        print(f"[{batch.processing_status}] "
              f"✓{counts.succeeded} ✗{counts.errored} ⏳{counts.processing}/{total}")

        if batch.processing_status == "ended":
            return batch

        time.sleep(intervalo)

batch_final = esperar_batch(client, batch.id)
```

## Descargar y procesar resultados

```python
# Iterar sobre los resultados (descarga streaming)
resultados_ok = []
resultados_error = []

for resultado in client.messages.batches.results(batch.id):
    custom_id = resultado.custom_id

    if resultado.result.type == "succeeded":
        texto_respuesta = resultado.result.message.content[0].text
        resultados_ok.append({
            "id": custom_id,
            "respuesta": texto_respuesta,
            "tokens": resultado.result.message.usage.output_tokens
        })

    elif resultado.result.type == "errored":
        resultados_error.append({
            "id": custom_id,
            "error": resultado.result.error.type,
            "mensaje": str(resultado.result.error)
        })

    elif resultado.result.type == "expired":
        print(f"⚠ {custom_id}: expirado sin procesar")

print(f"✓ Exitosos: {len(resultados_ok)}")
print(f"✗ Errores: {len(resultados_error)}")

# Guardar resultados
with open("batch_resultados.jsonl", "w") as f:
    for r in resultados_ok:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

## Listar y gestionar batches

```python
# Listar batches recientes
for batch in client.messages.batches.list(limit=10):
    print(f"{batch.id[:12]}... | {batch.processing_status} | "
          f"{batch.request_counts.succeeded}/{batch.request_counts.processing + batch.request_counts.succeeded} | "
          f"creado: {batch.created_at}")

# Cancelar un batch en progreso
client.messages.batches.cancel(batch_id="msgbatch_XXXX")

# Eliminar resultados (limpieza)
client.messages.batches.delete(batch_id="msgbatch_XXXX")
```

## Guardar batch localmente y reanudar

```python
import json

# Guardar ID para retomar en otra sesión
estado = {"batch_id": batch.id, "total_requests": len(requests)}
with open("batch_estado.json", "w") as f:
    json.dump(estado, f)

# En otra sesión: retomar
with open("batch_estado.json") as f:
    estado = json.load(f)

batch_retomado = client.messages.batches.retrieve(estado["batch_id"])
print(f"Estado: {batch_retomado.processing_status}")
```

## Pipeline completo: CSV → Batch → CSV enriquecido

```python
import pandas as pd
import anthropic
import json
import time

client = anthropic.Anthropic()

# 1. Cargar datos
df = pd.read_csv("productos.csv")  # Columnas: id, nombre, descripcion

# 2. Crear requests
requests = []
for _, fila in df.iterrows():
    requests.append({
        "custom_id": str(fila["id"]),
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 150,
            "messages": [{
                "role": "user",
                "content": f"""Categoriza este producto y extrae palabras clave.
Nombre: {fila['nombre']}
Descripción: {fila['descripcion']}
Responde JSON: {{"categoria": "...", "subcategoria": "...", "keywords": ["k1","k2","k3"]}}"""
            }]
        }
    })

# 3. Enviar y esperar
batch = client.messages.batches.create(requests=requests)
print(f"Batch {batch.id} enviado con {len(requests)} productos")

while True:
    b = client.messages.batches.retrieve(batch.id)
    if b.processing_status == "ended":
        break
    time.sleep(60)

# 4. Procesar resultados
enriquecidos = {}
for r in client.messages.batches.results(batch.id):
    if r.result.type == "succeeded":
        try:
            datos = json.loads(r.result.message.content[0].text)
            enriquecidos[r.custom_id] = datos
        except json.JSONDecodeError:
            pass

# 5. Enriquecer DataFrame
df["categoria"] = df["id"].astype(str).map(lambda x: enriquecidos.get(x, {}).get("categoria", ""))
df["keywords"] = df["id"].astype(str).map(lambda x: ", ".join(enriquecidos.get(x, {}).get("keywords", [])))
df.to_csv("productos_enriquecidos.csv", index=False)
print(f"Guardados {len(df)} productos enriquecidos")
```

## Comparativa: Batch API vs llamadas síncronas

| Aspecto | Llamadas síncronas | Batch API |
|---------|--------------------|-----------|
| Latencia | Inmediata | Hasta 24h |
| Coste | Precio normal | -50% |
| Rate limits | Afectado | No afectado |
| Paralelismo | Limitado por TPS | Gestionado automáticamente |
| Manejo de errores | Manual | Automático por request |
| Ideal para | Tiempo real | Procesamiento offline |

## Buenas prácticas

- Usa `custom_id` descriptivos para facilitar el debug (`email_001`, `ticket_abc`)
- Agrupa requests del mismo tipo de tarea en un solo batch
- Implementa reintentos para los requests con `errored`
- Guarda el `batch_id` persistentemente (base de datos o archivo)
- Para lotes >10.000 requests, divide en sub-batches de 10.000

## Recursos

- [Documentación Batch API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [Notebook interactivo](../notebooks/apis-avanzadas/02-batch-api.ipynb)
