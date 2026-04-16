# 05 — Mistral AI y Cohere

> **Bloque:** APIs de IA · **Nivel:** Intermedio · **Tiempo estimado:** 35 min

---

## Índice

1. [Mistral AI: modelos y filosofía](#1-mistral-ai-modelos-y-filosofía)
2. [API de Mistral: uso básico y avanzado](#2-api-de-mistral-uso-básico-y-avanzado)
3. [Cohere: especialización en empresa](#3-cohere-especialización-en-empresa)
4. [Embeddings y reranking con Cohere](#4-embeddings-y-reranking-con-cohere)
5. [Cuándo elegir Mistral vs Cohere vs Claude vs GPT-4](#5-cuándo-elegir-mistral-vs-cohere-vs-claude-vs-gpt-4)
6. [Extensiones sugeridas](#6-extensiones-sugeridas)

---

## 1. Mistral AI: modelos y filosofía

Mistral AI (empresa francesa) se distingue por:
- **Open weights**: sus modelos pequeños son open-source (Mistral 7B, Mixtral 8x7B)
- **Eficiencia**: los mejores ratios rendimiento/costo del mercado
- **Privacidad europea**: infraestructura en Europa, cumplimiento GDPR nativo

| Modelo | Parámetros | Contexto | Punto fuerte |
|---|---|---|---|
| **Mistral Small** | ~22B | 32K | Bajo costo, alta velocidad |
| **Mistral Large** | ~123B | 128K | Razonamiento complejo, multilingüe |
| **Codestral** | 22B | 32K | Código: completar, explicar, debuggear |
| **Mistral Embed** | — | — | Embeddings multilingual |
| **Mixtral 8x7B** | 47B (MoE) | 32K | Open source, deployable localmente |

---

## 2. API de Mistral: uso básico y avanzado

```bash
pip install mistralai
```

```python
# mistral_basics.py
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# ── Chat básico ──────────────────────────────────────────────────────────
response = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": "Eres un asistente experto en Python."},
        {"role": "user", "content": "¿Cuál es la diferencia entre un generator y un iterator?"}
    ],
    temperature=0.3,
    max_tokens=512
)
print(response.choices[0].message.content)
print(f"Tokens: {response.usage.total_tokens}")


# ── Streaming ────────────────────────────────────────────────────────────
with client.chat.stream(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Escribe una función Python para calcular números primos."}]
) as stream:
    for chunk in stream:
        if chunk.data.choices[0].delta.content:
            print(chunk.data.choices[0].delta.content, end="", flush=True)
print()


# ── Codestral: especializado en código ──────────────────────────────────
def completar_codigo(codigo_incompleto: str, lenguaje: str = "python") -> str:
    """Usa Codestral para completar código."""
    response = client.chat.complete(
        model="codestral-latest",
        messages=[{
            "role": "user",
            "content": (
                f"Completa este código {lenguaje}. "
                f"Devuelve SOLO el código completado, sin explicaciones:\n\n{codigo_incompleto}"
            )
        }],
        temperature=0.1  # temperatura baja para código
    )
    return response.choices[0].message.content


codigo = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    # Completar la función merge aquí
"""
print(completar_codigo(codigo))


# ── Function calling ─────────────────────────────────────────────────────
from mistralai.models import Tool, Function, FunctionParameters

herramientas_mistral = [
    Tool(
        function=Function(
            name="calcular_precio",
            description="Calcula el precio total con descuento e IVA",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "precio_base": {"type": "number", "description": "Precio sin IVA"},
                    "descuento_pct": {"type": "number", "description": "Porcentaje de descuento (0-100)"},
                    "iva_pct": {"type": "number", "description": "Porcentaje de IVA", "default": 21}
                },
                required=["precio_base"]
            )
        )
    )
]


def calcular_precio(precio_base: float, descuento_pct: float = 0, iva_pct: float = 21) -> dict:
    precio_con_descuento = precio_base * (1 - descuento_pct / 100)
    precio_final = precio_con_descuento * (1 + iva_pct / 100)
    return {
        "precio_base": precio_base,
        "descuento": round(precio_base * descuento_pct / 100, 2),
        "precio_con_iva": round(precio_final, 2)
    }


def agente_precios(consulta: str) -> str:
    import json
    messages = [{"role": "user", "content": consulta}]

    while True:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            tools=herramientas_mistral,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        if response.choices[0].finish_reason == "stop":
            return msg.content

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        for tc in (msg.tool_calls or []):
            args = json.loads(tc.function.arguments)
            resultado = calcular_precio(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": json.dumps(resultado)
            })
```

---

## 3. Cohere: especialización en empresa

Cohere se centra en aplicaciones empresariales, especialmente RAG y búsqueda semántica.

```bash
pip install cohere
```

```python
# cohere_basics.py
import cohere
import os

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])


# ── Chat con Command R+ ──────────────────────────────────────────────────
response = co.chat(
    model="command-r-plus",
    message="Explica el concepto de attention en transformers para un público no técnico.",
    preamble="Eres un divulgador científico especializado en IA.",
    temperature=0.4,
    max_tokens=512
)
print(response.text)


# ── RAG nativo con documentos (Grounded Generation) ────────────────────
response_rag = co.chat(
    model="command-r-plus",
    message="¿Qué dice la política de devoluciones?",
    documents=[
        {
            "title": "Política de devoluciones 2024",
            "snippet": "Los clientes pueden devolver cualquier producto en un plazo de 30 días desde la recepción. El artículo debe estar en su estado original con embalaje intacto. Las devoluciones por defecto de fabricación tienen un plazo de 2 años."
        },
        {
            "title": "FAQ — Tienda Online",
            "snippet": "Para iniciar una devolución, accede a 'Mi cuenta' > 'Mis pedidos' > 'Solicitar devolución'. Recibirás una etiqueta de envío por email en 24 horas."
        }
    ]
)
print(response_rag.text)

# Cohere incluye las citas en la respuesta
if response_rag.citations:
    print("\nCitas:")
    for cita in response_rag.citations:
        print(f"  - '{cita.text}' → fuente: {cita.document_ids}")
```

---

## 4. Embeddings y reranking con Cohere

Cohere tiene embeddings multilingüe de alta calidad y la mejor API de reranking del mercado.

```python
# cohere_embeddings_reranking.py
import cohere
import numpy as np

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])


# ── Embeddings multilingüe ────────────────────────────────────────────────
def obtener_embeddings(textos: list[str], tipo: str = "search_document") -> np.ndarray:
    """
    tipo: "search_document" para indexar, "search_query" para consultas,
          "classification" para clasificación, "clustering" para clustering
    """
    response = co.embed(
        texts=textos,
        model="embed-multilingual-v3.0",  # soporta 100+ idiomas
        input_type=tipo,
        embedding_types=["float"]
    )
    return np.array(response.embeddings.float_)


def busqueda_semantica(
    query: str,
    documentos: list[str],
    top_k: int = 5
) -> list[tuple[str, float]]:
    """Búsqueda semántica básica con cosine similarity."""
    query_emb = obtener_embeddings([query], tipo="search_query")[0]
    doc_embs = obtener_embeddings(documentos, tipo="search_document")

    # Cosine similarity
    from numpy.linalg import norm
    similitudes = [
        float(np.dot(query_emb, doc_emb) / (norm(query_emb) * norm(doc_emb)))
        for doc_emb in doc_embs
    ]

    # Ordenar por relevancia
    pares = sorted(zip(documentos, similitudes), key=lambda x: x[1], reverse=True)
    return pares[:top_k]


# ── Reranking ────────────────────────────────────────────────────────────
def rerankar_resultados(
    query: str,
    documentos: list[str],
    top_n: int = 3
) -> list[dict]:
    """
    El reranker de Cohere es un cross-encoder que evalúa la relevancia
    de cada documento para la query de forma más precisa que los embeddings.
    Úsalo como segundo paso después de una búsqueda inicial más rápida.
    """
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=documentos,
        top_n=top_n
    )

    return [
        {
            "documento": documentos[r.index],
            "score": round(r.relevance_score, 4),
            "posicion_original": r.index
        }
        for r in response.results
    ]


# Pipeline completo: embed → búsqueda → rerank
def pipeline_rag(query: str, base_conocimiento: list[str]) -> list[dict]:
    # 1. Búsqueda rápida con embeddings (recuperar top-20)
    candidatos = busqueda_semantica(query, base_conocimiento, top_k=20)
    docs_candidatos = [doc for doc, _ in candidatos]

    # 2. Reranking preciso (seleccionar top-3)
    return rerankar_resultados(query, docs_candidatos, top_n=3)


# Ejemplo
base = [
    "Python es un lenguaje interpretado de alto nivel creado por Guido van Rossum.",
    "Los transformers usan mecanismos de atención para procesar secuencias.",
    "El aprendizaje por refuerzo es un paradigma donde un agente aprende por recompensas.",
    "PyTorch es un framework de deep learning desarrollado por Meta.",
    "El BERT es un modelo de lenguaje bidireccional preentrenado por Google.",
]

resultados = pipeline_rag("¿Cómo funciona la atención en transformers?", base)
for r in resultados:
    print(f"Score {r['score']}: {r['documento'][:80]}...")
```

---

## 5. Cuándo elegir Mistral vs Cohere vs Claude vs GPT-4

```python
# guia_decision.py

GUIA = {
    "claude_opus": {
        "ideal_para": [
            "Razonamiento complejo y análisis largo",
            "Escritura creativa y redacción de alta calidad",
            "Tareas con instrucciones ambiguas o complejas",
            "Tool use sofisticado en agentes"
        ],
        "considerar_si": "necesitas la máxima calidad y el presupuesto lo permite",
        "precio_relativo": "alto"
    },
    "gpt_4o": {
        "ideal_para": [
            "Multimodalidad robusta (imagen + texto)",
            "Ecosistema de plugins y tools maduro",
            "Integración con Azure (empresas con Microsoft)",
            "Function calling con gran comunidad de ejemplos"
        ],
        "considerar_si": "ya usas Azure o necesitas integración profunda con el ecosistema OpenAI",
        "precio_relativo": "medio-alto"
    },
    "gemini_15_pro": {
        "ideal_para": [
            "Análisis de documentos muy largos (>100K tokens)",
            "Análisis de vídeo e imagen con contexto largo",
            "Integración con Google Workspace / GCP",
            "Grounding con búsqueda en tiempo real"
        ],
        "considerar_si": "trabajas con documentos masivos o videos, o usas Google Cloud",
        "precio_relativo": "medio"
    },
    "mistral_large": {
        "ideal_para": [
            "Proyectos en Europa con requisitos de privacidad (GDPR)",
            "Alto volumen con presupuesto ajustado",
            "Código y tareas técnicas (especialmente con Codestral)",
            "Cuando necesitas open-source (Mixtral) para on-premise"
        ],
        "considerar_si": "eres europeo, tienes requisitos de privacidad estrictos o necesitas on-premise",
        "precio_relativo": "bajo-medio"
    },
    "cohere_command_r": {
        "ideal_para": [
            "RAG empresarial con grounded generation y citas",
            "Embeddings multilingüe de alta calidad",
            "Reranking como segundo paso en pipelines de búsqueda",
            "Aplicaciones enterprise con SLA y soporte"
        ],
        "considerar_si": "tu caso de uso principal es búsqueda semántica o RAG empresarial",
        "precio_relativo": "medio"
    }
}

for modelo, info in GUIA.items():
    print(f"\n{'='*50}")
    print(f"Modelo: {modelo.upper()}")
    print(f"Precio relativo: {info['precio_relativo']}")
    print(f"Mejor opción si: {info['considerar_si']}")
    print("Ideal para:")
    for caso in info["ideal_para"]:
        print(f"  - {caso}")
```

---

## 6. Extensiones sugeridas

- **Mistral on-premise**: desplegar Mixtral 8x7B en servidor propio con vLLM para máxima privacidad
- **Cohere + LangChain**: integración nativa de Cohere en LangChain para RAG empresarial
- **Fine-tuning en Mistral**: la plataforma La Plateforme permite fine-tuning de Mistral Small
- **Comparativa automatizada**: benchmarks automáticos de LLMs con LangSmith o Weights & Biases

---

**Anterior:** [04 — Google Gemini](./04-google-gemini.md) · **Siguiente bloque:** [Bloque 4 — Python para IA](../python-para-ia/)
