# 03 — Optimización de costos con LLMs

> **Bloque:** Producción · **Nivel:** Práctico · **Tiempo estimado:** 45 min

---

## Índice

1. [Entender el coste de las APIs](#1-entender-el-coste-de-las-apis)
2. [Contar tokens antes de enviar](#2-contar-tokens-antes-de-enviar)
3. [Prompt Caching de Anthropic](#3-prompt-caching-de-anthropic)
4. [Batch API de Anthropic](#4-batch-api-de-anthropic)
5. [Estrategias de compresión de contexto](#5-estrategias-de-compresión-de-contexto)
6. [Calculadora de costes](#6-calculadora-de-costes)

---

## 1. Entender el coste de las APIs

Los LLMs en producción se pagan por tokens. Un token equivale aproximadamente a 4 caracteres en inglés o 3-4 en español. El coste tiene dos componentes independientes: los tokens de entrada (el prompt) y los de salida (la respuesta generada).

**Precios orientativos (USD por millón de tokens, abril 2025):**

| Modelo | Entrada | Salida | Caché (lectura) |
|---|---|---|---|
| claude-sonnet-4-6 | $3.00 | $15.00 | $0.30 |
| claude-haiku-3-5 | $0.80 | $4.00 | $0.08 |
| gpt-4o | $2.50 | $10.00 | $1.25 |
| gpt-4o-mini | $0.15 | $0.60 | $0.075 |

**La asimetría entrada/salida es el punto clave:** los tokens de salida cuestan entre 4 y 5 veces más que los de entrada. Reducir la longitud de las respuestas tiene un impacto mayor que reducir el prompt.

**Estrategias por orden de impacto:**

```
1. Elegir el modelo correcto para cada tarea    → ×10 reducción potencial
2. Prompt Caching (repetir el mismo prefijo)    → 90% descuento en entrada
3. Batch API (procesamiento diferido)           → 50% descuento en todo
4. Compresión de contexto (historial corto)     → variable según la app
5. Respuestas más cortas (instrucción explícita)→ 20-60% en salida
```

---

## 2. Contar tokens antes de enviar

Contar los tokens antes de la llamada permite estimar el coste, evitar errores de límite de contexto y decidir si aplicar compresión.

```bash
pip install anthropic tiktoken python-dotenv
```

```python
# costos/contar_tokens.py
"""
Contar tokens para Anthropic (Claude) y OpenAI (GPT) antes de enviar.
"""
import os
from dotenv import load_dotenv
import anthropic
import tiktoken

load_dotenv()

# ---------------------------------------------------------------------------
# Contador de tokens para Claude (Anthropic)
# ---------------------------------------------------------------------------

def contar_tokens_claude(
    messages: list[dict],
    system: str = "",
    model: str = "claude-sonnet-4-6",
) -> int:
    """
    Usa la API de Anthropic para contar los tokens exactos de un prompt.
    No genera respuesta: solo cuenta.
    """
    client = anthropic.Anthropic()

    kwargs = {
        "model": model,
        "max_tokens": 1,  # Requerido pero no se usa
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    respuesta = client.messages.count_tokens(**kwargs)
    return respuesta.input_tokens


def contar_tokens_claude_local(texto: str) -> int:
    """
    Estimación rápida sin llamada a la API.
    Claude usa tokenización similar a GPT-4 (cl100k_base).
    Error típico: ±5%.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(texto))


# ---------------------------------------------------------------------------
# Contador de tokens para GPT (OpenAI)
# ---------------------------------------------------------------------------

def contar_tokens_gpt(
    messages: list[dict],
    model: str = "gpt-4o",
) -> int:
    """
    Cuenta tokens para modelos GPT usando tiktoken.
    Incluye el overhead de formato de chat.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Overhead por mensaje (role + separadores)
    tokens_por_mensaje = 3
    tokens_reply = 3  # Overhead de respuesta

    total = tokens_reply
    for mensaje in messages:
        total += tokens_por_mensaje
        for clave, valor in mensaje.items():
            total += len(encoding.encode(str(valor)))
            if clave == "name":
                total += 1  # El campo 'name' añade un token extra

    return total


# ---------------------------------------------------------------------------
# Función para decidir si comprimir antes de enviar
# ---------------------------------------------------------------------------

LIMITES_CONTEXTO = {
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-3-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
}

def analizar_prompt(
    messages: list[dict],
    system: str = "",
    model: str = "claude-sonnet-4-6",
    max_tokens_respuesta: int = 1024,
) -> dict:
    """
    Analiza un prompt y devuelve: tokens estimados, tokens disponibles
    para respuesta y si es seguro enviar.
    """
    tokens_entrada = contar_tokens_claude_local(
        system + " ".join(m.get("content", "") for m in messages)
    )
    limite = LIMITES_CONTEXTO.get(model, 128_000)
    disponibles = limite - tokens_entrada
    seguro = disponibles >= max_tokens_respuesta

    return {
        "tokens_entrada": tokens_entrada,
        "limite_contexto": limite,
        "tokens_disponibles": disponibles,
        "max_tokens_respuesta": max_tokens_respuesta,
        "seguro_enviar": seguro,
        "uso_contexto_pct": round((tokens_entrada / limite) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    system_prompt = "Eres un asistente experto en inteligencia artificial que responde en español de forma clara y concisa."

    messages = [
        {"role": "user", "content": "¿Qué es el aprendizaje por refuerzo?"},
        {"role": "assistant", "content": "El aprendizaje por refuerzo es un paradigma de machine learning donde un agente aprende a tomar decisiones mediante interacción con un entorno, recibiendo recompensas o penalizaciones según sus acciones."},
        {"role": "user", "content": "Dame un ejemplo real de aplicación."},
    ]

    # Contador local (sin API)
    print("=== Estimación local (sin API) ===")
    analisis = analizar_prompt(messages, system=system_prompt)
    for clave, valor in analisis.items():
        print(f"  {clave:<25} {valor}")

    # Contador exacto con API de Anthropic
    print("\n=== Conteo exacto con API de Anthropic ===")
    tokens_exactos = contar_tokens_claude(messages, system=system_prompt)
    print(f"  Tokens exactos: {tokens_exactos}")

    # Contador GPT
    print("\n=== Tokens para GPT-4o (tiktoken) ===")
    tokens_gpt = contar_tokens_gpt(messages, model="gpt-4o")
    print(f"  Tokens estimados: {tokens_gpt}")
```

---

## 3. Prompt Caching de Anthropic

El Prompt Caching permite marcar partes del prompt para que Anthropic las almacene entre llamadas. Los tokens leídos de caché cuestan el **10% del precio normal** de entrada. Es ideal para system prompts largos, documentos de referencia o ejemplos few-shot que se repiten en cada llamada.

**Requisitos para que el caché funcione:**
- El bloque marcado debe tener **al menos 1024 tokens** (claude-haiku: 2048 mínimo)
- El caché tiene una duración de **5 minutos** (se renueva en cada uso)
- El prefijo cacheado debe ser **idéntico** en cada llamada (incluye el system prompt)

```bash
pip install anthropic python-dotenv
```

```python
# costos/prompt_caching.py
"""
Prompt Caching de Anthropic: reducir el coste de system prompts largos.

Flujo:
  1. Primera llamada: MISS → se paga el precio normal y se cachea
  2. Llamadas siguientes: HIT → se paga el 10% del precio normal
"""
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# System prompt largo (simulamos un documento de referencia extenso)
# ---------------------------------------------------------------------------

MANUAL_PRODUCTO = """
Eres el asistente de soporte técnico de TechCorp. Debes responder usando
exclusivamente la siguiente documentación oficial. No inventes información.

=== MANUAL DE PRODUCTO TECHCORP v3.2 ===

CAPÍTULO 1: INSTALACIÓN
El software TechCorp requiere Python 3.10 o superior y 4 GB de RAM mínimo.
Para instalar, ejecuta: pip install techcorp==3.2.0
La activación de la licencia se realiza con: techcorp activate --key TU_LICENCIA
Los logs de instalación se guardan en ~/.techcorp/install.log

CAPÍTULO 2: CONFIGURACIÓN
El archivo de configuración principal es ~/.techcorp/config.yaml
Parámetros principales:
  - max_workers: número máximo de procesos paralelos (defecto: 4)
  - timeout: tiempo máximo de espera en segundos (defecto: 30)
  - log_level: nivel de logging (DEBUG, INFO, WARNING, ERROR)
  - output_dir: directorio de salida (defecto: ~/techcorp_output)

CAPÍTULO 3: API REST
Base URL: https://api.techcorp.com/v3
Autenticación: Bearer token en el header Authorization
Endpoints principales:
  POST /process     → Enviar trabajo de procesamiento
  GET  /status/{id} → Consultar estado de un trabajo
  GET  /results/{id}→ Obtener resultados
  DELETE /job/{id}  → Cancelar un trabajo

CAPÍTULO 4: ERRORES COMUNES
Error 401: Token inválido o expirado. Renueva el token en el portal.
Error 429: Límite de requests alcanzado. Espera 60 segundos.
Error 503: Servicio temporalmente no disponible. Reintenta en 5 minutos.
TimeoutError: Aumenta el parámetro timeout en la configuración.
MemoryError: Reduce max_workers o cierra otras aplicaciones.

CAPÍTULO 5: OPTIMIZACIÓN
Para grandes volúmenes de datos, usa el modo batch:
  techcorp batch --input datos.csv --workers 8
El modo caché local reduce llamadas a la API:
  techcorp config set cache_enabled true
Para monitorizar el rendimiento:
  techcorp stats --last 24h
""" * 3  # Repetimos para superar el mínimo de 1024 tokens


def llamar_con_cache(pregunta: str) -> dict:
    """
    Llama a Claude con prompt caching en el system prompt.
    Retorna la respuesta y las métricas de uso de tokens.
    """
    inicio = time.perf_counter()

    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": MANUAL_PRODUCTO,
                "cache_control": {"type": "ephemeral"},  # Marcar para caché
            }
        ],
        messages=[{"role": "user", "content": pregunta}],
    )

    latencia_ms = round((time.perf_counter() - inicio) * 1000, 0)

    uso = mensaje.usage
    return {
        "respuesta": mensaje.content[0].text,
        "tokens_entrada": uso.input_tokens,
        "tokens_salida": uso.output_tokens,
        "cache_creados": getattr(uso, "cache_creation_input_tokens", 0),
        "cache_leidos": getattr(uso, "cache_read_input_tokens", 0),
        "latencia_ms": latencia_ms,
    }


def calcular_ahorro(metricas: dict) -> dict:
    """Calcula el ahorro real de usar caché."""
    precio_entrada_normal = 3.00 / 1_000_000   # USD por token
    precio_cache_lectura = 0.30 / 1_000_000     # 10% del precio normal

    tokens_normales = metricas["tokens_entrada"]
    tokens_en_cache = metricas["cache_leidos"]

    coste_sin_cache = (tokens_normales + tokens_en_cache) * precio_entrada_normal
    coste_con_cache = (tokens_normales * precio_entrada_normal) + (tokens_en_cache * precio_cache_lectura)
    ahorro = coste_sin_cache - coste_con_cache

    return {
        "coste_sin_cache_usd": round(coste_sin_cache, 6),
        "coste_con_cache_usd": round(coste_con_cache, 6),
        "ahorro_usd": round(ahorro, 6),
        "ahorro_pct": round((ahorro / coste_sin_cache * 100) if coste_sin_cache > 0 else 0, 1),
    }


# ---------------------------------------------------------------------------
# Ejemplo de uso — múltiples preguntas sobre el mismo manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preguntas = [
        "¿Cómo instalo TechCorp?",
        "¿Qué significa el error 429?",
        "¿Cómo activo el modo batch para procesar datos masivos?",
    ]

    print(f"Tamaño del manual: ~{len(MANUAL_PRODUCTO.split()):,} palabras\n")

    for i, pregunta in enumerate(preguntas, 1):
        print(f"--- Pregunta {i}: {pregunta} ---")
        resultado = llamar_con_cache(pregunta)

        tipo_cache = "MISS (primera vez)" if resultado["cache_creados"] > 0 else "HIT (desde caché)"
        print(f"Caché: {tipo_cache}")
        print(f"Tokens entrada:   {resultado['tokens_entrada']:>6}")
        print(f"Tokens en caché:  {resultado['cache_leidos']:>6}  ← se pagó el 10%")
        print(f"Tokens creados:   {resultado['cache_creados']:>6}  ← se pagó el 100%")
        print(f"Latencia:         {resultado['latencia_ms']:>6.0f} ms")

        ahorro = calcular_ahorro(resultado)
        print(f"Ahorro estimado:  {ahorro['ahorro_pct']}%  (${ahorro['ahorro_usd']:.6f} USD)")
        print(f"Respuesta: {resultado['respuesta'][:120]}...\n")
```

---

## 4. Batch API de Anthropic

La Batch API permite enviar miles de requests de forma asíncrona con un **50% de descuento** sobre el precio normal. Ideal para tareas que no requieren respuesta inmediata: clasificación masiva, generación de contenido, evaluaciones.

```python
# costos/batch_api.py
"""
Batch API de Anthropic: procesar miles de requests con 50% de descuento.

Flujo:
  1. Crear el batch con todos los requests
  2. Esperar a que se procese (puede tardar minutos u horas)
  3. Descargar y procesar los resultados
"""
import json
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Preparar los requests del batch
# ---------------------------------------------------------------------------

def crear_requests_clasificacion(textos: list[str]) -> list[anthropic.types.MessageCreateParamsNonStreaming]:
    """
    Crea la lista de requests para clasificar textos en lote.
    Cada request necesita un custom_id único para identificar el resultado.
    """
    requests = []
    for i, texto in enumerate(textos):
        request = anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            model="claude-sonnet-4-6",
            max_tokens=50,
            system="Clasifica el sentimiento del texto como: POSITIVO, NEGATIVO o NEUTRO. Responde solo con una de esas tres palabras.",
            messages=[{"role": "user", "content": texto}],
        )
        requests.append(
            anthropic.types.MessageBatchRequestCounts(
                custom_id=f"texto-{i:04d}",
                params=request,
            )
        )
    return requests


# ---------------------------------------------------------------------------
# Crear y monitorear el batch
# ---------------------------------------------------------------------------

def crear_batch(textos: list[str]) -> str:
    """Envía el batch a la API y retorna el batch_id."""
    print(f"Creando batch con {len(textos)} requests...")

    requests = []
    for i, texto in enumerate(textos):
        requests.append({
            "custom_id": f"texto-{i:04d}",
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 50,
                "system": "Clasifica el sentimiento del texto como: POSITIVO, NEGATIVO o NEUTRO. Responde solo con una de esas tres palabras.",
                "messages": [{"role": "user", "content": texto}],
            },
        })

    batch = client.beta.messages.batches.create(requests=requests)
    print(f"Batch creado: {batch.id}")
    print(f"Estado inicial: {batch.processing_status}")
    return batch.id


def esperar_batch(batch_id: str, intervalo_seg: int = 30) -> anthropic.types.MessageBatch:
    """
    Espera a que el batch termine de procesarse.
    Comprueba el estado cada 'intervalo_seg' segundos.
    """
    print(f"\nEsperando resultados del batch {batch_id}...")

    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        estado = batch.processing_status
        conteos = batch.request_counts

        print(
            f"Estado: {estado} | "
            f"Procesados: {conteos.succeeded + conteos.errored}/{conteos.succeeded + conteos.errored + conteos.processing} | "
            f"Éxitos: {conteos.succeeded} | Errores: {conteos.errored}"
        )

        if estado == "ended":
            print("Batch completado.")
            return batch

        time.sleep(intervalo_seg)


def procesar_resultados(batch_id: str) -> list[dict]:
    """Descarga y procesa los resultados del batch."""
    resultados = []

    for resultado in client.beta.messages.batches.results(batch_id):
        if resultado.result.type == "succeeded":
            respuesta = resultado.result.message.content[0].text.strip()
            uso = resultado.result.message.usage
            resultados.append({
                "id": resultado.custom_id,
                "sentimiento": respuesta,
                "tokens_entrada": uso.input_tokens,
                "tokens_salida": uso.output_tokens,
                "error": None,
            })
        else:
            resultados.append({
                "id": resultado.custom_id,
                "sentimiento": None,
                "tokens_entrada": 0,
                "tokens_salida": 0,
                "error": str(resultado.result.error),
            })

    return sorted(resultados, key=lambda r: r["id"])


# ---------------------------------------------------------------------------
# Ejemplo completo
# ---------------------------------------------------------------------------

TEXTOS_EJEMPLO = [
    "Me encanta este producto, es exactamente lo que necesitaba.",
    "El servicio de atención al cliente fue muy lento y poco útil.",
    "El paquete llegó en la fecha prevista.",
    "Calidad terrible, no lo recomendaría a nadie.",
    "Muy buena relación calidad-precio, repetiré sin duda.",
    "La interfaz es confusa y difícil de usar.",
    "El producto cumple con lo que promete.",
    "Decepcionante, esperaba mucho más por ese precio.",
    "Excelente experiencia de compra de principio a fin.",
    "El color no corresponde con las fotos de la web.",
]


def estimar_ahorro_batch(num_requests: int, tokens_por_request: int = 500) -> None:
    """Muestra el ahorro estimado al usar la Batch API."""
    precio_normal = 3.00 / 1_000_000
    precio_batch = 1.50 / 1_000_000  # 50% descuento

    coste_normal = num_requests * tokens_por_request * precio_normal
    coste_batch = num_requests * tokens_por_request * precio_batch

    print(f"\nEstimación de ahorro para {num_requests} requests ({tokens_por_request} tokens c/u):")
    print(f"  Coste sin Batch API: ${coste_normal:.4f} USD")
    print(f"  Coste con Batch API: ${coste_batch:.4f} USD")
    print(f"  Ahorro:              ${coste_normal - coste_batch:.4f} USD ({50}%)")


if __name__ == "__main__":
    estimar_ahorro_batch(len(TEXTOS_EJEMPLO))

    print("\n--- Iniciando batch real ---")
    print("Nota: Los batches pueden tardar desde segundos hasta horas según el volumen.\n")

    batch_id = crear_batch(TEXTOS_EJEMPLO)
    batch_completado = esperar_batch(batch_id, intervalo_seg=10)

    resultados = procesar_resultados(batch_id)

    print("\nResultados:")
    print(f"{'ID':<12} {'Sentimiento':<12} {'Tokens':<8} {'Error'}")
    print("-" * 55)
    for r in resultados:
        idx = int(r["id"].split("-")[1])
        texto_corto = TEXTOS_EJEMPLO[idx][:40] + "..."
        print(
            f"{r['id']:<12} {str(r['sentimiento']):<12} "
            f"{r['tokens_entrada'] + r['tokens_salida']:<8} "
            f"{r['error'] or '-'}"
        )

    # Guardar en JSONL
    with open("resultados_batch.jsonl", "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nResultados guardados en resultados_batch.jsonl")
```

---

## 5. Estrategias de compresión de contexto

En aplicaciones conversacionales, el historial crece con cada turno y puede representar el 90% del coste. Dos estrategias complementarias: truncación inteligente y resumen automático.

```python
# costos/compresion_contexto.py
"""
Estrategias para controlar el tamaño del contexto en aplicaciones conversacionales.
"""
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Estrategia 1: Truncación inteligente por número de tokens
# ---------------------------------------------------------------------------

def truncar_historial(
    historial: list[dict],
    max_tokens: int = 4_000,
    siempre_conservar_ultimos: int = 2,
) -> list[dict]:
    """
    Elimina los mensajes más antiguos hasta que el historial cabe en max_tokens.
    Siempre conserva al menos los últimos 'siempre_conservar_ultimos' mensajes.

    Args:
        historial: Lista de mensajes en formato {role, content}.
        max_tokens: Límite máximo de tokens para el historial.
        siempre_conservar_ultimos: Número de mensajes recientes a preservar siempre.

    Returns:
        Historial truncado.
    """
    def estimar_tokens(messages: list[dict]) -> int:
        texto = " ".join(m.get("content", "") for m in messages)
        return len(texto) // 4  # Estimación rápida: 1 token ≈ 4 caracteres

    if estimar_tokens(historial) <= max_tokens:
        return historial  # No hace falta truncar

    # Siempre conservar los últimos N mensajes (contexto reciente)
    protegidos = historial[-siempre_conservar_ultimos:]
    candidatos = historial[:-siempre_conservar_ultimos]

    # Eliminar desde el principio hasta que quepan
    while candidatos and estimar_tokens(candidatos + protegidos) > max_tokens:
        candidatos.pop(0)

    truncado = candidatos + protegidos
    print(f"Historial truncado: {len(historial)} → {len(truncado)} mensajes")
    return truncado


# ---------------------------------------------------------------------------
# Estrategia 2: Resumen automático del historial antiguo
# ---------------------------------------------------------------------------

def resumir_historial(
    historial: list[dict],
    max_mensajes_sin_resumir: int = 10,
    mensajes_recientes_a_conservar: int = 4,
) -> list[dict]:
    """
    Cuando el historial supera max_mensajes_sin_resumir, resume los mensajes
    antiguos en un bloque compacto y conserva los más recientes completos.

    Esto preserva el contexto semántico mejor que la truncación simple.
    """
    if len(historial) <= max_mensajes_sin_resumir:
        return historial

    # Separar: mensajes a resumir vs mensajes recientes (siempre completos)
    a_resumir = historial[:-mensajes_recientes_a_conservar]
    recientes = historial[-mensajes_recientes_a_conservar:]

    # Convertir a texto para el resumen
    texto_conversacion = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in a_resumir
    )

    print(f"Resumiendo {len(a_resumir)} mensajes antiguos...")
    resumen_msg = client.messages.create(
        model="claude-haiku-3-5",  # Modelo económico para el resumen
        max_tokens=300,
        system=(
            "Resume la siguiente conversación en un párrafo conciso. "
            "Incluye los temas tratados, decisiones tomadas y cualquier información clave. "
            "El resumen se usará como contexto para continuar la conversación."
        ),
        messages=[{"role": "user", "content": texto_conversacion}],
    )
    resumen = resumen_msg.content[0].text

    # Construir historial comprimido: resumen como contexto + mensajes recientes
    historial_comprimido = [
        {
            "role": "user",
            "content": f"[RESUMEN DE LA CONVERSACIÓN ANTERIOR]\n{resumen}",
        },
        {
            "role": "assistant",
            "content": "Entendido. Continúo con la conversación teniendo en cuenta ese contexto.",
        },
        *recientes,
    ]

    print(f"Historial comprimido: {len(historial)} → {len(historial_comprimido)} mensajes")
    return historial_comprimido


# ---------------------------------------------------------------------------
# Chatbot con compresión automática
# ---------------------------------------------------------------------------

class ChatbotEficiente:
    """
    Chatbot que aplica compresión de contexto automática para controlar costes.
    Usa resumen cuando el historial supera un umbral y truncación como respaldo.
    """

    def __init__(
        self,
        system: str = "Eres un asistente útil.",
        max_tokens_historial: int = 6_000,
        umbral_resumen: int = 12,
    ):
        self.system = system
        self.max_tokens_historial = max_tokens_historial
        self.umbral_resumen = umbral_resumen
        self.historial: list[dict] = []
        self.total_tokens_ahorrados = 0

    def chat(self, mensaje_usuario: str) -> str:
        self.historial.append({"role": "user", "content": mensaje_usuario})

        # Aplicar compresión si el historial es largo
        if len(self.historial) > self.umbral_resumen:
            historial_original_tokens = sum(len(m["content"]) for m in self.historial) // 4
            self.historial = resumir_historial(self.historial)
            historial_nuevo_tokens = sum(len(m["content"]) for m in self.historial) // 4
            self.total_tokens_ahorrados += historial_original_tokens - historial_nuevo_tokens

        # También truncar por seguridad
        self.historial = truncar_historial(self.historial, self.max_tokens_historial)

        # Llamar al modelo
        respuesta_msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=self.system,
            messages=self.historial,
        )
        respuesta = respuesta_msg.content[0].text
        self.historial.append({"role": "assistant", "content": respuesta})

        return respuesta

    def stats(self) -> dict:
        tokens_actuales = sum(len(m["content"]) for m in self.historial) // 4
        return {
            "mensajes_en_historial": len(self.historial),
            "tokens_estimados": tokens_actuales,
            "tokens_ahorrados_total": self.total_tokens_ahorrados,
        }


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = ChatbotEficiente(
        system="Eres un tutor de Python. Responde de forma concisa.",
        umbral_resumen=6,  # Resumir después de 6 mensajes (bajo para demostración)
    )

    conversacion = [
        "¿Qué es una lista en Python?",
        "¿Y un diccionario?",
        "¿Cuándo uso uno u otro?",
        "¿Las listas pueden contener diccionarios?",
        "¿Y diccionarios con listas?",
        "Dame un ejemplo de estructura anidada.",
        "¿Cómo accedo al segundo elemento de una lista dentro de un diccionario?",
        "¿Y si el diccionario está dentro de una lista?",
    ]

    for i, mensaje in enumerate(conversacion, 1):
        print(f"\n[Turno {i}] Usuario: {mensaje}")
        respuesta = bot.chat(mensaje)
        print(f"Bot: {respuesta[:120]}...")

        stats = bot.stats()
        print(f"Stats: {stats['mensajes_en_historial']} msgs | ~{stats['tokens_estimados']} tokens | ahorrados: ~{stats['tokens_ahorrados_total']}")
```

---

## 6. Calculadora de costes

Una función reutilizable para estimar el coste de cualquier llamada antes de ejecutarla o para calcular el coste real después.

```python
# costos/calculadora.py
"""
Calculadora de costes para APIs de LLMs.
Útil para estimar presupuestos y monitorizar gastos.
"""

# Precios en USD por millón de tokens (actualizar según tarifas vigentes)
PRECIOS = {
    "claude-sonnet-4-6": {
        "entrada": 3.00,
        "salida": 15.00,
        "cache_lectura": 0.30,
        "cache_escritura": 3.75,
    },
    "claude-haiku-3-5": {
        "entrada": 0.80,
        "salida": 4.00,
        "cache_lectura": 0.08,
        "cache_escritura": 1.00,
    },
    "claude-opus-4": {
        "entrada": 15.00,
        "salida": 75.00,
        "cache_lectura": 1.50,
        "cache_escritura": 18.75,
    },
    "gpt-4o": {
        "entrada": 2.50,
        "salida": 10.00,
        "cache_lectura": 1.25,
        "cache_escritura": 0,  # OpenAI no cobra por escritura de caché
    },
    "gpt-4o-mini": {
        "entrada": 0.15,
        "salida": 0.60,
        "cache_lectura": 0.075,
        "cache_escritura": 0,
    },
}


def calcular_coste_llamada(
    modelo: str,
    tokens_entrada: int,
    tokens_salida: int,
    tokens_cache_leidos: int = 0,
    tokens_cache_escritura: int = 0,
) -> dict:
    """
    Calcula el coste exacto de una llamada a la API.

    Args:
        modelo: Nombre del modelo (clave en PRECIOS).
        tokens_entrada: Tokens de entrada normales.
        tokens_salida: Tokens de salida generados.
        tokens_cache_leidos: Tokens leídos de caché (solo Anthropic).
        tokens_cache_escritura: Tokens escritos al caché (solo Anthropic).

    Returns:
        Diccionario con el desglose de costes en USD.
    """
    if modelo not in PRECIOS:
        modelos_disponibles = list(PRECIOS.keys())
        raise ValueError(f"Modelo desconocido: {modelo}. Disponibles: {modelos_disponibles}")

    p = PRECIOS[modelo]
    M = 1_000_000  # por millón

    coste_entrada = (tokens_entrada / M) * p["entrada"]
    coste_salida = (tokens_salida / M) * p["salida"]
    coste_cache_lectura = (tokens_cache_leidos / M) * p["cache_lectura"]
    coste_cache_escritura = (tokens_cache_escritura / M) * p["cache_escritura"]
    total = coste_entrada + coste_salida + coste_cache_lectura + coste_cache_escritura

    return {
        "modelo": modelo,
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_cache_leidos": tokens_cache_leidos,
        "tokens_cache_escritura": tokens_cache_escritura,
        "coste_entrada_usd": round(coste_entrada, 8),
        "coste_salida_usd": round(coste_salida, 8),
        "coste_cache_lectura_usd": round(coste_cache_lectura, 8),
        "coste_cache_escritura_usd": round(coste_cache_escritura, 8),
        "coste_total_usd": round(total, 8),
    }


def estimar_coste_mensual(
    modelo: str,
    tokens_entrada_por_llamada: int,
    tokens_salida_por_llamada: int,
    llamadas_por_dia: int,
    dias: int = 30,
    pct_cache_hit: float = 0.0,
) -> dict:
    """
    Estima el coste mensual de una aplicación.

    Args:
        pct_cache_hit: Porcentaje de llamadas que aciertan en caché (0.0 a 1.0).
    """
    llamadas_total = llamadas_por_dia * dias

    # Con caché, algunos tokens de entrada se leen al 10%
    tokens_entrada_normal = int(tokens_entrada_por_llamada * (1 - pct_cache_hit))
    tokens_entrada_cache = int(tokens_entrada_por_llamada * pct_cache_hit)

    coste_por_llamada = calcular_coste_llamada(
        modelo=modelo,
        tokens_entrada=tokens_entrada_normal,
        tokens_salida=tokens_salida_por_llamada,
        tokens_cache_leidos=tokens_entrada_cache,
    )

    coste_total = coste_por_llamada["coste_total_usd"] * llamadas_total

    return {
        "modelo": modelo,
        "llamadas_por_dia": llamadas_por_dia,
        "dias": dias,
        "llamadas_total": llamadas_total,
        "pct_cache_hit": f"{pct_cache_hit * 100:.0f}%",
        "coste_por_llamada_usd": coste_por_llamada["coste_total_usd"],
        "coste_total_usd": round(coste_total, 4),
        "coste_total_eur": round(coste_total * 0.92, 4),  # Tasa orientativa
    }


def comparar_modelos(
    tokens_entrada: int,
    tokens_salida: int,
    llamadas_por_mes: int = 10_000,
) -> None:
    """Muestra una tabla comparativa de costes por modelo."""
    print(f"\nComparación de modelos ({llamadas_por_mes:,} llamadas/mes)")
    print(f"Tokens por llamada: {tokens_entrada} entrada + {tokens_salida} salida")
    print("-" * 70)
    print(f"{'Modelo':<22} {'Por llamada':>12} {'Mensual (USD)':>14} {'Mensual (EUR)':>14}")
    print("-" * 70)

    filas = []
    for modelo in PRECIOS:
        coste = calcular_coste_llamada(modelo, tokens_entrada, tokens_salida)
        coste_mensual = coste["coste_total_usd"] * llamadas_por_mes
        filas.append((modelo, coste["coste_total_usd"], coste_mensual))

    for modelo, por_llamada, mensual in sorted(filas, key=lambda x: x[1]):
        print(
            f"{modelo:<22} ${por_llamada:>10.6f} ${mensual:>13.2f} €{mensual * 0.92:>13.2f}"
        )


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Coste de una llamada individual
    print("=== Coste de una llamada ===")
    coste = calcular_coste_llamada(
        modelo="claude-sonnet-4-6",
        tokens_entrada=1500,
        tokens_salida=300,
        tokens_cache_leidos=2000,
    )
    for k, v in coste.items():
        print(f"  {k:<30} {v}")

    # 2. Estimación mensual con caché
    print("\n=== Estimación mensual ===")
    estimacion = estimar_coste_mensual(
        modelo="claude-sonnet-4-6",
        tokens_entrada_por_llamada=2000,
        tokens_salida_por_llamada=400,
        llamadas_por_dia=1_000,
        dias=30,
        pct_cache_hit=0.7,  # 70% de llamadas leen del caché
    )
    for k, v in estimacion.items():
        print(f"  {k:<30} {v}")

    # 3. Comparar todos los modelos
    comparar_modelos(
        tokens_entrada=1_000,
        tokens_salida=500,
        llamadas_por_mes=50_000,
    )
```

---

**Siguiente:** [Despliegue con FastAPI y Docker](./04-despliegue.md)
