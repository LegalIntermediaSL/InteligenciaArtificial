# Prompt Caching: Reducción de Costes hasta 90%

## ¿Qué es el Prompt Caching?

El Prompt Caching permite a Anthropic **reutilizar el procesamiento de partes estáticas
del contexto** entre llamadas sucesivas. En lugar de procesar el mismo system prompt
o los mismos documentos de referencia en cada request, Claude los procesa una vez
y los almacena en caché durante 5 minutos (TTL renovable).

**Ahorro:**
- Tokens de entrada en caché: **-90% del coste** (se cobran a precio de cache read)
- Escritura inicial en caché: **+25%** sobre el precio normal (solo la primera vez)
- A partir de la segunda llamada: ahorro neto garantizado si el contexto es grande

## Cuándo tiene sentido

| Situación | ¿Vale la pena? |
|-----------|---------------|
| System prompt de >1.000 tokens, muchas llamadas | ✅ Sí |
| Base de datos de conocimiento fija en el contexto | ✅ Sí |
| Mismo PDF analizado por múltiples usuarios | ✅ Sí |
| Historial de conversación largo y estático | ✅ Sí |
| Llamadas ocasionales con contexto corto | ❌ No |
| Contexto completamente diferente en cada request | ❌ No |

**Mínimo para cachear:** 1.024 tokens (Haiku) / 2.048 tokens (Sonnet/Opus).

## Sintaxis básica

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT_LARGO = """Eres un asistente especializado en derecho mercantil español.
Tienes acceso al Código de Comercio completo, la Ley de Sociedades de Capital
y la Ley Concursal actualizada a 2025.

[... miles de tokens de contexto legal ...]
""" * 50  # Simular contexto largo

# Primera llamada: escribe en caché (cache_write)
respuesta1 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT_LARGO,
            "cache_control": {"type": "ephemeral"}   # Marcar para caché
        }
    ],
    messages=[{"role": "user", "content": "¿Qué es una sociedad limitada?"}]
)

print(f"Cache write tokens: {respuesta1.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {respuesta1.usage.cache_read_input_tokens}")
print(f"Input tokens: {respuesta1.usage.input_tokens}")

# Segunda llamada: lee de caché (cache_read — 90% más barato)
respuesta2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": SYSTEM_PROMPT_LARGO,
            "cache_control": {"type": "ephemeral"}   # Mismo texto → hit de caché
        }
    ],
    messages=[{"role": "user", "content": "¿Cuántos socios mínimos necesita una SL?"}]
)

print(f"\nSegunda llamada:")
print(f"Cache read tokens: {respuesta2.usage.cache_read_input_tokens}")  # Debería ser > 0
print(f"Ahorro en esta llamada: ~{respuesta2.usage.cache_read_input_tokens * 0.9 / 1000 * 0.25:.4f}$")
```

## Reglas de caché

1. **El texto debe ser idéntico** para obtener un cache hit. Cualquier cambio invalida la caché.
2. **TTL: 5 minutos**, renovado automáticamente con cada uso.
3. **Máximo 4 puntos de caché** por request.
4. **Los tokens de la parte variable** (historial de conversación, pregunta actual) nunca se cachean.
5. **Por modelo**: la caché de Haiku es independiente de la de Sonnet.

## Patrón: sistema RAG con caché

```python
# Cargar base de conocimiento una vez
with open("base_conocimiento.txt") as f:
    BASE_CONOCIMIENTO = f.read()  # Puede ser muy grande

def preguntar_con_cache(pregunta: str) -> str:
    """Cada llamada se beneficia del caché del contexto."""
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": "Eres un asistente experto. Responde basándote en este contexto:"
            },
            {
                "type": "text",
                "text": BASE_CONOCIMIENTO,
                "cache_control": {"type": "ephemeral"}   # Cachear la base de conocimiento
            }
        ],
        messages=[{"role": "user", "content": pregunta}]
    )
    return resp.content[0].text

# Las llamadas sucesivas reutilizan el contexto cacheado
r1 = preguntar_con_cache("¿Cómo funciona el proceso de onboarding?")
r2 = preguntar_con_cache("¿Cuáles son las políticas de devolución?")
r3 = preguntar_con_cache("¿Qué soporte incluye el plan Pro?")
```

## Múltiples puntos de caché

Puedes tener hasta 4 puntos de caché independientes en un mismo request:

```python
# Caché en sistema + en herramientas + en historial
respuesta = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=800,
    system=[
        {
            "type": "text",
            "text": CONTEXTO_EMPRESA_LARGO,
            "cache_control": {"type": "ephemeral"}   # Punto 1: contexto empresa
        }
    ],
    tools=[
        {
            "name": "buscar_cliente",
            "description": "Busca información de un cliente en el CRM",
            "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"}   # Punto 2: definición herramientas
        }
    ],
    messages=[
        # Punto 3: historial largo de conversación ya procesado
        *historial_largo,
        {"role": "user", "content": pregunta_nueva},  # Parte variable (no cacheada)
    ]
)
```

## Caché en conversaciones multi-turn

```python
historial = []

def chat_con_cache(mensaje_usuario: str) -> str:
    """Conversación donde el historial previo se cachea progresivamente."""
    historial.append({"role": "user", "content": mensaje_usuario})

    # Marcar el último mensaje del asistente como cacheable (si existe)
    mensajes_para_api = []
    for i, msg in enumerate(historial):
        if i == len(historial) - 2 and msg["role"] == "assistant":
            # Cachear el turn anterior del asistente
            mensajes_para_api.append({
                "role": "assistant",
                "content": [{"type": "text", "text": msg["content"],
                             "cache_control": {"type": "ephemeral"}}]
            })
        else:
            mensajes_para_api.append(msg)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=mensajes_para_api
    )

    respuesta_texto = resp.content[0].text
    historial.append({"role": "assistant", "content": respuesta_texto})
    return respuesta_texto

# Conversación larga con caché progresivo
r1 = chat_con_cache("Hola, tengo una duda sobre mi factura")
r2 = chat_con_cache("El importe parece incorrecto, pagué 350€ y me cobran 420€")
r3 = chat_con_cache("¿Puede alguien revisarlo manualmente?")
```

## Calcular el ahorro real

```python
def calcular_ahorro_cache(
    tokens_sistema: int,
    num_llamadas: int,
    modelo: str = "haiku"
) -> dict:
    """Calcula el ahorro estimado con prompt caching."""
    # Precios aproximados (USD por millón de tokens, 2025)
    precios = {
        "haiku": {"normal": 0.25, "cache_write": 0.30, "cache_read": 0.03},
        "sonnet": {"normal": 3.00, "cache_write": 3.75, "cache_read": 0.30},
    }
    p = precios.get(modelo, precios["haiku"])

    coste_sin_cache = tokens_sistema * num_llamadas * p["normal"] / 1_000_000
    coste_con_cache = (
        tokens_sistema * p["cache_write"] / 1_000_000 +           # 1 escritura
        tokens_sistema * (num_llamadas - 1) * p["cache_read"] / 1_000_000  # N-1 lecturas
    )
    ahorro = coste_sin_cache - coste_con_cache

    return {
        "sin_cache": f"${coste_sin_cache:.4f}",
        "con_cache": f"${coste_con_cache:.4f}",
        "ahorro": f"${ahorro:.4f}",
        "ahorro_porcentaje": f"{ahorro / coste_sin_cache * 100:.1f}%"
    }

# Ejemplo: 5.000 tokens de sistema, 100 llamadas
calculo = calcular_ahorro_cache(5000, 100, "haiku")
print(f"Sin caché: {calculo['sin_cache']}")
print(f"Con caché: {calculo['con_cache']}")
print(f"Ahorro: {calculo['ahorro']} ({calculo['ahorro_porcentaje']})")
```

## Monitorizar uso de caché

```python
def llamada_con_metricas(client, **kwargs) -> dict:
    """Wrapper que registra métricas de caché."""
    resp = client.messages.create(**kwargs)
    uso = resp.usage
    return {
        "respuesta": resp.content[0].text,
        "cache_write": getattr(uso, "cache_creation_input_tokens", 0),
        "cache_read": getattr(uso, "cache_read_input_tokens", 0),
        "input_normal": uso.input_tokens,
        "output": uso.output_tokens,
        "cache_hit": getattr(uso, "cache_read_input_tokens", 0) > 0
    }
```

## Checklist para maximizar el ahorro

- [x] System prompt > 1.024 tokens (Haiku) o > 2.048 (Sonnet)
- [x] El contenido estático va al principio del contexto
- [x] El contenido dinámico (pregunta del usuario) va al final
- [x] Marcar con `cache_control: {"type": "ephemeral"}` las partes estáticas
- [x] Las llamadas se hacen dentro de la ventana de 5 minutos
- [x] Monitorizar `cache_read_input_tokens` para confirmar hits
- [x] No mezclar contenido dinámico dentro del bloque cacheado

## Recursos

- [Documentación Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Notebook interactivo](../notebooks/apis-avanzadas/05-prompt-caching.ipynb)
