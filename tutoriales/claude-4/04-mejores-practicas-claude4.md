# Mejores prácticas con Claude 4.X en producción

## 1. Elige el modelo correcto desde el principio

El error más común en producción es usar Opus para todo. Un router de modelos puede reducir el coste total un 70% sin pérdida de calidad percibida:

```python
import anthropic
from enum import Enum

client = anthropic.Anthropic()

class Complejidad(Enum):
    SIMPLE = "simple"
    MODERADA = "moderada"
    COMPLEJA = "compleja"

MODELO_POR_COMPLEJIDAD = {
    Complejidad.SIMPLE: "claude-haiku-4-5-20251001",
    Complejidad.MODERADA: "claude-sonnet-4-6",
    Complejidad.COMPLEJA: "claude-opus-4-7",
}

def clasificar_complejidad(prompt: str) -> Complejidad:
    """Clasifica la complejidad con Haiku (muy barato)."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system='Responde solo "simple", "moderada" o "compleja".',
        messages=[{"role": "user", "content": f"Complejidad de esta tarea: {prompt[:200]}"}],
    )
    texto = response.content[0].text.strip().lower()
    for c in Complejidad:
        if c.value in texto:
            return c
    return Complejidad.MODERADA  # fallback seguro

def generar_respuesta(prompt: str) -> tuple[str, str]:
    complejidad = clasificar_complejidad(prompt)
    modelo = MODELO_POR_COMPLEJIDAD[complejidad]
    response = client.messages.create(
        model=modelo,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return modelo, response.content[0].text
```

## 2. Prompt Caching para reducir costes

Si tienes un system prompt largo o documentos que se reusan entre llamadas, el Prompt Caching puede reducir el coste hasta un 90%:

```python
import anthropic

client = anthropic.Anthropic()

# Sistema prompt largo (p.ej. 2000 tokens de instrucciones de empresa)
SYSTEM_PROMPT = """
Eres el asistente oficial de ACME Corp. Sigues estas normas:
[... 2000 palabras de instrucciones, documentos de empresa, FAQs ...]
"""

def preguntar_con_cache(pregunta: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # Cachea este bloque
            }
        ],
        messages=[{"role": "user", "content": pregunta}],
    )
    uso = response.usage
    print(f"Tokens cacheados: {getattr(uso, 'cache_read_input_tokens', 0)}")
    return response.content[0].text

# Primera llamada: crea la caché
r1 = preguntar_con_cache("¿Cuál es vuestra política de devoluciones?")

# Segunda llamada: usa la caché (90% más barato en el system prompt)
r2 = preguntar_con_cache("¿Tenéis envío gratuito?")
```

## 3. Retry logic con backoff exponencial

La API puede devolver errores `529 Overloaded` en picos de tráfico. Implementa retries:

```python
import anthropic
import time
import random
from typing import Any

client = anthropic.Anthropic()

def llamar_con_retry(
    model: str,
    messages: list,
    max_tokens: int = 1024,
    max_intentos: int = 3,
    **kwargs: Any,
) -> anthropic.types.Message:
    for intento in range(max_intentos):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs,
            )
        except anthropic.RateLimitError:
            if intento == max_intentos - 1:
                raise
            espera = (2 ** intento) + random.uniform(0, 1)
            print(f"Rate limit. Esperando {espera:.1f}s...")
            time.sleep(espera)
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and intento < max_intentos - 1:
                espera = (2 ** intento) * 5
                print(f"API sobrecargada. Esperando {espera}s...")
                time.sleep(espera)
            else:
                raise
    raise RuntimeError("Todos los intentos fallaron")
```

## 4. Rate limiting en tu aplicación

Si expones la API a múltiples usuarios, controla el consumo:

```python
import time
from collections import defaultdict, deque

class RateLimiter:
    """Limita a N peticiones por usuario por ventana de tiempo."""

    def __init__(self, max_peticiones: int = 10, ventana_segundos: int = 60):
        self.max_peticiones = max_peticiones
        self.ventana = ventana_segundos
        self.historial: dict[str, deque] = defaultdict(deque)

    def permite(self, usuario_id: str) -> bool:
        ahora = time.time()
        historial = self.historial[usuario_id]

        # Eliminar peticiones fuera de la ventana
        while historial and historial[0] < ahora - self.ventana:
            historial.popleft()

        if len(historial) >= self.max_peticiones:
            return False

        historial.append(ahora)
        return True

limiter = RateLimiter(max_peticiones=10, ventana_segundos=60)

def procesar_peticion(usuario_id: str, prompt: str) -> str:
    if not limiter.permite(usuario_id):
        raise ValueError(f"Límite de peticiones alcanzado para {usuario_id}")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

## 5. Monitorización con Langfuse

Registra cada llamada para detectar regresiones de calidad y optimizar prompts:

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import anthropic

langfuse = Langfuse()
client = anthropic.Anthropic()

@observe()
def generar_con_trazas(prompt: str, usuario_id: str) -> str:
    langfuse_context.update_current_trace(user_id=usuario_id)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    texto = response.content[0].text

    # Registrar métricas de uso
    langfuse_context.update_current_observation(
        usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
        output=texto,
    )
    return texto
```

## 6. Streaming para mejor UX

En aplicaciones de cara al usuario, el streaming reduce la latencia percibida:

```python
import anthropic

client = anthropic.Anthropic()

def responder_en_streaming(prompt: str):
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for texto in stream.text_stream:
            print(texto, end="", flush=True)
        print()
        # Estadísticas finales
        msg = stream.get_final_message()
        print(f"\n[{msg.usage.input_tokens}→{msg.usage.output_tokens} tokens]")

responder_en_streaming("Explica cómo funciona la atención en los Transformers.")
```

## 7. Structured output con Pydantic

Usa tool use para garantizar salidas estructuradas y validadas:

```python
import anthropic
from pydantic import BaseModel
import json

client = anthropic.Anthropic()

class AnalisisEmail(BaseModel):
    urgente: bool
    categoria: str  # "soporte", "ventas", "rrhh", "otro"
    resumen: str
    accion_recomendada: str

def analizar_email(texto_email: str) -> AnalisisEmail:
    tool = {
        "name": "clasificar_email",
        "description": "Clasifica y analiza un email entrante",
        "input_schema": AnalisisEmail.model_json_schema(),
    }

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku es suficiente para clasificación
        max_tokens=512,
        tools=[tool],
        tool_choice={"type": "tool", "name": "clasificar_email"},
        messages=[{"role": "user", "content": f"Analiza este email:\n\n{texto_email}"}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return AnalisisEmail(**tool_use.input)

resultado = analizar_email("Asunto: URGENTE - Servidor caído en producción. El cliente no puede acceder desde hace 2h.")
print(resultado.model_dump_json(indent=2))
```

## Resumen de mejores prácticas

| Práctica | Impacto | Esfuerzo |
|----------|---------|----------|
| Router de modelos | -70% coste | Medio |
| Prompt Caching | -90% en prompts estáticos | Bajo |
| Retry con backoff | +99.9% disponibilidad | Bajo |
| Rate limiting | Controla gasto | Medio |
| Monitorización (Langfuse) | Detecta regresiones | Medio |
| Streaming | Mejor UX | Bajo |
| Structured output | Parseo fiable | Bajo |
