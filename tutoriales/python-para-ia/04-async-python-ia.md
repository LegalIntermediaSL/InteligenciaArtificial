# Python Asíncrono para IA: Llamadas Concurrentes y Streaming

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/python-para-ia/04-async-python-ia.ipynb)

Cada llamada a un LLM puede tardar 1-10 segundos. Procesar 100 documentos en serie tomaría
horas; en paralelo, minutos. Este artículo muestra cómo usar `asyncio` con la API de Anthropic.

---

## 1. Por qué async en IA

```python
import time

# Simulación: comparar serie vs paralelo para 8 llamadas de 1s cada una
def simular_llamada_serie(n: int, latencia: float = 1.0) -> float:
    inicio = time.monotonic()
    for _ in range(n):
        time.sleep(latencia)   # bloqueante
    return time.monotonic() - inicio

# Serie: 8 llamadas × 1s = ~8s
# Paralelo (asyncio): 8 llamadas concurrentes = ~1s + overhead

print("Procesamiento serie:")
print(f"  8 documentos × 1s = ~8s de espera")
print()
print("Procesamiento asíncrono (asyncio.gather):")
print(f"  8 documentos concurrentes = ~1s + pequeño overhead")
print()
print("Casos de uso ideales para async:")
print("  • Clasificar N chunks de texto de forma independiente")
print("  • Traducir un artículo a 5 idiomas en paralelo")
print("  • Generar embeddings por lotes")
print("  • Resumir múltiples documentos simultáneamente")
```

---

## 2. `asyncio` básico

```python
import asyncio
import time

# --- async def y await ---
async def tarea(nombre: str, segundos: float) -> str:
    await asyncio.sleep(segundos)   # no bloquea el event loop
    return f"{nombre} completada en {segundos}s"

# --- asyncio.gather: ejecutar N corrutinas en paralelo ---
async def demo_gather() -> None:
    inicio = time.monotonic()
    resultados = await asyncio.gather(
        tarea("A", 0.3),
        tarea("B", 0.5),
        tarea("C", 0.2),
    )
    transcurrido = time.monotonic() - inicio
    for r in resultados:
        print(f"  {r}")
    print(f"  Tiempo total: {transcurrido:.2f}s (no ~1.0s — fueron concurrentes)")

asyncio.run(demo_gather())

# --- asyncio.Semaphore: limitar concurrencia ---
async def demo_semaforo() -> None:
    semaforo = asyncio.Semaphore(3)   # máximo 3 tareas simultáneas

    async def tarea_limitada(i: int) -> str:
        async with semaforo:          # espera si ya hay 3 ejecutándose
            await asyncio.sleep(0.1)
            return f"tarea-{i}"

    resultados = await asyncio.gather(*[tarea_limitada(i) for i in range(8)])
    print(f"  Completadas {len(resultados)} tareas con semáforo=3")

asyncio.run(demo_semaforo())
```

---

## 3. `anthropic.AsyncAnthropic`

```python
import asyncio
import anthropic

# El cliente async es idéntico al sync en firma — solo cambia el await
async def preguntar(pregunta: str, max_tokens: int = 256) -> str:
    client = anthropic.AsyncAnthropic()   # lee ANTHROPIC_API_KEY del entorno

    mensaje = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": pregunta}],
    )
    return mensaje.content[0].text

async def main() -> None:
    respuesta = await preguntar("¿Cuál es la capital de Francia? Responde en 1 línea.")
    print(respuesta)

asyncio.run(main())


# Patrón recomendado: crear el cliente una sola vez y reutilizarlo
async def main_reutilizar() -> None:
    async with anthropic.AsyncAnthropic() as client:
        # Todas las llamadas dentro del bloque comparten la sesión HTTP
        r1 = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": "Di 'hola'"}],
        )
        r2 = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=64,
            messages=[{"role": "user", "content": "Di 'adiós'"}],
        )
    print(r1.content[0].text)
    print(r2.content[0].text)
```

---

## 4. Procesamiento por lotes con semáforo

```python
import asyncio
import time
from dataclasses import dataclass
import anthropic

@dataclass
class ResultadoLote:
    indice:    int
    documento: str
    resumen:   str
    error:     str | None = None

class BatchProcessor:
    """
    Procesa N documentos con un máximo de `concurrencia` llamadas simultáneas.
    Captura errores por documento sin interrumpir el resto del lote.
    """

    def __init__(self, concurrencia: int = 5):
        self.concurrencia = concurrencia
        self._client: anthropic.AsyncAnthropic | None = None

    async def _resumir(
        self,
        semaforo: asyncio.Semaphore,
        indice: int,
        documento: str,
    ) -> ResultadoLote:
        async with semaforo:
            try:
                assert self._client is not None
                respuesta = await self._client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=128,
                    messages=[{
                        "role": "user",
                        "content": f"Resume en 1 oración:\n\n{documento[:2000]}",
                    }],
                )
                resumen = respuesta.content[0].text.strip()
                return ResultadoLote(indice, documento[:40] + "...", resumen)
            except Exception as exc:
                return ResultadoLote(
                    indice, documento[:40] + "...", "",
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def procesar(self, documentos: list[str]) -> list[ResultadoLote]:
        semaforo = asyncio.Semaphore(self.concurrencia)

        async with anthropic.AsyncAnthropic() as client:
            self._client = client
            tareas = [
                self._resumir(semaforo, i, doc)
                for i, doc in enumerate(documentos)
            ]
            return await asyncio.gather(*tareas)


# Demo
async def demo_lote() -> None:
    documentos = [
        "La inteligencia artificial es una rama de la informática que estudia sistemas capaces de realizar tareas que requieren inteligencia humana.",
        "El aprendizaje automático permite a las máquinas aprender de datos sin ser programadas explícitamente para cada tarea.",
        "Las redes neuronales profundas están inspiradas en el funcionamiento del cerebro humano y se usan en visión, lenguaje y audio.",
        "Los modelos de lenguaje grande como GPT o Claude son entrenados sobre enormes corpus de texto de internet.",
        "El fine-tuning adapta un modelo preentrenado a una tarea específica usando un conjunto de datos mucho más pequeño.",
    ]

    procesador = BatchProcessor(concurrencia=3)
    inicio = time.monotonic()
    resultados = await procesador.procesar(documentos)
    transcurrido = time.monotonic() - inicio

    for r in resultados:
        estado = "OK" if not r.error else f"ERROR: {r.error}"
        print(f"  [{r.indice}] {estado}")
        if r.resumen:
            print(f"       {r.resumen}")

    print(f"\n  {len(documentos)} documentos en {transcurrido:.1f}s con concurrencia=3")

asyncio.run(demo_lote())
```

---

## 5. Streaming asíncrono

```python
import asyncio
import anthropic

async def stream_respuesta(prompt: str) -> str:
    """Imprime tokens conforme llegan y devuelve el texto completo."""
    client = anthropic.AsyncAnthropic()
    texto_completo = ""

    async with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for evento in stream:
            if (
                evento.type == "content_block_delta"
                and hasattr(evento.delta, "text")
            ):
                fragmento = evento.delta.text
                print(fragmento, end="", flush=True)
                texto_completo += fragmento

    print()  # nueva línea al terminar
    return texto_completo


async def stream_multiples(prompts: list[str]) -> None:
    """Muestra el streaming de N peticiones en paralelo — una tarea por prompt."""

    async def worker(i: int, prompt: str) -> None:
        client = anthropic.AsyncAnthropic()
        tokens: list[str] = []
        async with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for evento in stream:
                if (
                    evento.type == "content_block_delta"
                    and hasattr(evento.delta, "text")
                ):
                    tokens.append(evento.delta.text)
        print(f"  Tarea {i}: {''.join(tokens)[:80]}")

    await asyncio.gather(*[worker(i, p) for i, p in enumerate(prompts)])


asyncio.run(stream_respuesta("Explica qué es asyncio en Python en 3 líneas."))
```

---

## 6. Rate limiting inteligente — token bucket asíncrono

```python
import asyncio
import time
import anthropic

class TokenBucket:
    """
    Algoritmo token bucket para respetar límites de la API.
    Permite ráfagas cortas pero mantiene la tasa media bajo control.
    """

    def __init__(self, tasa: float = 60.0, capacidad: float = 10.0):
        """
        tasa:     peticiones por minuto (60 req/min → 1 req/s)
        capacidad: máximo de tokens acumulables (ráfagas de hasta N req)
        """
        self.tasa      = tasa / 60.0    # convertir a req/segundo
        self.capacidad = capacidad
        self._tokens   = capacidad
        self._ultimo   = time.monotonic()
        self._lock     = asyncio.Lock()

    async def adquirir(self) -> None:
        async with self._lock:
            ahora = time.monotonic()
            # Recargar tokens según tiempo transcurrido
            ganados = (ahora - self._ultimo) * self.tasa
            self._tokens = min(self.capacidad, self._tokens + ganados)
            self._ultimo = ahora

            if self._tokens < 1.0:
                # Calcular cuánto esperar para tener 1 token
                espera = (1.0 - self._tokens) / self.tasa
                await asyncio.sleep(espera)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class ClienteConRateLimit:
    """Cliente de Anthropic con rate limiting automático."""

    def __init__(self, req_por_minuto: int = 50):
        self._client = anthropic.AsyncAnthropic()
        self._bucket = TokenBucket(tasa=float(req_por_minuto), capacidad=5.0)

    async def crear(self, **kwargs) -> anthropic.types.Message:
        await self._bucket.adquirir()   # esperar si estamos al límite
        return await self._client.messages.create(**kwargs)


# Demo: 12 llamadas con límite de 60 req/min
async def demo_rate_limit() -> None:
    cliente = ClienteConRateLimit(req_por_minuto=60)

    async def llamada(i: int) -> None:
        inicio = time.monotonic()
        # En un test real se haría la llamada; aquí simulamos
        await asyncio.sleep(0.05)   # simula latencia de red
        print(f"  req {i:2d} — t={time.monotonic() - inicio_global:.2f}s")

    inicio_global = time.monotonic()
    await asyncio.gather(*[llamada(i) for i in range(12)])
    print(f"\n  Total: {time.monotonic() - inicio_global:.2f}s")

asyncio.run(demo_rate_limit())
```

---

→ Anterior: [Jupyter Notebooks](03-jupyter-notebooks.md) | → Siguiente: [Producción](../produccion/01-deploy-api.md)
