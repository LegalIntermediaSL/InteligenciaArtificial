# Gestión de Errores y Resiliencia en APIs de IA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/apis/06-gestion-errores-resilience.ipynb)

Las APIs de IA fallan más que las APIs REST tradicionales: rate limits, sobrecarga del proveedor,
timeouts en respuestas largas. Este artículo muestra cómo construir clientes resilientes.

---

## 1. Errores comunes y cómo distinguirlos

```python
import anthropic
import openai

# Anthropic SDK — jerarquía de excepciones relevantes
ERRORES_ANTHROPIC = {
    anthropic.RateLimitError:      "429 — Demasiadas peticiones. Hacer backoff y reintentar.",
    anthropic.APITimeoutError:     "Timeout — La respuesta tardó más del límite. Reintentar.",
    anthropic.APIConnectionError:  "Red — Sin conectividad o DNS fallido. Reintentar con espera.",
    anthropic.BadRequestError:     "400 — Parámetros inválidos. NO reintentar (bug en el código).",
    anthropic.AuthenticationError: "401 — API key inválida. NO reintentar (corregir credencial).",
    anthropic.PermissionDeniedError: "403 — Sin permiso para este recurso.",
    anthropic.NotFoundError:       "404 — Recurso no encontrado.",
    anthropic.APIStatusError:      "5xx — Error del servidor. Reintentar con backoff.",
}

def clasificar_error(exc: Exception) -> tuple[bool, str]:
    """
    Devuelve (reintentable, descripcion).
    Los errores del cliente (4xx salvo 429) NO deben reintentarse.
    """
    reintentables = (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
    )
    # 5xx se mapean a APIStatusError con status_code >= 500
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True, f"Error del servidor ({exc.status_code})"

    if isinstance(exc, reintentables):
        return True, type(exc).__name__

    return False, f"Error no reintentable: {type(exc).__name__}"

# Demo de clasificación (sin llamada real)
errores_demo = [
    anthropic.RateLimitError("rate limit", response=None, body=None),
    anthropic.AuthenticationError("bad key", response=None, body=None),
]
for e in errores_demo:
    reintentable, desc = clasificar_error(e)
    icono = "↺" if reintentable else "✗"
    print(f"  {icono} {desc}")
```

---

## 2. Retry con exponential backoff

```python
import time
import random
import anthropic
from typing import Any

class RetryClient:
    """
    Envuelve el cliente de Anthropic con reintentos automáticos.
    Usa exponential backoff con jitter para evitar thundering herd.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 4,
        backoff_factor: float = 1.5,
        jitter: float = 0.3,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries   = max_retries
        self.backoff_factor = backoff_factor
        self.jitter        = jitter

    def _espera(self, intento: int) -> float:
        """Calcula el tiempo de espera con jitter multiplicativo."""
        base  = self.backoff_factor ** intento   # 1.5^0=1, 1.5^1=1.5, 1.5^2=2.25...
        ruido = random.uniform(1 - self.jitter, 1 + self.jitter)
        return base * ruido

    def create(self, **kwargs: Any) -> anthropic.types.Message:
        ultimo_error: Exception | None = None

        for intento in range(self.max_retries + 1):
            try:
                return self.client.messages.create(**kwargs)

            except (anthropic.RateLimitError,
                    anthropic.APITimeoutError,
                    anthropic.APIConnectionError) as exc:
                ultimo_error = exc
                if intento == self.max_retries:
                    break
                espera = self._espera(intento)
                print(f"  Intento {intento + 1}/{self.max_retries}: "
                      f"{type(exc).__name__} — esperando {espera:.1f}s")
                time.sleep(espera)

            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    ultimo_error = exc
                    if intento == self.max_retries:
                        break
                    espera = self._espera(intento)
                    print(f"  Intento {intento + 1}/{self.max_retries}: "
                          f"Error {exc.status_code} — esperando {espera:.1f}s")
                    time.sleep(espera)
                else:
                    raise  # 4xx no reintentable — propagar inmediatamente

        raise RuntimeError(
            f"Fallaron {self.max_retries} reintentos"
        ) from ultimo_error


# Uso
client = RetryClient(max_retries=4, backoff_factor=2.0)
# respuesta = client.create(
#     model="claude-haiku-4-5-20251001",
#     max_tokens=256,
#     messages=[{"role": "user", "content": "Hola"}],
# )
```

---

## 3. Circuit breaker

```python
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any

class Estado(Enum):
    CLOSED    = "closed"     # normal — deja pasar peticiones
    OPEN      = "open"       # bloqueado — rechaza peticiones sin llamar a la API
    HALF_OPEN = "half_open"  # prueba — deja pasar 1 petición para ver si se recuperó

@dataclass
class CircuitBreaker:
    """
    Patrón Circuit Breaker para llamadas a APIs externas.
    Evita acumular timeouts cuando el servicio está caído.
    """
    umbral_fallos:   int   = 5       # fallos consecutivos para OPEN
    timeout_reset:   float = 60.0    # segundos antes de probar HALF_OPEN
    _fallos:         int   = field(default=0, init=False)
    _estado:         Estado = field(default=Estado.CLOSED, init=False)
    _ultimo_fallo:   float = field(default=0.0, init=False)

    @property
    def estado(self) -> Estado:
        if self._estado == Estado.OPEN:
            if time.monotonic() - self._ultimo_fallo > self.timeout_reset:
                self._estado = Estado.HALF_OPEN
        return self._estado

    def llamar(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        estado = self.estado

        if estado == Estado.OPEN:
            raise RuntimeError("Circuit breaker OPEN — servicio no disponible")

        try:
            resultado = func(*args, **kwargs)
            self._en_exito()
            return resultado
        except Exception as exc:
            self._en_fallo()
            raise exc

    def _en_exito(self) -> None:
        self._fallos = 0
        self._estado = Estado.CLOSED

    def _en_fallo(self) -> None:
        self._fallos     += 1
        self._ultimo_fallo = time.monotonic()
        if self._fallos >= self.umbral_fallos:
            self._estado = Estado.OPEN
            print(f"  Circuit breaker → OPEN tras {self._fallos} fallos")

# Demo de estados
cb = CircuitBreaker(umbral_fallos=3, timeout_reset=5.0)

def llamada_fallida():
    raise anthropic.RateLimitError("rate limit", response=None, body=None)

for i in range(4):
    try:
        cb.llamar(llamada_fallida)
    except Exception as e:
        print(f"  Fallo {i+1}: estado = {cb.estado.value}")
```

---

## 4. Fallback entre proveedores

```python
import anthropic
import time
from dataclasses import dataclass
from typing import Any

@dataclass
class RespuestaFallback:
    contenido:  str
    proveedor:  str
    desde_cache: bool = False

class MultiproviderClient:
    """
    Intenta Claude → OpenAI → caché en ese orden.
    Nunca falla silenciosamente: siempre devuelve de dónde vino la respuesta.
    """

    def __init__(self, anthropic_key: str | None = None, openai_key: str | None = None):
        self._anthropic = anthropic.Anthropic(api_key=anthropic_key)
        self._cache: dict[str, str] = {}

        try:
            import openai as _openai
            self._openai = _openai.OpenAI(api_key=openai_key)
        except ImportError:
            self._openai = None

    def _intentar_anthropic(self, prompt: str, max_tokens: int) -> str:
        resp = self._anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _intentar_openai(self, prompt: str, max_tokens: int) -> str:
        if self._openai is None:
            raise RuntimeError("OpenAI no instalado")
        resp = self._openai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def completar(self, prompt: str, max_tokens: int = 512) -> RespuestaFallback:
        clave_cache = f"{prompt[:80]}:{max_tokens}"

        # 1. Intentar Claude
        try:
            texto = self._intentar_anthropic(prompt, max_tokens)
            self._cache[clave_cache] = texto
            return RespuestaFallback(texto, "claude-haiku")
        except Exception as e_claude:
            print(f"  Claude falló: {type(e_claude).__name__}")

        # 2. Intentar OpenAI
        try:
            texto = self._intentar_openai(prompt, max_tokens)
            self._cache[clave_cache] = texto
            return RespuestaFallback(texto, "gpt-4o-mini")
        except Exception as e_openai:
            print(f"  OpenAI falló: {type(e_openai).__name__}")

        # 3. Caché como último recurso
        if clave_cache in self._cache:
            print("  Usando respuesta cacheada")
            return RespuestaFallback(self._cache[clave_cache], "cache", desde_cache=True)

        raise RuntimeError("Todos los proveedores fallaron y no hay caché disponible")


# Uso
mp = MultiproviderClient()
# resultado = mp.completar("Resume en 2 líneas qué es un LLM")
# print(f"Proveedor: {resultado.proveedor}")
# print(resultado.contenido)
```

---

## 5. Timeout y límites de contexto

```python
import anthropic

client = anthropic.Anthropic()

def truncar_al_limite(
    texto: str,
    max_tokens_aprox: int = 180_000,
    chars_por_token: float = 4.0,
) -> tuple[str, bool]:
    """
    Truncado inteligente: corta por párrafo completo, no por carácter.
    Retorna (texto_truncado, fue_truncado).
    """
    limite_chars = int(max_tokens_aprox * chars_por_token)
    if len(texto) <= limite_chars:
        return texto, False

    # Buscar el último párrafo que cabe dentro del límite
    fragmento = texto[:limite_chars]
    ultimo_parrafo = fragmento.rfind("\n\n")
    if ultimo_parrafo > limite_chars // 2:
        fragmento = fragmento[:ultimo_parrafo]

    return fragmento + "\n\n[...texto truncado por límite de contexto...]", True


def llamada_con_timeout(
    prompt: str,
    timeout_segundos: int = 30,
    max_tokens: int = 1024,
) -> str:
    """Llama a la API con timeout explícito y manejo de contexto excedido."""

    # Truncar si el prompt es muy largo
    prompt_final, truncado = truncar_al_limite(prompt)
    if truncado:
        print("  Advertencia: prompt truncado por exceder límite de contexto")

    try:
        client_timeout = anthropic.Anthropic(timeout=timeout_segundos)
        respuesta = client_timeout.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt_final}],
        )
        return respuesta.content[0].text

    except anthropic.APITimeoutError:
        raise TimeoutError(f"La API no respondió en {timeout_segundos}s — considera reducir max_tokens")

    except anthropic.BadRequestError as exc:
        if "context_window" in str(exc).lower() or "too long" in str(exc).lower():
            # Reducir agresivamente y reintentar una vez
            prompt_reducido, _ = truncar_al_limite(prompt, max_tokens_aprox=50_000)
            client_timeout2 = anthropic.Anthropic(timeout=timeout_segundos)
            resp = client_timeout2.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt_reducido}],
            )
            return resp.content[0].text
        raise


# Ejemplo de uso
texto_largo = "Lorem ipsum. " * 10_000   # ~130k chars
resultado = llamada_con_timeout(texto_largo, timeout_segundos=20)
```

---

→ Anterior: [Mistral y Cohere](05-mistral-cohere.md) | → Siguiente: [APIs Avanzadas](../apis-avanzadas/01-streaming-avanzado.md)
