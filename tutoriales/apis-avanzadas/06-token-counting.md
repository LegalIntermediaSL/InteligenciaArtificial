# Token Counting API: Estimación de Costes Antes de Enviar

> **Bloque:** APIs Avanzadas · **Nivel:** Intermedio · **Tiempo estimado:** 35 min

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegalIntermediaSL/InteligenciaArtificial/blob/main/tutoriales/notebooks/apis-avanzadas/06-token-counting.ipynb)

---

## Índice

1. [Por qué contar tokens antes de enviar](#1-por-qué-contar-tokens-antes-de-enviar)
2. [API count_tokens básica](#2-api-count_tokens-básica)
3. [Conteo con system prompt, tools y multi-turno](#3-conteo-con-system-prompt-tools-y-multi-turno)
4. [TokenBudgetManager: control automático de presupuesto](#4-tokenbudgetmanager-control-automático-de-presupuesto)
5. [Estimación de coste en euros](#5-estimación-de-coste-en-euros)
6. [Integración con prompt caching](#6-integración-con-prompt-caching)
7. [Monitorización acumulada por sesión](#7-monitorización-acumulada-por-sesión)
8. [Comparativa español vs inglés](#8-comparativa-español-vs-inglés)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Por qué contar tokens antes de enviar

Los LLMs de Anthropic cobran por token (unidad de ~4 caracteres en inglés, ~3 en español). Sin control, una sola petición mal dimensionada puede:

- **Superar el límite de contexto** y fallar con error 400 (context_length_exceeded)
- **Disparar el coste** de una sesión interactiva por acumulación de historial
- **Cachear en falso:** activar el caché pensando que el prompt es grande cuando no lo es

La Token Counting API de Anthropic (`client.messages.count_tokens()`) resuelve esto permitiendo **conocer exactamente cuántos tokens consumirá una petición antes de enviarla**, sin incurrir en coste de inferencia.

**Comparativa de enfoques:**

| Enfoque | Precisión | Coste | Cuando usar |
|---------|-----------|-------|------------|
| Estimación por longitud de texto | ±30% | Gratis | Prototipos rápidos |
| Tokenizador local (tiktoken) | ±5% (para GPT; distinto para Claude) | CPU local | Aproximación offline |
| `count_tokens()` de Anthropic | **Exacto** | Gratis (sin inferencia) | Producción |

---

## 2. API count_tokens básica

```python
import anthropic

client = anthropic.Anthropic()

# Conteo simple: un solo mensaje
def contar_tokens_simple(texto: str, modelo: str = "claude-haiku-4-5-20251001") -> int:
    """
    Cuenta los tokens de un mensaje antes de enviarlo.
    No genera respuesta ni incurre en coste de inferencia.
    
    Returns:
        int: número de tokens de entrada
    """
    respuesta = client.messages.count_tokens(
        model=modelo,
        messages=[
            {"role": "user", "content": texto}
        ]
    )
    return respuesta.input_tokens


# Ejemplo básico
textos_prueba = [
    "Hola",
    "¿Cuál es la capital de Francia?",
    "Explícame en detalle cómo funciona la transformada de Fourier y sus aplicaciones en procesamiento de señales de audio.",
]

for texto in textos_prueba:
    tokens = contar_tokens_simple(texto)
    print(f"{tokens:4d} tokens | {len(texto):4d} chars | ratio: {len(texto)/tokens:.2f} chars/token | '{texto[:50]}'")
```

---

## 3. Conteo con system prompt, tools y multi-turno

En producción, el conteo debe incluir todos los componentes que se enviarán en la petición real.

```python
from typing import Optional

# Definición de herramienta de ejemplo
HERRAMIENTA_BUSQUEDA = {
    "name": "buscar_en_base_de_conocimiento",
    "description": "Busca información en la base de conocimiento interna de la empresa.",
    "input_schema": {
        "type": "object",
        "properties": {
            "consulta": {
                "type": "string",
                "description": "La pregunta o términos a buscar"
            },
            "categoria": {
                "type": "string",
                "enum": ["productos", "soporte", "facturacion", "general"],
                "description": "Categoría donde buscar"
            },
            "max_resultados": {
                "type": "integer",
                "description": "Número máximo de resultados (default: 5)"
            }
        },
        "required": ["consulta"]
    }
}

SYSTEM_PROMPT = """Eres un asistente especializado en soporte al cliente para TechStore.
Tienes acceso a nuestra base de conocimiento donde puedes buscar información sobre productos,
procedimientos de soporte, políticas de facturación y preguntas frecuentes.

Usa siempre la herramienta de búsqueda antes de responder preguntas específicas sobre productos
o procedimientos. Si no encuentras la información, dilo claramente.

Responde siempre en el idioma del cliente. Sé conciso y directo."""


def contar_tokens_completo(
    mensajes: list[dict],
    system: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    modelo: str = "claude-haiku-4-5-20251001"
) -> dict:
    """
    Cuenta tokens de una petición completa incluyendo system, tools y mensajes.
    
    Returns:
        dict con input_tokens y desglose por componente
    """
    kwargs = {
        "model": modelo,
        "messages": mensajes
    }
    
    if system:
        kwargs["system"] = system
    
    if tools:
        kwargs["tools"] = tools
    
    respuesta = client.messages.count_tokens(**kwargs)
    
    # Calcular desglose estimado (sin system/tools)
    tokens_solo_mensajes_resp = client.messages.count_tokens(
        model=modelo,
        messages=mensajes
    )
    tokens_solo_mensajes = tokens_solo_mensajes_resp.input_tokens
    
    overhead_system_tools = respuesta.input_tokens - tokens_solo_mensajes
    
    return {
        "total_input_tokens": respuesta.input_tokens,
        "tokens_mensajes": tokens_solo_mensajes,
        "overhead_system_tools": overhead_system_tools,
        "modelo": modelo
    }


# --- Ejemplo: conversación multi-turno acumulada ---
historial_conversacion = [
    {"role": "user", "content": "Hola, tengo una pregunta sobre mi pedido"},
    {"role": "assistant", "content": "Hola, claro. ¿Cuál es el número de tu pedido?"},
    {"role": "user", "content": "Es el pedido 12345. Quería saber cuándo llega."},
    {"role": "assistant", "content": "Déjame consultarlo. Un momento, por favor."},
    {"role": "user", "content": "Gracias. Y también, ¿puedo cambiar la dirección de entrega?"},
]

conteo = contar_tokens_completo(
    mensajes=historial_conversacion,
    system=SYSTEM_PROMPT,
    tools=[HERRAMIENTA_BUSQUEDA]
)

print("=== Conteo de petición completa ===")
print(f"Tokens totales de entrada: {conteo['total_input_tokens']:,}")
print(f"  - Solo mensajes: {conteo['tokens_mensajes']:,}")
print(f"  - System + tools: {conteo['overhead_system_tools']:,}")
print(f"Modelo: {conteo['modelo']}")
```

---

## 4. TokenBudgetManager: control automático de presupuesto

En aplicaciones con conversaciones largas, el historial crece y puede superar el límite del modelo (200K tokens en Claude) o simplemente costar demasiado. `TokenBudgetManager` previene ambos problemas.

```python
from dataclasses import dataclass, field
from enum import Enum

class AccionPresupuesto(Enum):
    PERMITIR = "permitir"
    ADVERTIR = "advertir"
    TRUNCAR = "truncar"
    RECHAZAR = "rechazar"


@dataclass
class TokenBudgetManager:
    """
    Gestiona el presupuesto de tokens para una sesión de conversación.
    
    Args:
        limite_tokens: Máximo de tokens de entrada permitidos por petición
        modelo: Modelo al que se enviarán las peticiones
        system_prompt: System prompt fijo (se incluye en el conteo)
        tools: Herramientas (se incluyen en el conteo)
        umbral_advertencia: Porcentaje del límite que activa advertencia (0-1)
        estrategia_truncado: "fifo" (elimina mensajes más antiguos) o "summarize"
    """
    limite_tokens: int
    modelo: str = "claude-haiku-4-5-20251001"
    system_prompt: Optional[str] = None
    tools: Optional[list] = None
    umbral_advertencia: float = 0.80
    estrategia_truncado: str = "fifo"
    
    _client: anthropic.Anthropic = field(init=False, default_factory=anthropic.Anthropic)
    
    def contar(self, mensajes: list[dict]) -> int:
        """Cuenta tokens de los mensajes con el contexto configurado."""
        kwargs = {"model": self.modelo, "messages": mensajes}
        if self.system_prompt:
            kwargs["system"] = self.system_prompt
        if self.tools:
            kwargs["tools"] = self.tools
        return self._client.messages.count_tokens(**kwargs).input_tokens
    
    def evaluar(self, mensajes: list[dict]) -> dict:
        """
        Evalúa si los mensajes caben en el presupuesto.
        
        Returns:
            dict con accion, tokens_actuales, porcentaje_uso, mensajes_procesados
        """
        tokens = self.contar(mensajes)
        porcentaje = tokens / self.limite_tokens
        
        if porcentaje >= 1.0:
            if self.estrategia_truncado == "fifo":
                mensajes_truncados = self._truncar_fifo(mensajes)
                tokens_tras_truncado = self.contar(mensajes_truncados)
                return {
                    "accion": AccionPresupuesto.TRUNCAR,
                    "tokens_originales": tokens,
                    "tokens_tras_truncado": tokens_tras_truncado,
                    "mensajes_eliminados": len(mensajes) - len(mensajes_truncados),
                    "mensajes_procesados": mensajes_truncados
                }
            else:
                return {
                    "accion": AccionPresupuesto.RECHAZAR,
                    "tokens_actuales": tokens,
                    "limite": self.limite_tokens,
                    "exceso": tokens - self.limite_tokens,
                    "mensajes_procesados": None
                }
        
        accion = AccionPresupuesto.ADVERTIR if porcentaje >= self.umbral_advertencia else AccionPresupuesto.PERMITIR
        
        return {
            "accion": accion,
            "tokens_actuales": tokens,
            "limite": self.limite_tokens,
            "porcentaje_uso": f"{porcentaje:.1%}",
            "tokens_disponibles": self.limite_tokens - tokens,
            "mensajes_procesados": mensajes
        }
    
    def _truncar_fifo(self, mensajes: list[dict]) -> list[dict]:
        """
        Elimina mensajes antiguos (FIFO) hasta que los tokens estén dentro del límite.
        Siempre conserva el último mensaje del usuario.
        """
        if not mensajes:
            return mensajes
        
        # Conservar siempre el último mensaje
        ultimo = mensajes[-1:]
        candidatos = list(mensajes[:-1])
        
        while candidatos:
            mensajes_candidatos = candidatos + ultimo
            if self.contar(mensajes_candidatos) <= self.limite_tokens * 0.9:
                return mensajes_candidatos
            # Eliminar el par más antiguo (user + assistant)
            if len(candidatos) >= 2:
                candidatos = candidatos[2:]
            else:
                candidatos = []
        
        return ultimo  # Caso extremo: solo el último mensaje


# --- Uso del TokenBudgetManager ---

manager = TokenBudgetManager(
    limite_tokens=4096,
    modelo="claude-haiku-4-5-20251001",
    system_prompt=SYSTEM_PROMPT,
    tools=[HERRAMIENTA_BUSQUEDA],
    umbral_advertencia=0.75
)

# Simular una conversación creciente
conversacion = []
for i in range(5):
    conversacion.append({
        "role": "user",
        "content": f"Pregunta {i+1}: ¿Cuáles son las características del producto {i+1}? Necesito detalles completos."
    })
    conversacion.append({
        "role": "assistant",
        "content": f"El producto {i+1} tiene las siguientes características: procesador de última generación, 16GB RAM, SSD de 512GB, pantalla de alta resolución, batería de larga duración y garantía de 2 años."
    })

evaluacion = manager.evaluar(conversacion)
print(f"Acción recomendada: {evaluacion['accion'].value}")
print(f"Tokens actuales: {evaluacion.get('tokens_actuales', 'N/A')}")
print(f"Uso del presupuesto: {evaluacion.get('porcentaje_uso', 'N/A')}")
```

---

## 5. Estimación de coste en euros

```python
from datetime import datetime

# Precios de Anthropic en USD (abril 2026) — verificar en https://www.anthropic.com/pricing
PRECIOS_POR_MILLON_TOKENS = {
    "claude-haiku-4-5-20251001": {
        "input": 0.80,       # USD por millón de tokens de entrada
        "output": 4.00,      # USD por millón de tokens de salida
        "cache_write": 1.00, # USD por millón de tokens escritos en caché
        "cache_read": 0.08,  # USD por millón de tokens leídos de caché
    },
    "claude-sonnet-4-5-20251001": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-opus-4-5": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    }
}

USD_A_EUR = 0.92  # Tasa aproximada — usar API de divisas en producción


def estimar_coste(
    tokens_entrada: int,
    tokens_salida_estimados: int,
    modelo: str = "claude-haiku-4-5-20251001",
    tokens_cache_write: int = 0,
    tokens_cache_read: int = 0
) -> dict:
    """
    Estima el coste en USD y EUR de una petición antes de enviarla.
    
    Args:
        tokens_entrada: Tokens de entrada (de count_tokens())
        tokens_salida_estimados: Estimación de tokens de salida
        modelo: Modelo a usar
        tokens_cache_write: Tokens que se escribirán en caché (primera vez)
        tokens_cache_read: Tokens que se leerán de caché
    
    Returns:
        dict con coste desglosado en USD y EUR
    """
    if modelo not in PRECIOS_POR_MILLON_TOKENS:
        raise ValueError(f"Modelo no reconocido: {modelo}. Disponibles: {list(PRECIOS_POR_MILLON_TOKENS.keys())}")
    
    precios = PRECIOS_POR_MILLON_TOKENS[modelo]
    
    # Tokens de entrada "normales" (sin caché)
    tokens_entrada_normales = tokens_entrada - tokens_cache_write - tokens_cache_read
    
    coste_entrada_usd = (tokens_entrada_normales / 1_000_000) * precios["input"]
    coste_salida_usd = (tokens_salida_estimados / 1_000_000) * precios["output"]
    coste_cache_write_usd = (tokens_cache_write / 1_000_000) * precios["cache_write"]
    coste_cache_read_usd = (tokens_cache_read / 1_000_000) * precios["cache_read"]
    
    coste_total_usd = coste_entrada_usd + coste_salida_usd + coste_cache_write_usd + coste_cache_read_usd
    
    return {
        "modelo": modelo,
        "tokens": {
            "entrada_normal": tokens_entrada_normales,
            "salida_estimada": tokens_salida_estimados,
            "cache_write": tokens_cache_write,
            "cache_read": tokens_cache_read,
        },
        "coste_usd": {
            "entrada": round(coste_entrada_usd, 6),
            "salida": round(coste_salida_usd, 6),
            "cache_write": round(coste_cache_write_usd, 6),
            "cache_read": round(coste_cache_read_usd, 6),
            "total": round(coste_total_usd, 6),
        },
        "coste_eur": {
            "total": round(coste_total_usd * USD_A_EUR, 6),
            "total_milesimas": f"€{coste_total_usd * USD_A_EUR * 1000:.4f} (por mil peticiones)",
        }
    }


# Ejemplo: estimar antes de una petición de análisis
mensajes_analisis = [
    {"role": "user", "content": "Analiza este contrato de 5 páginas y extrae todas las cláusulas de penalización..."}
]

tokens_entrada = contar_tokens_completo(mensajes_analisis, system=SYSTEM_PROMPT)["total_input_tokens"]
estimacion = estimar_coste(
    tokens_entrada=tokens_entrada,
    tokens_salida_estimados=500,  # Estimamos una respuesta moderada
    modelo="claude-haiku-4-5-20251001"
)

print("=== Estimación de coste ===")
print(f"Tokens entrada: {estimacion['tokens']['entrada_normal']:,}")
print(f"Tokens salida (estimados): {estimacion['tokens']['salida_estimada']:,}")
print(f"Coste USD: ${estimacion['coste_usd']['total']:.6f}")
print(f"Coste EUR: €{estimacion['coste_eur']['total']:.6f}")
print(f"Por 1.000 peticiones: {estimacion['coste_eur']['total_milesimas']}")
```

---

## 6. Integración con prompt caching

Antes de activar el caché, puedes predecir exactamente cuánto ahorrarás.

```python
def analizar_oportunidad_cache(
    mensajes: list[dict],
    system_prompt: str,
    n_peticiones_esperadas: int,
    modelo: str = "claude-haiku-4-5-20251001"
) -> dict:
    """
    Analiza si vale la pena activar prompt caching.
    
    Compara el coste de N peticiones con y sin caché.
    
    Returns:
        dict con análisis de rentabilidad del caché
    """
    precios = PRECIOS_POR_MILLON_TOKENS[modelo]
    
    # Tokens del system prompt (la parte que se cachearía)
    resp_system = client.messages.count_tokens(
        model=modelo,
        system=system_prompt,
        messages=[{"role": "user", "content": "x"}]  # Mensaje mínimo necesario
    )
    
    resp_total = client.messages.count_tokens(
        model=modelo,
        system=system_prompt,
        messages=mensajes
    )
    
    tokens_system = resp_system.input_tokens - 1  # Descontar el mensaje mínimo
    tokens_mensajes = resp_total.input_tokens - tokens_system
    tokens_salida_est = 300  # Estimación de salida por petición
    
    # --- Sin caché ---
    coste_sin_cache_por_peticion = (
        (resp_total.input_tokens / 1_000_000) * precios["input"] +
        (tokens_salida_est / 1_000_000) * precios["output"]
    )
    coste_total_sin_cache = coste_sin_cache_por_peticion * n_peticiones_esperadas
    
    # --- Con caché ---
    # Primera petición: coste normal + coste de escritura en caché
    coste_primera_con_cache = (
        (tokens_mensajes / 1_000_000) * precios["input"] +
        (tokens_system / 1_000_000) * precios["cache_write"] +
        (tokens_salida_est / 1_000_000) * precios["output"]
    )
    
    # Peticiones sucesivas: mensajes normales + lectura barata del caché
    coste_sucesiva_con_cache = (
        (tokens_mensajes / 1_000_000) * precios["input"] +
        (tokens_system / 1_000_000) * precios["cache_read"] +
        (tokens_salida_est / 1_000_000) * precios["output"]
    )
    
    coste_total_con_cache = coste_primera_con_cache + (coste_sucesiva_con_cache * (n_peticiones_esperadas - 1))
    
    ahorro_usd = coste_total_sin_cache - coste_total_con_cache
    ahorro_porcentaje = (ahorro_usd / coste_total_sin_cache) * 100 if coste_total_sin_cache > 0 else 0
    
    # Punto de break-even (¿desde cuántas peticiones es rentable el caché?)
    # Primera con caché es más cara; a partir de N la diferencia acumulada se vuelve positiva
    coste_overhead_primera = coste_primera_con_cache - coste_sin_cache_por_peticion
    ahorro_por_peticion_sucesiva = coste_sin_cache_por_peticion - coste_sucesiva_con_cache
    breakeven = max(1, int(coste_overhead_primera / ahorro_por_peticion_sucesiva) + 2) if ahorro_por_peticion_sucesiva > 0 else float("inf")
    
    return {
        "tokens_cacheables": tokens_system,
        "tokens_por_peticion": tokens_mensajes,
        "n_peticiones": n_peticiones_esperadas,
        "coste_sin_cache_eur": round(coste_total_sin_cache * USD_A_EUR, 4),
        "coste_con_cache_eur": round(coste_total_con_cache * USD_A_EUR, 4),
        "ahorro_eur": round(ahorro_usd * USD_A_EUR, 4),
        "ahorro_porcentaje": f"{ahorro_porcentaje:.1f}%",
        "breakeven_peticiones": breakeven,
        "recomendacion": "ACTIVAR caché" if ahorro_usd > 0 and breakeven <= n_peticiones_esperadas else "No necesario"
    }


# Análisis para un chatbot con system prompt largo
analisis = analizar_oportunidad_cache(
    mensajes=[{"role": "user", "content": "¿Cómo puedo devolver un producto?"}],
    system_prompt=SYSTEM_PROMPT * 5,  # System prompt más largo para el ejemplo
    n_peticiones_esperadas=100,
    modelo="claude-haiku-4-5-20251001"
)

print("=== Análisis de oportunidad de caché ===")
for k, v in analisis.items():
    print(f"  {k}: {v}")
```

---

## 7. Monitorización acumulada por sesión

```python
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class MonitorTokens:
    """
    Monitoriza el uso acumulado de tokens y costes por sesión.
    Registra tanto estimaciones (pre-envío) como valores reales (post-respuesta).
    """
    modelo: str = "claude-haiku-4-5-20251001"
    _registros: list = field(default_factory=list)
    _total_tokens_estimados: int = 0
    _total_tokens_reales_entrada: int = 0
    _total_tokens_reales_salida: int = 0
    
    def registrar_estimacion(self, tokens_estimados: int, contexto: str = "") -> None:
        """Registra una estimación pre-envío."""
        self._registros.append({
            "timestamp": datetime.now().isoformat(),
            "tipo": "estimacion",
            "tokens": tokens_estimados,
            "contexto": contexto
        })
        self._total_tokens_estimados += tokens_estimados
    
    def registrar_uso_real(self, usage, contexto: str = "") -> None:
        """
        Registra el uso real post-respuesta.
        usage: objeto usage de la respuesta de Anthropic
        """
        self._registros.append({
            "timestamp": datetime.now().isoformat(),
            "tipo": "real",
            "tokens_entrada": usage.input_tokens,
            "tokens_salida": usage.output_tokens,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
            "contexto": contexto
        })
        self._total_tokens_reales_entrada += usage.input_tokens
        self._total_tokens_reales_salida += usage.output_tokens
    
    def dashboard(self) -> dict:
        """Genera un resumen del uso de la sesión."""
        coste_total = estimar_coste(
            tokens_entrada=self._total_tokens_reales_entrada,
            tokens_salida_estimados=self._total_tokens_reales_salida,
            modelo=self.modelo
        )
        
        n_peticiones = sum(1 for r in self._registros if r["tipo"] == "real")
        
        return {
            "sesion": {
                "peticiones_realizadas": n_peticiones,
                "peticiones_estimadas": sum(1 for r in self._registros if r["tipo"] == "estimacion"),
                "inicio": self._registros[0]["timestamp"] if self._registros else None,
            },
            "tokens": {
                "entrada_total": self._total_tokens_reales_entrada,
                "salida_total": self._total_tokens_reales_salida,
                "total": self._total_tokens_reales_entrada + self._total_tokens_reales_salida,
                "media_por_peticion": int(self._total_tokens_reales_entrada / n_peticiones) if n_peticiones > 0 else 0,
            },
            "costes": {
                "total_eur": coste_total["coste_eur"]["total"],
                "media_por_peticion_eur": round(coste_total["coste_eur"]["total"] / n_peticiones, 6) if n_peticiones > 0 else 0,
                "proyeccion_1000_peticiones_eur": round(coste_total["coste_eur"]["total"] / n_peticiones * 1000, 4) if n_peticiones > 0 else 0,
            }
        }
    
    def imprimir_dashboard(self) -> None:
        """Muestra el dashboard formateado."""
        d = self.dashboard()
        print("\n" + "="*50)
        print("       DASHBOARD DE USO DE TOKENS")
        print("="*50)
        print(f"Peticiones realizadas: {d['sesion']['peticiones_realizadas']}")
        print(f"Tokens entrada total:  {d['tokens']['entrada_total']:,}")
        print(f"Tokens salida total:   {d['tokens']['salida_total']:,}")
        print(f"Media tokens/petición: {d['tokens']['media_por_peticion']:,}")
        print(f"Coste total sesión:    €{d['costes']['total_eur']:.6f}")
        print(f"Coste medio/petición:  €{d['costes']['media_por_peticion_eur']:.6f}")
        print(f"Proyección 1.000 pet.: €{d['costes']['proyeccion_1000_peticiones_eur']:.4f}")
        print("="*50)


# Uso del monitor en un flujo real
monitor = MonitorTokens(modelo="claude-haiku-4-5-20251001")

# Simular 3 peticiones con estimación previa y registro posterior
for i, mensaje in enumerate(["¿Precio del producto A?", "¿Disponibilidad del producto B?", "¿Garantía del producto C?"], 1):
    msgs = [{"role": "user", "content": mensaje}]
    
    # 1. Estimar antes
    tokens_est = contar_tokens_simple(mensaje)
    monitor.registrar_estimacion(tokens_est, contexto=f"peticion_{i}")
    
    # 2. Enviar
    respuesta = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=msgs
    )
    
    # 3. Registrar uso real
    monitor.registrar_uso_real(respuesta.usage, contexto=f"peticion_{i}")

monitor.imprimir_dashboard()
```

---

## 8. Comparativa español vs inglés

El español utiliza en promedio más tokens que el inglés para el mismo contenido porque el tokenizador de Claude fue entrenado principalmente con texto en inglés.

```python
def comparar_idiomas(contenido_base: str, contenido_ingles: str) -> dict:
    """
    Compara el número de tokens entre versiones en español e inglés.
    
    Args:
        contenido_base: Texto en español
        contenido_ingles: Misma idea en inglés
    
    Returns:
        dict con análisis comparativo
    """
    tokens_es = contar_tokens_simple(contenido_base)
    tokens_en = contar_tokens_simple(contenido_ingles)
    
    diferencia = tokens_es - tokens_en
    ratio = tokens_es / tokens_en if tokens_en > 0 else 1
    
    return {
        "espanol": {
            "texto": contenido_base[:80] + "..." if len(contenido_base) > 80 else contenido_base,
            "caracteres": len(contenido_base),
            "tokens": tokens_es,
            "chars_por_token": round(len(contenido_base) / tokens_es, 2)
        },
        "ingles": {
            "texto": contenido_ingles[:80] + "..." if len(contenido_ingles) > 80 else contenido_ingles,
            "caracteres": len(contenido_ingles),
            "tokens": tokens_en,
            "chars_por_token": round(len(contenido_ingles) / tokens_en, 2)
        },
        "diferencia_tokens": diferencia,
        "ratio_es_en": round(ratio, 3),
        "overhead_espanol_pct": f"+{(ratio-1)*100:.1f}%" if ratio > 1 else f"{(ratio-1)*100:.1f}%"
    }


# Pares de textos equivalentes
pares = [
    (
        "El procesamiento del lenguaje natural es una rama de la inteligencia artificial.",
        "Natural language processing is a branch of artificial intelligence."
    ),
    (
        "Por favor, proporcione información detallada sobre el procedimiento de devolución de productos.",
        "Please provide detailed information about the product return procedure."
    ),
    (
        "La transformada de Fourier permite descomponer señales complejas en sus componentes frecuenciales.",
        "The Fourier transform allows complex signals to be decomposed into their frequency components."
    ),
    (
        "Necesito que me expliques paso a paso cómo configurar el sistema de autenticación de dos factores.",
        "I need you to explain step by step how to configure the two-factor authentication system."
    ),
]

print("=== Comparativa de tokens: Español vs Inglés ===\n")
totales_es, totales_en = 0, 0
for texto_es, texto_en in pares:
    resultado = comparar_idiomas(texto_es, texto_en)
    print(f"ES ({resultado['espanol']['tokens']:3d} tokens): {resultado['espanol']['texto'][:60]}")
    print(f"EN ({resultado['ingles']['tokens']:3d} tokens): {resultado['ingles']['texto'][:60]}")
    print(f"Overhead español: {resultado['overhead_espanol_pct']} ({resultado['diferencia_tokens']:+d} tokens)")
    print()
    totales_es += resultado['espanol']['tokens']
    totales_en += resultado['ingles']['tokens']

print(f"TOTAL — ES: {totales_es} tokens | EN: {totales_en} tokens | Ratio: {totales_es/totales_en:.3f}x")
print(f"\nImplicación de coste: los prompts en español cuestan aprox. {(totales_es/totales_en-1)*100:.0f}% más que en inglés")
print("Estrategia: para reducir costes en producción, considera system prompts en inglés + respuestas en español")
```

---

## 9. Extensiones sugeridas

- **Alertas de coste:** integrar `MonitorTokens` con webhooks (Slack, email) cuando el coste acumulado supere un umbral
- **Presupuesto por usuario:** usar `TokenBudgetManager` con límites por usuario/plan en aplicaciones SaaS
- **Cache hit prediction:** antes de decidir si cachear, usar `count_tokens()` para verificar que el bloque cumple el mínimo (1.024 tokens para Haiku, 2.048 para Sonnet/Opus)
- **Optimización automática:** comparar automáticamente el coste de diferentes modelos para la misma tarea y seleccionar el más económico que cumpla la calidad requerida
- **Compresión de historial:** cuando el historial supera el 70% del presupuesto, usar un LLM para resumirlo antes de la siguiente petición

---

*Siguiente artículo: [07 — Structured Outputs: Respuestas JSON Garantizadas](07-structured-outputs.md)*
