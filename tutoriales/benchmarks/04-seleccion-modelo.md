# Guía de selección de modelo para producción

## Framework de decisión

La selección de modelo en producción depende de tres ejes: **calidad requerida**, **latencia aceptable** y **presupuesto**. Antes de elegir, responde estas preguntas:

1. ¿Cuánto puede costar cada llamada?
2. ¿Cuántos milisegundos de latencia tolera el usuario?
3. ¿Qué nivel de calidad necesita la tarea?
4. ¿Hay restricciones de privacidad de datos?
5. ¿Necesitas multimodalidad?

## Árbol de decisión

```
¿Los datos son altamente sensibles (no pueden salir de tu infraestructura)?
├── Sí → Modelo open-source self-hosted (Llama 3.1 70B con vLLM)
└── No → ¿Cuánta complejidad tiene la tarea?
    │
    ├── Baja (clasificación, extracción simple, FAQ)
    │   └── ¿Necesitas < 500ms de latencia?
    │       ├── Sí → claude-haiku-4-5-20251001 / gpt-4o-mini / gemini-flash
    │       └── No → claude-haiku-4-5-20251001
    │
    ├── Media (resumen, generación de contenido, chatbot, código moderado)
    │   └── claude-sonnet-4-6 (default recomendado)
    │
    └── Alta (análisis profundo, arquitecturas, razonamiento matemático)
        └── ¿Necesitas Extended Thinking?
            ├── Sí → claude-opus-4-7
            └── No → claude-opus-4-7 o claude-sonnet-4-6 con CoT explícito
```

## Recomendaciones por caso de uso

| Caso de uso | Modelo recomendado | Justificación |
|-------------|-------------------|---------------|
| Chatbot de FAQ | Haiku 4.5 | Respuestas predecibles, alto volumen |
| Clasificación de emails | Haiku 4.5 | Tarea simple, latencia baja |
| Resumen de documentos | Sonnet 4.6 | Requiere coherencia y comprensión |
| Generación de contenido | Sonnet 4.6 | Balance calidad/coste |
| Análisis de contratos | Opus 4.7 | Precisión legal crítica |
| Generación de código | Sonnet 4.6 | Excelente en código, más económico que Opus |
| Debugging complejo | Opus 4.7 | Razonamiento multi-paso |
| Computer Use / RPA | Sonnet 4.6 | Recomendado por Anthropic para CU |
| Embeddings semánticos | text-embedding-3-small / voyage-3 | Modelos especializados |
| Transcripción de audio | Whisper (OpenAI) | Especializado en ASR |

## Router de modelos en producción

```python
import anthropic
from enum import Enum
import re

client = anthropic.Anthropic()

class NivelModelo(Enum):
    RAPIDO = "claude-haiku-4-5-20251001"
    ESTANDAR = "claude-sonnet-4-6"
    POTENTE = "claude-opus-4-7"

# Patrones que indican alta complejidad
PATRONES_COMPLEJOS = re.compile(
    r"\b(analiza|diseña|compara|razona|evalúa|optimiza|arquitectura|estrategia|"
    r"complejidad|multi.?(paso|etapa)|exhaustivo|profund|detallad)\b",
    re.IGNORECASE,
)

# Patrones que indican tarea simple
PATRONES_SIMPLES = re.compile(
    r"\b(clasifica|resume brevemente|extrae|sí o no|verdadero o falso|traduce|"
    r"qué es|define|lista|enumera)\b",
    re.IGNORECASE,
)

def seleccionar_modelo(prompt: str, forzar_nivel: NivelModelo | None = None) -> NivelModelo:
    if forzar_nivel:
        return forzar_nivel

    palabras = len(prompt.split())

    if PATRONES_COMPLEJOS.search(prompt) or palabras > 500:
        return NivelModelo.POTENTE
    elif PATRONES_SIMPLES.search(prompt) or palabras < 30:
        return NivelModelo.RAPIDO
    else:
        return NivelModelo.ESTANDAR

def generar(
    prompt: str,
    system: str = "",
    forzar_nivel: NivelModelo | None = None,
    max_tokens: int = 1024,
) -> tuple[str, NivelModelo]:
    nivel = seleccionar_modelo(prompt, forzar_nivel)

    kwargs = {
        "model": nivel.value,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text, nivel

# Uso
texto, nivel_usado = generar("Clasifica este email: 'Reunión cancelada para mañana'")
print(f"[{nivel_usado.name}] {texto}")

texto, nivel_usado = generar("Diseña una arquitectura de microservicios para un SaaS de gestión de inventario con 50K usuarios concurrentes")
print(f"[{nivel_usado.name}] {texto[:200]}...")
```

## A/B testing de modelos

Antes de migrar de modelo, haz A/B testing con tráfico real:

```python
import random
import time
from collections import defaultdict
import anthropic

client = anthropic.Anthropic()

class ABTestModelos:
    def __init__(self, modelo_a: str, modelo_b: str, proporcion_b: float = 0.1):
        self.modelo_a = modelo_a
        self.modelo_b = modelo_b
        self.proporcion_b = proporcion_b
        self.metricas = defaultdict(lambda: {"llamadas": 0, "latencia_total": 0.0, "errores": 0})

    def llamar(self, prompt: str, **kwargs) -> tuple[str, str]:
        modelo = self.modelo_b if random.random() < self.proporcion_b else self.modelo_a
        inicio = time.time()
        try:
            response = client.messages.create(
                model=modelo,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            latencia = time.time() - inicio
            self.metricas[modelo]["llamadas"] += 1
            self.metricas[modelo]["latencia_total"] += latencia
            return response.content[0].text, modelo
        except Exception as e:
            self.metricas[modelo]["errores"] += 1
            raise

    def informe(self) -> dict:
        informe = {}
        for modelo, m in self.metricas.items():
            informe[modelo] = {
                "llamadas": m["llamadas"],
                "latencia_media": m["latencia_total"] / m["llamadas"] if m["llamadas"] > 0 else 0,
                "tasa_error": m["errores"] / (m["llamadas"] + m["errores"]) if (m["llamadas"] + m["errores"]) > 0 else 0,
            }
        return informe

# Ejemplo: probar Haiku en el 10% del tráfico antes de migrar desde Sonnet
ab_test = ABTestModelos("claude-sonnet-4-6", "claude-haiku-4-5-20251001", proporcion_b=0.1)
```

## Monitorización de calidad en producción

```python
from langfuse import Langfuse

langfuse = Langfuse()

def registrar_feedback_usuario(trace_id: str, puntuacion: float, comentario: str = ""):
    """Registra feedback del usuario (thumbs up/down, rating) en Langfuse."""
    langfuse.score(
        trace_id=trace_id,
        name="user_satisfaction",
        value=puntuacion,  # 0.0 - 1.0
        comment=comentario,
    )

# Alertas: si la puntuación media baja del umbral, notificar
def verificar_calidad_diaria(umbral_minimo: float = 0.7) -> bool:
    scores = langfuse.get_scores(name="user_satisfaction", limit=100)
    if not scores.data:
        return True
    media = sum(s.value for s in scores.data) / len(scores.data)
    if media < umbral_minimo:
        print(f"ALERTA: Calidad por debajo del umbral ({media:.2f} < {umbral_minimo})")
        return False
    return True
```

## Resumen

- Empieza siempre con **Sonnet 4.6** y optimiza si el coste o la calidad no son suficientes
- Usa **Haiku 4.5** para tareas simples y alto volumen (clasificación, FAQ, extracción)
- Reserva **Opus 4.7** para razonamiento complejo y decisiones de alto impacto
- Implementa A/B testing antes de cambiar de modelo en producción
- Monitoriza calidad y latencia continuamente para detectar regresiones
