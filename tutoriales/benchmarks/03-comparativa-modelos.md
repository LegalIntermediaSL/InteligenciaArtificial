# Comparativa práctica de modelos en 2025: ¿cuál elegir?

## Dimensiones de comparativa

Para comparar modelos de forma rigurosa hay que evaluar múltiples dimensiones:

| Dimensión | Qué mide | Herramienta |
|-----------|----------|-------------|
| Calidad de respuesta | Precisión, coherencia, instrucciones | LLM-as-judge, MMLU |
| Capacidad de código | Generación, debugging, refactoring | HumanEval, SWE-bench |
| Razonamiento | Multi-paso, matemáticas, lógica | GSM8K, MATH |
| Velocidad | Tokens por segundo | Medición directa |
| Coste | $/millón tokens | Tablas de precios |
| Contexto | Máximo tokens de entrada | Documentación oficial |
| Multimodalidad | Visión, documentos | Benchmarks visuales |

## Tabla comparativa de modelos principales (2025)

| Modelo | Proveedor | Contexto | Input $/M | Output $/M | MMLU | HumanEval |
|--------|-----------|----------|-----------|-----------|------|-----------|
| claude-opus-4-7 | Anthropic | 200K | $15 | $75 | ~90% | ~92% |
| claude-sonnet-4-6 | Anthropic | 200K | $3 | $15 | ~85% | ~88% |
| claude-haiku-4-5 | Anthropic | 200K | $0.80 | $4 | ~78% | ~75% |
| gpt-4o | OpenAI | 128K | $2.50 | $10 | ~88% | ~90% |
| gpt-4o-mini | OpenAI | 128K | $0.15 | $0.60 | ~82% | ~82% |
| gemini-1.5-pro | Google | 1M | $3.50 | $10.50 | ~86% | ~85% |
| gemini-1.5-flash | Google | 1M | $0.075 | $0.30 | ~78% | ~74% |
| llama-3.1-70b | Meta (open) | 128K | ~$0.50* | ~$0.75* | ~82% | ~81% |
| mistral-large-2 | Mistral | 128K | $2 | $6 | ~84% | ~83% |

*Precio orientativo en proveedores cloud (Together.ai, Groq, etc.)

## Benchmark automatizado multi-proveedor

```python
import anthropic
import openai
import time
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ResultadoModelo:
    modelo: str
    proveedor: str
    respuestas: list[str] = field(default_factory=list)
    latencias: list[float] = field(default_factory=list)
    errores: int = 0

    @property
    def latencia_media(self) -> float:
        return sum(self.latencias) / len(self.latencias) if self.latencias else 0

PROMPTS_TEST = [
    "Explica qué es el gradiente descendente en 3 frases.",
    "Escribe una función Python que invierta una lista sin usar reverse().",
    "¿Cuánto es 17 × 23? Muestra los pasos.",
    "Resume en 2 frases: 'El cambio climático es el aumento sostenido de la temperatura media global...'",
    "Clasifica como positivo/negativo: 'El producto llegó roto y el servicio al cliente fue pésimo'.",
]

# Cliente Anthropic
anthropic_client = anthropic.Anthropic()

def llamar_anthropic(modelo: str, prompt: str) -> str:
    response = anthropic_client.messages.create(
        model=modelo,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# Cliente OpenAI
openai_client = openai.OpenAI()

def llamar_openai(modelo: str, prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=modelo,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

MODELOS = [
    ("claude-sonnet-4-6", "Anthropic", llamar_anthropic),
    ("claude-haiku-4-5-20251001", "Anthropic", llamar_anthropic),
    ("gpt-4o-mini", "OpenAI", llamar_openai),
]

def benchmark_velocidad() -> list[ResultadoModelo]:
    resultados = []
    for modelo, proveedor, fn in MODELOS:
        resultado = ResultadoModelo(modelo=modelo, proveedor=proveedor)
        for prompt in PROMPTS_TEST:
            try:
                inicio = time.time()
                respuesta = fn(modelo, prompt)
                latencia = time.time() - inicio
                resultado.respuestas.append(respuesta)
                resultado.latencias.append(latencia)
            except Exception as e:
                resultado.errores += 1
                print(f"Error en {modelo}: {e}")
        resultados.append(resultado)

    # Mostrar resultados
    print(f"\n{'Modelo':<30} {'Proveedor':<12} {'Latencia media':<16} {'Errores'}")
    print("-" * 70)
    for r in sorted(resultados, key=lambda x: x.latencia_media):
        print(f"{r.modelo:<30} {r.proveedor:<12} {r.latencia_media:.2f}s{'':<10} {r.errores}")

    return resultados

resultados = benchmark_velocidad()
```

## Casos donde cada modelo gana

### Claude Opus 4.7 gana en:
- Análisis legal y compliance (razonamiento multi-documento)
- Arquitecturas de software complejas
- Investigación y síntesis de múltiples fuentes
- Matemáticas avanzadas con Extended Thinking

### Claude Sonnet 4.6 gana en:
- Computer Use (recomendado por Anthropic)
- Chatbots de producción con tráfico real
- Generación de contenido largo y coherente
- Balance calidad/coste para la mayoría de casos

### GPT-4o gana en:
- Integración con el ecosistema OpenAI (DALL-E, Whisper, etc.)
- Casos donde el cliente ya usa Azure OpenAI

### Gemini 1.5 Pro gana en:
- Contextos extremadamente largos (hasta 1M tokens)
- Integración con Google Workspace
- Vídeo multimodal nativo

### Modelos open-source (Llama 3.1 70B, Mistral Large) ganan en:
- Privacidad absoluta (datos sensibles que no pueden salir de la empresa)
- Coste a escala muy alta (hosting propio con vLLM)
- Fine-tuning sobre datos propios

## Herramientas de comparativa en línea

```python
# OpenRouter: accede a múltiples modelos con una sola API
import requests

def llamar_openrouter(modelo: str, prompt: str, api_key: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": modelo,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return response.json()["choices"][0]["message"]["content"]
```

## Resumen

- No existe un "mejor modelo" universal — depende del caso de uso, coste y velocidad requeridos
- **Regla 80/20**: Sonnet 4.6 resuelve el 80% de casos con el mejor equilibrio
- Para contextos > 200K tokens: Gemini 1.5 Pro
- Para privacidad total: modelos open-source con hosting propio
- Siempre valida con tu propio benchmark antes de decidir
