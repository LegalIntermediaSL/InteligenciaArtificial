# Benchmarks públicos de LLMs: MMLU, HumanEval, GSM8K y LMSYS Arena

## ¿Por qué importan los benchmarks?

Un benchmark es un conjunto de tareas estandarizadas que permiten comparar modelos de forma objetiva. Sin ellos, solo podríamos comparar modelos por intuición subjetiva. Sin embargo, los benchmarks tienen limitaciones importantes que veremos al final del artículo.

## MMLU — Massive Multitask Language Understanding

**Qué mide:** conocimiento general en 57 materias (matemáticas, historia, derecho, medicina, etc.)  
**Formato:** preguntas de opción múltiple (4 opciones)  
**Métrica:** accuracy (% de respuestas correctas)  
**Referencia humana:** ~89% (expertos en su campo)

### Puntuaciones actuales (2025)

| Modelo | MMLU |
|--------|------|
| Claude Opus 4.7 | ~90% |
| GPT-4o | ~88% |
| Claude Sonnet 4.6 | ~85% |
| Gemini 1.5 Pro | ~86% |
| Llama 3.1 70B | ~82% |
| Claude Haiku 4.5 | ~78% |

### Interpretación

Un modelo con 90% en MMLU no "sabe de todo" — memoriza patrones de texto de su dataset de entrenamiento. Temas subrepresentados en el entrenamiento pueden dar resultados mucho peores.

## HumanEval — Benchmark de código

**Qué mide:** capacidad para escribir código Python correcto  
**Formato:** descripción de función + tests unitarios; el modelo debe implementar la función  
**Métrica:** pass@1 (% de problemas resueltos en el primer intento)  
**Creado por:** OpenAI, 164 problemas

### Puntuaciones actuales

| Modelo | HumanEval pass@1 |
|--------|-----------------|
| Claude Opus 4.7 | ~92% |
| GPT-4o | ~90% |
| Claude Sonnet 4.6 | ~88% |
| Gemini 1.5 Pro | ~85% |
| Llama 3.1 70B | ~81% |

### Cómo ejecutar HumanEval tú mismo

```python
# pip install human-eval
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
import anthropic

client = anthropic.Anthropic()
problems = read_problems()

def generar_solucion(prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"Completa la siguiente función Python sin añadir texto extra:\n\n{prompt}"
        }],
    )
    return response.content[0].text

# Generar soluciones para los primeros 10 problemas
soluciones = []
for task_id, problem in list(problems.items())[:10]:
    solucion = generar_solucion(problem["prompt"])
    soluciones.append({"task_id": task_id, "completion": solucion})

write_jsonl("soluciones.jsonl", soluciones)
resultados = evaluate_functional_correctness("soluciones.jsonl")
print(f"pass@1: {resultados['pass@1']:.2%}")
```

## GSM8K — Razonamiento matemático

**Qué mide:** resolución de problemas matemáticos de nivel primaria/secundaria  
**Formato:** problema en lenguaje natural → solución paso a paso  
**Métrica:** accuracy en la respuesta numérica final  
**Por qué importa:** indica capacidad de razonamiento multi-paso, no solo memoria

| Modelo | GSM8K |
|--------|-------|
| Claude Opus 4.7 | ~97% |
| GPT-4o | ~95% |
| Claude Sonnet 4.6 | ~93% |
| Llama 3.1 70B | ~91% |

## LMSYS Chatbot Arena

El benchmark más diferente: en lugar de tareas automáticas, usa **votaciones humanas**.

**Cómo funciona:**
1. Un usuario escribe un prompt
2. Dos modelos responden (anónimos)
3. El usuario vota cuál respuesta prefiere
4. Se calcula un ranking tipo Elo (como en ajedrez)

**Por qué es valioso:** captura preferencias humanas reales, no solo capacidades técnicas. Un modelo puede tener MMLU alto pero respuestas que los humanos encuentran robóticas.

**Limitaciones:** sesgado hacia prompts en inglés, hacia usuarios técnicos, y hacia respuestas más largas ("verbosity bias").

### Ver el ranking actual

```python
import requests
import pandas as pd

# API pública de LMSYS Arena
url = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
# También disponible en: https://lmarena.ai/leaderboard
print("Visita https://lmarena.ai/leaderboard para el ranking actualizado")
```

## Limitaciones críticas de los benchmarks

1. **Contaminación del dataset de entrenamiento** — si un modelo "ha visto" las respuestas correctas durante el entrenamiento, su puntuación no refleja capacidad real.

2. **Saturación** — cuando los mejores modelos alcanzan 95%+ en un benchmark, deja de diferenciarlos. Hay que buscar benchmarks más difíciles.

3. **Generalización limitada** — un modelo puede tener 95% en MMLU y ser terrible en tu caso de uso específico.

4. **Benchmarks en inglés** — la mayoría de benchmarks están en inglés. Los resultados pueden ser muy distintos en español u otros idiomas.

5. **Gaming** — los laboratorios pueden optimizar específicamente para benchmarks conocidos, inflando artificialmente las puntuaciones.

## La regla de oro

> **Un benchmark público es solo el punto de partida. El benchmark que importa es el que mides tú sobre tu caso de uso real.**

En el siguiente artículo veremos cómo construir tu propia evaluación.

## Resumen

- **MMLU**: conocimiento general, 57 materias, opción múltiple
- **HumanEval**: código Python, pass@1
- **GSM8K**: matemáticas multi-paso
- **LMSYS Arena**: votaciones humanas, ranking Elo
- Los benchmarks públicos son útiles para comparar modelos, pero no reemplazan la evaluación en tu dominio específico
