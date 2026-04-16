# 03 — Evaluación de Modelos Fine-tuneados

> **Bloque:** Fine-tuning avanzado · **Nivel:** Avanzado · **Tiempo estimado:** 50 min

---

## Índice

1. [Por qué la evaluación es difícil en LLMs](#1-por-qué-la-evaluación-es-difícil-en-llms)
2. [Métricas automáticas: ROUGE, BLEU y perplexity](#2-métricas-automáticas-rouge-bleu-y-perplexity)
3. [LLM-as-judge con Claude y GPT-4](#3-llm-as-judge-con-claude-y-gpt-4)
4. [Benchmarks estándar: MMLU, HellaSwag, TruthfulQA](#4-benchmarks-estándar-mmlu-hellaswag-truthfulqa)
5. [Evaluación con lm-evaluation-harness](#5-evaluación-con-lm-evaluation-harness)
6. [Comparativa antes/después del fine-tuning](#6-comparativa-antesdespués-del-fine-tuning)
7. [Dashboard de métricas con Weights & Biases](#7-dashboard-de-métricas-con-weights--biases)
8. [Extensiones sugeridas](#8-extensiones-sugeridas)

---

## 1. Por qué la evaluación es difícil en LLMs

La evaluación de modelos de lenguaje es uno de los problemas más complejos del campo. A diferencia de la clasificación (accuracy, F1) o la regresión (MSE, MAE), los LLMs generan texto libre donde múltiples respuestas pueden ser igualmente correctas, útiles o apropiadas.

Considera la instrucción: "Explica qué es un decorador en Python." Estas tres respuestas son todas correctas:
- Una explicación de tres párrafos con ejemplos.
- Un ejemplo de código comentado sin explicación textual.
- Una analogía seguida de código.

¿Cuál es la mejor? Depende del contexto, el usuario, y el criterio de calidad que priorices.

### El triángulo de evaluación

Una evaluación robusta de LLMs combina tres enfoques complementarios:

```
         MÉTRICAS AUTOMÁTICAS
         (ROUGE, BLEU, perplexity)
         ┌────────────────────┐
         │  Rápidas, baratas  │
         │  Bajo correlación  │
         │  con calidad real  │
         └────────────────────┘
               /          \
              /            \
    LLM-AS-JUDGE       BENCHMARKS
    (Claude, GPT-4)    (MMLU, HellaSwag)
    ┌──────────────┐   ┌──────────────┐
    │ Alta calidad │   │ Reproducible │
    │ Costoso, lento│   │ Estándar     │
    │ Sesgo del juez│   │ No mide todo │
    └──────────────┘   └──────────────┘
```

---

## 2. Métricas automáticas: ROUGE, BLEU y perplexity

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE mide el solapamiento de n-gramas entre la respuesta generada y una referencia. Es común en evaluación de resúmenes.

```python
from evaluate import load
import numpy as np

# Cargar la métrica ROUGE
rouge = load("rouge")

# Ejemplos: respuestas generadas y referencias
predicciones = [
    "El aprendizaje supervisado usa datos etiquetados para entrenar modelos predictivos.",
    "Python es un lenguaje de programación interpretado de alto nivel muy popular en IA.",
    "La red neuronal aprende ajustando sus pesos mediante el algoritmo de backpropagation.",
]

referencias = [
    "En el aprendizaje supervisado, el modelo se entrena con ejemplos etiquetados.",
    "Python es un lenguaje interpretado, de alto nivel, ampliamente utilizado en ciencia de datos.",
    "Las redes neuronales ajustan sus parámetros usando backpropagation para minimizar el error.",
]

resultados_rouge = rouge.compute(
    predictions=predicciones,
    references=referencias,
    use_stemmer=False,      # True para inglés (normaliza variantes de palabras)
    use_aggregator=True,    # True = media sobre todos los ejemplos
)

print("Métricas ROUGE:")
for metrica, valor in resultados_rouge.items():
    print(f"  {metrica}: {valor:.4f}")

# ROUGE-1: solapamiento de unigramas (palabras individuales)
# ROUGE-2: solapamiento de bigramas (pares de palabras consecutivas)
# ROUGE-L: secuencia común más larga (captura orden)
# ROUGE-Lsum: variante de ROUGE-L para resúmenes (divide por oraciones)

# Valores de referencia para resúmenes en inglés:
# ROUGE-1 > 0.40 = bueno
# ROUGE-2 > 0.18 = bueno
# ROUGE-L > 0.35 = bueno
# Nota: para español los valores suelen ser más bajos
```

### BLEU (Bilingual Evaluation Understudy)

BLEU se diseñó para traducción automática y mide precisión de n-gramas con penalización por brevedad:

```python
from evaluate import load

bleu = load("bleu")

# BLEU espera las referencias como lista de listas (puede haber múltiples refs por predicción)
predicciones_traduccion = [
    "The model was trained for 3 days on 8 GPUs.",
    "The results show a significant improvement in accuracy.",
]

referencias_traduccion = [
    ["The model was trained during 3 days on 8 GPUs."],          # Solo una referencia aquí
    ["The results demonstrate a significant accuracy improvement."],
]

resultado_bleu = bleu.compute(
    predictions=predicciones_traduccion,
    references=referencias_traduccion,
)

print(f"BLEU score: {resultado_bleu['bleu']:.4f}")
print(f"Precisiones por n-grama: {resultado_bleu['precisions']}")
# BLEU < 0.30 = pobre
# BLEU 0.30-0.50 = aceptable
# BLEU > 0.50 = bueno (para traducción automática)

# Limitaciones importantes:
# - Requiere referencias de alta calidad
# - No captura calidad semántica
# - Penaliza paráfrasis correctas
# - Correlación con calidad humana moderada (~0.5-0.7)
```

### Perplexity

La perplexity mide la "sorpresa" del modelo ante el texto: a menor perplexity, el modelo predice mejor el texto. Se usa para medir si el modelo ha perdido fluidez general tras el fine-tuning (fenómeno llamado **catastrophic forgetting**):

```python
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def calcular_perplexity(
    ruta_modelo: str,
    textos: list[str],
    max_longitud: int = 512,
    stride: int = 256,
) -> float:
    """
    Calcula la perplexity del modelo sobre una lista de textos.

    Usa el método de ventana deslizante (stride) para manejar textos
    más largos que el contexto del modelo correctamente.

    Args:
        ruta_modelo: Ruta o nombre del modelo HuggingFace.
        textos: Lista de textos para evaluar.
        max_longitud: Longitud máxima de la ventana de evaluación.
        stride: Paso de la ventana deslizante (solapamiento).

    Returns:
        Perplexity media sobre todos los textos.
    """
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    modelo.eval()

    nlls_totales = []   # Negative log-likelihoods acumuladas
    tokens_totales = 0

    for texto in tqdm(textos, desc="Calculando perplexity"):
        input_ids = tokenizer.encode(texto, return_tensors="pt").to(modelo.device)
        n_tokens = input_ids.shape[1]

        for inicio in range(0, n_tokens, stride):
            fin = min(inicio + max_longitud, n_tokens)
            segmento = input_ids[:, inicio:fin]

            # Los primeros (max_longitud - stride) tokens solo se usan como contexto
            # Los últimos stride tokens son los que se evalúan
            inicio_objetivo = max(0, max_longitud - stride)
            n_tokens_objetivo = segmento.shape[1] - inicio_objetivo

            if n_tokens_objetivo <= 0:
                continue

            with torch.no_grad():
                outputs = modelo(segmento, labels=segmento)

            # La pérdida de CrossEntropy es la NLL media sobre todos los tokens del segmento
            # Necesitamos la NLL solo de los tokens objetivo
            logits = outputs.logits[:, :-1, :]                          # shift por predicción
            labels = segmento[:, 1:]                                    # shift de labels
            logits_obj = logits[:, inicio_objetivo:, :]
            labels_obj = labels[:, inicio_objetivo:]

            perdida = torch.nn.functional.cross_entropy(
                logits_obj.reshape(-1, logits_obj.shape[-1]),
                labels_obj.reshape(-1),
                reduction="sum",
            )

            nlls_totales.append(perdida.item())
            tokens_totales += labels_obj.numel()

    nll_media = sum(nlls_totales) / tokens_totales
    perplexity = math.exp(nll_media)
    return perplexity


# Evaluar perplexity en texto de dominio general (Wikitext)
dataset_wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
textos_wiki = [t for t in dataset_wiki["text"] if len(t) > 100][:50]

# Comparar modelo base vs modelo fine-tuneado
print("Calculando perplexity del modelo base...")
ppl_base = calcular_perplexity("mistralai/Mistral-7B-v0.1", textos_wiki)

print("Calculando perplexity del modelo fine-tuneado...")
ppl_finetuned = calcular_perplexity("./mistral-7b-sft-data-science/final", textos_wiki)

print(f"\nPerplexity en Wikitext-2:")
print(f"  Modelo base:        {ppl_base:.2f}")
print(f"  Modelo fine-tuneado: {ppl_finetuned:.2f}")
print(f"  Cambio:             {(ppl_finetuned / ppl_base - 1) * 100:+.1f}%")
# Si la perplexity sube mucho (>20%) en texto general, el modelo sufrió catastrophic forgetting
```

---

## 3. LLM-as-judge con Claude y GPT-4

La evaluación con LLM-as-judge es actualmente el método más fiable para evaluar calidad de texto libre, especialmente para instrucciones abiertas.

### Evaluación de respuesta única (single-answer grading)

```python
import anthropic
import json
from dataclasses import dataclass


@dataclass
class ResultadoEvaluacion:
    puntuacion: int          # 1-10
    correcto: bool
    justificacion: str
    problemas: list[str]
    puntos_fuertes: list[str]


def evaluar_respuesta_claude(
    instruccion: str,
    respuesta: str,
    rubrica: str | None = None,
) -> ResultadoEvaluacion:
    """
    Evalúa una respuesta usando Claude como juez.

    Args:
        instruccion: La instrucción o pregunta original.
        respuesta: La respuesta generada por el modelo evaluado.
        rubrica: Criterios específicos de evaluación (opcional).
    """
    cliente = anthropic.Anthropic()

    rubrica_texto = rubrica or (
        "- Corrección técnica y factual\n"
        "- Completitud (aborda todos los aspectos de la instrucción)\n"
        "- Claridad y estructura\n"
        "- Seguimiento preciso de la instrucción\n"
        "- Ausencia de alucinaciones o información incorrecta"
    )

    system_prompt = """Eres un evaluador experto de modelos de lenguaje. Tu tarea es evaluar 
la calidad de las respuestas de forma objetiva e imparcial, basándote en la instrucción dada 
y los criterios de evaluación proporcionados.

IMPORTANTE: Debes responder SOLO con el JSON solicitado, sin texto adicional."""

    prompt = f"""Evalúa la siguiente respuesta.

INSTRUCCIÓN ORIGINAL:
{instruccion}

RESPUESTA EVALUADA:
{respuesta}

CRITERIOS DE EVALUACIÓN:
{rubrica_texto}

Devuelve EXACTAMENTE este JSON (sin texto fuera del JSON):
{{
  "puntuacion": <entero del 1 al 10>,
  "correcto": <true si la respuesta es técnicamente correcta, false si tiene errores graves>,
  "justificacion": "Evaluación concisa en 2-3 oraciones",
  "puntos_fuertes": ["lista de lo que hace bien"],
  "problemas": ["lista de problemas o mejoras necesarias, vacía si no hay"]
}}"""

    respuesta_claude = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    texto = respuesta_claude.content[0].text.strip()

    # Limpiar el JSON si viene envuelto en bloques de código
    if "```" in texto:
        texto = texto.split("```")[1] if texto.startswith("```") else texto.split("```json\n")[1].split("```")[0]

    datos = json.loads(texto.strip())

    return ResultadoEvaluacion(
        puntuacion=datos["puntuacion"],
        correcto=datos["correcto"],
        justificacion=datos["justificacion"],
        problemas=datos.get("problemas", []),
        puntos_fuertes=datos.get("puntos_fuertes", []),
    )
```

### Evaluación comparativa (pairwise comparison)

```python
def comparar_respuestas_pairwise(
    instruccion: str,
    respuesta_a: str,
    respuesta_b: str,
    nombre_a: str = "Modelo A",
    nombre_b: str = "Modelo B",
) -> dict:
    """
    Compara dos respuestas y determina cuál es mejor.
    Usa evaluación doble (A vs B y B vs A) para mitigar el sesgo de posición.
    """
    cliente = anthropic.Anthropic()

    def comparacion_directa(resp_1: str, resp_2: str, label_1: str, label_2: str) -> dict:
        prompt = f"""Compara estas dos respuestas a la instrucción dada y determina cuál es mejor.

INSTRUCCIÓN: {instruccion}

RESPUESTA 1 ({label_1}):
{resp_1}

RESPUESTA 2 ({label_2}):
{resp_2}

Devuelve EXACTAMENTE este JSON:
{{
  "ganador": "1" o "2" o "empate",
  "margen": "claro" o "leve" o "empate",
  "razon": "Explicación en 1-2 oraciones de por qué un modelo es mejor"
}}"""

        respuesta = cliente.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        texto = respuesta.content[0].text.strip()
        if "```" in texto:
            texto = texto.split("```json\n")[1].split("```")[0] if "```json" in texto else texto.split("```")[1].split("```")[0]

        return json.loads(texto.strip())

    # Primera pasada: A vs B
    resultado_ab = comparacion_directa(respuesta_a, respuesta_b, nombre_a, nombre_b)

    # Segunda pasada: B vs A (para detectar sesgo de posición)
    resultado_ba = comparacion_directa(respuesta_b, respuesta_a, nombre_b, nombre_a)

    # Reconciliar resultados (invertir el ganador de la segunda pasada)
    mapa_inversion = {"1": "2", "2": "1", "empate": "empate"}
    ganador_ba_invertido = mapa_inversion[resultado_ba["ganador"]]

    # Si ambas pasadas coinciden → resultado confiable
    if resultado_ab["ganador"] == ganador_ba_invertido:
        ganador_final = resultado_ab["ganador"]
        confianza = "alta"
    else:
        # Discrepancia → posible sesgo de posición, considerar empate
        ganador_final = "empate"
        confianza = "baja"

    # Mapear número a nombre
    mapa_nombres = {"1": nombre_a, "2": nombre_b, "empate": "empate"}
    ganador_nombre = mapa_nombres[ganador_final]

    return {
        "ganador": ganador_nombre,
        "confianza": confianza,
        "razon_ab": resultado_ab["razon"],
        "razon_ba": resultado_ba["razon"],
    }


# Ejemplo de uso
instruccion = "¿Cuál es la complejidad temporal del algoritmo QuickSort y por qué?"

respuesta_base = "QuickSort tiene complejidad O(n log n). Es un algoritmo de ordenación."

respuesta_finetuned = (
    "QuickSort tiene una complejidad temporal de **O(n log n) en el caso promedio** y "
    "**O(n²) en el peor caso**.\n\n"
    "**Por qué O(n log n) en promedio:**\n"
    "- En cada llamada recursiva, el pivote divide el array en dos mitades aproximadamente iguales.\n"
    "- La profundidad de la recursión es O(log n) (número de divisiones hasta llegar a arrays de tamaño 1).\n"
    "- En cada nivel de la recursión, se hacen O(n) comparaciones en total.\n"
    "- Resultado: O(n) × O(log n) = O(n log n).\n\n"
    "**Por qué O(n²) en el peor caso:**\n"
    "- Si el pivote siempre es el elemento más pequeño o más grande (array ya ordenado + pivote en extremo),\n"
    "  la recursión tiene profundidad O(n) en lugar de O(log n).\n"
    "- Solución: usar pivote aleatorio o la mediana de tres elementos."
)

resultado = comparar_respuestas_pairwise(
    instruccion=instruccion,
    respuesta_a=respuesta_base,
    respuesta_b=respuesta_finetuned,
    nombre_a="Modelo base",
    nombre_b="Modelo fine-tuneado",
)

print(f"Ganador: {resultado['ganador']} (confianza: {resultado['confianza']})")
print(f"Razón: {resultado['razon_ab']}")
```

---

## 4. Benchmarks estándar: MMLU, HellaSwag, TruthfulQA

### MMLU (Massive Multitask Language Understanding)

MMLU evalúa conocimiento en 57 materias académicas (matemáticas, biología, derecho, historia, etc.) mediante preguntas de opción múltiple (4 opciones). Es el benchmark más usado para medir el "conocimiento general" del modelo.

```python
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluar_mmlu_subset(
    ruta_modelo: str,
    subjects: list[str] | None = None,
    n_ejemplos_por_materia: int = 50,
) -> dict:
    """
    Evalúa el modelo en un subconjunto de MMLU.

    Args:
        ruta_modelo: Ruta o nombre del modelo.
        subjects: Lista de materias de MMLU a evaluar. None = todas.
        n_ejemplos_por_materia: Cuántos ejemplos evaluar por materia.

    Returns:
        Diccionario con accuracy por materia y accuracy global.
    """
    if subjects is None:
        subjects = ["high_school_mathematics", "high_school_computer_science",
                    "college_mathematics", "abstract_algebra"]

    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    modelo.eval()

    opciones = ["A", "B", "C", "D"]
    resultados_por_materia = {}

    for materia in subjects:
        dataset_materia = load_dataset(
            "cais/mmlu",
            materia,
            split=f"test[:{n_ejemplos_por_materia}]",
        )

        correctos = 0
        total = len(dataset_materia)

        for ejemplo in dataset_materia:
            pregunta = ejemplo["question"]
            choices = ejemplo["choices"]
            respuesta_correcta_idx = ejemplo["answer"]  # 0-3

            # Construir el prompt de few-shot (0-shot aquí para simplicidad)
            prompt = (
                f"Question: {pregunta}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                f"Answer:"
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(modelo.device)

            # Calcular log-probabilidad de cada opción
            log_probs_opciones = []
            with torch.no_grad():
                for opcion in opciones:
                    opcion_ids = tokenizer.encode(f" {opcion}", add_special_tokens=False)
                    # Añadir la opción al final del prompt
                    ids_completos = torch.cat([
                        input_ids,
                        torch.tensor([opcion_ids]).to(modelo.device)
                    ], dim=1)

                    outputs = modelo(ids_completos)
                    logits = outputs.logits

                    # Log-prob del token de la opción (último token añadido)
                    log_prob = torch.nn.functional.log_softmax(
                        logits[0, -len(opcion_ids) - 1, :], dim=-1
                    )[opcion_ids[0]].item()

                    log_probs_opciones.append(log_prob)

            # Predecir la opción con mayor log-probabilidad
            prediccion = log_probs_opciones.index(max(log_probs_opciones))

            if prediccion == respuesta_correcta_idx:
                correctos += 1

        accuracy = correctos / total
        resultados_por_materia[materia] = {
            "accuracy": accuracy,
            "correctos": correctos,
            "total": total,
        }
        print(f"  {materia}: {accuracy:.1%} ({correctos}/{total})")

    # Accuracy global
    total_correctos = sum(r["correctos"] for r in resultados_por_materia.values())
    total_ejemplos = sum(r["total"] for r in resultados_por_materia.values())
    accuracy_global = total_correctos / total_ejemplos

    return {
        "por_materia": resultados_por_materia,
        "accuracy_global": accuracy_global,
        "total_correctos": total_correctos,
        "total_ejemplos": total_ejemplos,
    }
```

### TruthfulQA

TruthfulQA mide si el modelo genera respuestas veraces o reproduce mitos y creencias falsas comunes:

```python
from datasets import load_dataset


def evaluar_truthfulqa_generation(
    ruta_modelo: str,
    n_ejemplos: int = 50,
) -> None:
    """
    Evalúa el modelo en TruthfulQA en modo de generación libre.
    Requiere un juez (LLM o humano) para evaluar truthfulness.
    """
    dataset = load_dataset("truthful_qa", "generation", split=f"validation[:{n_ejemplos}]")

    # TruthfulQA tiene preguntas diseñadas para activar respuestas comúnmente incorrectas
    # Ejemplo de pregunta que los humanos suelen responder mal:
    # "¿Cuántos huesos tienen los humanos al nacer?"
    # (respuesta incorrecta común: 206; correcta: ~270-300)

    print("Preguntas de TruthfulQA (muestra):")
    for i, ejemplo in enumerate(dataset.select(range(5))):
        print(f"\n[{i+1}] {ejemplo['question']}")
        print(f"  Categoría: {ejemplo['category']}")
        print(f"  Respuestas correctas: {ejemplo['correct_answers'][:2]}")
        print(f"  Respuestas incorrectas frecuentes: {ejemplo['incorrect_answers'][:2]}")

    # Para evaluar el modelo en TruthfulQA de forma automática,
    # se recomienda usar lm-evaluation-harness (sección 5 de este tutorial)
```

---

## 5. Evaluación con lm-evaluation-harness

`lm-evaluation-harness` de EleutherAI es la herramienta estándar de la industria para evaluar LLMs en benchmarks. Soporta más de 60 benchmarks con una sola línea de comando.

### Instalación

```bash
pip install lm-eval
```

### Evaluación desde línea de comandos

```bash
# Evaluar en MMLU (todas las materias) con few-shot de 5 ejemplos
lm_eval \
    --model hf \
    --model_args pretrained=./mistral-7b-sft-final,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8 \
    --output_path ./resultados_mmlu.json \
    --log_samples

# Evaluar en múltiples benchmarks simultáneamente
lm_eval \
    --model hf \
    --model_args pretrained=./mistral-7b-sft-final,dtype=bfloat16 \
    --tasks mmlu,hellaswag,truthfulqa_mc1,arc_easy,arc_challenge \
    --num_fewshot 0 \
    --batch_size 4 \
    --output_path ./resultados_completos.json

# Evaluar con cuantización para modelos grandes
lm_eval \
    --model hf \
    --model_args "pretrained=./mistral-7b-sft-final,dtype=bfloat16,load_in_4bit=True" \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size 2
```

### Evaluación desde Python

```python
import lm_eval
from lm_eval.models.huggingface import HFLM
import json


def evaluar_con_harness(
    ruta_modelo: str,
    tareas: list[str],
    num_fewshot: int = 0,
    batch_size: int = 8,
    limite: int | None = None,
) -> dict:
    """
    Evalúa un modelo usando lm-evaluation-harness.

    Args:
        ruta_modelo: Ruta local o nombre HuggingFace del modelo.
        tareas: Lista de benchmarks (ej: ['mmlu', 'hellaswag']).
        num_fewshot: Número de ejemplos en el contexto (0 = zero-shot).
        batch_size: Tamaño de batch para evaluación.
        limite: Limitar evaluación a N ejemplos por tarea (para testing rápido).

    Returns:
        Diccionario con resultados por tarea.
    """
    # Cargar el modelo en el formato que espera lm-eval
    modelo_lm = HFLM(
        pretrained=ruta_modelo,
        dtype="bfloat16",
        batch_size=batch_size,
    )

    # Ejecutar evaluación
    resultados = lm_eval.simple_evaluate(
        model=modelo_lm,
        tasks=tareas,
        num_fewshot=num_fewshot,
        limit=limite,                    # None = evaluar todos los ejemplos
        log_samples=False,
    )

    # Extraer métricas principales
    metricas = {}
    for tarea in tareas:
        if tarea in resultados["results"]:
            res = resultados["results"][tarea]
            # La métrica principal varía por tarea
            # MMLU/ARC/HellaSwag: acc_norm
            # TruthfulQA: mc1/mc2
            for metrica in ["acc_norm,none", "acc,none", "mc1,none", "exact_match,none"]:
                if metrica in res:
                    metricas[tarea] = res[metrica]
                    break

    return {
        "metricas": metricas,
        "resultados_completos": resultados["results"],
    }


# Evaluar y comparar dos modelos
tareas_evaluacion = ["mmlu", "hellaswag", "truthfulqa_mc1", "arc_challenge"]

print("Evaluando modelo base...")
res_base = evaluar_con_harness(
    "mistralai/Mistral-7B-v0.1",
    tareas=tareas_evaluacion,
    num_fewshot=5,
    limite=100,   # Solo 100 ejemplos por tarea para testing
)

print("\nEvaluando modelo fine-tuneado...")
res_ft = evaluar_con_harness(
    "./mistral-7b-sft-final",
    tareas=tareas_evaluacion,
    num_fewshot=5,
    limite=100,
)

# Comparar
print("\n" + "="*60)
print(f"{'Benchmark':<25} {'Base':>10} {'Fine-tuned':>12} {'Cambio':>8}")
print("-"*60)
for tarea in tareas_evaluacion:
    base = res_base["metricas"].get(tarea, float("nan"))
    ft = res_ft["metricas"].get(tarea, float("nan"))
    cambio = ft - base if not (isinstance(base, float) and isinstance(ft, float) and
                                (base != base or ft != ft)) else float("nan")
    print(f"{tarea:<25} {base:>10.1%} {ft:>12.1%} {cambio:>+8.1%}")
```

---

## 6. Comparativa antes/después del fine-tuning

### Pipeline completo de comparación

```python
import json
from pathlib import Path
from datetime import datetime


def ejecutar_evaluacion_completa(
    modelo_base: str,
    modelo_finetuned: str,
    instrucciones_test: list[dict],
    nombre_experimento: str,
) -> dict:
    """
    Pipeline completo de evaluación: métricas automáticas + LLM-as-judge.
    Guarda los resultados en un archivo JSON con timestamp.
    """
    from evaluate import load
    import anthropic
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rouge = load("rouge")
    cliente_anthropic = anthropic.Anthropic()

    def generar_respuestas_modelo(ruta_modelo: str, instrucciones: list[dict]) -> list[str]:
        tok = AutoTokenizer.from_pretrained(ruta_modelo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(
            ruta_modelo, torch_dtype=torch.bfloat16, device_map="auto"
        )
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok,
                        max_new_tokens=256, do_sample=False)

        respuestas = []
        for item in instrucciones:
            msgs = [{"role": "user", "content": item["instruccion"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            out = pipe(prompt)[0]["generated_text"]
            respuesta = out[len(prompt):].strip()
            respuestas.append(respuesta)
        return respuestas

    # Generar respuestas
    print(f"Generando respuestas con modelo base: {modelo_base}")
    respuestas_base = generar_respuestas_modelo(modelo_base, instrucciones_test)

    print(f"Generando respuestas con modelo fine-tuneado: {modelo_finetuned}")
    respuestas_ft = generar_respuestas_modelo(modelo_finetuned, instrucciones_test)

    # Métricas ROUGE (contra respuestas de referencia si existen)
    referencias = [item.get("respuesta_referencia", item["instruccion"])
                   for item in instrucciones_test]

    rouge_base = rouge.compute(predictions=respuestas_base, references=referencias,
                                use_aggregator=True)
    rouge_ft = rouge.compute(predictions=respuestas_ft, references=referencias,
                              use_aggregator=True)

    # LLM-as-judge (comparativo)
    victorias_base = 0
    victorias_ft = 0
    empates = 0

    for i, item in enumerate(instrucciones_test):
        resultado = comparar_respuestas_pairwise(
            instruccion=item["instruccion"],
            respuesta_a=respuestas_base[i],
            respuesta_b=respuestas_ft[i],
            nombre_a="Base",
            nombre_b="Fine-tuneado",
        )
        if resultado["ganador"] == "Base":
            victorias_base += 1
        elif resultado["ganador"] == "Fine-tuneado":
            victorias_ft += 1
        else:
            empates += 1

    n = len(instrucciones_test)
    resultados_finales = {
        "experimento": nombre_experimento,
        "timestamp": timestamp,
        "modelo_base": modelo_base,
        "modelo_finetuned": modelo_finetuned,
        "n_ejemplos": n,
        "metricas_rouge": {
            "base": rouge_base,
            "finetuned": rouge_ft,
            "delta_rouge1": rouge_ft["rouge1"] - rouge_base["rouge1"],
            "delta_rouge2": rouge_ft["rouge2"] - rouge_base["rouge2"],
        },
        "llm_judge_pairwise": {
            "victorias_base": victorias_base,
            "victorias_finetuned": victorias_ft,
            "empates": empates,
            "win_rate_finetuned": victorias_ft / n,
        },
    }

    # Guardar resultados
    ruta_salida = Path(f"./evaluacion_{nombre_experimento}_{timestamp}.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados_finales, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en {ruta_salida}")
    print(f"\nRESUMEN:")
    print(f"  ROUGE-1: base={rouge_base['rouge1']:.3f} | ft={rouge_ft['rouge1']:.3f} "
          f"(Δ{rouge_ft['rouge1'] - rouge_base['rouge1']:+.3f})")
    print(f"  Win rate fine-tuneado: {victorias_ft}/{n} ({victorias_ft/n:.0%})")

    return resultados_finales
```

---

## 7. Dashboard de métricas con Weights & Biases

```python
import wandb
import json
from pathlib import Path


def crear_dashboard_evaluacion(
    resultados: dict,
    nombre_proyecto: str = "llm-finetuning-eval",
) -> str:
    """
    Crea un run en W&B con todas las métricas de evaluación.
    Returns la URL del dashboard.
    """
    run = wandb.init(
        project=nombre_proyecto,
        name=f"eval_{resultados['experimento']}_{resultados['timestamp']}",
        config={
            "modelo_base": resultados["modelo_base"],
            "modelo_finetuned": resultados["modelo_finetuned"],
            "n_ejemplos": resultados["n_ejemplos"],
        },
        tags=["evaluation", "comparison"],
    )

    # Loggear métricas escalares
    rouge = resultados["metricas_rouge"]
    judge = resultados["llm_judge_pairwise"]

    wandb.log({
        "rouge/base_rouge1": rouge["base"]["rouge1"],
        "rouge/base_rouge2": rouge["base"]["rouge2"],
        "rouge/base_rougeL": rouge["base"]["rougeL"],
        "rouge/ft_rouge1": rouge["finetuned"]["rouge1"],
        "rouge/ft_rouge2": rouge["finetuned"]["rouge2"],
        "rouge/ft_rougeL": rouge["finetuned"]["rougeL"],
        "rouge/delta_rouge1": rouge["delta_rouge1"],
        "rouge/delta_rouge2": rouge["delta_rouge2"],
        "judge/win_rate_finetuned": judge["win_rate_finetuned"],
        "judge/victorias_finetuned": judge["victorias_finetuned"],
        "judge/victorias_base": judge["victorias_base"],
        "judge/empates": judge["empates"],
    })

    # Crear tabla comparativa
    tabla = wandb.Table(columns=["Métrica", "Modelo Base", "Fine-tuneado", "Delta"])
    metricas_tabla = [
        ("ROUGE-1", rouge["base"]["rouge1"], rouge["finetuned"]["rouge1"],
         rouge["delta_rouge1"]),
        ("ROUGE-2", rouge["base"]["rouge2"], rouge["finetuned"]["rouge2"],
         rouge["delta_rouge2"]),
        ("Win Rate (Judge)", judge["victorias_base"] / resultados["n_ejemplos"],
         judge["win_rate_finetuned"],
         judge["win_rate_finetuned"] - judge["victorias_base"] / resultados["n_ejemplos"]),
    ]

    for nombre, base, ft, delta in metricas_tabla:
        tabla.add_data(nombre, f"{base:.3f}", f"{ft:.3f}", f"{delta:+.3f}")

    wandb.log({"tabla_comparativa": tabla})

    # Gráfico de barras para win rate
    datos_winrate = [[k, v] for k, v in {
        "Fine-tuneado gana": judge["victorias_finetuned"],
        "Base gana": judge["victorias_base"],
        "Empate": judge["empates"],
    }.items()]
    tabla_winrate = wandb.Table(data=datos_winrate, columns=["Resultado", "Conteo"])
    wandb.log({
        "win_rate_chart": wandb.plot.bar(tabla_winrate, "Resultado", "Conteo",
                                          title="Pairwise Comparison Results")
    })

    url = run.get_url()
    wandb.finish()

    print(f"Dashboard disponible en: {url}")
    return url


# Usar en el pipeline de evaluación:
# resultados = ejecutar_evaluacion_completa(...)
# url = crear_dashboard_evaluacion(resultados)
```

### Loggear métricas durante el entrenamiento para correlación posterior

```python
import wandb

# Durante el entrenamiento con SFTTrainer o DPOTrainer,
# configurar el logging con callbacks personalizados:

from transformers import TrainerCallback


class EvaluacionCallback(TrainerCallback):
    """
    Callback que evalúa el modelo en un conjunto de prueba
    al final de cada época y loggea los resultados en W&B.
    """

    def __init__(self, instrucciones_test: list[dict], tokenizer, cliente_anthropic):
        self.instrucciones_test = instrucciones_test
        self.tokenizer = tokenizer
        self.cliente = cliente_anthropic

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Ejecutar evaluación al final de cada época."""
        if model is None:
            return

        epoch = state.epoch
        model.eval()

        puntuaciones = []
        for item in self.instrucciones_test[:5]:  # Solo 5 ejemplos para rapidez
            instruccion = item["instruccion"]
            msgs = [{"role": "user", "content": instruccion}]
            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False
                )
            respuesta = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            eval_result = evaluar_respuesta_claude(instruccion, respuesta)
            puntuaciones.append(eval_result.puntuacion)

        media_puntuacion = sum(puntuaciones) / len(puntuaciones)
        wandb.log({
            "eval/llm_judge_score": media_puntuacion,
            "epoch": epoch,
        })

        print(f"\nÉpoca {epoch:.1f} - Puntuación LLM-judge: {media_puntuacion:.1f}/10")
        model.train()
```

---

## 8. Extensiones sugeridas

- **MT-Bench**: benchmark de conversaciones multi-turno de LMSYS, diseñado específicamente para evaluar la capacidad de seguimiento de instrucciones complejas.
- **AlpacaEval**: evalúa win rate contra text-davinci-003 usando GPT-4 como juez. Muy correlacionado con preferencias humanas en la práctica.
- **RAGAS para RAG**: si el modelo fine-tuneado se usa en un pipeline RAG, RAGAS provee métricas específicas (faithfulness, answer relevancy, context precision).
- **Evaluación de seguridad con PromptBench**: biblioteca para evaluar robustez del modelo ante adversarial prompts y jailbreaks.
- **Evaluación de calibración**: medir si el modelo es honesto sobre su incertidumbre (Expected Calibration Error, reliability diagrams).

---

**Anterior:** [02 — Instruction Tuning](./02-instruction-tuning.md) · **Siguiente:** [04 — Despliegue de Modelos Propios](./04-despliegue-modelos-propios.md)
