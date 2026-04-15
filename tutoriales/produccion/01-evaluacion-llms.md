# 01 — Evaluación de LLMs en producción

> **Bloque:** Producción · **Nivel:** Práctico · **Tiempo estimado:** 45 min

---

## Índice

1. [Por qué evaluar outputs de LLMs](#1-por-qué-evaluar-outputs-de-llms)
2. [Métricas automáticas — exactitud, BLEU y ROUGE](#2-métricas-automáticas--exactitud-bleu-y-rouge)
3. [LLM-as-judge — usar Claude para evaluar respuestas](#3-llm-as-judge--usar-claude-para-evaluar-respuestas)
4. [Framework de evaluación con dataset de referencia](#4-framework-de-evaluación-con-dataset-de-referencia)
5. [Evaluación de RAG con ragas](#5-evaluación-de-rag-con-ragas)
6. [Extensiones sugeridas](#6-extensiones-sugeridas)

---

## 1. Por qué evaluar outputs de LLMs

Desplegar un LLM en producción sin un sistema de evaluación es como publicar software sin tests. Los LLMs generan texto libre, y eso hace que la evaluación sea fundamentalmente diferente a otros sistemas de software: no hay un `assert respuesta == "correcto"` que funcione de forma universal.

**Los problemas más comunes sin evaluación:**

- Regresiones silenciosas al cambiar el modelo o el prompt
- Respuestas que parecen correctas pero contienen errores factuales
- Degradación gradual de la calidad que nadie detecta
- Imposibilidad de comparar objetivamente dos versiones del sistema

**Las tres estrategias principales:**

| Estrategia | Cuándo usarla | Costo |
|---|---|---|
| Métricas automáticas (BLEU, ROUGE) | Tareas con respuesta de referencia fija | Bajo |
| LLM-as-judge | Tareas abiertas donde la calidad es subjetiva | Medio |
| Evaluación humana | Decisiones críticas, calibración inicial | Alto |

En producción se combinan las tres: métricas automáticas para el CI/CD, LLM-as-judge para monitoreo continuo y evaluación humana para calibrar periódicamente.

---

## 2. Métricas automáticas — exactitud, BLEU y ROUGE

Las métricas automáticas comparan la respuesta del modelo contra una respuesta de referencia escrita por humanos.

```bash
pip install anthropic nltk rouge-score python-dotenv
```

```python
# evaluacion/metricas_automaticas.py
import re
import math
from collections import Counter
from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# 1. Exactitud exacta (Exact Match)
# ---------------------------------------------------------------------------

def exact_match(prediccion: str, referencia: str) -> float:
    """1.0 si las cadenas son idénticas (normalizado), 0.0 si no."""
    def normalizar(texto: str) -> str:
        texto = texto.lower().strip()
        texto = re.sub(r"[^\w\s]", "", texto)
        texto = re.sub(r"\s+", " ", texto)
        return texto

    return 1.0 if normalizar(prediccion) == normalizar(referencia) else 0.0


# ---------------------------------------------------------------------------
# 2. BLEU — mide n-gramas compartidos (útil para traducción y QA corto)
# ---------------------------------------------------------------------------

def bleu_score(prediccion: str, referencia: str, max_n: int = 4) -> float:
    """BLEU simplificado sin dependencias externas."""
    pred_tokens = prediccion.lower().split()
    ref_tokens = referencia.lower().split()

    if not pred_tokens:
        return 0.0

    # Penalización por brevedad (brevity penalty)
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )

        matches = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    # Media geométrica de las precisiones
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# 3. ROUGE-L — mide la subsecuencia común más larga (útil para resúmenes)
# ---------------------------------------------------------------------------

def rouge_l_score(prediccion: str, referencia: str) -> dict:
    """Retorna precisión, recall y F1 de ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    resultado = scorer.score(referencia, prediccion)
    r = resultado["rougeL"]
    return {
        "precision": round(r.precision, 4),
        "recall": round(r.recall, 4),
        "f1": round(r.fmeasure, 4),
    }


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    referencia = "La capital de Francia es París y tiene más de dos millones de habitantes."
    predicciones = [
        "París es la capital de Francia y alberga a más de dos millones de personas.",  # Buena
        "La capital de Francia es París.",  # Parcial
        "No sé cuál es la capital de Francia.",  # Mala
    ]

    print(f"{'Predicción':<55} {'EM':>4} {'BLEU':>6} {'ROUGE-L F1':>10}")
    print("-" * 80)

    for pred in predicciones:
        em = exact_match(pred, referencia)
        bleu = bleu_score(pred, referencia)
        rouge = rouge_l_score(pred, referencia)
        etiqueta = pred[:52] + "..." if len(pred) > 52 else pred
        print(f"{etiqueta:<55} {em:>4.2f} {bleu:>6.4f} {rouge['f1']:>10.4f}")
```

**Cuándo usar cada métrica:**

- **Exact Match**: preguntas de trivia, extracción de entidades, respuestas sí/no
- **BLEU**: traducción automática, generación de código corto
- **ROUGE-L**: resúmenes, paráfrasis, respuestas largas

---

## 3. LLM-as-judge — usar Claude para evaluar respuestas

Cuando no existe una respuesta de referencia única o la calidad es subjetiva, se usa otro LLM como juez. Claude es especialmente bueno en este rol porque sigue instrucciones de evaluación con precisión.

```bash
pip install anthropic python-dotenv
```

```python
# evaluacion/llm_as_judge.py
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Rúbrica de evaluación estructurada
# ---------------------------------------------------------------------------

RUBRICA = """
Eres un evaluador experto de sistemas de IA. Tu tarea es puntuar una respuesta
generada por un asistente de IA según los siguientes criterios.

Devuelve EXCLUSIVAMENTE un objeto JSON con esta estructura (sin texto adicional):
{
  "precision_factual": <0-10>,
  "relevancia": <0-10>,
  "claridad": <0-10>,
  "completitud": <0-10>,
  "puntuacion_total": <0-10>,
  "justificacion": "<explicación breve en español>"
}

Definición de criterios:
- precision_factual: ¿La respuesta contiene información correcta y sin errores?
- relevancia: ¿La respuesta responde directamente a la pregunta?
- claridad: ¿La respuesta es fácil de entender?
- completitud: ¿La respuesta abarca todos los aspectos importantes de la pregunta?
- puntuacion_total: Valoración global (no tiene que ser la media exacta).
"""


def evaluar_respuesta(
    pregunta: str,
    respuesta: str,
    contexto: str = "",
) -> dict:
    """
    Usa Claude como juez para evaluar la calidad de una respuesta.

    Args:
        pregunta: La pregunta original del usuario.
        respuesta: La respuesta generada por el modelo a evaluar.
        contexto: Información adicional de referencia (opcional).

    Returns:
        Diccionario con las puntuaciones y justificación.
    """
    contenido_usuario = f"""**Pregunta:** {pregunta}

**Respuesta a evaluar:** {respuesta}
"""
    if contexto:
        contenido_usuario += f"\n**Contexto de referencia:** {contexto}"

    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=RUBRICA,
        messages=[{"role": "user", "content": contenido_usuario}],
    )

    texto = mensaje.content[0].text.strip()

    # Extraer el JSON aunque el modelo añada texto extra
    inicio = texto.find("{")
    fin = texto.rfind("}") + 1
    if inicio == -1 or fin == 0:
        raise ValueError(f"No se encontró JSON en la respuesta: {texto}")

    return json.loads(texto[inicio:fin])


def evaluar_lote(casos: list[dict]) -> list[dict]:
    """
    Evalúa varios pares pregunta/respuesta y devuelve los resultados.

    Cada elemento de 'casos' debe tener: 'pregunta', 'respuesta' y
    opcionalmente 'contexto' e 'id'.
    """
    resultados = []
    for i, caso in enumerate(casos):
        print(f"  Evaluando caso {i + 1}/{len(casos)}...")
        puntuacion = evaluar_respuesta(
            pregunta=caso["pregunta"],
            respuesta=caso["respuesta"],
            contexto=caso.get("contexto", ""),
        )
        resultados.append(
            {
                "id": caso.get("id", i + 1),
                "pregunta": caso["pregunta"],
                **puntuacion,
            }
        )
    return resultados


# ---------------------------------------------------------------------------
# Ejemplo de uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    casos_prueba = [
        {
            "id": "Q1",
            "pregunta": "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?",
            "respuesta": (
                "El aprendizaje supervisado usa datos etiquetados para entrenar modelos "
                "que predicen una salida concreta, como clasificar emails como spam. "
                "El no supervisado encuentra patrones en datos sin etiquetar, como "
                "agrupar clientes por comportamiento."
            ),
        },
        {
            "id": "Q2",
            "pregunta": "¿Qué es una red neuronal?",
            "respuesta": "Es algo que funciona como el cerebro.",
        },
        {
            "id": "Q3",
            "pregunta": "¿Cuándo se fundó Anthropic?",
            "respuesta": "Anthropic fue fundada en 2021 por Dario Amodei, Daniela Amodei y otros ex-empleados de OpenAI.",
        },
    ]

    print("Evaluando respuestas con LLM-as-judge (Claude)...\n")
    resultados = evaluar_lote(casos_prueba)

    print(f"\n{'ID':<5} {'Factual':>7} {'Relev.':>7} {'Claridad':>8} {'Complet.':>9} {'Total':>6}")
    print("-" * 50)
    for r in resultados:
        print(
            f"{r['id']:<5} {r['precision_factual']:>7} {r['relevancia']:>7} "
            f"{r['claridad']:>8} {r['completitud']:>9} {r['puntuacion_total']:>6}"
        )
        print(f"       {r['justificacion'][:70]}")
        print()
```

---

## 4. Framework de evaluación con dataset de referencia

Un framework de evaluación completo combina métricas automáticas con LLM-as-judge y genera un informe estructurado. Aquí evaluamos un chatbot contra 10 preguntas con respuestas esperadas.

```bash
pip install anthropic python-dotenv
```

```python
# evaluacion/framework_evaluacion.py
import json
import statistics
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Dataset de referencia (10 preguntas de ejemplo)
# ---------------------------------------------------------------------------

DATASET = [
    {
        "id": 1,
        "categoria": "conceptos",
        "pregunta": "¿Qué es un transformer en el contexto de los LLMs?",
        "respuesta_esperada": (
            "Un transformer es una arquitectura de red neuronal basada en el mecanismo "
            "de atención (attention). Procesa tokens en paralelo y usa capas de "
            "self-attention para capturar relaciones entre palabras a cualquier distancia. "
            "Es la base de modelos como GPT, Claude y BERT."
        ),
    },
    {
        "id": 2,
        "categoria": "conceptos",
        "pregunta": "¿Qué significa que un modelo sea 'autoregresivo'?",
        "respuesta_esperada": (
            "Un modelo autoregresivo genera texto token a token, usando los tokens "
            "anteriores como contexto para predecir el siguiente. GPT y Claude son "
            "modelos autoregresivos: generan la respuesta de izquierda a derecha."
        ),
    },
    {
        "id": 3,
        "categoria": "practica",
        "pregunta": "¿Cómo puedo reducir las alucinaciones de un LLM?",
        "respuesta_esperada": (
            "Para reducir alucinaciones: usar RAG para grounding en documentos reales, "
            "pedir al modelo que cite fuentes, aplicar temperature baja, usar chain-of-thought "
            "para razonamiento explícito, y evaluar las respuestas con LLM-as-judge."
        ),
    },
    {
        "id": 4,
        "categoria": "practica",
        "pregunta": "¿Qué es el temperature en un LLM y cómo afecta a las respuestas?",
        "respuesta_esperada": (
            "Temperature controla la aleatoriedad del muestreo. Valores bajos (0-0.3) "
            "producen respuestas más deterministas y conservadoras. Valores altos (0.7-1.0) "
            "aumentan la variedad y creatividad. Para tareas factuales se recomienda "
            "temperature baja; para generación creativa, alta."
        ),
    },
    {
        "id": 5,
        "categoria": "conceptos",
        "pregunta": "¿Qué es el context window de un LLM?",
        "respuesta_esperada": (
            "El context window es el número máximo de tokens que el modelo puede procesar "
            "en una sola llamada, incluyendo el prompt y la respuesta. Claude 3.5 Sonnet "
            "tiene 200k tokens. Superar este límite provoca errores o truncación del contexto."
        ),
    },
    {
        "id": 6,
        "categoria": "practica",
        "pregunta": "¿Cuál es la diferencia entre un system prompt y un user message?",
        "respuesta_esperada": (
            "El system prompt define el comportamiento, personalidad y restricciones del "
            "asistente. Se envía una vez al inicio y persiste durante la conversación. "
            "Los user messages son las entradas del usuario en cada turno del diálogo."
        ),
    },
    {
        "id": 7,
        "categoria": "conceptos",
        "pregunta": "¿Qué es el fine-tuning y cuándo conviene usarlo?",
        "respuesta_esperada": (
            "El fine-tuning adapta un modelo preentrenado a una tarea específica usando "
            "ejemplos supervisados. Conviene cuando el prompting no da resultados suficientes, "
            "cuando se necesita un estilo muy específico o cuando se requiere latencia baja "
            "con un modelo pequeño. Para la mayoría de casos, prompting con ejemplos es suficiente."
        ),
    },
    {
        "id": 8,
        "categoria": "practica",
        "pregunta": "¿Cómo funciona el RAG (Retrieval-Augmented Generation)?",
        "respuesta_esperada": (
            "RAG combina recuperación de documentos con generación de texto. Al recibir "
            "una pregunta, se buscan los fragmentos más relevantes en una base de datos "
            "vectorial, se insertan en el prompt como contexto y el LLM genera la respuesta "
            "basándose en ellos. Reduce alucinaciones y permite conocimiento actualizado."
        ),
    },
    {
        "id": 9,
        "categoria": "conceptos",
        "pregunta": "¿Qué son los embeddings de texto?",
        "respuesta_esperada": (
            "Los embeddings son representaciones numéricas (vectores) del texto que capturan "
            "su significado semántico. Textos con significado similar tienen vectores cercanos "
            "en el espacio vectorial. Se usan en búsqueda semántica, clustering y RAG."
        ),
    },
    {
        "id": 10,
        "categoria": "practica",
        "pregunta": "¿Qué es el prompt caching de Anthropic?",
        "respuesta_esperada": (
            "El prompt caching permite reutilizar partes del prompt (como system prompts "
            "largos o documentos de referencia) entre llamadas. Anthropic cachea el "
            "prefijo marcado con cache_control y cobra solo el 10% del coste normal "
            "por los tokens leídos de caché."
        ),
    },
]


# ---------------------------------------------------------------------------
# Función que genera una respuesta del chatbot a evaluar
# ---------------------------------------------------------------------------

def chatbot_responder(pregunta: str) -> str:
    """El sistema que vamos a evaluar — reemplaza esto con tu propio chatbot."""
    mensaje = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system="Eres un asistente experto en inteligencia artificial. Responde de forma clara y concisa en español.",
        messages=[{"role": "user", "content": pregunta}],
    )
    return mensaje.content[0].text.strip()


# ---------------------------------------------------------------------------
# Evaluador con LLM-as-judge
# ---------------------------------------------------------------------------

SYSTEM_JUEZ = """Eres un evaluador de sistemas de IA. Compara la respuesta del asistente
con la respuesta de referencia y puntúa del 0 al 10.

Devuelve SOLO un JSON con esta forma:
{"puntuacion": <0-10>, "justificacion": "<máximo 50 palabras>"}
"""

def evaluar_con_juez(pregunta: str, respuesta: str, referencia: str) -> dict:
    prompt = f"""Pregunta: {pregunta}

Respuesta de referencia: {referencia}

Respuesta del asistente: {respuesta}"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=SYSTEM_JUEZ,
        messages=[{"role": "user", "content": prompt}],
    )
    texto = msg.content[0].text.strip()
    inicio = texto.find("{")
    fin = texto.rfind("}") + 1
    return json.loads(texto[inicio:fin])


# ---------------------------------------------------------------------------
# Ejecutar evaluación completa
# ---------------------------------------------------------------------------

def ejecutar_evaluacion(dataset: list[dict], verbose: bool = True) -> dict:
    resultados = []
    puntuaciones_por_categoria = {}

    print(f"Evaluando {len(dataset)} preguntas...\n")

    for caso in dataset:
        if verbose:
            print(f"[{caso['id']:02d}/{len(dataset)}] {caso['pregunta'][:60]}...")

        # Obtener respuesta del chatbot
        respuesta = chatbot_responder(caso["pregunta"])

        # Evaluar con LLM-as-judge
        evaluacion = evaluar_con_juez(
            pregunta=caso["pregunta"],
            respuesta=respuesta,
            referencia=caso["respuesta_esperada"],
        )

        resultado = {
            "id": caso["id"],
            "categoria": caso["categoria"],
            "pregunta": caso["pregunta"],
            "respuesta_generada": respuesta,
            "respuesta_esperada": caso["respuesta_esperada"],
            "puntuacion": evaluacion["puntuacion"],
            "justificacion": evaluacion["justificacion"],
        }
        resultados.append(resultado)

        # Acumular por categoría
        cat = caso["categoria"]
        if cat not in puntuaciones_por_categoria:
            puntuaciones_por_categoria[cat] = []
        puntuaciones_por_categoria[cat].append(evaluacion["puntuacion"])

        if verbose:
            print(f"    Puntuación: {evaluacion['puntuacion']}/10 — {evaluacion['justificacion'][:60]}\n")

    # Calcular estadísticas
    todas = [r["puntuacion"] for r in resultados]
    informe = {
        "timestamp": datetime.now().isoformat(),
        "total_preguntas": len(dataset),
        "puntuacion_media": round(statistics.mean(todas), 2),
        "puntuacion_mediana": round(statistics.median(todas), 2),
        "desviacion_tipica": round(statistics.stdev(todas), 2) if len(todas) > 1 else 0,
        "por_categoria": {
            cat: round(statistics.mean(punts), 2)
            for cat, punts in puntuaciones_por_categoria.items()
        },
        "resultados": resultados,
    }

    return informe


def imprimir_informe(informe: dict) -> None:
    print("\n" + "=" * 60)
    print("INFORME DE EVALUACIÓN")
    print("=" * 60)
    print(f"Fecha:              {informe['timestamp'][:19]}")
    print(f"Total preguntas:    {informe['total_preguntas']}")
    print(f"Puntuación media:   {informe['puntuacion_media']}/10")
    print(f"Puntuación mediana: {informe['puntuacion_mediana']}/10")
    print(f"Desviación típica:  {informe['desviacion_tipica']}")
    print("\nPor categoría:")
    for cat, media in informe["por_categoria"].items():
        print(f"  {cat:<15} {media}/10")
    print("=" * 60)


if __name__ == "__main__":
    informe = ejecutar_evaluacion(DATASET, verbose=True)
    imprimir_informe(informe)

    # Guardar resultados
    with open("informe_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=2)
    print("\nResultados guardados en informe_evaluacion.json")
```

---

## 5. Evaluación de RAG con ragas

`ragas` es una librería especializada en evaluar sistemas RAG. Mide si las respuestas son fieles al contexto recuperado y si el contexto es relevante.

```bash
pip install ragas anthropic openai datasets
```

```python
# evaluacion/evaluar_rag.py
"""
Evaluación de un sistema RAG con ragas.

Métricas:
  - faithfulness:       ¿La respuesta se basa en el contexto? (evita alucinaciones)
  - answer_relevancy:   ¿La respuesta responde la pregunta?
  - context_precision:  ¿Los fragmentos recuperados son relevantes?
"""
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# Dataset de evaluación RAG
# Cada ejemplo tiene: pregunta, contextos recuperados, respuesta generada
# y (opcional) respuesta de referencia para context_precision
# ---------------------------------------------------------------------------

datos_evaluacion = {
    "question": [
        "¿Qué es el prompt caching de Anthropic?",
        "¿Cuántos tokens procesa Claude 3.5 Sonnet?",
        "¿Qué modelos ofrece Anthropic?",
    ],
    "contexts": [
        # Contextos recuperados por el sistema RAG (lista de fragmentos por pregunta)
        [
            "El prompt caching de Anthropic permite almacenar prefijos de prompt de más de "
            "1024 tokens. Los tokens leídos de caché tienen un coste del 10% respecto a "
            "los tokens normales de entrada.",
            "Para activar el caché, se usa el campo cache_control con type: ephemeral "
            "en el bloque de contenido que se quiere cachear.",
        ],
        [
            "Claude 3.5 Sonnet tiene una ventana de contexto de 200.000 tokens de entrada "
            "y puede generar hasta 8.192 tokens de salida.",
        ],
        [
            "Anthropic ofrece la familia Claude 3: Haiku (rápido y económico), "
            "Sonnet (equilibrado) y Opus (más capaz). También Claude 3.5 Sonnet y Haiku.",
        ],
    ],
    "answer": [
        # Respuestas generadas por el sistema RAG
        "El prompt caching permite reutilizar partes del prompt entre llamadas. "
        "Los tokens en caché cuestan solo el 10% del precio normal.",
        "Claude 3.5 Sonnet puede procesar hasta 200.000 tokens de entrada.",
        "Anthropic ofrece Claude 3 Haiku, Claude 3 Sonnet, Claude 3 Opus y "
        "Claude 3.5 Sonnet. Haiku es el más rápido y económico.",
    ],
    "ground_truth": [
        # Respuestas de referencia escritas por humanos (necesarias para context_precision)
        "Prompt caching almacena prefijos del prompt. Los tokens en caché cuestan el 10% del precio normal.",
        "Claude 3.5 Sonnet tiene 200.000 tokens de contexto.",
        "Anthropic tiene Claude 3 Haiku, Sonnet y Opus, más Claude 3.5 Sonnet y Haiku.",
    ],
}

# ---------------------------------------------------------------------------
# Configurar los modelos para ragas
# ragas usa LangChain internamente
# ---------------------------------------------------------------------------

llm_juez = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

# ---------------------------------------------------------------------------
# Ejecutar evaluación
# ---------------------------------------------------------------------------

def evaluar_rag() -> None:
    dataset = Dataset.from_dict(datos_evaluacion)

    print("Ejecutando evaluación RAG con ragas...\n")
    resultado = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm_juez,
        embeddings=embeddings,
    )

    print("Resultados:")
    print(f"  faithfulness:      {resultado['faithfulness']:.4f}  (fidelidad al contexto, 1.0 = perfecto)")
    print(f"  answer_relevancy:  {resultado['answer_relevancy']:.4f}  (relevancia de la respuesta)")
    print(f"  context_precision: {resultado['context_precision']:.4f}  (precisión del contexto recuperado)")

    # Detalle por pregunta
    df = resultado.to_pandas()
    print("\nDetalle por pregunta:")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_string(index=False))


if __name__ == "__main__":
    evaluar_rag()
```

**Interpretación de las métricas:**

| Métrica | Qué mide | Valor ideal |
|---|---|---|
| `faithfulness` | La respuesta no inventa cosas que no están en el contexto | 1.0 |
| `answer_relevancy` | La respuesta responde la pregunta formulada | 1.0 |
| `context_precision` | Los documentos recuperados son los relevantes | 1.0 |

---

## 6. Extensiones sugeridas

- **Evaluación continua en CI/CD**: ejecutar el framework de evaluación en cada Pull Request para detectar regresiones automáticamente.
- **A/B testing de prompts**: comparar dos versiones del system prompt usando el mismo dataset de referencia.
- **Panel de control**: almacenar los resultados en una base de datos (SQLite o PostgreSQL) y visualizarlos con Grafana o Streamlit.
- **Evaluación de seguridad**: añadir casos de prueba específicos para detectar fugas de información, prompt injection o respuestas inapropiadas.
- **Anotación humana**: integrar una interfaz sencilla (Label Studio, Argilla) para que expertos revisen los casos con puntuación baja.

---

**Siguiente:** [Observabilidad y tracing](./02-observabilidad.md)
