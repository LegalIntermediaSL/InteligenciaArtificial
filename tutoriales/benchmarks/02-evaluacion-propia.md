# Crear tu propio benchmark: evaluación específica del dominio

## Por qué los benchmarks públicos no son suficientes

Supón que estás construyendo un chatbot de soporte técnico para software de contabilidad. MMLU mide conocimiento en 57 materias generales. HumanEval mide código Python. Ninguno te dice si tu chatbot responde correctamente a "¿Cómo concilio una factura duplicada en el libro mayor?".

**La evaluación específica del dominio responde preguntas que los benchmarks públicos no pueden:**
- ¿Responde correctamente en mi vocabulario y contexto?
- ¿Sigue el tono y las políticas de mi empresa?
- ¿Mejora o empeora al cambiar el modelo o el prompt?

## Paso 1: Construir el golden dataset

Un golden dataset es una colección de pares (pregunta, respuesta esperada) representativos de tu caso de uso.

### Fuentes para construir el dataset

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class EjemploEval:
    pregunta: str
    respuesta_esperada: str
    categoria: str
    dificultad: str  # "facil", "media", "dificil"
    notas: Optional[str] = None

# Ejemplos para un chatbot de soporte de software de contabilidad
golden_dataset = [
    EjemploEval(
        pregunta="¿Cómo elimino una factura ya contabilizada?",
        respuesta_esperada="No es posible eliminar una factura contabilizada directamente. Debes crear un abono o nota de crédito que anule el importe, y luego marcar ambos documentos como compensados.",
        categoria="facturas",
        dificultad="media",
    ),
    EjemploEval(
        pregunta="¿Qué es el libro mayor?",
        respuesta_esperada="El libro mayor es el registro contable principal donde se agregan todas las transacciones clasificadas por cuenta. Muestra el saldo acumulado de cada cuenta en un periodo.",
        categoria="conceptos_basicos",
        dificultad="facil",
    ),
    # Añade 50-200 ejemplos para un dataset robusto
]

# Guardar dataset
with open("golden_dataset.json", "w", encoding="utf-8") as f:
    json.dump([vars(e) for e in golden_dataset], f, ensure_ascii=False, indent=2)
```

### Consejo: genera ejemplos con Claude

```python
import anthropic

client = anthropic.Anthropic()

def generar_ejemplos(categoria: str, n: int = 10) -> list[dict]:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Genera {n} pares pregunta/respuesta para evaluar un chatbot de soporte contable.
Categoría: {categoria}
Formato JSON array: [{{"pregunta": "...", "respuesta_esperada": "...", "dificultad": "facil|media|dificil"}}]
Solo el JSON, sin texto adicional."""
        }],
    )
    return json.loads(response.content[0].text)
```

## Paso 2: LLM-as-judge

En lugar de comparar texto exacto (frágil), usamos un LLM para evaluar si la respuesta del modelo es correcta:

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class ResultadoEval:
    puntuacion: float  # 0.0 - 1.0
    correcto: bool
    razon: str

def evaluar_respuesta(
    pregunta: str,
    respuesta_esperada: str,
    respuesta_modelo: str,
) -> ResultadoEval:
    prompt = f"""Evalúa si la respuesta del modelo es correcta y completa.

PREGUNTA: {pregunta}

RESPUESTA ESPERADA: {respuesta_esperada}

RESPUESTA DEL MODELO: {respuesta_modelo}

Evalúa en una escala del 0 al 10:
- 10: Correcta, completa y bien explicada
- 7-9: Correcta pero incompleta o con pequeños errores
- 4-6: Parcialmente correcta
- 1-3: Mayormente incorrecta
- 0: Completamente incorrecta o no responde

Responde en JSON: {{"puntuacion": N, "correcto": true/false, "razon": "explicación breve"}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku es suficiente para evaluar
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    datos = json.loads(response.content[0].text)
    return ResultadoEval(
        puntuacion=datos["puntuacion"] / 10,
        correcto=datos["correcto"],
        razon=datos["razon"],
    )
```

## Paso 3: Pipeline de evaluación completo

```python
import anthropic
import json
from pathlib import Path

client = anthropic.Anthropic()

def generar_respuesta(modelo: str, pregunta: str, system_prompt: str = "") -> str:
    kwargs = {"model": modelo, "max_tokens": 512, "messages": [{"role": "user", "content": pregunta}]}
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[0].text

def evaluar_modelo(
    modelo: str,
    dataset_path: str,
    system_prompt: str = "",
) -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    resultados = []
    for ejemplo in dataset:
        respuesta = generar_respuesta(modelo, ejemplo["pregunta"], system_prompt)
        eval_result = evaluar_respuesta(
            ejemplo["pregunta"],
            ejemplo["respuesta_esperada"],
            respuesta,
        )
        resultados.append({
            "pregunta": ejemplo["pregunta"],
            "categoria": ejemplo.get("categoria", "sin_categoria"),
            "puntuacion": eval_result.puntuacion,
            "correcto": eval_result.correcto,
        })

    total = len(resultados)
    correctos = sum(1 for r in resultados if r["correcto"])
    puntuacion_media = sum(r["puntuacion"] for r in resultados) / total

    return {
        "modelo": modelo,
        "total": total,
        "accuracy": correctos / total,
        "puntuacion_media": puntuacion_media,
        "por_categoria": _agrupar_por_categoria(resultados),
    }

def _agrupar_por_categoria(resultados: list) -> dict:
    categorias = {}
    for r in resultados:
        cat = r["categoria"]
        if cat not in categorias:
            categorias[cat] = []
        categorias[cat].append(r["puntuacion"])
    return {cat: sum(scores)/len(scores) for cat, scores in categorias.items()}

# Comparar dos modelos
resultado_sonnet = evaluar_modelo("claude-sonnet-4-6", "golden_dataset.json")
resultado_haiku = evaluar_modelo("claude-haiku-4-5-20251001", "golden_dataset.json")

print(f"Sonnet 4.6 — Accuracy: {resultado_sonnet['accuracy']:.1%}, Score: {resultado_sonnet['puntuacion_media']:.2f}")
print(f"Haiku 4.5  — Accuracy: {resultado_haiku['accuracy']:.1%}, Score: {resultado_haiku['puntuacion_media']:.2f}")
```

## Paso 4: Métricas automáticas complementarias

```python
# pip install rouge-score bert-score
from rouge_score import rouge_scorer

def calcular_rouge(referencia: str, prediccion: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(referencia, prediccion)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# ROUGE es útil para resúmenes y respuestas donde el texto esperado es específico
# Para respuestas abiertas, LLM-as-judge es más fiable
```

## Paso 5: Evolucionar el benchmark

Un benchmark es un activo vivo:

```python
class GestorBenchmark:
    def __init__(self, path: str):
        self.path = Path(path)
        self._cargar()

    def _cargar(self):
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.datos = json.load(f)
        else:
            self.datos = {"version": "1.0", "ejemplos": []}

    def agregar_ejemplo(self, ejemplo: dict):
        self.datos["ejemplos"].append(ejemplo)
        self._guardar()

    def agregar_desde_feedback(self, pregunta: str, respuesta_mala: str, respuesta_correcta: str):
        """Convierte un error de producción en un caso de test."""
        self.agregar_ejemplo({
            "pregunta": pregunta,
            "respuesta_esperada": respuesta_correcta,
            "notas": f"Caso real: el modelo respondió '{respuesta_mala[:100]}...'",
            "origen": "produccion",
        })

    def _guardar(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.datos, f, ensure_ascii=False, indent=2)
```

## Resumen

1. **Golden dataset** — 50-200 ejemplos representativos de tu dominio
2. **LLM-as-judge** — evalúa con Haiku (barato) usando una rúbrica estructurada
3. **Pipeline completo** — compara modelos y prompts automáticamente
4. **Métricas complementarias** — ROUGE para resúmenes, BLEU para traducción
5. **Evolución continua** — añade casos de errores reales en producción
