# 02 — Instruction Tuning con SFTTrainer

> **Bloque:** Fine-tuning avanzado · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [Qué es instruction tuning y por qué funciona](#1-qué-es-instruction-tuning-y-por-qué-funciona)
2. [Formatos de datasets: Alpaca, ShareGPT y ChatML](#2-formatos-de-datasets-alpaca-sharegpt-y-chatml)
3. [Chat templates con el tokenizer](#3-chat-templates-con-el-tokenizer)
4. [Creación de un dataset de instrucciones propio](#4-creación-de-un-dataset-de-instrucciones-propio)
5. [Fine-tuning supervisado con SFTTrainer](#5-fine-tuning-supervisado-con-sfttrainer)
6. [Evaluación cualitativa de respuestas](#6-evaluación-cualitativa-de-respuestas)
7. [Extensiones sugeridas](#7-extensiones-sugeridas)

---

## 1. Qué es instruction tuning y por qué funciona

Un modelo de lenguaje base (preentrenado) sabe completar texto: dado un fragmento, predice el siguiente token. Pero si le das una instrucción como "Traduce este párrafo al inglés:", el modelo puede continuar el texto de formas inesperadas: puede añadir más instrucciones similares, puede escribir una respuesta parcial, puede ignorar la instrucción por completo.

**Instruction tuning** es el proceso de fine-tuning que enseña al modelo a distinguir entre el rol de "usuario que da instrucciones" y el rol de "asistente que las ejecuta", y a responder de manera útil y consistente.

### Por qué funciona

La intuición detrás del instruction tuning es sorprendentemente sencilla: durante el preentrenamiento, el modelo ha visto millones de ejemplos de conversaciones, tutoriales, documentación y textos donde humanos responden preguntas. El conocimiento para responder bien **ya está en los pesos del modelo**. Lo que falta es el mecanismo de activación: aprender que cuando el texto tiene el formato "Instrucción: X", el comportamiento esperado es "Respuesta: Y".

Un estudio seminal (Wei et al., 2021 - FLAN) demostró que fine-tuning sobre un conjunto diverso de tareas formuladas como instrucciones mejora drásticamente la capacidad de generalización a tareas **nuevas** nunca vistas durante el fine-tuning.

### Cuándo usar SFT vs DPO

- Modelo base sin instrucciones (LLaMA-2 base, Mistral base): **empieza con SFT**.
- Modelo de instrucciones que responde pero con baja calidad o mala alineación: añade **DPO** después del SFT.
- Para un dominio específico (medicina, derecho, código): **SFT con datos de dominio** + DPO opcional.

---

## 2. Formatos de datasets: Alpaca, ShareGPT y ChatML

Existen tres formatos ampliamente utilizados. Es importante entender cada uno porque los datasets públicos usan diferentes convenciones y el tokenizer necesita saber cuál aplicar.

### Formato Alpaca

El formato Alpaca (Stanford, 2023) tiene tres campos:

```json
{
  "instruction": "Clasifica el siguiente texto como positivo, negativo o neutro.",
  "input": "El servicio fue aceptable pero el producto llegó tarde.",
  "output": "Negativo"
}
```

- `instruction`: la tarea a realizar.
- `input`: contexto o dato de entrada (puede estar vacío).
- `output`: la respuesta esperada.

Es el más sencillo. Útil para tareas bien definidas con formato fijo. No soporta múltiples turnos de conversación.

### Formato ShareGPT

El formato ShareGPT modela conversaciones multi-turno:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "¿Cuál es la diferencia entre herencia y composición en POO?"
    },
    {
      "from": "gpt",
      "value": "La herencia (is-a) define una jerarquía de tipos: un Perro es un Animal..."
    },
    {
      "from": "human",
      "value": "¿Puedes darme un ejemplo en Python?"
    },
    {
      "from": "gpt",
      "value": "Claro. Con herencia:\n\n```python\nclass Animal:\n    def hablar(self):\n        pass\n```"
    }
  ]
}
```

Es el formato más común en datasets de alta calidad (Vicuna, WizardLM, etc.).

### Formato ChatML

ChatML es el formato estandarizado que usan la mayoría de modelos modernos internamente:

```
<|im_start|>system
Eres un asistente técnico experto en Python.<|im_end|>
<|im_start|>user
¿Qué es un generador en Python?<|im_end|>
<|im_start|>assistant
Un generador es una función especial que usa `yield` en lugar de `return`...<|im_end|>
```

Los tokens especiales `<|im_start|>` y `<|im_end|>` marcan inicio y fin de cada turno. Modelos como Qwen, Phi-3 y muchos otros usan variantes de este formato.

---

## 3. Chat templates con el tokenizer

Los tokenizers modernos de HuggingFace implementan el método `apply_chat_template` que convierte una lista de mensajes al formato correcto para cada modelo. Esto abstrae las diferencias entre Llama-3, Mistral, ChatML, etc.

```python
from transformers import AutoTokenizer

# Llama-3 usa su propio formato con tokens especiales
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

mensajes = [
    {"role": "system", "content": "Eres un experto en Python."},
    {"role": "user", "content": "¿Qué es una list comprehension?"},
    {"role": "assistant", "content": "Una list comprehension es una forma concisa de crear listas..."},
    {"role": "user", "content": "Dame un ejemplo."},
]

# apply_chat_template convierte la lista de mensajes al formato del modelo
texto_llama = tokenizer_llama.apply_chat_template(
    mensajes,
    tokenize=False,         # False = devuelve string; True = devuelve token IDs
    add_generation_prompt=True,  # Añade el inicio del turno del asistente al final
)
print("=== Llama-3 ===")
print(repr(texto_llama[:500]))

# Mistral usa su propio formato sin system message separado
tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# Mistral no soporta system message directamente; hay que integrarlo en el primer user message
mensajes_mistral = [
    {"role": "user", "content": "Eres un experto en Python.\n\n¿Qué es una list comprehension?"},
    {"role": "assistant", "content": "Una list comprehension es una forma concisa de crear listas..."},
    {"role": "user", "content": "Dame un ejemplo."},
]

texto_mistral = tokenizer_mistral.apply_chat_template(
    mensajes_mistral,
    tokenize=False,
    add_generation_prompt=True,
)
print("\n=== Mistral ===")
print(repr(texto_mistral[:500]))
```

### Inspeccionar el chat template de un modelo

```python
from transformers import AutoTokenizer
import jinja2

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# El chat template es una plantilla Jinja2 almacenada en el tokenizer
print("Chat template del tokenizer:")
print(tokenizer.chat_template)

# Ver qué tokens especiales usa el modelo
print("\nTokens especiales:")
print(f"  BOS: {tokenizer.bos_token!r} (id: {tokenizer.bos_token_id})")
print(f"  EOS: {tokenizer.eos_token!r} (id: {tokenizer.eos_token_id})")
print(f"  PAD: {tokenizer.pad_token!r} (id: {tokenizer.pad_token_id})")
```

### Función auxiliar para formatear datasets

```python
def formatear_instruccion_alpaca(ejemplo: dict, tokenizer) -> dict:
    """Convierte el formato Alpaca al formato de chat del tokenizer."""
    instruccion = ejemplo["instruction"]
    entrada = ejemplo.get("input", "")
    salida = ejemplo["output"]

    # Construir el mensaje del usuario
    if entrada:
        contenido_usuario = f"{instruccion}\n\nEntrada: {entrada}"
    else:
        contenido_usuario = instruccion

    mensajes = [
        {"role": "user", "content": contenido_usuario},
        {"role": "assistant", "content": salida},
    ]

    # apply_chat_template genera el texto completo con tokens especiales
    texto = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=False,  # False porque incluimos la respuesta del asistente
    )

    return {"text": texto}


def formatear_sharegpt(ejemplo: dict, tokenizer) -> dict:
    """Convierte el formato ShareGPT al formato de chat del tokenizer."""
    conversaciones = ejemplo["conversations"]

    mensajes = []
    for turno in conversaciones:
        rol = "user" if turno["from"] in ("human", "user") else "assistant"
        mensajes.append({"role": rol, "content": turno["value"]})

    texto = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": texto}
```

---

## 4. Creación de un dataset de instrucciones propio

### Dataset mínimo manual

```python
from datasets import Dataset
import json
from pathlib import Path


# Ejemplos de instrucciones para un asistente de análisis de datos
datos_instrucciones = [
    {
        "instruction": "Explica qué es la normalización de datos y cuándo aplicarla.",
        "input": "",
        "output": (
            "La normalización de datos es el proceso de reescalar los valores de una variable "
            "para que queden dentro de un rango definido, típicamente [0, 1] o [-1, 1].\n\n"
            "**Cuándo aplicarla:**\n"
            "- Antes de entrenar modelos que usan distancias (k-NN, SVM, redes neuronales).\n"
            "- Cuando las variables tienen escalas muy diferentes (p.ej., edad en [0-100] y salario en [0-100000]).\n"
            "- No es necesaria para modelos basados en árboles (Random Forest, XGBoost) que son invariantes a escala.\n\n"
            "**Tipos comunes:**\n"
            "- Min-Max scaling: `x' = (x - min) / (max - min)` → rango [0, 1]\n"
            "- Z-score: `x' = (x - μ) / σ` → media 0, desviación estándar 1\n"
            "- Robust scaling: usa mediana y rango intercuartílico, robusto ante outliers."
        ),
    },
    {
        "instruction": "Escribe una función Python para calcular la media móvil de una serie temporal.",
        "input": "Serie: [10, 12, 13, 12, 15, 14, 16, 18, 17, 19], ventana: 3",
        "output": (
            "```python\ndef media_movil(serie: list[float], ventana: int) -> list[float]:\n"
            "    \"\"\"\n"
            "    Calcula la media móvil de una serie temporal.\n\n"
            "    Args:\n"
            "        serie: Lista de valores numéricos.\n"
            "        ventana: Número de períodos para calcular la media.\n\n"
            "    Returns:\n"
            "        Lista con la media móvil. Los primeros (ventana-1) valores son None.\n"
            "    \"\"\"\n"
            "    resultado = [None] * (ventana - 1)\n"
            "    for i in range(ventana - 1, len(serie)):\n"
            "        media = sum(serie[i - ventana + 1 : i + 1]) / ventana\n"
            "        resultado.append(round(media, 2))\n"
            "    return resultado\n\n\n"
            "# Uso con el ejemplo:\n"
            "serie = [10, 12, 13, 12, 15, 14, 16, 18, 17, 19]\n"
            "mm3 = media_movil(serie, ventana=3)\n"
            "print(mm3)\n"
            "# [None, None, 11.67, 12.33, 13.33, 13.67, 15.0, 16.0, 17.0, 18.0]\n```"
        ),
    },
    {
        "instruction": "¿Qué diferencia hay entre correlación y causalidad? Da un ejemplo.",
        "input": "",
        "output": (
            "**Correlación** significa que dos variables tienden a cambiar juntas estadísticamente. "
            "**Causalidad** significa que el cambio en una variable *produce* el cambio en la otra.\n\n"
            "**Ejemplo clásico (correlación sin causalidad):**\n"
            "Las ventas de helados y los ahogamientos en playas están altamente correlacionados: "
            "cuando suben las ventas de helados, aumentan los ahogamientos. ¿Los helados causan ahogamientos? "
            "No: ambas variables tienen una causa común oculta (el calor/verano), llamada variable confusora.\n\n"
            "**Por qué importa:**\n"
            "- Las decisiones basadas en correlación sin causalidad llevan a intervenciones ineficaces "
            "(prohibir helados no reduciría ahogamientos).\n"
            "- Para establecer causalidad se necesitan experimentos controlados (A/B tests) o "
            "métodos de inferencia causal (diferencias en diferencias, variables instrumentales).\n\n"
            "**Regla:** correlación es necesaria pero no suficiente para la causalidad."
        ),
    },
]

# Crear dataset de HuggingFace
dataset = Dataset.from_list(datos_instrucciones)
print(f"Dataset creado: {dataset}")
print(f"\nPrimer ejemplo:\n{json.dumps(datos_instrucciones[0], indent=2, ensure_ascii=False)}")
```

### Generación de datos sintéticos con Claude

Para escalar el dataset, podemos usar Claude para generar instrucciones y respuestas de alta calidad:

```python
import anthropic
import json
import time
from datasets import Dataset

cliente = anthropic.Anthropic()


SISTEMA_GENERADOR = """Eres un generador de datos de entrenamiento para un asistente de análisis de datos.
Tu tarea es generar ejemplos de instrucciones variadas y respuestas de alta calidad.

Genera exactamente el JSON solicitado, sin texto adicional antes o después."""


def generar_ejemplos_sinteticos(
    tema: str,
    n_ejemplos: int = 5,
    nivel: str = "intermedio",
) -> list[dict]:
    """
    Genera ejemplos de instrucciones sintéticas usando Claude.

    Args:
        tema: El tema sobre el que generar instrucciones.
        n_ejemplos: Cuántos ejemplos generar por llamada.
        nivel: 'básico', 'intermedio' o 'avanzado'.

    Returns:
        Lista de dicts con campos instruction, input, output.
    """
    prompt = f"""Genera {n_ejemplos} ejemplos de instrucciones de nivel {nivel} sobre el tema: {tema}

Devuelve un JSON con este formato exacto:
{{
  "ejemplos": [
    {{
      "instruction": "La instrucción clara y específica",
      "input": "Contexto o datos de entrada (puede ser cadena vacía '')",
      "output": "Respuesta completa, detallada y de alta calidad"
    }}
  ]
}}

Requisitos:
- Las instrucciones deben ser diversas (explicación, código, análisis, comparación, etc.)
- Las respuestas deben ser técnicamente correctas y bien estructuradas
- Usa markdown donde ayude a la claridad (listas, código, negrita)
- Al menos la mitad de los ejemplos deben tener un 'input' no vacío"""

    respuesta = cliente.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        system=SISTEMA_GENERADOR,
    )

    texto = respuesta.content[0].text.strip()

    # Extraer el JSON de la respuesta
    if "```json" in texto:
        texto = texto.split("```json")[1].split("```")[0].strip()
    elif "```" in texto:
        texto = texto.split("```")[1].split("```")[0].strip()

    datos = json.loads(texto)
    return datos["ejemplos"]


def crear_dataset_sintetico(
    temas: list[str],
    ejemplos_por_tema: int = 10,
) -> Dataset:
    """Crea un dataset completo generando ejemplos para múltiples temas."""
    todos_los_ejemplos = []

    for tema in temas:
        print(f"Generando {ejemplos_por_tema} ejemplos para: {tema}")
        try:
            ejemplos = generar_ejemplos_sinteticos(
                tema=tema,
                n_ejemplos=ejemplos_por_tema,
            )
            todos_los_ejemplos.extend(ejemplos)
            print(f"  Generados: {len(ejemplos)} ejemplos")
            time.sleep(1)  # Evitar rate limiting
        except Exception as e:
            print(f"  Error al generar ejemplos para '{tema}': {e}")

    dataset = Dataset.from_list(todos_los_ejemplos)
    print(f"\nDataset total: {len(dataset)} ejemplos")
    return dataset


# Crear dataset para un asistente de ciencia de datos
temas = [
    "pandas y manipulación de DataFrames",
    "visualización de datos con matplotlib y seaborn",
    "preprocesamiento de datos para machine learning",
    "evaluación de modelos de clasificación",
    "series temporales y forecasting",
]

dataset_sintetico = crear_dataset_sintetico(temas, ejemplos_por_tema=5)
dataset_sintetico.save_to_disk("./dataset_data_science")
```

---

## 5. Fine-tuning supervisado con SFTTrainer

### Configuración completa del entrenamiento

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset

# ── Configuración ────────────────────────────────────────────────────────────
MODELO_BASE = "mistralai/Mistral-7B-v0.1"    # Modelo BASE (sin instrucciones)
NOMBRE_EXPERIMENTO = "mistral-7b-sft-data-science"
DIRECTORIO_SALIDA = f"./{NOMBRE_EXPERIMENTO}"

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)

# Modelos base a menudo no tienen pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Para SFT, padding por la derecha es más común (a diferencia de DPO)
tokenizer.padding_side = "right"

# ── Modelo con cuantización 4-bit ────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ── LoRA ─────────────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=64,                           # Rango más alto para SFT (más capacidad de aprendizaje)
    lora_alpha=16,                  # alpha = r/4 es una heurística alternativa
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── Dataset ──────────────────────────────────────────────────────────────────
# Cargar el dataset creado en la sección anterior
# (o usar un dataset público como Alpaca para pruebas)
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Convertir al formato de chat del modelo
def formatear_ejemplo(ejemplo: dict) -> dict:
    """Formatea un ejemplo Alpaca usando el chat template del tokenizer."""
    instruccion = ejemplo["instruction"]
    entrada = ejemplo.get("input", "")
    salida = ejemplo["output"]

    contenido_usuario = f"{instruccion}\n\n{entrada}".strip() if entrada else instruccion

    mensajes = [
        {"role": "user", "content": contenido_usuario},
        {"role": "assistant", "content": salida},
    ]

    # SFTTrainer espera el campo "text" con el texto formateado completo
    texto = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": texto}

dataset_formateado = dataset.map(
    formatear_ejemplo,
    remove_columns=dataset.column_names,
    desc="Formateando dataset",
)

# División train/eval
split = dataset_formateado.train_test_split(test_size=0.02, seed=42)
print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

# ── Configuración de SFT ─────────────────────────────────────────────────────
sft_config = SFTConfig(
    # Rutas
    output_dir=DIRECTORIO_SALIDA,

    # Longitud
    max_seq_length=2048,            # Máxima longitud de secuencia (truncar si es mayor)

    # Épocas y batches
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Batch efectivo = 4 * 4 = 16

    # Optimizador
    learning_rate=2e-4,             # SFT puede usar LR más alto que DPO
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # Precisión
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",       # Optimizador cuantizado (ahorra VRAM)

    # Packing: concatena ejemplos cortos en secuencias más largas
    # Aumenta la eficiencia GPU significativamente
    packing=True,

    # Logging
    logging_steps=10,
    eval_steps=100,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    # W&B
    report_to="wandb",
    run_name=NOMBRE_EXPERIMENTO,
)

# ── SFTTrainer ───────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=modelo,
    args=sft_config,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
    peft_config=lora_config,        # SFTTrainer aplica LoRA internamente
    # dataset_text_field="text",    # Campo con el texto formateado (default: "text")
)

# ── Entrenar ─────────────────────────────────────────────────────────────────
print(f"Parámetros entrenables: {trainer.model.num_parameters(only_trainable=True) / 1e6:.1f}M")
trainer.train()

# Guardar
trainer.save_model(f"{DIRECTORIO_SALIDA}/final")
tokenizer.save_pretrained(f"{DIRECTORIO_SALIDA}/final")
```

### Seguimiento del entrenamiento con Weights & Biases

```python
import wandb

# Inicializar el experimento (alternativa a configurar report_to="wandb" en SFTConfig)
wandb.init(
    project="finetuning-llms",
    name=NOMBRE_EXPERIMENTO,
    config={
        "modelo_base": MODELO_BASE,
        "lora_r": 64,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size_efectivo": 16,
        "dataset": "alpaca",
        "n_ejemplos_train": len(split["train"]),
    },
)

# Las métricas se loggean automáticamente durante trainer.train()
# Métricas clave a monitorear:
# - train/loss: debe decrecer monotónicamente
# - eval/loss: debe decrecer (si sube → overfitting)
# - train/learning_rate: debe seguir el scheduler cosine
# - train/tokens_per_second: eficiencia del entrenamiento
```

---

## 6. Evaluación cualitativa de respuestas

### Generar y comparar respuestas

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def generar_respuesta(
    pipe: pipeline,
    instruccion: str,
    tokenizer,
    system_prompt: str = "",
    max_tokens: int = 512,
) -> str:
    """Genera una respuesta para una instrucción dada."""
    mensajes = []
    if system_prompt:
        mensajes.append({"role": "system", "content": system_prompt})
    mensajes.append({"role": "user", "content": instruccion})

    prompt = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True,
    )

    salida = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )

    texto_generado = salida[0]["generated_text"]
    # Eliminar el prompt del inicio
    respuesta = texto_generado[len(prompt):].strip()
    return respuesta


def evaluar_instruction_following(
    ruta_modelo: str,
    instrucciones_test: list[dict],
    system_prompt: str = "",
) -> list[dict]:
    """
    Evalúa el modelo en un conjunto de instrucciones de prueba.

    Returns:
        Lista de dicts con instruccion, respuesta_generada y anotaciones.
    """
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
    )

    resultados = []
    for item in instrucciones_test:
        instruccion = item["instruccion"]
        respuesta_esperada = item.get("respuesta_esperada", "")

        respuesta_generada = generar_respuesta(
            pipe, instruccion, tokenizer, system_prompt
        )

        resultados.append({
            "instruccion": instruccion,
            "respuesta_esperada": respuesta_esperada,
            "respuesta_generada": respuesta_generada,
        })

        print(f"\n{'─'*60}")
        print(f"INSTRUCCIÓN: {instruccion}")
        print(f"\nRESPUESTA GENERADA:\n{respuesta_generada}")
        if respuesta_esperada:
            print(f"\nRESPUESTA ESPERADA:\n{respuesta_esperada}")

    return resultados


# Casos de prueba de instruction following
instrucciones_test = [
    {
        "instruccion": "Escribe una función Python que cuente las palabras en un texto.",
        "respuesta_esperada": "Función que toma un string y devuelve un entero o dict.",
    },
    {
        "instruccion": "Explica qué es el overfitting en machine learning y cómo prevenirlo.",
        "respuesta_esperada": "Definición clara + técnicas: regularización, dropout, early stopping, más datos.",
    },
    {
        "instruccion": "Traduce al inglés: 'El modelo fue entrenado durante 3 días en 8 GPUs A100.'",
        "respuesta_esperada": "The model was trained for 3 days on 8 A100 GPUs.",
    },
    {
        "instruccion": "Dame una lista de 5 libros sobre inteligencia artificial ordenados por dificultad.",
        "respuesta_esperada": "Lista numerada con títulos, autores y breve descripción.",
    },
]

resultados = evaluar_instruction_following(
    ruta_modelo="./mistral-7b-sft-data-science/final",
    instrucciones_test=instrucciones_test,
    system_prompt="Eres un asistente experto en ciencia de datos y programación en Python.",
)
```

### Evaluación automática con LLM-as-judge

```python
import anthropic
import json
from typing import Literal


cliente_claude = anthropic.Anthropic()


def evaluar_con_llm_judge(
    instruccion: str,
    respuesta: str,
    criterios: list[str] | None = None,
) -> dict:
    """
    Usa Claude como juez para evaluar la calidad de una respuesta.

    Returns:
        Dict con puntuaciones por criterio y justificación.
    """
    if criterios is None:
        criterios = [
            "Corrección técnica: ¿La respuesta es factualmente correcta?",
            "Completitud: ¿La respuesta aborda todos los aspectos de la instrucción?",
            "Claridad: ¿La respuesta es fácil de entender?",
            "Seguimiento de instrucciones: ¿La respuesta hace exactamente lo que pide la instrucción?",
        ]

    criterios_str = "\n".join(f"- {c}" for c in criterios)

    prompt = f"""Evalúa la siguiente respuesta a la instrucción dada.

INSTRUCCIÓN: {instruccion}

RESPUESTA EVALUADA:
{respuesta}

Evalúa según estos criterios (puntuación 1-5 cada uno, donde 5 es perfecto):
{criterios_str}

Devuelve un JSON con este formato:
{{
  "puntuaciones": {{
    "correccion_tecnica": <1-5>,
    "completitud": <1-5>,
    "claridad": <1-5>,
    "seguimiento_instrucciones": <1-5>
  }},
  "puntuacion_total": <promedio de las puntuaciones>,
  "justificacion": "Explicación breve de los puntos fuertes y débiles",
  "mejora_sugerida": "Qué cambiarías para mejorar la respuesta"
}}"""

    respuesta_juez = cliente_claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    texto = respuesta_juez.content[0].text.strip()

    # Extraer JSON
    if "```json" in texto:
        texto = texto.split("```json")[1].split("```")[0].strip()
    elif "```" in texto:
        texto = texto.split("```")[1].split("```")[0].strip()

    evaluacion = json.loads(texto)
    return evaluacion


def evaluar_conjunto_respuestas(resultados: list[dict]) -> dict:
    """Evalúa un conjunto de respuestas y calcula estadísticas agregadas."""
    evaluaciones = []

    for resultado in resultados:
        print(f"\nEvaluando: {resultado['instruccion'][:60]}...")
        evaluacion = evaluar_con_llm_judge(
            instruccion=resultado["instruccion"],
            respuesta=resultado["respuesta_generada"],
        )
        evaluaciones.append({
            **resultado,
            "evaluacion": evaluacion,
        })
        print(f"  Puntuación total: {evaluacion['puntuacion_total']:.1f}/5")

    # Estadísticas agregadas
    puntuaciones_totales = [e["evaluacion"]["puntuacion_total"] for e in evaluaciones]
    media = sum(puntuaciones_totales) / len(puntuaciones_totales)

    print(f"\n{'='*60}")
    print(f"RESUMEN DE EVALUACIÓN")
    print(f"  Ejemplos evaluados: {len(evaluaciones)}")
    print(f"  Puntuación media: {media:.2f}/5")
    print(f"  Puntuación máxima: {max(puntuaciones_totales):.2f}")
    print(f"  Puntuación mínima: {min(puntuaciones_totales):.2f}")

    return {
        "evaluaciones": evaluaciones,
        "media": media,
        "max": max(puntuaciones_totales),
        "min": min(puntuaciones_totales),
    }


# Evaluar las respuestas generadas
resumen = evaluar_conjunto_respuestas(resultados)

# Guardar resultados
with open("evaluacion_sft.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
```

---

## 7. Extensiones sugeridas

- **Flash Attention 2**: instalar `flash-attn` y pasar `attn_implementation="flash_attention_2"` al cargar el modelo. Acelera el entrenamiento 2-4x y permite secuencias mucho más largas.
- **Unsloth**: biblioteca optimizada que acelera el entrenamiento con LoRA hasta 2x con menos consumo de VRAM. Compatible con Llama, Mistral, Gemma y otros.
- **Dataset filtering con perplexity**: calcular la perplexity del modelo base sobre cada ejemplo y filtrar los que tengan valores extremos (demasiado fáciles o demasiado difíciles).
- **Curriculum learning**: ordenar los ejemplos de entrenamiento de más sencillos a más complejos para mejorar la convergencia.
- **Multi-task SFT**: incluir múltiples tipos de tareas (resumen, traducción, QA, código) para producir un asistente más generalista.

---

**Anterior:** [01 — DPO y RLHF](./01-dpo-rlhf.md) · **Siguiente:** [03 — Evaluación de Modelos Fine-tuneados](./03-evaluacion-modelos-finetuneados.md)
