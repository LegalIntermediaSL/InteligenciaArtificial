# 06 — Fine-tuning con LoRA

> **Bloque:** LLMs · **Nivel:** Avanzado · **Tiempo estimado:** 45 min

---

## Índice

1. [¿Qué es el fine-tuning y cuándo usarlo?](#1-qué-es-el-fine-tuning-y-cuándo-usarlo)
2. [LoRA: el método eficiente](#2-lora-el-método-eficiente)
3. [Preparar el dataset](#3-preparar-el-dataset)
4. [Entorno y requisitos](#4-entorno-y-requisitos)
5. [Fine-tuning con QLoRA (local)](#5-fine-tuning-con-qlora-local)
6. [Fine-tuning en la nube](#6-fine-tuning-en-la-nube)
7. [Evaluar el modelo ajustado](#7-evaluar-el-modelo-ajustado)
8. [Desplegar el modelo](#8-desplegar-el-modelo)
9. [Resumen y decisión](#9-resumen-y-decisión)

---

## 1. ¿Qué es el fine-tuning y cuándo usarlo?

El **fine-tuning** consiste en continuar el entrenamiento de un modelo preentrenado con datos específicos de tu dominio, ajustando sus pesos para que aprenda comportamientos o conocimientos nuevos.

### Cuándo tiene sentido hacer fine-tuning

| Situación | Fine-tuning recomendado |
|---|---|
| Necesitas un estilo o tono muy específico y consistente | ✅ Sí |
| El modelo debe hablar siempre como tu marca | ✅ Sí |
| Terminología muy especializada y poco común | ✅ Sí |
| Formato de salida muy estricto (JSON, SQL, código interno) | ✅ Sí |
| Latencia crítica (no puedes pagar el overhead de RAG) | ✅ Sí |
| Conocimiento que cambia frecuentemente | ❌ Usa RAG |
| Tienes menos de 200 ejemplos | ❌ Usa few-shot |
| Necesitas citar fuentes | ❌ Usa RAG |

### Lo que fine-tuning NO hace

- **No añade conocimiento actualizado** — el modelo sigue sin saber qué pasó después de su fecha de corte
- **No elimina alucinaciones** — puede reducirlas en el dominio entrenado, pero no las elimina
- **No es magia** — si los datos de entrenamiento son malos, el modelo será malo

---

## 2. LoRA: el método eficiente

### El problema del full fine-tuning

Ajustar todos los parámetros de un modelo grande es prohibitivamente caro:

| Modelo | Parámetros | VRAM necesaria (full FT) |
|---|---|---|
| Llama 3 8B | 8B | ~80 GB |
| Mistral 7B | 7B | ~60 GB |
| Llama 3 70B | 70B | ~600 GB |

### ¿Qué es LoRA?

**LoRA (Low-Rank Adaptation)** congela los pesos originales del modelo y añade matrices de bajo rango entrenables en puntos clave de la arquitectura Transformer.

```
Pesos originales W (congelados)
            +
Matrices LoRA: W_A × W_B  (entrenables, mucho más pequeñas)
            ↓
Resultado: W + α × W_A × W_B
```

**Intuición:** en lugar de modificar una matriz grande de 4096×4096 = 16M parámetros, entrenas dos matrices pequeñas de 4096×16 y 16×4096 = 131K parámetros. Una reducción de ~99%.

### QLoRA: LoRA + cuantización

**QLoRA** añade cuantización de 4-bit al modelo base, reduciendo aún más la memoria:

| Método | VRAM para Llama 3 8B | Calidad |
|---|---|---|
| Full fine-tuning | ~80 GB | Máxima |
| LoRA (bf16) | ~20 GB | Casi igual |
| QLoRA (4-bit) | ~6 GB | Muy buena |

Con QLoRA puedes hacer fine-tuning de Llama 3 8B en una **GPU de 8 GB** (RTX 3070, A10, etc.).

### Parámetros clave de LoRA

```python
# Configuración típica de LoRA
lora_config = {
    "r": 16,              # Rango de las matrices (8-64, más alto = más capacidad y coste)
    "lora_alpha": 32,     # Factor de escala (suele ser 2*r)
    "target_modules": [   # Módulos donde aplicar LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",  # Atención
        "gate_proj", "up_proj", "down_proj"       # Feed-forward
    ],
    "lora_dropout": 0.05, # Regularización
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

---

## 3. Preparar el dataset

La calidad del dataset es lo más importante. Un dataset pequeño y de calidad supera a uno grande y ruidoso.

### Formato estándar: Alpaca / ChatML

```python
# Formato Alpaca (instrucción → respuesta)
ejemplo_alpaca = {
    "instruction": "Clasifica el siguiente email como urgente, normal o spam.",
    "input": "Asunto: GANASTE UN IPHONE GRATIS!!!",
    "output": "spam"
}

# Formato ChatML (conversación multivuelta)
ejemplo_chatml = {
    "messages": [
        {"role": "system", "content": "Eres el asistente de atención al cliente de TechCorp."},
        {"role": "user", "content": "No puedo acceder a mi cuenta."},
        {"role": "assistant", "content": "Lo siento. ¿Has probado a restablecer la contraseña desde la página de login?"}
    ]
}
```

### Crear el dataset

```python
import json
from pathlib import Path

def crear_dataset(ejemplos: list[dict], ruta_salida: str):
    """Guarda el dataset en formato JSONL (un JSON por línea)."""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for ejemplo in ejemplos:
            f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
    print(f"Dataset guardado: {len(ejemplos)} ejemplos en {ruta_salida}")


# Ejemplo: dataset de clasificación de tickets de soporte
dataset = [
    {
        "instruction": "Clasifica este ticket de soporte.",
        "input": "La aplicación no carga en mi móvil desde ayer.",
        "output": "BUG"
    },
    {
        "instruction": "Clasifica este ticket de soporte.",
        "input": "¿Cómo puedo cambiar mi contraseña?",
        "output": "PREGUNTA"
    },
    {
        "instruction": "Clasifica este ticket de soporte.",
        "input": "Me gustaría poder exportar los informes a PDF.",
        "output": "FEATURE_REQUEST"
    },
    # ... mínimo 200 ejemplos por categoría para buenos resultados
]

crear_dataset(dataset, "dataset_tickets.jsonl")
```

### Generar datos sintéticos con Claude

Cuando no tienes suficientes datos, Claude puede ayudarte a generar ejemplos:

```python
import anthropic

client = anthropic.Anthropic()

def generar_datos_sinteticos(categoria: str, n: int = 20) -> list[dict]:
    """Genera ejemplos sintéticos de una categoría usando Claude."""
    prompt = f"""Genera {n} ejemplos de tickets de soporte de la categoría '{categoria}'.
Cada ejemplo debe ser diferente y realista.
Devuelve SOLO un JSON array: [{{"texto": "...", "categoria": "{categoria}"}}]"""

    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = r.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]

    return json.loads(raw)


# Generar 20 ejemplos por categoría
categorias = ["BUG", "PREGUNTA", "FEATURE_REQUEST", "QUEJA"]
todos_los_datos = []
for cat in categorias:
    print(f"Generando ejemplos de {cat}...")
    ejemplos = generar_datos_sinteticos(cat, n=20)
    todos_los_datos.extend(ejemplos)

print(f"\nTotal ejemplos generados: {len(todos_los_datos)}")
```

### Checklist de calidad del dataset

```
✅ Mínimo 200 ejemplos por categoría / comportamiento
✅ Ejemplos variados (no repetitivos)
✅ Formato consistente en todos los ejemplos
✅ Sin errores tipográficos graves en las respuestas esperadas
✅ Distribución equilibrada entre clases (si es clasificación)
✅ Dividir en train (80%) / validation (20%)
✅ Revisar manualmente una muestra antes de entrenar
```

---

## 4. Entorno y requisitos

```bash
# GPU recomendada: NVIDIA con ≥8 GB VRAM (RTX 3070, A10, T4...)
# En Google Colab: seleccionar GPU T4 (gratis) o A100 (Colab Pro)

pip install transformers datasets peft trl accelerate bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```python
import torch

# Verificar GPU
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 5. Fine-tuning con QLoRA (local)

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ── 1. Configuración QLoRA ────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Requiere aceptar licencia en HF
OUTPUT_DIR = "./modelo_ajustado"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Cuantización 4-bit
    bnb_4bit_quant_type="nf4",               # Tipo de cuantización
    bnb_4bit_compute_dtype=torch.bfloat16,   # Tipo de cómputo
    bnb_4bit_use_double_quant=True           # Doble cuantización (ahorra ~0.4 bits/param)
)

# ── 2. Cargar modelo y tokenizer ─────────────────────────────────────────────
print("Cargando modelo base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

modelo_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
modelo_base.config.use_cache = False

# ── 3. Configurar LoRA ────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

modelo_lora = get_peft_model(modelo_base, lora_config)
modelo_lora.print_trainable_parameters()
# → trainable params: 41,943,040 || all params: 8,071,626,752 || trainable%: 0.52%

# ── 4. Cargar dataset ─────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files={
    "train": "dataset_tickets_train.jsonl",
    "validation": "dataset_tickets_val.jsonl"
})

def formatear_prompt(ejemplo):
    """Convierte el ejemplo al formato de instrucción de Llama 3."""
    return {
        "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres un clasificador de tickets de soporte.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{ejemplo['instruction']}

{ejemplo['input']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{ejemplo['output']}<|eot_id|>"""
    }

dataset = dataset.map(formatear_prompt)

# ── 5. Configurar entrenamiento ───────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # Efectivo: batch_size = 4 × 4 = 16
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    bf16=True,                            # Más estable que fp16 en GPUs modernas
    optim="paged_adamw_32bit",            # Optimizador eficiente en memoria
    report_to="none"                      # Cambiar a "wandb" para tracking
)

# ── 6. Entrenar ───────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=modelo_lora,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False
)

print("Iniciando entrenamiento...")
trainer.train()

# ── 7. Guardar adaptadores LoRA ───────────────────────────────────────────────
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modelo guardado en {OUTPUT_DIR}")
```

---

## 6. Fine-tuning en la nube

Si no tienes GPU local, estas opciones son las más accesibles:

### Google Colab (gratuito / Pro)

```python
# En Colab, selecciona Runtime → Change runtime type → GPU (T4 gratis, A100 en Pro)
# El proceso es idéntico al local
# Monta Drive para no perder el modelo:
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = '/content/drive/MyDrive/modelos/mi_modelo_ajustado'
```

### Modal (serverless, pago por uso)

```python
# modal run train.py
import modal

app = modal.App("fine-tuning-lora")
image = modal.Image.debian_slim().pip_install("transformers", "peft", "trl", "bitsandbytes")

@app.function(gpu="A10G", image=image, timeout=3600)
def entrenar():
    # Todo el código de entrenamiento aquí
    pass
```

### Alternativa: Fine-tuning gestionado

Algunos proveedores ofrecen fine-tuning sin gestionar infraestructura:

| Proveedor | Modelos | Notas |
|---|---|---|
| OpenAI | GPT-4o mini, GPT-3.5 | API sencilla, caro |
| Together AI | Llama, Mistral | Más económico |
| Replicate | Múltiples | Pay-per-use |
| Hugging Face AutoTrain | Múltiples | Sin código |

```python
# Fine-tuning con OpenAI (el más sencillo)
from openai import OpenAI
client = OpenAI()

# 1. Subir dataset
with open("dataset_train.jsonl", "rb") as f:
    fichero = client.files.create(file=f, purpose="fine-tune")

# 2. Lanzar fine-tuning
job = client.fine_tuning.jobs.create(
    training_file=fichero.id,
    model="gpt-4o-mini-2024-07-18"
)
print(f"Job ID: {job.id}")

# 3. Monitorizar
job_estado = client.fine_tuning.jobs.retrieve(job.id)
print(f"Estado: {job_estado.status}")

# 4. Usar el modelo ajustado
r = client.chat.completions.create(
    model=job_estado.fine_tuned_model,  # ID del modelo ajustado
    messages=[{"role": "user", "content": "Clasifica: El botón no funciona"}]
)
```

---

## 7. Evaluar el modelo ajustado

```python
from peft import PeftModel

def cargar_modelo_ajustado(model_id: str, lora_path: str):
    """Carga el modelo base con los adaptadores LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    modelo_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return PeftModel.from_pretrained(modelo_base, lora_path), tokenizer


def inferencia(modelo, tokenizer, instruccion: str, entrada: str,
               max_tokens: int = 100) -> str:
    """Genera una respuesta del modelo ajustado."""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruccion}

{entrada}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    respuesta = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return respuesta.strip()


def evaluar_modelo(modelo, tokenizer, dataset_eval: list[dict]) -> dict:
    """Evalúa el modelo en un dataset de test."""
    correctos = 0
    resultados = []

    for ejemplo in dataset_eval:
        prediccion = inferencia(modelo, tokenizer, ejemplo["instruction"], ejemplo["input"])
        correcto = prediccion.strip().upper() == ejemplo["output"].strip().upper()
        correctos += correcto
        resultados.append({
            "entrada": ejemplo["input"],
            "esperado": ejemplo["output"],
            "predicho": prediccion,
            "correcto": correcto
        })

    return {
        "accuracy": correctos / len(dataset_eval),
        "total": len(dataset_eval),
        "resultados": resultados
    }
```

---

## 8. Desplegar el modelo

### Combinar adaptadores con el modelo base (para producción)

```python
# Merge LoRA + base model → modelo standalone (sin dependencia de PEFT)
from peft import PeftModel
import torch

modelo_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
modelo_peft = PeftModel.from_pretrained(modelo_base, OUTPUT_DIR)

# Merge y descargar adaptadores
modelo_merged = modelo_peft.merge_and_unload()
modelo_merged.save_pretrained("./modelo_final_merged")

# Publicar en Hugging Face Hub (opcional)
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./modelo_final_merged",
    repo_id="tu-usuario/mi-modelo-ajustado",
    repo_type="model"
)
```

### Servir con Ollama (local)

```bash
# Convertir a formato GGUF y ejecutar con Ollama
pip install llama-cpp-python

# Crear Modelfile
echo 'FROM ./modelo_final_merged' > Modelfile
ollama create mi-modelo -f Modelfile
ollama run mi-modelo
```

---

## 9. Resumen y decisión

```
¿Vale la pena hacer fine-tuning?

¿Tienes al menos 200 ejemplos de calidad por comportamiento?
    └── NO → Usa few-shot prompting con el modelo base

¿El conocimiento cambia frecuentemente?
    └── SÍ → Usa RAG en lugar de fine-tuning

¿Tienes GPU o presupuesto para cloud?
    └── NO → Usa API de OpenAI fine-tuning (gestionado)
    └── SÍ → QLoRA local (Llama 3 8B, Mistral 7B)

¿Qué rank de LoRA usar?
    r=8  → Cambios leves de estilo
    r=16 → Uso general (recomendado)
    r=32 → Tareas complejas / mucho conocimiento nuevo
    r=64 → Casos muy exigentes (raramente necesario)
```

**Tiempo estimado de entrenamiento (QLoRA, Llama 3 8B):**

| Dataset | GPU A10G (24 GB) | GPU T4 (16 GB) |
|---|---|---|
| 1.000 ejemplos, 3 épocas | ~20 min | ~45 min |
| 5.000 ejemplos, 3 épocas | ~90 min | ~3.5 h |
| 20.000 ejemplos, 3 épocas | ~6 h | ~14 h |

---

**Anterior:** [05 — RAG con ChromaDB](./05-rag-chromadb.md) · **Bloque completo: LLMs**
