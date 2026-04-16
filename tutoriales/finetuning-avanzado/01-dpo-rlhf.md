# 01 — DPO y RLHF: Ajuste por Preferencias Humanas

> **Bloque:** Fine-tuning avanzado · **Nivel:** Avanzado · **Tiempo estimado:** 55 min

---

## Índice

1. [Qué es RLHF y por qué importa](#1-qué-es-rlhf-y-por-qué-importa)
2. [El ciclo completo de RLHF](#2-el-ciclo-completo-de-rlhf)
3. [DPO: Direct Preference Optimization](#3-dpo-direct-preference-optimization)
4. [DPO vs RLHF: cuándo usar cada uno](#4-dpo-vs-rlhf-cuándo-usar-cada-uno)
5. [Dataset de preferencias: formato chosen/rejected](#5-dataset-de-preferencias-formato-chosenrejected)
6. [Implementación DPO con TRL](#6-implementación-dpo-con-trl)
7. [Entrenamiento con DPOTrainer](#7-entrenamiento-con-dpotrainer)
8. [Evaluación del modelo fine-tuneado](#8-evaluación-del-modelo-fine-tuneado)
9. [Extensiones sugeridas](#9-extensiones-sugeridas)

---

## 1. Qué es RLHF y por qué importa

El preentrenamiento de un LLM optimiza una sola señal: predecir el siguiente token. Esto produce modelos que saben mucho sobre el lenguaje pero que no están alineados con lo que los humanos consideran respuestas útiles, honestas o seguras.

**RLHF** (Reinforcement Learning from Human Feedback) es el proceso que usaron OpenAI, Anthropic y Google para convertir modelos base en asistentes. La idea central es sencilla: en lugar de mostrarle al modelo ejemplos de respuestas correctas, le enseñamos qué respuestas son *mejores que otras* según evaluadores humanos.

### Por qué el ajuste supervisado solo no es suficiente

Con SFT puro (imitar ejemplos), el modelo aprende a generar texto similar al training set. Pero:

- Los ejemplos humanos son **inconsistentes**: diferentes anotadores escriben con estilos distintos.
- No todos los ejemplos son de igual calidad; el modelo imita lo bueno y lo malo por igual.
- El modelo no aprende a **rechazar peticiones dañinas** ni a calibrar su incertidumbre.

RLHF resuelve esto entrenando sobre señales de comparación: "esta respuesta A es mejor que esta respuesta B".

---

## 2. El ciclo completo de RLHF

El pipeline clásico de RLHF tiene tres fases:

```
┌─────────────────────────────────────────────────────────────┐
│  FASE 1: Supervised Fine-Tuning (SFT)                        │
│  Modelo base → fine-tuning sobre demostraciones humanas      │
│  Resultado: SFT model (sabe seguir instrucciones básicas)    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  FASE 2: Reward Model (RM)                                   │
│  Humanos comparan pares de respuestas → (chosen, rejected)   │
│  Entrenamos un clasificador que puntúa calidad de respuesta  │
│  Resultado: Reward model r(x, y) → score                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  FASE 3: RL con PPO                                          │
│  El SFT model genera respuestas                              │
│  El RM las puntúa                                            │
│  PPO actualiza el SFT model para maximizar la recompensa     │
│  Un "KL penalty" evita que el modelo se aleje demasiante     │
│  del SFT original (evita reward hacking)                     │
└─────────────────────────────────────────────────────────────┘
```

### El reward model en detalle

El reward model toma un par `(prompt, respuesta)` y produce un escalar que representa la calidad percibida por humanos. Se entrena con una función de pérdida de ranking:

```
L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
```

Donde `r(x, y)` es la puntuación que el reward model asigna a la respuesta `y` dado el prompt `x`.

### PPO con penalización KL

La fase de RL minimiza:

```
L_PPO = -E[r(x, y)] + β · KL[π_θ(y|x) || π_ref(y|x)]
```

- `π_θ` es la política actual (modelo en entrenamiento).
- `π_ref` es la política de referencia (SFT model congelado).
- `β` controla cuánto se permite al modelo alejarse de la referencia.

Sin la penalización KL, el modelo aprende a maximizar la recompensa de formas inesperadas (**reward hacking**): repetir frases que el RM puntúa bien, volverse excesivamente sycophantic, etc.

```python
# Instalación de dependencias
# pip install trl transformers datasets peft accelerate bitsandbytes wandb
```

---

## 3. DPO: Direct Preference Optimization

DPO (Rafailov et al., 2023) es una alternativa a RLHF que elimina la necesidad de entrenar un reward model explícito y de ejecutar PPO. La intuición clave es que **el propio LLM puede actuar como reward model implícito**.

### La pérdida DPO

DPO demuestra que el objetivo de RL de RLHF tiene una solución analítica en forma de política óptima. A partir de esa solución, deriva una función de pérdida que puede optimizarse directamente sobre los datos de preferencia:

```
L_DPO = -E[log σ(
    β · log(π_θ(y_w|x) / π_ref(y_w|x))
  - β · log(π_θ(y_l|x) / π_ref(y_l|x))
)]
```

Donde:
- `y_w` es la respuesta preferida (chosen/winner).
- `y_l` es la respuesta rechazada (rejected/loser).
- `π_ref` es el modelo de referencia (SFT model congelado).
- `β` controla la divergencia respecto a la referencia (típicamente 0.1–0.5).

### Qué hace DPO en la práctica

- **Aumenta** la probabilidad de generar respuestas que los humanos prefieren.
- **Disminuye** la probabilidad de generar respuestas que los humanos rechazan.
- El ratio `π_θ / π_ref` actúa como una señal de cuánto se aleja el modelo de su prior.

---

## 4. DPO vs RLHF: cuándo usar cada uno

| Criterio | RLHF (con PPO) | DPO |
|---|---|---|
| **Complejidad de implementación** | Alta (3 modelos en memoria simultáneos) | Baja (1-2 modelos) |
| **Coste computacional** | Muy alto | Moderado |
| **Estabilidad de entrenamiento** | Frágil (hiperparámetros PPO delicados) | Estable (similar a SFT estándar) |
| **Reward hacking** | Posible sin penalización KL adecuada | Menos probable (β controla la divergencia) |
| **Flexibilidad** | Alta (se puede cambiar el RM) | Menor (la señal viene del propio LM) |
| **Reproducibilidad** | Difícil | Sencilla |
| **Requisito de datos** | Pares comparativos + demostraciones | Solo pares comparativos |
| **Cuándo usarlo** | Proyectos con recursos computacionales amplios y requisitos de alineación muy estrictos | La mayoría de casos prácticos de fine-tuning por preferencias |

> **Recomendación práctica:** Para el 90% de los casos de uso, DPO es la elección correcta. RLHF con PPO solo se justifica si tienes acceso a un equipo de investigación, hardware de escala (decenas de A100) y un proceso de anotación humana de alta calidad.

---

## 5. Dataset de preferencias: formato chosen/rejected

El dataset de DPO tiene tres campos por ejemplo:

```json
{
  "prompt": "¿Cómo puedo mejorar mis habilidades de programación?",
  "chosen": "Para mejorar tus habilidades de programación, te recomiendo: 1) Practica a diario resolviendo problemas en plataformas como LeetCode o Codewars. 2) Lee código de otros en GitHub. 3) Trabaja en proyectos propios que te motiven. 4) Estudia los fundamentos de algoritmia y estructuras de datos.",
  "rejected": "Practica mucho y lee libros."
}
```

### Crear un dataset de preferencias propio

```python
from datasets import Dataset
import json

# Ejemplo: dataset de preferencias para un asistente de soporte técnico
datos_preferencias = [
    {
        "prompt": "Mi aplicación se cuelga al iniciar. ¿Qué hago?",
        "chosen": (
            "Para diagnosticar el problema, sigue estos pasos:\n"
            "1. Revisa los logs de la aplicación: busca archivos .log en el directorio de la app "
            "o en /var/log/ (Linux) o Visor de Eventos (Windows).\n"
            "2. Identifica el último mensaje antes del cuelgue.\n"
            "3. Si ves un error de memoria, prueba aumentar el heap con `-Xmx2g` (Java) o "
            "verificar fugas de memoria con herramientas como Valgrind.\n"
            "4. Comprueba si hay actualizaciones pendientes del sistema o dependencias.\n"
            "¿Puedes compartir los primeros líneas del log de error?"
        ),
        "rejected": (
            "Reinicia el ordenador y vuelve a intentarlo. Si no funciona, reinstala la aplicación."
        ),
    },
    {
        "prompt": "¿Cuál es la diferencia entre una lista y una tupla en Python?",
        "chosen": (
            "La diferencia fundamental es la **mutabilidad**:\n\n"
            "- **Lista** (`list`): mutable. Puedes añadir, eliminar o modificar elementos después de crearla. "
            "Se define con corchetes: `mi_lista = [1, 2, 3]`.\n"
            "- **Tupla** (`tuple`): inmutable. Una vez creada, no puede modificarse. "
            "Se define con paréntesis: `mi_tupla = (1, 2, 3)`.\n\n"
            "Cuándo usar cada una:\n"
            "- Usa listas cuando los datos puedan cambiar (p.ej., una cola de tareas).\n"
            "- Usa tuplas para datos constantes (coordenadas, claves de diccionario, valores de retorno múltiple).\n"
            "- Las tuplas son ligeramente más rápidas y consumen menos memoria."
        ),
        "rejected": (
            "Las listas usan [] y las tuplas usan (). Las listas son mutables."
        ),
    },
    {
        "prompt": "Explícame qué es una API REST",
        "chosen": (
            "Una **API REST** (Representational State Transfer) es un conjunto de convenciones para "
            "diseñar servicios web que permiten a diferentes sistemas comunicarse a través de HTTP.\n\n"
            "Los principios clave son:\n"
            "1. **Sin estado (stateless)**: cada petición contiene toda la información necesaria; "
            "el servidor no recuerda peticiones anteriores.\n"
            "2. **Recursos identificados por URLs**: `/usuarios/42` representa al usuario con id 42.\n"
            "3. **Verbos HTTP con significado semántico**: GET (leer), POST (crear), PUT/PATCH (actualizar), DELETE (eliminar).\n"
            "4. **Respuestas en JSON o XML**: formato estándar e interoperable.\n\n"
            "Ejemplo práctico: `GET /api/productos/123` devuelve los datos del producto 123 en JSON."
        ),
        "rejected": (
            "Es una forma de conectar aplicaciones por internet usando HTTP."
        ),
    },
]

# Crear el dataset de HuggingFace
dataset = Dataset.from_list(datos_preferencias)
print(dataset)
# Dataset({features: ['prompt', 'chosen', 'rejected'], num_rows: 3})

# Guardar en disco
dataset.save_to_disk("./dataset_preferencias")

# O exportar a JSONL
with open("dataset_preferencias.jsonl", "w", encoding="utf-8") as f:
    for ejemplo in datos_preferencias:
        f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

print(f"Dataset guardado con {len(datos_preferencias)} ejemplos")
```

### Cargar datasets de preferencias públicos

```python
from datasets import load_dataset

# Anthropic HH-RLHF: dataset de preferencias de Anthropic
dataset_hh = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
print(dataset_hh[0])
# {'chosen': "Human: ...\nAssistant: ...", 'rejected': "Human: ...\nAssistant: ..."}

# Ultrafeedback: dataset de preferencias sintéticas de alta calidad
dataset_uf = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs[:1000]"
)
# Tiene campos: prompt, chosen, rejected, score_chosen, score_rejected

# Intel ORCA DPO: instrucciones sintéticas con pares de preferencia
dataset_orca = load_dataset("Intel/orca_dpo_pairs", split="train[:500]")

print("Columnas:", dataset_hh.column_names)
```

### Preprocesar el formato HH-RLHF para DPO

El dataset HH-RLHF usa un formato de conversación concatenada. Hay que extraer el prompt y las respuestas:

```python
from datasets import load_dataset
import re


def extraer_ultimo_turno(texto: str) -> tuple[str, str]:
    """
    Extrae el prompt (todo hasta el último 'Assistant:') y
    la respuesta del asistente del texto del dataset HH-RLHF.
    """
    # El formato es: "Human: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
    partes = re.split(r"\n\nAssistant:", texto)
    
    if len(partes) < 2:
        return texto, ""
    
    # El prompt es todo hasta el último bloque del asistente
    prompt = "Assistant:".join(partes[:-1]) + "\n\nAssistant:"
    respuesta = partes[-1].strip()
    
    return prompt, respuesta


def preprocesar_hh_rlhf(ejemplo: dict) -> dict:
    """Convierte el formato HH-RLHF al formato DPO estándar."""
    prompt_chosen, chosen = extraer_ultimo_turno(ejemplo["chosen"])
    prompt_rejected, rejected = extraer_ultimo_turno(ejemplo["rejected"])
    
    # El prompt debe ser el mismo para chosen y rejected
    # (tomamos el del chosen como referencia)
    return {
        "prompt": prompt_chosen,
        "chosen": chosen,
        "rejected": rejected,
    }


dataset_hh = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
dataset_dpo = dataset_hh.map(preprocesar_hh_rlhf, remove_columns=dataset_hh.column_names)

# Filtrar ejemplos donde la extracción falló
dataset_dpo = dataset_dpo.filter(
    lambda x: len(x["chosen"]) > 20 and len(x["rejected"]) > 20
)

print(f"Dataset procesado: {len(dataset_dpo)} ejemplos")
print("\nEjemplo:")
print("PROMPT:", dataset_dpo[0]["prompt"][:200])
print("\nCHOSEN:", dataset_dpo[0]["chosen"][:200])
print("\nREJECTED:", dataset_dpo[0]["rejected"][:200])
```

---

## 6. Implementación DPO con TRL

### Configuración del entorno

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset

# Verificar GPU disponible
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cargar el modelo base con cuantización 4-bit

Usamos **QLoRA**: el modelo base se carga en 4-bit (para ahorrar VRAM) y los adaptadores LoRA se entrenan en bfloat16.

```python
# Nombre del modelo base (SFT model o modelo de instrucciones)
# Para DPO necesitamos un modelo que ya sepa seguir instrucciones
MODELO_BASE = "mistralai/Mistral-7B-Instruct-v0.2"

# Configuración de cuantización 4-bit con bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Cargar en 4 bits (NF4)
    bnb_4bit_quant_type="nf4",                  # NormalFloat4: mejor precisión que int4
    bnb_4bit_compute_dtype=torch.bfloat16,      # Tipo de datos para el cómputo
    bnb_4bit_use_double_quant=True,             # Doble cuantización: ahorra ~0.4 bits/param adicionales
)

# Cargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODELO_BASE,
    trust_remote_code=True,
)

# Asegurarse de que el tokenizer tenga pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Padding por la izquierda: necesario para LLMs causales durante entrenamiento
tokenizer.padding_side = "left"

# Cargar el modelo con cuantización
modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map="auto",              # Distribuye automáticamente entre GPUs disponibles
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"Modelo cargado. Parámetros: {modelo.num_parameters() / 1e9:.2f}B")
```

### Configurar LoRA para DPO

```python
from peft import prepare_model_for_kbit_training

# Prepara el modelo para entrenamiento en k-bit (activa gradient checkpointing, etc.)
modelo = prepare_model_for_kbit_training(modelo)

# Configuración de LoRA
# Para DPO, típicamente se aplica a más módulos que en SFT básico
lora_config = LoraConfig(
    r=16,                           # Rango de las matrices de adaptación (16-64 para DPO)
    lora_alpha=32,                  # Factor de escala (alpha/r = escala efectiva; 2x r es común)
    target_modules=[                # Módulos donde aplicar LoRA
        "q_proj",                   # Query projection
        "k_proj",                   # Key projection
        "v_proj",                   # Value projection
        "o_proj",                   # Output projection
        "gate_proj",                # FFN gate (importante para DPO)
        "up_proj",                  # FFN up projection
        "down_proj",                # FFN down projection
    ],
    lora_dropout=0.05,              # Dropout en las capas LoRA
    bias="none",                    # No entrenar bias
    task_type=TaskType.CAUSAL_LM,   # Tipo de tarea
)

# Aplicar LoRA al modelo
modelo = get_peft_model(modelo, lora_config)
modelo.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 7,325,184,000 || trainable%: 1.14%
```

---

## 7. Entrenamiento con DPOTrainer

```python
from trl import DPOConfig
from datasets import load_dataset

# ── Dataset ─────────────────────────────────────────────────────────────────
# Usamos un dataset de ejemplo; en producción reemplaza con tu dataset propio
dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

# Dividir en train/eval
split = dataset.train_test_split(test_size=0.05, seed=42)
dataset_train = split["train"]
dataset_eval = split["test"]

print(f"Train: {len(dataset_train)} ejemplos")
print(f"Eval:  {len(dataset_eval)} ejemplos")


# ── Configuración del entrenamiento ─────────────────────────────────────────
dpo_config = DPOConfig(
    # Directorio de salida
    output_dir="./mistral-7b-dpo",

    # Hiperparámetros de DPO
    beta=0.1,                           # Coeficiente KL: controla la divergencia de la política de referencia
                                        # Valores típicos: 0.01 (más libertad) a 0.5 (más conservador)

    # Hiperparámetros de entrenamiento
    num_train_epochs=1,                 # 1-3 épocas suele ser suficiente para DPO
    per_device_train_batch_size=2,      # Batch size por GPU (ajustar según VRAM)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,     # Batch efectivo = 2 * 8 = 16
    learning_rate=5e-7,                 # DPO requiere LR muy bajo (1e-7 a 1e-6)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # Longitudes máximas
    max_length=1024,                    # Longitud total prompt + respuesta
    max_prompt_length=512,              # Longitud máxima del prompt

    # Precisión y rendimiento
    bf16=True,                          # bfloat16 (mejor que fp16 para LLMs)
    gradient_checkpointing=True,        # Ahorra VRAM a costa de velocidad

    # Logging y evaluación
    logging_steps=25,
    eval_steps=100,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,

    # Weights & Biases
    report_to="wandb",
    run_name="mistral-7b-dpo-ultrafeedback",

    # Configuración de datos
    remove_unused_columns=False,
)


# ── Entrenar ─────────────────────────────────────────────────────────────────
# DPOTrainer maneja automáticamente:
# - El modelo de referencia (copia del modelo con LoRA desactivado)
# - El cálculo de la pérdida DPO
# - La tokenización del formato de chat

trainer = DPOTrainer(
    model=modelo,
    args=dpo_config,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    # model_ref: si no se especifica, DPOTrainer crea automáticamente una copia
    # congelada del modelo como referencia. Solo especificar si quieres usar
    # un modelo de referencia diferente.
)

# Iniciar entrenamiento
trainer.train()

# Guardar el modelo final
trainer.save_model("./mistral-7b-dpo-final")
tokenizer.save_pretrained("./mistral-7b-dpo-final")
print("Modelo guardado en ./mistral-7b-dpo-final")
```

### Fusionar los adaptadores LoRA con el modelo base

Después del entrenamiento, puedes fusionar los pesos LoRA con el modelo base para obtener un modelo standalone que no necesita la biblioteca PEFT para inferencia:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODELO_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTADOR_DPO = "./mistral-7b-dpo-final"
SALIDA_FUSIONADA = "./mistral-7b-dpo-merged"

# Cargar el modelo base en precisión completa (para fusionar correctamente)
print("Cargando modelo base...")
modelo_base = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    torch_dtype=torch.bfloat16,
    device_map="cpu",           # Fusionar en CPU para evitar problemas de VRAM
)

# Cargar los adaptadores LoRA sobre el modelo base
print("Cargando adaptadores LoRA...")
modelo_peft = PeftModel.from_pretrained(modelo_base, ADAPTADOR_DPO)

# Fusionar y descargar (merge_and_unload combina los pesos y devuelve un modelo normal)
print("Fusionando pesos LoRA...")
modelo_fusionado = modelo_peft.merge_and_unload()

# Guardar el modelo fusionado
print("Guardando modelo fusionado...")
modelo_fusionado.save_pretrained(SALIDA_FUSIONADA)

tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
tokenizer.save_pretrained(SALIDA_FUSIONADA)

print(f"Modelo fusionado guardado en {SALIDA_FUSIONADA}")
print("Ahora puedes usar el modelo sin PEFT:")
print(f"  model = AutoModelForCausalLM.from_pretrained('{SALIDA_FUSIONADA}')")
```

---

## 8. Evaluación del modelo fine-tuneado

### Comparación cualitativa

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def cargar_modelo_para_evaluacion(ruta_modelo: str) -> pipeline:
    """Carga un modelo y devuelve un pipeline de generación de texto."""
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipeline(
        "text-generation",
        model=modelo,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,            # Greedy decoding para evaluación reproducible
        temperature=1.0,
        repetition_penalty=1.1,
    )


def formatear_prompt_chat(instruccion: str, tokenizer) -> str:
    """Aplica el chat template del tokenizer al prompt."""
    mensajes = [{"role": "user", "content": instruccion}]
    return tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True,
    )


def comparar_modelos(
    prompts: list[str],
    modelo_base_path: str,
    modelo_dpo_path: str,
) -> None:
    """Compara las respuestas del modelo base y el modelo DPO."""
    print("Cargando modelo base...")
    pipe_base = cargar_modelo_para_evaluacion(modelo_base_path)

    print("Cargando modelo DPO...")
    pipe_dpo = cargar_modelo_para_evaluacion(modelo_dpo_path)

    tokenizer_base = AutoTokenizer.from_pretrained(modelo_base_path)

    for i, prompt in enumerate(prompts, 1):
        prompt_formateado = formatear_prompt_chat(prompt, tokenizer_base)

        respuesta_base = pipe_base(prompt_formateado)[0]["generated_text"]
        respuesta_dpo = pipe_dpo(prompt_formateado)[0]["generated_text"]

        # Extraer solo la respuesta generada (quitar el prompt)
        respuesta_base = respuesta_base[len(prompt_formateado):].strip()
        respuesta_dpo = respuesta_dpo[len(prompt_formateado):].strip()

        print(f"\n{'='*70}")
        print(f"PROMPT {i}: {prompt}")
        print(f"\n[BASE]:\n{respuesta_base}")
        print(f"\n[DPO]:\n{respuesta_dpo}")


# Prompts de evaluación
prompts_evaluacion = [
    "¿Cuál es la capital de Francia y qué es lo más famoso de esa ciudad?",
    "Explícame el concepto de recursión en programación con un ejemplo.",
    "¿Puedes ayudarme a escribir un correo profesional para pedir un aumento de sueldo?",
    "¿Cuáles son los riesgos de tomar aspirina con el estómago vacío?",
]

comparar_modelos(
    prompts_evaluacion,
    modelo_base_path="mistralai/Mistral-7B-Instruct-v0.2",
    modelo_dpo_path="./mistral-7b-dpo-merged",
)
```

### Métricas de reward implícitas

DPOTrainer registra automáticamente métricas clave durante el entrenamiento. Las más importantes son:

```python
# Las siguientes métricas se loggean automáticamente en W&B / TensorBoard:

# rewards/chosen: recompensa implícita media para las respuestas elegidas
#   → debe subir durante el entrenamiento

# rewards/rejected: recompensa implícita media para las respuestas rechazadas
#   → debe bajar durante el entrenamiento

# rewards/margins: diferencia entre chosen y rejected (rewards/chosen - rewards/rejected)
#   → debe ser positivo y crecer

# rewards/accuracies: fracción de ejemplos donde chosen > rejected
#   → objetivo: > 0.8 para un buen modelo

# logps/chosen, logps/rejected: log-probabilidades de las secuencias
#   → chosen debe subir, rejected debe bajar

# loss: pérdida DPO
#   → debe decrecer estabilmente

# Leer las métricas desde el log de entrenamiento:
import json

with open("./mistral-7b-dpo/trainer_state.json") as f:
    estado = json.load(f)

# Ver las últimas métricas
ultimas_metricas = estado["log_history"][-1]
print("\nÚltimas métricas de entrenamiento:")
for clave, valor in ultimas_metricas.items():
    if isinstance(valor, float):
        print(f"  {clave}: {valor:.4f}")
```

### Evaluación de accuracy de preferencias sobre test set

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def calcular_log_prob(modelo, tokenizer, prompt: str, respuesta: str) -> float:
    """
    Calcula la log-probabilidad de una respuesta dado un prompt.
    Esto es lo que DPO usa internamente para calcular las recompensas implícitas.
    """
    # Tokenizar prompt + respuesta
    input_ids_completo = tokenizer.encode(
        prompt + respuesta,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(modelo.device)

    # Tokenizar solo el prompt para saber qué tokens son respuesta
    input_ids_prompt = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(modelo.device)

    len_prompt = input_ids_prompt.shape[1]

    # Calcular logits
    with torch.no_grad():
        outputs = modelo(input_ids_completo)
        logits = outputs.logits

    # Calcular log-probabilidades solo para los tokens de la respuesta
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    tokens_respuesta = input_ids_completo[:, len_prompt:]

    if tokens_respuesta.shape[1] == 0:
        return 0.0

    # Recolectar log-prob de los tokens correctos
    log_probs_respuesta = log_probs[
        :, len_prompt - 1: len_prompt - 1 + tokens_respuesta.shape[1], :
    ]
    log_probs_seleccionadas = log_probs_respuesta.gather(
        2, tokens_respuesta.unsqueeze(-1)
    ).squeeze(-1)

    # Normalizar por longitud (log-prob media por token)
    return log_probs_seleccionadas.mean().item()


def evaluar_accuracy_preferencias(
    ruta_modelo: str,
    dataset_eval,
    n_ejemplos: int = 100,
) -> dict:
    """
    Evalúa cuántas veces el modelo asigna mayor probabilidad a la respuesta
    elegida que a la rechazada.
    """
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(
        ruta_modelo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    modelo.eval()

    correctos = 0
    total = min(n_ejemplos, len(dataset_eval))
    margenes = []

    for i in tqdm(range(total), desc="Evaluando"):
        ejemplo = dataset_eval[i]
        prompt = ejemplo["prompt"]

        lp_chosen = calcular_log_prob(modelo, tokenizer, prompt, ejemplo["chosen"])
        lp_rejected = calcular_log_prob(modelo, tokenizer, prompt, ejemplo["rejected"])

        if lp_chosen > lp_rejected:
            correctos += 1

        margenes.append(lp_chosen - lp_rejected)

    accuracy = correctos / total
    margen_medio = sum(margenes) / len(margenes)

    return {
        "accuracy_preferencias": accuracy,
        "margen_medio": margen_medio,
        "n_evaluados": total,
    }


# Evaluar
dataset_test = load_dataset(
    "HuggingFaceH4/ultrafeedback_binarized",
    split="test_prefs"
)

print("Evaluando modelo DPO...")
metricas = evaluar_accuracy_preferencias(
    "./mistral-7b-dpo-merged",
    dataset_test,
    n_ejemplos=200,
)
print(f"\nResultados:")
print(f"  Accuracy de preferencias: {metricas['accuracy_preferencias']:.2%}")
print(f"  Margen medio (log-prob chosen - rejected): {metricas['margen_medio']:.4f}")
print(f"  Evaluados: {metricas['n_evaluados']} ejemplos")
# Un buen modelo DPO debería tener accuracy > 75-80%
```

---

## 9. Extensiones sugeridas

- **IPO (Identity Preference Optimization)**: variante de DPO más robusta ante datasets con ruido en las etiquetas de preferencia.
- **KTO (Kahneman-Tversky Optimization)**: no requiere pares chosen/rejected, solo etiquetas binarias de si la respuesta es "buena" o "mala".
- **ORPO (Odds Ratio Preference Optimization)**: combina SFT y DPO en una sola pérdida, ahorrando el modelo de referencia.
- **Constitutional AI (CAI)**: pipeline de Anthropic que usa al propio LLM para generar pares de preferencia y retroalimentación sin anotadores humanos.
- **Datos de preferencia sintéticos**: usar GPT-4 o Claude para generar pares chosen/rejected a escala, usando un conjunto de criterios de calidad predefinidos.

---

**Anterior:** [README — Bloque 14: Fine-tuning Avanzado](./README.md) · **Siguiente:** [02 — Instruction Tuning](./02-instruction-tuning.md)
